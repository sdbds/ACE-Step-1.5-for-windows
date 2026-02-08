from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

from .models import AudioSample


class LabelAllMixin:
    """Label all samples in the dataset."""

    def _batch_encode_audio_codes(
        self,
        samples_to_label: list,
        dit_handler,
        progress_callback,
        total: int,
        chunk_size: int,
        batch_size: int,
    ) -> Dict[int, Optional[str]]:
        codes_cache: Dict[int, Optional[str]] = {}

        if not hasattr(dit_handler, "convert_src_audio_to_codes"):
            logger.error("DiT handler missing convert_src_audio_to_codes method")
            for i, _ in samples_to_label:
                codes_cache[i] = None
            return codes_cache

        uses_context = hasattr(dit_handler, "_load_model_context")
        if uses_context:
            return self._batch_encode_with_context(
                samples_to_label=samples_to_label,
                dit_handler=dit_handler,
                progress_callback=progress_callback,
                total=total,
                chunk_size=chunk_size,
                batch_size=batch_size,
            )

        with torch.inference_mode():
            for idx, (i, sample) in enumerate(samples_to_label):
                if progress_callback and idx % 5 == 0:
                    progress_callback(f"Encoding {idx+1}/{total}: {sample.filename}")
                try:
                    codes = dit_handler.convert_src_audio_to_codes(sample.audio_path)
                    if codes and str(codes).startswith("❌"):
                        codes = None
                    codes_cache[i] = codes
                except Exception:
                    logger.exception(f"Failed to convert audio to codes: {sample.filename}")
                    codes_cache[i] = None

        return codes_cache

    def _batch_encode_with_context(
        self,
        samples_to_label: list,
        dit_handler,
        progress_callback,
        total: int,
        chunk_size: int,
        batch_size: int,
    ) -> Dict[int, Optional[str]]:
        codes_cache: Dict[int, Optional[str]] = {}

        effective_chunk_size = max(1, int(chunk_size) if chunk_size else 1)
        effective_batch_size = max(1, int(batch_size) if batch_size else 1)

        for chunk_start in range(0, len(samples_to_label), effective_chunk_size):
            chunk = samples_to_label[chunk_start:chunk_start + effective_chunk_size]
            latents_cache: Dict[int, torch.Tensor] = {}

            with dit_handler._load_model_context("vae"):
                with torch.inference_mode():
                    for j, (i, sample) in enumerate(chunk):
                        global_idx = chunk_start + j
                        if progress_callback and global_idx % 5 == 0:
                            progress_callback(f"VAE encoding {global_idx+1}/{total}: {sample.filename}")
                        try:
                            processed_audio = dit_handler.process_src_audio(sample.audio_path)
                            if processed_audio is None:
                                continue
                            if dit_handler.is_silence(processed_audio.unsqueeze(0)):
                                continue
                            latents = dit_handler._encode_audio_to_latents(processed_audio)
                            latents_cache[i] = latents.cpu()
                        except Exception as e:
                            logger.warning(f"VAE encode failed for {sample.filename}: {e}")

            with dit_handler._load_model_context("model"):
                try:
                    model = getattr(dit_handler, "model", None)
                    if model is None or not hasattr(model, "tokenize"):
                        raise RuntimeError("dit_handler.model is missing or has no tokenize()")

                    silence_latent = getattr(dit_handler, "silence_latent", None)
                    if silence_latent is None:
                        raise RuntimeError("dit_handler.silence_latent is missing")

                    target_device = dit_handler.device
                    if isinstance(target_device, str):
                        target_device = torch.device(target_device)
                    if silence_latent.device != target_device:
                        raise RuntimeError(
                            f"silence_latent on {silence_latent.device}, expected {target_device}"
                        )
                except Exception as e:
                    logger.error(f"Tokenize precheck failed: {e}")
                    for i, _ in chunk:
                        codes_cache[i] = None
                    continue

                def _pad_latents_to_length(lat: torch.Tensor, target_len: int) -> torch.Tensor:
                    if lat.shape[0] >= target_len:
                        return lat
                    pad_len = target_len - lat.shape[0]
                    pad_src = silence_latent[0]
                    if pad_src.shape[0] < pad_len:
                        repeat = (pad_len + pad_src.shape[0] - 1) // pad_src.shape[0]
                        pad_src = pad_src.repeat(repeat, 1)
                    pad = pad_src[:pad_len]
                    return torch.cat([lat, pad], dim=0)

                def _indices_to_codes(idx_tensor: torch.Tensor, pooled_mask: torch.Tensor) -> str:
                    valid = pooled_mask > 0
                    if idx_tensor.dim() == 1:
                        vals = idx_tensor[valid]
                    elif idx_tensor.dim() == 2:
                        vals = idx_tensor[valid].flatten()
                    else:
                        vals = idx_tensor.flatten()
                    vals_list = vals.detach().cpu().tolist()
                    return "".join(f"<|audio_code_{v}|>" for v in vals_list)

                pending = [(j, i, sample) for j, (i, sample) in enumerate(chunk) if i in latents_cache]

                with torch.inference_mode():
                    for start in range(0, len(pending), effective_batch_size):
                        micro = pending[start:start + effective_batch_size]
                        if not micro:
                            continue

                        max_len = max(latents_cache[i].shape[0] for _, i, _ in micro)
                        hidden_list = []
                        mask_list = []
                        ids = []
                        samples = []

                        for j, i, sample in micro:
                            ids.append(i)
                            samples.append(sample)
                            lat_cpu = latents_cache[i]
                            lat = lat_cpu.to(device=dit_handler.device, dtype=dit_handler.dtype)
                            orig_len = lat.shape[0]
                            lat = _pad_latents_to_length(lat, max_len)
                            hidden_list.append(lat)
                            mask = torch.zeros((max_len,), dtype=torch.bool, device=dit_handler.device)
                            mask[:orig_len] = True
                            mask_list.append(mask)

                        hidden_states = torch.stack(hidden_list, dim=0)
                        attention_mask = torch.stack(mask_list, dim=0)

                        global_idx = chunk_start + micro[0][0]
                        if progress_callback and global_idx % 5 == 0:
                            progress_callback(f"Tokenizing {global_idx+1}/{total}: {samples[0].filename}")

                        try:
                            _, indices, pooled_mask = model.tokenize(
                                hidden_states, silence_latent, attention_mask
                            )
                            for k, i in enumerate(ids):
                                codes_cache[i] = _indices_to_codes(indices[k], pooled_mask[k])
                        except Exception as e:
                            for k, i in enumerate(ids):
                                logger.warning(f"Tokenize failed for {samples[k].filename}: {e}")
                                codes_cache[i] = None

            del latents_cache

        return codes_cache

    def label_all_samples(
        self,
        dit_handler,
        llm_handler,
        format_lyrics: bool = False,
        transcribe_lyrics: bool = False,
        skip_metas: bool = False,
        only_unlabeled: bool = False,
        chunk_size: int = 16,
        batch_size: int = 1,
        progress_callback=None,
    ) -> Tuple[List[AudioSample], str]:
        """Label all samples in the dataset."""
        if not self.samples:
            return [], "❌ No samples to label. Please scan a directory first."

        if only_unlabeled:
            samples_to_label = [
                (i, s) for i, s in enumerate(self.samples) if not s.labeled or not s.caption
            ]
        else:
            samples_to_label = [(i, s) for i, s in enumerate(self.samples)]

        if not samples_to_label:
            return self.samples, "✅ All samples already labeled"

        total = len(samples_to_label)

        if progress_callback:
            progress_callback(f"Phase 1/{2}: Encoding audio for {total} samples...")

        codes_cache = self._batch_encode_audio_codes(
            samples_to_label=samples_to_label,
            dit_handler=dit_handler,
            progress_callback=progress_callback,
            total=total,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )

        if progress_callback:
            progress_callback(f"Phase 2/{2}: Labeling {total} samples with LLM...")

        success_count = 0
        fail_count = 0
        total = len(samples_to_label)

        for idx, (i, sample) in enumerate(samples_to_label):
            if progress_callback:
                progress_callback(f"Labeling {idx+1}/{total}: {sample.filename}")

            _, status = self._label_sample_with_codes(
                sample_idx=i,
                audio_codes=codes_cache.get(i),
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                format_lyrics=format_lyrics,
                transcribe_lyrics=transcribe_lyrics,
                skip_metas=skip_metas,
                progress_callback=progress_callback,
            )

            if "✅" in status:
                success_count += 1
            else:
                fail_count += 1

        status_msg = f"✅ Labeled {success_count}/{total} samples"
        if fail_count > 0:
            status_msg += f" ({fail_count} failed)"
        if only_unlabeled:
            status_msg += f" (unlabeled only, {len(self.samples)} total)"

        return self.samples, status_msg
