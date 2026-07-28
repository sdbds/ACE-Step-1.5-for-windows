"""Model-context audio-code encoding for low-memory dataset labeling."""

from typing import Any, Callable, Optional

import torch
from loguru import logger

from .label_audio_codes_tensor import (
    build_latent_batch,
    indices_to_audio_codes,
)
from .models import AudioSample


def encode_audio_codes_with_context(
    samples_to_label: list[tuple[int, AudioSample]],
    dit_handler: Any,
    progress_callback: Optional[Callable[[str], None]],
    total: int,
    chunk_size: int,
    batch_size: int,
) -> dict[int, Optional[str]]:
    """Encode samples in separate VAE and tokenizer model contexts."""
    codes_cache: dict[int, Optional[str]] = {}
    effective_chunk_size = max(1, int(chunk_size) if chunk_size else 1)
    effective_batch_size = max(1, int(batch_size) if batch_size else 1)

    for chunk_start in range(0, len(samples_to_label), effective_chunk_size):
        chunk = samples_to_label[chunk_start:chunk_start + effective_chunk_size]
        latents_cache = _encode_vae_chunk(
            chunk=chunk,
            chunk_start=chunk_start,
            dit_handler=dit_handler,
            progress_callback=progress_callback,
            total=total,
        )
        _tokenize_chunk(
            chunk=chunk,
            chunk_start=chunk_start,
            latents_cache=latents_cache,
            codes_cache=codes_cache,
            dit_handler=dit_handler,
            progress_callback=progress_callback,
            total=total,
            batch_size=effective_batch_size,
        )
        del latents_cache

    return codes_cache


def _encode_vae_chunk(
    chunk: list[tuple[int, AudioSample]],
    chunk_start: int,
    dit_handler: Any,
    progress_callback: Optional[Callable[[str], None]],
    total: int,
) -> dict[int, torch.Tensor]:
    """Encode one audio chunk to CPU latents inside the VAE context."""
    latents_cache: dict[int, torch.Tensor] = {}
    with dit_handler._load_model_context("vae"):
        with torch.inference_mode():
            for offset, (sample_idx, sample) in enumerate(chunk):
                global_offset = chunk_start + offset
                if progress_callback and global_offset % 5 == 0:
                    progress_callback(
                        f"VAE encoding {global_offset + 1}/{total}: {sample.filename}"
                    )
                try:
                    processed_audio = dit_handler.process_src_audio(sample.audio_path)
                    if processed_audio is None:
                        continue
                    if dit_handler.is_silence(processed_audio.unsqueeze(0)):
                        continue
                    latents = dit_handler._encode_audio_to_latents(processed_audio)
                    latents_cache[sample_idx] = latents.cpu()
                except Exception as exc:
                    logger.warning(f"VAE encode failed for {sample.filename}: {exc}")

    return latents_cache


def _tokenize_chunk(
    chunk: list[tuple[int, AudioSample]],
    chunk_start: int,
    latents_cache: dict[int, torch.Tensor],
    codes_cache: dict[int, Optional[str]],
    dit_handler: Any,
    progress_callback: Optional[Callable[[str], None]],
    total: int,
    batch_size: int,
) -> None:
    """Tokenize cached latents in microbatches inside the model context."""
    with dit_handler._load_model_context("model"):
        try:
            model, silence_latent = _validate_tokenizer_state(dit_handler)
        except Exception as exc:
            logger.error(f"Tokenize precheck failed: {exc}")
            for sample_idx, _sample in chunk:
                codes_cache[sample_idx] = None
            return

        pending = [
            (offset, sample_idx, sample)
            for offset, (sample_idx, sample) in enumerate(chunk)
            if sample_idx in latents_cache
        ]
        with torch.inference_mode():
            for start in range(0, len(pending), batch_size):
                microbatch = pending[start:start + batch_size]
                if not microbatch:
                    continue
                sample_indices, samples, hidden_states, attention_mask = build_latent_batch(
                    microbatch=microbatch,
                    latents_cache=latents_cache,
                    dit_handler=dit_handler,
                    silence_latent=silence_latent,
                )
                global_offset = chunk_start + microbatch[0][0]
                if progress_callback and global_offset % 5 == 0:
                    progress_callback(
                        f"Tokenizing {global_offset + 1}/{total} "
                        f"(bs={len(microbatch)}): {samples[0].filename}"
                    )
                try:
                    _, indices, pooled_mask = model.tokenize(
                        hidden_states,
                        silence_latent,
                        attention_mask,
                    )
                    for offset, sample_idx in enumerate(sample_indices):
                        codes_cache[sample_idx] = indices_to_audio_codes(
                            indices[offset],
                            pooled_mask[offset],
                        )
                except Exception as exc:
                    for offset, sample_idx in enumerate(sample_indices):
                        logger.warning(
                            f"Tokenize failed for {samples[offset].filename}: {exc}"
                        )
                        codes_cache[sample_idx] = None


def _validate_tokenizer_state(
    dit_handler: Any,
) -> tuple[Any, torch.Tensor]:
    """Validate tokenizer objects and device placement for the model context."""
    model = getattr(dit_handler, "model", None)
    if model is None or not hasattr(model, "tokenize"):
        raise RuntimeError("dit_handler.model is missing or has no tokenize()")

    silence_latent = getattr(dit_handler, "silence_latent", None)
    if silence_latent is None:
        raise RuntimeError("dit_handler.silence_latent is missing")

    if getattr(dit_handler, "offload_to_cpu", False):
        return model, silence_latent

    target_device = dit_handler.device
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
    silence_device = silence_latent.device
    if silence_device.type != target_device.type:
        raise RuntimeError(
            f"silence_latent on {silence_device}, expected {target_device}"
        )
    if (
        target_device.type == "cuda"
        and target_device.index is not None
        and silence_device.index is not None
        and silence_device.index != target_device.index
    ):
        raise RuntimeError(
            f"silence_latent on {silence_device}, expected {target_device}"
        )
    return model, silence_latent
