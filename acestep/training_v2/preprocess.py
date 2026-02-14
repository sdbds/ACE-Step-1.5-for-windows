"""
Two-Pass CLI Preprocessing for ACE-Step Training V2.

Converts raw audio files into ``.pt`` tensor files compatible with
``PreprocessedDataModule``.  Uses upstream sub-functions directly and
loads models **sequentially** to minimise peak VRAM:

    Pass 1 (Light ~3 GB):  VAE + Text Encoder  -> intermediate ``.tmp.pt``
    Pass 2 (Heavy ~6 GB):  DIT encoder          -> final ``.pt``

Input modes:
    * With ``--dataset-json``: rich per-sample metadata (lyrics, genre, BPM, …)
    * Without JSON: scan directory, default to ``[Instrumental]``, filename caption
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Supported audio extensions (same as upstream)
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}

# Target sample rate for ACE-Step models
_TARGET_SR = 48000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_audio_files(
    audio_dir: Optional[str],
    output_dir: str,
    checkpoint_dir: str,
    variant: str = "turbo",
    max_duration: float = 240.0,
    dataset_json: Optional[str] = None,
    device: str = "auto",
    precision: str = "auto",
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Preprocess audio files into .pt tensor format (two-pass pipeline).

    Audio files are discovered from one of two sources:

    * **Dataset JSON** (preferred): each entry's ``audio_path`` or
      ``filename`` field locates the audio file directly.
    * **Audio directory** (fallback): scanned **recursively** for
      supported audio formats when no JSON is provided.

    Args:
        audio_dir: Directory containing audio files (scanned recursively).
            May be ``None`` when *dataset_json* provides audio paths.
        output_dir: Directory for output .pt files.
        checkpoint_dir: Path to ACE-Step model checkpoints.
        variant: Model variant (turbo, base, sft).
        max_duration: Maximum audio duration in seconds.
        dataset_json: Optional JSON file with per-sample metadata and
            audio paths.
        device: Target device (``"auto"`` to auto-detect).
        precision: Target precision (``"auto"`` to auto-detect).
        progress_callback: ``(current, total, message) -> None``.
        cancel_check: ``() -> bool`` -- return True to cancel.

    Returns:
        Dict with keys: ``processed``, ``failed``, ``total``, ``output_dir``.
    """
    from acestep.training_v2.gpu_utils import detect_gpu

    gpu = detect_gpu(device, precision)
    dev = gpu.device
    prec = gpu.precision

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -- Discover audio files -----------------------------------------------
    audio_files = _discover_audio_files(audio_dir, dataset_json)
    if not audio_files:
        logger.warning("[Side-Step] No audio files found")
        return {"processed": 0, "failed": 0, "total": 0, "output_dir": str(out_path)}

    total = len(audio_files)
    logger.info("[Side-Step] Found %d audio files to preprocess", total)

    # -- Load per-sample metadata (optional) --------------------------------
    sample_meta = _load_sample_metadata(dataset_json, audio_files)

    # -- Pass 1: VAE + Text Encoder -----------------------------------------
    intermediates, pass1_failed = _pass1_light(
        audio_files=audio_files,
        sample_meta=sample_meta,
        out_path=out_path,
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=dev,
        precision=prec,
        max_duration=max_duration,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    # -- Pass 2: DIT Encoder ------------------------------------------------
    processed, pass2_failed = _pass2_heavy(
        intermediates=intermediates,
        out_path=out_path,
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=dev,
        precision=prec,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    failed = pass1_failed + pass2_failed
    result = {
        "processed": processed,
        "failed": failed,
        "total": total,
        "output_dir": str(out_path),
    }
    logger.info(
        "[Side-Step] Preprocessing complete: %d/%d processed, %d failed",
        processed, total, failed,
    )
    return result


# ---------------------------------------------------------------------------
# Audio file discovery
# ---------------------------------------------------------------------------

def _discover_audio_files(
    audio_dir: Optional[str],
    dataset_json: Optional[str],
) -> List[Path]:
    """Discover audio files from a dataset JSON or by scanning a directory.

    Resolution order:

    1. If *dataset_json* is provided, extract ``audio_path`` (or fall back
       to ``filename``) from each entry.  Missing files are skipped with a
       warning.
    2. Otherwise, recursively scan *audio_dir* for supported audio
       extensions (``_AUDIO_EXTENSIONS``).
    """
    # -- JSON-driven discovery ----------------------------------------------
    if dataset_json and Path(dataset_json).is_file():
        try:
            raw = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
            samples = raw if isinstance(raw, list) else raw.get("samples", [])
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[Side-Step] Failed to read dataset JSON: %s", exc)
            samples = []

        audio_files: List[Path] = []
        json_dir = Path(dataset_json).parent  # resolve relative paths vs JSON
        for entry in samples:
            ap = entry.get("audio_path") or entry.get("filename", "")
            if not ap:
                continue
            p = Path(ap)
            if not p.is_absolute():
                p = json_dir / p
            if p.is_file():
                audio_files.append(p)
            else:
                logger.warning("[Side-Step] Audio file from JSON not found: %s", p)

        if audio_files:
            logger.info(
                "[Side-Step] Resolved %d audio files from dataset JSON", len(audio_files),
            )
            return sorted(audio_files)
        else:
            logger.warning(
                "[Side-Step] Dataset JSON contained no resolvable audio paths; "
                "falling back to directory scan"
            )

    # -- Recursive directory scan -------------------------------------------
    if not audio_dir:
        return []

    source_path = Path(audio_dir)
    if not source_path.is_dir():
        logger.warning("[Side-Step] Audio directory does not exist: %s", audio_dir)
        return []

    audio_files = sorted(
        f for f in source_path.rglob("*")
        if f.is_file() and f.suffix.lower() in _AUDIO_EXTENSIONS
    )
    if audio_files:
        logger.info(
            "[Side-Step] Found %d audio files (recursive scan of %s)",
            len(audio_files), audio_dir,
        )
    return audio_files


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def _load_sample_metadata(
    dataset_json: Optional[str],
    audio_files: List[Path],
) -> Dict[str, Dict[str, Any]]:
    """Build a filename -> metadata mapping.

    If *dataset_json* is provided, load it and index by filename.
    Otherwise return defaults for every audio file.
    """
    meta: Dict[str, Dict[str, Any]] = {}

    if dataset_json and Path(dataset_json).is_file():
        try:
            raw = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
            samples = raw if isinstance(raw, list) else raw.get("samples", [])
            for s in samples:
                fname = s.get("filename", "")
                if fname:
                    meta[fname] = s
            logger.info("[Side-Step] Loaded metadata for %d samples from %s", len(meta), dataset_json)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[Side-Step] Failed to load dataset JSON: %s", exc)

    # Fill defaults for any audio file without metadata
    for af in audio_files:
        if af.name not in meta:
            meta[af.name] = {
                "filename": af.name,
                "caption": af.stem.replace("_", " ").replace("-", " "),
                "lyrics": "[Instrumental]",
                "genre": "",
                "bpm": None,
                "keyscale": "",
                "timesignature": "",
                "duration": 0,
                "is_instrumental": True,
            }

    return meta


# ---------------------------------------------------------------------------
# Tiled VAE encoding (standalone -- no handler dependency)
# ---------------------------------------------------------------------------

def _tiled_vae_encode(
    vae: Any,
    audio: torch.Tensor,
    dtype: torch.dtype,
    chunk_size: Optional[int] = None,
    overlap: int = 96000,
) -> torch.Tensor:
    """Encode audio through the VAE using overlap-discard tiling.

    Processes long audio in chunks to avoid OOM on the monolithic
    ``vae.encode()`` call.  Mirrors the tiling strategy from
    ``handler.tiled_encode`` but as a standalone function with no
    ``self`` / handler dependency.

    Args:
        vae: The ``AutoencoderOobleck`` VAE model (on device, in eval mode).
        audio: Audio tensor ``[B, C, S]`` (batch, channels, samples).
        dtype: Target dtype for the output latents.
        chunk_size: Audio samples per chunk.  ``None`` = auto-select
            based on available GPU memory (30 s for >=8 GB, 15 s otherwise).
        overlap: Overlap in audio samples between adjacent chunks
            (default 2 s at 48 kHz = 96 000).

    Returns:
        Latent tensor ``[B, T, 64]`` (same format as upstream
        ``vae_encode``), cast to *dtype*.
    """
    vae_device = next(vae.parameters()).device
    vae_dtype = vae.dtype

    # Auto-select chunk size based on GPU VRAM
    if chunk_size is None:
        gpu_mem_gb = 0.0
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(vae_device)
                gpu_mem_gb = props.total_mem / (1024 ** 3)
            except Exception:
                pass
        chunk_size = _TARGET_SR * 15 if gpu_mem_gb <= 8 else _TARGET_SR * 30

    B, C, S = audio.shape

    # Short audio -- direct encode (no tiling needed)
    if S <= chunk_size:
        vae_input = audio.to(vae_device, dtype=vae_dtype)
        with torch.inference_mode():
            latents = vae.encode(vae_input).latent_dist.sample()
        return latents.transpose(1, 2).to(dtype)

    # Calculate stride (core region per chunk, excluding overlap)
    stride = chunk_size - 2 * overlap
    if stride <= 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be > 2 * overlap ({overlap})"
        )

    num_steps = math.ceil(S / stride)
    downsample_factor: Optional[float] = None
    latent_write_pos = 0
    final_latents: Optional[torch.Tensor] = None

    for i in range(num_steps):
        core_start = i * stride
        core_end = min(core_start + stride, S)

        # Window with overlap on both sides
        win_start = max(0, core_start - overlap)
        win_end = min(S, core_end + overlap)

        chunk = audio[:, :, win_start:win_end].to(vae_device, dtype=vae_dtype)

        with torch.inference_mode():
            latent_chunk = vae.encode(chunk).latent_dist.sample()

        # Determine downsample factor from the first chunk
        if downsample_factor is None:
            downsample_factor = chunk.shape[-1] / latent_chunk.shape[-1]
            total_latent_len = int(round(S / downsample_factor))
            final_latents = torch.zeros(
                B, latent_chunk.shape[1], total_latent_len,
                dtype=latent_chunk.dtype, device="cpu",
            )

        # Trim the overlap regions from the latent
        added_start = core_start - win_start
        trim_start = int(round(added_start / downsample_factor))

        added_end = win_end - core_end
        trim_end = int(round(added_end / downsample_factor))

        lat_len = latent_chunk.shape[-1]
        end_idx = lat_len - trim_end if trim_end > 0 else lat_len
        latent_core = latent_chunk[:, :, trim_start:end_idx]

        # Copy to pre-allocated CPU tensor
        core_len = latent_core.shape[-1]
        assert final_latents is not None
        final_latents[:, :, latent_write_pos:latent_write_pos + core_len] = latent_core.cpu()
        latent_write_pos += core_len

        del chunk, latent_chunk, latent_core

    # Trim to actual written length
    assert final_latents is not None
    final_latents = final_latents[:, :, :latent_write_pos]

    # Transpose to (B, T, 64) and cast -- matches vae_encode output format
    return final_latents.transpose(1, 2).to(dtype)


# ---------------------------------------------------------------------------
# Pass 1 -- Light models (VAE + Text Encoder)
# ---------------------------------------------------------------------------

def _pass1_light(
    audio_files: List[Path],
    sample_meta: Dict[str, Dict[str, Any]],
    out_path: Path,
    checkpoint_dir: str,
    variant: str,
    device: str,
    precision: str,
    max_duration: float,
    progress_callback: Optional[Callable],
    cancel_check: Optional[Callable],
) -> tuple[List[Path], int]:
    """Load audio, VAE-encode, text-encode, save intermediates.

    Returns ``(list_of_intermediate_paths, fail_count)``.
    """
    from acestep.training_v2.model_loader import (
        load_vae,
        load_text_encoder,
        load_silence_latent,
        unload_models,
        _resolve_dtype,
    )
    from acestep.training.dataset_builder_modules.preprocess_audio import load_audio_stereo
    from acestep.training.dataset_builder_modules.preprocess_text import encode_text
    from acestep.training.dataset_builder_modules.preprocess_lyrics import encode_lyrics

    dtype = _resolve_dtype(precision)

    logger.info("[Side-Step] Pass 1/2: Loading VAE + Text Encoder ...")
    vae = load_vae(checkpoint_dir, device, precision)
    tokenizer, text_enc = load_text_encoder(checkpoint_dir, device, precision)
    silence_latent = load_silence_latent(checkpoint_dir, device, precision, variant=variant)

    intermediates: List[Path] = []
    failed = 0
    total = len(audio_files)

    try:
        for i, af in enumerate(audio_files):
            if cancel_check and cancel_check():
                logger.info("[Side-Step] Cancelled at %d/%d", i, total)
                break

            if progress_callback:
                progress_callback(i, total, f"[Pass 1] {af.name}")

            # Skip if final .pt already exists (resumable)
            final_pt = out_path / f"{af.stem}.pt"
            if final_pt.exists():
                logger.info("[Side-Step] Skipping (final exists): %s", af.name)
                continue

            try:
                # 1. Load audio (stereo, 48 kHz)
                audio, _sr = load_audio_stereo(str(af), _TARGET_SR, max_duration)
                audio = audio.unsqueeze(0).to(device).to(vae.dtype)

                # 2. VAE encode (tiled for long audio)
                with torch.no_grad():
                    target_latents = _tiled_vae_encode(vae, audio, dtype)

                latent_length = target_latents.shape[1]
                attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)

                # 3. Text encode
                sm = sample_meta.get(af.name, {})
                caption = sm.get("caption", af.stem)
                lyrics = sm.get("lyrics", "[Instrumental]")

                # Build a simple text prompt (matches upstream SFT_GEN_PROMPT format)
                text_prompt = _build_simple_prompt(sm)

                with torch.no_grad():
                    text_hs, text_mask = encode_text(text_enc, tokenizer, text_prompt, device, dtype)
                    lyric_hs, lyric_mask = encode_lyrics(text_enc, tokenizer, lyrics, device, dtype)

                # 4. Save intermediate
                tmp_path = out_path / f"{af.stem}.tmp.pt"
                torch.save({
                    "target_latents": target_latents.squeeze(0).cpu(),
                    "attention_mask": attention_mask.squeeze(0).cpu(),
                    "text_hidden_states": text_hs.cpu(),
                    "text_attention_mask": text_mask.cpu(),
                    "lyric_hidden_states": lyric_hs.cpu(),
                    "lyric_attention_mask": lyric_mask.cpu(),
                    "silence_latent": silence_latent.cpu(),
                    "latent_length": latent_length,
                    "metadata": {
                        "audio_path": str(af),
                        "filename": af.name,
                        "caption": caption,
                        "lyrics": lyrics,
                        "duration": sm.get("duration", 0),
                        "bpm": sm.get("bpm"),
                        "keyscale": sm.get("keyscale", ""),
                        "timesignature": sm.get("timesignature", ""),
                        "is_instrumental": sm.get("is_instrumental", True),
                        "preprocess_mode": "lora",
                    },
                }, tmp_path)
                intermediates.append(tmp_path)
                logger.info("[Side-Step] Pass 1 OK: %s", af.name)

            except Exception as exc:
                failed += 1
                logger.error("[Side-Step] Pass 1 FAIL %s: %s", af.name, exc)

    finally:
        logger.info("[Side-Step] Unloading VAE + Text Encoder ...")
        unload_models(vae, text_enc, tokenizer, silence_latent)

    if progress_callback:
        progress_callback(total, total, "[Pass 1] Done")

    return intermediates, failed


# ---------------------------------------------------------------------------
# Pass 2 -- Heavy model (DIT encoder)
# ---------------------------------------------------------------------------

def _pass2_heavy(
    intermediates: List[Path],
    out_path: Path,
    checkpoint_dir: str,
    variant: str,
    device: str,
    precision: str,
    progress_callback: Optional[Callable],
    cancel_check: Optional[Callable],
) -> tuple[int, int]:
    """Run DIT encoder on intermediates and write final .pt files.

    Returns ``(processed_count, fail_count)``.
    """
    if not intermediates:
        return 0, 0

    from acestep.training_v2.model_loader import (
        load_decoder_for_training,
        unload_models,
        _resolve_dtype,
    )
    from acestep.training.dataset_builder_modules.preprocess_encoder import run_encoder
    from acestep.training.dataset_builder_modules.preprocess_context import build_context_latents

    dtype = _resolve_dtype(precision)

    logger.info("[Side-Step] Pass 2/2: Loading DIT model (variant=%s) ...", variant)
    model = load_decoder_for_training(checkpoint_dir, variant, device, precision)

    processed = 0
    failed = 0
    total = len(intermediates)

    try:
        for i, tmp_path in enumerate(intermediates):
            if cancel_check and cancel_check():
                logger.info("[Side-Step] Cancelled at %d/%d", i, total)
                break

            if progress_callback:
                progress_callback(i, total, f"[Pass 2] {tmp_path.stem}")

            try:
                data = torch.load(str(tmp_path), weights_only=False)

                text_hs = data["text_hidden_states"].to(device, dtype=dtype)
                text_mask = data["text_attention_mask"].to(device, dtype=dtype)
                lyric_hs = data["lyric_hidden_states"].to(device, dtype=dtype)
                lyric_mask = data["lyric_attention_mask"].to(device, dtype=dtype)
                silence_latent = data["silence_latent"].to(device, dtype=dtype)
                latent_length = data["latent_length"]

                # Move model tensors to model device if needed
                model_device = next(model.parameters()).device
                model_dtype = next(model.parameters()).dtype
                text_hs = text_hs.to(model_device, dtype=model_dtype)
                text_mask = text_mask.to(model_device, dtype=model_dtype)
                lyric_hs = lyric_hs.to(model_device, dtype=model_dtype)
                lyric_mask = lyric_mask.to(model_device, dtype=model_dtype)

                # DIT encoder pass
                encoder_hs, encoder_mask = run_encoder(
                    model,
                    text_hidden_states=text_hs,
                    text_attention_mask=text_mask,
                    lyric_hidden_states=lyric_hs,
                    lyric_attention_mask=lyric_mask,
                    device=str(model_device),
                    dtype=model_dtype,
                )

                # Build context latents
                silence_latent = silence_latent.to(model_device, dtype=model_dtype)
                if silence_latent.dim() == 2:
                    silence_latent = silence_latent.unsqueeze(0)
                context_latents = build_context_latents(
                    silence_latent, latent_length, str(model_device), model_dtype,
                )

                # Write final .pt  (strip ".tmp" from "song.tmp.pt" -> "song.pt")
                base_name = tmp_path.name.replace(".tmp.pt", ".pt")
                final_path = out_path / base_name
                torch.save({
                    "target_latents": data["target_latents"],
                    "attention_mask": data["attention_mask"],
                    "encoder_hidden_states": encoder_hs.squeeze(0).cpu(),
                    "encoder_attention_mask": encoder_mask.squeeze(0).cpu(),
                    "context_latents": context_latents.squeeze(0).cpu(),
                    "metadata": data["metadata"],
                }, final_path)

                # Remove intermediate
                tmp_path.unlink(missing_ok=True)

                processed += 1
                logger.info("[Side-Step] Pass 2 OK: %s", tmp_path.stem)

            except Exception as exc:
                failed += 1
                logger.error("[Side-Step] Pass 2 FAIL %s: %s", tmp_path.stem, exc)

    finally:
        logger.info("[Side-Step] Unloading DIT model ...")
        unload_models(model)

    if progress_callback:
        progress_callback(total, total, "[Pass 2] Done")

    return processed, failed


# ---------------------------------------------------------------------------
# Prompt builder (simplified -- no DatasetBuilder / AudioSample dependency)
# ---------------------------------------------------------------------------

def _build_simple_prompt(meta: Dict[str, Any]) -> str:
    """Build a text prompt from sample metadata.

    Mimics the upstream ``build_text_prompt`` + ``build_metas_str`` but
    without requiring an ``AudioSample`` dataclass or ``DatasetBuilder``.
    """
    from acestep.constants import DEFAULT_DIT_INSTRUCTION, SFT_GEN_PROMPT

    caption = meta.get("caption", "")
    bpm = meta.get("bpm", "N/A") or "N/A"
    ts = meta.get("timesignature", "N/A") or "N/A"
    ks = meta.get("keyscale", "N/A") or "N/A"
    dur = meta.get("duration", 0)

    metas_str = (
        f"- bpm: {bpm}\n"
        f"- timesignature: {ts}\n"
        f"- keyscale: {ks}\n"
        f"- duration: {dur} seconds\n"
    )
    return SFT_GEN_PROMPT.format(DEFAULT_DIT_INSTRUCTION, caption, metas_str)
