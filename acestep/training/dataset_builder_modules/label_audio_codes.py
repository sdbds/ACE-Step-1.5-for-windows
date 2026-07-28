"""Audio-code encoding strategies for bulk dataset labeling."""

from typing import Any, Callable, Optional

from loguru import logger

from .models import AudioSample


def encode_audio_codes(
    samples_to_label: list[tuple[int, AudioSample]],
    dit_handler: Any,
    progress_callback: Optional[Callable[[str], None]],
    total: int,
    chunk_size: int,
    batch_size: int,
) -> dict[int, Optional[str]]:
    """Encode audio codes using the handler's direct or model-context path."""
    if not hasattr(dit_handler, "convert_src_audio_to_codes"):
        logger.error("DiT handler missing convert_src_audio_to_codes method")
        return {sample_idx: None for sample_idx, _sample in samples_to_label}

    if hasattr(dit_handler, "_load_model_context"):
        from .label_audio_codes_context import encode_audio_codes_with_context

        return encode_audio_codes_with_context(
            samples_to_label=samples_to_label,
            dit_handler=dit_handler,
            progress_callback=progress_callback,
            total=total,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )

    return _encode_audio_codes_direct(
        samples_to_label=samples_to_label,
        dit_handler=dit_handler,
        progress_callback=progress_callback,
        total=total,
    )


def _encode_audio_codes_direct(
    samples_to_label: list[tuple[int, AudioSample]],
    dit_handler: Any,
    progress_callback: Optional[Callable[[str], None]],
    total: int,
) -> dict[int, Optional[str]]:
    """Encode samples one at a time when model contexts are unavailable."""
    import torch

    codes_cache: dict[int, Optional[str]] = {}
    with torch.inference_mode():
        for offset, (sample_idx, sample) in enumerate(samples_to_label):
            if progress_callback and offset % 5 == 0:
                progress_callback(f"Encoding {offset + 1}/{total}: {sample.filename}")
            try:
                codes = dit_handler.convert_src_audio_to_codes(sample.audio_path)
                if codes and str(codes).startswith("❌"):
                    codes = None
                codes_cache[sample_idx] = codes
            except Exception:
                logger.exception(f"Failed to convert audio to codes: {sample.filename}")
                codes_cache[sample_idx] = None

    return codes_cache
