"""Tensor preparation helpers for context-aware audio-code encoding."""

from typing import Any

import torch

from .models import AudioSample


def build_latent_batch(
    microbatch: list[tuple[int, int, AudioSample]],
    latents_cache: dict[int, torch.Tensor],
    dit_handler: Any,
    silence_latent: torch.Tensor,
) -> tuple[list[int], list[AudioSample], torch.Tensor, torch.Tensor]:
    """Pad cached latents and stack them with masks for model tokenization."""
    max_length = max(latents_cache[sample_idx].shape[0] for _, sample_idx, _ in microbatch)
    sample_indices: list[int] = []
    samples: list[AudioSample] = []
    hidden_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []

    for _offset, sample_idx, sample in microbatch:
        sample_indices.append(sample_idx)
        samples.append(sample)
        latent = latents_cache[sample_idx].to(
            device=dit_handler.device,
            dtype=dit_handler.dtype,
        )
        original_length = latent.shape[0]
        hidden_list.append(_pad_latent(latent, max_length, silence_latent))
        mask = torch.zeros(
            (max_length,),
            dtype=torch.bool,
            device=dit_handler.device,
        )
        mask[:original_length] = True
        mask_list.append(mask)

    return (
        sample_indices,
        samples,
        torch.stack(hidden_list, dim=0),
        torch.stack(mask_list, dim=0),
    )


def indices_to_audio_codes(
    indices: torch.Tensor,
    pooled_mask: torch.Tensor,
) -> str:
    """Format valid tokenizer indices as the LLM audio-code token string."""
    valid = pooled_mask > 0
    if indices.dim() == 1:
        values = indices[valid]
    elif indices.dim() == 2:
        values = indices[valid].flatten()
    else:
        values = indices.flatten()
    return "".join(
        f"<|audio_code_{value}|>"
        for value in values.detach().cpu().tolist()
    )


def _pad_latent(
    latent: torch.Tensor,
    target_length: int,
    silence_latent: torch.Tensor,
) -> torch.Tensor:
    """Pad a latent sequence to the target length with silence frames."""
    if latent.shape[0] >= target_length:
        return latent

    pad_length = target_length - latent.shape[0]
    pad_source = silence_latent[0]
    if pad_source.shape[0] < pad_length:
        repeat = (pad_length + pad_source.shape[0] - 1) // pad_source.shape[0]
        pad_source = pad_source.repeat(repeat, 1)
    return torch.cat([latent, pad_source[:pad_length]], dim=0)
