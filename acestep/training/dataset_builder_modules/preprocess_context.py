import torch


def build_context_latents(silence_latent, latent_length: int, device, dtype):
    """Build context latents for text2music."""
    context_latents = torch.empty((1, latent_length, 128), device=device, dtype=dtype)

    src = silence_latent
    if src.dtype != dtype:
        src = src.to(dtype)

    src_len = src.shape[1]
    take = min(latent_length, src_len)
    context_latents[:, :take, :64] = src[:, :take, :]

    if take < latent_length:
        remaining = latent_length - take
        pad_src = src[:, :min(src_len, remaining), :]
        while remaining > 0:
            chunk = min(remaining, pad_src.shape[1])
            context_latents[:, take : take + chunk, :64] = pad_src[:, :chunk, :]
            take += chunk
            remaining -= chunk

    context_latents[:, :, 64:] = 1
    return context_latents
