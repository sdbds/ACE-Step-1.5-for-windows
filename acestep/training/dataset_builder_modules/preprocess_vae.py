import torch


def vae_encode(vae, audio, dtype):
    """VAE encode audio to get target latents.

    Note: Caller is responsible for ensuring audio is on the correct device
    and dtype before calling this function (avoids per-call device probing).
    """
    latent = vae.encode(audio).latent_dist.sample()
    target_latents = latent.transpose(1, 2).to(dtype)
    return target_latents
