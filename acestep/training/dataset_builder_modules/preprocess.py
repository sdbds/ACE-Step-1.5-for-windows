import os
from typing import List, Tuple

import torch
from loguru import logger

from .models import AudioSample
from .preprocess_audio import load_audio_stereo
from .preprocess_context import build_context_latents
from .preprocess_encoder import run_encoder
from .preprocess_lyrics import encode_lyrics
from .preprocess_manifest import save_manifest
from .preprocess_text import build_text_prompt, encode_text
from .preprocess_utils import select_genre_indices
from .preprocess_vae import vae_encode


class PreprocessMixin:
    """Preprocess labeled samples to tensor files."""

    def preprocess_to_tensors(
        self,
        dit_handler,
        output_dir: str,
        max_duration: float = 240.0,
        progress_callback=None,
    ) -> Tuple[List[str], str]:
        """Preprocess all labeled samples to tensor files for efficient training."""
        if not self.samples:
            return [], "❌ No samples to preprocess"

        labeled_samples = [s for s in self.samples if s.labeled]
        if not labeled_samples:
            return [], "❌ No labeled samples to preprocess"

        if dit_handler is None or dit_handler.model is None:
            return [], "❌ Model not initialized. Please initialize the service first."

        os.makedirs(output_dir, exist_ok=True)

        output_paths: List[str] = []
        success_count = 0
        fail_count = 0

        model = dit_handler.model
        vae = dit_handler.vae
        text_encoder = dit_handler.text_encoder
        text_tokenizer = dit_handler.text_tokenizer
        silence_latent = dit_handler.silence_latent
        device = dit_handler.device
        dtype = dit_handler.dtype

        target_sample_rate = 48000

        genre_indices = select_genre_indices(labeled_samples, self.metadata.genre_ratio)

        for i, sample in enumerate(labeled_samples):
            try:
                if progress_callback:
                    progress_callback(f"Preprocessing {i+1}/{len(labeled_samples)}: {sample.filename}")

                use_genre = i in genre_indices

                audio, _ = load_audio_stereo(sample.audio_path, target_sample_rate, max_duration)
                audio = audio.unsqueeze(0).to(device).to(vae.dtype)

                with torch.no_grad():
                    target_latents = vae_encode(vae, audio, dtype)

                latent_length = target_latents.shape[1]
                attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)

                caption = sample.get_training_prompt(self.metadata.tag_position, use_genre=use_genre)
                text_prompt = build_text_prompt(sample, self.metadata.tag_position, use_genre)

                if i == 0:
                    logger.info(f"\n{'='*70}")
                    logger.info("🔍 [DEBUG] DiT TEXT ENCODER INPUT (Training Preprocess)")
                    logger.info(f"{'='*70}")
                    logger.info(f"text_prompt:\n{text_prompt}")
                    logger.info(f"{'='*70}\n")

                text_hidden_states, text_attention_mask = encode_text(
                    text_encoder, text_tokenizer, text_prompt, device, dtype
                )

                lyrics = sample.lyrics if sample.lyrics else "[Instrumental]"
                lyric_hidden_states, lyric_attention_mask = encode_lyrics(
                    text_encoder, text_tokenizer, lyrics, device, dtype
                )

                encoder_hidden_states, encoder_attention_mask = run_encoder(
                    model,
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    device=device,
                    dtype=dtype,
                )

                context_latents = build_context_latents(silence_latent, latent_length, device, dtype)

                output_data = {
                    "target_latents": target_latents.squeeze(0).cpu(),
                    "attention_mask": attention_mask.squeeze(0).cpu(),
                    "encoder_hidden_states": encoder_hidden_states.squeeze(0).cpu(),
                    "encoder_attention_mask": encoder_attention_mask.squeeze(0).cpu(),
                    "context_latents": context_latents.squeeze(0).cpu(),
                    "metadata": {
                        "audio_path": sample.audio_path,
                        "filename": sample.filename,
                        "caption": caption,
                        "lyrics": lyrics,
                        "duration": sample.duration,
                        "bpm": sample.bpm,
                        "keyscale": sample.keyscale,
                        "timesignature": sample.timesignature,
                        "language": sample.language,
                        "is_instrumental": sample.is_instrumental,
                    },
                }

                output_path = os.path.join(output_dir, f"{sample.id}.pt")
                torch.save(output_data, output_path)
                output_paths.append(output_path)
                success_count += 1

            except Exception as e:
                logger.exception(f"Error preprocessing {sample.filename}")
                fail_count += 1
                if progress_callback:
                    progress_callback(f"❌ Failed: {sample.filename}: {str(e)}")

        save_manifest(output_dir, self.metadata, output_paths)

        status = f"✅ Preprocessed {success_count}/{len(labeled_samples)} samples to {output_dir}"
        if fail_count > 0:
            status += f" ({fail_count} failed)"

        return output_paths, status
