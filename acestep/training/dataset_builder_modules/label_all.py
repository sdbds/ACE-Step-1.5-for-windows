"""Bulk dataset labeling orchestration."""

from typing import Callable, List, Optional, Tuple

from loguru import logger

from .label_audio_codes import encode_audio_codes
from .models import AudioSample


class LabelAllMixin:
    """Label all samples in the dataset."""

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
        sample_labeled_callback: Optional[Callable[[int, AudioSample, str], None]] = None,
        on_phase_complete: Optional[Callable[[int], None]] = None,
    ) -> Tuple[List[AudioSample], str]:
        """Encode and label samples, notifying callers after the encoding phase."""
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
            progress_callback(f"Phase 1/2: Encoding audio for {total} samples...")

        codes_cache = encode_audio_codes(
            samples_to_label=samples_to_label,
            dit_handler=dit_handler,
            progress_callback=progress_callback,
            total=total,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )

        if on_phase_complete:
            on_phase_complete(1)

        if progress_callback:
            progress_callback(f"Phase 2/2: Labeling {total} samples with LLM...")

        success_count = 0
        fail_count = 0

        for idx, (i, sample) in enumerate(samples_to_label):
            if progress_callback:
                progress_callback(f"Labeling {idx+1}/{total}: {sample.filename}")

            labeled_sample, status = self._label_sample_with_codes(
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

            if sample_labeled_callback is not None and labeled_sample is not None:
                try:
                    sample_labeled_callback(i, labeled_sample, status)
                except Exception:
                    logger.exception("sample_labeled_callback failed")

        status_msg = f"✅ Labeled {success_count}/{total} samples"
        if fail_count > 0:
            status_msg += f" ({fail_count} failed)"
        if only_unlabeled:
            status_msg += f" (unlabeled only, {len(self.samples)} total)"

        return self.samples, status_msg
