"""Tests for bulk dataset labeling orchestration."""

import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType

from loguru import logger

_PACKAGE_NAME = "_label_all_test_modules"
_PACKAGE_PATH = Path(__file__).resolve().parent
_PACKAGE = ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(_PACKAGE_PATH)]
sys.modules[_PACKAGE_NAME] = _PACKAGE

LabelAllMixin = importlib.import_module(f"{_PACKAGE_NAME}.label_all").LabelAllMixin
AudioSample = importlib.import_module(f"{_PACKAGE_NAME}.models").AudioSample


class _RecordingHandler:
    """Record audio-code conversion order for orchestration tests."""

    def __init__(self, events: list[tuple]) -> None:
        """Store the shared event log."""
        self.events = events

    def convert_src_audio_to_codes(self, audio_path: str) -> str:
        """Return deterministic audio codes and record the conversion."""
        self.events.append(("encode", audio_path))
        return f"codes:{audio_path}"


class _LabelAllHost(LabelAllMixin):
    """Provide concrete labeling methods required by ``LabelAllMixin``."""

    def __init__(
        self,
        samples: list[AudioSample],
        events: list[tuple],
        failed_indices: set[int] | None = None,
    ) -> None:
        """Store samples and the shared event log."""
        self.samples = samples
        self.events = events
        self.failed_indices = failed_indices or set()

    def label_sample(
        self,
        sample_idx,
        dit_handler,
        llm_handler,
        format_lyrics,
        transcribe_lyrics,
        skip_metas,
        progress_callback,
    ):
        """Model the upstream per-sample encode-and-label behavior."""
        sample = self.samples[sample_idx]
        codes = dit_handler.convert_src_audio_to_codes(sample.audio_path)
        return self._label_sample_with_codes(
            sample_idx,
            codes,
            dit_handler,
            llm_handler,
            format_lyrics,
            transcribe_lyrics,
            skip_metas,
            progress_callback,
        )

    def _label_sample_with_codes(
        self,
        sample_idx,
        audio_codes,
        dit_handler,
        llm_handler,
        format_lyrics,
        transcribe_lyrics,
        skip_metas,
        progress_callback,
    ):
        """Record labeling with precomputed codes and mark the sample labeled."""
        del dit_handler, llm_handler, format_lyrics
        del transcribe_lyrics, skip_metas, progress_callback
        sample = self.samples[sample_idx]
        self.events.append(("label", sample.audio_path, audio_codes))
        if sample_idx in self.failed_indices:
            return sample, f"❌ Failed: {sample.filename}"
        sample.labeled = True
        return sample, f"✅ Labeled: {sample.filename}"


class TestLabelAllMixin(unittest.TestCase):
    """Verify bulk labeling preserves local batching and upstream callbacks."""

    def test_encodes_all_samples_before_labeling_and_calls_callback_once(self) -> None:
        """Catch per-sample interleaving and duplicate incremental-save callbacks."""
        events: list[tuple] = []
        samples = [
            AudioSample(audio_path="first.wav", filename="first.wav"),
            AudioSample(audio_path="second.wav", filename="second.wav"),
        ]
        callback_indices: list[int] = []
        host = _LabelAllHost(samples, events)

        labeled_samples, status = host.label_all_samples(
            dit_handler=_RecordingHandler(events),
            llm_handler=object(),
            sample_labeled_callback=lambda index, _sample, _status: callback_indices.append(index),
        )

        self.assertEqual(
            [
                ("encode", "first.wav"),
                ("encode", "second.wav"),
                ("label", "first.wav", "codes:first.wav"),
                ("label", "second.wav", "codes:second.wav"),
            ],
            events,
        )
        self.assertEqual([0, 1], callback_indices)
        self.assertIs(samples, labeled_samples)
        self.assertEqual("✅ Labeled 2/2 samples", status)

    def test_callback_failure_does_not_stop_remaining_samples(self) -> None:
        """Catch callback failures escaping and aborting later labeling."""
        events: list[tuple] = []
        samples = [
            AudioSample(audio_path="first.wav", filename="first.wav"),
            AudioSample(audio_path="second.wav", filename="second.wav"),
        ]
        callback_indices: list[int] = []
        host = _LabelAllHost(samples, events, failed_indices={0})

        def failing_callback(index: int, _sample: AudioSample, _status: str) -> None:
            """Fail only for the first incrementally saved sample."""
            callback_indices.append(index)
            if index == 0:
                raise RuntimeError("save failed")

        logger.disable(_PACKAGE_NAME)
        try:
            try:
                labeled_samples, status = host.label_all_samples(
                    dit_handler=_RecordingHandler(events),
                    llm_handler=object(),
                    sample_labeled_callback=failing_callback,
                )
            except RuntimeError as exc:
                self.fail(f"callback exception escaped: {exc}")
        finally:
            logger.enable(_PACKAGE_NAME)

        self.assertTrue(samples[1].labeled)
        self.assertEqual([0, 1], callback_indices)
        self.assertIs(samples, labeled_samples)
        self.assertEqual("✅ Labeled 1/2 samples (1 failed)", status)

    def test_phase_completion_runs_between_encoding_and_labeling(self) -> None:
        """Catch model-offload hooks running before encoding or after labeling."""
        events: list[tuple] = []
        samples = [
            AudioSample(audio_path="first.wav", filename="first.wav"),
            AudioSample(audio_path="second.wav", filename="second.wav"),
        ]
        host = _LabelAllHost(samples, events)

        def phase_complete(phase: int) -> None:
            """Record the completed labeling phase."""
            events.append(("phase", phase))

        try:
            host.label_all_samples(
                dit_handler=_RecordingHandler(events),
                llm_handler=object(),
                on_phase_complete=phase_complete,
            )
        except TypeError as exc:
            self.fail(f"phase completion hook is unsupported: {exc}")

        self.assertEqual(
            [
                ("encode", "first.wav"),
                ("encode", "second.wav"),
                ("phase", 1),
                ("label", "first.wav", "codes:first.wav"),
                ("label", "second.wav", "codes:second.wav"),
            ],
            events,
        )


if __name__ == "__main__":
    unittest.main()
