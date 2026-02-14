"""Unit tests for generation input event handlers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from acestep.gradio_ui.events import generation_handlers
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependency guard
    generation_handlers = None
    _IMPORT_ERROR = exc


class _FakeDitHandler:
    """Minimal DiT handler stub for analyze-src-audio tests."""

    def __init__(self, convert_result):
        self._convert_result = convert_result

    def convert_src_audio_to_codes(self, _src_audio):
        """Return configured conversion output."""
        return self._convert_result


@unittest.skipIf(generation_handlers is None, f"generation_handlers import unavailable: {_IMPORT_ERROR}")
class GenerationHandlersTests(unittest.TestCase):
    """Tests for source-audio analysis validation behavior."""

    @patch("acestep.gradio_ui.events.generation_handlers.gr.Warning")
    @patch("acestep.gradio_ui.events.generation_handlers.understand_music")
    def test_analyze_src_audio_rejects_non_audio_code_output(
        self,
        understand_music_mock,
        warning_mock,
    ):
        """Reject conversion output that has no serialized audio-code tokens."""
        dit_handler = _FakeDitHandler("ERROR: not an audio file")
        llm_handler = SimpleNamespace(llm_initialized=True)

        result = generation_handlers.analyze_src_audio(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            src_audio="fake.mp3",
            constrained_decoding_debug=False,
        )

        self.assertEqual(result, ("", "", "", "", None, None, "", "", "", False))
        understand_music_mock.assert_not_called()
        warning_mock.assert_called_once()

    @patch("acestep.gradio_ui.events.generation_handlers.gr.Warning")
    @patch("acestep.gradio_ui.events.generation_handlers.understand_music")
    def test_analyze_src_audio_allows_valid_audio_code_output(
        self,
        understand_music_mock,
        warning_mock,
    ):
        """Pass valid audio codes through to LM understanding."""
        dit_handler = _FakeDitHandler("<|audio_code_123|><|audio_code_456|>")
        llm_handler = SimpleNamespace(llm_initialized=True)
        understand_music_mock.return_value = SimpleNamespace(
            success=True,
            status_message="ok",
            caption="caption",
            lyrics="lyrics",
            bpm=120,
            duration=30.0,
            keyscale="C major",
            language="en",
            timesignature="4",
        )

        result = generation_handlers.analyze_src_audio(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            src_audio="real.mp3",
            constrained_decoding_debug=False,
        )

        self.assertEqual(result[0], "<|audio_code_123|><|audio_code_456|>")
        self.assertEqual(result[1], "ok")
        understand_music_mock.assert_called_once()
        warning_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
