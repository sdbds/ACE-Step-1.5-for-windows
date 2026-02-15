"""Unit tests for generation input event handlers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

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

    @patch("acestep.gradio_ui.events.generation_handlers.get_global_gpu_config")
    @patch("acestep.gradio_ui.events.generation_handlers.get_model_type_ui_settings")
    def test_init_service_wrapper_preserves_batch_size(
        self,
        get_model_type_ui_settings_mock,
        get_global_gpu_config_mock,
    ):
        """Verify that init_service_wrapper preserves current batch_size when provided."""
        # Setup mocks
        gpu_config_mock = MagicMock()
        gpu_config_mock.max_batch_size_with_lm = 8
        gpu_config_mock.max_batch_size_without_lm = 4
        gpu_config_mock.max_duration_with_lm = 600
        gpu_config_mock.max_duration_without_lm = 300
        gpu_config_mock.tier = "tier5"
        gpu_config_mock.available_lm_models = ["acestep-5Hz-lm-1.7B"]
        get_global_gpu_config_mock.return_value = gpu_config_mock

        get_model_type_ui_settings_mock.return_value = (None,) * 9  # 9 model type settings

        dit_handler = MagicMock()
        dit_handler.model = MagicMock()
        dit_handler.is_turbo_model.return_value = True
        dit_handler.initialize_service.return_value = ("Success", True)

        llm_handler = MagicMock()
        llm_handler.llm_initialized = True

        # Test with current_batch_size = 5
        result = generation_handlers.init_service_wrapper(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            checkpoint=None,
            config_path="acestep-v15-turbo",
            device="cuda",
            init_llm=True,
            lm_model_path=None,
            backend="vllm",
            use_flash_attention=True,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            compile_model=False,
            quantization=False,
            mlx_dit=False,
            current_mode="Custom",
            current_batch_size=5,
        )

        # Result is a tuple: (status, btn_update, accordion, *model_settings, duration_update, batch_update, think_update)
        # batch_update is at index -2 (second to last)
        batch_update = result[-2]
        
        # Verify batch_update preserves the value 5 (clamped to max_batch of 8)
        self.assertEqual(batch_update["value"], 5)
        self.assertEqual(batch_update["maximum"], 8)

    @patch("acestep.gradio_ui.events.generation_handlers.get_global_gpu_config")
    @patch("acestep.gradio_ui.events.generation_handlers.get_model_type_ui_settings")
    def test_init_service_wrapper_defaults_batch_size_when_none(
        self,
        get_model_type_ui_settings_mock,
        get_global_gpu_config_mock,
    ):
        """Verify that init_service_wrapper uses default batch_size when current_batch_size is None."""
        # Setup mocks
        gpu_config_mock = MagicMock()
        gpu_config_mock.max_batch_size_with_lm = 8
        gpu_config_mock.max_batch_size_without_lm = 4
        gpu_config_mock.max_duration_with_lm = 600
        gpu_config_mock.max_duration_without_lm = 300
        gpu_config_mock.tier = "tier5"
        gpu_config_mock.available_lm_models = ["acestep-5Hz-lm-1.7B"]
        get_global_gpu_config_mock.return_value = gpu_config_mock

        get_model_type_ui_settings_mock.return_value = (None,) * 9

        dit_handler = MagicMock()
        dit_handler.model = MagicMock()
        dit_handler.is_turbo_model.return_value = True
        dit_handler.initialize_service.return_value = ("Success", True)

        llm_handler = MagicMock()
        llm_handler.llm_initialized = True

        # Test with current_batch_size = None (should default to 2)
        result = generation_handlers.init_service_wrapper(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            checkpoint=None,
            config_path="acestep-v15-turbo",
            device="cuda",
            init_llm=True,
            lm_model_path=None,
            backend="vllm",
            use_flash_attention=True,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            compile_model=False,
            quantization=False,
            mlx_dit=False,
            current_mode="Custom",
            current_batch_size=None,
        )

        batch_update = result[-2]
        
        # Verify batch_update defaults to min(2, max_batch)
        self.assertEqual(batch_update["value"], 2)
        self.assertEqual(batch_update["maximum"], 8)


if __name__ == "__main__":
    unittest.main()
