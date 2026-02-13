"""Unit tests for LLMHandler._compute_max_new_tokens.

Validates that the progress bar total (max_new_tokens) is computed correctly
for both the CoT and codes generation phases, ensuring the progress bar
reaches ~100% instead of appearing to finish early.

Also validates that duration clamping uses DURATION_MIN / DURATION_MAX from
constants and respects GPU-config-dependent max_duration bounds.
"""

import unittest
from unittest.mock import patch, MagicMock

from acestep.constants import DURATION_MIN, DURATION_MAX


class TestComputeMaxNewTokens(unittest.TestCase):
    """Tests for _compute_max_new_tokens helper method."""

    def _make_handler(self, max_model_len: int = 4096):
        """Create a minimal LLMHandler with mocked dependencies."""
        from acestep.llm_inference import LLMHandler

        handler = LLMHandler.__new__(LLMHandler)
        handler.max_model_len = max_model_len
        return handler

    @staticmethod
    def _mock_gpu_config(max_duration_with_lm: int = DURATION_MAX):
        """Create a mock GPUConfig with a specific max_duration_with_lm."""
        cfg = MagicMock()
        cfg.max_duration_with_lm = max_duration_with_lm
        return cfg

    # ------------------------------------------------------------------
    # Codes phase: should use target_codes + 10 (small buffer)
    # ------------------------------------------------------------------

    def test_codes_phase_195s(self):
        """195s duration in codes phase -> 975 + 10 = 985."""
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(DURATION_MAX)):
            result = handler._compute_max_new_tokens(
                target_duration=195.0, generation_phase="codes"
            )
        self.assertEqual(result, 985)

    def test_codes_phase_60s(self):
        """60s duration in codes phase -> 300 + 10 = 310."""
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(DURATION_MAX)):
            result = handler._compute_max_new_tokens(
                target_duration=60.0, generation_phase="codes"
            )
        self.assertEqual(result, 310)

    # ------------------------------------------------------------------
    # CoT phase: should use target_codes + 500 (large buffer for metadata)
    # ------------------------------------------------------------------

    def test_cot_phase_195s(self):
        """195s duration in cot phase -> 975 + 500 = 1475."""
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(DURATION_MAX)):
            result = handler._compute_max_new_tokens(
                target_duration=195.0, generation_phase="cot"
            )
        self.assertEqual(result, 1475)

    def test_cot_phase_60s(self):
        """60s duration in cot phase -> 300 + 500 = 800."""
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(DURATION_MAX)):
            result = handler._compute_max_new_tokens(
                target_duration=60.0, generation_phase="cot"
            )
        self.assertEqual(result, 800)

    # ------------------------------------------------------------------
    # Capping at max_model_len
    # ------------------------------------------------------------------

    def test_capped_by_max_model_len(self):
        """Result should be capped at max_model_len - 64."""
        handler = self._make_handler(max_model_len=512)
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(DURATION_MAX)):
            result = handler._compute_max_new_tokens(
                target_duration=195.0, generation_phase="codes"
            )
        self.assertEqual(result, 512 - 64)

    # ------------------------------------------------------------------
    # Duration clamping — uses DURATION_MIN and DURATION_MAX constants
    # ------------------------------------------------------------------

    def test_duration_clamp_low(self):
        """Duration below DURATION_MIN is clamped to DURATION_MIN."""
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(DURATION_MAX)):
            result = handler._compute_max_new_tokens(
                target_duration=3.0, generation_phase="codes"
            )
        expected = int(DURATION_MIN * 5) + 10  # 50 + 10 = 60
        self.assertEqual(result, expected)

    def test_duration_clamp_high_unlimited_gpu(self):
        """Duration above DURATION_MAX clamped to DURATION_MAX when GPU allows it."""
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(DURATION_MAX)):
            result = handler._compute_max_new_tokens(
                target_duration=999.0, generation_phase="codes"
            )
        expected = int(DURATION_MAX * 5) + 10  # 3000 + 10 = 3010
        self.assertEqual(result, expected)

    # ------------------------------------------------------------------
    # GPU-config-dependent clamping
    # ------------------------------------------------------------------

    def test_gpu_config_lower_max_duration_codes(self):
        """GPU config with max_duration_with_lm=240 should clamp duration to 240s."""
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(240)):
            result = handler._compute_max_new_tokens(
                target_duration=480.0, generation_phase="codes"
            )
        # effective_duration = min(240, 480) = 240, target_codes = 1200
        expected = int(240 * 5) + 10  # 1200 + 10 = 1210
        self.assertEqual(result, expected)

    def test_gpu_config_lower_max_duration_cot(self):
        """GPU config with max_duration_with_lm=240 affects CoT phase too."""
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(240)):
            result = handler._compute_max_new_tokens(
                target_duration=480.0, generation_phase="cot"
            )
        expected = int(240 * 5) + 500  # 1200 + 500 = 1700
        self.assertEqual(result, expected)

    def test_gpu_config_does_not_exceed_duration_max(self):
        """Even if GPU config allows more than DURATION_MAX, cap at DURATION_MAX."""
        handler = self._make_handler()
        # GPU config says 900s but DURATION_MAX is 600
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(900)):
            result = handler._compute_max_new_tokens(
                target_duration=800.0, generation_phase="codes"
            )
        expected = int(DURATION_MAX * 5) + 10  # 3000 + 10 = 3010
        self.assertEqual(result, expected)

    def test_gpu_config_within_limit(self):
        """Duration within GPU config limit is not clamped."""
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(480)):
            result = handler._compute_max_new_tokens(
                target_duration=300.0, generation_phase="codes"
            )
        expected = int(300 * 5) + 10  # 1500 + 10 = 1510
        self.assertEqual(result, expected)

    def test_gpu_config_unavailable_falls_back(self):
        """If get_global_gpu_config raises, fall back to DURATION_MAX."""
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   side_effect=RuntimeError("no GPU")):
            result = handler._compute_max_new_tokens(
                target_duration=800.0, generation_phase="codes"
            )
        expected = int(DURATION_MAX * 5) + 10  # 3010
        self.assertEqual(result, expected)

    # ------------------------------------------------------------------
    # Fallback when target_duration is None
    # ------------------------------------------------------------------

    def test_fallback_with_explicit_value(self):
        """When target_duration is None, use fallback_max."""
        handler = self._make_handler()
        result = handler._compute_max_new_tokens(
            target_duration=None, generation_phase="codes", fallback_max=2048
        )
        self.assertEqual(result, 2048)

    def test_fallback_default(self):
        """When target_duration is None and no fallback_max, use max_model_len - 64."""
        handler = self._make_handler(max_model_len=4096)
        result = handler._compute_max_new_tokens(
            target_duration=None, generation_phase="codes"
        )
        self.assertEqual(result, 4096 - 64)

    # ------------------------------------------------------------------
    # Regression: the original bug scenario
    # ------------------------------------------------------------------

    def test_regression_progress_bar_not_inflated(self):
        """
        Regression test for the misleading progress bar issue.

        With 195s duration and codes phase, the old code produced 1475 tokens
        (975 + 500) but the constrained decoder forced EOS at 975, making the
        progress bar stop at 66%. The fix should produce 985 (975 + 10).
        """
        handler = self._make_handler()
        with patch("acestep.llm_inference.get_global_gpu_config",
                   return_value=self._mock_gpu_config(DURATION_MAX)):
            result = handler._compute_max_new_tokens(
                target_duration=195.0, generation_phase="codes"
            )
        target_codes = int(195.0 * 5)  # 975
        # max_new_tokens should be close to target_codes, not inflated by +500
        self.assertLessEqual(result - target_codes, 20)
        self.assertGreater(result, target_codes)


if __name__ == "__main__":
    unittest.main()
