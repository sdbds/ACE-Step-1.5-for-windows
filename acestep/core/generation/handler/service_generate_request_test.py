"""Unit tests for service-generation request normalization helpers."""

import types
import unittest
from unittest.mock import patch

from acestep.core.generation.handler.service_generate_request import ServiceGenerateRequestMixin


class _Host(ServiceGenerateRequestMixin):
    """Test host exposing the minimum runtime attributes for request helpers."""

    def __init__(self, is_turbo: bool):
        """Configure turbo flag and no-op normalizers."""
        self.config = types.SimpleNamespace(is_turbo=is_turbo)
        self._normalize_instructions = lambda instructions, _batch, _default: instructions
        self._normalize_audio_code_hints = lambda hints, _batch: hints


class ServiceGenerateRequestMixinTests(unittest.TestCase):
    """Verify request normalization behavior stays stable after extraction."""

    def test_normalize_inputs_clamps_turbo_steps_and_expands_lists(self):
        """Turbo path should clamp infer steps and normalize list-like inputs."""
        host = _Host(is_turbo=True)
        out = host._normalize_service_generate_inputs(
            captions="cap",
            lyrics=["lyric"],
            keys="k1",
            metas={"bpm": 120},
            vocal_languages="en",
            repainting_start=0.1,
            repainting_end=1.2,
            instructions=["i1"],
            audio_code_hints=["h1"],
            infer_steps=20,
            seed=[1],
        )

        self.assertEqual(out["infer_steps"], 8)
        self.assertEqual(out["captions"], ["cap"])
        self.assertEqual(out["lyrics"], ["lyric"])
        self.assertEqual(out["keys"], ["k1"])
        self.assertEqual(out["metas"], [{"bpm": 120}])
        self.assertEqual(out["vocal_languages"], ["en"])
        self.assertEqual(out["repainting_start"], [0.1])
        self.assertEqual(out["repainting_end"], [1.2])
        self.assertEqual(out["seed_list"], [1])

    def test_build_service_seed_list_duplicates_scalar_seed(self):
        """Scalar seed should be copied across the normalized batch size."""
        host = _Host(is_turbo=False)
        out = host._normalize_service_generate_inputs(
            captions=["a", "b", "c"],
            lyrics=["l1"],
            keys=None,
            metas=None,
            vocal_languages=None,
            repainting_start=None,
            repainting_end=None,
            instructions=None,
            audio_code_hints=None,
            infer_steps=12,
            seed=7,
        )

        self.assertEqual(out["infer_steps"], 12)
        self.assertEqual(out["lyrics"], ["l1", "l1", "l1"])
        self.assertEqual(out["seed_list"], [7, 7, 7])

    def test_seed_list_shorter_than_batch_gets_padded(self):
        """Short seed lists should preserve existing entries and append random seeds."""
        host = _Host(is_turbo=False)
        with patch("acestep.core.generation.handler.service_generate_request.random.randint", return_value=99):
            out = host._normalize_service_generate_inputs(
                captions=["a", "b", "c"],
                lyrics=["l1"],
                keys=None,
                metas=None,
                vocal_languages=None,
                repainting_start=None,
                repainting_end=None,
                instructions=None,
                audio_code_hints=None,
                infer_steps=12,
                seed=[3],
            )
        self.assertEqual(out["seed_list"], [3, 99, 99])

    def test_seed_list_longer_than_batch_gets_truncated(self):
        """Long seed lists should be truncated to batch size."""
        host = _Host(is_turbo=False)
        out = host._normalize_service_generate_inputs(
            captions=["a", "b"],
            lyrics=["l1"],
            keys=None,
            metas=None,
            vocal_languages=None,
            repainting_start=None,
            repainting_end=None,
            instructions=None,
            audio_code_hints=None,
            infer_steps=12,
            seed=[11, 12, 13, 14],
        )
        self.assertEqual(out["seed_list"], [11, 12])

    def test_lyrics_longer_than_batch_get_truncated(self):
        """Lyrics list should be truncated to match caption batch size."""
        host = _Host(is_turbo=False)
        out = host._normalize_service_generate_inputs(
            captions=["a", "b"],
            lyrics=["l1", "l2", "l3"],
            keys=None,
            metas=None,
            vocal_languages=None,
            repainting_start=None,
            repainting_end=None,
            instructions=None,
            audio_code_hints=None,
            infer_steps=12,
            seed=[1, 2],
        )
        self.assertEqual(out["lyrics"], ["l1", "l2"])

    def test_batch_size_at_maximum_is_not_clamped(self):
        """Batch size exactly at MAX_BATCH_SIZE (8) should not be clamped."""
        host = _Host(is_turbo=False)
        captions = [f"caption_{i}" for i in range(8)]
        out = host._normalize_service_generate_inputs(
            captions=captions,
            lyrics=["lyric"],
            keys=None,
            metas=None,
            vocal_languages=None,
            repainting_start=None,
            repainting_end=None,
            instructions=None,
            audio_code_hints=None,
            infer_steps=12,
            seed=None,
        )
        self.assertEqual(len(out["captions"]), 8)
        self.assertEqual(out["captions"], captions)
        self.assertEqual(len(out["lyrics"]), 8)

    def test_batch_size_exceeds_maximum_gets_clamped(self):
        """Batch size exceeding MAX_BATCH_SIZE (8) should be clamped to 8."""
        host = _Host(is_turbo=False)
        captions = [f"caption_{i}" for i in range(10)]
        out = host._normalize_service_generate_inputs(
            captions=captions,
            lyrics=["lyric"],
            keys=None,
            metas=None,
            vocal_languages=None,
            repainting_start=None,
            repainting_end=None,
            instructions=None,
            audio_code_hints=None,
            infer_steps=12,
            seed=None,
        )
        # Batch size should be clamped to 8
        self.assertEqual(len(out["captions"]), 8)
        self.assertEqual(out["captions"], captions[:8])
        # Lyrics should be expanded to match clamped batch size
        self.assertEqual(len(out["lyrics"]), 8)
        # Seed list should match clamped batch size
        self.assertIsNone(out["seed_list"])

    def test_batch_size_clamping_with_seed_list(self):
        """Batch size clamping should work correctly with seed lists."""
        host = _Host(is_turbo=False)
        captions = [f"caption_{i}" for i in range(12)]
        seeds = [i for i in range(12)]
        out = host._normalize_service_generate_inputs(
            captions=captions,
            lyrics=["lyric"],
            keys=None,
            metas=None,
            vocal_languages=None,
            repainting_start=None,
            repainting_end=None,
            instructions=None,
            audio_code_hints=None,
            infer_steps=12,
            seed=seeds,
        )
        # Batch size should be clamped to 8
        self.assertEqual(len(out["captions"]), 8)
        # Seed list should be generated for clamped batch size of 8
        self.assertEqual(len(out["seed_list"]), 8)
        self.assertEqual(out["seed_list"], seeds[:8])

    def test_batch_size_below_maximum_is_not_affected(self):
        """Batch sizes below MAX_BATCH_SIZE should pass through unchanged."""
        host = _Host(is_turbo=False)
        for batch_size in [1, 2, 4, 7]:
            captions = [f"caption_{i}" for i in range(batch_size)]
            out = host._normalize_service_generate_inputs(
                captions=captions,
                lyrics=["lyric"],
                keys=None,
                metas=None,
                vocal_languages=None,
                repainting_start=None,
                repainting_end=None,
                instructions=None,
                audio_code_hints=None,
                infer_steps=12,
                seed=None,
            )
            self.assertEqual(len(out["captions"]), batch_size)
            self.assertEqual(out["captions"], captions)

    def test_batch_size_clamping_truncates_all_list_fields(self):
        """Batch size clamping should truncate all optional list fields."""
        host = _Host(is_turbo=False)
        # Create lists with 10 items each
        captions = [f"caption_{i}" for i in range(10)]
        keys = [f"key_{i}" for i in range(10)]
        metas = [{"id": i} for i in range(10)]
        vocal_languages = [f"lang_{i}" for i in range(10)]
        repainting_start = [float(i) for i in range(10)]
        repainting_end = [float(i + 1) for i in range(10)]
        
        out = host._normalize_service_generate_inputs(
            captions=captions,
            lyrics=["lyric"],
            keys=keys,
            metas=metas,
            vocal_languages=vocal_languages,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
            instructions=None,
            audio_code_hints=None,
            infer_steps=12,
            seed=None,
        )
        
        # All fields should be clamped to 8
        self.assertEqual(len(out["captions"]), 8)
        self.assertEqual(out["captions"], captions[:8])
        self.assertEqual(len(out["keys"]), 8)
        self.assertEqual(out["keys"], keys[:8])
        self.assertEqual(len(out["metas"]), 8)
        self.assertEqual(out["metas"], metas[:8])
        self.assertEqual(len(out["vocal_languages"]), 8)
        self.assertEqual(out["vocal_languages"], vocal_languages[:8])
        self.assertEqual(len(out["repainting_start"]), 8)
        self.assertEqual(out["repainting_start"], repainting_start[:8])
        self.assertEqual(len(out["repainting_end"]), 8)
        self.assertEqual(out["repainting_end"], repainting_end[:8])
        self.assertEqual(len(out["lyrics"]), 8)

    def test_batch_size_clamping_with_single_value_fields(self):
        """Batch size clamping should work correctly when single values are expanded."""
        host = _Host(is_turbo=False)
        # 10 captions but single values for other fields
        captions = [f"caption_{i}" for i in range(10)]
        
        out = host._normalize_service_generate_inputs(
            captions=captions,
            lyrics="single_lyric",
            keys="single_key",
            metas={"bpm": 120},
            vocal_languages="en",
            repainting_start=0.5,
            repainting_end=1.5,
            instructions=None,
            audio_code_hints=None,
            infer_steps=12,
            seed=42,
        )
        
        # Batch size should be clamped to 8
        self.assertEqual(len(out["captions"]), 8)
        self.assertEqual(out["captions"], captions[:8])
        # Single values are converted to single-item lists (not expanded to batch size)
        # This matches the original behavior - only lyrics/instructions/audio_code_hints expand
        self.assertEqual(len(out["keys"]), 1)  # Single value becomes list of 1
        self.assertEqual(out["keys"], ["single_key"])
        self.assertEqual(len(out["metas"]), 1)  # Single value becomes list of 1
        self.assertEqual(out["metas"], [{"bpm": 120}])
        self.assertEqual(len(out["vocal_languages"]), 1)  # Single value becomes list of 1
        self.assertEqual(out["vocal_languages"], ["en"])
        self.assertEqual(len(out["repainting_start"]), 1)  # Single value becomes list of 1
        self.assertEqual(out["repainting_start"], [0.5])
        self.assertEqual(len(out["repainting_end"]), 1)  # Single value becomes list of 1
        self.assertEqual(out["repainting_end"], [1.5])
        # Lyrics should be expanded to match clamped batch size
        self.assertEqual(len(out["lyrics"]), 8)
        self.assertEqual(out["lyrics"], ["single_lyric"] * 8)
        # Seed should create list for clamped batch size
        self.assertEqual(len(out["seed_list"]), 8)
        self.assertEqual(out["seed_list"], [42] * 8)


if __name__ == "__main__":
    unittest.main()
