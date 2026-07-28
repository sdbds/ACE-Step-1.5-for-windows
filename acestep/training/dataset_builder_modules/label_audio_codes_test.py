"""Tests for bulk audio-code encoding strategies."""

import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType

import torch

_PACKAGE_NAME = "_label_audio_codes_test_modules"
_PACKAGE_PATH = Path(__file__).resolve().parent
_PACKAGE = ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(_PACKAGE_PATH)]
sys.modules[_PACKAGE_NAME] = _PACKAGE

encode_audio_codes = importlib.import_module(
    f"{_PACKAGE_NAME}.label_audio_codes"
).encode_audio_codes
AudioSample = importlib.import_module(f"{_PACKAGE_NAME}.models").AudioSample


class _RecordedContext:
    """Record entry and exit for a model-loading context."""

    def __init__(self, name: str, events: list[tuple]) -> None:
        """Store the context name and shared event log."""
        self.name = name
        self.events = events

    def __enter__(self):
        """Record entry and return the context."""
        self.events.append(("enter", self.name))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Record exit without suppressing exceptions."""
        del exc_type, exc_value, traceback
        self.events.append(("exit", self.name))


class _TokenizeModel:
    """Return deterministic token indices for encoded latent batches."""

    def __init__(self, events: list[tuple]) -> None:
        """Store the shared event log."""
        self.events = events
        self.next_index = 1

    def tokenize(
        self,
        hidden_states: torch.Tensor,
        silence_latent: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[None, torch.Tensor, torch.Tensor]:
        """Record the batch and return indices masked to original lengths."""
        del silence_latent
        self.events.append(("tokenize", tuple(hidden_states.shape)))
        count = hidden_states.shape[0] * hidden_states.shape[1]
        indices = torch.arange(
            self.next_index,
            self.next_index + count,
            device=hidden_states.device,
        ).reshape(hidden_states.shape[:2])
        self.next_index += count
        return None, indices, attention_mask


class _ContextHandler:
    """Provide the model-context API used by local low-memory labeling."""

    def __init__(
        self,
        events: list[tuple],
        offload_to_cpu: bool = False,
        silence_device: str = "cpu",
    ) -> None:
        """Initialize deterministic CPU tensors and model state."""
        self.events = events
        self.model = _TokenizeModel(events)
        self.silence_latent = torch.zeros((1, 3, 2), device=silence_device)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.offload_to_cpu = offload_to_cpu

    def convert_src_audio_to_codes(self, audio_path: str) -> str:
        """Reject the direct path when model contexts are available."""
        raise AssertionError(f"direct conversion used for {audio_path}")

    def _load_model_context(self, name: str) -> _RecordedContext:
        """Return a context that records model loading boundaries."""
        return _RecordedContext(name, self.events)

    def process_src_audio(self, audio_path: str) -> torch.Tensor:
        """Return different latent lengths for the two test samples."""
        self.events.append(("process", audio_path))
        length = {
            "first.wav": 2, "second.wav": 3, "third.wav": 1, "fourth.wav": 2
        }[audio_path]
        return torch.ones((length, 2))

    def is_silence(self, audio: torch.Tensor) -> bool:
        """Treat all test audio as non-silent."""
        del audio
        return False

    def _encode_audio_to_latents(self, audio: torch.Tensor) -> torch.Tensor:
        """Use the deterministic processed tensor as its latent encoding."""
        self.events.append(("vae_encode", audio.shape[0]))
        return audio


class TestContextAudioCodeEncoding(unittest.TestCase):
    """Verify the low-memory context path preserves batching semantics."""

    def test_context_path_honors_chunk_and_microbatch_sizes(self) -> None:
        """Catch ignored chunk sizes, batch sizes, or direct conversion."""
        events: list[tuple] = []
        samples = [
            (0, AudioSample(audio_path="first.wav", filename="first.wav")),
            (1, AudioSample(audio_path="second.wav", filename="second.wav")),
            (2, AudioSample(audio_path="third.wav", filename="third.wav")),
            (3, AudioSample(audio_path="fourth.wav", filename="fourth.wav")),
        ]

        try:
            codes = encode_audio_codes(
                samples_to_label=samples,
                dit_handler=_ContextHandler(events),
                progress_callback=None,
                total=4,
                chunk_size=3,
                batch_size=2,
            )
        except ModuleNotFoundError as exc:
            self.fail(f"context encoder is missing: {exc}")

        self.assertEqual(
            {
                0: "<|audio_code_1|><|audio_code_2|>",
                1: "<|audio_code_4|><|audio_code_5|><|audio_code_6|>",
                2: "<|audio_code_7|>",
                3: "<|audio_code_8|><|audio_code_9|>",
            },
            codes,
        )
        self.assertEqual(
            [
                ("enter", "vae"),
                ("process", "first.wav"),
                ("vae_encode", 2),
                ("process", "second.wav"),
                ("vae_encode", 3),
                ("process", "third.wav"),
                ("vae_encode", 1),
                ("exit", "vae"),
                ("enter", "model"),
                ("tokenize", (2, 3, 2)),
                ("tokenize", (1, 1, 2)),
                ("exit", "model"),
                ("enter", "vae"),
                ("process", "fourth.wav"),
                ("vae_encode", 2),
                ("exit", "vae"),
                ("enter", "model"),
                ("tokenize", (1, 2, 2)),
                ("exit", "model"),
            ],
            events,
        )

    def test_cpu_offload_skips_transient_silence_device_validation(self) -> None:
        """Catch strict device validation rejecting CPU-offloaded model state."""
        events: list[tuple] = []
        sample = AudioSample(audio_path="first.wav", filename="first.wav")
        handler = _ContextHandler(
            events,
            offload_to_cpu=True,
            silence_device="meta",
        )

        codes = encode_audio_codes(
            samples_to_label=[(0, sample)],
            dit_handler=handler,
            progress_callback=None,
            total=1,
            chunk_size=1,
            batch_size=1,
        )

        self.assertEqual(
            {0: "<|audio_code_1|><|audio_code_2|>"},
            codes,
        )


if __name__ == "__main__":
    unittest.main()
