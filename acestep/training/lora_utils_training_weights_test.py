"""Unit tests for load_lora_training_weights."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from acestep.training.lora_utils import load_lora_training_weights


def _make_fake_model(state_dict_keys=("w1", "w2")):
    """Build a minimal model stub whose decoder supports load_state_dict."""
    decoder = MagicMock()
    decoder.load_state_dict.return_value = SimpleNamespace(
        missing_keys=[], unexpected_keys=[]
    )
    model = SimpleNamespace(decoder=decoder)
    return model


class TestLoadLoraTrainingWeightsSafetensors(unittest.TestCase):
    """Loading from .safetensors files."""

    def test_loads_safetensors_successfully(self):
        model = _make_fake_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weights.safetensors")
            from safetensors.torch import save_file
            save_file({"w1": torch.zeros(2)}, path)

            info = load_lora_training_weights(model, path)

            model.decoder.load_state_dict.assert_called_once()
            self.assertIn("missing_keys", info)


class TestLoadLoraTrainingWeightsPt(unittest.TestCase):
    """Loading from .pt files."""

    def test_loads_pt_successfully(self):
        model = _make_fake_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weights.pt")
            torch.save({"w1": torch.zeros(2)}, path)

            info = load_lora_training_weights(model, path)

            model.decoder.load_state_dict.assert_called_once()
            self.assertIn("missing_keys", info)


class TestLoadLoraTrainingWeightsMissing(unittest.TestCase):
    """Error handling for missing files."""

    def test_raises_on_missing_file(self):
        model = _make_fake_model()
        with self.assertRaises(FileNotFoundError):
            load_lora_training_weights(model, "/nonexistent/path/weights.safetensors")


class TestLoadLoraTrainingWeightsUnsupported(unittest.TestCase):
    """Error handling for unsupported formats."""

    def test_raises_on_unsupported_format(self):
        model = _make_fake_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weights.bin")
            with open(path, "wb") as f:
                f.write(b"\x00")
            with self.assertRaises(ValueError):
                load_lora_training_weights(model, path)


if __name__ == "__main__":
    unittest.main()
