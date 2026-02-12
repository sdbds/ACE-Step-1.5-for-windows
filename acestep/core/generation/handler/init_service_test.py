import os
import builtins
import types
import unittest
from unittest.mock import Mock, patch

import torch

from acestep.core.generation.handler.init_service import InitServiceMixin


class _Host(InitServiceMixin):
    def __init__(self, project_root: str, device: str = "cpu", config=None):
        self._project_root = project_root
        self.device = device
        self.config = config

    def _get_project_root(self):
        return self._project_root


class InitServiceMixinTests(unittest.TestCase):
    def test_device_type_normalizes_device(self):
        host = _Host(project_root="K:/fake_root", device="cuda:0")
        self.assertEqual(host._device_type(), "cuda")

    def test_is_on_target_device_handles_device_alias(self):
        host = _Host(project_root="K:/fake_root", device="cpu")
        t = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
        self.assertTrue(host._is_on_target_device(t, "cuda:0"))
        self.assertFalse(host._is_on_target_device(t, "cpu"))

    def test_is_on_target_device_fallback_does_not_assume_cuda(self):
        host = _Host(project_root="K:/fake_root", device="cpu")
        t = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
        self.assertFalse(host._is_on_target_device(t, "mps:0"))

    def test_is_on_target_device_malformed_target_logs_and_returns_false(self):
        host = _Host(project_root="K:/fake_root", device="cpu")
        t = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
        with patch("acestep.core.generation.handler.init_service.logger.warning") as warning:
            self.assertFalse(host._is_on_target_device(t, ":0"))
        warning.assert_called_once()

    def test_move_module_recursive_preserves_parameter_type(self):
        host = _Host(project_root="K:/fake_root", device="cpu")
        module = torch.nn.Linear(2, 2)
        with patch.object(host, "_is_on_target_device", return_value=False):
            host._move_module_recursive(module, "cpu")
        self.assertIsInstance(module.weight, torch.nn.Parameter)
        self.assertIsInstance(module.bias, torch.nn.Parameter)

    def test_move_quantized_param_fallback_wraps_parameter(self):
        host = _Host(project_root="K:/fake_root", device="cpu")
        param = torch.nn.Parameter(torch.randn(2), requires_grad=True)
        moved = host._move_quantized_param(param, "cpu")
        self.assertIsInstance(moved, torch.nn.Parameter)
        self.assertTrue(moved.requires_grad)

    def test_get_available_checkpoints_returns_expected_list(self):
        host = _Host(project_root="K:/fake_root")
        with patch("os.path.exists", return_value=False):
            self.assertEqual(host.get_available_checkpoints(), [])

        with patch("os.path.exists", return_value=True):
            self.assertEqual(host.get_available_checkpoints(), [os.path.join("K:/fake_root", "checkpoints")])

    def test_get_available_acestep_v15_models_filters_and_sorts(self):
        host = _Host(project_root="K:/fake_root")
        with patch("os.path.exists", return_value=True), patch(
            "os.listdir",
            return_value=["acestep-v15-zeta", "acestep-v15-alpha", "not-a-model", "acestep-v15-file"],
        ), patch(
            "os.path.isdir",
            side_effect=lambda p: p.endswith("acestep-v15-zeta")
            or p.endswith("acestep-v15-alpha")
            or p.endswith("not-a-model"),
        ):
            self.assertEqual(
                host.get_available_acestep_v15_models(),
                ["acestep-v15-alpha", "acestep-v15-zeta"],
            )

    def test_is_turbo_model_uses_config_flag(self):
        host = _Host(project_root="K:/fake_root", config=None)
        self.assertFalse(host.is_turbo_model())

        host.config = types.SimpleNamespace(is_turbo=True)
        self.assertTrue(host.is_turbo_model())

    def test_is_flash_attention_available_rejects_non_cuda(self):
        host = _Host(project_root="K:/fake_root", device="cpu")
        self.assertFalse(host.is_flash_attention_available())
        self.assertFalse(host.is_flash_attention_available(device="mps"))

    def test_is_flash_attention_available_true_when_cuda_and_module_present(self):
        host = _Host(project_root="K:/fake_root", device="cuda")
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
                with patch.dict("sys.modules", {"flash_attn": types.SimpleNamespace()}):
                    self.assertTrue(host.is_flash_attention_available())

    def test_is_flash_attention_available_false_when_pre_ampere_gpu(self):
        host = _Host(project_root="K:/fake_root", device="cuda")
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
                with patch.dict("sys.modules", {"flash_attn": types.SimpleNamespace()}):
                    self.assertFalse(host.is_flash_attention_available())

    def test_is_flash_attention_available_false_when_module_missing(self):
        host = _Host(project_root="K:/fake_root", device="cuda")
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "flash_attn":
                raise ImportError("flash_attn missing")
            return real_import(name, globals, locals, fromlist, level)

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
                with patch("builtins.__import__", side_effect=fake_import):
                    self.assertFalse(host.is_flash_attention_available())

    def test_empty_cache_routes_to_cuda(self):
        host = _Host(project_root="K:/fake_root", device="cuda")
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as empty_cache:
            host._empty_cache()
            empty_cache.assert_called_once()

    def test_empty_cache_routes_to_xpu(self):
        host = _Host(project_root="K:/fake_root", device="xpu")
        empty_cache = Mock()
        xpu_stub = types.SimpleNamespace(is_available=lambda: True, empty_cache=empty_cache)
        with patch("torch.xpu", new=xpu_stub, create=True):
            host._empty_cache()
        empty_cache.assert_called_once()

    def test_empty_cache_routes_to_mps(self):
        host = _Host(project_root="K:/fake_root", device="mps")
        with patch("torch.backends.mps.is_available", return_value=True), patch("torch.mps.empty_cache") as empty_cache:
            host._empty_cache()
            empty_cache.assert_called_once()

    def test_synchronize_routes_to_cuda(self):
        host = _Host(project_root="K:/fake_root", device="cuda")
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.synchronize") as sync:
            host._synchronize()
            sync.assert_called_once()

    def test_synchronize_routes_to_xpu(self):
        host = _Host(project_root="K:/fake_root", device="xpu")
        sync = Mock()
        xpu_stub = types.SimpleNamespace(is_available=lambda: True, synchronize=sync)
        with patch("torch.xpu", new=xpu_stub, create=True):
            host._synchronize()
        sync.assert_called_once()

    def test_synchronize_routes_to_mps(self):
        host = _Host(project_root="K:/fake_root", device="mps")
        with patch("torch.backends.mps.is_available", return_value=True), patch("torch.mps.synchronize") as sync:
            host._synchronize()
            sync.assert_called_once()

    def test_memory_queries_use_cuda_only(self):
        host = _Host(project_root="K:/fake_root", device="cpu")
        self.assertEqual(host._memory_allocated(), 0)
        self.assertEqual(host._max_memory_allocated(), 0)

        host.device = "cuda"
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=123), patch(
                "torch.cuda.max_memory_allocated", return_value=456
            ):
                self.assertEqual(host._memory_allocated(), 123)
                self.assertEqual(host._max_memory_allocated(), 456)


if __name__ == "__main__":
    unittest.main()
