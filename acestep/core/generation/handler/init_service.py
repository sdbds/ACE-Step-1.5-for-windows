"""Initialization-adjacent utility mixin for AceStepHandler."""

import os
import time
from contextlib import contextmanager
from typing import List, Optional

import torch
from loguru import logger


class InitServiceMixin:
    def _device_type(self) -> str:
        """Normalize the host device value to a backend type string."""
        if isinstance(self.device, str):
            return self.device.split(":", 1)[0]
        return self.device.type

    def get_available_checkpoints(self) -> List[str]:
        """Return available checkpoint directory paths under the project root.

        Uses ``self._get_project_root()`` to resolve the checkpoints directory and
        returns a single-item list when present, otherwise an empty list.
        """
        # Get project root (handler.py is in acestep/, so go up two levels to project root)
        project_root = self._get_project_root()
        # default checkpoints
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        if os.path.exists(checkpoint_dir):
            return [checkpoint_dir]
        else:
            return []

    def get_available_acestep_v15_models(self) -> List[str]:
        """Scan and return all model directory names starting with 'acestep-v15-'"""
        # Get project root
        project_root = self._get_project_root()
        checkpoint_dir = os.path.join(project_root, "checkpoints")

        models = []
        if os.path.exists(checkpoint_dir):
            # Scan all directories starting with 'acestep-v15-' in checkpoints folder
            for item in os.listdir(checkpoint_dir):
                item_path = os.path.join(checkpoint_dir, item)
                if os.path.isdir(item_path) and item.startswith("acestep-v15-"):
                    models.append(item)

        # Sort by name
        models.sort()
        return models

    def is_flash_attention_available(self, device: Optional[str] = None) -> bool:
        """Check whether flash attention can be used on the target device."""
        target_device = str(device or self.device or "auto").split(":", 1)[0]
        if target_device == "auto":
            if not torch.cuda.is_available():
                return False
        else:
            if target_device != "cuda" or not torch.cuda.is_available():
                return False
        # FlashAttention requires Ampere (compute capability >= 8.0) or newer
        try:
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                logger.info(
                    f"[is_flash_attention_available] GPU compute capability {major}.x < 8.0 "
                    f"(pre-Ampere) — FlashAttention not supported, will use SDPA instead."
                )
                return False
        except Exception:
            return False
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def is_turbo_model(self) -> bool:
        """Check if the currently loaded model is a turbo model"""
        if self.config is None:
            return False
        return getattr(self.config, "is_turbo", False)

    def _empty_cache(self):
        """Clear accelerator memory cache (CUDA, XPU, or MPS)."""
        device_type = self._device_type()
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif device_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _synchronize(self):
        """Synchronize accelerator operations (CUDA, XPU, or MPS)."""
        device_type = self._device_type()
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif device_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()

    def _memory_allocated(self):
        """Get current accelerator memory usage in bytes, or 0 for unsupported backends."""
        device_type = self._device_type()
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        # MPS and XPU don't expose per-tensor memory tracking
        return 0

    def _max_memory_allocated(self):
        """Get peak accelerator memory usage in bytes, or 0 for unsupported backends."""
        device_type = self._device_type()
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
        return 0

    def _is_on_target_device(self, tensor, target_device):
        """Check if tensor is on the target device (handles cuda vs cuda:0 comparison)."""
        if tensor is None:
            return True
        try:
            if isinstance(target_device, torch.device):
                target_type = target_device.type
            else:
                target_type = torch.device(str(target_device)).type
        except Exception:
            # Keep fallback conservative: derive backend token instead of assuming CUDA.
            target_type = str(target_device).strip().lower().split(":", 1)[0]
            if not target_type:
                logger.warning(
                    "[_is_on_target_device] Malformed target device value: {!r}",
                    target_device,
                )
                return False
        return tensor.device.type == target_type

    @staticmethod
    def _get_affine_quantized_tensor_class():
        """Return the AffineQuantizedTensor class from torchao, or None if unavailable.

        Supports both old (torchao.quantization.affine_quantized) and new
        (torchao.dtypes.affine_quantized_tensor) import paths across torchao versions.
        """
        try:
            from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
            return AffineQuantizedTensor
        except ImportError:
            pass
        try:
            from torchao.quantization.affine_quantized import AffineQuantizedTensor
            return AffineQuantizedTensor
        except ImportError:
            pass
        return None

    def _is_quantized_tensor(self, t):
        """True if t is a torchao AffineQuantizedTensor (calling .to() on it can raise NotImplementedError)."""
        if t is None:
            return False
        cls = self._get_affine_quantized_tensor_class()
        if cls is None:
            return False
        return isinstance(t, cls)

    def _has_quantized_params(self, module):
        """True if module (or any submodule) has at least one AffineQuantizedTensor parameter."""
        cls = self._get_affine_quantized_tensor_class()
        if cls is None:
            return False
        for _, param in module.named_parameters():
            if param is not None and isinstance(param, cls):
                return True
        return False

    def _ensure_silence_latent_on_device(self):
        """Ensure silence_latent is on the correct device (self.device)."""
        if hasattr(self, "silence_latent") and self.silence_latent is not None:
            if not self._is_on_target_device(self.silence_latent, self.device):
                self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)

    def _move_module_recursive(self, module, target_device, dtype=None, visited=None):
        """
        Recursively move a module and all its submodules to the target device.
        This handles modules that may not be properly registered.
        """
        if visited is None:
            visited = set()

        module_id = id(module)
        if module_id in visited:
            return
        visited.add(module_id)

        # Move the module itself
        module.to(target_device)
        if dtype is not None:
            module.to(dtype)

        # Move all direct parameters
        for param_name, param in module._parameters.items():
            if param is not None and not self._is_on_target_device(param, target_device):
                if self._is_quantized_tensor(param):
                    moved_param = self._move_quantized_param(param, target_device)
                else:
                    moved_param = torch.nn.Parameter(
                        param.data.to(target_device), requires_grad=param.requires_grad
                    )
                if dtype is not None and moved_param.is_floating_point():
                    moved_param = torch.nn.Parameter(
                        moved_param.data.to(dtype), requires_grad=param.requires_grad
                    )
                module._parameters[param_name] = moved_param

        # Move all direct buffers
        for buf_name, buf in module._buffers.items():
            if buf is not None and not self._is_on_target_device(buf, target_device):
                module._buffers[buf_name] = buf.to(target_device)

        # Recursively process all submodules (registered and unregistered)
        for name, child in module._modules.items():
            if child is not None:
                self._move_module_recursive(child, target_device, dtype, visited)

        # Also check for any nn.Module attributes that might not be in _modules
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(module, attr_name, None)
                if isinstance(attr, torch.nn.Module) and id(attr) not in visited:
                    self._move_module_recursive(attr, target_device, dtype, visited)
            except Exception:
                pass

    def _move_quantized_param(self, param, target_device):
        """Move an AffineQuantizedTensor to target_device using _apply_fn_to_data.

        This is the safe fallback for older torch versions where model.to(device) raises
        NotImplementedError on AffineQuantizedTensor (because aten._has_compatible_shallow_copy_type
        is not implemented). _apply_fn_to_data recursively applies a function to all inner
        tensors (int_data, scale, zero_point, etc.) without going through Module._apply.
        """
        if hasattr(param, '_apply_fn_to_data'):
            return torch.nn.Parameter(
                param._apply_fn_to_data(lambda x: x.to(target_device)),
                requires_grad=param.requires_grad,
            )
        # Last resort: try direct .to() (may raise), but preserve Parameter registration.
        moved = param.to(target_device)
        return torch.nn.Parameter(moved, requires_grad=param.requires_grad)

    def _recursive_to_device(self, model, device, dtype=None):
        """
        Recursively move all parameters and buffers of a model to the specified device.
        This is more thorough than model.to() for some custom HuggingFace models.

        Handles torchao AffineQuantizedTensor parameters that may raise NotImplementedError
        on model.to(device) in older torch versions (where Module._apply calls
        _has_compatible_shallow_copy_type, which is not implemented for AffineQuantizedTensor).
        In that case, falls back to moving quantized parameters individually via _apply_fn_to_data.
        """
        target_device = torch.device(device) if isinstance(device, str) else device

        # Method 1: Standard .to() call — works on newer torch where _apply uses swap_tensors
        try:
            model.to(target_device)
            if dtype is not None:
                model.to(dtype)
        except NotImplementedError:
            # Older torch: Module._apply calls _has_compatible_shallow_copy_type which is
            # not implemented for AffineQuantizedTensor. Move parameters manually.
            logger.info(
                "[_recursive_to_device] model.to() raised NotImplementedError "
                "(AffineQuantizedTensor on older torch). Moving parameters individually."
            )
            for module in model.modules():
                # Move non-quantized parameters and buffers directly
                for param_name, param in module._parameters.items():
                    if param is None:
                        continue
                    if self._is_on_target_device(param, target_device):
                        continue
                    if self._is_quantized_tensor(param):
                        module._parameters[param_name] = self._move_quantized_param(param, target_device)
                    else:
                        module._parameters[param_name] = torch.nn.Parameter(
                            param.data.to(target_device), requires_grad=param.requires_grad
                        )
                        if dtype is not None:
                            module._parameters[param_name] = torch.nn.Parameter(
                                module._parameters[param_name].data.to(dtype),
                                requires_grad=param.requires_grad,
                            )
                for buf_name, buf in module._buffers.items():
                    if buf is not None and not self._is_on_target_device(buf, target_device):
                        module._buffers[buf_name] = buf.to(target_device)

        # Method 2: Use our thorough recursive moving for any missed modules
        # (skip if model.to() failed — we already moved everything above)
        try:
            self._move_module_recursive(model, target_device, dtype)
        except NotImplementedError:
            pass  # Already handled above

        # Method 3: Force move via state_dict if there are still parameters on wrong device
        wrong_device_params = []
        for name, param in model.named_parameters():
            if not self._is_on_target_device(param, device):
                wrong_device_params.append(name)

        if wrong_device_params and device != "cpu":
            logger.warning(f"[_recursive_to_device] {len(wrong_device_params)} parameters on wrong device after initial move, retrying individually")
            for module in model.modules():
                for param_name, param in module._parameters.items():
                    if param is None or self._is_on_target_device(param, target_device):
                        continue
                    if self._is_quantized_tensor(param):
                        module._parameters[param_name] = self._move_quantized_param(param, target_device)
                    else:
                        module._parameters[param_name] = torch.nn.Parameter(
                            param.data.to(target_device), requires_grad=param.requires_grad
                        )
                        if dtype is not None and module._parameters[param_name].is_floating_point():
                            module._parameters[param_name] = torch.nn.Parameter(
                                module._parameters[param_name].data.to(dtype),
                                requires_grad=param.requires_grad,
                            )

        # Synchronize accelerator to ensure all transfers are complete
        if device != "cpu":
            self._synchronize()

        # Final verification
        if device != "cpu":
            still_wrong = []
            for name, param in model.named_parameters():
                if not self._is_on_target_device(param, device):
                    still_wrong.append(f"{name} on {param.device}")
            if still_wrong:
                logger.error(f"[_recursive_to_device] CRITICAL: {len(still_wrong)} parameters still on wrong device: {still_wrong[:10]}")

    @contextmanager
    def _load_model_context(self, model_name: str):
        """
        Context manager to load a model to GPU and offload it back to CPU after use.

        Args:
            model_name: Name of the model to load ("text_encoder", "vae", "model")
        """
        if not self.offload_to_cpu:
            yield
            return

        # If model is DiT ("model") and offload_dit_to_cpu is False, do not offload
        if model_name == "model" and not self.offload_dit_to_cpu:
            # Ensure it's on device if not already (should be handled by init, but safe to check)
            model = getattr(self, model_name, None)
            if model is not None:
                # Check if model is on CPU, if so move to device (one-time move if it was somehow on CPU)
                # We check the first parameter's device
                try:
                    param = next(model.parameters())
                    if param.device.type == "cpu":
                        logger.info(f"[_load_model_context] Moving {model_name} to {self.device} (persistent)")
                        self._recursive_to_device(model, self.device, self.dtype)
                        if hasattr(self, "silence_latent"):
                            self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)
                except StopIteration:
                    pass
            yield
            return

        model = getattr(self, model_name, None)
        if model is None:
            yield
            return

        # Load to GPU
        logger.info(f"[_load_model_context] Loading {model_name} to {self.device}")
        start_time = time.time()
        if model_name == "vae":
            vae_dtype = self._get_vae_dtype()
            self._recursive_to_device(model, self.device, vae_dtype)
        else:
            self._recursive_to_device(model, self.device, self.dtype)

        if model_name == "model" and hasattr(self, "silence_latent"):
            self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)

        load_time = time.time() - start_time
        self.current_offload_cost += load_time
        logger.info(f"[_load_model_context] Loaded {model_name} to {self.device} in {load_time:.4f}s")

        try:
            yield
        finally:
            # Offload to CPU
            logger.info(f"[_load_model_context] Offloading {model_name} to CPU")
            start_time = time.time()
            if model_name == "vae":
                self._recursive_to_device(model, "cpu", self._get_vae_dtype("cpu"))
            else:
                self._recursive_to_device(model, "cpu")

            # NOTE: Do NOT offload silence_latent to CPU here!
            # silence_latent is used in many places outside of model context,
            # so it should stay on GPU to avoid device mismatch errors.

            self._empty_cache()
            offload_time = time.time() - start_time
            self.current_offload_cost += offload_time
            logger.info(f"[_load_model_context] Offloaded {model_name} to CPU in {offload_time:.4f}s")
