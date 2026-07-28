"""Top-level initialization orchestration for the handler."""

import os
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger

from acestep import gpu_config

_ROCM_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _cuda_supports_bfloat16() -> bool:
    """Return whether the active CUDA device supports native bfloat16 kernels."""
    return gpu_config.cuda_supports_bfloat16()


def _resolve_rocm_dtype() -> torch.dtype:
    """Return a safe model dtype for ROCm/HIP devices.

    Uses ``float32`` by default to avoid segfaults from incomplete
    ``bfloat16`` kernel support on some ROCm GPU configurations (e.g.
    AMD iGPUs on Strix Halo).  Set the ``ACESTEP_ROCM_DTYPE`` environment
    variable to ``float16`` or ``bfloat16`` to override for hardware that
    fully supports those formats.
    """
    raw = os.environ.get("ACESTEP_ROCM_DTYPE", "float32").strip().lower()
    dtype = _ROCM_DTYPE_MAP.get(raw)
    if dtype is None:
        logger.warning(
            f"[initialize_service] Unknown ACESTEP_ROCM_DTYPE={raw!r}; "
            "falling back to float32."
        )
        dtype = torch.float32
    return dtype


class InitServiceOrchestratorMixin:
    """Public ``initialize_service`` orchestration entrypoint."""

    def _reset_lora_state(self) -> None:
        """Reset all LoRA/LoKr state to defaults.

        Must be called when the underlying model is replaced (switch or reinit)
        to prevent stale _base_decoder from causing dimension mismatches.
        """
        self._base_decoder = None
        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0
        self._active_loras = {}
        self._lora_adapter_registry = {}
        self._lora_active_adapter = None
        if hasattr(self, "_adapter_type"):
            self._adapter_type = None
        if hasattr(self, "_lora_scale_state"):
            self._lora_scale_state = {}
        # Reset LoraService registry if present
        lora_svc = getattr(self, "_lora_service", None)
        if lora_svc is not None:
            lora_svc.registry = {}
            lora_svc.scale_state = {}
            lora_svc.active_adapter = None
            lora_svc.last_scale_report = {}
        logger.info("[_reset_lora_state] LoRA/LoKr state cleared")

    def switch_dit_model(
        self,
        config_path: str,
        use_flash_attention: bool = False,
    ) -> Tuple[str, bool]:
        """Hot-swap only the DiT model checkpoint without reloading VAE/text encoder.

        Cleans up LoRA state and resets _base_decoder to prevent stale backups.

        Args:
            config_path: Model directory name under checkpoints/ (e.g. 'acestep-v15-turbo').
            use_flash_attention: Whether to request flash attention for the new model.

        Returns:
            Tuple of (status_message, success_bool).
        """
        try:
            # Clean up LoRA state from previous model
            if self.lora_loaded:
                try:
                    msg = self.unload_lora()
                    logger.info(f"[switch_dit_model] Unloaded LoRA before switch: {msg}")
                except Exception as exc:
                    logger.warning(f"[switch_dit_model] Failed to unload LoRA: {exc}")
            self._reset_lora_state()

            # Resolve checkpoint path
            base_root = (self.last_init_params or {}).get("project_root") or self._get_project_root()
            checkpoint_dir = os.path.join(base_root, "checkpoints")
            model_path = os.path.join(checkpoint_dir, config_path)

            if not os.path.exists(model_path):
                return f"Model checkpoint not found: {model_path}", False

            # Reuse compile/quantization settings from last init
            params = self.last_init_params or {}
            compile_model = params.get("compile_model", False)
            quantization = params.get("quantization", None)

            self._sync_model_code_if_needed(config_path, Path(checkpoint_dir))

            self._load_main_model_from_checkpoint(
                model_checkpoint_path=model_path,
                device=str(self.device),
                use_flash_attention=use_flash_attention,
                compile_model=compile_model,
                quantization=quantization,
            )

            # Update last_init_params to reflect new config
            if self.last_init_params is not None:
                self.last_init_params["config_path"] = config_path
                self.last_init_params["use_flash_attention"] = use_flash_attention

            attn = getattr(self.config, "_attn_implementation", "eager")
            status = f"[OK] Switched to {config_path} on {self.device} (attn={attn})"
            logger.info(f"[switch_dit_model] {status}")
            return status, True
        except Exception as exc:
            error_msg = f"Failed to switch model to {config_path}: {exc}"
            logger.exception(f"[switch_dit_model] {error_msg}")
            return error_msg, False

    def initialize_service(
        self,
        project_root: str,
        config_path: str,
        device: str = "auto",
        use_flash_attention: bool = False,
        compile_model: bool = False,
        offload_to_cpu: bool = False,
        offload_dit_to_cpu: bool = False,
        quantization: Optional[str] = None,
        prefer_source: Optional[str] = None,
        use_mlx_dit: bool = True,
        vae_checkpoint: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Initialize model artifacts and runtime backends for generation.

        This method intentionally supports repeated calls to reinitialize models
        with new settings; it does not short-circuit when components are already loaded.
        """
        try:
            # Clean up stale LoRA state from any previous model before reinit
            self._reset_lora_state()

            if config_path is None:
                config_path = "acestep-v15-turbo"
                logger.warning(
                    "[initialize_service] config_path not set; defaulting to 'acestep-v15-turbo'."
                )

            resolved_device = self._resolve_initialize_device(device)
            self.device = resolved_device
            self.offload_to_cpu = offload_to_cpu
            self.offload_dit_to_cpu = offload_dit_to_cpu

            normalized_compile, normalized_quantization, mlx_compile_requested = self._configure_initialize_runtime(
                device=resolved_device,
                compile_model=compile_model,
                quantization=quantization,
            )
            self.compiled = normalized_compile
            if resolved_device == "cuda" and gpu_config.is_rocm_available():
                self.dtype = _resolve_rocm_dtype()
                logger.info(
                    f"[initialize_service] ROCm/HIP device detected: using dtype={self.dtype} "
                    "(set ACESTEP_ROCM_DTYPE=bfloat16 or float16 to override)"
                )
            elif resolved_device == "cuda":
                if gpu_config.cuda_supports_bfloat16():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16
                    logger.info(
                        "[initialize_service] Pre-Ampere CUDA detected: "
                        "using float16 instead of bfloat16."
                    )
            else:
                self.dtype = torch.bfloat16 if resolved_device == "xpu" else torch.float32
            self.quantization = normalized_quantization
            try:
                self._validate_quantization_setup(
                    quantization=self.quantization,
                    compile_model=normalized_compile,
                )
            except ImportError as exc:
                if self.quantization is not None:
                    logger.warning(
                        "[initialize_service] Quantization disabled: {}",
                        exc,
                    )
                    self.quantization = None
                else:
                    raise

            from acestep.model_downloader import (
                DEFAULT_VAE_VARIANT,
                get_checkpoints_dir,
            )
            env_ckpt = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
            if env_ckpt:
                checkpoint_dir = str(get_checkpoints_dir())
            elif project_root:
                checkpoint_dir = os.path.join(project_root, "checkpoints")
            else:
                checkpoint_dir = str(get_checkpoints_dir())
            checkpoint_path = Path(checkpoint_dir)

            # Resolve VAE selection: explicit param > env var > default.
            resolved_vae_variant = (
                vae_checkpoint
                or os.environ.get("ACESTEP_VAE_CHECKPOINT")
                or DEFAULT_VAE_VARIANT
            )

            precheck_failure = self._ensure_models_present(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                prefer_source=prefer_source,
                vae_variant=resolved_vae_variant,
            )
            if precheck_failure is not None:
                self.model = None
                self.vae = None
                self.text_encoder = None
                self.text_tokenizer = None
                self.config = None
                self.silence_latent = None
                return precheck_failure

            self._sync_model_code_if_needed(config_path, checkpoint_path)

            model_path = os.path.join(checkpoint_dir, config_path)
            self._load_main_model_from_checkpoint(
                model_checkpoint_path=model_path,
                device=resolved_device,
                use_flash_attention=use_flash_attention,
                compile_model=normalized_compile,
                quantization=self.quantization,
            )
            vae_path = self._load_vae_model(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
                compile_model=normalized_compile,
                vae_variant=resolved_vae_variant,
            )
            text_encoder_path = self._load_text_encoder_and_tokenizer(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
            )

            mlx_dit_status, mlx_vae_status = self._initialize_mlx_backends(
                device=resolved_device,
                use_mlx_dit=use_mlx_dit,
                mlx_compile_requested=mlx_compile_requested,
            )

            status_msg = self._build_initialize_status_message(
                device=resolved_device,
                model_path=model_path,
                vae_path=vae_path,
                text_encoder_path=text_encoder_path,
                dtype=self.dtype,
                attention=getattr(self.config, "_attn_implementation", "eager"),
                compile_model=normalized_compile,
                mlx_compile_requested=mlx_compile_requested,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
                quantization=self.quantization,
                mlx_dit_status=mlx_dit_status,
                mlx_vae_status=mlx_vae_status,
            )

            self.last_init_params = {
                "project_root": project_root,
                "config_path": config_path,
                "device": resolved_device,
                "use_flash_attention": use_flash_attention,
                "compile_model": normalized_compile,
                "offload_to_cpu": offload_to_cpu,
                "offload_dit_to_cpu": offload_dit_to_cpu,
                "quantization": self.quantization,
                "use_mlx_dit": use_mlx_dit,
                "prefer_source": prefer_source,
                "vae_checkpoint": resolved_vae_variant,
            }

            return status_msg, True
        except Exception as exc:
            self.model = None
            self.vae = None
            self.text_encoder = None
            self.text_tokenizer = None
            self.config = None
            self.silence_latent = None
            error_msg = f"Error initializing model: {str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.exception(error_msg)
            return error_msg, False
