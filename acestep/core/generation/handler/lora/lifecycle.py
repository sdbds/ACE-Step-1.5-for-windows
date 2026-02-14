"""LoRA/LoKr adapter load/unload lifecycle management."""

import json
import os
from typing import Any

from loguru import logger

from acestep.constants import DEBUG_MODEL_LOADING
from acestep.debug_utils import debug_log
from acestep.training.configs import LoKRConfig

LOKR_WEIGHTS_FILENAME = "lokr_weights.safetensors"


def _resolve_lokr_weights_path(adapter_path: str) -> str | None:
    """Return LoKr safetensors path when ``adapter_path`` points to LoKr artifacts."""
    if os.path.isfile(adapter_path):
        return adapter_path if os.path.basename(adapter_path) == LOKR_WEIGHTS_FILENAME else None
    if os.path.isdir(adapter_path):
        weights_path = os.path.join(adapter_path, LOKR_WEIGHTS_FILENAME)
        return weights_path if os.path.exists(weights_path) else None
    return None


def _load_lokr_config(weights_path: str) -> LoKRConfig:
    """Build ``LoKRConfig`` from safetensors metadata, with defaults on parse failure."""
    config = LoKRConfig()
    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors metadata reader unavailable; using default LoKr config.")
        return config

    try:
        with safe_open(weights_path, framework="pt", device="cpu") as sf:
            metadata: dict[str, Any] = sf.metadata() or {}
    except Exception as exc:
        logger.warning(f"Unable to read LoKr metadata from {weights_path}: {exc}")
        return config

    raw_config = metadata.get("lokr_config")
    if not isinstance(raw_config, str) or not raw_config.strip():
        return config

    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        logger.warning(f"Invalid LoKr metadata config JSON in {weights_path}: {exc}")
        return config

    if not isinstance(parsed, dict):
        return config

    allowed_keys = set(LoKRConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in parsed.items() if k in allowed_keys}
    if not filtered:
        return config

    try:
        return LoKRConfig(**filtered)
    except Exception as exc:
        logger.warning(f"Failed to apply LoKr metadata config from {weights_path}: {exc}")
        return config


def _load_lokr_adapter(decoder: Any, weights_path: str) -> Any:
    """Inject and load a LoKr LyCORIS adapter into ``decoder``."""
    try:
        from lycoris import LycorisNetwork, create_lycoris
    except ImportError as exc:
        raise ImportError("LyCORIS library not installed. Please install with: pip install lycoris-lora") from exc

    lokr_config = _load_lokr_config(weights_path)
    LycorisNetwork.apply_preset(
        {
            "unet_target_name": lokr_config.target_modules,
            "target_name": lokr_config.target_modules,
        }
    )
    lycoris_net = create_lycoris(
        decoder,
        1.0,
        linear_dim=lokr_config.linear_dim,
        linear_alpha=lokr_config.linear_alpha,
        algo="lokr",
        factor=lokr_config.factor,
        decompose_both=lokr_config.decompose_both,
        use_tucker=lokr_config.use_tucker,
        use_scalar=lokr_config.use_scalar,
        full_matrix=lokr_config.full_matrix,
        bypass_mode=lokr_config.bypass_mode,
        rs_lora=lokr_config.rs_lora,
        unbalanced_factorization=lokr_config.unbalanced_factorization,
    )

    if lokr_config.weight_decompose:
        try:
            lycoris_net = create_lycoris(
                decoder,
                1.0,
                linear_dim=lokr_config.linear_dim,
                linear_alpha=lokr_config.linear_alpha,
                algo="lokr",
                factor=lokr_config.factor,
                decompose_both=lokr_config.decompose_both,
                use_tucker=lokr_config.use_tucker,
                use_scalar=lokr_config.use_scalar,
                full_matrix=lokr_config.full_matrix,
                bypass_mode=lokr_config.bypass_mode,
                rs_lora=lokr_config.rs_lora,
                unbalanced_factorization=lokr_config.unbalanced_factorization,
                dora_wd=True,
            )
        except Exception as exc:
            logger.warning(f"DoRA mode not supported in current LyCORIS build: {exc}")

    lycoris_net.apply_to()
    decoder._lycoris_net = lycoris_net
    lycoris_net.load_weights(weights_path)
    return lycoris_net


def load_lora(self, lora_path: str) -> str:
    """Load LoRA adapter into the decoder."""
    if self.model is None:
        return "❌ Model not initialized. Please initialize service first."

    if self.quantization is not None:
        return (
            "❌ LoRA loading is not supported on quantized models. "
            f"Current quantization: {self.quantization}. "
            "Please re-initialize the service with quantization disabled, then try loading the LoRA adapter again."
        )

    if not lora_path or not lora_path.strip():
        return "❌ Please provide a LoRA path."

    lora_path = lora_path.strip()
    if not os.path.exists(lora_path):
        return f"❌ LoRA path not found: {lora_path}"

    lokr_weights_path = _resolve_lokr_weights_path(lora_path)
    if lokr_weights_path is None:
        config_file = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_file):
            return (
                "❌ Invalid adapter: expected PEFT LoRA directory containing adapter_config.json "
                f"or LoKr artifact {LOKR_WEIGHTS_FILENAME} in {lora_path}"
            )

    try:
        from peft import PeftModel
    except ImportError:
        if lokr_weights_path is None:
            return "❌ PEFT library not installed. Please install with: pip install peft"
        PeftModel = None  # type: ignore[assignment]

    try:
        # Memory-efficient state_dict backup instead of deepcopy
        if self._base_decoder is None:
            # Log memory before backup
            if hasattr(self, '_memory_allocated'):
                mem_before = self._memory_allocated() / (1024**3)
                logger.info(f"VRAM before LoRA load: {mem_before:.2f}GB")
            
            # Save only the base model weights as state_dict (CPU to save VRAM)
            try:
                state_dict = self.model.decoder.state_dict()
                if not state_dict:
                    raise ValueError("state_dict is empty - cannot backup decoder")
                self._base_decoder = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
            except Exception as e:
                logger.error(f"Failed to create state_dict backup: {e}")
                raise
            
            # Calculate backup size in MB
            backup_size_mb = sum(v.numel() * v.element_size() for v in self._base_decoder.values()) / (1024**2)
            logger.info(f"Base decoder state_dict backed up to CPU ({backup_size_mb:.1f}MB)")
        else:
            # Restore base decoder from state_dict backup
            logger.info("Restoring base decoder from state_dict backup")
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)

        loaded_kind = "LoRA"
        loaded_source = lora_path
        if lokr_weights_path is not None:
            loaded_kind = "LoKr"
            loaded_source = lokr_weights_path
            logger.info(f"Loading LoKr adapter from {lokr_weights_path}")
            _load_lokr_adapter(self.model.decoder, lokr_weights_path)
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()
        else:
            logger.info(f"Loading LoRA adapter from {lora_path}")
            self.model.decoder = PeftModel.from_pretrained(self.model.decoder, lora_path, is_trainable=False)
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()

        # Log memory after LoRA load
        if hasattr(self, '_memory_allocated'):
            mem_after = self._memory_allocated() / (1024**3)
            logger.info(f"VRAM after LoRA load: {mem_after:.2f}GB")

        self.lora_loaded = True
        self.use_lora = True
        self._ensure_lora_registry()
        self._lora_active_adapter = None
        target_count, adapters = self._rebuild_lora_registry(lora_path=lora_path)

        logger.info(
            f"{loaded_kind} adapter loaded successfully from {loaded_source} "
            f"(adapters={adapters}, targets={target_count})"
        )
        debug_log(
            lambda: f"LoRA registry snapshot: {self._debug_lora_registry_snapshot()}",
            mode=DEBUG_MODEL_LOADING,
            prefix="lora",
        )
        return f"✅ {loaded_kind} loaded from {loaded_source}"
    except Exception as e:
        logger.exception("Failed to load LoRA adapter")
        return f"❌ Failed to load LoRA: {str(e)}"


def unload_lora(self) -> str:
    """Unload LoRA adapter and restore base decoder."""
    if not self.lora_loaded:
        return "⚠️ No LoRA adapter loaded."

    if self._base_decoder is None:
        return "❌ Base decoder backup not found. Cannot restore."

    try:
        # Log memory before unload (track before any operations)
        mem_before = None
        if hasattr(self, '_memory_allocated'):
            mem_before = self._memory_allocated() / (1024**3)
            logger.info(f"VRAM before LoRA unload: {mem_before:.2f}GB")

        # If this decoder has an attached LyCORIS net, restore original module graph first.
        lycoris_net = getattr(self.model.decoder, "_lycoris_net", None)
        if lycoris_net is not None:
            restore_fn = getattr(lycoris_net, "restore", None)
            if callable(restore_fn):
                logger.info("Restoring decoder structure from LyCORIS adapter")
                restore_fn()
            else:
                logger.warning("Decoder has _lycoris_net but no restore() method; continuing with state_dict restore")
            self.model.decoder._lycoris_net = None

        # Get the base model from the PEFT wrapper if it exists.
        # This is more memory-efficient than recreating from state_dict.
        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None  # type: ignore[assignment]

        if PeftModel is not None and isinstance(self.model.decoder, PeftModel):
            logger.info("Extracting base model from PEFT wrapper")
            # PEFT's get_base_model() returns the underlying base model without copying
            self.model.decoder = self.model.decoder.get_base_model()
            # Restore state_dict from backup to ensure clean state
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")
        else:
            # Fallback: restore from state_dict backup
            logger.info("Restoring base decoder from state_dict backup")
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")
        
        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()

        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0
        self._ensure_lora_registry()
        self._lora_service.registry = {}
        self._lora_service.scale_state = {}
        self._lora_service.active_adapter = None
        self._lora_service.last_scale_report = {}
        self._lora_adapter_registry = {}
        self._lora_active_adapter = None
        self._lora_scale_state = {}

        # Log memory after unload
        if mem_before is not None and hasattr(self, '_memory_allocated'):
            mem_after = self._memory_allocated() / (1024**3)
            freed = mem_before - mem_after
            logger.info(f"VRAM after LoRA unload: {mem_after:.2f}GB (freed: {freed:.2f}GB)")

        logger.info("LoRA unloaded, base decoder restored")
        return "✅ LoRA unloaded, using base model"
    except Exception as e:
        logger.exception("Failed to unload LoRA")
        return f"❌ Failed to unload LoRA: {str(e)}"
