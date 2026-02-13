"""LoRA adapter load/unload lifecycle management."""

import os

from loguru import logger

from acestep.constants import DEBUG_MODEL_LOADING
from acestep.debug_utils import debug_log


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

    config_file = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_file):
        return f"❌ Invalid LoRA adapter: adapter_config.json not found in {lora_path}"

    try:
        from peft import PeftModel, PeftConfig  # noqa: F401
    except ImportError:
        return "❌ PEFT library not installed. Please install with: pip install peft"

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
            f"LoRA adapter loaded successfully from {lora_path} "
            f"(adapters={adapters}, targets={target_count})"
        )
        debug_log(
            lambda: f"LoRA registry snapshot: {self._debug_lora_registry_snapshot()}",
            mode=DEBUG_MODEL_LOADING,
            prefix="lora",
        )
        return f"✅ LoRA loaded from {lora_path}"
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
        
        # Get the base model from the PEFT wrapper if it exists
        # This is more memory-efficient than recreating from state_dict
        from peft import PeftModel
        
        if isinstance(self.model.decoder, PeftModel):
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
