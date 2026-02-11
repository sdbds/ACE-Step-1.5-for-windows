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
        import copy

        if self._base_decoder is None:
            self._base_decoder = copy.deepcopy(self.model.decoder)
            logger.info("Base decoder backed up")
        else:
            self.model.decoder = copy.deepcopy(self._base_decoder)
            logger.info("Restored base decoder before loading new LoRA")

        logger.info(f"Loading LoRA adapter from {lora_path}")
        self.model.decoder = PeftModel.from_pretrained(self.model.decoder, lora_path, is_trainable=False)
        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()

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
        import copy

        self.model.decoder = copy.deepcopy(self._base_decoder)
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

        logger.info("LoRA unloaded, base decoder restored")
        return "✅ LoRA unloaded, using base model"
    except Exception as e:
        logger.exception("Failed to unload LoRA")
        return f"❌ Failed to unload LoRA: {str(e)}"
