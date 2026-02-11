"""Runtime controls for LoRA enablement, scaling, and adapter selection."""

import math
from typing import Any

from loguru import logger

from acestep.constants import DEBUG_MODEL_LOADING
from acestep.debug_utils import debug_log


def set_use_lora(self, use_lora: bool) -> str:
    """Toggle LoRA usage for inference."""
    if use_lora and not self.lora_loaded:
        return "❌ No LoRA adapter loaded. Please load a LoRA first."

    self.use_lora = use_lora
    model = getattr(self, "model", None)
    decoder = getattr(model, "decoder", None) if model is not None else None
    if self.lora_loaded and decoder is None:
        logger.warning("LoRA is marked as loaded, but model/decoder is unavailable during toggle.")

    if self.lora_loaded and decoder is not None and hasattr(decoder, "disable_adapter_layers"):
        try:
            if use_lora:
                if self._lora_active_adapter and hasattr(decoder, "set_adapter"):
                    try:
                        decoder.set_adapter(self._lora_active_adapter)
                    except Exception:
                        pass
                decoder.enable_adapter_layers()
                logger.info("LoRA adapter enabled")
                if self.lora_scale != 1.0:
                    self.set_lora_scale(self.lora_scale)
            else:
                decoder.disable_adapter_layers()
                logger.info("LoRA adapter disabled")
        except Exception as e:
            logger.warning(f"Could not toggle adapter layers: {e}")

    status = "enabled" if use_lora else "disabled"
    return f"✅ LoRA {status}"


def set_lora_scale(self, scale: float) -> str:
    """Set LoRA adapter scale/weight (0-1 range)."""
    if not self.lora_loaded:
        return "⚠️ No LoRA loaded"

    try:
        scale_value = float(scale)
    except (TypeError, ValueError):
        return "❌ Invalid LoRA scale: please provide a numeric value between 0 and 1."
    if not math.isfinite(scale_value):
        return "❌ Invalid LoRA scale: please provide a finite numeric value between 0 and 1."

    self.lora_scale = max(0.0, min(1.0, scale_value))
    if not self.use_lora:
        logger.info(f"LoRA scale set to {self.lora_scale:.2f} (will apply when LoRA is enabled)")
        return f"✅ LoRA scale: {self.lora_scale:.2f} (LoRA disabled)"

    try:
        rebuilt_adapters: list[str] | None = None
        if not getattr(self, "_lora_adapter_registry", None):
            _, rebuilt_adapters = self._rebuild_lora_registry()

        if rebuilt_adapters is not None:
            active_adapter = rebuilt_adapters[0] if rebuilt_adapters else self._lora_service.active_adapter
            if active_adapter is not None:
                self._lora_service.set_active_adapter(active_adapter)
        else:
            active_adapter = self._lora_service.ensure_active_adapter()
        self._lora_active_adapter = active_adapter
        self._sync_lora_state_from_service()
        adapter_names = list(self._lora_service.registry.keys())

        debug_log(
            lambda: (
                f"LoRA scale request: slider={self.lora_scale:.3f} "
                f"active_adapter={active_adapter} adapters={adapter_names}"
            ),
            mode=DEBUG_MODEL_LOADING,
            prefix="lora",
        )

        modified = self._apply_scale_to_adapter(active_adapter, self.lora_scale) if active_adapter else 0
        report = getattr(self, "_lora_last_scale_report", {})
        skipped_total = sum(report.get("skipped_by_kind", {}).values())

        if modified > 0 and active_adapter:
            logger.info(
                f"LoRA scale set to {self.lora_scale:.2f} "
                f"(adapter={active_adapter}, modified={modified}, "
                f"by_kind={report.get('modified_by_kind', {})}, skipped={report.get('skipped_by_kind', {})})"
            )
            return (
                f"✅ LoRA scale: {self.lora_scale:.2f}"
                if skipped_total == 0
                else f"✅ LoRA scale: {self.lora_scale:.2f} (skipped {skipped_total} targets)"
            )

        if skipped_total > 0:
            logger.warning(
                f"No LoRA targets were modified for active adapter "
                f"(adapter={active_adapter}, skipped={report.get('skipped_by_kind', {})})"
            )
            return f"⚠️ LoRA scale unchanged: {self.lora_scale:.2f} (skipped {skipped_total} targets)"

        logger.warning(f"No registered LoRA scaling targets found for active adapter (skipped={report.get('skipped_by_kind', {})})")
        return f"⚠️ Scale set to {self.lora_scale:.2f} (no modules found)"
    except Exception as e:
        logger.warning(f"Could not set LoRA scale: {e}")
        return f"⚠️ Scale set to {self.lora_scale:.2f} (partial)"


def set_active_lora_adapter(self, adapter_name: str) -> str:
    """Set the active LoRA adapter for scaling/inference."""
    self._ensure_lora_registry()
    if not self._lora_service.set_active_adapter(adapter_name):
        return f"❌ Unknown adapter: {adapter_name}"
    self._lora_active_adapter = self._lora_service.active_adapter
    debug_log(f"Selected active LoRA adapter: {adapter_name}", mode=DEBUG_MODEL_LOADING, prefix="lora")
    if self.model is not None and hasattr(self.model, "decoder") and hasattr(self.model.decoder, "set_adapter"):
        try:
            self.model.decoder.set_adapter(adapter_name)
        except Exception:
            pass
    return f"✅ Active LoRA adapter: {adapter_name}"


def get_lora_status(self) -> dict[str, Any]:
    """Get current LoRA status."""
    self._ensure_lora_registry()
    return {
        "loaded": self.lora_loaded,
        "active": self.use_lora,
        "scale": self.lora_scale,
        "active_adapter": self._lora_active_adapter,
        "adapters": list(self._lora_service.registry.keys()),
        "synthetic_default_mode": self._lora_service.synthetic_default_mode,
    }
