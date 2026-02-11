import unittest

from acestep.core.generation.handler.lora_manager import LoraManagerMixin


class FakeDecoder:
    def __init__(self, modules, adapter_names):
        self._modules = modules
        self._adapter_names = adapter_names

    def named_modules(self):
        return self._modules

    def get_adapter_names(self):
        return self._adapter_names


class FakeModel:
    def __init__(self, decoder):
        self.decoder = decoder


class FakeSetScaleModule:
    def __init__(self):
        self.scaling = {}
        self.lora_alpha = {"main": 2.0}
        self.r = {"main": 2.0}

    def set_scale(self, adapter_name, factor):
        raise AssertionError(f"set_scale should be skipped for {adapter_name} with factor={factor}")


class MinimalHandler(LoraManagerMixin):
    def __init__(self, decoder):
        self.model = FakeModel(decoder)
        self.device = "cpu"
        self.dtype = "float32"
        self.quantization = None
        self._base_decoder = None
        self.lora_loaded = True
        self.use_lora = True
        self.lora_scale = 1.0


class LoraHandlerIntegrationTests(unittest.TestCase):
    def test_handler_state_snapshot_does_not_mutate_service(self):
        decoder = FakeDecoder(modules=[("lora_block", FakeSetScaleModule())], adapter_names=["main"])
        handler = MinimalHandler(decoder)

        _, adapters = handler._rebuild_lora_registry()
        self.assertEqual(adapters, ["main"])
        self.assertIn("main", handler._lora_service.registry)
        handler_target = handler._lora_adapter_registry["main"]["targets"][0]
        self.assertNotIn("module", handler_target)
        self.assertEqual(handler_target["module_class"], "FakeSetScaleModule")

        handler._lora_adapter_registry.clear()
        self.assertIn("main", handler._lora_service.registry)

    def test_set_lora_scale_reports_skipped_targets(self):
        decoder = FakeDecoder(modules=[("lora_block", FakeSetScaleModule())], adapter_names=["main"])
        handler = MinimalHandler(decoder)
        handler._rebuild_lora_registry()

        message = handler.set_lora_scale(0.7)

        self.assertIn("skipped", message)
        self.assertIn("unchanged", message)

    def test_set_lora_scale_reports_no_modules_found(self):
        decoder = FakeDecoder(modules=[], adapter_names=["main"])
        handler = MinimalHandler(decoder)
        handler._rebuild_lora_registry()

        message = handler.set_lora_scale(0.7)

        self.assertIn("no modules found", message)

    def test_get_lora_status_exposes_synthetic_default_mode(self):
        decoder = FakeDecoder(modules=[], adapter_names=[])
        handler = MinimalHandler(decoder)
        handler._rebuild_lora_registry()

        status = handler.get_lora_status()

        self.assertIn("synthetic_default_mode", status)
        self.assertTrue(status["synthetic_default_mode"])

    def test_set_lora_scale_rejects_non_numeric_input(self):
        decoder = FakeDecoder(modules=[], adapter_names=["main"])
        handler = MinimalHandler(decoder)

        message = handler.set_lora_scale("abc")

        self.assertIn("Invalid LoRA scale", message)

    def test_set_use_lora_handles_missing_model_decoder(self):
        decoder = FakeDecoder(modules=[], adapter_names=["main"])
        handler = MinimalHandler(decoder)
        handler.model = None

        message = handler.set_use_lora(False)

        self.assertEqual(message, "✅ LoRA disabled")


if __name__ == "__main__":
    unittest.main()
