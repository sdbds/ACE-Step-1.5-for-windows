import unittest

from acestep.core.lora import LoraService
from acestep.core.lora.scaling import apply_scale_to_adapter


class FakeDecoder:
    def __init__(self, modules, adapter_names=None):
        self._modules = modules
        self._adapter_names = adapter_names

    def named_modules(self):
        return self._modules

    def get_adapter_names(self):
        if self._adapter_names is None:
            raise AttributeError("no adapter names")
        return self._adapter_names


class FakeScaleLayerModule:
    scaling = 1.0

    def __init__(self):
        self.current = 1.0

    def unscale_layer(self):
        self.current = 1.0

    def scale_layer(self, value):
        self.current *= value


class FakeSetScaleFactorModule:
    def __init__(self, scaling):
        self.scaling = scaling
        self.lora_alpha = {"main": 2.0}
        self.r = {"main": 2.0}
        self.calls = []

    def set_scale(self, adapter_name, factor):
        self.calls.append((adapter_name, factor))


class ExplodingSetScaleModule:
    def set_scale(self, adapter_name, factor):
        raise RuntimeError("boom")


class LoraServiceTests(unittest.TestCase):
    def test_rebuild_registry_keeps_adapter_agnostic_targets_without_adapter_names(self):
        decoder = FakeDecoder(modules=[("lora_layer", FakeScaleLayerModule())], adapter_names=[])
        service = LoraService(decoder=decoder)

        total_targets, adapters = service.rebuild_registry()

        self.assertEqual(adapters, ["default"])
        self.assertEqual(total_targets, 1)
        self.assertEqual(service.active_adapter, "default")
        target = service.registry["default"]["targets"][0]
        self.assertEqual(target["kind"], "scale_layer")

    def test_apply_scale_scale_layer_is_idempotent_with_unscale_layer(self):
        module = FakeScaleLayerModule()
        decoder = FakeDecoder(modules=[("lora_layer", module)], adapter_names=[])
        service = LoraService(decoder=decoder)
        service.rebuild_registry()

        modified_first = service.apply_scale("default", 0.5)
        first_value = module.current
        modified_second = service.apply_scale("default", 0.5)

        self.assertEqual(modified_first, 1)
        self.assertEqual(modified_second, 1)
        self.assertEqual(first_value, 0.5)
        self.assertEqual(module.current, 0.5)
        self.assertEqual(len(service.scale_state), 1)

    def test_set_scale_factor_unanchored_is_reported_and_skipped(self):
        module = FakeSetScaleFactorModule(scaling={})
        decoder = FakeDecoder(modules=[("lora_block", module)], adapter_names=["main"])
        service = LoraService(decoder=decoder)
        service.rebuild_registry()

        modified = service.apply_scale("main", 0.5)

        self.assertEqual(modified, 0)
        self.assertEqual(module.calls, [])
        report = service.last_scale_report
        self.assertEqual(report["adapter"], "main")
        self.assertIn("set_scale_factor_unanchored", report["skipped_by_kind"])

    def test_scaling_scalar_non_numeric_is_skipped_with_warning(self):
        class BadScalarModule:
            scaling = "not-a-number"

        warnings = []
        registry = {
            "main": {
                "path": None,
                "targets": [{"module": BadScalarModule(), "kind": "scaling_scalar", "module_name": "bad_scalar"}],
            }
        }
        modified, report = apply_scale_to_adapter(
            registry=registry,
            scale_state={},
            adapter_name="main",
            scale=0.5,
            warn_hook=warnings.append,
        )

        self.assertEqual(modified, 0)
        self.assertIn("scaling_scalar", report["skipped_by_kind"])
        self.assertTrue(any("non-numeric scaling" in msg for msg in warnings))

    def test_target_exception_surfaces_warning(self):
        warnings = []
        debug = []
        registry = {
            "main": {
                "path": None,
                "targets": [
                    {
                        "module": ExplodingSetScaleModule(),
                        "kind": "set_scale_unknown",
                        "module_name": "explode",
                        "base_scale": 1.0,
                    }
                ],
            }
        }
        modified, report = apply_scale_to_adapter(
            registry=registry,
            scale_state={},
            adapter_name="main",
            scale=0.5,
            warn_hook=warnings.append,
            debug_hook=debug.append,
        )

        self.assertEqual(modified, 0)
        self.assertIn("set_scale_unknown", report["skipped_by_kind"])
        self.assertTrue(any("Failed to apply LoRA scale target" in msg for msg in warnings))
        self.assertTrue(any("Scale application exception" in msg for msg in debug))


if __name__ == "__main__":
    unittest.main()
