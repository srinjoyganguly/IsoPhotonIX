import unittest
from pydantic import ValidationError
from backend.model import WaveguideConfig


class PhysicalValidationTests(unittest.TestCase):
    def test_example_is_valid_and_bounded(self):
        model = WaveguideConfig()
        self.assertEqual(model.estimate()["grid"], [200, 100, 1])
        self.assertGreaterEqual(model.estimate()["cells_per_core_wavelength"], 10)

    def test_invalid_physics_and_resources_rejected(self):
        for change in [dict(core_index=1.4, cladding_index=1.5), dict(wavelength_um=float("nan")),
                       dict(wavelength_um=float("inf")), dict(resolution=10, core_index=3),
                       dict(core_width_um=2.5), dict(length_um=20, height_um=12, resolution=60),
                       dict(solver="meep", device="gpu"), dict(device="auto"),
                       dict(unrecognized_parameter=1)]:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                WaveguideConfig(**change)

    def test_target_change_preserves_physical_model(self):
        cpu = WaveguideConfig(solver="fdtdx", device="cpu")
        gpu = WaveguideConfig(solver="fdtdx", device="gpu")
        self.assertEqual(cpu.model_dump(exclude={"device"}), gpu.model_dump(exclude={"device"}))
