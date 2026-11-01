"""Known synthetic E/H waves test flux math, not numerical solver accuracy."""
import unittest

import numpy as np

from backend.adapters.fdtdx_ports_adapter import measured_energy_tail, normalized_power, spectral_flux


def plane(ez, hy):
    fields = np.zeros((3, 6, 1, 4, 3), dtype=np.complex128)
    fields[:, 2] = ez
    fields[:, 4] = hy
    return fields


class SpectralPowerTests(unittest.TestCase):
    def test_incident_subtraction_precedes_bilinear_poynting_product(self):
        incident = plane(1, -1)
        reflection = plane(.3+.4j, .3+.4j)
        total = incident+reflection
        transmission = plane(np.sqrt(.6)*1j, -np.sqrt(.6)*1j)
        incident_flux = spectral_flux(incident, .25)
        reflected_flux = spectral_flux(total-incident, .25)
        transmitted_flux = spectral_flux(transmission, .25)
        np.testing.assert_allclose(incident_flux, .5)
        np.testing.assert_allclose(reflected_flux, -.125)
        r, t = normalized_power(incident_flux, reflected_flux, transmitted_flux, np.ones(3, dtype=bool))
        np.testing.assert_allclose(r, .25)
        np.testing.assert_allclose(t, .6)

    def test_both_transverse_polarizations_and_per_invariant_length_integration(self):
        fields = plane(1, -1)
        fields[:, 1] = 2
        fields[:, 5] = 2
        np.testing.assert_allclose(spectral_flux(fields, .25), 2.5)
        np.testing.assert_allclose(spectral_flux(fields[:, :, :, :, :1], .25), 2.5)

    def test_invalid_bins_are_undefined_and_ratios_are_not_clipped(self):
        r, t = normalized_power(np.array([1., 0., 1., -1.]), np.array([.1, 0., -.5, -.5]),
            np.array([1.2, 0., .4, .4]), np.array([True, False, False, True]))
        self.assertAlmostEqual(r[0], -.1)
        self.assertAlmostEqual(t[0], 1.2)
        self.assertTrue(np.isnan(r[1:]).all())
        self.assertTrue(np.isnan(t[1:]).all())

    def test_settling_excludes_unexecuted_zero_detector_slots(self):
        energy = np.array([10., 2., 1., 0., 0., 0.])
        recorded, times, ratio = measured_energy_tail(energy, np.arange(6)*10, 21, 1., 5.)
        np.testing.assert_array_equal(recorded, [10., 2., 1.])
        np.testing.assert_array_equal(times, [0., 10., 20.])
        self.assertEqual(ratio, .1)

    def test_no_measured_energy_cannot_claim_settling(self):
        for energy in ([0., 0.], [1., np.nan]):
            with self.assertRaises(RuntimeError):
                measured_energy_tail(energy, [0, 10], 20, 1.)
        with self.assertRaises(RuntimeError):
            measured_energy_tail([1.], [10], 0, 1.)


if __name__ == '__main__':
    unittest.main()
