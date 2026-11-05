"""The pole extractor must estimate both frequency and damping from the measured signal."""
import unittest

import numpy as np

from scripts.benchmark_passive_cavity import C_MHZ, measured_modes


class CavityMeasurement(unittest.TestCase):
    def test_independent_frequencies_and_decay_rates(self):
        # Regression fixture deliberately differs from the four published values.
        frequencies = np.array([30., 64.5, 102., 152.])
        quality = np.array([4., 11., 8., 17.])
        times = np.arange(380) * .125
        omega = 2 * np.pi * frequencies / C_MHZ
        poles = 1j * omega - omega / (2 * quality)
        samples = (np.exp(times[:, None] * poles) @ np.array([1., .4, .9, .1])).real
        result = measured_modes(times, samples, 8)
        np.testing.assert_allclose([m['frequency_MHz'] for m in result['modes']], frequencies, rtol=1e-9)
        np.testing.assert_allclose([m['quality_factor'] for m in result['modes']], quality, rtol=1e-9)
        self.assertLess(result['relative_field_fit_residual'], 1e-10)

    def test_rejects_unmeasured_or_irregular_field_trace(self):
        times = np.arange(320) * .125
        with self.assertRaisesRegex(ValueError, 'No field'):
            measured_modes(times, np.zeros_like(times), 16)
        times[80] += .001
        with self.assertRaisesRegex(ValueError, 'equally spaced'):
            measured_modes(times, np.ones_like(times), 16)
