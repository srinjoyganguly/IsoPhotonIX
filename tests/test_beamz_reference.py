"""Check signed-field measurement independently of the optional solver environment."""
import unittest

import numpy as np

from scripts.benchmark_beamz_reference import phase_and_impedance


class BeamZMeasurement(unittest.TestCase):
    def test_staggered_forward_wave_recovers_angular_frequency_and_impedance(self):
        x_e = np.arange(60) / 20
        x_h = x_e + .025
        k = 2 * np.pi
        angular_frequency, impedance, duration = 3.75, 2.4, .25
        e0 = np.cos(k*x_e)
        e1 = np.cos(k*x_e-angular_frequency*duration)
        h1 = -np.cos(k*x_h-angular_frequency*(duration-.012))/impedance
        rate, z, amplitude = phase_and_impedance(e0,e1,h1,x_e,x_h,k,duration)
        np.testing.assert_allclose([rate,z,amplitude],[angular_frequency,impedance,1.],rtol=1e-12)
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            phase_and_impedance(0*e0,e1,h1,x_e,x_h,k,duration)
        with self.assertRaisesRegex(ValueError,'forward'):
            phase_and_impedance(e1,e0,h1,x_e,x_h,k,duration)
