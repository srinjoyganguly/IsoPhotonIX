"""Synthetic math fixtures only; these do not validate Maxwell simulations."""
import numpy as np
import unittest

from scripts.benchmark_slab_propagation import C_UM_PER_FS, harmonic_phase_fit, transverse_projection
from backend.mode_benchmarks import analytic_te0


class SlabPropagationFitTests(unittest.TestCase):
    def test_transverse_projection_does_not_impose_reference_longitudinal_index(self):
        x, y = np.linspace(-2, 3, 151), np.linspace(-4.5, 4.5, 61)
        t = np.arange(25)*7.5+np.linspace(0, .1, 25)
        k0, reference = 2*np.pi/1.55, analytic_te0(1.5, 1., 1.)
        h, alpha = k0*np.sqrt(1.5**2-reference**2), k0*np.sqrt(reference**2-1.)
        phi = np.where(abs(y) <= .5, np.cos(h*y), np.cos(h*.5)*np.exp(-alpha*(abs(y)-.5)))
        # Deliberately use a propagation index different from the template's.
        # An odd transverse contaminant has zero discrete overlap with phi.
        omega, neff = k0*C_UM_PER_FS, 1.37
        field = phi[None, :, None]*np.cos(omega*t[:, None, None]-k0*neff*x[None, None, :])
        field += 20*(y*phi)[None, :, None]*np.cos(omega*t[:, None, None]-k0*1.1*x[None, None, :])
        measured, derived = transverse_projection(t, field, x, y)
        self.assertAlmostEqual(measured['effective_index'], neff, places=12)
        self.assertNotAlmostEqual(measured['effective_index'], reference, places=2)
        self.assertEqual(derived['full_phasor'].shape, (len(y), len(x)))
        self.assertLess(measured['projection_vs_temporal_fit_max_abs_difference'], 1e-13)

    def test_known_traveling_wave_with_spatial_amplitude_and_dc(self):
        x = np.linspace(-2, 3, 151)
        # Deliberately sparse, slightly irregular saved times; frequency is known.
        t = np.arange(25)*7.5+np.linspace(0, .1, 25)
        neff, wavelength = 1.4029, 1.55
        omega, beta = 2*np.pi*C_UM_PER_FS/wavelength, 2*np.pi*neff/wavelength
        values = (1+.1*x)[None, :]*np.cos(omega*t[:, None]-beta*x[None, :]+.7)+.4+.02*x
        metric, derived = harmonic_phase_fit(t, values, x)
        self.assertAlmostEqual(metric["effective_index"], neff, places=12)
        self.assertAlmostEqual(metric["phase_slope_rad_per_um"], -beta, places=12)
        assert metric["temporal_relative_l2_residual"] < 1e-13
        assert metric["phase_rmse_rad"] < 1e-13
        assert np.array_equal(derived["samples"], values[np.ix_(t >= 100, (x >= -1)&(x <= 2))])


    def test_zero_or_dc_only_signal_has_no_measurable_propagation_phase(self):
        for value in (0., 1.):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "No nonzero oscillating"):
                    harmonic_phase_fit(np.linspace(0, 180, 25), np.full((25, 50), value), np.linspace(-1, 2, 50))


    def test_bad_time_coverage_and_phase_degenerate_sampling_are_rejected(self):
        x = np.linspace(-1, 2, 30)
        with self.assertRaisesRegex(ValueError, "six late-time"):
            harmonic_phase_fit(np.linspace(0, 20, 25), np.ones((25, 30)), x)
        with self.assertRaisesRegex(ValueError, "two optical periods"):
            harmonic_phase_fit(np.linspace(100, 101, 25), np.ones((25, 30)), x)
        t = np.arange(21)*1.55/C_UM_PER_FS
        with self.assertRaisesRegex(ValueError, "cannot identify"):
            harmonic_phase_fit(t, np.ones((21, 30)), x, start_fs=0)


    def test_nonfinite_and_malformed_saved_coordinates_rejected(self):
        t = np.linspace(0, 180, 25)
        x = np.linspace(-1, 2, 30)
        data = np.ones((25, 30)); data[4, 5] = np.nan
        with self.assertRaisesRegex(ValueError, "Finite"):
            harmonic_phase_fit(t, data, x)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            harmonic_phase_fit(t[::-1], np.ones((25, 30)), x)
