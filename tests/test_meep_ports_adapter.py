"""Real port-pipeline sanity checks, not a general accuracy certificate."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.adapters.meep_ports_adapter import run


def configuration(kind='gap'):
    return dict(schema_version=3, study_type='waveguide-ports-2d', solver='meep', device='cpu',
                device_kind=kind, gap_um=.25 if kind == 'gap' else 0., core_width_um=.625,
                core_index=2., cladding_index=1., length_um=12., height_um=6., pml_um=1.,
                resolution=32, wavelength_min_um=1.45, wavelength_max_um=1.65,
                frequency_points=21, decay_by=1e-8, spectrum_tolerance=1e-4,
                max_after_sources_fs=1000.)


class MeepPortTests(unittest.TestCase):
    def test_gpu_is_rejected_before_solver_import(self):
        config = configuration()
        config['device'] = 'gpu'
        with self.assertRaisesRegex(ValueError, 'CPU target'):
            run(config, Path('unused'))

    @unittest.skipUnless(importlib.util.find_spec('meep'), 'Meep runtime required for actual numerical test')
    def test_straight_calibration_gap_response_and_unsettled_results(self):
        with tempfile.TemporaryDirectory() as directory:
            measurements = {}
            for kind in ('straight', 'gap'):
                output = Path(directory)/kind
                config = configuration(kind)
                metadata = run(config, output)
                self.assertTrue(metadata['spectra_valid'])
                for stage in metadata['settlement'].values():
                    self.assertTrue(stage['settled'])
                    self.assertLessEqual(stage['energy_ratio'], config['decay_by'])
                    self.assertLessEqual(stage['spectral_relative_change'], config['spectrum_tolerance'])
                with np.load(output/'spectra.npz', allow_pickle=False) as data:
                    self.assertTrue(data['valid_bin'].all())
                    self.assertTrue((np.diff(data['wavelength_um']) > 0).all())
                    np.testing.assert_allclose(data['reflectance'], -data['reflected_flux']/data['incident_flux'])
                    np.testing.assert_allclose(data['transmittance'], data['transmitted_flux']/data['incident_flux'])
                    np.testing.assert_allclose(data['uncollected_fraction'], 1-data['reflectance']-data['transmittance'])
                    measurements[kind] = (data['reflectance'].copy(), data['transmittance'].copy())
                    self.assertTrue(np.iscomplexobj(data['reference_dft_e']))
                with np.load(output/'arrays.npz', allow_pickle=False) as data:
                    self.assertEqual(data['field'].shape, (len(data['y_um']), len(data['x_um'])))
                    self.assertTrue(np.isfinite(data['field']).all())
                    ix, iy = np.argmin(abs(data['x_um'])), np.argmin(abs(data['y_um']))
                    self.assertAlmostEqual(data['epsilon'][iy, ix], 4. if kind == 'straight' else 1., places=5)
            straight_r, straight_t = measurements['straight']
            self.assertLess(np.max(abs(straight_r)), 1e-6)
            self.assertLess(np.max(abs(straight_t-1)), 1e-3)
            gap_r, gap_t = measurements['gap']
            self.assertTrue((gap_r > .1).all())
            self.assertTrue(((gap_t > .4) & (gap_t < .9)).all())
            self.assertGreater(np.max(abs(gap_t-straight_t)), .2)
            # Intentionally below the UI's allowed time cap to exercise the
            # adapter's failure semantics; incomplete values remain exported.
            config = configuration()
            config['max_after_sources_fs'] = 5.
            output = Path(directory)/'timeout'
            metadata = run(config, output)
            self.assertFalse(metadata['spectra_valid'])
            self.assertTrue(any(not stage['settled'] for stage in metadata['settlement'].values()))
            with np.load(output/'spectra.npz', allow_pickle=False) as data:
                self.assertFalse(data['valid_bin'].any())
                self.assertTrue(np.isfinite(data['transmittance']).all())
