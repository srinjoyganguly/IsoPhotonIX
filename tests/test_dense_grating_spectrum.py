"""Regressions for the qualified dense monitor grid and its external reference."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from scripts import calibrate_finite_grating as grating


class DenseGratingSpectrum(unittest.TestCase):
    def test_monitor_band_keeps_the_source_and_five_original_wavelengths(self):
        nominal=grating.settings('lumerical',12)
        dense=grating.settings('lumerical',12,grating_spectrum='dense')
        self.assertEqual({k:v for k,v in dense.items() if k!='grating_spectrum'},nominal)
        wave=grating.monitoring_wavelengths(dense)
        self.assertEqual(len(wave),21)
        np.testing.assert_array_equal(wave[::5],grating.WL)
        self.assertEqual(wave[10],1.55)
        self.assertNotIn('grating_spectrum',nominal)
        for geometry,mesh in [('tidy3d',12),('compact',12),('lumerical',14)]:
            with self.assertRaises(ValueError):
                grating.settings(geometry,mesh,grating_spectrum='dense')

    def test_dense_direct_yee_requires_the_larger_memory_admission(self):
        dense=grating.settings('lumerical',12,'direct-yee','dense')
        with TemporaryDirectory() as folder:
            root=Path(folder); destination=root/'docs/verification';destination.mkdir(parents=True)
            baseline=grating.ROOT/'docs/verification/v012-grating-memory-budget.json'
            (destination/baseline.name).write_bytes(baseline.read_bytes())
            threshold=12345.6
            profile={'required_free_with_reserve_mib':threshold,'wavelength_um':grating.monitoring_wavelengths(dense).tolist(),'resolution':12}
            filename=destination/'v013-grating-dense-memory.json'
            filename.write_text(json.dumps(profile))
            with patch.object(grating,'ROOT',root):
                self.assertEqual(grating.memory_admission(dense,12346)['required_free_with_reserve_mib'],threshold)
                with self.assertRaisesRegex(ValueError,'requires 12346'):
                    grating.memory_admission(dense,12000)
                profile['wavelength_um']=grating.WL.tolist(); filename.write_text(json.dumps(profile))
                with self.assertRaisesRegex(ValueError,'differs'):
                    grating.memory_admission(dense,16000)

    def test_external_dense_reference_is_bound_to_the_published_file(self):
        path=grating.ROOT/'docs/verification'
        baseline=json.loads((path/'v012-grating-commercial-references.json').read_text())['lumerical']
        ref=json.loads((path/'v013-grating-dense-reference.json').read_text())
        self.assertEqual(ref['s_file_sha256'],baseline['s_file_sha256'])
        self.assertEqual(ref['input_samples'],50)
        np.testing.assert_array_equal(ref['published_s21_power'][::5],baseline['samples']['S21']['power'])
        np.testing.assert_array_equal(ref['wavelength_um'],grating.monitoring_wavelengths({'grating_spectrum':'dense'}))
