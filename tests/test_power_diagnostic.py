import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from backend.calibrations import Calibration, design_archive
from scripts.compare_loaded_ring3d import lumerical_port_spectra
from scripts.diagnose_component_power import gain_interval


class PowerDiagnosticTests(unittest.TestCase):
    def test_four_port_mapping_and_power_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'vendor.txt'
            file.write_text(''.join(f'("port {p}", "TE", 1, "port 1", 1, "transmission")\n'
                                    f'(2, 3)\n1e14 {p/10} 0\n2e14 {p/20} 1\n' for p in (4, 1, 3, 2)))
            with patch('scripts.compare_loaded_ring3d.ANSYS_SHA', hashlib.sha256(file.read_bytes()).hexdigest()):
                _, data, _ = lumerical_port_spectra(file)
            for p, key in enumerate(('reflection','drop','through','add_side'), start=1):
                np.testing.assert_allclose(data[key], [(p/20)**2,(p/10)**2])

    def test_one_positive_gain_must_meet_every_wavelength(self):
        inconsistent = gain_interval([.3,.05],[.15,.225],.05)
        self.assertFalse(inconsistent['feasible'])
        self.assertEqual(inconsistent['minimum_set_by_index'],1)
        self.assertTrue(gain_interval([.2,.4],[.4,.8],.01)['feasible'])

    def test_interface_selection_is_validated_and_captured_in_both_programs(self):
        for case, field, chosen, flag in (
            ('ring-3d','ring_interface','exact-area','--ring-interface'),
            ('grating-3d','grating_material_model','direct-yee','--grating-material-model')):
            base = dict(case=case,geometry_3d='lumerical',solver='fdtdx',device='gpu')
            self.assertNotIn(field,Calibration(**base).model_dump())
            config = Calibration(**base, **{field:chosen})
            with zipfile.ZipFile(io.BytesIO(design_archive(config))) as bundle:
                self.assertEqual(json.loads(bundle.read('config.json'))[field],chosen)
                args=json.loads(bundle.read('manifest.json'))['arguments']
                self.assertEqual(args[args.index(flag)+1],chosen)
                self.assertTrue({'run.py','calibration.ipynb'}<=set(bundle.namelist()))
        with self.assertRaises(ValueError):
            Calibration(case='grating-3d',solver='fdtdx',device='gpu',geometry_3d='tidy3d',grating_material_model='direct-yee')
        with self.assertRaises(ValueError):
            Calibration(case='ring',solver='meep',device='cpu',ring_interface='exact-area')
