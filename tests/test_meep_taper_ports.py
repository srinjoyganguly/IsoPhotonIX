"""Real Meep two-port checks for different input and output guide widths."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.adapters.meep_ports3d_adapter import run
from backend.component_library import build_component
from backend.sparameters import validate_sparameters

ROOT = Path(__file__).resolve().parents[1]


def configs():
    base = json.loads((ROOT/'examples/ports3d-straight.json').read_text())
    settings = {k:v for k,v in base.items() if k not in ('schema_version','study_type','geometry','gap_um','device_kind')}
    def taper(left, right, resolution=16):
        design = build_component('taper', {'length_um':2., 'input_width_um':left,
                                           'output_width_um':right,'height_um':.5})['geometry']
        return dict(settings, schema_version=7, study_type='taper-ports-3d', geometry=design, resolution=resolution)
    return {'straight':taper(.75,.75), 'legacy':base, 'unequal':taper(.75,1.),
            'reverse':taper(1.,.75), 'finer':taper(.75,1.,20)}


class MeepTaperPorts(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('meep'), 'Requires the separate native Meep/MPB interpreter')
    def test_straight_limit_mirrored_taper_refinement_and_mode_normalization(self):
        retained = os.environ.get('PHOTONICS_MEEP_TAPER_PORT_CASES')
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(retained or scratch)
            matrices, evidence = {}, {}
            for name, config in configs().items():
                case_name = {'straight':'straight-prism','legacy':'legacy','unequal':'pair-a',
                             'reverse':'reverse-prism','finer':'finer-final'}[name] if retained else name
                case = folder / ('result-' + case_name)
                if retained:
                    self.assertEqual(json.loads((folder/(case_name+'.json')).read_text()),config)
                    metadata = json.loads((case/'metadata.json').read_text())
                else:
                    metadata = run(config,case)
                self.assertTrue(metadata['sparameters_valid'])
                self.assertTrue(all(s['settled'] for s in metadata['settlement'].values()))
                self.assertEqual(metadata['actual_device'],'CPU')
                self.assertEqual(metadata['ports']['port_1_x_um'],-1.75)
                self.assertEqual(metadata['ports']['port_2_x_um'],1.75)
                with np.load(case/'sparameters.npz') as d:
                    validate_sparameters(d, {**metadata, 'config':config})
                    s = matrices[name] = d['s'].copy()
                    self.assertTrue(d['valid_bin'].all() and d['mode_valid'].all())
                    np.testing.assert_allclose(s,d['outgoing_amplitude']/d['incident_amplitude'][:,None,:],rtol=1e-12,atol=1e-12)
                    port_index_errors = []
                    if name != 'legacy':
                        self.assertEqual(d['mode_neff_ports'].shape,(5,2))
                        self.assertTrue(np.all(d['mode_neff_ports'].real > config['cladding_index']))
                        np.testing.assert_allclose(d['mode_neff_ports'][:,0],d['mode_neff'],rtol=1e-8)
                        for j,stage in enumerate(('reference-forward','reference-reverse')):
                            with np.load(case/(stage+'-monitors.npz')) as monitors:
                                sampled = monitors[f'port{j+1}_mode_neff']
                                np.testing.assert_allclose(sampled,d['mode_neff_ports'][::-1,j],rtol=1e-8)
                                port_index_errors.append(float(np.max(abs(sampled-d['mode_neff_ports'][::-1,j]))))
                        if name in ('unequal','finer'):
                            self.assertTrue(np.all(d['mode_neff_ports'][:,1].real > d['mode_neff_ports'][:,0].real + .05))
                    power = np.sum(np.abs(s)**2,axis=1)
                    entry = {'resolution':config['resolution'],'wavelength_um':d['wavelength_um'].tolist(),
                             's_real':s.real.tolist(),'s_imag':s.imag.tolist(),
                             'native_array_sha256':hashlib.sha256((case/'sparameters.npz').read_bytes()).hexdigest(),
                             'reference_device_port_index_max_abs_difference':port_index_errors,
                             'column_power':power.tolist(), 'max_reciprocity_abs':float(np.max(abs(s[:,1,0]-s[:,0,1]))),
                             'max_power_excess':float(np.max(power-1)), 'mode_neff_ports_real':
                                 d['mode_neff_ports'].real.tolist() if name!='legacy' else None,
                             'reference_through_power':(abs(d['reference_through'])**2).tolist(),
                             'source_x_um':metadata['source']['source_x_um'],'elapsed_s':metadata['total_seconds'],
                             'stages':{k:{n:v[n] for n in ('settled','elapsed_fs','energy_ratio','spectral_relative_change')}
                                       for k,v in metadata['settlement'].items()}}
                    self.assertLess(entry['max_reciprocity_abs'],.005)
                    self.assertLess(entry['max_power_excess'],.005)
                    evidence[name] = entry
            straight = matrices['straight']
            self.assertLess(np.max(abs(straight[:,0,0])),1e-5)
            np.testing.assert_allclose(abs(straight[:,1,0])**2,1.,atol=.005)
            straight_error = float(np.max(abs(straight-matrices['legacy'])))
            mirror_error = float(np.max(abs(matrices['reverse']-matrices['unequal'][:,::-1,::-1])))
            mesh_change = float(np.max(abs(matrices['finer']-matrices['unequal'])))
            self.assertLess(straight_error,2e-5)
            self.assertLess(mirror_error,.005)
            report = {'passed':True,'solver':'Meep/MPB CPU','versions':metadata['versions'],
                      'adapter_sha256':hashlib.sha256((ROOT/'backend/adapters/meep_ports3d_adapter.py').read_bytes()).hexdigest(),
                      'scope':'Complex fundamental-mode scattering at fixed physical planes for the recorded taper. Monitors are 0.75 µm into each uniform lead.',
                      'straight_vs_existing_study_max_complex_difference':straight_error,
                      'mirrored_taper_port_permutation_max_complex_difference':mirror_error,
                      'mesh_16_to_20_max_complex_difference':mesh_change,
                      'mesh_16_to_20_max_transmission_power_difference':float(np.max(abs(
                          abs(matrices['finer'][:,1,0])**2-abs(matrices['unequal'][:,1,0])**2))),
                      'mesh_16_to_20_max_transmission_phase_difference_radians':float(np.max(abs(np.angle(
                          matrices['finer'][:,1,0]*np.conj(matrices['unequal'][:,1,0]))))),
                      'criteria':'Equal-width control agrees with the existing native block, mirrored taper swaps ports, finite guided mode indices at both leads, incident-normalized S, measured energy and complex DFT settling. For the tested cases max reciprocity difference and max excess summed fundamental power must be below 0.005.',
                      'cases':evidence,
                      'limits':['Two comparable meshes measure sensitivity and do not establish continuum convergence.','Only one Ey-like guided mode per lead. Residual fundamental power can contain radiation/higher modes or numerical error. No commercial taper reference attached.']}
            save = os.environ.get('PHOTONICS_MEEP_TAPER_PORT_REPORT')
            if save:
                Path(save).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    unittest.main()
