import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'calibrate_{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResonatorAndGrating(unittest.TestCase):
    def test_peak_width_resolution_gate(self):
        # Synthetic Lorentzian tests only the extraction, never the numerical evidence.
        try:
            ring=load('ring')
        except ModuleNotFoundError as exc:
            if exc.name != 'scipy':
                raise
            self.skipTest('ring analysis imports SciPy from the Meep numerical environment')
        wl=np.linspace(1.50,1.60,1001)
        for width,resolved in [(0.004,True),(.0001,False)]:
            power=1/(1+(2*(wl-1.55)/width)**2)
            row=ring.characterize(wl,power)[0]
            self.assertAlmostEqual(row['wavelength_um'],1.55)
            if resolved:
                self.assertAlmostEqual(row['loaded_q_half_max'],1.55/width,places=6)
            else:
                self.assertIsNone(row['loaded_q_half_max'])
        power=1/(1+(2*(wl-1.55)/.004)**2)
        self.assertIsNone(ring.characterize(wl,power,post_source_record=900.)[0]['loaded_q_half_max'])
        self.assertIsNotNone(ring.characterize(wl,power,post_source_record=2000.)[0]['loaded_q_half_max'])

    def test_straight_vertical_reduction_is_a_guided_root(self):
        try:
            ring=load('ring')
        except ModuleNotFoundError as exc:
            if exc.name != 'scipy':
                raise
            self.skipTest('ring analysis imports SciPy from the Meep numerical environment')
        n=ring.slab_mode(3.48,.18,1.55,False)
        self.assertAlmostEqual(n,2.69580907588,places=9)
        u=np.pi*.18/1.55*np.sqrt(3.48**2-n**2)
        tail=np.pi*.18/1.55*np.sqrt(n*n-1.44**2)
        self.assertAlmostEqual(u*np.tan(u),tail,places=10)

    def test_finest_ring_pair_must_straddle_center_at_adjacent_orders(self):
        try:
            ring=load('ring')
        except ModuleNotFoundError as exc:
            if exc.name != 'scipy':
                raise
            self.skipTest('ring analysis imports SciPy from the Meep numerical environment')
        rows=[dict(angular_order=m,second_post_source_window={'wavelength_um':w})
              for m,w in [(26,1.512),(25,1.538),(24,1.565)]]
        self.assertEqual([r['angular_order'] for r in ring.bracketed_pole_pair(rows)],[25,24])
        with self.assertRaisesRegex(RuntimeError,'do not bracket'):
            ring.bracketed_pole_pair(rows[:2])
        with self.assertRaisesRegex(RuntimeError,'missing angular order'):
            ring.bracketed_pole_pair([rows[0],rows[-1]])

    def test_two_vendor_gratings_have_their_own_observable_and_geometry(self):
        grating=load('grating')
        tidy,ansys=grating.CASES['tidy3d'],grating.CASES['lumerical']
        self.assertEqual((tidy['teeth'],tidy['period'],tidy['fill'],tidy['published']), (20,.620,.80,.5623))
        self.assertEqual((ansys['teeth']*ansys['period'],ansys['etch'],ansys['published']), (50.,.030,.557))
        self.assertNotEqual(tidy['tooth_at_start'],ansys['tooth_at_start'])

    def test_slab_phase_matching_and_source_project_coordinates(self):
        grating=load('grating')
        self.assertAlmostEqual(grating.phase_matching_estimate(grating.CASES['tidy3d'])['estimated_period_nm'],607.541,places=3)
        self.assertAlmostEqual(grating.phase_matching_estimate(grating.CASES['lumerical'])['estimated_period_nm'],499.393,places=3)
        project=grating.ANSYS_FILED
        self.assertEqual((project['oxide_n'],project['beam_x'],project['waist_radius']), (1.44,14.8,17.288))
        self.assertAlmostEqual(project['beam_z'],2.-(-1.3))

    def test_percentage_point_comparison_reorders_descending_wavelengths(self):
        grating=load('grating')
        measured=grating.compare_ansys_spectrum([1.34,1.31,1.28],[.6,.5,.4])
        i=measured['wavelength_nm'].index(1310.)
        self.assertAlmostEqual(measured['measured_percent'][i],50.)
        self.assertAlmostEqual(measured['difference_percentage_points'][i],-5.61)
        with self.assertRaisesRegex(ValueError,'cover the published'):
            grating.compare_ansys_spectrum([1.30,1.32],[.4,.5])

    def test_weak_band_edges_are_masked_and_power_gain_fails(self):
        grating=load('grating')
        incident=np.ones(21)
        incident[[0,-1]]=.005
        response=np.full(21,.5)
        self.assertEqual(np.count_nonzero(grating.valid_beam_bins(incident,response)),19)
        self.assertTrue(grating.beam_band_passes(incident,response))
        response[10]=1.1
        self.assertFalse(grating.beam_band_passes(incident,response))

    def test_checkpoint_reuse_settings_isolation_and_integrity(self):
        spec=importlib.util.spec_from_file_location('calibration_checkpoint',ROOT/'scripts/calibration_checkpoint.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as dirname:
            folder=Path(dirname)
            with patch.dict(sys.modules,{'meep':SimpleNamespace(__version__='test'),
                                         'scipy':SimpleNamespace(__version__='test')}):
                cache=module.MeasurementCheckpoints(folder,__file__)
            calls=[]
            def measure():
                calls.append(True)
                return {'power':.25},{'mode':np.array([.5+0j])}
            first=cache.acquire('port',{'mesh':20},measure)
            second=cache.acquire('port',{'mesh':20},measure)
            np.testing.assert_array_equal(first[1]['mode'],second[1]['mode'])
            cache.acquire('port',{'mesh':28},measure)
            self.assertEqual(len(calls),2)
            for path in folder.glob('*/arrays.npz'):
                with path.open('ab') as stream:
                    stream.write(b'damaged')
            with self.assertRaisesRegex(ValueError,'integrity'):
                cache.acquire('port',{'mesh':20},measure)

    def test_published_grating_power_and_ring_complex_coefficients_reconstruct(self):
        folder=ROOT/'docs/verification'
        for report_name,prefix in [('v011-grating-compact-tidy3d','tidy_compact'),
                                   ('v011-grating-ansys-article-r40','lumerical')]:
            report=json.loads((folder/(report_name+'.json')).read_text())
            archive=folder/report['arrays_file']
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),report['arrays_sha256'])
            with np.load(archive,allow_pickle=False) as raw:
                power=abs(raw[prefix+'_device_modes'][:,1])**2/-raw[prefix+'_reference'][0]
                np.testing.assert_allclose(power,report['result']['beam_to_guide'])
        report=json.loads((folder/'v011-ring-loaded-lumerical-fine.json').read_text())
        series=report['series']['lumerical']
        prefix=series['cases']['fine']['array_prefix']
        archive=folder/report['arrays_file']
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),report['arrays_sha256'])
        with np.load(archive,allow_pickle=False) as raw:
            reference,device=raw[prefix+'_reference_modes'],raw[prefix+'_device_modes']
            outgoing=np.stack([device[:,0,1]-reference[:,0,1],device[:,1,0],
                               device[:,2,1],device[:,3,0]],axis=1)
            measured=outgoing/reference[:,0,0,None]
            np.testing.assert_allclose(measured,raw[prefix+'_s'])
            np.testing.assert_allclose(abs(measured[:,2])**2,series['drop'])

    def test_canonical_meep_reports_retain_weak_bins_and_actual_fluxes(self):
        folder=ROOT/'docs/verification'
        grating=json.loads((folder/'v011-grating.json').read_text())
        raw=folder/grating['arrays_file']
        self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),grating['arrays_sha256'])
        with np.load(raw,allow_pickle=False) as arrays:
            for vendor,controls in grating['studies'].items():
                for row in controls.values():
                    label=row['label']
                    left,right,up,down=arrays[label+'_guide_device']/arrays[label+'_guide_reference'][0]
                    np.testing.assert_allclose(row['power_sum'],1-left+right+up-down)
                    np.testing.assert_allclose(row['guide_radiation_up'],up)
                    self.assertEqual(sum(row['valid_beam_bins']),19)
                if vendor=='lumerical':
                    self.assertLess(sum(controls['fine']['valid_energy_bins']),len(controls['fine']['wavelength_um']))
        ring=json.loads((folder/'v011-ring.json').read_text())
        self.assertTrue(ring['passed'])
        for row in ring['series'].values():
            self.assertEqual(sum(row['valid_bins']),561)
        short=ring['series']['tidy3d']['cases']['short-fine']
        self.assertLess(sum(short['valid_bins']),len(short['valid_bins']))
        for case in ring['isolated_finite_height_ring'].values():
            self.assertLess(case['measured_bare_ring_pair_um'][0],1.55)
            self.assertGreater(case['measured_bare_ring_pair_um'][1],1.55)
            self.assertFalse(case['absolute_poles_stable_within_1nm'])

    def test_ansys_total_flux_and_late_window_are_separate_measured_controls(self):
        folder=ROOT/'docs/verification'
        far_reports=[]
        for mesh in (40,60):
            far=json.loads((folder/f'v011-grating-ansys-output-r{mesh}.json').read_text())
            result=far['result']
            far_reports.append(far)
            self.assertEqual((result['output_x_um'],result['output_aperture_um']),(-40.,1.))
            archive=folder/far['arrays_file']
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),far['arrays_sha256'])
            with np.load(archive,allow_pickle=False) as raw:
                incoming=-raw['lumerical_reference'][0]
                np.testing.assert_allclose(result['direct_output_fraction'],-raw['lumerical_output_flux']/incoming)
                np.testing.assert_allclose(result['beam_to_guide'],abs(raw['lumerical_device_modes'][:,1])**2/incoming)
        coarse,fine=far_reports
        self.assertEqual(coarse['physical_model'],fine['physical_model'])
        change=100*(fine['result']['center_direct_output_fraction']-coarse['result']['center_direct_output_fraction'])
        self.assertAlmostEqual(fine['same_observable_refinement']['delta_direct_percentage_points'],change)
        long=json.loads((folder/'v011-grating-guided-lumerical-r40-long.json').read_text())
        control=long['window_control']
        self.assertEqual((control['earlier_qualified_bins'],control['longer_qualified_bins']),(13,19))
        self.assertLess(control['longer_max_abs_balance_error'],.01)
