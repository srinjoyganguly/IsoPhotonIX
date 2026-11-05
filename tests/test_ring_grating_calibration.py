import importlib.util
from pathlib import Path
import unittest

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
