"""Physical reference-plane and independent modal basis contracts."""
import unittest
import numpy as np
from backend.taper_ports import example_taper_ports, TaperPortStudy
from backend.sparameters import validate_sparameters
from backend.study3d import parse_study
from tests.test_modal_ports import fixture


class TaperPortContracts(unittest.TestCase):
    def test_leads_and_two_reference_planes_are_outside_the_transition(self):
        study=example_taper_ports()
        self.assertEqual(parse_study(study.model_dump()).model_dump(),study.model_dump())
        layout=study.layout()
        self.assertLess(layout['source_forward_x_um'],layout['port_1_x_um'])
        self.assertGreaterEqual(layout['x_in']-layout['port_1_x_um'],.75)
        self.assertGreaterEqual(layout['port_2_x_um']-layout['x_out'],.75)
        self.assertLess(layout['port_2_x_um'],layout['source_reverse_x_um'])
        self.assertNotEqual(layout['width_in'],layout['width_out'])
        for changes in ({'solver':'meep','device':'gpu'},{'wavelength_min_um':1.9},
                        {'resolution':32},{'frequency_points':1}):
            with self.assertRaises(ValueError):TaperPortStudy.model_validate({**study.model_dump(),**changes})

    def test_spectrum_requires_guided_modes_at_both_physical_ends(self):
        a,m=fixture();m['config']=example_taper_ports().model_dump()
        with self.assertRaisesRegex(ValueError,'both port-specific'):
            validate_sparameters(a,m)
        a['mode_neff_ports']=np.column_stack([a['mode_neff'],np.full(5,1.8+0j)])
        valid=validate_sparameters(a,m)
        self.assertTrue(valid['valid_bin'].all())
        for modes in (np.column_stack([a['mode_neff'],np.ones(5,complex)*.90]),
                      np.column_stack([np.ones(5,complex)*1.9,np.ones(5,complex)*1.8])):
            with self.assertRaises(ValueError):validate_sparameters({**a,'mode_neff_ports':modes},m)
