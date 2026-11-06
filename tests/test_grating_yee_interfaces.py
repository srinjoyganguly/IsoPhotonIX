"""Physical controls for direct Yee-cell constitutive sampling of the curved chip."""
import unittest
from unittest.mock import patch
import numpy as np
import importlib.util
import struct

from scripts import calibrate_finite_grating as grating
from scripts import lumerical_grating3d as curved


class GratingInterfaces(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('scipy'),'The external field diagnostic uses SciPy.')
    def test_filed_complex_vectors_preserve_component_order_and_port_number(self):
        from scripts.compare_grating_port_basis import read_matrix
        name=b'freqEHny0\0'
        shape=(1,2,3,1)
        real=np.arange(6,dtype='<f8')
        def entry(scale):
            header=struct.pack('<I',len(name))+name+struct.pack('<II',3,4)+b'E1y\0'
            dims=struct.pack('<8I',6,4,*shape,0,1)
            return header+dims+(scale*real).tobytes()+(2*scale*real).tobytes()
        data=entry(1)+entry(3)
        observed=read_matrix(data,'freqEHny0',1)
        np.testing.assert_array_equal(observed,(3*real+6j*real).reshape(2,3,order='F'))

    @unittest.skipUnless(importlib.util.find_spec('scipy'),'Numerical phase screening uses SciPy.')
    def test_two_sampling_models_approach_the_independent_continuous_slab(self):
        from scripts.grating_phase_matching import screen
        continuous=screen(None)['estimated_first_order_wavelength_um']
        coarse,refined=([screen(mesh,kind)['estimated_first_order_wavelength_um'] for kind in
                         ('volume-average','direct-yee')] for mesh in (14,96))
        self.assertLess(coarse[0],continuous)
        self.assertGreater(coarse[1],continuous)
        for actual in refined:
            self.assertLess(abs(actual-continuous),.001)
    def test_model_selection_preserves_previous_settings(self):
        original=grating.settings('lumerical',14)
        refined=grating.settings('lumerical',14,'direct-yee')
        self.assertEqual(refined.pop('grating_material_model'),'direct-yee')
        self.assertEqual(refined,original)
        for name in ('compact','tidy3d','compact-radiation'):
            with self.assertRaises(ValueError):
                grating.settings(name,12,'direct-yee')

    def test_layer_limits_and_tangential_normal_interface_conditions(self):
        m=grating.settings('lumerical',12)
        dx=1/m['resolution']
        axes=(np.array([-.5,0.,.5]),np.array([-.5,0.,.5]),np.arange(-4,10)*dx+.5*dx)
        ref=np.full((3,3,3,len(axes[2])),m['n_oxide']**2,dtype=np.float32)
        # Three lateral columns are all oxide, residual film and full ridge.
        base=np.broadcast_to(np.array([0.,1.,1.])[:,None],(3,3))
        ridge=np.broadcast_to(np.array([0.,0.,1.])[:,None],(3,3))
        with patch.object(curved,'xy_coverage',return_value=(base,ridge)):
            actual=curved.silicon_on_yee_grid(m,axes,ref)
        for comp in range(3):
            z=axes[2]-(0. if comp==2 else dx/2)
            for i,h in enumerate((0.,m['height']-m['etch'],m['height'])):
                f=np.clip((np.minimum(z+dx/2,h)-np.maximum(z-dx/2,0))/dx,0,1)
                if comp==2:
                    expected=1/(f/m['n_si']**2+(1-f)/m['n_oxide']**2)
                else:
                    expected=m['n_oxide']**2+f*(m['n_si']**2-m['n_oxide']**2)
                np.testing.assert_allclose(actual[comp,i,1],expected,rtol=2e-7)
                self.assertAlmostEqual(dx*f.sum(),h,places=12)

    def test_material_outside_chip_is_unchanged(self):
        m=grating.settings('lumerical',12)
        dx=1/m['resolution']
        axes=(np.arange(-4,4)*dx,np.arange(-4,4)*dx,np.arange(-16,19)*dx)
        original=grating.aligned(curved.epsilon(m,axes,False))
        refined=curved.silicon_on_yee_grid(m,axes,original.copy())
        for comp in range(3):
            z=axes[2]-(0. if comp==2 else dx/2)
            outside=(z < -1.5*dx)|(z > m['height']+1.5*dx)
            np.testing.assert_array_equal(original[comp,:,:,outside],refined[comp,:,:,outside])
        self.assertGreater(float(abs(refined-original).max()),.1)
