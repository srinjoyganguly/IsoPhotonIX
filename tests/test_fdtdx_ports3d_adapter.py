"""Analytical waves test complex projection math; real runs are separate evidence."""
import unittest

import numpy as np

from backend.adapters.fdtdx_ports3d_adapter import canonical_source_profile, collocate_mode, longitudinal_yee_permittivity, modal_amplitudes, require_quasi_te_mode


class ModalProjectionTests(unittest.TestCase):
    def test_taper_reference_uses_the_incident_ends_width_over_the_whole_cell(self):
        from backend.taper_port_geometry import port_layout
        from backend.taper_ports import example_taper_ports
        from backend.adapters.fdtdx_ports3d_adapter import taper_stage_material

        config = example_taper_ports().model_dump()
        model = port_layout(config)
        device = taper_stage_material(model, config, reverse=False, reference=False)['occupied']
        reverse_device = taper_stage_material(model, config, reverse=True, reference=False)['occupied']
        left = taper_stage_material(model, config, reverse=False, reference=True)['occupied']
        right = taper_stage_material(model, config, reverse=True, reference=True)['occupied']
        np.testing.assert_array_equal(device, reverse_device)
        self.assertTrue((left == device[0]).all())
        self.assertTrue((right == device[-1]).all())
        self.assertGreater(right.sum(), left.sum())

    def test_modes_must_actually_have_predominantly_ey_polarization(self):
        e=np.zeros((3,1,4,4),complex);e[1]=2j;e[2]=1
        self.assertAlmostEqual(require_quasi_te_mode(e),.8)
        e[1]=1;e[2]=2j
        with self.assertRaisesRegex(RuntimeError,'predominantly Ey'):
            require_quasi_te_mode(e)
        with self.assertRaisesRegex(RuntimeError,'zero transverse'):
            require_quasi_te_mode(np.zeros_like(e))
        e[1,0,0,0]=np.nan
        with self.assertRaisesRegex(RuntimeError,'finite'):
            require_quasi_te_mode(e)

    def test_gap_material_is_mirror_symmetric_at_each_actual_yee_position(self):
        eps=np.full((16,4,4),4.);eps[6:10]=1.
        components=longitudinal_yee_permittivity(eps)
        np.testing.assert_array_equal(components[0],components[0,::-1])
        np.testing.assert_array_equal(components[1,1:],components[1,:0:-1])
        np.testing.assert_array_equal(components[2,1:],components[2,:0:-1])
        self.assertEqual(components[1,6,1,1],2.5)
        self.assertEqual(components[1,10,1,1],2.5)
        uniform=longitudinal_yee_permittivity(np.full((8,4,4),2.25))
        np.testing.assert_array_equal(uniform,np.full((3,8,4,4),2.25))

    def test_pure_imaginary_source_gauge_cannot_erase_reverse_excitation(self):
        e=np.zeros((3,1,4,4),complex);h=np.zeros_like(e)
        e[1]=2j;h[2]=-3j
        electric,magnetic,phase=canonical_source_profile(e,h)
        np.testing.assert_allclose(electric[1],2)
        np.testing.assert_allclose(magnetic[2],-3)
        self.assertAlmostEqual(abs(phase),1)

    def test_complex_superposition_separates_counterpropagating_amplitudes(self):
        e=np.zeros((3,3,4,5),complex);h=np.zeros_like(e)
        e[:,1]=2;h[:,2]=3
        forward=np.array([.7+.2j,-.3+.8j,1.1-.4j])
        backward=np.array([.1-.3j,.2+.2j,-.1+.05j])
        p=np.concatenate([(forward+backward)[:,None,None,None]*e,
            (forward-backward)[:,None,None,None]*h],axis=1)
        a,power=modal_amplitudes(p,e,h,.05)
        np.testing.assert_allclose(power,3.)
        np.testing.assert_allclose(a[:,0],forward*np.sqrt(3.))
        np.testing.assert_allclose(a[:,1],backward*np.sqrt(3.))
        self.assertTrue(np.any(np.abs(a.imag)>0))

    def test_reference_subtraction_is_complex_and_preserves_signed_phase(self):
        e=np.zeros((3,3,2,2),complex);h=np.zeros_like(e);e[:,1]=1;h[:,2]=1
        ref=np.concatenate([e,h],axis=1)*(1+.5j)
        reflected=np.concatenate([e,-h],axis=1)*(.2-.3j)
        reference,_=modal_amplitudes(ref,e,h,.5)
        total,_=modal_amplitudes(ref+reflected,e,h,.5)
        actual=(total[:,1]-reference[:,1])/reference[:,0]
        np.testing.assert_allclose(actual,(.2-.3j)/(1+.5j))
        self.assertTrue(np.all(actual.imag<0))

    def test_normalization_is_invariant_to_positive_modal_scale(self):
        e=np.zeros((3,3,2,3),complex);h=np.zeros_like(e);e[:,1]=1;h[:,2]=2
        p=np.concatenate([e,h],axis=1)*(.8-.3j)
        a,_=modal_amplitudes(p,e,h,.01)
        scaled,_=modal_amplitudes(p,e*7,h*7,.01)
        np.testing.assert_allclose(a,scaled)

    def test_colocation_applies_longitudinal_and_time_yee_factors(self):
        e=np.ones((3,1,7,8),complex);h=e*2
        ec,hc=collocate_mode(e,h,.4,.2)
        np.testing.assert_allclose(ec[1,2:-2,2:-2],1)
        np.testing.assert_allclose(ec[2],1)
        np.testing.assert_allclose(hc[1],2*np.cos(.2)*np.cos(.1))
        np.testing.assert_allclose(hc[2,2:-2,2:-2],2*np.cos(.2)*np.cos(.1))

    def test_invalid_forward_mode_or_inconsistent_grid_rejected(self):
        e=np.zeros((3,3,2,3),complex);h=np.zeros_like(e);e[:,1]=1;h[:,2]=-1
        p=np.concatenate([e,h],axis=1)
        with self.assertRaises(ValueError):modal_amplitudes(p,e,h,.01)
        with self.assertRaises(ValueError):modal_amplitudes(p,e[:,:,:,:2],h,.01)


if __name__=='__main__':unittest.main()
