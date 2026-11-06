"""Native FDTDX source-only regression for a tilted fiber crossing a Yee-sampled interface."""
import importlib.util
import unittest


class FiberSourceProfile(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('fdtdx'), 'Requires the isolated FDTDX runtime.')
    def test_lp01_sheet_preserves_actual_staggered_material(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        import fdtdx
        from scripts import calibrate_finite_grating as solver
        from scripts import lumerical_grating3d as ansys
        m=solver.settings('lumerical',12)
        # FDTDX source-only initialization. No time stepping or hardware substitution.
        dx=1/12; n=64
        sim=fdtdx.SimulationConfig(time=1e-15,resolution=dx*1e-6,backend='cpu',dtype=jnp.float32)
        key=jax.random.PRNGKey(0)
        lp=ansys.lp01(m['fiber_radius'],m['fiber_core_n'],m['fiber_clad_n'],1.55)
        volume=fdtdx.SimulationVolume(name='domain',partial_grid_shape=(n,n,8),material=fdtdx.Material(permittivity=1.))
        pulse=fdtdx.WaveCharacter(wavelength=1.55e-6)
        Source=solver.waist_source_type(m,lp)
        source=Source(name='fiber',partial_grid_shape=(n,n,1),direction='-',fixed_E_polarization_vector=(0.,1.,0.),
                      azimuth_angle=10.,normalize_by_energy=False,center_x=0.,center_y=n/2,waist_cells=4.5/dx,wave_character=pulse)
        rules=[source.set_grid_coordinates(axes=(0,1,2),sides=('-','-','-'),coordinates=(0,0,4))]
        obj,arr,params,sim,_=fdtdx.place_objects(object_list=[volume,source],config=sim,constraints=rules,key=key)
        values=np.broadcast_to(np.where(np.hypot(np.arange(n)[:,None],np.arange(n)[None,:]-n/2)*dx<m['fiber_radius'],m['fiber_core_n']**2,m['fiber_clad_n']**2)[:,:,None],(n,n,8))
        staggered=solver.aligned(values.astype(np.float32))
        self.assertTrue(np.any(staggered[0]!=staggered[1]))
        arr=arr.aset('inv_permittivities',jnp.array(1/staggered))
        arr,obj,_=fdtdx.apply_params(arr,obj,params,key)
        np.testing.assert_allclose(np.array(arr.inv_permittivities),1/staggered)
        e,h=np.asarray(obj['fiber']._E),np.asarray(obj['fiber']._H)
        err=float(np.max(abs(np.linalg.norm(h,axis=0)-lp[2]*np.linalg.norm(e,axis=0)))/np.max(np.linalg.norm(h,axis=0)))
        self.assertLess(err,2.e-4)
        k=np.cross(e[:,:, :,0].transpose(1,2,0),h[:,:,:,0].transpose(1,2,0)).sum(axis=(0,1)); k/=np.linalg.norm(k)
        np.testing.assert_allclose(k,[-np.sin(np.deg2rad(10)),0,-np.cos(np.deg2rad(10))],atol=2.e-4)
        # TFSF's time offset is minus the dot product of position and wavevector.
        # A tilted fiber eigenmode must use a single propagation constant throughout its rim.
        delay=np.asarray(obj['fiber']._time_offset_E)[1,:,:,0]
        slope=np.sin(np.deg2rad(10))*dx*1e-6*lp[2]/(299792458*sim.time_step_duration)
        np.testing.assert_allclose(np.diff(delay,axis=0),slope,rtol=3e-5,atol=3e-5)
        np.testing.assert_allclose(np.diff(delay,axis=1),0.,atol=1e-5)
