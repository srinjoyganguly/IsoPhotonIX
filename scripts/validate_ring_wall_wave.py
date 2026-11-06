"""Independent one-dimensional standing wave check of the native rotated-wall electric update."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from scripts.ring_wall_tensor import forward_lossless_xy, in_plane_inverse


def measure(engine):
    import fdtdx
    import jax
    import jax.numpy as jnp
    from fdtdx.fdtd.forward import forward

    # A 1D closed periodic test can run independently of the occupied device GPU.
    target = jax.devices('cpu')[0]
    shape=(3,3,32)
    key=jax.random.PRNGKey(2)
    sim=fdtdx.SimulationConfig(time=1e-12,resolution=50e-9,backend='cpu',dtype=jnp.float32,courant_factor=.99)
    volume=fdtdx.SimulationVolume(name='domain',partial_grid_shape=shape,material=fdtdx.Material(permittivity=1.))
    boundaries=fdtdx.BoundaryConfig(**{f'boundary_type_{face}{axis}':'periodic' for face in ('min','max') for axis in 'xyz'})
    bounds,constraints=fdtdx.boundary_objects_from_config(boundaries,volume)
    objects,arrays,params,sim,_=fdtdx.place_objects(object_list=[volume,*bounds.values()],constraints=constraints,key=key,config=sim)
    arrays,objects,_=fdtdx.apply_params(arrays,objects,params,key)
    n=np.array([np.cos(np.pi/5),np.sin(np.pi/5)])
    t=np.array([-n[1],n[0]])
    f,si,bg=.47,3.48**2,1.
    xx,xy,yy=in_plane_inverse(f,*n,1.,0.,si,1.44**2,bg)
    inv=np.array([xx,xy,0.,xy,yy,0.,0.,0.,1.],np.float32)
    arrays=arrays.aset('inv_permittivities',jax.device_put(np.broadcast_to(inv[:,None,None,None],(9,*shape)),target))
    field=np.zeros((3,*shape),np.float32)
    field[0]=np.cos(2*np.pi*np.arange(shape[-1])/shape[-1])
    arrays=arrays.aset('fields->E',jax.device_put(field,target))
    def step(state,_):
        if engine=='native_dense':
            state=forward(state=state,config=sim,objects=objects,key=key,record_detectors=False,
                          record_boundaries=False,simulate_boundaries=True)
        else:
            state=forward_lossless_xy(state,sim,objects)
        return state,state[1].fields.E[:,1,1,0]
    steps=256
    _,samples=jax.jit(lambda state:jax.lax.scan(step,state,None,length=steps))((jnp.array(0,jnp.int32),arrays))
    if samples.device!=target:
        raise ValueError('The native reference should execute on CPU.')
    signals=np.asarray(samples)
    principal=np.array([f/si+(1-f)/bg,1/(f*si+(1-f)*bg)])
    angles=2*np.arcsin(float(sim.courant_number)*np.sqrt(principal)*np.sin(np.pi/shape[-1]))
    # With H initially zero and E updated first, E(0) = E(1).
    analytic=np.cos((np.arange(steps)[:,None]+.5)*angles)/np.cos(.5*angles)
    expected=(analytic[:,0]*n[0])[:,None]*n+(analytic[:,1]*t[0])[:,None]*t
    error=float(np.max(abs(expected-signals[:,:2])))
    cross=float(np.max(abs(signals[:,1])))
    passed=error<2e-5 and cross>.3 and np.max(abs(signals[:,2]))<1e-6
    return dict(scope='One-dimensional field varying along z. All axes periodic. Uniform off-diagonal tensor; both principal waves excited. This checks the native update independently of source, PML and curved-ring geometry.',
                electric_update=engine,platform=target.platform,solver=version('fdtdx'),jax=version('jax'),shape=shape,steps=steps,
                courant_number=float(sim.courant_number),fraction=f,angle_radians=float(np.pi/5),inverse_tensor=inv.tolist(),
                oracle='Leapfrog Yee recurrence with analytic normal/tangential eigenvalues, E initially cos(kz) on x and H initially zero.',
                max_field_absolute_error=error,maximum_cross_polarized_field=cross,
                passing_max_error=2e-5,passed=bool(passed),sample_z_index=0,
                measured_ex_ey=signals[:,:2].tolist(),expected_ex_ey=expected.tolist())


def check_spatial_fields():
    """Compare native and specialized fields including source injection and all PML sides."""
    import fdtdx
    import jax
    import jax.numpy as jnp
    from fdtdx.fdtd.forward import forward

    target=jax.devices('cpu')[0]
    shape=(15,14,13)
    steps=13
    key=jax.random.PRNGKey(0)
    sim=fdtdx.SimulationConfig(time=200e-15,resolution=50e-9,backend='cpu',dtype=jnp.float32,courant_factor=.95)
    volume=fdtdx.SimulationVolume(name='domain',partial_grid_shape=shape,material=fdtdx.Material(permittivity=1.))
    boundaries=fdtdx.BoundaryConfig(**{f'thickness_grid_{face}{axis}':3 for face in ('min','max') for axis in 'xyz'})
    bounds,constraints=fdtdx.boundary_objects_from_config(boundaries,volume)
    source=fdtdx.UniformPlaneSource(name='source',partial_grid_shape=(1,6,6),fixed_E_polarization_vector=(0.,1.,0.),
        direction='+',normalize_by_energy=False,wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6))
    constraints=list(constraints)+[source.set_grid_coordinates(axes=(0,1,2),sides=('-','-','-'),coordinates=(4,4,4))]
    objects,arrays,params,sim,_=fdtdx.place_objects(object_list=[volume,*bounds.values(),source],constraints=constraints,
                                                    key=key,config=sim)
    arrays,objects,_=fdtdx.apply_params(arrays,objects,params,key)
    rng=np.random.default_rng(348)
    inv=np.zeros((9,*shape),np.float32)
    angle=rng.uniform(-np.pi,np.pi,shape)
    fraction=rng.uniform(0,1,shape)
    inv[0],inv[1],inv[4]=in_plane_inverse(fraction,np.cos(angle),np.sin(angle),1.,0.,3.48**2,1.44**2,1.)
    inv[3]=inv[1]
    inv[8]=.52
    arrays=arrays.aset('inv_permittivities',jax.device_put(inv,target))
    for field in ('E','H'):
        arrays=arrays.aset('fields->'+field,jax.device_put(rng.normal(0,.1,(3,*shape)).astype(np.float32),target))
    def native(i,state):
        return forward(state,sim,objects,key,False,False,True)
    def block(i,state):
        return forward_lossless_xy(state,sim,objects)
    start=(jnp.array(0,jnp.int32),arrays)
    states=[jax.jit(lambda st:jax.lax.fori_loop(0,steps,step,st))(start)[1].fields for step in (native,block)]
    differences={}
    signals={}
    for name in ('E','H','psi_E','psi_H'):
        left,right=(np.asarray(getattr(fields,name)) for fields in states)
        differences[name]=float(np.abs(left-right).max())
        signals[name]=float(np.abs(left).max())
    passed=all(v<2e-5 for v in differences.values()) and all(v>1e-5 for v in signals.values())
    return dict(shape=shape,steps=steps,device=str(target),random_field_seed=348,pml_cells_each_face=3,
                source='FDTDX UniformPlaneSource',maximum_field_and_pml_memory_differences=differences,
                maximum_native_amplitudes=signals,pass_limit=2e-5,passed=passed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={name:measure(name) for name in ('native_dense','lossless_xy')}
    result['spatially_varying_tensor']=check_spatial_fields()
    result['passed']=all(item['passed'] for item in result.values())
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not result['passed']:
        raise SystemExit('Native rotated tensor analytic control failed.')
