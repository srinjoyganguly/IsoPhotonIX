"""Check reciprocity, dielectric eigenvalue bounds and long source-free time evolution on real ring-wall cells."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, factorized

from scripts.calibrate_loaded_ring3d import geometry,disk_fractions,lateral_fraction,permittivity
from scripts.ring_wall_tensor import wall_inverse,forward_lossless_xy
from scripts.reciprocal_ring_wall import reciprocal_links,OFFSETS


def matrix(inverse,links):
    shape=inverse.shape[1:]; n=np.prod(shape); local=np.arange(n).reshape(shape)
    rows=[np.arange(3*n)]; columns=[rows[0]]; values=[inverse[[0,4,8]].astype(float).ravel()]
    for i,(sx,sy) in enumerate(OFFSETS):
        ex=local.ravel();ey=np.roll(local,(-sx,-sy),axis=(0,1)).ravel()+n; weights=links[i].ravel()
        rows.extend((ex,ey));columns.extend((ey,ex));values.extend((weights,weights))
    return coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),shape=(3*n,3*n)).tocsc()


def stress(platform):
    import fdtdx
    import jax
    import jax.numpy as jnp
    from scripts.gpu_timing import StageTiming
    target=jax.devices(platform)[0]
    timer=StageTiming(target)
    g=geometry('lumerical',24,.65,1.8,'wall-tensor')
    full=wall_inverse(g,False,disk_fractions,lateral_fraction,permittivity)
    nx,ny,nz=g['shape']
    ix=nx//2+round(g['radius_um']*24/np.sqrt(2))
    iy=ny//2+round(g['radius_um']*24/np.sqrt(2))
    tensor=full[:,ix-8:ix+8,iy-8:iy+8,nz//2-3:nz//2+3].copy()
    del full
    links,metadata=reciprocal_links(tensor,1/g['silicon_index']**2,1.,periodic=True)
    operator=matrix(tensor,links)
    discrepancy=operator-operator.T
    symmetric_error=float(np.max(abs(discrepancy.data),initial=0.))
    extremal=[float(eigsh(operator,k=1,which=which,return_eigenvectors=False,tol=1e-9)[0]) for which in ('SA','LA')]
    if symmetric_error>1e-12 or extremal[0]<metadata['inverse_lower']-2e-6 or extremal[1]>metadata['inverse_upper']+2e-6:
        raise ValueError('Staggered inverse material failed symmetry or spectrum bounds.')
    solve=factorized(operator)
    shape=tensor.shape[1:]
    key=jax.random.PRNGKey(2026)
    config=fdtdx.SimulationConfig(time=1e-12,resolution=1e-6/24,backend=platform,dtype=jnp.float32,courant_factor=.99)
    volume=fdtdx.SimulationVolume(name='tile',partial_grid_shape=shape,material=fdtdx.Material(permittivity=1.))
    boundary=fdtdx.BoundaryConfig(**{f'boundary_type_{s}{a}':'periodic' for s in ('min','max') for a in 'xyz'})
    bounds,constraints=fdtdx.boundary_objects_from_config(boundary,volume)
    objects,arrays,params,config,_=fdtdx.place_objects(object_list=[volume,*bounds.values()],constraints=constraints,key=key,config=config)
    arrays,objects,_=fdtdx.apply_params(arrays,objects,params,key)
    rng=np.random.default_rng(2026)
    initial={field:rng.normal(size=(3,*shape)).astype(np.float32) for field in ('E','H')}
    arrays=arrays.aset('inv_permittivities',jax.device_put(tensor,target))
    for field in ('E','H'):
        arrays=arrays.aset('fields->'+field,jax.device_put(initial[field],target))
    device_links=jax.device_put(links,target)
    state=(jnp.array(0,jnp.int32),arrays)
    chunk=256; batches=32
    def evolve(st,weights):
        step=lambda i,current:forward_lossless_xy(current,config,objects,weights)
        prior=jax.lax.fori_loop(0,chunk-1,step,st)
        return step(0,prior),prior[1].fields.H
    jax.block_until_ready((state,device_links))
    with timer.compile():
        advance=jax.jit(evolve,donate_argnums=(0,)).lower(state,device_links).compile()
    records=[]
    timer.begin_steps()
    for k in range(batches):
        with timer.step():
            state,h_previous=advance(state,device_links)
            jax.block_until_ready((state,h_previous))
        if state[1].fields.E.device!=target:
            raise ValueError('Stress test field left the selected device.')
        e=np.asarray(state[1].fields.E,dtype=float).ravel()
        h=np.asarray(state[1].fields.H,dtype=float).ravel()
        previous=np.asarray(h_previous,dtype=float).ravel()
        records.append(float(e @ solve(e) + h @ previous))
    timer.end_steps()
    drift=float(np.max(abs(np.asarray(records)/records[0]-1)))
    passed=bool(np.isfinite(records).all() and np.min(records)>0 and drift<2e-4 and metadata['legacy_max_transpose_discrepancy']>.001)
    root=Path(__file__).resolve().parents[1]
    files=('scripts/validate_reciprocal_ring_wall.py','scripts/validate_ring_wall_wave.py','scripts/ring_wall_tensor.py',
           'scripts/reciprocal_ring_wall.py','scripts/calibrate_loaded_ring3d.py','scripts/gpu_timing.py')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    return dict(passed=passed,numerical_source_sha256=hashes,scope='Source-free periodic tile cut from the 24-cell Lumerical ring near its 45° curved wall. Spatially varying coefficients include Si, oxide, air and corners; randomized E/H excite all grid modes. No device transfer spectrum is inferred.',
                source_geometry=g,patch_center_indices=[ix,iy,nz//2],shape=list(shape),physical_gpu=target.device_kind,
                reciprocal_operator=metadata,symmetry_max_abs_error=symmetric_error,extremal_eigenvalues=extremal,
                random_seed=2026,energy_definition='E_n^T M^{-1} E_n + H_n^T H_(n-1), the lossless discrete leapfrog invariant (mu=1).',
                recorded_energy_every_256_steps=records,relative_energy_drift=drift,drift_limit=2e-4,
                performance=timer.report(shape=shape,steps=chunk*batches,dt_seconds=config.time_step_duration,spacing_um=1/24,
                                        pml_cells={f'{s}_{a}':0 for s in ('min','max') for a in 'xyz'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=('cpu','gpu'),required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    r=stress(args.device)
    if args.device=='cpu':
        from scripts.validate_ring_wall_wave import measure,check_spatial_fields
        r['analytic_wave']=measure('reciprocal')
        r['uniform_tensor_with_source_and_six_pml_faces']=check_spatial_fields(reciprocal=True)
        r['passed'] &= r['analytic_wave']['passed'] and r['uniform_tensor_with_source_and_six_pml_faces']['passed']
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    if not r['passed']:
        raise SystemExit('Reciprocal ring energy check failed.')
