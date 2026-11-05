"""Compare signed FDTDX plane reflections with both direct fields and Fresnel theory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "backend").is_dir():
    sys.path.insert(0, str(ROOT))
from scripts.calibrate_fdtdx_multiport import _ports_runtime, yee_material, modal_amplitudes


def pair(v):
    return [float(v.real), float(v.imag)]


def stage(runtime,resolution,side,device):
    # Positive and negative launches are independently simulated at one physical plane.
    jax,jnp,fdtdx,target=runtime
    dx=1./resolution
    frequency=1/1.55
    shape=(8*resolution,3,3)
    config=fdtdx.SimulationConfig(time=135/299792458e6,resolution=dx*1e-6,backend='cpu',dtype=jnp.float32,courant_factor=.99)
    n=1. if side<0 else 2.
    volume=fdtdx.SimulationVolume(name='domain',partial_grid_shape=shape,material=fdtdx.Material(permittivity=n*n if not device else 1.))
    bc=fdtdx.BoundaryConfig(thickness_grid_minx=resolution,thickness_grid_maxx=resolution,
                       boundary_type_miny='periodic',boundary_type_maxy='periodic',boundary_type_minz='periodic',boundary_type_maxz='periodic')
    bounds, constraints=fdtdx.boundary_objects_from_config(bc,volume)
    waves=(fdtdx.WaveCharacter(frequency=frequency*299792458e6),)
    source=fdtdx.UniformPlaneSource(name='source',partial_grid_shape=(1,3,3),fixed_E_polarization_vector=(0,1,0),
                  direction='+' if side<0 else '-',normalize_by_energy=False,wave_character=waves[0],
                  temporal_profile=fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(frequency=.16*299792458e6/(2*np.pi)),center_wave=waves[0]),
                  switch=fdtdx.OnOffSwitch(end_time=75/299792458e6))
    detector=fdtdx.PhasorDetector(name='port',partial_grid_shape=(1,3,3),wave_characters=waves,scaling_mode='pulse',exact_interpolation=True)
    energy=fdtdx.EnergyDetector(name='energy',partial_grid_shape=(6*resolution,3,3),reduce_volume=True,exact_interpolation=True,
                switch=fdtdx.OnOffSwitch(interval=max(1,round(1e-15/config.time_step_duration))),plot=False)
    constraints=list(constraints)
    for item,x in ((source,4*resolution+2*side*resolution),(detector,4*resolution+side*resolution),(energy,resolution)):
        constraints.append(item.set_grid_coordinates(axes=(0,1,2),sides=('-','-','-'),coordinates=(x,0,0)))
    key=jax.random.PRNGKey(0)
    objects,arrays,params,sim,_=fdtdx.place_objects(object_list=[volume,*bounds.values(),source,detector,energy],constraints=constraints,key=key,config=config)
    eps=np.full(shape,n*n if not device else 1.,dtype=np.float32)
    if device: eps[shape[0]//2:]=4.
    staggered=yee_material(eps,'yee-aligned')
    arrays=arrays.aset('inv_permittivities',jax.device_put(1/staggered,target))
    arrays,objects,_=fdtdx.apply_params(arrays,objects,params,key)
    if not np.allclose(1/np.asarray(arrays.inv_permittivities),staggered):
        raise ValueError('Unexpected stepped material.')
    step,end=jax.jit(lambda a:fdtdx.run_fdtd(arrays=a,objects=objects,config=sim,key=key,show_progress=False))(arrays)
    jax.block_until_ready(end)
    if end.fields.E.device!=target:
        raise ValueError('Final field on unexpected target.')
    ph=np.asarray(end.detector_states['port']['phasor'])[0]
    t=np.asarray(end.detector_states['energy']['energy']).reshape(-1)
    peak=max(t)
    tail=t[-40:].max()/peak
    if tail>1e-4 or not np.isfinite(ph).all():
        raise ValueError(f'Unsettled pulse tail {tail:g}.')
    # Normalized fields in FDTDX have H_z/E_y=n for a +x plane wave in
    # the uniform incident material. Correct component interpolation factors.
    omega_dt=2*np.pi*frequency*299792458e6*sim.time_step_duration
    beta_dx=2*np.arcsin(n*dx*1e-6/(299792458*sim.time_step_duration)*np.sin(omega_dt/2))
    c=np.zeros((1,3,3,3),dtype=float)
    h=np.zeros_like(c)
    c[:,1]=1.
    h[:,2]=n*np.cos(beta_dx/2)*np.cos(omega_dt/2)
    amps,_=modal_amplitudes(ph,c,h,dx*dx)
    return ph[:,1].mean(),amps[0],float(tail),int(step)


def record(rows):
    import importlib.metadata
    passed = all(
        row['projection_minus_direct_abs'] < 0.004
        and row['opposite_sign_minus_direct_abs'] > 0.55
        and row['projection_minus_analytic_abs'] < (0.018 if row['resolution'] == 32 else 0.005)
        for row in rows
    )
    return {
        'schema_version': 1,
        'passed': passed,
        'solver': 'FDTDX',
        'version': importlib.metadata.version('fdtdx'),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'physical_model': 'A plane wave normally incident on a step between n=1 and n=2 at x=0, in both directions; uniform periodic transverse cells and axial PML. Each port is one micrometre from the interface.',
        'measurement': 'Subtract each matched straight-reference backwave, divide by its measured incident wave, then compare with direct Ey subtraction and the complex electric Fresnel coefficient propagated back to the physical plane.',
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resolutions', nargs='+', choices=(32,64), type=int, default=[32,64])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output path.')
    runtime = _ports_runtime('cpu')
    rows = []
    for res in args.resolutions:
        for side, n1, n2 in ((-1, 1., 2.), (1, 2., 1.)):
            e0, a, tail0, steps = stage(runtime, res, side, False)
            e1, b, tail1, _ = stage(runtime, res, side, True)
            branch = 0 if side < 0 else 1
            direct = (e1 - e0) / e0
            projected = (b[1-branch] - a[1-branch]) / a[branch]
            theory = (n1-n2)/(n1+n2) * np.exp(4j*np.pi*n1/1.55)
            row = {
                'resolution': res, 'side': 'left' if side < 0 else 'right', 'physical_distance_um': 1.,
                'analytic': pair(theory), 'direct_Ey': pair(direct), 'reciprocal_EH': pair(projected),
                'projection_minus_direct_abs': float(abs(projected-direct)),
                'projection_minus_analytic_abs': float(abs(projected-theory)),
                'opposite_sign_minus_direct_abs': float(abs(projected+direct)),
                'actual_device': str(runtime[-1]), 'native_time_steps': steps, 'energy_tails': [tail0,tail1],
            }
            rows.append(row)
            print(row, flush=True)
    report = record(rows)
    with args.output.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    if not report['passed']:
        raise SystemExit('Reflected field disagrees with its direct or analytic control.')


if __name__ == '__main__':
    main()
