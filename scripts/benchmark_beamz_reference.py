"""Run a local BeamZ plane-wave propagation reference on one explicit JAX device."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np

RELEASE = '0.6.2'
RELEASE_COMMIT = 'b2db93f11b6ae94160b7980866106320d2f161f5'
MODE_REFERENCE = 'https://optics.ansys.com/hc/en-us/articles/1500007378182-Inverse-design-of-an-efficient-narrow-band-grating-coupler'
MODE_NEFF = 2.84825
WAVELENGTH_M = 1.55e-6


def phase_and_impedance(e_before, e_after, h_after, x_e, x_h, wavenumber, duration):
    """Project signed E and H samples over complete periods at their staggered positions."""
    ae0 = np.mean(e_before * np.exp(-1j*wavenumber*x_e))
    ae1 = np.mean(e_after * np.exp(-1j*wavenumber*x_e))
    ah1 = np.mean(h_after * np.exp(-1j*wavenumber*x_h))
    if min(abs(ae0), abs(ae1), abs(ah1)) <= 1e-12:
        raise ValueError('Insufficient field amplitude in the measurement interval.')
    advance = -np.angle(ae1/ae0)
    if not 0 < advance < np.pi:
        raise ValueError('The forward phase increment must be smaller than π.')
    return float(advance/duration), float(abs(ae1/ah1)), float(abs(ae1/ae0))


def plane_wave(points_per_period, platform, arrays):
    import jax.numpy as jnp
    from beamz import Design, EPS_0, LIGHT_SPEED, MU_0, Material, Simulation

    n = 1.5
    period = WAVELENGTH_M / n
    dx = period/points_per_period
    dt = .9 * dx / (LIGHT_SPEED*np.sqrt(2))
    k = 2*np.pi/period
    omega = LIGHT_SPEED*k/n
    impedance = np.sqrt(MU_0/EPS_0)/n
    steps = 8
    simulation = Simulation(
        design=Design(width=10*period, height=6*period, material=Material(permittivity=n*n)),
        sources=[], boundaries=[], resolution=dx, time=np.arange(steps)*dt,
    )
    state = simulation.initial_state()
    x_e = np.arange(state.ez.shape[1])*dx
    x_h = (np.arange(state.hy.shape[1])+.5)*dx
    # Initialize a continuum forward wave on the staggered grids. No fitted or discrete
    # frequency is used to prepare it. The evolution independently measures numerical dispersion.
    e0 = np.cos(k*x_e)
    h0 = -np.cos(k*x_h + omega*dt/2)/impedance
    state = state._replace(ez=jnp.asarray(np.broadcast_to(e0,state.ez.shape), dtype=state.ez.dtype),
                           hy=jnp.asarray(np.broadcast_to(h0,state.hy.shape), dtype=state.hy.dtype))
    final = simulation.advance(state=state, backend='jax', progress=False, performance=False).state
    device = final.ez.device
    if device.platform != platform or final.hy.device != device:
        raise RuntimeError(f'Expected {platform} field arrays, found {device}.')
    ez, hy = np.asarray(final.ez), np.asarray(final.hy)
    if not (np.isfinite(ez).all() and np.isfinite(hy).all()):
        raise ValueError('Non-finite BeamZ field.')
    interior = slice(3*points_per_period,7*points_per_period)
    rate, z, amplitude = phase_and_impedance(
        np.asarray(state.ez)[ez.shape[0]//2,interior], ez[ez.shape[0]//2,interior],
        hy[hy.shape[0]//2,interior], x_e[interior], x_h[interior], k, steps*dt)
    discrete_rate = 2/dt * np.arcsin(LIGHT_SPEED*dt/(n*dx)*np.sin(k*dx/2))
    label = f'p{points_per_period}'
    arrays.update({label+'_Ez':ez, label+'_Hy':hy, label+'_initial_Ez_row':np.asarray(state.ez)[ez.shape[0]//2],
                   label+'_initial_Hy_row':np.asarray(state.hy)[hy.shape[0]//2], label+'_x_E_m':x_e, label+'_x_H_m':x_h})
    return {'points_per_material_wavelength':points_per_period, 'n':n, 'vacuum_wavelength_m':WAVELENGTH_M,
            'dx_m':dx,'dt_s':dt,'steps':steps,'measurement_slice':[interior.start,interior.stop],
            'grid_shape_Ez_yx':list(ez.shape),'actual_device':str(device),'device_kind':device.device_kind,
            'actual_platform':device.platform,'field_dtype':str(ez.dtype),'phase_velocity_m_s':rate/k,
            'continuum_velocity_m_s':LIGHT_SPEED/n,'discrete_velocity_m_s':discrete_rate/k,
            'continuum_relative_error':abs(rate/omega-1),'yee_relative_error':abs(rate/discrete_rate-1),
            'impedance_ohm':z,'analytic_impedance_ohm':impedance,'impedance_relative_error':abs(z/impedance-1),
            'amplitude_ratio':amplitude}


def slab_modes(arrays):
    """A separate CPU eigenproblem for the 220 nm TE slab in the Ansys example."""
    from beamz import LIGHT_SPEED
    from beamz.devices.modes.plane import solve_modes
    from scipy.optimize import brentq

    n_core, n_clad, thickness = 3.48, 1.44, 220e-9
    half = thickness/2
    v = 2*np.pi/WAVELENGTH_M*half*np.sqrt(n_core*n_core - n_clad*n_clad)
    u = brentq(lambda u:u*np.tan(u)-np.sqrt(v*v-u*u),0,min(v,np.pi/2-1e-12))
    analytic = float(np.sqrt(n_core*n_core - (u*WAVELENGTH_M/(half*2*np.pi))**2))
    cases = []
    for dx_nm, span_um in [(20,4.),(10,4.),(5,4.),(2.5,4.),(2.5,6.)]:
        dx = dx_nm*1e-9
        cells, core_cells = round(span_um*1e-6/dx), round(thickness/dx)
        start = (cells-core_cells)//2
        epsilon = np.full(cells,n_clad*n_clad)
        epsilon[start:start+core_cells] = n_core*n_core
        neff, electric, magnetic, _ = solve_modes(epsilon,omega=2*np.pi*LIGHT_SPEED/WAVELENGTH_M,
            dL=dx,m=1,direction='+x',filter_pol='tm',return_fields=True,target_neff=.95*n_core)
        if abs(neff[0].imag)>1e-9 or not n_clad<neff[0].real<n_core:
            raise ValueError('The lossless slab fundamental must have a real guided index.')
        # For this line profile, BeamZ's "tm" filter selects out-of-plane Ez (slab TE).
        strengths = np.linalg.norm(electric[0],axis=1)
        if np.argmax(strengths)!=2 or not np.isfinite(electric).all() or not np.isfinite(magnetic).all():
            raise ValueError('The measured electric field must have dominant out-of-plane polarization.')
        label = f'slab_{dx_nm}_{span_um}'
        arrays.update({label+'_epsilon':epsilon,label+'_E':electric[0],label+'_H':magnetic[0],
                       label+'_y_m':(np.arange(cells)+.5-cells/2)*dx})
        value = float(neff[0].real)
        cases.append({'grid_nm':dx_nm,'aperture_um':span_um,'neff':value,'analytic_neff':analytic,
                      'published_neff':MODE_NEFF,'analytic_relative_error':abs(value/analytic-1),
                      'published_relative_error':abs(value/MODE_NEFF-1)})
    return cases


def replay_notebook(source, target, output):
    intro = ('# BeamZ passive reference\nChoose a kernel with BeamZ 0.6.2 or set `BEAMZ_PYTHON` '
             'to that interpreter. The complete source is embedded. Each run needs a fresh output folder.\n')
    code = """import json, os, subprocess, sys, tempfile
from pathlib import Path
output = Path('beamz-result').resolve()
target = TARGET
with tempfile.TemporaryDirectory() as tmp:
    script = Path(tmp) / 'run_beamz_reference.py'
    script.write_text(source, encoding='utf-8')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['MPLBACKEND'] = 'Agg'
    subprocess.run([env.get('BEAMZ_PYTHON',sys.executable),str(script),'--device',target,
                    '--output',str(output),'--no-export'],check=True,env=env)
report = json.loads((output / 'report.json').read_text())
[(row['points_per_material_wavelength'], row['phase_velocity_m_s'], row['continuum_relative_error'])
 for row in report['propagation']]""".replace('TARGET',repr(target))
    cells = [{'cell_type':'markdown','id':'intro','metadata':{},'source':intro}]
    for name, snippet in [('source','source = '+repr(source)),('measure',code)]:
        cells.append({'cell_type':'code','id':name,'metadata':{},'outputs':[],'execution_count':None,'source':snippet})
    nb = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{
        'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    output.write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8')


def run(target, out, export):
    # This executable owns its process. Select JAX before BeamZ is imported.
    os.environ['JAX_PLATFORMS'] = 'cuda' if target=='gpu' else 'cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if version('beamz') != RELEASE:
        raise ValueError(f'This reference is qualified for BeamZ {RELEASE}.')
    import jax
    platform = 'gpu' if target=='gpu' else 'cpu'
    device = jax.devices()[0]
    if device.platform != platform:
        raise RuntimeError(f'Requested {target}, JAX initialized {device}.')
    arrays = {}
    wave = [plane_wave(ppw,platform,arrays) for ppw in (12,24,48)]
    modes = slab_modes(arrays) if target == 'cpu' else []
    checks = {'continuum_phase_error_reduces_on_each_refinement':
                 wave[0]['continuum_relative_error']>wave[1]['continuum_relative_error']>wave[2]['continuum_relative_error'],
              'finest_phase_within_0_1_percent':wave[-1]['continuum_relative_error']<=.001,
              'discrete_phase_within_0_08_percent':all(r['yee_relative_error']<=.0008 for r in wave),
              'impedance_and_amplitude_within_0_1_percent':all(max(r['impedance_relative_error'],abs(r['amplitude_ratio']-1))<=.001 for r in wave)}
    if modes:
        fine, wider = modes[-2:]
        checks.update(slab_fine_analytic_within_0_02_percent=fine['analytic_relative_error']<=.0002,
                      slab_fine_published_within_0_2_percent=fine['published_relative_error']<=.002,
                      slab_wider_domain_change_below_1e_6=abs(fine['neff']-wider['neff'])<=1e-6)
    np.savez_compressed(out/'fields.npz',**arrays)
    source = Path(__file__).read_bytes()
    report={'schema_version':1,'case_id':'beamz-passive-reference','passed':all(checks.values()),'checks':checks,
            'release':RELEASE,'tag_commit':RELEASE_COMMIT,'solver_source_sha256':hashlib.sha256(source).hexdigest(),
            'device_requested':target,'implementation':'BeamZ JAX Maxwell time stepper','actual_device':str(device),
            'device_kind':device.device_kind,'versions':{p:version(p) for p in ('beamz','jax','jaxlib','numpy','scipy')},
            'python':sys.version.split()[0],'propagation':wave,'slab_modes_cpu':modes,
            'slab_reference_url':MODE_REFERENCE,'fields_sha256':hashlib.sha256((out/'fields.npz').read_bytes()).hexdigest(),
            'limits':'Short-time homogeneous dielectric propagation and a separate CPU slab-mode comparison. General 3D layouts, port measurements, gain and BeamZ native CUDA kernels require separate qualification.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if export:
        (out/'run_beamz_reference.py').write_bytes(source)
        replay_notebook(source.decode('utf-8'),target,out/'beamz-reference.ipynb')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','gpu'],required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--no-export',action='store_true')
    args=parser.parse_args()
    dest=args.output.resolve()
    if dest.exists():
        parser.error('output exists; choose a fresh directory')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staging=Path(tmp)/'completed'
        staging.mkdir()
        report=run(args.device,staging,not args.no_export)
        staging.rename(dest)
    print(dest/'report.json', 'pass' if report['passed'] else 'outside benchmark tolerances')
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
