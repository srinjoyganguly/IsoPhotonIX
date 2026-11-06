"""Calibrate fundamental quasi-TE ports of a finite-height silicon polygon layout on CUDA."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import stat
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import planar_port_layout as layout
from backend.adapters.fdtdx_ports_adapter import _ports_runtime, measured_energy_tail
from backend.adapters.fdtdx_ports3d_adapter import canonical_source_profile, modal_amplitudes, port_basis
from scripts.calibrate_fdtdx_multiport import digest, normalize, DFT_LIMIT, ENERGY_LIMIT
from scripts.grating_cuda_identity import cache_salt, fingerprint

SOURCES = (
    'scripts/calibrate_layout_ports3d.py', 'scripts/calibrate_fdtdx_multiport.py',
    'scripts/grating_cuda_identity.py', 'scripts/calibration_checkpoint.py',
    'backend/__init__.py', 'backend/planar_port_layout.py', 'backend/geometry_draft.py',
    'backend/calibrated_geometry.py', 'backend/adapters/fdtdx_ports_adapter.py',
    'backend/adapters/fdtdx_ports3d_adapter.py', 'backend/adapters/fdtdx3d_adapter.py',
    'backend/adapters/fdtdx_adapter.py',
)
SILICON, OXIDE, HEIGHT = 3.48, 1.44, .25
WAVELENGTHS = np.array([1.60, 1.55, 1.50])
FREQ = 1 / WAVELENGTHS
LIGHT = 299792458.
PULSE_WIDTH, PULSE_END, AFTER = .16, 62.5, 80.

def gpu_lock_path():
    override = os.environ.get('PHOTONICS_GPU_LOCK')
    if override:
        return Path(override).expanduser()
    uid = os.getuid()
    runtime = os.environ.get('XDG_RUNTIME_DIR')
    fallback = Path(tempfile.gettempdir())/f'photonics-workbench-{uid}'
    for directory in ([Path(runtime),fallback] if runtime else [fallback]):
        if directory==fallback:
            directory.mkdir(mode=0o700,exist_ok=True)
        try:
            info = directory.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(info.st_mode) and info.st_uid==uid and info.st_mode & 0o077==0:
            return directory/'photonics-workbench-gpu.lock'
    raise PermissionError('A private per-user directory is required for the GPU lock.')


def file_digest(path):
    with path.open('rb') as data:
        return hashlib.file_digest(data, 'sha256').hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def grid(design, resolution):
    shape = (8*resolution, 8*resolution, 4*resolution)
    aperture = round(layout.APERTURE * resolution)
    ports = []
    for port in design['ports']:
        axis, sign = layout.OUTWARD[port['outward']]
        start = [0, 0, (shape[2]-aperture)//2]
        start[axis] = round(port['center_um'][axis]*resolution + shape[axis]/2)
        transverse = 1-axis
        start[transverse] = round(port['center_um'][transverse]*resolution + (shape[transverse]-aperture)/2)
        extent = [aperture]*3
        extent[axis] = 1
        source = list(start)
        source[axis] += sign*round(.5*resolution)
        ports.append({**port, 'axis':axis, 'inward_sign':-sign, 'start':start, 'shape':extent, 'source':source,
                      'sampled_center_um':[(start[k]+(aperture/2 if k!=axis else 0)-shape[k]/2)/resolution for k in (0,1)]})
    return shape, ports


def permittivity(design, resolution, launch, reference):
    """Area quadrature on each actual component grid, exact thickness overlap.

    For Ez, mix in series across the horizontal interfaces, then average lateral
    areas. For Ex/Ey use parallel area/volume mixing at each component sample.
    """
    shape, ports = grid(design, resolution)
    polygons = design['polygons']
    leads = [layout.lead_bounds(p) for p in ports]
    port = ports[launch]
    dx = 1/resolution
    substeps = (np.arange(8)+.5)/8-.5
    eps = np.empty((3, *shape), np.float32)
    for c in range(3):
        xs, ys, zs = [(np.arange(n)-n/2+.5*(k==c))*dx for k,n in enumerate(shape)]
        fill = np.zeros(shape[:2], dtype=float)
        for ox in substeps:
            x = (xs+ox*dx)[:,None]
            for oy in substeps:
                y = (ys+oy*dx)[None,:]
                if reference:
                    transverse = y if port['axis']==0 else x
                    center = port['center_um'][1-port['axis']]
                    inside = (transverse>=center-port['width_um']/2) & (transverse<center+port['width_um']/2)
                else:
                    inside = np.zeros(shape[:2], bool)
                    for polygon in polygons:
                        hit = np.zeros_like(inside)
                        for (x0,y0),(x1,y1) in zip(polygon, polygon[1:]+polygon[:1]):
                            if x0!=x1:
                                hit ^= (x>=min(x0,x1)) & (x<max(x0,x1)) & (y < y0+(x-x0)*(y1-y0)/(x1-x0))
                        inside |= hit
                    for x0,x1,y0,y1 in leads:
                        inside |= (x>=x0)&(x<x1)&(y>=y0)&(y<y1)
                    # Carry the half-open lead through the last boundary sample's exterior half cell.
                    for p in ports:
                        normal = x if p['axis']==0 else y
                        transverse = y if p['axis']==0 else x
                        t = p['center_um'][1-p['axis']]
                        inside |= ((-p['inward_sign']*normal>=layout.HALF_CELL-dx) &
                                   (transverse>=t-p['width_um']/2)&(transverse<t+p['width_um']/2))
                fill += inside
        fill /= len(substeps)**2
        fraction = np.clip(np.minimum(zs+dx/2, HEIGHT/2)-np.maximum(zs-dx/2, -HEIGHT/2), 0, dx)/dx
        vertical = (1/(fraction/SILICON**2+(1-fraction)/OXIDE**2) if c==2
                    else OXIDE**2+fraction*(SILICON**2-OXIDE**2))
        eps[c] = OXIDE**2+fill[:,:,None]*(vertical-OXIDE**2)
    return eps


def frame(values, axis, axial=False):
    """Exchange x and y without reversing the transverse samples.

    The exchange reverses handedness: H is axial, so its three components
    change sign; E is polar. This transform and its inverse are identical.
    """
    if axis==0:
        return values
    swapped = values[[1,0,2]].swapaxes(1,2)
    return -swapped if axial else swapped


def local_phasors(values, axis):
    if axis==0:
        return values
    p = values[:,[1,0,2,4,3,5]].swapaxes(2,3).copy()
    p[:,3:] *= -1
    return p


def stage(design, resolution, launch, reference, identity):
    jax,jnp,fdtdx,target = _ports_runtime('gpu')
    from fdtdx.core.physics.modes import compute_mode
    from fdtdx.fdtd.forward import forward

    if target.device_kind!=identity['name'] or target.id!=0:
        raise RuntimeError('CUDA target does not match the checkpoint device.')
    shape, ports = grid(design, resolution)
    port = ports[launch]
    pml = round(layout.PML*resolution)
    dx = 1/resolution
    sim = fdtdx.SimulationConfig(time=(PULSE_END+AFTER)*1e-6/LIGHT, resolution=dx*1e-6,
                                 backend='gpu',dtype=jnp.float32,courant_factor=.99)
    volume = fdtdx.SimulationVolume(name='domain',partial_grid_shape=shape,material=fdtdx.Material(permittivity=OXIDE**2))
    boundary = fdtdx.BoundaryConfig(**{f'thickness_grid_{side}{axis}':pml for side in ('min','max') for axis in 'xyz'},
                                    **{f'boundary_type_{side}{axis}':'pml' for side in ('min','max') for axis in 'xyz'})
    bounds,constraints = fdtdx.boundary_objects_from_config(boundary,volume)
    items = [volume,*bounds.values()]
    constraints = list(constraints)
    def place(item, coordinates):
        items.append(item)
        constraints.append(item.set_grid_coordinates(axes=(0,1,2),sides=('-','-','-'),coordinates=tuple(coordinates)))
    direction = '+' if port['inward_sign']==1 else '-'
    center = LIGHT*1e6 / 1.55
    source_wave = fdtdx.WaveCharacter(frequency=center)
    source = fdtdx.ModePlaneSource(name='launch',partial_grid_shape=tuple(port['shape']),mode_index=0,
        filter_pol='te',direction=direction,wave_character=source_wave,
        temporal_profile=fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(frequency=PULSE_WIDTH*LIGHT*1e6/(2*np.pi)),
                                                    center_wave=source_wave),
        switch=fdtdx.OnOffSwitch(end_time=PULSE_END*1e-6/LIGHT))
    place(source,port['source'])
    if reference:
        end = list(port['start'])
        end[port['axis']] = shape[port['axis']] - end[port['axis']]
        monitors = [port, {**port,'name':'opposite reference plane','start':end}]
    else:
        monitors = ports
    waves = tuple(fdtdx.WaveCharacter(frequency=float(f*LIGHT*1e6)) for f in FREQ)
    for i,m in enumerate(monitors):
        place(fdtdx.PhasorDetector(name=f'port{i}',partial_grid_shape=tuple(m['shape']),wave_characters=waves,
                                   scaling_mode='pulse',exact_interpolation=True), m['start'])
    if not reference:
        place(fdtdx.PhasorDetector(name='preview',partial_grid_shape=(*shape[:2],1),wave_characters=(waves[1],),
                                   components=('Ex','Ey','Ez'),scaling_mode='pulse',exact_interpolation=True),(0,0,shape[2]//2))
    place(fdtdx.EnergyDetector(name='energy',partial_grid_shape=tuple(n-2*pml for n in shape),reduce_volume=True,
          exact_interpolation=True,switch=fdtdx.OnOffSwitch(interval=max(1,round(1e-15/sim.time_step_duration))),plot=False),
          (pml,pml,pml))
    key = jax.random.PRNGKey(0)
    objects,arrays,params,sim,_ = fdtdx.place_objects(object_list=items,config=sim,constraints=constraints,key=key)
    eps = permittivity(design,resolution,launch,reference)
    arrays = arrays.aset('inv_permittivities',jax.device_put(1/eps,target))
    arrays,objects,_ = fdtdx.apply_params(arrays,objects,params,key)
    if arrays.fields.E.device!=target or not np.allclose(np.asarray(1/arrays.inv_permittivities),eps,rtol=1e-6):
        raise RuntimeError('Native material coefficients or GPU field placement differ from the requested grid.')
    prepared = objects['launch']
    e,h,neff = compute_mode(frequency=center,inv_permittivities=prepared._inv_permittivity,
        inv_permeabilities=prepared._inv_permeability,resolution=dx*1e-6,direction=direction,mode_index=0,
        filter_pol='te',dtype=jnp.float32)
    axis = port['axis']
    e,h,_ = canonical_source_profile(frame(np.asarray(e),axis), frame(np.asarray(h),axis,axial=True))
    e,h = frame(e,axis),frame(h,axis,axial=True)
    if not OXIDE<float(np.real(neff))<SILICON:
        raise RuntimeError('The source cross-section has no guided fundamental quasi-TE mode.')
    prepared = prepared.aset('_E',jax.device_put(e,target)).aset('_H',jax.device_put(h,target))
    objects = objects.aset('object_list',[prepared if item.name=='launch' else item for item in objects.object_list])
    raw = {'source_e':e,'source_h':h,'midplane_permittivity':eps[:,:,:,shape[2]//2]}
    basis_by_slice, bases = {},[]
    for i,m in enumerate(monitors):
        slices = tuple(slice(s,s+n) for s,n in zip(m['start'],m['shape']))
        local = frame(np.array(arrays.inv_permittivities[(slice(None),*slices)]),m['axis'])
        signature = digest(local)
        if signature not in basis_by_slice:
            basis_by_slice[signature] = port_basis(jax.device_put(local,target),FREQ,FREQ*LIGHT*1e6,dx,
                                                    sim.time_step_duration,OXIDE,SILICON)
        basis,details = basis_by_slice[signature]
        bases.append((basis,details))
        raw[f'basis_e_{i}'],raw[f'basis_h_{i}'] = basis['mode_e'],basis['mode_h']
    def evolve(state,end):
        return jax.lax.while_loop(lambda st:st[0]<end,lambda st:forward(state=st,config=sim,objects=objects,key=key,
            record_detectors=True,record_boundaries=False,simulate_boundaries=True),state)
    state = (jnp.asarray(0,dtype=jnp.int32),arrays)
    dt_fs = sim.time_step_duration*1e15
    window_fs = 8e9/LIGHT
    end = math.ceil((PULSE_END*1e9/LIGHT+window_fs)/dt_fs)
    compiled = jax.jit(evolve,donate_argnums=(0,)).lower(state,jnp.int32(end)).compile()
    schedule = np.flatnonzero(np.asarray(objects['energy']._is_on_at_time_step_arr))
    previous,consecutive,history = None,0,[]
    while True:
        state = compiled(state,jnp.int32(end))
        step,final = state
        jax.block_until_ready(final)
        step = int(step)
        amplitudes = []
        for i,(m,(basis,_)) in enumerate(zip(monitors,bases)):
            p = np.asarray(final.detector_states[f'port{i}']['phasor'])[0]
            raw[f'phasors_{i}'] = p
            amplitudes.append(modal_amplitudes(local_phasors(p,m['axis']),basis['mode_e'],basis['mode_h'],dx**2)[0])
        amplitudes = np.asarray(amplitudes)
        if not np.isfinite(amplitudes).all() or np.max(abs(amplitudes))<=0:
            raise RuntimeError('Invalid measured modal amplitudes.')
        energies,times,ratio = measured_energy_tail(np.asarray(final.detector_states['energy']['energy'])[:,0],schedule,step,dt_fs,window_fs)
        change = None if previous is None else float(np.max(abs(amplitudes-previous))/np.max(abs(amplitudes)))
        consecutive = consecutive+1 if ratio<=ENERGY_LIMIT and change is not None and change<=DFT_LIMIT else 0
        history.append({'time_fs':step*dt_fs,'tail_energy_ratio':ratio,'relative_amplitude_change':change})
        if consecutive>=2 or step>=sim.time_steps_total:
            break
        previous = amplitudes.copy()
        end = min(sim.time_steps_total, step+math.ceil(window_fs/dt_fs))
    if final.fields.E.device!=target:
        raise RuntimeError('Final fields left the requested GPU.')
    raw.update(amplitudes=amplitudes,energy=energies,energy_time_fs=times)
    if not reference:
        raw['midplane_e'] = np.asarray(final.detector_states['preview']['phasor'])[0,0,:,:,:,0]
    if not all(np.isfinite(a).all() for a in raw.values()):
        raise RuntimeError('Nonfinite native field sample.')
    report = {'field_device':str(target),'mode_neff':{m['name']:b[1]['neff_real'] for m,b in zip(monitors,bases)},
              'max_basis_edge_fraction':max(max(b[1]['edge_electric_l2_fraction']) for b in bases),
              'source_neff':float(np.real(neff)),'source_profile_sha256':digest(np.stack([e,h])),
              'material_sha256':digest(eps),'settled':consecutive>=2,'history':history,'steps':step}
    return report,raw


def stage_spec(design, resolution, launch, reference, checkpoint):
    return {'design':design,'resolution':resolution,'launch':launch,'reference':reference,
            'sources':{name:file_digest(ROOT/name) for name in SOURCES},
            'versions':{p:importlib.metadata.version(p) for p in ('fdtdx','jax','jaxlib','numpy','tidy3d')},
            'device':fingerprint(cache_salt(checkpoint))}


def acquire_child(spec, checkpoint, destination):
    label = json_digest(spec)
    args = [sys.executable,str(Path(__file__).resolve()),'--native',str(checkpoint),str(destination),label]
    subprocess.run(args,input=json.dumps(spec),text=True,check=True)


def native_stage(checkpoint, destination, label):
    spec = json.load(sys.stdin)
    with gpu_lock_path().open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        expected = stage_spec(spec['design'],spec['resolution'],spec['launch'],spec['reference'],checkpoint)
        if spec!=expected or json_digest(expected)!=label:
            raise ValueError('Source, GPU or configuration changed between orchestration and acquisition.')
        entry = checkpoint/label
        if not entry.exists():
            print('Measuring', spec['design']['name'], spec['launch'], 'reference' if spec['reference'] else 'device',flush=True)
            report, raw = stage(spec['design'],spec['resolution'],spec['launch'],spec['reference'],spec['device'])
            with tempfile.TemporaryDirectory(dir=checkpoint) as temp:
                sample = Path(temp)/'stage'
                sample.mkdir()
                np.savez_compressed(sample/'arrays.npz',**raw)
                receipt = {'specification':spec,'report':report,'report_sha256':json_digest(report),
                           'archive_sha256':file_digest(sample/'arrays.npz')}
                (sample/'receipt.json').write_text(json.dumps(receipt,allow_nan=False))
                sample.rename(entry)
        receipt = json.loads((entry/'receipt.json').read_text())
        if (receipt['specification']!=expected or json_digest(receipt['report'])!=receipt['report_sha256']
                or file_digest(entry/'arrays.npz')!=receipt['archive_sha256']):
            raise ValueError('Stage checkpoint failed integrity verification.')
        destination.write_text(label)


def run(design, resolution, dest, checkpoint):
    shape,ports = grid(design,resolution)
    checkpoint.mkdir(parents=True,exist_ok=True)
    raw, pairs, stages = {}, [], []
    for i,p in enumerate(ports):
        row = []
        for ref in (True,False):
            label = f"port{i}-{'reference' if ref else 'device'}"
            spec = stage_spec(design,resolution,i,ref,checkpoint)
            path = dest/'stage-key'
            acquire_child(spec,checkpoint,path)
            entry = checkpoint/path.read_text()
            path.unlink()
            receipt = json.loads((entry/'receipt.json').read_text())
            with np.load(entry/'arrays.npz',allow_pickle=False) as data:
                fields = {key:data[key] for key in data.files}
            raw.update({f'{label}_{k}':v for k,v in fields.items()})
            stages.append({'label':label,'stage_key':entry.name,'archive_sha256':receipt['archive_sha256'],
                           **receipt['report']})
            row.append((fields,receipt['report']))
        (ref,rs),(dev,ds) = row
        error = max(float(np.linalg.norm(ref[k]-dev[k])/np.linalg.norm(ref[k])) for k in ('source_e','source_h'))
        if not np.isfinite(error) or error>1e-5:
            raise ValueError('The measured reference and device injection profiles differ.')
        ds['reference_source_relative_difference'] = error
        stages[-1]['reference_source_relative_difference'] = error
        pairs.append((ref['amplitudes'],dev['amplitudes'],rs,ds))
    model = {'ports':ports,'frequency_per_um':FREQ}
    s,incident,through,valid = normalize(model,pairs)
    power = abs(s)**2
    checks = {'reference_power_error':float(np.max(abs(abs(through)**2-1))),
              'max_complex_reciprocity_residual':float(np.max(abs(s-s.transpose(0,2,1)))),
              'largest_squared_singular_value':float(np.max(np.linalg.svd(s,compute_uv=False)**2)),
              'max_collected_port_power':float(power.sum(axis=1).max())}
    limits = {'reference_power_error':.03,'max_complex_reciprocity_residual':.05,
              'largest_squared_singular_value':1.05,'max_collected_port_power':1.03}
    passed = bool(np.all(valid) and all(checks[k]<=lim for k,lim in limits.items()))
    device = {r['field_device'] for r in stages}
    if len(device)!=1:
        raise ValueError('The source and device stages ran on different targets.')
    np.savez_compressed(dest/'arrays.npz',s=s,power=power,wavelength_um=WAVELENGTHS,incident=incident,
                        reference_through=through,valid_column=valid,**raw)
    preview = {'array':'port0-device_midplane_e','index':[1-ports[0]['axis']],'axis_order':'xy',
               'horizontal_axis':'x','vertical_axis':'y','horizontal_um':((np.arange(shape[0])-shape[0]/2)/resolution).tolist(),
               'vertical_um':((np.arange(shape[1])-shape[1]/2)/resolution).tolist(),'component':'Ey' if ports[0]['axis']==0 else 'Ex',
               'wavelength_um':1.55,'description':'Real part of the port 1 device electric phasor on the central Ez grid plane.'}
    report = {'schema_version':1,'case':'layout-ports-3d','solver':'FDTDX','actual_device':device.pop(),
              'device_name':spec['device']['name'],'versions':spec['versions'],'sources':spec['sources'],'passed':passed,
              'channels':[p['name'] for p in ports],'wavelength_um':WAVELENGTHS.tolist(),'s_real':s.real.tolist(),
              's_imag':s.imag.tolist(),'power':power.tolist(),'reference_through_power':(abs(through)**2).tolist(),
              'checks':checks,'limits':limits,'parameters':{'design':design,'resolution':resolution,'core_height_um':HEIGHT,
              'core_index':SILICON,'cladding_index':OXIDE,'cell_um':[8.,8.,4.],'pml_um':layout.PML,'aperture_um':[2.,2.]},
              'sampled_ports':ports,'stages':stages,'field_preview':preview,
              'matrix_axes':['wavelength','outgoing port','incident port'],
              'phase':'Physical exp(-i omega t), +Ey at x-normal ports and +Ex at y-normal ports for both directions. Reflection reference subtracted. Propagation phase retained.',
              'scope':'Fundamental quasi-TE submatrix of a fixed-height lossless oxide-clad silicon extrusion with cardinal ports. Uncollected power includes radiation, other modes and numerical error.'}
    (dest/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return passed


def main():
    if sys.argv[1:2]==['--native'] and len(sys.argv)==5:
        native_stage(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path)
    parser.add_argument('--device',choices=('gpu',),default='gpu')
    parser.add_argument('--resolution',type=int,choices=(12,16),default=16)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output directory.')
    design = layout.validate_planar_port_layout(json.loads(args.design.read_text()) if args.design else layout.default_planar_port_layout())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        dest = Path(tmp)/'result'
        dest.mkdir()
        passed = run(design,args.resolution,dest,args.checkpoint or Path(tmp)/'checkpoint')
        dest.rename(args.output)
    if not passed:
        raise SystemExit('Numerical qualification limits were exceeded. Inspect report.json.')


if __name__=='__main__':
    main()
