"""Measure a finite-height four-port directional coupler or three-port Y splitter with Meep."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
import hashlib
import inspect
import json
import os
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.calibrated_geometry import multiport_model, straight_polygon
from scripts.calibration_checkpoint import read_locked_plan

C_UM_PER_FS = .299792458


def as_pairs(values):
    return np.stack([values.real, values.imag], axis=-1).tolist()


def build_simulation(mp, model, launch, resolution, reference):
    p = model['ports'][launch]
    polys = [straight_polygon(model, launch)] if reference else model['polygons']
    prisms = [mp.Prism(vertices=[mp.Vector3(x, y, -model['height']/2) for x, y in vertices],
                       height=model['height'], material=mp.Medium(index=model['core_index'])) for vertices in polys]
    source = mp.EigenModeSource(mp.GaussianSource(1/1.55, fwidth=model['pulse_fwidth'], cutoff=model['pulse_cutoff']),
           center=mp.Vector3(p['source_x'], p['y']), size=mp.Vector3(0, *model['aperture']),
           direction=mp.X, eig_kpoint=mp.Vector3(p['inward_sign']), eig_band=1, eig_match_freq=True,
           eig_parity=mp.ODD_Y + mp.EVEN_Z, eig_resolution=resolution, eig_tolerance=1e-12)
    mp.set_random_seed(0)
    sim = mp.Simulation(cell_size=mp.Vector3(*model['cell']), dimensions=3, resolution=resolution, Courant=.5,
                        default_material=mp.Medium(index=model['cladding_index']), geometry=prisms,
                        sources=[source], boundary_layers=[mp.PML(model['pml_um'])],
                        symmetries=[mp.Mirror(mp.Z, phase=1)])
    mp.set_random_seed(0)
    sim.init_sim()
    ports = [launch] if reference else list(range(len(model['ports'])))
    monitors = {}
    for i in ports:
        p = model['ports'][i]
        monitors[i] = sim.add_flux(1/1.55, .04, 3, mp.FluxRegion(center=mp.Vector3(p['x'], p['y']),
                        size=mp.Vector3(0, *model['aperture']), direction=mp.X), decimation_factor=1)
    return sim, monitors


def run_until_settled(mp, sim, monitors, model, after_time, minimum_ringdown=0.):
    """Use a relative whole-cell energy and complex-monitor increment on a common incident scale."""
    end = sim.fields.last_source_time()
    cap = end + after_time
    peak_energy = 0.
    peak_norm = 0.
    previous = None
    consecutive = 0
    history = []
    volume = mp.Volume(center=mp.Vector3(), size=mp.Vector3(*model['cell']))

    def sample(s):
        nonlocal peak_energy, peak_norm, previous, consecutive
        energy = abs(float(s.field_energy_in_box(box=volume)))
        data = [s.get_flux_data(m) for m in monitors.values()]
        current = np.concatenate([np.r_[d.E, d.H] for d in data])
        if not np.isfinite(energy) or not np.all(np.isfinite(current)):
            raise RuntimeError('Nonfinite native Meep field or DFT.')
        peak_energy = max(peak_energy, energy)
        peak_norm = max(peak_norm, float(np.linalg.norm(current)))
        ratio = energy / peak_energy if peak_energy else 1.
        change = float(np.linalg.norm(current-previous) / peak_norm) if previous is not None and peak_norm else 1.
        previous = current
        t = float(s.meep_time())
        consecutive = consecutive + 1 if t > end and ratio <= 1e-4 and change <= .002 else 0
        history.append([t, ratio, change])

    sim.run(mp.at_every(8., sample), until=lambda s: (consecutive >= 2 and s.meep_time() >= end+minimum_ringdown) or s.meep_time() >= cap)
    return {'settled': consecutive >= 2 and sim.meep_time() >= end+minimum_ringdown, 'elapsed_fs': float(sim.meep_time()/C_UM_PER_FS),
            'source_end_fs': float(end/C_UM_PER_FS), 'steps': int(sim.timestep()), 'history': history,
            'energy_limit': 1e-4, 'complex_monitor_increment_limit': .002, 'consecutive_required': 2}


def project(mp, sim, monitor, port, model, sign, resolution):
    """Signed MPB mode with a positive-real Ey anchor at its physical port centre."""
    frequency = np.asarray(mp.get_flux_freqs(monitor))
    volume = mp.Volume(center=mp.Vector3(port['x'], port['y']), size=mp.Vector3(0,*model['aperture']))
    parity = mp.ODD_Y + mp.EVEN_Z
    mp.set_random_seed(0)
    values = sim.get_eigenmode_coefficients(monitor, [1], eig_vol=volume, direction=mp.X, eig_parity=parity,
           eig_resolution=resolution, eig_tolerance=1e-12,
           kpoint_func=lambda f, band: mp.Vector3(sign * f * (model['core_index'] + model['cladding_index']) / 2))
    neff = np.array([k.x for k in values.kpoints]) / frequency
    if not np.all(sign * neff > model['cladding_index']) or not np.all(sign * np.asarray(values.vgrp) > 0):
        raise RuntimeError('A port has no guided mode with the requested propagation direction.')
    alpha = np.array(values.alpha[0,:,0 if sign == 1 else 1]).copy()
    anchors, fractions, edge_ratios = [], [], []
    for f, k in zip(frequency, values.kpoints):
        mp.set_random_seed(0)
        mode = sim.get_eigenmode(f, mp.X, volume, 1, k, parity=parity, resolution=resolution, eigensolver_tol=1e-12)
        center = mp.Vector3(port['x'],port['y'])
        ey = complex(mode.amplitude(center, mp.Ey))
        ys = port['y'] + np.linspace(-model['aperture'][0]/2, model['aperture'][0]/2, 13)
        zs = np.linspace(-model['aperture'][1]/2, model['aperture'][1]/2, 13)
        field = np.array([[[mode.amplitude(mp.Vector3(port['x'], y, z), cmp) for y in ys] for z in zs]
                          for cmp in (mp.Ex, mp.Ey, mp.Ez)])
        energy = abs(field)**2
        intensity = np.sum(energy, axis=0)
        fraction = np.sum(energy[1]) / np.sum(energy)
        if not np.isfinite(field).all() or abs(ey)<1e-8 or fraction <= .5:
            raise RuntimeError('Selected MPB mode is missing the TE-like centre-Ey gauge.')
        anchors.append(ey)
        fractions.append(fraction)
        edge_ratios.append(max(np.max(intensity[[0,-1],:]),np.max(intensity[:,[0,-1]])) / np.max(intensity))
    gauge = np.asarray(anchors) / np.abs(anchors)
    # Meep 1.35 uses the signed normal-flux overlap with a positive
    # sqrt(1/abs(normfac)) scale, even for a negative-vg same-direction mode.
    # Express the overlap in the centre-Ey electric basis, independently for
    # each requested signed mode. See src/mpb.cpp lines 967–981.
    return alpha*gauge*np.sign(values.vgrp), neff, fractions, edge_ratios


def atomic_json(path, value):
    temporary=path.with_suffix('.pending-json')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def atomic_archive(path, **values):
    temporary=path.with_suffix('.pending-npz')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**values)
    os.replace(temporary,path)


def read_checkpoint(metadata, archive, key):
    if not metadata.exists():
        return None
    receipt=json.loads(metadata.read_text())
    if receipt['checkpoint_key']!=key or receipt['archive_sha256']!=hashlib.sha256(archive.read_bytes()).hexdigest():
        raise ValueError(f'Cannot resume {metadata.name}: measured archive or simulation configuration changed.')
    return receipt


def measure_column(launch, model, resolution, after_time, directory, minimum_ringdown=0., stage='all', checkpoint_key=''):
    import meep as mp
    from meep.simulation import FluxData
    mp.verbosity(0)
    start = time.monotonic()
    out = Path(directory)
    port = model['ports'][launch]
    name = port['name']
    ref_file=out/f'{name}-reference.npz'
    ref_receipt=out/f'{name}-reference.json'
    column_file=out/f'{name}-monitors.npz'
    column_receipt=out/f'{name}.json'
    key=f'{checkpoint_key}/{name}'
    saved=read_checkpoint(ref_receipt,ref_file,key)
    if saved is None:
        if stage=='device':
            raise ValueError(f'{name} requires its completed native reference first.')
        print(f'{name}: exciting the isolated incident reference',flush=True)
        ref,ref_mon=build_simulation(mp,model,launch,resolution,reference=True)
        try:
            reference=run_until_settled(mp,ref,ref_mon,model,after_time,minimum_ringdown)
            incident,neff_in,frac,edge=project(mp,ref,ref_mon[launch],port,model,port['inward_sign'],resolution)
            ref_data=ref.get_flux_data(ref_mon[launch])
            atomic_archive(ref_file,E=ref_data.E,H=ref_data.H,incident=incident,neff=neff_in,
                           ey_fraction=frac,edge_fraction=edge)
        finally:
            ref.reset_meep()
        saved={'checkpoint_key':key,'archive_sha256':hashlib.sha256(ref_file.read_bytes()).hexdigest(),
               'seconds':time.monotonic()-start,'reference':reference}
        atomic_json(ref_receipt,saved)
        print(f'{name}: reference checkpoint after {saved["seconds"]:.1f}s, settled={reference["settled"]}',flush=True)
    if stage=='reference':
        return saved
    completed=read_checkpoint(column_receipt,column_file,key)
    if completed is not None:
        return completed
    # Reload even on the initial run so the resume path uses the same native bytes.
    with np.load(ref_file,allow_pickle=False) as arrays:
        ref_data=FluxData(E=arrays['E'],H=arrays['H'])
        incident,neff_in,frac,edge=(arrays[k] for k in ('incident','neff','ey_fraction','edge_fraction'))
    native={'reference_E':ref_data.E,'reference_H':ref_data.H}
    reference=saved['reference']
    device_start=time.monotonic()
    dev, mon = build_simulation(mp, model, launch, resolution, reference=False)
    dev.load_minus_flux_data(mon[launch], ref_data)
    device = run_until_settled(mp, dev, mon, model, after_time, minimum_ringdown)
    print(f'{name}: device fields evolved after {time.monotonic()-start:.1f}s, settled={device["settled"]}',flush=True)
    outgoing = []
    neff = []
    all_fractions, all_edges = list(frac), list(edge)
    for i,p in enumerate(model['ports']):
        d = dev.get_flux_data(mon[i])
        native[f'port{i+1}_E'] = np.asarray(d.E)
        native[f'port{i+1}_H'] = np.asarray(d.H)
        amp, ns, fs, es = project(mp, dev, mon[i], p, model, -p['inward_sign'], resolution)
        outgoing.append(amp)
        neff.append(ns)
        all_fractions.extend(fs)
        all_edges.extend(es)
    dev.reset_meep()
    outgoing = np.array(outgoing).T
    if not np.isfinite(incident).all() or (np.abs(incident)**2 < np.max(np.abs(incident)**2)*1e-6).any():
        raise RuntimeError('An incident spectral bin is insufficient for normalization.')
    column = outgoing/incident[:,None]
    atomic_archive(column_file, **native, incident=incident, outgoing=outgoing, s=column,
                        neff_reference=neff_in, neff_device_outward=np.array(neff).T)
    receipt = {'launch':name, 'reference':reference, 'device':device, 'seconds':saved['seconds']+time.monotonic()-device_start,
               'checkpoint_key':key,'archive_sha256':hashlib.sha256(column_file.read_bytes()).hexdigest(),
               'phase_convention_version':2, 'minimum_Ey_fraction':float(np.min(all_fractions)), 'maximum_edge_intensity_fraction':float(np.max(all_edges)),
               'incident':as_pairs(incident), 'outgoing':as_pairs(outgoing), 's':as_pairs(column),
               'min_incident_relative_power':float(np.min(abs(incident)**2)/np.max(abs(incident)**2))}
    atomic_json(column_receipt,receipt)
    print(f'{name}: {receipt["seconds"]:.1f}s, settled {reference["settled"]}/{device["settled"]}', flush=True)
    return receipt


def matrix_diagnostics(s, complete):
    if s.ndim != 3 or not np.isfinite(s).all():
        raise ValueError('Scattering coefficients must be a finite [frequency, outgoing, incoming] array.')
    power = np.sum(np.abs(s)**2, axis=1)
    singular = np.linalg.svd(s, compute_uv=False)[:,0]
    reciprocity = None
    if complete:
        if s.shape[1] != s.shape[2]:
            raise ValueError('Full reciprocity requires every incident port in the same basis.')
        reciprocity = np.max(np.abs(s-s.transpose(0,2,1)), axis=(1,2))
    return power, singular, reciprocity


def measure(case, output, resolution=16, pml_um=.75, aperture_um=2., after_time=80., jobs=1, launch_port=None, minimum_ringdown=0., resume=False, stage='all'):
    import meep as mp
    import fcntl
    if resolution not in (16,20,24) or jobs not in (1,2,3,4) or not 40 <= after_time <= 140 or not 0 <= minimum_ringdown <= after_time or stage not in ('all','reference','device'):
        raise ValueError('Unsupported mesh, worker count or ringdown window.')
    model = multiport_model(case,pml_um,aperture_um)
    names = [p['name'] for p in model['ports']]
    if launch_port is not None and launch_port not in names:
        raise ValueError(f'Choose a launch port from {names}.')
    ids = list(range(len(names))) if launch_port is None else [names.index(launch_port)]
    numerical_code='\n'.join(inspect.getsource(f) for f in (build_simulation,run_until_settled,project,measure_column,matrix_diagnostics))
    config=dict(model=model,resolution=resolution,max_after_source_time=after_time,minimum_ringdown=minimum_ringdown,
                version=mp.__version__,numpy_version=np.__version__,input_ports=[names[i] for i in ids],phase_convention_version=2,
                numerical_sha256=hashlib.sha256(numerical_code.encode()).hexdigest())
    config=json.loads(json.dumps(config))
    out=Path(output)
    if not resume:
        out.mkdir(parents=True,exist_ok=False)
    elif not out.is_dir():
        raise ValueError('Resume requires an existing study directory.')
    with (out/'.measurement.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=read_locked_plan(out/'config.json','.measurement.lock','config.pending-json') if resume else None
        if saved is not None and json.loads(saved)!=config:
            raise ValueError('Resume refused: physical model, solver, numerical code or inputs differ.')
        if saved is None:
            atomic_json(out/'config.json',config)
        key=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return measure_locked(mp,model,resolution,after_time,minimum_ringdown,out,jobs,names,ids,launch_port,stage,key)


def measure_locked(mp,model,resolution,after_time,minimum_ringdown,out,jobs,names,ids,launch_port,stage,key):
    fn = partial(measure_column, model=model, resolution=resolution, after_time=after_time, directory=out,
                 minimum_ringdown=minimum_ringdown,stage=stage,checkpoint_key=key)
    if jobs == 1:
        columns = list(map(fn,ids))
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context('spawn')) as pool:
            columns = list(pool.map(fn,ids))
    if stage=='reference':
        return {'case':model['case'],'stage':stage,'all_stages_settled':all(c['reference']['settled'] for c in columns)}
    s = np.stack([np.array(c['s'])[...,0]+1j*np.array(c['s'])[...,1] for c in columns],axis=-1)
    valid = all(c[k]['settled'] for c in columns for k in ('reference','device'))
    power, singular, reciprocity = matrix_diagnostics(s, launch_port is None)
    atomic_archive(out/'arrays.npz', frequency_per_um=model['frequency_per_um'], s=s,
                        incident=np.array([np.asarray(c['incident'])[:,0]+1j*np.asarray(c['incident'])[:,1] for c in columns]).T,
                        mode_power_sum=power, largest_singular_value=singular)
    report = {'schema_version':1,'case':model['case'],'checked_at':datetime.now(timezone.utc).isoformat(),
        'solver':'Meep', 'version':mp.__version__, 'numpy_version':np.__version__, 'python_version':sys.version.split()[0], 'actual_device':'CPU', 'resolution':resolution, 'model':model,
        'ports':names, 'input_ports':[names[i] for i in ids], 'full_matrix':launch_port is None, 'matrix_order':'[frequency, outgoing port, incident port]; [real,imag] pairs',
        'wavelength_um':(1/np.asarray(model['frequency_per_um'])).tolist(), 's':as_pairs(s),
        'power_sums':power.tolist(),'maximum_reciprocity_error':None if reciprocity is None else reciprocity.tolist(),'largest_singular_value':singular.tolist(),
        'phase_convention_version':2,
        'phase_convention':'Positive-real centre Ey for each signed-k MPB mode; native overlap multiplied by sign(group velocity).',
        'all_stages_settled':valid,'valid_bins':[valid]*len(s), 'stage_checks':[
            {k:c[k] for k in ('launch','minimum_Ey_fraction','maximum_edge_intensity_fraction','min_incident_relative_power','seconds')}|
            {k:{key:c[k][key] for key in ('elapsed_fs','settled','steps')} for k in ('reference','device')} for c in columns],
        'numerical_scope':'Squared amplitude is fundamental-mode power; the recorded matrix omits radiation and higher modes. Mesh and port sensitivities require separate paired studies.',
        'maximum_column_power_excess':float(max(0., np.max(power)-1)),
        'largest_singular_value_excess':float(max(0., np.max(singular)-1)),
        'arrays_sha256':hashlib.sha256((out/'arrays.npz').read_bytes()).hexdigest()}
    report['native_monitor_sha256'] = {f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                                      for f in sorted(out.glob('*-monitors.npz'))}
    report['passed']=valid
    atomic_json(out/'report.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('coupler','splitter'), required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resolution',choices=(16,20,24),type=int,default=16)
    parser.add_argument('--pml-um',type=float,default=.75)
    parser.add_argument('--aperture-um',type=float,default=2.)
    parser.add_argument('--after-time',type=float,default=80.,help='maximum ringdown in µm/c after the source')
    parser.add_argument('--jobs',type=int,choices=(1,2,3,4),default=1)
    parser.add_argument('--launch-port', help='Measure only this column for a numerical control; omit to measure every port.')
    parser.add_argument('--minimum-ringdown',type=float,default=0.,help='continue DFT accumulation for at least this many µm/c after the pulse')
    parser.add_argument('--resume',action='store_true',help='reuse validated native checkpoints in this exact study directory')
    parser.add_argument('--stage',choices=('all','reference','device'),default='all',
                        help='optionally run the straight references and device columns in separate processes')
    args=parser.parse_args()
    if not 40. <= args.after_time <= 140.: parser.error('after-time must be between 40 and 140 µm/c.')
    result=measure(args.case,args.output,args.resolution,args.pml_um,args.aperture_um,args.after_time,args.jobs,args.launch_port,args.minimum_ringdown,args.resume,args.stage)
    print(json.dumps({k:result[k] for k in ('case','stage','power_sums','maximum_reciprocity_error','all_stages_settled') if k in result},indent=2))
    if not result['all_stages_settled']: sys.exit(2)
