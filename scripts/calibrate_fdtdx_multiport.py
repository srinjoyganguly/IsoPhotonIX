"""Local multiport FDTDX 0.6.2 calibration of two finite-height PIC test circuits.

Run in the registered FDTDX interpreter. The shared physical polygons and
ported solver helpers accompany this script in a Workbench export.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'backend').is_dir():
    sys.path.insert(0, str(ROOT))

from backend.calibrated_geometry import multiport_model, straight_polygon
from scripts.calibration_checkpoint import read_locked_plan
from scripts import phase_polygon_material
from backend.adapters.fdtdx_ports_adapter import _ports_runtime, measured_energy_tail
from backend.adapters.fdtdx_ports3d_adapter import (
    canonical_source_profile, longitudinal_yee_permittivity, modal_amplitudes, port_basis,
)

C_UM_PER_FS = .299792458
ENERGY_LIMIT = 1e-4
DFT_LIMIT = .002
DEFAULT_MATERIAL_MODEL = 'yee-aligned'


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def complex_components(a):
    return np.stack((a.real, a.imag), axis=-1).tolist()


def grid(model, resolution):
    """Explicit snapped FDTDX grid and per-port apertures. Coordinates are µm."""
    dx = 1 / resolution
    shape = tuple(round(size * resolution) for size in model['cell'])
    pml = round(model['pml_um'] * resolution)
    if pml < 6:
        raise ValueError('Need at least six cells across each PML.')
    axes = [(np.arange(n) + .5 - n/2) * dx for n in shape]
    # Optional benchmark registration translates the computational lattice,
    # including PML faces and finite apertures. Physical polygons/core stay fixed.
    z_origin = model.get('z_grid_origin_um',0.)
    if z_origin:
        axes[2] += z_origin
    sy, sz = (round(size * resolution) for size in model['aperture'])
    ports = []
    for port in model['ports']:
        to_x = lambda x: round(x * resolution + shape[0]/2)
        start = (to_x(port['x']), round(port['y'] * resolution + shape[1]/2 - sy/2),
                 round((shape[2] - sz)/2))
        if start[1] < pml or start[1]+sy > shape[1]-pml or start[2] < pml or start[2]+sz > shape[2]-pml:
            raise ValueError('A guided aperture cannot extend into PML.')
        ports.append({**port, 'start':start, 'source_ix':to_x(port['source_x']),
                      'shape':(1,sy,sz), 'actual_x':(start[0]-shape[0]/2)*dx,
                      'actual_y':(start[1]+sy/2-shape[1]/2)*dx})
        if 'z_grid_origin_um' in model:
            ports[-1]['actual_z']=(start[2]+sz/2-shape[2]/2)*dx+z_origin
    return shape, pml, axes, ports


def raster(model, resolution, launch=None):
    """Cell-centre staircase: interpolate the x-monotone rail intervals exactly.

    General even-odd crossings also handle a Y's concave branching contour.
    Treat longitudinal rows and transverse voxel intervals as half-open.
    """
    shape, _, (xs, ys, zs), _ = grid(model, resolution)
    polygons = model['polygons'] if launch is None else [straight_polygon(model, launch)]
    footprint = np.zeros(shape[:2], dtype=bool)
    for polygon in polygons:
        inside = np.zeros_like(footprint)
        for (x0,y0), (x1,y1) in zip(polygon, polygon[1:]+polygon[:1]):
            cross = (xs >= min(x0,x1)) & (xs < max(x0,x1))
            if np.any(cross):
                edge = y0 + (xs[cross]-x0) * ((y1-y0)/(x1-x0))
                inside[cross] ^= ys[None,:] < edge[:,None]
        footprint |= inside
    height = model['height']
    zmask = (zs >= -height/2) & (zs < height/2)
    return footprint[:,:,None] & zmask[None,None,:]


def yee_material(epsilon, material_model):
    """Map scalar cell-center samples to the three component locations on the Yee grid."""
    if material_model == 'legacy-x':
        return longitudinal_yee_permittivity(epsilon)
    if material_model != 'yee-aligned':
        raise ValueError('Choose legacy-x or yee-aligned material sampling.')
    # Ex lies at (i+1/2,j,k), Ey at (i,j+1/2,k), Ez at (i,j,k+1/2).
    # Each tangential half-cell displacement averages adjacent scalar samples.
    # Extend the exterior by its nearest value rather than wrapping across PML.
    def average_to_face(values, axis):
        first = [slice(None)] * 3
        first[axis] = slice(0,1)
        previous = [slice(None)] * 3
        previous[axis] = slice(0,-1)
        adjacent = np.concatenate([values[tuple(first)], values[tuple(previous)]], axis=axis)
        return .5*(values+adjacent)
    return np.stack([average_to_face(average_to_face(epsilon,1),2),
                     average_to_face(average_to_face(epsilon,0),2),
                     average_to_face(average_to_face(epsilon,0),1)])


def stage(runtime, model, resolution, after_time, launch, reference, output, basis_cache, material_model='legacy-x'):
    jax, jnp, fdtdx, target = runtime
    from fdtdx.core.jax.pytrees import autoinit, field
    from fdtdx.materials import compute_ordered_names
    from fdtdx.objects.static_material.static import StaticMultiMaterialObject
    from scripts.gpu_timing import StageTiming
    timer = StageTiming(target)
    from fdtdx.core.physics.modes import compute_mode
    from fdtdx.fdtd.forward import forward

    @autoinit
    class CircuitMaterial(StaticMultiMaterialObject):
        occupancy: jax.Array = field()

        def get_voxel_mask_for_shape(self):
            return jnp.ones(self.occupancy.shape, dtype=bool)

        def get_material_mapping(self):
            return self.occupancy

    shape,pml,axes,ports = grid(model, resolution)
    dx = 1/resolution
    port = ports[launch]
    mask = raster(model, resolution, launch if reference else None)
    indices = [model['cladding_index'], model['core_index']]
    epsilon = np.where(mask, indices[1]**2, indices[0]**2).astype(np.float32)
    materials = {str(i):fdtdx.Material(permittivity=n*n) for i,n in enumerate(indices)}
    order = compute_ordered_names(materials)
    material = CircuitMaterial(name='circuit', partial_grid_shape=shape, materials=materials,
                               occupancy=jax.device_put(np.where(mask,order.index('1'),order.index('0')).astype('int32'),target))
    frequencies = np.array(model['frequency_per_um'])
    freq_hz = frequencies*299792458e6
    center = float(np.mean(freq_hz))
    # Meep GaussianSource uses sigma_t=1/fwidth in optical time units.
    source_end_fs = 2*model['pulse_cutoff']/model['pulse_fwidth']/C_UM_PER_FS
    total_fs = source_end_fs + after_time / C_UM_PER_FS
    sim = fdtdx.SimulationConfig(time=total_fs*1e-15, resolution=dx*1e-6, backend='gpu' if target.platform=='gpu' else 'cpu',
                                 dtype=jnp.float32, courant_factor=.99)
    expected_backend = 'gpu' if target.platform=='gpu' else 'cpu'
    if sim.backend != expected_backend:
        raise RuntimeError('The FDTDX backend changed.')
    volume = fdtdx.SimulationVolume(name='domain', partial_grid_shape=shape, material=materials['0'])
    boundary = fdtdx.BoundaryConfig(**{f'thickness_grid_{side}{axis}':pml for side in ('min','max') for axis in 'xyz'},
                                    **{f'boundary_type_{side}{axis}':'pml' for side in ('min','max') for axis in 'xyz'})
    bounds, constraints = fdtdx.boundary_objects_from_config(boundary, volume)
    items = [volume, *bounds.values(), material]
    constraints = list(constraints)
    def locate(item, start):
        constraints.append(item.set_grid_coordinates(axes=(0,1,2), sides=('-','-','-'), coordinates=start))
    locate(material, (0,0,0))
    direction = '+' if port['inward_sign']==1 else '-'
    source = fdtdx.ModePlaneSource(name='launch', partial_grid_shape=port['shape'], mode_index=0, filter_pol='te',
        direction=direction, wave_character=fdtdx.WaveCharacter(frequency=center),
        temporal_profile=fdtdx.GaussianPulseProfile(
            spectral_width=fdtdx.WaveCharacter(frequency=model['pulse_fwidth']*299792458e6/(2*math.pi)),
            center_wave=fdtdx.WaveCharacter(frequency=center)),
        switch=fdtdx.OnOffSwitch(end_time=source_end_fs*1e-15))
    items.append(source)
    locate(source,(port['source_ix'],*port['start'][1:]))
    if reference:
        through = {**port,'name':'straight_through','start':(shape[0]-port['start'][0],*port['start'][1:])}
        monitors = [port,through]
    else:
        monitors = ports
    waves = tuple(fdtdx.WaveCharacter(frequency=float(f)) for f in freq_hz)
    for m in monitors:
        detector = fdtdx.PhasorDetector(name=m['name'],partial_grid_shape=m['shape'],wave_characters=waves,
                                        scaling_mode='pulse',exact_interpolation=True)
        items.append(detector)
        locate(detector,m['start'])
    energy = fdtdx.EnergyDetector(name='energy',partial_grid_shape=tuple(n-2*pml for n in shape),
        reduce_volume=True,exact_interpolation=True,
        switch=fdtdx.OnOffSwitch(interval=max(1,round(1e-15/sim.time_step_duration))),plot=False)
    items.append(energy)
    locate(energy,(pml,pml,pml))
    key = jax.random.PRNGKey(0)
    objects, arrays, params, sim, _ = fdtdx.place_objects(object_list=items,config=sim,constraints=constraints,key=key)
    if not np.allclose(np.asarray(1/arrays.inv_permittivities[0]),epsilon,rtol=1e-6,atol=1e-6):
        raise RuntimeError('FDTDX material initialization differs from the requested circuit.')
    if material_model == 'polygon-area':
        polygons = [straight_polygon(model,launch)] if reference else model['polygons']
        staggered = phase_polygon_material.extruded_yee(model,resolution,axes,polygons)
    else:
        staggered = yee_material(epsilon,material_model)
    arrays = arrays.aset('inv_permittivities',jax.device_put(1/staggered,target))
    arrays,objects,_ = fdtdx.apply_params(arrays,objects,params,key)
    if arrays.fields.E.device!=target or not np.allclose(np.asarray(1/arrays.inv_permittivities),staggered,rtol=1e-6,atol=1e-6):
        raise RuntimeError('Initial field device or stepped permittivities differ.')
    native_source = objects['launch']
    e,h,n = compute_mode(frequency=center,inv_permittivities=native_source._inv_permittivity,
                         inv_permeabilities=native_source._inv_permeability,resolution=dx*1e-6,
                         direction=direction,mode_index=0,filter_pol='te',dtype=jnp.float32)
    e,h,_ = canonical_source_profile(e,h)
    native_source = native_source.aset('_E',jax.device_put(e,target)).aset('_H',jax.device_put(h,target))
    objects = objects.aset('object_list',[native_source if o.name=='launch' else o for o in objects.object_list])
    if not indices[0] < np.real(n) < indices[1]:
        raise RuntimeError('Source mode is outside the guided range.')
    bases = {}
    for m in monitors:
        i,y,z = m['start']; _,ny,nz = m['shape']
        local = arrays.inv_permittivities[:,i:i+1,y:y+ny,z:z+nz]
        fingerprint = digest(np.asarray(local))
        if fingerprint not in basis_cache:
            basis_cache[fingerprint] = port_basis(local, frequencies, freq_hz, dx, sim.time_step_duration, *indices)
        bases[m['name']] = basis_cache[fingerprint]
    def evolve(state, end):
        return jax.lax.while_loop(lambda s:s[0]<end,
            lambda s:forward(state=s,config=sim,objects=objects,key=key,record_detectors=True,
                             record_boundaries=False,simulate_boundaries=True),state)
    state = (jnp.asarray(0,dtype=jnp.int32),arrays)
    dt_fs = sim.time_step_duration*1e15
    window_fs = 8/C_UM_PER_FS
    end = min(sim.time_steps_total,math.ceil((source_end_fs+window_fs)/dt_fs))
    with timer.compile():
        compiled = jax.jit(evolve).lower(state,jnp.asarray(end,dtype=jnp.int32)).compile()
    schedule = np.flatnonzero(np.asarray(objects['energy']._is_on_at_time_step_arr))
    previous = None
    consecutive = 0
    history = []
    jax.block_until_ready(state)
    timer.begin_steps()
    while True:
        with timer.step():
            state = compiled(state,jnp.asarray(end,dtype=jnp.int32)); step,final = state
            jax.block_until_ready(final)
        step = int(step)
        phasors = {m['name']:np.asarray(final.detector_states[m['name']]['phasor'])[0] for m in monitors}
        amplitudes = np.array([modal_amplitudes(phasors[m['name']],bases[m['name']][0]['mode_e'],
                              bases[m['name']][0]['mode_h'],dx**2)[0] for m in monitors])
        energies,times,ratio = measured_energy_tail(np.asarray(final.detector_states['energy']['energy'])[:,0],
                                                  schedule, step,dt_fs,window_fs)
        if not np.isfinite(amplitudes).all() or np.max(abs(amplitudes))<=0:
            raise RuntimeError('Invalid measured modal amplitudes.')
        change = None if previous is None else float(np.max(abs(amplitudes-previous))/np.max(abs(amplitudes)))
        passed = ratio<=ENERGY_LIMIT and change is not None and change<=DFT_LIMIT
        consecutive = consecutive+1 if passed else 0
        history.append({'time_fs':step*dt_fs,'energy_ratio':ratio,'amplitude_change':change})
        if consecutive>=2 or step>=sim.time_steps_total:
            break
        previous = amplitudes.copy()
        end = min(sim.time_steps_total, step+math.ceil(window_fs/dt_fs))
    timer.end_steps()
    performance = timer.report(shape=shape, steps=step, dt_seconds=sim.time_step_duration,
                               spacing_um=dx, pml_cells=pml)
    if final.fields.E.device!=target:
        raise RuntimeError('Final EM fields left the selected device.')
    detail = {'performance':performance, 'monitor_names':[m['name'] for m in monitors], 'source_mode_neff':float(np.real(n)),
              'mode_neff': {m['name']:bases[m['name']][1]['neff_real'] for m in monitors},
              'mode_edge_fractions':{m['name']:bases[m['name']][1]['edge_electric_l2_fraction'] for m in monitors},
              'time_steps':step,'history':history,'settled':consecutive>=2,'material_grid_sha256':digest(epsilon),
              'field_device':str(final.fields.E.device),'material_model':material_model,
              'yee_component_grid_sha256':digest(staggered),'source_profile_sha256':digest(np.stack([e,h]))}
    np.savez_compressed(output,amplitudes=amplitudes,frequency_per_um=frequencies,
                        source_e=e,source_h=h,energy=energies,energy_time_fs=times,
                        **{'phasors_'+name:value for name,value in phasors.items()})
    return amplitudes,detail


def normalize(model, columns):
    nf,np_ = len(model['frequency_per_um']),len(model['ports'])
    result = np.empty((nf,np_,np_),np.complex128)
    incident, reference_through = np.empty((2,nf,np_),np.complex128)
    settled = []
    for i,(ref,dev,ref_details,dev_details) in enumerate(columns):
        name = model['ports'][i]['name']
        profiles_match = (ref_details['source_profile_sha256'] == dev_details['source_profile_sha256'] or
                          dev_details.get('reference_source_relative_difference',float('inf')) <= 1e-5)
        if not profiles_match or not np.allclose(
                ref_details['mode_neff'][name],dev_details['mode_neff'][name],rtol=1e-6,atol=1e-6):
            raise ValueError('Launch modes must agree between reference and device.')
        incoming = 0 if model['ports'][i]['inward_sign']==1 else 1
        incident[:,i] = ref[0,:,incoming]
        if np.any(abs(incident[:,i])**2 <= max(abs(incident[:,i])**2)*1e-6):
            raise ValueError('A spectral sample lacks a sufficient incident amplitude.')
        reference_through[:,i] = ref[1,:,incoming]/incident[:,i]
        for j,port in enumerate(model['ports']):
            outward = 1 if port['inward_sign']==1 else 0
            outgoing = dev[j,:,outward].copy()
            if j==i:
                outgoing -= ref[0,:,outward]
            result[:,j,i] = outgoing/incident[:,i]
        settled.append(ref_details['settled'] and dev_details['settled'])
    return result,incident,reference_through,np.asarray(settled)


def numerical_digest():
    """Fingerprint the transported worker/physics, independent of archive packaging."""
    import backend.calibrated_geometry as geometry
    import backend.adapters.fdtdx_ports3d_adapter as projection
    import backend.adapters.fdtdx_ports_adapter as runtime
    from scripts import gpu_timing
    code = [inspect.getsource(f) for f in (gpu_timing,geometry,projection,runtime,phase_polygon_material,grid,raster,yee_material,stage)]
    constants = [C_UM_PER_FS, ENERGY_LIMIT, DFT_LIMIT]
    return hashlib.sha256(('\n'.join(code)+repr(constants)).encode()).hexdigest()


def assemble(model, device, resolution, after_time, output, stages):
    """Reconstruct the normalized matrix solely from the recorded complex monitor arrays."""
    columns = []
    for i,port in enumerate(model['ports']):
        ref_key, dev_key = [port['name']+'-'+kind for kind in ('reference','device')]
        reference, device_record = stages[ref_key],stages[dev_key]
        with np.load(output/(ref_key+'.npz'),allow_pickle=False) as ref, np.load(output/(dev_key+'.npz'),allow_pickle=False) as dev:
            difference=max(float(np.linalg.norm(ref[k]-dev[k])/np.linalg.norm(ref[k])) for k in ('source_e','source_h'))
            if not np.isfinite(difference) or difference>1e-5:
                raise ValueError('The reference and device launches have different measured field profiles.')
            device_record['reference_source_relative_difference']=difference
            if not all(np.array_equal(v['frequency_per_um'],model['frequency_per_um']) for v in (ref,dev)):
                raise ValueError('Monitor frequencies differ from the requested bins.')
            columns.append((ref['amplitudes'],dev['amplitudes'],reference,device_record))
    matrix,incident,through,valid = normalize(model,columns)
    np.savez_compressed(output/'arrays.npz',s=matrix,incident=incident,reference_through=through,
                        valid_column=valid,frequency_per_um=model['frequency_per_um'],wavelength_um=1/np.array(model['frequency_per_um']))
    power = abs(matrix)**2
    if not np.isfinite(matrix).all():
        raise RuntimeError('One or more measured S coefficients are nonfinite.')
    grid_shape,_,_,planes = grid(model,resolution)
    expected_device = next(iter(stages.values()))['field_device']
    if not all(s['field_device']==expected_device for s in stages.values()):
        raise ValueError('The stages were measured on different devices.')
    maximum_reciprocity = float(np.max(abs(matrix-np.swapaxes(matrix,1,2))))
    singular_values = np.linalg.svd(matrix,compute_uv=False)[:,0]
    straight_error = float(np.max(abs(abs(through)**2-1)))
    under_test = {'complete_settling':bool(np.all(valid)),
                  'straight_reference_power_within_0_03':straight_error<=.03,
                  'max_complex_reciprocity_difference_below_0_05':maximum_reciprocity<=.05,
                  'max_singular_value_squared_at_most_1_05':bool(np.all(singular_values**2<=1.05))}
    left_right_mirror = [1,0,3,2] if model['case']=='coupler' else [0,2,1]
    mirrored_power = power[:,left_right_mirror][:,:,left_right_mirror]
    mask = raster(model,resolution)
    return {'study':'finite-height multiport fundamental quasi-TE','solver':'fdtdx','case':model['case'],'requested_device':device,
            'actual_device':expected_device,'frequency_per_um':model['frequency_per_um'],
            'wavelength_um':[1/f for f in model['frequency_per_um']],'port_order':[p['name'] for p in model['ports']],
            'matrix_axes':['frequency','outgoing_port','incoming_port'],'complex_encoding':['real','imag'],
            's':complex_components(matrix),'modal_power':power.tolist(),'sum_observed_power':power.sum(axis=1).tolist(),
            'max_reciprocity_complex_difference':maximum_reciprocity,
            'max_modal_power_excess':max(0.,float(power.sum(axis=1).max()-1)),
            'mirror_symmetry_max_guided_power_delta':float(np.max(abs(power-mirrored_power))),
            'scalar_mask_mirror_disagreeing_voxels':int(np.count_nonzero(mask!=mask[:,::-1,:])),
            'maximum_singular_value':singular_values.tolist(),'sampled_setup_checks':under_test,
            'sampled_setup_checks_passed':all(under_test.values()),
            'reference_through_power':(abs(through)**2).tolist(),'valid_column':valid.tolist(),'settled':bool(np.all(valid)),
            'physical_model':model,'resolution':resolution,'grid_shape':grid_shape,'sampled_ports':planes,
            'after_source_cap_um_over_c':after_time,'stages':stages,
            'phase':'exp(-i omega t), common forward Ey-positive power basis, matched reference backwave subtracted only at launched port',
            'scope':'The named straight-guide fundamental modes are observed at finite apertures. Radiation and other modes are not included.'}


def build_plan(model,device,resolution,after_time,target,material_model):
    return {'schema_version':1,'model':model,'device':device,'actual_device':str(target),'device_name':target.device_kind,
            'versions':{p:importlib.metadata.version(p) for p in ('fdtdx','jax','jaxlib','numpy','tidy3d')},
            'resolution':resolution,'after_source_cap_um_over_c':after_time,'numerical_sha256':numerical_digest(),
            'material_model':material_model}


def reference_cache_key(plan,launch):
    """Identical reference only: include the actual material, native grids, source, monitors and target."""
    model,resolution=plan['model'],plan['resolution']
    shape,pml,_,ports=grid(model,resolution)
    ref={'shape':shape,'pml_cells':pml,'port':ports[launch], 'material_model':plan.get('material_model','legacy-x'),'occupancy_u8_sha256':digest(raster(model,resolution,launch)),
         **{key:model[key] for key in ('core_index','cladding_index','frequency_per_um','pulse_fwidth','pulse_cutoff')},
         **{key:plan[key] for key in ('device','actual_device','device_name','resolution','after_source_cap_um_over_c',
                                       'numerical_sha256','versions')}}
    if ref['material_model'] == 'polygon-area':
        ref['straight_polygon_um'] = straight_polygon(model,launch)
        ref['physical_height_um'] = model['height']
        if 'z_grid_origin_um' in model:
            ref['z_grid_origin_um'] = model['z_grid_origin_um']
    return hashlib.sha256(json.dumps(ref,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_stage(record,archive):
    details=json.loads(record.read_text())
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest()!=details.get('archive_sha256'):
        raise ValueError(f'The completed stage archive for {record.stem} is missing or changed.')
    return details


def save_record(path,record):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def compute_stage(runtime,plan,launch,reference,archive,bases):
    start=time.perf_counter()
    _,details=stage(runtime,plan['model'],plan['resolution'],plan['after_source_cap_um_over_c'],launch,reference,archive,bases,plan['material_model'])
    return {**details,'seconds':time.perf_counter()-start,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}


def device_cache_key(plan, launch):
    """Only the identical physical circuit, source port, solver, target and numerical settings can share a device stage."""
    import platform
    port = plan['model']['ports'][launch]['name']
    architecture = {'machine':platform.machine()}
    cpu_info = Path('/proc/cpuinfo')
    if cpu_info.is_file():
        fields = dict(line.split(':',1) for line in cpu_info.read_text().split('\n\n',1)[0].splitlines() if ':' in line)
        architecture.update({name.strip():value.strip() for name,value in fields.items()
                             if name.strip() in {'vendor_id','cpu family','model','model name','stepping','flags','Features','CPU implementer','CPU part'}})
    payload = {'kind':'device', 'launch':port, 'plan':plan, 'host_architecture':architecture}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def measured_stage(runtime,plan,launch,reference,archive,bases,cache,device_cache=None):
    cache = cache if reference else device_cache
    if cache is None:
        return compute_stage(runtime,plan,launch,reference,archive,bases)
    import fcntl
    cache.mkdir(parents=True,exist_ok=True)
    kind = 'reference' if reference else 'device'
    key = reference_cache_key(plan,launch) if reference else device_cache_key(plan,launch)
    key_field, reuse_field = kind+'_cache_key', 'reuse_existing_'+kind
    report,stored=cache/(key+'.json'),cache/(key+'.npz')
    with (cache/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if report.exists():
            entry=load_stage(report,stored)
            if entry.get(key_field)!=key or entry['field_device']!=plan['actual_device'] or not entry['settled']:
                raise ValueError('The cached stage is unqualified or belongs to another physical study or device.')
            if not reference and (entry.get('cache_launch_port') != plan['model']['ports'][launch]['name']
                    or entry.get('material_model') != plan['material_model']
                    or entry.get('monitor_names') != [p['name'] for p in plan['model']['ports']]):
                raise ValueError('The cached device stage has different ports or material sampling.')
            shutil.copyfile(stored,archive)
            return {**entry,reuse_field:True}
        details=compute_stage(runtime,plan,launch,reference,archive,bases)
        details.update({key_field:key, reuse_field:False})
        if not reference:
            details['cache_launch_port'] = plan['model']['ports'][launch]['name']
        if details['settled']:
            shutil.copyfile(archive,stored)
            save_record(report,details)
        return details


def resolved_material_model(requested, saved_plan=None):
    """New studies use the qualified grid; an unspecified resume keeps its recorded physical representation."""
    if requested is None:
        requested = DEFAULT_MATERIAL_MODEL if saved_plan is None else saved_plan.get('material_model', 'legacy-x')
    if requested not in ('legacy-x', 'yee-aligned', 'polygon-area'):
        raise ValueError('Choose legacy-x, yee-aligned or polygon-area material sampling.')
    return requested


def run(case,device,resolution,output,pml=.75,aperture=2.,after_time=80.,resume=False,reference_cache=None,material_model=None,device_cache=None):
    import fcntl
    if resolution not in (12,16,20,24) or not 24<=after_time<=160:
        raise ValueError('Use a supported resolution (12/16/20/24) and an after-source time between 24 and 160 µm/c.')
    if material_model is not None:
        resolved_material_model(material_model)
    if resume and not output.is_dir():
        raise ValueError('Resume requires an existing study folder.')
    model=multiport_model(case,pml,aperture)
    runtime=_ports_runtime(device)
    jax,_,_,target=runtime
    output.mkdir(parents=True,exist_ok=resume)
    stages,bases={},{}
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=output/'run.json'
        saved=read_locked_plan(manifest,'.run.lock','run.pending-json') if resume else None
        saved_plan = json.loads(saved) if saved is not None else None
        if saved is not None and not isinstance(saved_plan, dict):
            raise ValueError('Cannot resume a study with an invalid manifest.')
        material_model = resolved_material_model(material_model, saved_plan)
        plan=build_plan(model,device,resolution,after_time,target,material_model)
        encoded=json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False)
        if saved is not None and saved!=encoded:
            raise ValueError('Cannot resume a different model, worker version, numerical source or target.')
        if saved is None:
            pending=manifest.with_suffix('.pending-json')
            pending.write_text(encoded+'\n')
            pending.replace(manifest)
        for i,port in enumerate(model['ports']):
            for kind in ('reference','device'):
                key=f"{port['name']}-{kind}"
                archive,record=output/(key+'.npz'),output/(key+'.json')
                if resume and record.exists():
                    details=load_stage(record,archive)
                    origin='reused_verified_receipt'
                else:
                    print(f'{case} {device} {key}: calibrating',flush=True)
                    details=measured_stage(runtime,plan,i,kind=='reference',archive,bases,reference_cache,device_cache)
                    save_record(record,details)
                    origin='reused_verified_receipt' if details.get('reuse_existing_'+kind) else 'computed_this_invocation'
                    print(key,details['settled'],details['history'][-1],f"{details['seconds']:.1f}s",flush=True)
                    jax.clear_caches()
                stages[key]={**details,'invocation_acquisition':origin}
        report=assemble(model,device,resolution,after_time,output,stages)
        report.update(material_model=material_model,
                      material_sampling={'yee-aligned':'Adjacent-cell tangential averaging on both transverse axes at each electric component.',
                          'legacy-x':'Legacy adjacent-x averaging of tangential components only.',
                          'polygon-area':'Exact polygon and height coverage on each Yee dual cell; diagonal interface-normal mixing.'}[material_model],
                      device_name=target.device_kind,versions=plan['versions'],numerical_sha256=plan['numerical_sha256'],
                      stage_seconds=sum(s['seconds'] for s in stages.values()))
        save_record(output/'report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('coupler','splitter'),required=True)
    parser.add_argument('--device',choices=('cpu','gpu'),required=True)
    parser.add_argument('--resolution',choices=(12,16,20,24),type=int,default=16)
    parser.add_argument('--pml',type=float,default=.75)
    parser.add_argument('--aperture',type=float,default=2.)
    parser.add_argument('--after-time',type=float,default=80.,help='after-source time cap in µm/c')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='verify and reuse completed stages from this exact study folder')
    parser.add_argument('--reference-cache',type=Path,help='optional cache of exactly matching measured straight references')
    parser.add_argument('--device-cache',type=Path,help='coordinate native device stages from identical study plans and input ports')
    parser.add_argument('--material-model',choices=('legacy-x','yee-aligned','polygon-area'),
                        help='fresh runs use yee-aligned; omit on resume to keep the recorded material grid')
    args=parser.parse_args()
    report=run(args.case,args.device,args.resolution,args.output,args.pml,args.aperture,args.after_time,args.resume,args.reference_cache,args.material_model,args.device_cache)
    if not report['sampled_setup_checks_passed']:
        raise SystemExit('The sampled setup missed a numerical check; inspect report.json before using the matrix.')


if __name__=='__main__':
    main()
