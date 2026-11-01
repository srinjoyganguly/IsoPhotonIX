"""Finite-height FDTDX port modes and complex two-port scattering matrices.

FDTDX 0.6.2 provides the actual Yee evolution, local mode solver and six-field
phasor monitors. This adapter performs the documented reciprocal E/H overlap
explicitly because the installed ModeOverlapDetector supports one frequency.
"""
from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path
import time

from .fdtdx_ports_adapter import _ports_runtime, measured_energy_tail
from .fdtdx3d_adapter import cell_center_fields


def canonical_source_profile(electric, magnetic):
    """Fix the arbitrary eigenvector phase before real-valued TFSF injection."""
    import numpy as np
    e,h=np.asarray(electric),np.asarray(magnetic)
    anchor=e[1,0,e.shape[2]//2,e.shape[3]//2]
    if not np.isfinite(anchor) or abs(anchor)<1e-9:
        raise RuntimeError('The fundamental source-mode Ey phase anchor is unavailable.')
    phase=np.exp(-1j*np.angle(anchor))
    return np.real(e*phase).copy(),np.real(h*phase).copy(),phase


def require_quasi_te_mode(electric):
    """Verify physical Ey polarization; the native TE filter only sorts modes."""
    import numpy as np
    e=np.asarray(electric)
    if e.ndim!=4 or e.shape[0]!=3 or not np.isfinite(e).all():
        raise RuntimeError('Expected a finite three-component local electric mode.')
    transverse_power=float(np.sum(abs(e[1:])**2))
    if transverse_power<=0:
        raise RuntimeError('The local mode has zero transverse electric power.')
    fraction=float(np.sum(abs(e[1])**2)/transverse_power)
    if fraction<.5:
        raise RuntimeError('The local mode is not predominantly Ey (quasi-TE).')
    return fraction


def longitudinal_yee_permittivity(voxels):
    """Place gap interfaces at the same physical x for all electric components.

    Input scalar voxels live at x=(i+1/2)dx, as does Ex. Ey/Ez live at x=i dx;
    their interface-tangential permittivity is the adjacent-cell arithmetic
    average. Uniform-x guide/source cross-sections remain exactly unchanged.
    This bounded translation is not a general 3D subpixel smoothing algorithm.
    """
    import numpy as np
    eps=np.asarray(voxels)
    if eps.ndim!=3 or not np.isfinite(eps).all() or np.any(eps<1):
        raise ValueError('Expected finite positive dielectric voxels.')
    previous=np.concatenate([eps[:1],eps[:-1]],axis=0)
    transverse=(eps+previous)/2
    return np.stack([eps,transverse,transverse])


def collocate_mode(electric, magnetic, beta_dx, omega_dt):
    """Match FDTDX exact-interpolation phasors at the Ez Yee point.

    Modes are native Yee cross-section samples [3,1,y,z]. Spatial averaging
    in x gives cos(beta*dx/2); H's adjacent-time average gives cos(omega*dt/2).
    Transverse boundary continuation is zero, as in the finite mode aperture.
    """
    import numpy as np
    e, h = np.asarray(electric)[:, 0], np.asarray(magnetic)[:, 0]
    if e.shape != h.shape or e.ndim != 3 or e.shape[0] != 3:
        raise ValueError('Expected matching three-component Yee mode planes.')
    def yz_average(a):
        p = np.pad(a, ((1, 1), (1, 1)))
        return (p[1:-1, 1:-1]+p[:-2, 1:-1]+p[1:-1, 2:]+p[:-2, 2:])/4
    def y_average(a):
        p = np.pad(a, ((1, 1), (0, 0)))
        return (p[1:-1]+p[:-2])/2
    def z_average(a):
        p = np.pad(a, ((0, 0), (1, 1)))
        return (p[:, 1:-1]+p[:, 2:])/2
    x_factor, t_factor = np.cos(beta_dx/2), np.cos(omega_dt/2)
    ec = np.stack([z_average(e[0])*x_factor, yz_average(e[1]), e[2]])
    hc = np.stack([y_average(h[0]), h[1]*x_factor, yz_average(h[2])*x_factor])*t_factor
    return ec, hc


def modal_amplitudes(phasors, mode_e, mode_h, cell_area_um2):
    """Power-normalized forward/backward amplitudes from reciprocal E/H overlap.

    All inputs use colocated transverse fields and the exp(-i omega t) physical
    convention. The common lossless-mode transverse electric gauge is positive
    Ey at the core center, shared by both ports and propagation directions.
    """
    import numpy as np
    p = np.asarray(phasors)
    if p.ndim == 5:
        p = p[:, :, 0]
    e, h = np.asarray(mode_e), np.asarray(mode_h)
    if p.ndim != 4 or p.shape[1] != 6 or e.shape != h.shape or e.shape != (p.shape[0], 3, *p.shape[2:]):
        raise ValueError('Phasor and modal planes must agree in frequency and transverse grid.')
    cross_eh = e[:, 1]*np.conj(h[:, 2])-e[:, 2]*np.conj(h[:, 1])
    power = .5*np.real(cross_eh.sum(axis=(1, 2)))*cell_area_um2
    if not np.isfinite(power).all() or np.any(power <= 0):
        raise ValueError('Forward mode must carry finite positive +x power.')
    first = p[:, 1]*np.conj(h[:, 2])-p[:, 2]*np.conj(h[:, 1])
    second = np.conj(e[:, 1])*p[:, 5]-np.conj(e[:, 2])*p[:, 4]
    plus = (first+second).sum(axis=(1, 2))*cell_area_um2/(4*np.sqrt(power))
    minus = (first-second).sum(axis=(1, 2))*cell_area_um2/(4*np.sqrt(power))
    return np.stack([plus, minus], axis=1), power


def run(config: dict, output_dir: Path) -> dict:
    import numpy as np

    if config.get('study_type') != 'waveguide-ports-3d' or config.get('schema_version') != 4:
        raise ValueError('Expected a schema-version-4 finite-height port study.')
    started = time.perf_counter()
    jax, jnp, fdtdx, target = _ports_runtime(config['device'])
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    dx = 1/config['resolution']
    domain = [config['geometry']['domain'][axis] for axis in 'xyz']
    shape = tuple(round(value/dx) for value in domain)
    nx, ny, nz = shape
    pml = round(config['pml_um']/dx)
    aperture = (ny-2*pml, nz-2*pml)
    objects_by_id = {item['id']: item for item in config['geometry']['objects']}
    core = objects_by_id['core']
    core_shape = (nx, round(core['size']['y']/dx), round(core['size']['z']/dx))
    core_start = (0, (ny-core_shape[1])//2, (nz-core_shape[2])//2)
    frequencies = np.linspace(1/config['wavelength_max_um'], 1/config['wavelength_min_um'], config['frequency_points'])
    frequencies_hz = frequencies*299792458e6
    center_hz, span_hz = float(np.mean(frequencies_hz)), float(np.ptp(frequencies_hz))
    sigma_frequency, sigma_time = span_hz/math.pi, 1/(2*span_hz)
    source_end_fs = 12*sigma_time*1e15
    total_fs = source_end_fs+config['max_after_sources_fs']
    source_left = -domain[0]/2+config['pml_um']+.5
    port_left, port_right = source_left+1, -source_left-1
    plane_index = {name: round((coordinate+nx*dx/2)/dx) for name, coordinate in
        [('source_forward', source_left), ('source_reverse', -source_left), ('left', port_left), ('right', port_right)]}
    waves = tuple(fdtdx.WaveCharacter(frequency=float(f)) for f in frequencies_hz)
    profile = fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(frequency=sigma_frequency),
        center_wave=fdtdx.WaveCharacter(frequency=center_hz))
    versions = {name: importlib.metadata.version(name) for name in ('fdtdx', 'jax', 'jaxlib', 'numpy', 'tidy3d')}
    stages, results, mode_record, captured = {}, {}, {}, {}
    basis_e = basis_h = neffs = None

    def stage_run(stage_name, reverse, gap_um):
        nonlocal basis_e, basis_h, neffs
        stage_started = time.perf_counter()
        print(f'PORT_STAGE={stage_name}', flush=True)
        simulation = fdtdx.SimulationConfig(time=total_fs*1e-15, resolution=dx*1e-6,
            backend=config['device'], dtype=jnp.float32, courant_factor=.99)
        if simulation.backend != config['device']:
            raise RuntimeError('Execution target changed; CPU fallback is disabled.')
        volume = fdtdx.SimulationVolume(name='domain', partial_grid_shape=shape,
            material=fdtdx.Material(permittivity=config['cladding_index']**2))
        bounds, constraints = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig(
            **{f'boundary_type_{side}{axis}':'pml' for side in ('min','max') for axis in 'xyz'},
            **{f'thickness_grid_{side}{axis}':pml for side in ('min','max') for axis in 'xyz'}), volume)
        constraints = list(constraints); items = [volume, *bounds.values()]
        expected = np.full(shape, config['cladding_index']**2, dtype=np.float32)
        def material(name, grid_shape, grid_start, index, order):
            item = fdtdx.UniformMaterialObject(name=name, partial_grid_shape=grid_shape,
                material=fdtdx.Material(permittivity=index**2), placement_order=order)
            constraints.append(item.set_grid_coordinates(axes=(0,1,2), sides=('-','-','-'), coordinates=grid_start))
            items.append(item)
            expected[tuple(slice(a,a+b) for a,b in zip(grid_start,grid_shape))] = index**2
        if 'substrate' in objects_by_id:
            material('substrate', (nx,ny,core_start[2]), (0,0,0), objects_by_id['substrate']['refractive_index'], 0)
        material('core', core_shape, core_start, core['refractive_index'], 1)
        gap_cells = round(gap_um/dx)
        if gap_cells:
            material('gap', (gap_cells,*core_shape[1:]), ((nx-gap_cells)//2,*core_start[1:]), config['cladding_index'], 2)
        source = fdtdx.ModePlaneSource(name='source', partial_grid_shape=(1,*aperture),
            mode_index=0, filter_pol='te', direction='-' if reverse else '+',
            wave_character=fdtdx.WaveCharacter(frequency=center_hz), temporal_profile=profile,
            switch=fdtdx.OnOffSwitch(end_time=source_end_fs*1e-15))
        constraints.append(source.set_grid_coordinates(axes=(0,1,2), sides=('-','-','-'),
            coordinates=(plane_index['source_reverse' if reverse else 'source_forward'],pml,pml)))
        items.append(source)
        for name in ('left','right'):
            detector = fdtdx.PhasorDetector(name=name, partial_grid_shape=(1,*aperture),
                wave_characters=waves, scaling_mode='pulse', exact_interpolation=True)
            constraints.append(detector.set_grid_coordinates(axes=(0,1,2), sides=('-','-','-'), coordinates=(plane_index[name],pml,pml)))
            items.append(detector)
        energy = fdtdx.EnergyDetector(name='energy', partial_grid_shape=tuple(v-2*pml for v in shape),
            reduce_volume=True, exact_interpolation=True, switch=fdtdx.OnOffSwitch(interval=max(1,round(1e-15/simulation.time_step_duration))), plot=False)
        constraints.append(energy.set_grid_coordinates(axes=(0,1,2), sides=('-','-','-'), coordinates=(pml,pml,pml)))
        items.append(energy)
        key = jax.random.PRNGKey(0)
        print(f'{stage_name}: local CPU source-mode preparation; full 3D Maxwell evolution on {target}', flush=True)
        objects, arrays, params, simulation, _ = fdtdx.place_objects(object_list=items, config=simulation, constraints=constraints, key=key)
        initial_epsilon=np.asarray(1/arrays.inv_permittivities[0])
        if not np.allclose(initial_epsilon,expected,rtol=1e-6,atol=1e-6):
            raise RuntimeError('Initial material voxels differ from the explicit geometry.')
        component_epsilon=longitudinal_yee_permittivity(expected)
        arrays=arrays.aset('inv_permittivities',jax.device_put(1/component_epsilon,target))
        arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
        # FDTDX 0.6.2 takes real(mode_E/H) without fixing its arbitrary gauge.
        # A backward eigensolve can return a purely imaginary transverse mode,
        # which otherwise injects zero. Keep its native direction/time offsets
        # and canonicalize the actual eigenvector before the real TFSF profile.
        from fdtdx.core.physics.modes import compute_mode
        prepared=objects['source']
        source_mode_e,source_mode_h,_=compute_mode(frequency=center_hz,
            inv_permittivities=prepared._inv_permittivity,inv_permeabilities=prepared._inv_permeability,
            resolution=dx*1e-6,direction='-' if reverse else '+',mode_index=0,filter_pol='te',dtype=jnp.float32)
        canonical_e,canonical_h,source_gauge=canonical_source_profile(source_mode_e,source_mode_h)
        prepared=prepared.aset('_E',jax.device_put(canonical_e,target)).aset('_H',jax.device_put(canonical_h,target))
        objects=objects.aset('object_list',[prepared if item.name=='source' else item for item in objects.object_list])
        jax.block_until_ready(arrays)
        if arrays.fields.E.device != target:
            raise RuntimeError('Initial electromagnetic fields are not on the requested execution target.')
        epsilon = np.asarray(1/arrays.inv_permittivities[0])
        if not np.allclose(np.asarray(1/arrays.inv_permittivities), component_epsilon, rtol=1e-6, atol=1e-6):
            raise RuntimeError('Material voxels differ from the requested supported geometry translation.')
        source_e = np.asarray(objects['source']._E)
        source_h = np.asarray(objects['source']._H)
        source_neff = complex(np.asarray(objects['source']._neff))
        if not np.isfinite(source_e).all() or not np.isfinite(source_h).all() or not np.any(source_e[1:]):
            raise RuntimeError('Source has no finite nonzero transverse electric profile.')
        source_fraction = require_quasi_te_mode(source_e)
        highest_cladding = max(config['cladding_index'],objects_by_id.get('substrate',{}).get('refractive_index',1))
        if source_fraction < .5 or not highest_cladding < source_neff.real < core['refractive_index']:
            raise RuntimeError('Source mode is not a guided fundamental quasi-TE mode.')
        if basis_e is None:
            print('Preparing independent local modes at every measured frequency', flush=True)
            from fdtdx.core.physics.modes import compute_mode
            ep = arrays.inv_permittivities[:,plane_index['left']:plane_index['left']+1,pml:ny-pml,pml:nz-pml]
            es, hs, ns, gauges, edge_fractions, ey_fractions = [], [], [], [], [], []
            raw_es, raw_hs = [], []
            for f, f_hz in zip(frequencies, frequencies_hz):
                e, h, n = compute_mode(frequency=float(f_hz), inv_permittivities=ep,
                    inv_permeabilities=1., resolution=dx*1e-6, direction='+', mode_index=0, filter_pol='te', dtype=jnp.float32)
                e,h,n = np.asarray(e),np.asarray(h),complex(np.asarray(n))
                ey_fractions.append(require_quasi_te_mode(e))
                anchor = e[1,0,aperture[0]//2,aperture[1]//2]
                if not np.isfinite(anchor) or abs(anchor) < 1e-9:
                    raise RuntimeError('Fundamental Ey mode phase anchor is unavailable.')
                gauge = np.exp(-1j*np.angle(anchor)); e=e*gauge;h=h*gauge
                transverse = np.concatenate([e[1:].ravel(),h[1:].ravel()])
                imaginary_fraction = np.linalg.norm(transverse.imag)/np.linalg.norm(transverse)
                if imaginary_fraction > 1e-4 or not highest_cladding < n.real < core['refractive_index'] or abs(n.imag)>1e-5:
                    raise RuntimeError('The measured-frequency mode is not a supported lossless guided transverse basis.')
                ec,hc = collocate_mode(e,h,2*np.pi*f*n.real*dx,2*np.pi*f_hz*simulation.time_step_duration)
                intensity = np.sum(abs(e[:,0])**2,axis=0)
                edge = intensity.copy();edge[2:-2,2:-2]=0
                edge_fractions.append(float(edge.sum()/intensity.sum()))
                es.append(ec);hs.append(hc);ns.append(n);gauges.append(gauge)
                raw_es.append(e);raw_hs.append(h)
            basis_e,basis_h,neffs = np.asarray(es),np.asarray(hs),np.asarray(ns)
            mode_record.update(neff_real=neffs.real.tolist(),neff_imag=neffs.imag.tolist(),
                ey_transverse_fraction=ey_fractions, minimum_ey_transverse_fraction=.5,
                edge_electric_l2_fraction=edge_fractions, gauge='Ey at the center sample is real and positive; common +x basis at both ports')
            np.savez_compressed(output/'port-modes.npz', frequency_per_um=frequencies,
                raw_mode_e=np.asarray(raw_es),raw_mode_h=np.asarray(raw_hs),mode_e=basis_e,mode_h=basis_h,
                mode_neff=neffs,gauge_factors=np.asarray(gauges),ey_transverse_fraction=np.asarray(ey_fractions),
                y_um=np.arange(pml,ny-pml)*dx-ny*dx/2,z_um=(np.arange(pml,nz-pml)+.5)*dx-nz*dx/2)
        from fdtdx.fdtd.forward import forward
        def evolve(state,end_step):
            return jax.lax.while_loop(lambda s:s[0]<end_step,
                lambda s:forward(state=s,config=simulation,objects=objects,key=key,
                    record_detectors=True,record_boundaries=False,simulate_boundaries=True),state)
        state=(jnp.asarray(0,dtype=jnp.int32),arrays)
        first=min(simulation.time_steps_total,math.ceil((source_end_fs+40.)*1e-15/simulation.time_step_duration))
        compile_start=time.perf_counter()
        compiled=jax.jit(evolve).lower(state,jnp.asarray(first,dtype=jnp.int32)).compile()
        compile_s=time.perf_counter()-compile_start
        schedule=np.flatnonzero(np.asarray(objects['energy']._is_on_at_time_step_arr))
        previous=None;history=[];settled=False;end=first
        compute_start=time.perf_counter()
        while True:
            state=compiled(state,jnp.asarray(end,dtype=jnp.int32));step,final=state
            jax.block_until_ready(final);step=int(step)
            phasors={name:np.asarray(final.detector_states[name]['phasor'])[0] for name in ('left','right')}
            if not all(np.isfinite(v).all() for v in phasors.values()):
                raise RuntimeError('Recorded complex port fields are nonfinite.')
            amplitudes={name:modal_amplitudes(p,basis_e,basis_h,dx**2)[0] for name,p in phasors.items()}
            energies,times,ratio=measured_energy_tail(np.asarray(final.detector_states['energy']['energy'])[:,0],
                schedule,step,simulation.time_step_duration*1e15,40.)
            spectral_change=None
            if previous is not None:
                scale=max(float(np.max(abs(v))) for v in amplitudes.values())
                spectral_change=max(float(np.max(abs(amplitudes[name]-previous[name]))) for name in amplitudes)/scale if scale>0 else None
                settled=bool(ratio<=config['decay_by'] and spectral_change is not None and spectral_change<=config['spectrum_tolerance'])
            history.append({'time_fs':step*simulation.time_step_duration*1e15,'energy_ratio':ratio,
                'spectral_relative_change':spectral_change,'settled':settled})
            print(f'{stage_name}: {history[-1]}',flush=True)
            if settled or step>=simulation.time_steps_total:break
            previous={name:value.copy() for name,value in amplitudes.items()}
            end=min(simulation.time_steps_total,step+math.ceil(40e-15/simulation.time_step_duration))
        if final.fields.E.device!=target:
            raise RuntimeError('Final fields are not on the requested execution target; fallback refused.')
        elapsed=step*simulation.time_step_duration*1e15
        stage={'settled':settled,'energy_ratio':ratio,'spectral_relative_change':spectral_change,
            'thresholds':{'energy_decay':config['decay_by'],'spectral_relative_change':config['spectrum_tolerance']},
            'decay_by':config['decay_by'],'spectrum_tolerance':config['spectrum_tolerance'],
            'elapsed_fs':elapsed,'actual_duration_fs':elapsed,'source_effective_end_fs':source_end_fs,
            'time_steps':step,'reached_time_cap':step>=simulation.time_steps_total,'history':history,
            'actual_device':str(target),'source_mode_neff_real':source_neff.real,
            'source_eigenmode_gauge_factor':[float(source_gauge.real),float(source_gauge.imag)],
            'source_ey_transverse_fraction':source_fraction,'compile_seconds':compile_s,
            'compute_seconds':time.perf_counter()-compute_start,'total_seconds':time.perf_counter()-stage_started,
            'settlement_definition':'Maximum sampled non-PML energy over final40fs / peak overrun; complex mode-amplitude change between40fs checks / peak current mode amplitude.'}
        np.savez_compressed(output/(stage_name.replace('_','-')+'-monitors.npz'),
            frequency_per_um=frequencies,left_phasors=phasors['left'],right_phasors=phasors['right'],
            left_amplitudes=amplitudes['left'],right_amplitudes=amplitudes['right'],
            source_e=source_e,source_h=source_h,energy=energies,energy_times_fs=times)
        if stage_name=='device_forward':
            raw=np.asarray(final.fields.E).transpose(0,3,2,1).copy()
            fields=cell_center_fields(raw);eps=epsilon.transpose(2,1,0)
            epsilon_components=component_epsilon.transpose(0,3,2,1)[:,:-1,:-1,:-1]
            axes={f'{axis}_um':(np.arange(count-1)+.5)*dx-count*dx/2 for axis,count in zip('xyz',shape)}
            np.savez_compressed(output/'arrays.npz',fields=fields,field=fields[1],epsilon=epsilon_components[1],epsilon_components=epsilon_components,
                raw_fields=raw,raw_epsilon=eps,raw_permittivity_components=component_epsilon.transpose(0,3,2,1),**axes)
            captured.update(field_time_fs=elapsed,field_shape=list(fields.shape[1:]))
        return amplitudes,stage,source_e,source_h

    gap=config['gap_um'] if config['device_kind']=='gap' else 0.
    for reverse,direction in ((False,'forward'),(True,'reverse')):
        reference_e=reference_h=None
        for kind,stage_gap in (('reference',0.),('device',gap)):
            name=f'{kind}_{direction}'
            amplitudes,stage,source_e,source_h=stage_run(name,reverse,stage_gap)
            if kind=='reference':reference_e,reference_h=source_e,source_h
            else:
                difference=max(float(np.linalg.norm(source_e-reference_e)/np.linalg.norm(reference_e)),
                    float(np.linalg.norm(source_h-reference_h)/np.linalg.norm(reference_h)))
                if difference>1e-5:raise RuntimeError('Reference and device injected mode profiles differ.')
                stage['reference_source_relative_difference']=difference
            stages[name]=stage;results[name]=amplitudes
    print('PORT_STAGE=normalization',flush=True)
    incident=np.stack([results['reference_forward']['left'][:,0],results['reference_reverse']['right'][:,1]],axis=1)
    outgoing=np.empty((len(frequencies),2,2),dtype=np.complex128)
    outgoing[:,0,0]=results['device_forward']['left'][:,1]-results['reference_forward']['left'][:,1]
    outgoing[:,1,0]=results['device_forward']['right'][:,0]
    outgoing[:,0,1]=results['device_reverse']['left'][:,1]
    outgoing[:,1,1]=results['device_reverse']['right'][:,0]-results['reference_reverse']['right'][:,0]
    reference_output=np.stack([results['reference_forward']['right'][:,0],results['reference_reverse']['left'][:,1]],axis=1)
    valid=np.all(abs(incident)**2>np.max(abs(incident)**2,axis=0)*1e-6,axis=1)
    valid &= all(stage['settled'] for stage in stages.values())
    if not np.isfinite(incident).all() or not np.isfinite(outgoing).all() or not np.all(np.max(abs(incident),axis=0)>0):
        raise RuntimeError('Incident or outgoing modal coefficients are invalid.')
    matrix=np.divide(outgoing,incident[:,None,:],out=np.full_like(outgoing,np.nan+1j*np.nan),where=abs(incident[:,None,:])>0)
    through=np.divide(reference_output,incident,out=np.full_like(reference_output,np.nan+1j*np.nan),where=abs(incident)>0)
    order=np.argsort(1/frequencies)
    np.savez_compressed(output/'sparameters.npz',wavelength_um=(1/frequencies)[order],frequency_per_um=frequencies[order],
        s=matrix[order],incident_amplitude=incident[order],outgoing_amplitude=outgoing[order],
        reference_through=through[order],valid_bin=valid[order],mode_neff=neffs[order],mode_valid=np.ones(len(frequencies),dtype=bool))
    return {'solver':'fdtdx','requested_device':config['device'],'actual_device':str(target),'device_name':target.device_kind,
        'versions':versions,'dimensions':'3D','field_component':'Ey','field_components':['Ex','Ey','Ez'],
        'field_units':'arbitrary source units','field_capture_stage':'device_forward',**captured,
        'field_sampling':'Final device-forward Yee electric fields averaged to common cell centers; raw staggered fields retained.',
        'material_sampling':'Ex uses voxel permittivity; Ey/Ez use adjacent-x-cell arithmetic average to place tangential gap interfaces at their actual Yee x locations. epsilon_components retains these actual coefficients cropped to the display grid (not spatially averaged with fields); raw_permittivity_components preserves all3component Yee coefficients. No transverse or general3D subpixel averaging.',
        'actual_duration_fs':captured['field_time_fs'],'settlement':stages,'sparameters_valid':bool(np.all(valid)),
        'spectra_valid':bool(np.all(valid)),'total_seconds':time.perf_counter()-started,
        'source':{'kind':'Local numerical fundamental quasi-TE ModePlaneSource; finite-band Gaussian pulse',
            'mode_preparation_device':'Local CPU callback; requested CPU/CUDA performs full3D time evolution',
            'gaussian_sigma_time_fs':sigma_time*1e15,'effective_end_fs':source_end_fs,
            'mode_index':0,'polarization':'Predominantly Ey transverse electric field'},
        'ports':{'numbering':'1=left,2=right','requested_x_um':[port_left,port_right],
            'actual_x_um':[plane_index[name]*dx-nx*dx/2 for name in ('left','right')],
            'port_1_x_um':plane_index['left']*dx-nx*dx/2,'port_2_x_um':plane_index['right']*dx-nx*dx/2,
            'phase_convention':'DFT exp(+i omega t), physical exp(-i omega t), forward exp(+i beta x)',
            'mode_gauge':mode_record['gauge'],
            'normalization':'Unit-power reciprocal E/H modal projection divided by separately measured incident coefficient; reflection reference subtracted; transmission propagation phase retained',
            'aperture_yz_um':[v*dx for v in aperture],'mode':mode_record},
        'normalization':{'definition':'S[frequency,output_port,input_port] = outgoing unit-power modal amplitude / separately measured incoming reference amplitude',
            'reflection':'Complex reference backward amplitude is subtracted before incident normalization',
            'transmission':'Actual device output / reference input; straight-guide propagation phase retained, no through correction or deembedding',
            'projection':'a+/-=[integral(E_sim cross H_mode* +/- E_mode* cross H_sim).x dA]/(4 sqrt(P_mode)); P_mode=.5 Re integral(E_mode cross H_mode*).x dA',
            'phase_convention':'DFT sum(field(t)*exp(+i omega t)); physical exp(-i omega t); forward propagation exp(+i beta x)',
            'mode_gauge':mode_record['gauge'],'sampling':'Both mode profiles and measured E/H use Ez Yee-point spatial interpolation; H includes adjacent-time average cos(omega*dt/2)'},
        'geometry_translation':{'grid_shape':list(shape),'grid_spacing_um':dx,'core_start':list(core_start),'core_shape':list(core_shape),
            'actual_gap_um':round(gap/dx)*dx,'pml_cells':pml,'requested_domain_um':domain,
            'actual_domain_um':[v*dx for v in shape],'material_grid_verified':True,
            'convention':'Dimensions rounded to cells; centered lower indices floored; source/port planes rounded to grid coordinates'},
        'boundaries':{'x':'PML','y':'PML','z':'PML','pml_cells':pml},
        'limitations':['Fundamental-mode submatrix only; radiation and higher-order modes are outside this2x2matrix.',
            'Lossless nondispersive straight or centeredgap finite-height guide; geometric and modal discretization error remains.',
            'Settling checks and CPU/GPU repeatability do not establish mesh/domain convergence or commercial accuracy.',
            'Absolute phase uses declared physical port planes and a shared electric-mode gauge; no arbitrary phase alignment is applied.']}
