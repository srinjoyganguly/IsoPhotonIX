"""Local FDTDX mode excitation and incident-normalized cross-section flux.

Uses FDTDX 0.6.2 ModePlaneSource and PhasorDetector. The local Tidy3D
eigensolver called by FDTDX requires no account, API key, or remote solve.
"""
from __future__ import annotations

import importlib.metadata
import math
import os
from pathlib import Path
import sys
import time


def _ports_runtime(device):
    """Allow the CPU callback required for local modes; never fall back for FDTD."""
    if device not in {'cpu', 'gpu'} or 'jax' in sys.modules:
        raise RuntimeError('An explicit CPU/GPU target and fresh worker are required.')
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu' if device == 'gpu' else 'cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    import jax
    import jax.numpy as jnp
    try:
        target = jax.devices('cuda' if device == 'gpu' else 'cpu')[0]
        jax.config.update('jax_default_device', target)
        probe = jax.jit(lambda x: x*x)(jax.device_put(jnp.arange(4), target))
        probe.block_until_ready()
        if probe.device != target or target.platform != ('gpu' if device == 'gpu' else 'cpu'):
            raise RuntimeError('Actual probe placement differs from the requested device.')
    except Exception as error:
        raise RuntimeError(f'Requested {device} initialization failed; CPU fallback is disabled: {error}') from error
    import fdtdx
    return jax, jnp, fdtdx, target


def spectral_flux(phasors, dx_um):
    """Positive-x power per invariant-z length, in common arbitrary units.

    Input is [frequency,component,x=1,y,z], components Ex/Ey/Ez/Hx/Hy/Hz.
    Phasors are synchronized by the solver's exact interpolation. The 1/2
    factor and integration weights are the same for reference and device.
    """
    import numpy as np
    p = np.asarray(phasors)
    s = .5*np.real(p[:, 1]*np.conj(p[:, 5])-p[:, 2]*np.conj(p[:, 4]))
    return np.mean(s[:, 0], axis=-1).sum(axis=-1)*dx_um


def normalized_power(incident, reflected_signed, transmitted, valid):
    """Return R/T only for valid bins; reflected input is signed +x flux."""
    import numpy as np
    incident = np.asarray(incident)
    valid = np.asarray(valid, dtype=bool)&np.isfinite(incident)&(incident > 0)
    reflectance = np.divide(-np.asarray(reflected_signed), incident,
        out=np.full(incident.shape, np.nan, dtype=np.float64), where=valid)
    transmittance = np.divide(np.asarray(transmitted), incident,
        out=np.full(incident.shape, np.nan, dtype=np.float64), where=valid)
    return reflectance, transmittance


def measured_energy_tail(recorded_energy, scheduled_steps, completed_steps, dt_fs, window_fs=50.):
    """Inspect only completed samples, never zero-filled future detector slots."""
    import numpy as np
    steps = np.asarray(scheduled_steps)
    steps = steps[steps < completed_steps]
    energies = np.asarray(recorded_energy)[:len(steps)]
    times = steps*dt_fs
    if not len(energies) or not np.isfinite(energies).all() or np.max(energies) <= 0:
        raise RuntimeError('The pulse has no finite nonzero measured electromagnetic energy.')
    ratio = float(np.max(energies[times >= times[-1]-window_fs])/np.max(energies))
    return energies, times, ratio


def run(config: dict, output_dir: Path) -> dict:
    import numpy as np

    started = time.perf_counter()
    requested = config['device']
    jax, jnp, fdtdx, target = _ports_runtime(requested)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    dx = 1/config['resolution']
    nx, ny = round(config['length_um']/dx), round(config['height_um']/dx)
    pml, core_cells = round(config['pml_um']/dx), round(config['core_width_um']/dx)
    aperture = ny-2*pml
    frequencies = np.linspace(1/config['wavelength_max_um'], 1/config['wavelength_min_um'], config['frequency_points'])
    frequencies_hz = frequencies*299792458e6
    center_hz, span_hz = float(np.mean(frequencies_hz)), float(np.ptp(frequencies_hz))
    # Gaussian sigma_t=1/(2*span), with the shared six-sigma cutoff at each end.
    sigma_frequency = span_hz/math.pi
    sigma_time = 1/(2*math.pi*sigma_frequency)
    source_end_fs = 12*sigma_time*1e15
    total_fs = source_end_fs+config['max_after_sources_fs']
    source_x = -config['length_um']/2+config['pml_um']+.5
    reflection_x, transmission_x = source_x+1, config['length_um']/2-config['pml_um']-1.5
    plane_indices = {name: round((coordinate+nx*dx/2)/dx) for name, coordinate in
                     [('source', source_x), ('reflection', reflection_x), ('transmission', transmission_x)]}
    wave = fdtdx.WaveCharacter(frequency=center_hz)
    profile = fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(frequency=sigma_frequency), center_wave=wave)
    waves = tuple(fdtdx.WaveCharacter(frequency=float(f)) for f in frequencies_hz)
    versions = {name: importlib.metadata.version(name) for name in ('fdtdx', 'jax', 'jaxlib', 'numpy', 'tidy3d')}
    stages = {}

    def execute_stage(name, gap_um):
        print(f'PORT_STAGE={name}', flush=True)
        stage_started = time.perf_counter()
        simulation = fdtdx.SimulationConfig(time=total_fs*1e-15, resolution=dx*1e-6,
            backend=requested, dtype=jnp.float32, courant_factor=.99)
        if simulation.backend != requested:
            raise RuntimeError('FDTDX changed the requested execution backend; CPU fallback is disabled.')
        volume = fdtdx.SimulationVolume(name='domain', partial_grid_shape=(nx, ny, 3),
            material=fdtdx.Material(permittivity=config['cladding_index']**2))
        core = fdtdx.UniformMaterialObject(name='core', partial_grid_shape=(nx, core_cells, 3),
            material=fdtdx.Material(permittivity=config['core_index']**2), placement_order=0)
        bounds, constraints = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig(
            boundary_type_minz='periodic', boundary_type_maxz='periodic',
            thickness_grid_minx=pml, thickness_grid_maxx=pml,
            thickness_grid_miny=pml, thickness_grid_maxy=pml,
            thickness_grid_minz=1, thickness_grid_maxz=1), volume)
        constraints = list(constraints)
        y0 = (ny-core_cells)//2
        constraints.append(core.set_grid_coordinates(axes=(0, 1, 2), sides=('-', '-', '-'), coordinates=(0, y0, 0)))
        items = [volume, core, *bounds.values()]
        gap_cells = round(gap_um/dx)
        if gap_cells:
            gap = fdtdx.UniformMaterialObject(name='gap', partial_grid_shape=(gap_cells, core_cells, 3),
                material=fdtdx.Material(permittivity=config['cladding_index']**2), placement_order=1)
            constraints.append(gap.set_grid_coordinates(axes=(0, 1, 2), sides=('-', '-', '-'), coordinates=((nx-gap_cells)//2, y0, 0)))
            items.append(gap)
        source = fdtdx.ModePlaneSource(name='mode_source', partial_grid_shape=(1, aperture, 3),
            mode_index=0, filter_pol='tm', direction='+', wave_character=wave, temporal_profile=profile,
            switch=fdtdx.OnOffSwitch(end_time=source_end_fs*1e-15))
        constraints.append(source.set_grid_coordinates(axes=(0, 1, 2), sides=('-', '-', '-'), coordinates=(plane_indices['source'], pml, 0)))
        items.append(source)
        for detector_name in ('reflection', 'transmission'):
            detector = fdtdx.PhasorDetector(name=detector_name, partial_grid_shape=(1, aperture, 3),
                wave_characters=waves, scaling_mode='pulse', exact_interpolation=True)
            constraints.append(detector.set_grid_coordinates(axes=(0, 1, 2), sides=('-', '-', '-'), coordinates=(plane_indices[detector_name], pml, 0)))
            items.append(detector)
        interval = max(1, round(1e-15/simulation.time_step_duration))
        energy = fdtdx.EnergyDetector(name='settling_energy', partial_grid_shape=(nx-2*pml, aperture, 3),
            reduce_volume=True, exact_interpolation=True, switch=fdtdx.OnOffSwitch(interval=interval), plot=False)
        constraints.append(energy.set_grid_coordinates(axes=(0, 1, 2), sides=('-', '-', '-'), coordinates=(pml, pml, 0)))
        items.append(energy)
        key = jax.random.PRNGKey(0)
        print(f'FDTDX ports {name}: local CPU mode preparation; FDTD execution target remains {target}', flush=True)
        objects, arrays, params, simulation, _ = fdtdx.place_objects(object_list=items,
            config=simulation, constraints=constraints, key=key)
        arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
        jax.block_until_ready(arrays)
        if arrays.fields.E.device != target:
            raise RuntimeError('FDTDX initial fields are not on the requested device.')
        placed_source = objects['mode_source']
        mode_e, mode_h = np.asarray(placed_source._E), np.asarray(placed_source._H)
        polarization = np.sum(mode_e.astype(np.float64)**2, axis=(1, 2, 3))/np.sum(mode_e.astype(np.float64)**2)
        neff = complex(np.asarray(placed_source._neff))
        print(f'FDTDX ports {name}: {target}; mode neff={neff.real:.8f}; Ez fraction={polarization[2]:.8f}', flush=True)
        if not np.isfinite(mode_e).all() or not np.isfinite(mode_h).all() or polarization[2] < .99:
            raise RuntimeError('Local mode solve did not select the required Ez-polarized slab mode.')
        invariant_error = float(np.linalg.norm(mode_e-mode_e.mean(axis=-1, keepdims=True))/np.linalg.norm(mode_e))
        if invariant_error > 1e-4:
            raise RuntimeError('Mode source varies along the supposedly invariant periodic z axis.')
        epsilon = np.asarray(1/arrays.inv_permittivities[0])
        expected = np.full((nx, ny, 3), config['cladding_index']**2, dtype=np.float32)
        expected[:, y0:y0+core_cells] = config['core_index']**2
        if gap_cells:
            expected[(nx-gap_cells)//2:(nx-gap_cells)//2+gap_cells, y0:y0+core_cells] = config['cladding_index']**2
        if not np.allclose(epsilon, expected, rtol=1e-6, atol=1e-6):
            raise RuntimeError('Initialized material grid differs from the declared waveguide/gap translation.')
        compile_started = time.perf_counter()
        # Continue the installed solver's unchanged Yee update without resetting
        # fields/DFTs between checks. run_fdtd resets state on each invocation.
        from fdtdx.fdtd.forward import forward
        def evolve(state, end_step):
            return jax.lax.while_loop(lambda s: s[0] < end_step,
                lambda s: forward(state=s, config=simulation, objects=objects, key=key,
                    record_detectors=True, record_boundaries=False, simulate_boundaries=True), state)
        state = (jnp.asarray(0, dtype=jnp.int32), arrays)
        first_check = min(simulation.time_steps_total, math.ceil((source_end_fs+50.)*1e-15/simulation.time_step_duration))
        compiled = jax.jit(evolve).lower(state, jnp.asarray(first_check, dtype=jnp.int32)).compile()
        compile_s = time.perf_counter()-compile_started
        print(f'FDTDX ports {name}: evolving up to {simulation.time_steps_total} steps; measured settling checks every 50 fs', flush=True)
        compute_started = time.perf_counter()
        previous_phasors = None
        history = []
        scheduled_steps = np.flatnonzero(np.asarray(objects['settling_energy']._is_on_at_time_step_arr))
        end_step = first_check
        settled = False
        while True:
            state = compiled(state, jnp.asarray(end_step, dtype=jnp.int32))
            final_step, final = state
            jax.block_until_ready(final)
            step = int(final_step)
            phasors = {d: np.asarray(final.detector_states[d]['phasor'])[0] for d in ('reflection', 'transmission')}
            if not all(np.isfinite(v).all() for v in phasors.values()):
                raise RuntimeError('A spectral field monitor returned nonfinite samples.')
            energies, times, ratio = measured_energy_tail(
                np.asarray(final.detector_states['settling_energy']['energy'])[:, 0],
                scheduled_steps, step, simulation.time_step_duration*1e15)
            spectral_change = float('inf')
            tail_phasors = {d: np.zeros_like(v) for d, v in phasors.items()}
            if previous_phasors is not None:
                spectral_changes = []
                for d in ('reflection', 'transmission'):
                    full_flux = spectral_flux(phasors[d].astype(np.complex128), dx)
                    previous_flux = spectral_flux(previous_phasors[d].astype(np.complex128), dx)
                    denominator = float(np.max(np.abs(full_flux)))
                    spectral_changes.append(float(np.max(abs(full_flux-previous_flux))/denominator) if denominator > 0 else float('inf'))
                    tail_phasors[d] = phasors[d]-previous_phasors[d]
                spectral_change = max(spectral_changes)
                settled = bool(ratio <= config['decay_by'] and np.isfinite(spectral_change) and spectral_change <= config['spectrum_tolerance'])
            history.append({'time_fs': step*simulation.time_step_duration*1e15,
                'energy_ratio': ratio, 'spectral_relative_change': spectral_change if np.isfinite(spectral_change) else None,
                'settled': settled})
            if settled or step >= simulation.time_steps_total:
                break
            previous_phasors = {d: v.copy() for d, v in phasors.items()}
            end_step = min(simulation.time_steps_total, step+math.ceil(50e-15/simulation.time_step_duration))
        if final.fields.E.device != target:
            raise RuntimeError('FDTDX final fields are not on the requested device; fallback refused.')
        compute_s = time.perf_counter()-compute_started
        stage = {'actual_device': str(target), 'mode_neff_real': neff.real, 'mode_neff_imag': neff.imag,
            'source_e_component_l2_fractions': polarization.tolist(), 'source_z_invariance_relative_l2': invariant_error,
            'source_mode_kind': 'Local numerical Ez fundamental slab mode; FDTDX tm filter maps to physical Ez under its transverse-axis convention',
            'grid_shape': [nx, ny, 3], 'time_steps': int(final_step),
            'actual_duration_fs': float(final_step)*simulation.time_step_duration*1e15,
            'settled': settled, 'energy_tail_peak_over_run_peak': ratio, 'energy_tail_window_fs': 50.,
            'energy_ratio': ratio, 'spectral_relative_change': spectral_change if np.isfinite(spectral_change) else None,
            'spectrum_tolerance': config['spectrum_tolerance'],
            'thresholds': {'energy_decay': config['decay_by'], 'spectral_relative_change': config['spectrum_tolerance']},
            'elapsed_fs': float(final_step)*simulation.time_step_duration*1e15,
            'spectral_settlement_definition': 'Largest normalized accumulated spectral-flux change at either monitor between consecutive approximately 50 fs checks; max over frequencies divided by monitor peak absolute current flux',
            'settlement_definition': 'Maximum sampled non-PML electromagnetic energy in final 50 fs divided by maximum over run; samples approximately 1 fs apart; unexecuted detector slots excluded',
            'source_effective_end_fs': source_end_fs,
            'source_end_definition': 'Gaussian pulse centered six sigma after start, switched off six sigma after peak; cutoff envelope exp(-18)',
            'pulse_truncated': bool(float(final_step)*simulation.time_step_duration*1e15 < source_end_fs),
            'reached_time_cap': bool(int(final_step) >= simulation.time_steps_total), 'stopped_for_settling': settled,
            'settlement_history': history,
            'max_after_sources_fs': config['max_after_sources_fs'], 'decay_by': config['decay_by'],
            'compile_s': compile_s, 'compute_s': compute_s, 'total_s': time.perf_counter()-stage_started}
        np.savez_compressed(output/f'{name}-monitors.npz', **{f'{d}_phasors': v for d, v in phasors.items()},
            **{f'{d}_tail_phasors': v for d, v in tail_phasors.items()},
            mode_e=mode_e, mode_h=mode_h, energy=energies, energy_times_fs=times,
            frequency_per_um=frequencies, monitor_y_um=np.arange(pml, ny-pml)*dx-ny*dx/2)
        return phasors, stage, mode_e, mode_h, final, epsilon

    print(f'FDTDX ports: verified {target.device_kind} ({target}); starting incident reference', flush=True)
    reference, stages['reference'], source_e, source_h, _, _ = execute_stage('reference', 0.)
    device_gap = config['gap_um'] if config['device_kind'] == 'gap' else 0.
    device, stages['device'], device_e, device_h, final, epsilon = execute_stage('device', device_gap)
    print('PORT_STAGE=normalization', flush=True)
    source_difference = max(float(np.linalg.norm(device_e-source_e)/np.linalg.norm(source_e)),
                            float(np.linalg.norm(device_h-source_h)/np.linalg.norm(source_h)))
    if source_difference > 1e-5:
        raise RuntimeError('Reference and device mode injections differ; incident subtraction is invalid.')
    incident = spectral_flux(reference['reflection'], dx)
    reflected = spectral_flux(device['reflection']-reference['reflection'], dx)
    transmitted = spectral_flux(device['transmission'], dx)
    baseline = spectral_flux(reference['transmission'], dx)
    if not all(np.isfinite(v).all() for v in (incident, reflected, transmitted, baseline)):
        raise RuntimeError('Raw spectral flux contains a nonfinite value; normalization refused.')
    valid = np.isfinite(incident)&(incident > max(float(np.max(incident))*1e-6, 0))
    valid &= stages['reference']['settled'] and stages['device']['settled']
    reflectance, transmittance = normalized_power(incident, reflected, transmitted, valid)
    order = np.argsort(1/frequencies)
    np.savez_compressed(output/'spectra.npz', wavelength_um=(1/frequencies)[order],
        frequency_per_um=frequencies[order], incident_flux=incident[order], reflected_flux=reflected[order],
        transmitted_flux=transmitted[order], reflectance=reflectance[order], transmittance=transmittance[order],
        uncollected_fraction=(1-reflectance-transmittance)[order], valid_bin=valid[order],
        reference_transmitted_flux=baseline[order])
    field = np.asarray(final.fields.E[2, :, :, 1]).T
    np.savez_compressed(output/'arrays.npz', field=field, epsilon=epsilon[:, :, 1].T,
        x_um=np.arange(nx)*dx-nx*dx/2, y_um=np.arange(ny)*dx-ny*dx/2)
    return {'solver': 'fdtdx', 'device': requested, 'actual_device': str(target), 'device_name': target.device_kind,
        'versions': versions, 'field_component': 'Ez', 'field_units': 'arbitrary source units',
        'field_time_fs': stages['device']['actual_duration_fs'],
        'actual_duration_fs': stages['device']['actual_duration_fs'],
        'dimensions': '2D invariant-z slab via three-cell periodic extrusion', 'field_shape': list(field.shape),
        'spectra_valid': bool(np.all(valid)), 'settlement': stages,
        'source': {'kind': 'ModePlaneSource; local Tidy3D numerical cross-section mode',
            'mode_preparation_device': 'CPU callback; FDTD arrays and time evolution use the explicitly selected target',
            'frequency_hz': center_hz, 'gaussian_sigma_frequency_hz': sigma_frequency, 'direction': '+x',
            'gaussian_sigma_time_fs': sigma_time*1e15, 'effective_end_fs': source_end_fs,
            'mode_index': 0, 'physical_polarization': 'Ez, even transverse slab fundamental',
            'requested_x_um': source_x, 'actual_x_um': plane_indices['source']*dx-nx*dx/2,
            'reference_device_mode_relative_difference': source_difference},
        'monitor': {'kind': 'Total signed cross-section Poynting flux, not mode-resolved S-parameters',
            'reflection_x_um': plane_indices['reflection']*dx-nx*dx/2,
            'transmission_x_um': plane_indices['transmission']*dx-nx*dx/2, 'aperture_um': aperture*dx,
            'field_sampling': 'FDTDX exact interpolation co-locates fields at Ez Yee positions and averages adjacent H time levels; phasor DFT uses every time step',
            'normalization': 'Incident reference at reflection plane; complex reference E/H subtracted before reflected Poynting product',
            'raw_flux_orientation': 'All raw fluxes are signed +x; reflected_flux is usually negative and R=-reflected_flux/incident_flux',
            'flux_units': 'common arbitrary spectral power per invariant-z length; ratios dimensionless',
            'phasor_convention': 'sum(field(t)*exp(+i*omega*t)), pulse scaling; common time-step factors cancel in ratios'},
        'geometry': {'actual_core_width_um': core_cells*dx, 'actual_gap_um': round(device_gap/dx)*dx,
            'requested_domain_xy_um': [config['length_um'], config['height_um']],
            'requested_core_width_um': config['core_width_um'], 'requested_gap_um': device_gap,
            'requested_pml_um': config['pml_um'],
            'actual_domain_um': [nx*dx, ny*dx, 3*dx], 'actual_pml_um': pml*dx,
            'actual_core_y_bounds_um': [(ny-core_cells)//2*dx-ny*dx/2,
                ((ny-core_cells)//2+core_cells)*dx-ny*dx/2],
            'actual_gap_x_bounds_um': ([(nx-round(device_gap/dx))//2*dx-nx*dx/2,
                ((nx-round(device_gap/dx))//2+round(device_gap/dx))*dx-nx*dx/2] if device_gap else None),
            'material_grid_verified': True,
            'discretization': 'Lengths rounded to nearest grid-cell count; centered box lower index floored. Bounds describe material voxels, while exported Ez samples use Yee positions.'},
        'field_sampling': 'Raw final Ez at (i*dx,j*dx,(k+0.5)*dx); z middle cell retained. x/y export coordinates are the actual Ez Yee positions, not voxel centers.',
        'boundaries': 'PML both x/y faces; periodic z', 'total_seconds': time.perf_counter()-started,
        'limitations': ['1-R-T is uncollected flux including lateral radiation and numerical error, not absorption.',
            'Source mode profile is computed at central frequency and reused across the pulse spectrum.',
            'No clipping of negative or above-one power ratios; unsettled or weak incident bins are invalid.',
            'Reference and device share the same grid, source, monitor sampling and requested execution target.']}
