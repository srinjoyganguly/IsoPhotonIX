"""Finite-height 3D modal two-port scattering measured with Meep/MPB.

Each column is independently excited and incident-normalized. The signed-k
mode bases have positive-real Ey at the core center at both physical planes.
No through calibration removes the actual propagation phase or power loss.
"""
from pathlib import Path
import platform
import time

C_UM_PER_FS = .299792458


def run(config, output_dir):
    if config.get('solver') != 'meep' or config.get('device') != 'cpu':
        raise ValueError('Meep 3D ports require an explicitly selected Meep CPU target.')
    if config.get('schema_version') != 4 or config.get('study_type') != 'waveguide-ports-3d':
        raise ValueError('Expected a schema-version-4 finite-height 3D modal-port study.')
    if config.get('device_kind') not in ('straight', 'gap'):
        raise ValueError('Only straight and centered cladding-gap studies are supported.')
    import meep as mp
    import numpy as np

    mp.verbosity(0)
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geometry = config['geometry']
    objects = {item['id']: item for item in geometry['objects']}
    if set(objects)-{'core', 'substrate'} or 'core' not in objects:
        raise ValueError('A centered rectangular core and optional substrate are required.')
    if any(item['kind'] != 'box' or item['rotation_deg'] % 360 for item in objects.values()):
        raise ValueError('The modal-port adapter supports only unrotated rectangular solids.')
    core, substrate = objects['core'], objects.get('substrate')
    length, height, depth = (float(geometry['domain'][axis]) for axis in 'xyz')
    width, core_height = core['size']['y'], core['size']['z']
    resolution, pml = config['resolution'], config['pml_um']
    cladding_index, core_index = config['cladding_index'], core['refractive_index']
    high_cladding = max(cladding_index, substrate['refractive_index'] if substrate else cladding_index)
    source_offset = length/2-pml-.5
    port_offset = source_offset-1
    port_positions = (-port_offset, port_offset)
    aperture = (height-2*pml, depth-2*pml)
    frequency_min, frequency_max = 1/config['wavelength_max_um'], 1/config['wavelength_min_um']
    frequency, span = (frequency_min+frequency_max)/2, frequency_max-frequency_min
    count, source_end = config['frequency_points'], 12/(2*span)
    cap = source_end+config['max_after_sources_fs']*C_UM_PER_FS
    check_interval, eig_tolerance = 5., 1e-12
    parity = mp.ODD_Y if substrate else mp.ODD_Y+mp.EVEN_Z
    symmetries = [mp.Mirror(mp.Y, phase=-1)]
    if not substrate:
        symmetries.append(mp.Mirror(mp.Z, phase=1))
    cell = mp.Vector3(length, height, depth)
    volume = mp.Volume(center=mp.Vector3(), size=cell)
    plane_size = mp.Vector3(0, *aperture)
    settlement, native_stages = {}, {}
    modal_checks = []
    field_time_fs = None

    def materials(kind):
        result = []
        for identifier in ('substrate', 'core'):
            if identifier not in objects:
                continue
            obj = objects[identifier]
            size = [float(obj['size'][axis]) for axis in 'xyz']
            if identifier == 'core':
                size[0] = mp.inf
            result.append(mp.Block(size=mp.Vector3(*size),
                center=mp.Vector3(*(obj['position'][axis] for axis in 'xyz')),
                material=mp.Medium(index=obj['refractive_index'])))
        if kind == 'gap':
            result.append(mp.Block(size=mp.Vector3(config['gap_um'], width, core_height),
                                   material=mp.Medium(index=cladding_index)))
        return result

    def build(kind, direction):
        mp.set_random_seed(0)
        sim = mp.Simulation(cell_size=cell, dimensions=3, resolution=resolution, Courant=.5,
            default_material=mp.Medium(index=cladding_index), geometry=materials(kind),
            boundary_layers=[mp.PML(pml)], symmetries=symmetries,
            sources=[mp.EigenModeSource(mp.GaussianSource(frequency, fwidth=2*span, cutoff=6),
                center=mp.Vector3(-direction*source_offset), size=plane_size,
                direction=mp.X, eig_band=1, eig_kpoint=mp.Vector3(direction),
                eig_match_freq=True, eig_parity=parity, eig_resolution=resolution,
                eig_tolerance=eig_tolerance)])
        monitors = [sim.add_flux(frequency, span, count,
            mp.FluxRegion(center=mp.Vector3(position), size=plane_size, direction=mp.X),
            decimation_factor=1) for position in port_positions]
        return sim, monitors

    def norm(data):
        return float(np.linalg.norm(np.r_[data.E, data.H]))

    def evolve(sim, monitors, stage, reference_data=None):
        print(f'PORT_STAGE={stage}', flush=True)
        stage_started = time.perf_counter()
        state = {'settled': False, 'energy_ratio': None, 'spectral_relative_change': None,
                 'sampled_energy_peak': 0., 'consecutive_checks': 0, 'history': []}
        previous, maximum_norms = None, np.zeros(2)

        def check(s):
            nonlocal previous, maximum_norms
            energy = abs(float(s.field_energy_in_box(box=volume)))
            if not np.isfinite(energy):
                raise RuntimeError(f'{stage}: nonfinite electromagnetic energy.')
            state['sampled_energy_peak'] = max(state['sampled_energy_peak'], energy)
            current = [s.get_flux_data(monitor) for monitor in monitors]
            if any(not np.isfinite(np.r_[item.E, item.H]).all() for item in current):
                raise RuntimeError(f'{stage}: nonfinite complex monitor fields.')
            current_norms = np.array([norm(item) for item in current])
            maximum_norms = np.maximum(maximum_norms, current_norms)
            denominator = np.array([norm(item) for item in reference_data]) if reference_data else maximum_norms
            denominator = np.maximum(denominator, max(float(np.max(denominator))*1e-6, 1e-100))
            change = None if previous is None else float(max(
                np.linalg.norm(np.r_[item.E-old.E, item.H-old.H])/scale
                for item, old, scale in zip(current, previous, denominator)))
            ratio = energy/state['sampled_energy_peak'] if state['sampled_energy_peak'] else None
            state.update(energy_ratio=ratio, spectral_relative_change=change)
            okay = (s.meep_time() > source_end+check_interval and ratio is not None and
                    change is not None and ratio <= config['decay_by'] and change <= config['spectrum_tolerance'])
            state['consecutive_checks'] = state['consecutive_checks']+1 if okay else 0
            state['settled'] = state['consecutive_checks'] >= 2
            state['history'].append({'time_fs': float(s.meep_time()/C_UM_PER_FS),
                                    'energy_ratio': ratio, 'spectral_relative_change': change})
            previous = current

        sim.run(mp.at_every(check_interval, check), until=lambda s: state['settled'] or s.meep_time() >= cap)
        state.update(elapsed_fs=float(sim.meep_time()/C_UM_PER_FS),
            actual_source_end_fs=float(sim.fields.last_source_time()/C_UM_PER_FS),
            time_steps=int(sim.timestep()), total_seconds=time.perf_counter()-stage_started,
            reason='energy_and_complex_dft_settled' if state['settled'] else 'maximum_time_reached',
            thresholds={'energy_decay': config['decay_by'], 'spectral_relative_change': config['spectrum_tolerance']},
            decay_by=config['decay_by'], spectrum_tolerance=config['spectrum_tolerance'],
            check_interval_fs=check_interval/C_UM_PER_FS, required_consecutive_checks=2,
            definition='Whole-cell energy/sample peak and L2 change of complex E/H DFT streams/reference norm, or largest sampled norm for reference stages; both pass twice after pulse turn-off.')
        print(f"Meep 3D ports {stage}: {state['reason']} at {state['elapsed_fs']:.3f} fs", flush=True)
        return state

    def modal_data(sim, monitor, position):
        # Solve the signed-k basis independently for each propagation direction.
        # The opposite-direction branch of one sampled +k profile need not have
        # the same Yee-grid normalization or electric gauge as the true -k mode.
        frequencies = np.asarray(mp.get_flux_freqs(monitor))
        by_direction = []
        transverse_y = np.linspace(-aperture[0]/2, aperture[0]/2, 11)
        transverse_z = np.linspace(-aperture[1]/2, aperture[1]/2, 11)
        for branch, sign in enumerate((1, -1)):
            mp.set_random_seed(0)
            values = sim.get_eigenmode_coefficients(monitor, [1], eig_parity=parity,
                eig_resolution=resolution, eig_tolerance=eig_tolerance, direction=mp.X,
                kpoint_func=lambda f, band: mp.Vector3(sign*f*(core_index+high_cladding)/2))
            raw_alpha = np.array(values.alpha[0, :, branch], copy=True)
            anchors, fractions, edge_ratios = [], [], []
            for f, k in zip(frequencies, values.kpoints):
                mp.set_random_seed(0)
                mode = sim.get_eigenmode(f, mp.X, mp.Volume(center=mp.Vector3(position), size=plane_size),
                    1, k, parity=parity, resolution=resolution, eigensolver_tol=eig_tolerance)
                anchor = complex(mode.amplitude(mp.Vector3(position), mp.Ey))
                samples = np.array([[[mode.amplitude(mp.Vector3(position, y, z), component)
                    for y in transverse_y] for z in transverse_z] for component in (mp.Ex, mp.Ey, mp.Ez)])
                if not np.isfinite(samples).all() or not np.isfinite(anchor) or abs(anchor) <= 1e-8:
                    raise RuntimeError('The selected mode has no finite nonzero center-Ey phase anchor.')
                powers = np.sum(np.abs(samples)**2, axis=(1, 2))
                fractions.append(powers/np.sum(powers))
                intensity = np.sum(np.abs(samples)**2, axis=0)
                edge = max(np.max(intensity[0]), np.max(intensity[-1]), np.max(intensity[:, 0]), np.max(intensity[:, -1]))
                edge_ratios.append(float(edge/np.max(intensity)))
                anchors.append(anchor)
            anchors = np.asarray(anchors)
            gauge = anchors/np.abs(anchors)
            alpha = raw_alpha*gauge
            neff = np.asarray([point.x for point in values.kpoints])/frequencies
            fractions = np.asarray(fractions)
            guided = np.isfinite(neff) & (sign*neff > high_cladding) & (sign*np.asarray(values.vgrp) > 0)
            polarization = fractions[:, 1] > .5
            if not np.isfinite(alpha).all() or not np.all(guided & polarization):
                raise RuntimeError('MPB did not find the required signed-k guided Ey-dominant mode at every frequency.')
            by_direction.append({'alpha': alpha, 'raw_alpha': raw_alpha, 'gauge': gauge,
                'ey_anchor': anchors, 'mode_neff': neff, 'group_velocity': np.asarray(values.vgrp),
                'electric_l2_fraction': fractions, 'edge_intensity_relative_peak': np.asarray(edge_ratios)})
        result = {key: np.stack([item[key] for item in by_direction], axis=1)
                  for key in by_direction[0] if key != 'mode_neff'}
        result.update(mode_neff=by_direction[0]['mode_neff'],
                      signed_mode_neff=np.stack([item['mode_neff'] for item in by_direction], axis=1),
                      frequency_per_um=frequencies)
        return result

    def execute(kind, direction, stage, reference_data=None):
        nonlocal field_time_fs
        sim, monitors = build(kind, direction)
        input_port = 0 if direction == 1 else 1
        if reference_data:
            sim.load_minus_flux_data(monitors[input_port], reference_data[input_port])
        settlement[stage] = evolve(sim, monitors, stage, reference_data)
        data = [sim.get_flux_data(monitor) for monitor in monitors]
        coefficients = [modal_data(sim, monitor, position) for monitor, position in zip(monitors, port_positions)]
        modal_checks.append({'stage': stage, 'minimum_neff': float(min(np.min(item['mode_neff']) for item in coefficients)),
            'minimum_ey_l2_fraction': float(min(np.min(item['electric_l2_fraction'][:, :, 1]) for item in coefficients)),
            'maximum_edge_intensity_relative_peak': float(max(np.max(item['edge_intensity_relative_peak']) for item in coefficients)),
            'guided': True, 'group_velocity_matches_direction': True, 'gauge': 'Center Ey real and positive; independently solved +k and -k bases'})
        saved = {'frequency_per_um': coefficients[0]['frequency_per_um'], 'port_x_um': np.asarray(port_positions)}
        for port, (native, modal) in enumerate(zip(data, coefficients), start=1):
            saved.update({f'port{port}_dft_e': np.asarray(native.E), f'port{port}_dft_h': np.asarray(native.H)})
            saved.update({f'port{port}_{key}': value for key, value in modal.items() if key != 'frequency_per_um'})
        np.savez_compressed(output/(stage.replace('_', '-')+'-monitors.npz'), **saved)
        native_stages[stage] = coefficients
        if stage == 'device_forward':
            fields = np.stack([np.asarray(sim.get_array(component=component, vol=volume), dtype=np.float32).transpose(2, 1, 0)
                               for component in (mp.Ex, mp.Ey, mp.Ez)])
            epsilon = np.asarray(sim.get_array(component=mp.Dielectric, vol=volume), dtype=np.float32).transpose(2, 1, 0)
            x, y, z, _ = sim.get_array_metadata(vol=volume)
            np.savez_compressed(output/'arrays.npz', fields=fields, field=fields[1], epsilon=epsilon,
                                x_um=np.asarray(x), y_um=np.asarray(y), z_um=np.asarray(z))
            field_time_fs = settlement[stage]['elapsed_fs']
        sim.reset_meep()
        return data, coefficients

    for direction, suffix in ((1, 'forward'), (-1, 'reverse')):
        reference_data, _ = execute('straight', direction, 'reference_'+suffix)
        execute(config['device_kind'], direction, 'device_'+suffix, reference_data)

    print('PORT_STAGE=normalization', flush=True)
    reference_forward, reference_reverse = native_stages['reference_forward'], native_stages['reference_reverse']
    device_forward, device_reverse = native_stages['device_forward'], native_stages['device_reverse']
    incident = np.stack((reference_forward[0]['alpha'][:, 0], reference_reverse[1]['alpha'][:, 1]), axis=1)
    outgoing = np.empty((count, 2, 2), dtype=np.complex128)
    outgoing[:, 0, 0], outgoing[:, 1, 0] = device_forward[0]['alpha'][:, 1], device_forward[1]['alpha'][:, 0]
    outgoing[:, 0, 1], outgoing[:, 1, 1] = device_reverse[0]['alpha'][:, 1], device_reverse[1]['alpha'][:, 0]
    positive = np.all(np.abs(incident)**2 > np.max(np.abs(incident)**2, axis=0)*1e-6, axis=1)
    s = np.divide(outgoing, incident[:, None, :], out=np.full_like(outgoing, np.nan+1j*np.nan), where=positive[:, None, None])
    reference_output = np.stack((reference_forward[1]['alpha'][:, 0], reference_reverse[0]['alpha'][:, 1]), axis=1)
    reference_through = np.divide(reference_output, incident, out=np.full_like(incident, np.nan+1j*np.nan), where=positive[:, None])
    valid = positive & all(item['settled'] for item in settlement.values()) & np.isfinite(s).all(axis=(1, 2))
    frequencies = reference_forward[0]['frequency_per_um']
    order = np.argsort(1/frequencies)
    np.savez_compressed(output/'sparameters.npz', wavelength_um=(1/frequencies)[order], frequency_per_um=frequencies[order],
        s=s[order], incident_amplitude=incident[order], outgoing_amplitude=outgoing[order],
        reference_through=reference_through[order], valid_bin=valid[order],
        mode_neff=reference_forward[0]['mode_neff'][order].astype(np.complex128),
        mode_valid=np.ones(count, dtype=bool))
    with np.load(output/'arrays.npz', allow_pickle=False) as arrays:
        field_shape = list(arrays['field'].shape)
    return {'solver': 'meep', 'device': 'cpu', 'actual_device': 'CPU', 'device_name': 'CPU',
        'versions': {'meep': mp.__version__, 'numpy': np.__version__, 'python': platform.python_version()},
        'dimensions': '3D', 'field_component': 'Ey', 'field_components': ['Ex', 'Ey', 'Ez'],
        'field_shape': field_shape, 'array_order': 'fields[component,z,y,x]; final device_forward ringdown',
        'field_units': 'arbitrary eigenmode pulse units', 'field_capture_stage': 'device_forward',
        'field_time_fs': field_time_fs, 'actual_duration_fs': field_time_fs,
        'field_sampling': 'Meep get_array interpolation on common coordinates; exact mirror-symmetry reconstruction of full 3D volume',
        'precision': 'Meep float64 evolution and complex128 modal coefficients; final volume exported float32',
        'total_seconds': time.perf_counter()-started, 'settlement': settlement,
        'sparameters_valid': bool(np.all(valid)), 'modal_checks': modal_checks,
        'source': {'type': '3D MPB fundamental Ey-like eigenmode Gaussian pulse', 'band': 1,
            'eig_parity': 'ODD_Y' if substrate else 'ODD_Y + EVEN_Z', 'eig_resolution': resolution,
            'eig_tolerance': eig_tolerance, 'center_frequency_per_um': frequency, 'fwidth_per_um': 2*span,
            'cutoff': 6, 'source_duration_fs': source_end/C_UM_PER_FS, 'source_x_um': [-source_offset, source_offset],
            'profile_note': 'Source mode is solved at pulse center; projection modes are solved independently at every reported frequency.'},
        'boundaries': {'all_six_faces': 'PML', 'pml_um': pml,
            'exact_symmetries': ['Y odd'] if substrate else ['Y odd', 'Z even']},
        'ports': {'port_1_x_um': port_positions[0], 'port_2_x_um': port_positions[1], 'aperture_yz_um': list(aperture),
            'basis': 'One Ey-dominant guided mode per lead, evaluated at each frequency; squared amplitude is modal power',
            'phase_convention': 'DFT exp(+i omega t); forward propagation exp(+i beta x); center Ey is real positive at each physical port; both directions use the same transverse E orientation',
            'mode_gauge': 'Positive-real Ey at the core center at each physical plane; independent +k and -k modes for the two propagation directions',
            'normalization': 'Direction-specific unit-power MPB coefficients on the Meep Yee integration grid; outgoing device coefficients divided by matching measured incident reference coefficients',
            'matrix_order': 's[frequency,outgoing_port,incoming_port]; rows and columns are [port1,port2]',
            'reference_planes': 'Physical monitor planes; propagation phase retained; no reference-through division or de-embedding'},
        'normalization': {'incident': 'Measured incoming modal coefficient of the matching straight reference',
            'reflection': 'Reference complex Fourier E/H subtracted at input plane before native unit-power modal decomposition',
            'transmission': 'Actual outgoing device modal coefficient divided by measured incident coefficient',
            'weak_incident_power_relative_threshold': 1e-6, 'clipping': False,
            'raw_monitors': 'Native *_dft_e/h are opaque Meep FluxData streams; per-port alpha arrays are [ascending frequency,+x/-x] from independent signed-k mode solves with exported center-Ey gauges'},
        'limitations': ['One fundamental Ey-like mode per physical port; radiation and other modes are not a full loss or absorption accounting.',
            'Lossless nondispersive straight or centered-gap finite-height guide; optional substrate breaks Z mirror symmetry.',
            'Temporal settling and device repeatability do not establish spatial/PML convergence or commercial agreement.',
            'Finite transverse modal aperture and fixed-profile broadband injection require separate sensitivity checks.',
            'S values and passivity/reciprocity discrepancies are never clipped or symmetrized.']}
