"""Reference-normalized 2D slab power spectra using actual Meep eigenmode pulses.

This dedicated invariant-z study is separate from finite-height 3D fields.
Official API: https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/
and https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/ .
"""
from pathlib import Path
import platform
import time

C_UM_PER_FS = .299792458


def run(config, output_dir):
    if config.get('solver') != 'meep' or config.get('device') != 'cpu':
        raise ValueError('This Meep port adapter requires an explicitly selected CPU target.')
    if config.get('study_type') != 'waveguide-ports-2d' or config.get('device_kind') not in ('straight', 'gap'):
        raise ValueError('Only the dedicated straight/gap 2D port study is supported.')
    import meep as mp
    import numpy as np

    mp.verbosity(0)
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    length, height = float(config['length_um']), float(config['height_um'])
    width, pml = float(config['core_width_um']), float(config['pml_um'])
    resolution = int(config['resolution'])
    aperture = height - 2*pml
    source_x = -length/2 + pml + .5
    reflection_x, transmission_x = source_x+1, length/2-pml-1.5
    freq_min, freq_max = 1/config['wavelength_max_um'], 1/config['wavelength_min_um']
    frequency, bandwidth = (freq_min+freq_max)/2, freq_max-freq_min
    pulse_width, cutoff = 2*bandwidth, 6.
    source_end = 2*cutoff/pulse_width
    max_after = config['max_after_sources_fs']*C_UM_PER_FS
    decay_by = float(config['decay_by'])
    spectrum_tolerance = float(config.get('spectrum_tolerance', 1e-4))
    check_interval = 5.  # µm/c, several optical cycles; documented in metadata.
    count = int(config['frequency_points'])
    core = mp.Medium(index=float(config['core_index']))
    cladding = mp.Medium(index=float(config['cladding_index']))
    cell = mp.Vector3(length, height)
    volume = mp.Volume(center=mp.Vector3(), size=cell)
    parity = mp.ODD_Z + mp.EVEN_Y

    def build(device_kind):
        geometry = [mp.Block(size=mp.Vector3(mp.inf, width, mp.inf), material=core)]
        if device_kind == 'gap':
            # Later material takes precedence, opening a gap in the otherwise
            # identical guide. The reference retains the uninterrupted core.
            geometry.append(mp.Block(size=mp.Vector3(config['gap_um'], width, mp.inf), material=cladding))
        sim = mp.Simulation(cell_size=cell, dimensions=2, resolution=resolution, Courant=.5,
                            geometry=geometry, default_material=cladding,
                            boundary_layers=[mp.PML(pml)],
                            sources=[mp.EigenModeSource(
                                mp.GaussianSource(frequency, fwidth=pulse_width, cutoff=cutoff),
                                center=mp.Vector3(source_x), size=mp.Vector3(0, aperture),
                                direction=mp.X, eig_band=1, eig_kpoint=mp.Vector3(1),
                                eig_match_freq=True, eig_parity=parity,
                                eig_resolution=resolution, eig_tolerance=1e-10)])
        reflected = sim.add_flux(frequency, bandwidth, count,
            mp.FluxRegion(center=mp.Vector3(reflection_x), size=mp.Vector3(0, aperture), direction=mp.X))
        transmitted = sim.add_flux(frequency, bandwidth, count,
            mp.FluxRegion(center=mp.Vector3(transmission_x), size=mp.Vector3(0, aperture), direction=mp.X))
        return sim, reflected, transmitted

    def evolve(sim, monitors, stage, incident=None):
        stage_start = time.perf_counter()
        print(f'PORT_STAGE={stage}', flush=True)
        state = {'settled': False, 'peak_energy': 0., 'last_energy_ratio': None,
                 'last_flux_change': None, 'consecutive_checks': 0, 'history': []}
        previous = None

        def check(s):
            nonlocal previous
            energy = abs(float(s.field_energy_in_box(box=volume)))
            if not np.isfinite(energy):
                raise RuntimeError(f'{stage} energy became non-finite.')
            state['peak_energy'] = max(state['peak_energy'], energy)
            current = np.array([mp.get_fluxes(monitor) for monitor in monitors])
            if not np.isfinite(current).all():
                raise RuntimeError(f'{stage} flux became non-finite.')
            denominator = np.abs(current[1]) if incident is None else np.abs(incident)
            denominator = np.maximum(denominator, max(float(np.max(denominator))*1e-6, 1e-100))
            change = None if previous is None else float(np.max(np.abs(current-previous)/denominator))
            ratio = energy/state['peak_energy'] if state['peak_energy'] else None
            state['last_energy_ratio'], state['last_flux_change'] = ratio, change
            state['history'].append({'time_fs': float(s.meep_time()/C_UM_PER_FS),
                                     'energy_ratio': ratio, 'flux_change': change})
            if s.meep_time() > source_end+check_interval and ratio is not None and change is not None and ratio <= decay_by and change <= spectrum_tolerance:
                state['consecutive_checks'] += 1
            else:
                state['consecutive_checks'] = 0
            state['settled'] = state['consecutive_checks'] >= 2
            previous = current

        def stop(s):
            return state['settled'] or s.meep_time() >= source_end+max_after

        print(f'Meep ports: {stage} pulse and measured decay', flush=True)
        sim.run(mp.at_every(check_interval, check), until=stop)
        state.update(actual_time_fs=float(sim.meep_time()/C_UM_PER_FS),
                     elapsed_fs=float(sim.meep_time()/C_UM_PER_FS),
                     after_source_fs=float((sim.meep_time()-sim.fields.last_source_time())/C_UM_PER_FS),
                     actual_source_end_fs=float(sim.fields.last_source_time()/C_UM_PER_FS),
                     time_steps=int(sim.fields.t), elapsed_s=time.perf_counter()-stage_start,
                     reason='energy_and_flux_decay' if state['settled'] else 'maximum_time_reached',
                     decay_by=decay_by, spectrum_tolerance=spectrum_tolerance,
                     energy_ratio=state['last_energy_ratio'], spectral_relative_change=state['last_flux_change'],
                     thresholds={'energy_decay': decay_by, 'spectral_relative_change': spectrum_tolerance},
                     check_interval_fs=check_interval/C_UM_PER_FS,
                     required_consecutive_checks=2,
                     definition='Whole-cell electromagnetic energy relative to sampled peak and maximum per-bin flux change relative to incident power; both must pass twice after source turn-off.')
        print(f"Meep ports: {stage} {state['reason']}; energy ratio={state['last_energy_ratio']}; flux change={state['last_flux_change']}", flush=True)
        return state

    reference, reference_r, reference_t = build('straight')
    reference_state = evolve(reference, [reference_r, reference_t], 'reference')
    incident = np.asarray(mp.get_fluxes(reference_r))
    reference_transmitted = np.asarray(mp.get_fluxes(reference_t))
    reference_data = reference.get_flux_data(reference_r)
    frequencies = np.asarray(mp.get_flux_freqs(reference_r))
    reference.reset_meep()

    device, reflected_monitor, transmitted_monitor = build(config['device_kind'])
    device.load_minus_flux_data(reflected_monitor, reference_data)
    device_state = evolve(device, [reflected_monitor, transmitted_monitor], 'device', incident)
    reflected = np.asarray(mp.get_fluxes(reflected_monitor))
    transmitted = np.asarray(mp.get_fluxes(transmitted_monitor))
    scattered_data = device.get_flux_data(reflected_monitor)
    transmitted_data = device.get_flux_data(transmitted_monitor)
    field = np.asarray(device.get_array(component=mp.Ez, vol=volume), dtype=np.float32).T
    epsilon = np.asarray(device.get_array(component=mp.Dielectric, vol=volume), dtype=np.float32).T
    x, y, _, _ = device.get_array_metadata(vol=volume)
    x, y = np.asarray(x), np.asarray(y)
    if field.shape != (len(y), len(x)) or epsilon.shape != field.shape or not np.isfinite(field).all() or not np.isfinite(epsilon).all():
        raise RuntimeError('Actual Meep field/material export is malformed or non-finite.')
    np.savez_compressed(output/'arrays.npz', field=field, epsilon=epsilon, x_um=x, y_um=y)

    print('PORT_STAGE=normalization', flush=True)
    positive_incident = incident > max(float(np.max(incident))*1e-6, 0.)
    settled = reference_state['settled'] and device_state['settled']
    # Undefined ratios remain NaN in NPZ and are invalid, never fabricated zero.
    reflectance = np.divide(-reflected, incident, out=np.full_like(incident, np.nan), where=positive_incident)
    transmittance = np.divide(transmitted, incident, out=np.full_like(incident, np.nan), where=positive_incident)
    valid = positive_incident & settled & np.isfinite(reflectance) & np.isfinite(transmittance)
    reference_ratio = np.divide(reference_transmitted, incident, out=np.full_like(incident, np.nan), where=positive_incident)
    finite_reference_ratio = reference_ratio[np.isfinite(reference_ratio)]
    order = np.argsort(1/frequencies)
    spectra = dict(wavelength_um=1/frequencies, frequency_per_um=frequencies,
                   incident_flux=incident, reflected_flux=reflected, transmitted_flux=transmitted,
                   reference_transmitted_flux=reference_transmitted,
                   reflectance=reflectance, transmittance=transmittance,
                   uncollected_fraction=1-reflectance-transmittance, valid_bin=valid)
    saved_spectra = {key: value[order] for key, value in spectra.items()}
    saved_spectra.update(reference_dft_e=np.asarray(reference_data.E), reference_dft_h=np.asarray(reference_data.H),
                         scattered_dft_e=np.asarray(scattered_data.E), scattered_dft_h=np.asarray(scattered_data.H),
                         transmitted_dft_e=np.asarray(transmitted_data.E), transmitted_dft_h=np.asarray(transmitted_data.H))
    np.savez_compressed(output/'spectra.npz', **saved_spectra)
    elapsed = time.perf_counter()-started
    result = {'solver': 'meep', 'actual_device': 'CPU',
              'versions': {'meep': mp.__version__, 'numpy': np.__version__, 'python': platform.python_version()},
              'dimensions': '2D (invariant z)', 'field_component': 'Ez',
              'field_units': 'arbitrary eigenmode-pulse source units',
              'field_time_fs': device_state['actual_time_fs'],
              'field_sampling': 'Meep get_array interpolation onto common output coordinates; final-time ringdown snapshot.',
              'material_sampling': 'Meep subpixel averaging; scalar Dielectric export at interfaces.',
              'actual_duration_fs': device_state['actual_time_fs'],
              'grid_shape': [len(x), len(y)], 'timings_s': {'total_s': elapsed},
              'source': {'type': 'Gaussian eigenmode pulse', 'component': 'Ez', 'mode': 'fundamental TE0, even y',
                         'eig_band': 1, 'eig_parity': 'ODD_Z + EVEN_Y', 'direction': '+x',
                         'x_um': source_x, 'aperture_um': aperture, 'center_frequency_per_um': frequency,
                         'fwidth_per_um': pulse_width, 'cutoff': cutoff,
                         'source_duration_fs': source_end/C_UM_PER_FS,
                         'note': 'Mode profile solved at pulse centre frequency; broadband injection error is possible and requires calibration/convergence.'},
              'monitors': {'reflection_x_um': reflection_x, 'transmission_x_um': transmission_x,
                           'aperture_um': aperture, 'normal': '+x'},
              'boundaries': {'x': 'PML', 'y': 'PML', 'z': 'invariant', 'pml_um': pml},
              'normalization': {'reference': 'uninterrupted straight guide with identical source, mesh and monitors',
                                'reflection': 'negative flux after subtracting reference complex DFT E/H fields, divided by incident reference flux',
                                'transmission': 'device transmitted total cross-section flux divided by incident reference flux',
                                'weak_incident_relative_threshold': 1e-6, 'clipping': False,
                                'reference_monitor_transmission_min': float(np.min(finite_reference_ratio)) if finite_reference_ratio.size else None,
                                'reference_monitor_transmission_max': float(np.max(finite_reference_ratio)) if finite_reference_ratio.size else None,
                                'opaque_dft_data': 'Extra *_dft_e/h arrays are exact opaque Meep FluxData streams in native order, for identical monitor/chunk reload only. They are not wavelength-indexed or spatially labelled field arrays.'},
              'settlement': {'reference': reference_state, 'device': device_state},
              'spectra_valid': bool(np.all(valid)), 'valid_frequency_count': int(np.sum(valid)),
              'notes': ['Lossless nondispersive invariant-z slab; this is not a finite-height 3D calculation.',
                        'Total cross-sectional flux, not solved modal S-parameters or guaranteed mode purity.',
                        'Uncollected fraction 1-R-T includes radiation and numerical/monitor residual; it is not absorption.',
                        'Ratios are never clipped. Unsettled or weak-incident bins are invalid and must remain visible as such.',
                        'Meeting the temporal stopping criterion does not establish grid/PML/domain convergence.']}
    device.reset_meep()
    return result
