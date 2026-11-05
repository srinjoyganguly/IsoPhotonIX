"""Compare mesh sensitivity of measured finite-taper modal scattering."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .passport import file_record
from .source_snapshot import FILENAME, read_implementation
from .sparameters import ENTRIES, PHASE_FLOOR, STAGES, stored_sparameters
from .taper_ports import TaperPortStudy


def _physical_coordinates(metadata, resolution):
    ports = metadata['ports']
    translation = metadata['geometry_translation']
    planes = [ports['port_1_x_um'], ports['port_2_x_um']]
    aperture = ports['aperture_yz_um']
    if metadata['solver'] == 'fdtdx':
        # The adapter rounds source positions to an integer Yee index. Use
        # the recorded grid dimensions/spacing to recover the applied planes.
        dx = translation['grid_spacing_um']
        domain = translation['actual_domain_um']
        nx = translation['grid_shape'][0]
        half = nx * dx / 2
        profile = translation['profile']
        sources = [round((profile[key] + half) / dx) * dx - half
                   for key in ('source_forward_x_um', 'source_reverse_x_um')]
        pml = dx * translation['pml_cells']
        if not np.isclose(dx, 1 / resolution, atol=1e-12, rtol=0):
            raise ValueError('Recorded FDTDX grid spacing differs from the selected mesh.')
        if not translation.get('material_grid_verified'):
            raise ValueError('This FDTDX taper did not verify its material grid.')
    else:
        domain = [translation['domain'][axis] for axis in 'xyz']
        sources = metadata['source']['source_x_um']
        pml = metadata['boundaries']['pml_um']
    positions = {'domain_um': domain, 'source_planes_um': sources, 'reference_planes_um': planes,
                 'aperture_yz_um': aperture, 'pml_um': pml}
    for key, shape in [('domain_um', (3,)), ('source_planes_um', (2,)), ('reference_planes_um', (2,)),
                       ('aperture_yz_um', (2,)), ('pml_um', ())]:
        values = np.asarray(positions[key])
        if values.shape != shape or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError(f'Expected finite physical {key} in micrometres.')
    if min(*domain, *aperture, pml) <= 0 or not sources[0] < planes[0] < planes[1] < sources[1]:
        raise ValueError('Invalid recorded domain, source planes, port planes, aperture or PML thickness.')
    return positions


def _read_run(folder):
    folder = Path(folder)
    if (folder / 'job.json').exists():
        job = json.loads((folder / 'job.json').read_text())
        if job.get('status') != 'completed':
            raise ValueError('Only completed taper-port jobs can be compared.')
    config = json.loads((folder / 'config.json').read_text())
    metadata = json.loads((folder / 'metadata.json').read_text())
    if config.get('study_type') != 'taper-ports-3d' or config.get('schema_version') != 7:
        raise ValueError('Both runs must be finite taper-port studies (schema 7).')
    if metadata.get('config') != config:
        raise ValueError('The saved configuration differs from its metadata.')
    study = TaperPortStudy.model_validate(config)
    if (metadata.get('requested_device'), metadata.get('solver')) != (study.device, study.solver):
        raise ValueError('Recorded solver or device differs from the requested study.')
    platform = metadata['actual_device'].lower().split(':')[0]
    if platform != ('cuda' if study.device == 'gpu' else 'cpu'):
        raise ValueError('Recorded hardware differs from the requested target.')
    implementation = metadata['implementation_snapshot']['sha256']
    read_implementation(folder / FILENAME, implementation)
    spectra = stored_sparameters(folder, metadata)
    coordinates = _physical_coordinates(metadata, study.resolution)
    source = dict(metadata['source'])
    if study.solver == 'meep' and source.pop('eig_resolution') != study.resolution:
        raise ValueError('The Meep mode grid must track the requested Maxwell grid.')
    boundaries = dict(metadata['boundaries'])
    boundaries.pop('pml_cells', None)  # The physical thickness is compared below.
    ports = metadata['ports']
    convention = {key: ports[key] for key in ('phase_convention', 'mode_gauge', 'normalization')}
    numerical = {'implementation_sha256': implementation, 'versions': metadata['versions'],
                 'actual_device': metadata['actual_device'], 'device_name': metadata['device_name'],
                 'source': source, 'boundaries': boundaries, 'port_conventions': convention,
                 'normalization': metadata['normalization']}
    if not all(numerical.values()):
        raise ValueError('Complete solver, source, hardware and normalization records are required.')
    if study.solver == 'fdtdx' and any(metadata['settlement'][stage].get('actual_device') != metadata['actual_device']
                                       for stage in STAGES):
        raise ValueError('The four FDTDX stages must all record the selected hardware.')
    summary = {'resolution_cells_per_um': study.resolution,
               'files': [file_record(folder / name) for name in ('config.json', 'metadata.json', 'sparameters.npz')],
               'valid_bins': spectra['valid_bin'].tolist(),
               'settled_stages': {stage: metadata['settlement'][stage]['settled'] for stage in STAGES}}
    return study, numerical, coordinates, spectra, summary


def _entry_metrics(coarse, fine, valid):
    """Only take phases at jointly qualified, resolvable amplitudes."""
    # Slice first. Invalid bins may contain NaNs even in wholly usable runs.
    x, y = coarse[valid], fine[valid]
    complex_delta = np.abs(y - x)
    power_delta = np.abs(y)**2 - np.abs(x)**2
    phase_mask = (np.abs(x) > PHASE_FLOOR) & (np.abs(y) > PHASE_FLOOR)
    phase_difference = np.angle(y[phase_mask]) - np.angle(x[phase_mask])
    phase = np.arctan2(np.sin(phase_difference), np.cos(phase_difference))
    power_output = np.full(len(valid), np.nan)
    phase_output = np.full(len(valid), np.nan)
    power_output[valid] = power_delta
    phase_output[np.flatnonzero(valid)[phase_mask]] = phase
    def nullable(values):
        return [float(value) if np.isfinite(value) else None for value in values]
    return {'maximum_absolute_complex_change': float(complex_delta.max()),
            'maximum_absolute_power_change': float(np.max(np.abs(power_delta))),
            'maximum_absolute_phase_change_rad': float(np.max(np.abs(phase))) if phase.size else None,
            'power_change': nullable(power_output), 'principal_phase_change_rad': nullable(phase_output)}


def compare_taper_mesh(coarse_folder, fine_folder):
    coarse, c_method, c_positions, c, c_summary = _read_run(coarse_folder)
    fine, f_method, f_positions, f, f_summary = _read_run(fine_folder)
    if coarse.resolution >= fine.resolution:
        raise ValueError('Provide a coarse run first and a strictly finer mesh second.')
    if coarse.model_dump(exclude={'resolution'}) != fine.model_dump(exclude={'resolution'}):
        raise ValueError('Both runs must use the same saved study and target. Only resolution may differ.')
    for key in c_method:
        if c_method[key] != f_method[key]:
            raise ValueError(f'The runs differ in {key}. Use matching solver sources, versions, hardware and conventions.')
    for key, value in c_positions.items():
        if not np.allclose(value, f_positions[key], atol=1e-10, rtol=0):
            raise ValueError(f'Applied {key} differs between meshes. Choose physically aligned grids.')
    if not np.allclose(c['wavelength_um'], f['wavelength_um'], atol=1e-12, rtol=0):
        raise ValueError('Wavelength sample grids differ. This report compares measured bins without interpolation.')
    valid = c['valid_bin'] & f['valid_bin']
    if not np.any(valid):
        raise ValueError('No wavelength bin has settled, guided, sufficiently strong measurements on both meshes.')
    entries = {name: _entry_metrics(c['s'][:, i, j], f['s'][:, i, j], valid) for name, i, j in ENTRIES}
    coarse_matrix = c['s'][valid]
    denominator = np.linalg.norm(coarse_matrix)
    relative = float(np.linalg.norm(f['s'][valid] - coarse_matrix) / denominator) if denominator else None
    mode_changes = np.max(np.abs(f['mode_neff_ports'][valid] - c['mode_neff_ports'][valid]), axis=0)
    return {'schema_version': 1, 'kind': 'taper-port-mesh-sensitivity',
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'solver': coarse.solver, 'device': coarse.device, **c_method, **c_positions,
            'coarse': c_summary, 'fine': f_summary,
            'wavelength_um': c['wavelength_um'].tolist(), 'jointly_valid_bins': valid.tolist(),
            'compared_bins': int(valid.sum()), 'complex_matrix_relative_l2': relative,
            'maximum_absolute_port_index_change': mode_changes.tolist(), 'entries': entries,
            'phase_amplitude_floor': PHASE_FLOOR,
            'interpretation': 'Changes are fine minus coarse at fixed physical planes. Power is |S|². '
            'Phase is the principal angle in radians, where both amplitudes exceed the floor. '
            'The relative matrix L2 norm uses the coarse matrix norm over mutually valid bins. '
            'Two resolutions measure sensitivity. An absolute error bound or convergence order requires additional evidence.'}
