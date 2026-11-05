"""Repeatability metrics for compatible saved studies, not external accuracy."""
from datetime import datetime, timezone
import json

import numpy as np

from .volume import COMPONENTS, stored_volume


def physics_spec(config):
    # Presentation state and execution device do not change this physical study.
    if config.get('study_type') == 'layout-3d':
        from .layout_grid import rasterize
        spec, _, _ = rasterize(config)
        keys = ('study_type', 'wavelength_um', 'resolution', 'duration_fs', 'pml_um', 'cladding_index', 'solver')
        return {k: config[k] for k in keys} | {k: spec[k] for k in ('grid','source_um','epsilon_xyz_sha256')}
    if config.get('study_type') == 'taper-3d':
        from .taper_geometry import taper_geometry
        keys = ('study_type', 'schema_version', 'wavelength_um', 'resolution', 'duration_fs', 'pml_um', 'cladding_index', 'solver')
        return {key: config[key] for key in keys} | {'physical_layout': taper_geometry(config)}
    geometry = config['geometry']
    objects = [{key: obj[key] for key in ('id', 'kind', 'position', 'size', 'rotation_deg', 'refractive_index')}
               for obj in geometry['objects']]
    return {key: config[key] for key in ('schema_version', 'study_type', 'wavelength_um', 'resolution', 'duration_fs', 'pml_um', 'cladding_index', 'solver')} | {
        'domain': geometry['domain'], 'objects': sorted(objects, key=lambda item: item['id'])}


def compare_runs(reference, candidate, directory):
    if reference['id'] == candidate['id']:
        raise ValueError('Choose two different completed 3D runs.')
    for job in (reference, candidate):
        if job['status'] != 'completed' or job['config'].get('study_type') not in ('waveguide-3d', 'taper-3d', 'layout-3d'):
            raise ValueError('Both comparison runs must be completed finite-height 3D studies.')
    if physics_spec(reference['config']) != physics_spec(candidate['config']):
        raise ValueError('Same-study comparison requires the same solver, geometry, materials, source settings, resolution and duration. Only the execution device and presentation settings may differ. No automatic rescaling or source calibration is applied.')
    metadata = []
    for job in (reference, candidate):
        saved = json.loads((directory/job['id']/'metadata.json').read_text())
        config = json.loads((directory/job['id']/'config.json').read_text())
        if saved.get('config') != config or config != job['config']:
            raise ValueError('Saved run configuration does not match its metadata.')
        metadata.append(saved)
    a, b = metadata
    for saved in metadata:
        if saved.get('dimensions') != '3D' or saved.get('field_components') != list(COMPONENTS) or saved.get('field_component') != 'Ey':
            raise ValueError('Saved metadata must identify 3D Ex/Ey/Ez fields with Ey as the primary component.')
    for key in ('versions', 'source', 'field_sampling', 'material_sampling', 'boundaries', 'field_units', 'array_order'):
        if not a.get(key) or not b.get(key):
            raise ValueError(f'Runs must record {key}; missing conventions cannot establish a matched comparison.')
        if a.get(key) != b.get(key):
            raise ValueError(f'Runs have different recorded {key}; a matched comparison needs the same numerical conventions.')
    ta = a.get('actual_duration_fs', a.get('simulated_time_fs'))
    tb = b.get('actual_duration_fs', b.get('simulated_time_fs'))
    if type(ta) not in (float, int) or type(tb) not in (float, int) or not np.isfinite([ta, tb]).all() or min(ta, tb) <= 0 or abs(ta-tb) > 1e-6:
        raise ValueError('Final recorded physical times differ. Comparing snapshots at different phases is not a repeatability test.')
    ra, ca = (stored_volume(directory/job['id']) for job in (reference, candidate))
    if any(not np.array_equal(ra[key], ca[key]) for key in ('x_um', 'y_um', 'z_um')):
        raise ValueError('Saved coordinate grids differ. This comparison never resamples or interpolates between runs.')
    if not np.array_equal(ra['epsilon'], ca['epsilon']):
        raise ValueError('Actual material grids differ; these are not identical discretized studies.')
    def metrics(left, right):
        left, right = left.astype(np.float64), right.astype(np.float64)
        delta = right-left
        denominator = float(np.linalg.norm(left.ravel()))
        return {'relative_l2': float(np.linalg.norm(delta.ravel())/denominator) if denominator else None,
                'max_abs_error': float(np.max(np.abs(delta))),
                'reference_peak_abs': float(np.max(np.abs(left)))}
    components = {component: metrics(ra['fields'][index], ca['fields'][index]) for index, component in enumerate(COMPONENTS)}
    return {'schema_version': 1, 'kind': 'same-study-repeatability', 'generated_at': datetime.now(timezone.utc).isoformat(),
            'reference_run_id': reference['id'], 'candidate_run_id': candidate['id'],
            'reference_target': f"{reference['config']['solver']} / {reference['config']['device']}",
            'candidate_target': f"{candidate['config']['solver']} / {candidate['config']['device']}",
            'components': components, 'overall_relative_l2': metrics(ra['fields'], ca['fields'])['relative_l2'],
            'epsilon_identical': True, 'coordinate_grids_identical': True, 'actual_time_fs': ta,
            'commercial_agreement': 'not_validated',
            'notes': ['Metrics describe differences between actual saved samples on identical grids at the same physical time.',
                      'Physical inputs, source conventions, numerical versions and material arrays were checked before comparing.',
                      'These are repeatability metrics. A small difference does not establish convergence or agreement with commercial software.',
                      'No arbitrary pass/fail threshold, phase fitting, amplitude normalization or resampling was applied.']}
