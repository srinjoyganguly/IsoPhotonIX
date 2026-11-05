"""Controller copied verbatim into standalone Python/notebook study bundles.

This file deliberately has no dependency on a running API or this repository.
Solver execution happens in a fresh, explicitly selected local interpreter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

BUNDLE = Path(__file__).resolve().parent


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def check_bundle():
    """Detect changed solver code or comparison data; edited CONFIG is allowed."""
    manifest = json.loads((BUNDLE / 'bundle-manifest.json').read_text())
    for name, record in manifest['files'].items():
        if name.startswith(('source/', 'reference/', 'design_source/')) or name in ('workbench_runtime.py', 'workbench_solver_environment.py', 'geometry_contract.py'):
            if _hash(BUNDLE / name) != record['sha256']:
                raise ValueError(f'Bundled source or reference file changed: {name}. Keep an original bundle for GUI reproduction.')
    return manifest


def compare_results(output, *, rtol=1e-6, atol=1e-12):
    """Compare numerical samples, not ZIP bytes, wall times or human labels.

    Boolean/integer arrays, coordinate axes and validity masks must match
    exactly. Physical fields, S and raw E/H retain strict pointwise tolerances.
    A separately reported amplitude-scale criterion is available only for the
    unused incoming Meep modal residual channels. Strict failures stay visible.
    """
    import numpy as np

    output, reference = Path(output), BUNDLE / 'reference'
    if not reference.exists():
        return {'status': 'no_reference', 'passed': None,
                'reason': 'This was exported before execution; no GUI result was supplied.'}
    manifest = check_bundle()
    expected_config = json.loads((reference / 'config.json').read_text())
    actual_config = json.loads((output / 'config.json').read_text())
    if actual_config != expected_config:
        return {'status': 'changed_configuration', 'passed': None,
                'reason': 'CONFIG was edited. This is a new experiment, not a replay of the saved GUI result.'}
    old_meta = json.loads((reference / 'metadata.json').read_text())
    new_meta = json.loads((output / 'metadata.json').read_text())
    versions_match = new_meta.get('versions') == old_meta.get('versions')
    old_implementation = old_meta.get('implementation_snapshot')
    new_implementation = new_meta.get('implementation_snapshot')
    implementation_preserved = None
    if old_implementation is not None:
        implementation_preserved = bool(isinstance(new_implementation, dict) and
            new_implementation.get('format') == old_implementation.get('format') and
            new_implementation.get('sha256') == old_implementation.get('sha256') and
            (output / 'implementation.zip').is_file() and
            _hash(output / 'implementation.zip') == old_implementation.get('sha256'))
    details, passed, exact = {}, True, True
    physical_passed, pointwise_passed, diagnostic_variations = True, True, []
    for original in sorted(reference.glob('*.npz')):
        path = output / original.name
        if not path.is_file():
            details[original.name] = {'passed': False, 'reason': 'Missing reproduced array file'}
            passed = exact = physical_passed = pointwise_passed = False
            continue
        with np.load(original, allow_pickle=False) as old, np.load(path, allow_pickle=False) as new:
            if set(old.files) != set(new.files):
                details[original.name] = {'passed': False, 'reason': 'Array names differ'}
                passed = exact = physical_passed = pointwise_passed = False
                continue
            records = {}
            for key in old.files:
                a, b = old[key], new[key]
                same_shape = a.shape == b.shape and a.dtype == b.dtype
                same = bool(same_shape and np.array_equal(a, b, equal_nan=True))
                categorical = a.dtype.kind in 'biu' or key.endswith(('_um', '_fs')) or 'valid' in key
                comparable = same_shape and a.dtype.kind in 'biufc'
                within = bool(comparable and (same if categorical else np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)))
                physical_within, qualified = within, within
                record = {'shape': list(a.shape), 'dtype': str(a.dtype), 'exact': same,
                          'passed': within, 'pointwise_passed': within,
                          'comparison': 'exact' if categorical else 'fixed pointwise numeric tolerances'}
                if comparable and a.dtype.kind in 'fc':
                    finite = np.isfinite(a) & np.isfinite(b)
                    delta = b[finite].astype(np.complex128) - a[finite].astype(np.complex128)
                    norm = float(np.linalg.norm(a[finite].astype(np.complex128)))
                    record.update(max_abs_error=float(np.max(np.abs(delta))) if delta.size else 0.,
                                  relative_l2=float(np.linalg.norm(delta) / norm) if norm else None,
                                  undefined_masks_equal=bool(np.array_equal(np.isnan(a), np.isnan(b))))
                # Only these Meep device-run incoming residual channels are
                # unused by S, incident normalization and reference-through.
                # Preserve the strict pointwise result; supplement it with a
                # separately reported bound in the reference array's amplitude
                # units. Outgoing channels and every raw E/H sample stay strict.
                diagnostic_column = (0 if key in ('port1_alpha', 'port1_raw_alpha') else
                                     1 if key in ('port2_alpha', 'port2_raw_alpha') else None)
                is_residual = (expected_config.get('schema_version') == 4 and
                    expected_config.get('study_type') == 'waveguide-ports-3d' and
                    expected_config['solver'] == 'meep' and original.name in
                    ('device-forward-monitors.npz', 'device-reverse-monitors.npz') and
                    diagnostic_column is not None and comparable and a.ndim == 2 and a.shape[1] == 2)
                if is_residual:
                    other = 1 - diagnostic_column
                    physical_within = bool(np.allclose(a[:, other], b[:, other], rtol=rtol, atol=atol, equal_nan=True))
                    residual_delta = b[:, diagnostic_column] - a[:, diagnostic_column]
                    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
                    peak, reference_norm = float(np.max(np.abs(a))), float(np.linalg.norm(a))
                    maximum = float(np.max(np.abs(residual_delta))) if finite else None
                    residual_norm = float(np.linalg.norm(residual_delta)) if finite else None
                    supplemental = bool(finite and versions_match and
                        maximum <= atol + rtol * peak and
                        residual_norm <= atol * math.sqrt(len(residual_delta)) + rtol * reference_norm)
                    qualified = physical_within and (within or supplemental)
                    residual_record = {'file': original.name, 'array': key, 'column': diagnostic_column,
                        'role': 'Unused incoming residual; excluded from published S, outgoing amplitudes, incident reference and reference-through.',
                        'pointwise_passed': within, 'amplitude_scaled_passed': supplemental,
                        'max_abs_error': maximum, 'residual_l2': residual_norm,
                        'reference_array_peak': peak, 'reference_array_l2': reference_norm,
                        'max_abs_bound': atol + rtol * peak,
                        'l2_bound': atol * math.sqrt(len(residual_delta)) + rtol * reference_norm,
                        'rule': 'Same declared rtol/atol; maximum residual and L2 error bounded using the original full two-direction coefficient array. No fit or phase alignment.'}
                    record['incoming_residual_diagnostic'] = residual_record
                    if not same:
                        diagnostic_variations.append(residual_record)
                record['reproduction_criteria_passed'] = bool(qualified)
                records[key] = record
                passed &= qualified
                pointwise_passed &= within
                physical_passed &= physical_within
                exact &= same
            details[original.name] = records
    requested_target_preserved = new_meta.get('config', {}).get('solver') == expected_config['solver'] and new_meta.get('config', {}).get('device') == expected_config['device']
    actual = str(new_meta.get('actual_device', '')).lower()
    actual_target_preserved = (actual.startswith(('cuda', 'gpu')) if expected_config['device'] == 'gpu' else actual.startswith('cpu'))
    recorded_times = {}
    for key in ('field_time_fs', 'simulated_time_fs', 'actual_duration_fs', 'time_steps', 'time_step_fs'):
        if key in old_meta:
            recorded_times[key] = {'reference': old_meta[key], 'reproduced': new_meta.get(key), 'exact': old_meta[key] == new_meta.get(key)}
    times_preserved = all(value['exact'] for value in recorded_times.values())
    execution_preserved = (requested_target_preserved and actual_target_preserved and times_preserved and
                           versions_match and implementation_preserved is not False)
    passed &= execution_preserved
    physical_passed &= execution_preserved
    result = {'status': 'compared', 'passed': bool(passed), 'all_arrays_exact': bool(exact),
              'physical_results_passed': bool(physical_passed), 'all_arrays_pointwise_passed': bool(pointwise_passed),
              'diagnostic_variations': diagnostic_variations,
              'rtol': rtol, 'atol': atol, 'requested_target_preserved': requested_target_preserved,
              'actual_target_preserved': actual_target_preserved, 'recorded_physical_times': recorded_times,
              'actual_device': new_meta.get('actual_device'), 'reference_actual_device': old_meta.get('actual_device'),
              'versions_match': versions_match,
              'snapshot_origin': manifest.get('snapshot_origin', 'export_time_compatibility'),
              'recorded_implementation_preserved': implementation_preserved,
              'arrays': details, 'excluded': ['wall-clock timings', 'NPZ compression bytes', 'human-readable labels'],
              'meaning': 'Replaying this implementation and configuration is separate from external accuracy or convergence.'}
    (output / 'reproduction-check.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def run(config, *, output=None, solver_python=None, timeout_seconds=1800, compare=True):
    """Validate editable settings and run the exact bundled GUI worker locally."""
    check_bundle()
    source = BUNDLE / 'source'
    sys.path.insert(0, str(source))
    from backend.study3d import parse_study
    validated = parse_study(config).model_dump(by_alias=True)
    solver, device = validated['solver'], validated['device']
    executable = solver_python
    if not executable:
        if __package__:
            from . import solver_environment
        else:
            solver_environment = _bundled_module('workbench_solver_environment')
        executable = solver_environment.solver_interpreter(solver, config['device'], solver_environment.default_user_state())
    if output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        output = BUNDLE / 'output' / f'{stamp}-{uuid.uuid4().hex[:8]}'
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(validated, indent=2, allow_nan=False) + '\n')
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', MPLCONFIGDIR=str(output / '.matplotlib'),
               MPLBACKEND='Agg', PYTHONPATH=str(source))
    if solver == 'fdtdx':
        env['JAX_PLATFORMS'] = 'cuda' if device == 'gpu' else 'cpu'
    command = [str(executable), '-m', 'backend.worker', str(output / 'config.json'), str(output)]
    print(f'Running {solver} on explicitly selected {device}; results: {output}', flush=True)
    print('A GPU request never retries on CPU. See worker.log for solver progress.', flush=True)
    with (output / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, cwd=BUNDLE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != 'nt')
        try:
            code = process.wait(timeout=timeout_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == 'nt':
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code or not (output / 'metadata.json').exists():
        raise RuntimeError(f'Solver failed (exit {code}); inspect {output / "worker.log"}. No fallback was attempted.')
    result = compare_results(output) if compare else {'status': 'not_requested', 'passed': None}
    print(json.dumps({k: v for k, v in result.items() if k != 'arrays'}, indent=2))
    if result.get('passed') is False:
        raise RuntimeError(f'Reproduction checks failed for numerical arrays, execution or versions. Inspect {output / "reproduction-check.json"}.')
    return output


def summarize(output):
    import numpy as np
    output = Path(output)
    metadata = json.loads((output / 'metadata.json').read_text())
    print('Actual device:', metadata.get('actual_device'))
    print('Solver versions:', metadata.get('versions'))
    for path in sorted(output.glob('*.npz')):
        with np.load(path, allow_pickle=False) as arrays:
            print(path.name, {key: list(arrays[key].shape) for key in arrays.files})
    return metadata


def plot_results(output):
    """Plot saved numerical samples without changing normalization or sampling."""
    import numpy as np
    import matplotlib.pyplot as plt
    output = Path(output)
    with np.load(output / 'arrays.npz', allow_pickle=False) as arrays:
        field, material = arrays['field'], arrays['epsilon']
        if field.ndim == 3:
            field, material = field[len(field) // 2], material[len(material) // 2]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        extent = [arrays['x_um'][0], arrays['x_um'][-1], arrays['y_um'][0], arrays['y_um'][-1]]
        peak = float(np.max(np.abs(field))) or 1.
        for ax, values, title, options in (
            (axes[0], field, 'Actual final field · central XY slice for 3D', {'cmap': 'RdBu_r', 'vmin': -peak, 'vmax': peak}),
            (axes[1], material, 'Actual sampled permittivity', {'cmap': 'viridis'})):
            image = ax.imshow(values, origin='lower', extent=extent, aspect='equal', **options)
            ax.set(xlabel='x (µm)', ylabel='y (µm)', title=title)
            fig.colorbar(image, ax=ax)
        fig.tight_layout()
    if (output / 'spectra.npz').exists():
        with np.load(output / 'spectra.npz', allow_pickle=False) as values:
            fig2, ax = plt.subplots(figsize=(8, 4))
            for key in ('reflectance', 'transmittance', 'uncollected_fraction'):
                ax.plot(values['wavelength_um'], np.where(values['valid_bin'], values[key], np.nan), '.-', label=key)
            ax.set(xlabel='Wavelength (µm)', ylabel='Incident-normalized power')
            ax.legend(); fig2.tight_layout()
    if (output / 'sparameters.npz').exists():
        with np.load(output / 'sparameters.npz', allow_pickle=False) as values:
            fig3, axes = plt.subplots(1, 2, figsize=(11, 4))
            valid = values['valid_bin'] & values['mode_valid']
            for outgoing in range(2):
                for incoming in range(2):
                    amplitude = values['s'][:, outgoing, incoming]
                    label = f'S{outgoing + 1}{incoming + 1}'
                    axes[0].plot(values['wavelength_um'], np.where(valid, abs(amplitude) ** 2, np.nan), '.-', label=label)
                    phase_valid = valid & (abs(amplitude) >= 1e-8)
                    axes[1].plot(values['wavelength_um'], np.where(phase_valid, np.angle(amplitude), np.nan), '.-', label=label)
            axes[0].set(xlabel='Wavelength (µm)', ylabel='Fundamental-mode power |S|²')
            axes[1].set(xlabel='Wavelength (µm)', ylabel='Principal phase (rad); |S| < 1e−8 masked')
            for ax in axes: ax.legend()
            fig3.tight_layout()
    return fig


def _bundled_module(name):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _geometry_contract():
    # Repository and standalone bundles use the same first-party validator.
    if __package__:
        from . import geometry_draft
        return geometry_draft
    return _bundled_module('geometry_contract')


def validate_geometry(value):
    """Validate the full versioned draft, including materials and component recipes."""
    return _geometry_contract().validate_geometry(value)


def evaluate_material(record, wavelength_um):
    """Evaluate the saved optical record; tabulated values never extrapolate."""
    return _geometry_contract().evaluate_material(record, wavelength_um)


def _design_module(name):
    import importlib
    if __package__:
        return importlib.import_module('.' + name, __package__)
    import importlib.util
    folder = BUNDLE / 'design_source'
    identity = hashlib.sha256(b''.join((folder / (part + '.py')).read_bytes()
        for part in ('component_library', 'geometry_draft', 'library_layout'))).hexdigest()[:16]
    package = 'photonics_design_' + identity
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, folder / '__init__.py', submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return importlib.import_module('.' + name, package)


def regenerate_component(geometry, component_id, *, parameters=None, placement=None, fit_domain=False):
    """Regenerate one instance using the bundled generator, preserving its neighbors.

    This creates a new design, not a reproduced electromagnetic result. All
    dimensions/topology/material references pass the same editor validation.
    """
    check_bundle()
    geometry = validate_geometry(geometry)
    instance = next((item for item in geometry.get('components', []) if item['id'] == component_id), None)
    if instance is None:
        raise ValueError('Choose a parametric component instance from this geometry.')
    settings = {**instance['parameters'], **(parameters or {})}
    return _design_module('library_layout').place_component(geometry, instance['family'], settings,
        placement or instance['placement'], instance['material_id'], component_id, fit_domain)['geometry']


def align_ports(geometry, moving_component, moving_port, fixed_component, fixed_port, *, fit_domain=False):
    """Align two named layout ports using the same rigid transform as the editor."""
    check_bundle()
    return _design_module('library_layout').align_ports(
        geometry, moving_component, moving_port, fixed_component, fixed_port, fit_domain)


def plot_geometry(geometry, *, show_hidden=False):
    """Render the exact draft solids; curved surfaces are visualization meshes."""
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, pathpatch_2d_to_3d
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    geometry = validate_geometry(geometry)
    fig = plt.figure(figsize=(10, 6)); ax = fig.add_subplot(111, projection='3d')
    for index, item in enumerate(geometry['objects']):
        if not show_hidden and not item['visible']: continue
        color = plt.cm.viridis(index / max(1, len(geometry['objects']) - 1))
        p = np.array([item['position'][a] for a in 'xyz'])
        h = item['size']['z'] / 2
        if item['kind'] == 'polygon':
            angle = math.radians(item['rotation_deg']); c, s = math.cos(angle), math.sin(angle)
            contours = []
            for contour in [item['vertices'], *item['holes']]:
                local = np.array([[point['x'], point['y']] for point in contour])
                contours.append(local @ np.array([[c, -s], [s, c]]).T + p[:2])
            vertices, codes, faces = [], [], []
            for contour in contours:
                vertices.extend([*contour, contour[0]])
                codes.extend([PlotPath.MOVETO, *([PlotPath.LINETO] * (len(contour)-1)), PlotPath.CLOSEPOLY])
                for a, b in zip(contour, np.roll(contour, -1, axis=0)):
                    faces.append([[*a, p[2]-h], [*b, p[2]-h], [*b, p[2]+h], [*a, p[2]+h]])
            # Compound paths retain actual holes and concave outlines. An
            # unconstrained point-cloud triangulation would bridge voids.
            for z in (p[2]-h, p[2]+h):
                patch = PathPatch(PlotPath(vertices, codes), facecolor=color, edgecolor=color, alpha=.35, linewidth=.25)
                ax.add_patch(patch)
                pathpatch_2d_to_3d(patch, z=z, zdir='z')
        elif item['kind'] == 'box':
            half = np.array([item['size'][a] for a in 'xyz']) / 2
            corners = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]) * half
            angle = math.radians(item['rotation_deg']); c, s = math.cos(angle), math.sin(angle)
            corners = corners @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + p
            faces = [[corners[i] for i in ids] for ids in ([0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5])]
        else:
            theta = np.linspace(0, 2 * np.pi, 65)
            radii = [item['radius'], item['inner_radius'] if item['kind'] == 'ring' else 0.]
            rings = [np.stack([p[0] + radius * np.cos(theta), p[1] + radius * np.sin(theta), np.full_like(theta, p[2] + z)], axis=-1) for radius in radii for z in (-h, h)]
            faces = []
            for j in range(len(theta) - 1):
                for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
                    faces.append([rings[a][j], rings[a][j + 1], rings[b][j + 1], rings[b][j]])
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=.35, linewidths=.25, edgecolors=color))
        ax.text(*p, item['name'], fontsize=8, parse_math=False)
    visible_ids = {item['id'] for item in geometry['objects'] if show_hidden or item['visible']}
    for component in geometry.get('components', []):
        if not visible_ids.intersection(component['object_ids']):
            continue
        for port in component['ports']:
            angle = math.radians(port['angle_deg'])
            length = max(.15, min(port['width_um'], .7))
            ax.quiver(port['x'], port['y'], port['z'], math.cos(angle)*length,
                      math.sin(angle)*length, 0, color='#149b86', arrow_length_ratio=.3)
            ax.text(port['x'], port['y'], port['z'], port['name'], fontsize=7, parse_math=False)
    for axis in 'xyz': getattr(ax, f'set_{axis}lim')(-geometry['domain'][axis] / 2, geometry['domain'][axis] / 2)
    ax.set_box_aspect([geometry['domain'][a] for a in 'xyz'])
    ax.set(xlabel='x (µm)', ylabel='y (µm)', zlabel='z (µm)')
    ax.set_title(geometry['name'] + ' · geometry only, no electromagnetic result', parse_math=False)
    fig.tight_layout()
    return fig


def main(config):
    parser = argparse.ArgumentParser(description='Reproduce an exported local photonics study with the bundled solver implementation.')
    parser.add_argument('--solver-python', help='Interpreter containing the selected solver; defaults to SOLVER_PYTHON or this Python.')
    parser.add_argument('--output', type=Path, help='New output directory; never overwrites a previous run.')
    parser.add_argument('--timeout', type=int, default=1800, help='Wall-clock cap in seconds; does not alter physical simulation time.')
    args = parser.parse_args()
    output = run(config, output=args.output, solver_python=args.solver_python, timeout_seconds=args.timeout)
    summarize(output)
