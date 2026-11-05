"""Local full-vector FDTDX evolution for the bounded finite-height taper study.

The core fills the cells whose centres lie inside the taper or its access
leads. Contiguous columns with the same cross-section form a single native
FDTDX material block. This is a staircase discretization of a linear contour.
"""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import time

from .fdtdx_adapter import _runtime
from .fdtdx3d_adapter import cell_center_fields


def raster_layout(profile, resolution, pml_um):
    """Translate a validated continuous profile into grid-aligned material runs."""
    import numpy as np

    dx = 1. / resolution
    shape = tuple(round(profile['domain'][axis] * resolution) for axis in 'xyz')
    if min(shape) < 2 or np.prod(shape) > 1_200_000:
        raise ValueError('Taper grid exceeds the allowed 3D cell budget.')
    x, y, z = [(np.arange(n) + .5 - n / 2) * dx for n in shape]
    width = np.interp(x, [profile['x_in'], profile['x_out']],
                      [profile['width_in'], profile['width_out']])
    # Include cell centres on the contour. Use a grid-scaled tolerance only
    # for floating-point equality, including symmetric half-cell boundaries.
    tolerance = dx * 1e-9
    inside_y = np.abs(y[np.newaxis, :]) <= width[:, np.newaxis] / 2 + tolerance
    inside_z = np.abs(z) <= profile['height'] / 2 + tolerance
    widths = np.count_nonzero(inside_y, axis=1)
    if widths.min() < 4 or np.count_nonzero(inside_z) < 4:
        raise ValueError('The taper and access leads need at least four occupied cells across width and height.')
    y_starts = np.argmax(inside_y, axis=1)
    z_start = int(np.flatnonzero(inside_z)[0])
    z_count = int(np.count_nonzero(inside_z))
    runs = []
    start_x = 0
    for end_x in range(1, shape[0] + 1):
        if end_x < shape[0] and y_starts[end_x] == y_starts[start_x] and widths[end_x] == widths[start_x]:
            continue
        runs.append(((start_x, int(y_starts[start_x]), z_start),
                     (end_x - start_x, int(widths[start_x]), z_count)))
        start_x = end_x
    pml_cells = round(pml_um * resolution)
    # The E/H sheet lies at a staggered-grid x index. The analytic core is
    # constant across this entire lead, so its aperture equals the local core.
    source_ix = round(profile['source_x'] * resolution + shape[0] / 2)
    source_x = (source_ix - shape[0] / 2) * dx
    if source_ix <= pml_cells or source_ix >= shape[0] - pml_cells or source_x + dx >= profile['x_in']:
        raise ValueError('Source must fit in the input lead outside its PML.')
    source_start = (source_ix, int(y_starts[source_ix]), z_start)
    source_shape = (1, int(widths[source_ix]), z_count)
    return {'shape': shape, 'dx': dx, 'centres': (x, y, z), 'runs': runs,
            'source_start': source_start, 'source_shape': source_shape, 'source_x': source_x,
            'pml_cells': pml_cells, 'occupied': inside_y[:, :, None] & inside_z[None, None, :]}


def _material_objects(fdtdx, layout, index, constraints):
    material = fdtdx.Material(permittivity=index**2)
    solids = []
    for number, (start, extent) in enumerate(layout['runs']):
        obj = fdtdx.UniformMaterialObject(name=f'core-{number}', partial_grid_shape=extent, material=material)
        solids.append(obj)
        constraints.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=('-', '-', '-'), coordinates=start))
    return solids


def _save_fields(output, layout, arrays):
    import numpy as np

    raw_fields = np.asarray(arrays.fields.E).transpose(0, 3, 2, 1).copy()
    raw_eps = np.asarray(1. / arrays.inv_permittivities[0]).transpose(2, 1, 0).copy()
    if not np.isfinite(raw_fields).all() or not np.isfinite(raw_eps).all():
        raise RuntimeError('FDTDX returned non-finite field or material samples.')
    fields = cell_center_fields(raw_fields)
    if not np.isfinite(fields).all() or not np.any(fields[1]):
        raise RuntimeError('FDTDX returned invalid displayed fields or a zero Ey field.')
    axes = {f'{a}_um': values[:-1] for a, values in zip('xyz', layout['centres'])}
    raw_axes = {f'raw_{a}_um': values - layout['dx'] / 2 for a, values in zip('xyz', layout['centres'])}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'arrays.npz', field=fields[1], fields=fields,
                        epsilon=raw_eps[:-1, :-1, :-1], raw_fields=raw_fields, raw_epsilon=raw_eps,
                        **axes, **raw_axes)
    peaks = {name: float(np.max(np.abs(component))) for name, component in zip(('Ex', 'Ey', 'Ez'), fields)}
    return list(fields.shape[1:]), peaks


def run(config: dict, output_dir: Path) -> dict:
    import numpy as np
    from ..taper_geometry import taper_geometry

    started = time.perf_counter()
    if (config.get('schema_version'), config.get('study_type'), config.get('solver')) != (5, 'taper-3d', 'fdtdx'):
        raise ValueError('This adapter requires a version 5 FDTDX taper study.')
    profile = taper_geometry(config)
    layout = raster_layout(profile, config['resolution'], config['pml_um'])
    requested = config['device']
    jax, jnp, fdtdx, target = _runtime(requested)
    print(f'FDTDX taper: verified {target.device_kind} ({target})', flush=True)
    simulation = fdtdx.SimulationConfig(time=config['duration_fs'] * 1e-15,
        resolution=layout['dx'] * 1e-6, backend=requested, dtype=jnp.float32, courant_factor=.99)
    if simulation.backend != requested:
        raise RuntimeError(f'FDTDX selected {simulation.backend} for requested {requested}; fallback refused.')
    volume = fdtdx.SimulationVolume(name='domain', partial_grid_shape=layout['shape'],
        material=fdtdx.Material(permittivity=config['cladding_index']**2))
    pml = layout['pml_cells']
    boundary_config = fdtdx.BoundaryConfig(**{f'thickness_grid_{s}{a}': pml for s in ('min','max') for a in 'xyz'},
        **{f'boundary_type_{s}{a}': 'pml' for s in ('min','max') for a in 'xyz'})
    boundaries, constraints = fdtdx.boundary_objects_from_config(boundary_config, volume)
    constraints = list(constraints)
    solids = _material_objects(fdtdx, layout, profile['core_index'], constraints)
    source = fdtdx.UniformPlaneSource(name='input', partial_grid_shape=layout['source_shape'],
        fixed_E_polarization_vector=(0, 1, 0), direction='+', normalize_by_energy=False,
        temporal_profile=fdtdx.SingleFrequencyProfile(num_startup_periods=4),
        wave_character=fdtdx.WaveCharacter(wavelength=config['wavelength_um'] * 1e-6))
    constraints.append(source.set_grid_coordinates(axes=(0, 1, 2), sides=('-', '-', '-'),
                                                    coordinates=layout['source_start']))
    key = jax.random.PRNGKey(0)
    objects, arrays, parameters, simulation, _ = fdtdx.place_objects(
        object_list=[volume, *solids, source, *boundaries.values()], config=simulation, constraints=constraints, key=key)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, parameters, key)
    jax.block_until_ready(arrays)
    if arrays.fields.E.device != target:
        raise RuntimeError('FDTDX input fields were placed on the wrong execution device.')
    actual_epsilon = np.asarray(1. / arrays.inv_permittivities[0])
    expected = np.where(layout['occupied'], profile['core_index']**2, config['cladding_index']**2)
    if actual_epsilon.shape != expected.shape or not np.allclose(actual_epsilon, expected, atol=1e-6, rtol=1e-6):
        raise RuntimeError('FDTDX material grid differs from the requested taper and its access leads.')
    material_error = float(np.max(np.abs(actual_epsilon - expected)))
    setup_s = time.perf_counter() - started
    print(f'FDTDX taper: verified {layout["shape"]} cells; compiling {simulation.time_steps_total} steps', flush=True)
    compile_start = time.perf_counter()
    compiled = jax.jit(lambda a: fdtdx.run_fdtd(arrays=a, objects=objects, config=simulation,
                                                key=key, show_progress=False)).lower(arrays).compile()
    compile_s = time.perf_counter() - compile_start
    compute_start = time.perf_counter()
    final_step, final_arrays = compiled(arrays)
    jax.block_until_ready(final_arrays)
    elapsed = time.perf_counter() - compute_start
    actual = final_arrays.fields.E.device
    if actual != target:
        raise RuntimeError('FDTDX result fields were placed on the wrong execution device.')
    field_shape, peaks = _save_fields(Path(output_dir), layout, final_arrays)
    print(f'FDTDX taper: saved actual field volume {field_shape}; peak |Ey| {peaks["Ey"]:.6g}', flush=True)
    return {'solver': 'fdtdx', 'device': requested, 'actual_device': str(actual), 'device_name': actual.device_kind,
            'versions': {'fdtdx': importlib.metadata.version('fdtdx'), 'jax': jax.__version__,
                         'jaxlib': importlib.metadata.version('jaxlib'), 'numpy': np.__version__},
            'dimensions': '3D', 'field_component': 'Ey', 'field_components': ['Ex','Ey','Ez'],
            'field_units': 'arbitrary source units', 'array_order': 'fields[component,z,y,x]; field and epsilon[z,y,x]',
            'field_sampling': 'Four neighbouring Yee field samples averaged to each cell centre. The positive outermost layer on each axis is omitted from the displayed volume. Raw staggered fields are retained.',
            'material_sampling': 'Scalar staircase permittivity evaluated at voxel centres across the finite taper and both access leads.',
            'grid_shape': list(layout['shape']), 'grid_spacing_um': layout['dx'], 'field_shape': field_shape,
            'precision': 'float32', 'field_peaks': peaks, 'time_steps': int(final_step),
            'time_step_fs': simulation.time_step_duration * 1e15,
            'simulated_time_fs': int(final_step) * simulation.time_step_duration * 1e15,
            'source': {'type': 'finite uniform E/H sheet directed +x', 'component': 'Ey',
                       'temporal_profile': 'sinusoid with four-period linear turn-on', 'normalization': 'unit polarization',
                       'requested_x_um': profile['source_x'], 'x_um': layout['source_x'],
                       'grid_start_xyz': list(layout['source_start']), 'grid_shape_xyz': list(layout['source_shape']),
                       'wavelength_um': config['wavelength_um']},
            'boundaries': {'x': 'PML', 'y': 'PML', 'z': 'PML', 'pml_cells': pml, 'actual_pml_um': pml*layout['dx']},
            'geometry_translation': {'profile': profile, 'staircase_blocks': len(solids)},
            'material_grid_check': {'passed': True, 'maximum_absolute_error': material_error,
                                    'checked_voxels': int(expected.size), 'core_voxels': int(layout['occupied'].sum())},
            'assumptions': ['Lossless isotropic dielectric and homogeneous cladding.',
                            'Continuous linear taper between straight input and output leads extending through the longitudinal PML.',
                            'Source is a finite driven sheet. This study reports final-time fields only.'],
            'raw_coordinate_convention': 'raw_x/y/z_um mark voxel origins. Add the component Yee offsets times grid_spacing_um to locate raw_fields.',
            'yee_component_offsets_cells': {'Ex': [.5,0,0], 'Ey': [0,.5,0], 'Ez': [0,0,.5]},
            'setup_seconds': setup_s, 'compile_seconds': compile_s, 'compute_seconds': elapsed,
            'total_seconds': time.perf_counter() - started}
