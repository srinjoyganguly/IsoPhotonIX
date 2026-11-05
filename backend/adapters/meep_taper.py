"""Meep CPU field snapshot for a finite-height polygon taper and its two leads.

Prism vertices are on the bottom face in world coordinates. Omitting ``center``
keeps their coordinates unchanged during extrusion. Meep documents this rule at
https://meep.readthedocs.io/en/latest/Python_User_Interface/#prism .
"""
from __future__ import annotations

import math
import platform
import time
from pathlib import Path

from ..taper_geometry import taper_geometry
from ..volume import validate_volume
from .meep3d_adapter import C_UM_PER_FS


def run(config: dict, output_dir: Path) -> dict:
    if config.get('device') != 'cpu':
        raise ValueError('Meep supports CPU only. Select CPU explicitly.')
    if config.get('solver') != 'meep' or config.get('study_type') != 'taper-3d' or config.get('schema_version') != 5:
        raise ValueError('Expected a schema-version-5 Meep finite-height taper study.')
    model = taper_geometry(config)
    started = time.perf_counter()
    import meep as mp
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mp.verbosity(0)
    domain = model['domain']
    cell = mp.Vector3(*(domain[a] for a in 'xyz'))
    height = model['height']
    core = mp.Medium(index=model['core_index'])
    half_length = domain['x'] / 2
    leads = [(-half_length, model['x_in'], model['width_in']),
             (model['x_out'], half_length, model['width_out'])]
    # All three regions are one dielectric. Independent touching solids cause
    # Meep's subpixel averaging to introduce a seam even in a uniform guide.
    # Extrude their exact union, retaining the taper vertices where width varies.
    lower_in, lower_out, upper_out, upper_in = model['vertices']
    outline = [(-half_length, lower_in[1]), lower_in, lower_out, (half_length, lower_out[1]),
               (half_length, upper_out[1]), upper_out, upper_in, (-half_length, upper_in[1])]
    if model['width_in'] == model['width_out']:
        outline = [outline[0], outline[3], outline[4], outline[7]]
    prism = mp.Prism(vertices=[mp.Vector3(x, y, -height / 2) for x, y in outline],
                     height=height, material=core)
    wavelength = config['wavelength_um']
    resolution = config['resolution']

    def source_profile(t):
        return -math.cos(2 * math.pi * t / wavelength) * min(1., max(0., t / (4 * wavelength)))

    sim = mp.Simulation(
        cell_size=cell, dimensions=3, resolution=resolution, Courant=.5,
        default_material=mp.Medium(index=config['cladding_index']), geometry=[prism],
        boundary_layers=[mp.PML(config['pml_um'])], sources=[mp.Source(
            mp.CustomSource(source_profile, start_time=0, center_frequency=1 / wavelength),
            component=mp.Ey, center=mp.Vector3(model['source_x'], 0, 0),
            size=mp.Vector3(0, model['width_in'], height))],
    )
    print('Meep: initializing the finite-height polygonal taper and input/output access guides.', flush=True)
    try:
        sim.init_sim()
        initialized = time.perf_counter()
        sim.run(until=config['duration_fs'] * C_UM_PER_FS)
        solved = time.perf_counter()
        volume = mp.Volume(center=mp.Vector3(), size=cell)
        fields = np.stack([np.asarray(sim.get_array(component=c, vol=volume), dtype=np.float32).transpose(2, 1, 0)
                           for c in (mp.Ex, mp.Ey, mp.Ez)])
        epsilon = np.asarray(sim.get_array(component=mp.Dielectric, vol=volume), dtype=np.float32).transpose(2, 1, 0)
        x, y, z, _ = sim.get_array_metadata(vol=volume)
        arrays = validate_volume({'fields': fields, 'field': fields[1], 'epsilon': epsilon,
                                  'x_um': np.asarray(x), 'y_um': np.asarray(y), 'z_um': np.asarray(z)})
        actual_duration_fs = float(sim.meep_time() / C_UM_PER_FS)
        steps = int(sim.timestep())
        np.savez_compressed(output_dir / 'arrays.npz', **arrays)
    finally:
        sim.reset_meep()
    finished = time.perf_counter()
    peak = float(np.max(np.abs(fields[1])))
    print(f'Meep: saved final taper volume {epsilon.shape} [z,y,x], peak |Ey| {peak:.6g}.', flush=True)
    return {
        'solver': 'meep', 'device': 'cpu', 'actual_device': 'CPU', 'device_name': 'CPU',
        'versions': {'meep': mp.__version__, 'numpy': np.__version__, 'python': platform.python_version()},
        'dimensions': '3D', 'field_component': 'Ey', 'field_components': ['Ex', 'Ey', 'Ez'],
        'array_order': 'fields[component,z,y,x]; field and epsilon[z,y,x]',
        'field_units': 'arbitrary electric-current source amplitude', 'frame_count': 0,
        'field_shape': list(epsilon.shape), 'grid_shape': [len(x), len(y), len(z)], 'field_max_abs': peak,
        'actual_duration_fs': actual_duration_fs, 'time_steps': steps, 'time_step_fs': .5 / resolution / C_UM_PER_FS,
        'grid_spacing_um': 1 / resolution, 'courant_factor': .5,
        'precision': 'Meep float64 evolution, exported electric fields and permittivity float32',
        'timings': {'initialization_s': initialized - started, 'compute_s': solved - initialized,
                    'export_s': finished - solved, 'total_s': finished - started},
        'source': {'type': 'bidirectional finite electric current sheet, ramped continuous wave', 'component': 'Ey',
                   'x_um': model['source_x'], 'y_um': 0., 'z_um': 0., 'width_um': model['width_in'],
                   'height_um': height, 'wavelength_um': wavelength, 'amplitude': 1., 'startup_periods': 4,
                   'temporal_profile': '-cos(2*pi*t/T) * clip(t/(4*T),0,1)'},
        'taper': {'world_vertices_um': model['vertices'], 'full_outline_with_leads_um': outline,
                  'z_min_um': -height / 2, 'z_max_um': height / 2,
                  'core_index': model['core_index'], 'cladding_index': config['cladding_index'],
                  'leads': [{'x_min_um': x0, 'x_max_um': x1, 'width_um': w} for x0, x1, w in leads]},
        'boundaries': {'x': 'PML both faces', 'y': 'PML both faces', 'z': 'PML both faces', 'pml_um': config['pml_um']},
        'material_sampling': 'One native Meep Prism covers the exact union of the taper and its leads with subpixel smoothing. Dielectric output reports the harmonic mean of permittivity eigenvalues.',
        'field_sampling': 'Final solver Ex/Ey/Ez sampled on the common output grid with Meep get_array.',
        'assumptions': ['Uniform, lossless, isotropic core and cladding. The two access guides reach the X boundaries.',
                        'All six faces have PML. The saved volume includes those absorbing regions.',
                        'Finite current sheet launches bidirectional radiation. The snapshot measures fields only.'],
    }
