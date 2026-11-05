"""Physical translation shared by the Meep and FDTDX taper field adapters."""
import math

from .component_library import build_component
from .geometry_draft import validate_geometry


def taper_geometry(config):
    """Check the entire saved scene and resolve its taper, access leads and cell.

    The editor's box bounds the design. The solver cell may grow to include
    straight access guides, free cladding and the six absorbing boundaries.
    Return the actual saved contour after checking it against its recipe.
    """
    geometry = validate_geometry(config['geometry'])
    if (geometry['schema_version'] != 2 or geometry['mode'] != 'draft-3d' or
            len(geometry['objects']) != 1 or len(geometry['components']) != 1):
        raise ValueError('Open one linear taper as a new design for this field study.')
    instance = geometry['components'][0]
    solid = geometry['objects'][0]
    if instance['family'] != 'taper' or instance['object_ids'] != [solid['id']]:
        raise ValueError('The taper study requires one intact linear taper instance.')
    if any(instance['placement'].values()) or any(solid['position'].values()) or solid['rotation_deg']:
        raise ValueError('Place the taper at the origin with zero rotation for this study.')
    recipe = build_component('taper', parameters=instance['parameters'])
    reference = recipe['geometry']['objects'][0]
    if (instance['parameters'] != recipe['component']['parameters'] or
            instance['ports'] != recipe['component']['ports'] or
            solid['kind'] != 'polygon' or solid['holes'] or solid['vertices'] != reference['vertices'] or
            solid['size']['z'] != reference['size']['z']):
        raise ValueError('The saved solid or port geometry differs from the linear taper recipe. Regenerate the taper before running.')
    if solid.get('material_id') != instance['material_id']:
        raise ValueError('The taper and its solid must use the same optical material.')
    material = next(m for m in geometry['materials'] if m['id'] == instance['material_id'])
    if material['model'] != 'constant-nk' or material['k'] != 0:
        raise ValueError('The taper field study requires a constant-index material with k = 0.')
    if not config['cladding_index'] < material['n'] <= 5:
        raise ValueError('Choose a core index above the cladding index and at most 5.')
    p = instance['parameters']
    w_in, w_out = p['input_width_um'], p['output_width_um']
    height = solid['size']['z']
    resolution, pml = config['resolution'], config['pml_um']
    if min(w_in, w_out, height) * resolution < 4:
        raise ValueError('Both taper widths and its height must each span at least four grid cells. Increase the resolution or dimensions.')
    if pml * resolution < 8 or config['wavelength_um'] * resolution / material['n'] < 10:
        raise ValueError('Use at least eight cells per PML and ten cells per core wavelength.')
    # Centre the device on a common integer grid. Grow only the simulation cell.
    required = {'x': p['length_um'] + 2 * (pml + 1.5),
                'y': max(3., max(w_in, w_out) + 2 * (pml + .6)),
                'z': max(3., height + 2 * (pml + .6))}
    counts = {axis: math.ceil(max(geometry['domain'][axis], required[axis]) * resolution - 1e-10)
              for axis in 'xyz'}
    domain = {axis: count / resolution for axis, count in counts.items()}
    return {'domain': domain, 'vertices': [[v['x'], v['y']] for v in solid['vertices']],
            'x_in': instance['ports'][0]['x'], 'x_out': instance['ports'][1]['x'],
            'width_in': w_in, 'width_out': w_out, 'height': height, 'core_index': material['n'],
            'source_x': -domain['x'] / 2 + pml + .5}
