"""Pure-Python draft geometry/material contract, independent of any EM solver.

Version 1 remains the original primitive format. Version 2 adds polygon solids,
material records and parametric component provenance. These are design data,
not a promise that an electromagnetic adapter supports the complete model.
"""
from __future__ import annotations

import math
import re

EPS = 1e-9
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z')
BASE_KEYS = {'schema', 'schema_version', 'units', 'name', 'mode', 'domain', 'objects'}
SOLID_KEYS = {'id', 'name', 'visible', 'kind', 'position', 'size', 'radius', 'inner_radius', 'rotation_deg', 'refractive_index'}


def _record(value, required, optional=(), label='Record'):
    if not isinstance(value, dict) or not set(required) <= value.keys() or not value.keys() <= set(required) | set(optional):
        raise ValueError(f'{label} fields do not match the versioned geometry format.')
    return value


def _number(value, low=-math.inf, high=math.inf, label='Value', *, positive=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or not low <= value <= high or (positive and value <= 0):
        raise ValueError(f'{label} must be a finite number in the supported range.')
    return 0 if value == 0 else value


def _name(value, label='Name'):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 80 or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f'{label} must contain 1–80 printable characters.')
    return value.strip()


def _text(value, maximum, label):
    if not isinstance(value, str) or len(value) > maximum or any((ord(c) < 32 and c not in '\n\r\t') or ord(c) == 127 for c in value):
        raise ValueError(f'{label} must be text of at most {maximum} characters.')
    return value


def _id(value, label='Identifier'):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError(f'{label} must use 1–80 letters, digits, underscores or hyphens, starting with a letter or digit.')
    return value


def _vector(value, low, high, label='Vector'):
    _record(value, set('xyz'), label=label)
    return {a: _number(value[a], low, high, f'{label}.{a}') for a in 'xyz'}


def _angle(value, label='Angle'):
    return _number(value, -360, 360, label) % 360


def validate_material(value):
    """Validate passive scalar optical design data; no implicit fit/extrapolation."""
    if not isinstance(value, dict):
        raise ValueError('Material must be a record.')
    common = {'id', 'name', 'model', 'reference_wavelength_um', 'source', 'notes'}
    model = value.get('model')
    if model not in ('constant-nk', 'tabulated-nk'):
        raise ValueError('Material model must be constant-nk or tabulated-nk; gain and other models are not implemented.')
    _record(value, common | ({'n', 'k'} if model == 'constant-nk' else {'wavelength_um', 'n', 'k'}), label='Material')
    result = {'id': _id(value['id'], 'Material id'), 'name': _name(value['name'], 'Material name'), 'model': model,
              'reference_wavelength_um': _number(value['reference_wavelength_um'], 0, math.inf, 'Reference wavelength', positive=True),
              'source': _text(value['source'], 1024, 'Material source'), 'notes': _text(value['notes'], 2000, 'Material notes')}
    if model == 'constant-nk':
        result.update(n=_number(value['n'], 0, 20, 'Refractive index n', positive=True),
                      k=_number(value['k'], 0, 20, 'Extinction coefficient k'))
    else:
        waves, n, k = value['wavelength_um'], value['n'], value['k']
        if not all(isinstance(a, list) for a in (waves, n, k)) or not 2 <= len(waves) <= 2048 or len(n) != len(waves) or len(k) != len(waves):
            raise ValueError('Tabulated n/k requires 2–2048 matching wavelength, n and k samples.')
        waves = [_number(w, 0, math.inf, 'Material wavelength', positive=True) for w in waves]
        if any(b <= a for a, b in zip(waves, waves[1:])):
            raise ValueError('Material wavelengths must be strictly increasing.')
        if not waves[0] <= result['reference_wavelength_um'] <= waves[-1]:
            raise ValueError('Material reference wavelength is outside the table; extrapolation is not allowed.')
        result.update(wavelength_um=waves, n=[_number(v, 0, 20, 'Refractive index n', positive=True) for v in n],
                      k=[_number(v, 0, 20, 'Extinction coefficient k') for v in k])
    return result


def evaluate_material(value, wavelength_um):
    """Return n/k at a wavelength, linearly interpolating only inside tables."""
    material = validate_material(value)
    wavelength = _number(wavelength_um, 0, math.inf, 'Wavelength', positive=True)
    if material['model'] == 'constant-nk':
        return {key: material[key] for key in ('n', 'k')}
    waves = material['wavelength_um']
    if not waves[0] <= wavelength <= waves[-1]:
        raise ValueError('Wavelength is outside the material table; extrapolation is not allowed.')
    for i in range(len(waves) - 1):
        if wavelength <= waves[i + 1]:
            fraction = (wavelength - waves[i]) / (waves[i + 1] - waves[i])
            return {key: material[key][i] + fraction * (material[key][i + 1] - material[key][i]) for key in ('n', 'k')}
    raise ValueError('Material interpolation failed.')


def _cross(a, b, c):
    return (b['x'] - a['x']) * (c['y'] - a['y']) - (b['y'] - a['y']) * (c['x'] - a['x'])


def _on_segment(p, a, b):
    scale = max(1., math.hypot(b['x'] - a['x'], b['y'] - a['y']))
    return abs(_cross(a, b, p)) <= EPS * scale and min(a['x'], b['x']) - EPS <= p['x'] <= max(a['x'], b['x']) + EPS and min(a['y'], b['y']) - EPS <= p['y'] <= max(a['y'], b['y']) + EPS


def _intersects(a, b, c, d):
    if max(a['x'], b['x']) < min(c['x'], d['x']) - EPS or max(c['x'], d['x']) < min(a['x'], b['x']) - EPS or max(a['y'], b['y']) < min(c['y'], d['y']) - EPS or max(c['y'], d['y']) < min(a['y'], b['y']) - EPS:
        return False
    if any((_on_segment(c, a, b), _on_segment(d, a, b), _on_segment(a, c, d), _on_segment(b, c, d))):
        return True
    return (_cross(a, b, c) > 0) != (_cross(a, b, d) > 0) and (_cross(c, d, a) > 0) != (_cross(c, d, b) > 0)


def _edges(contour):
    return zip(contour, contour[1:] + contour[:1])


def _inside(point, contour):
    inside = False
    for a, b in _edges(contour):
        if _on_segment(point, a, b):
            return False
        if (a['y'] > point['y']) != (b['y'] > point['y']):
            x = a['x'] + (point['y'] - a['y']) * (b['x'] - a['x']) / (b['y'] - a['y'])
            if point['x'] < x:
                inside = not inside
    return inside


def _contour(value, label, ccw):
    if not isinstance(value, list) or not 3 <= len(value) <= 256:
        raise ValueError(f'{label} must contain 3–256 XY vertices without a repeated closing point.')
    points = []
    for p in value:
        _record(p, set('xy'), label=f'{label} vertex')
        points.append({a: _number(p[a], -1000, 1000, f'{label} vertex {a}') for a in 'xy'})
    for a, b in _edges(points):
        if math.hypot(b['x'] - a['x'], b['y'] - a['y']) <= EPS:
            raise ValueError(f'{label} has a zero-length or repeated edge.')
    n = len(points)
    for i in range(n):
        a, b, c = points[i - 1], points[i], points[(i + 1) % n]
        if abs(_cross(a, b, c)) <= EPS and (a['x'] - b['x']) * (c['x'] - b['x']) + (a['y'] - b['y']) * (c['y'] - b['y']) > EPS:
            raise ValueError(f'{label} doubles back along an adjacent edge.')
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _intersects(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                raise ValueError(f'{label} self-intersects or touches itself.')
    area2 = sum(a['x'] * b['y'] - b['x'] * a['y'] for a, b in _edges(points))
    if abs(area2) <= EPS:
        raise ValueError(f'{label} must have nonzero area.')
    if (area2 > 0) != ccw:
        points.reverse()
    return points


def validate_polygon(vertices, holes):
    outer = _contour(vertices, 'Polygon outline', True)
    if not isinstance(holes, list) or len(holes) > 32:
        raise ValueError('Polygon holes must be an array of at most 32 contours.')
    if len(outer) + sum(len(h) if isinstance(h, list) else 1025 for h in holes) > 1024:
        raise ValueError('A polygon may contain at most 1024 total vertices.')
    rings = [_contour(h, f'Polygon hole {i + 1}', False) for i, h in enumerate(holes)]
    for i, ring in enumerate(rings):
        if not all(_inside(p, outer) for p in ring) or any(_intersects(a, b, c, d) for a, b in _edges(ring) for c, d in _edges(outer)):
            raise ValueError('Polygon holes must lie strictly inside the outline without touching it.')
        for other in rings[:i]:
            if _inside(ring[0], other) or _inside(other[0], ring) or any(_intersects(a, b, c, d) for a, b in _edges(ring) for c, d in _edges(other)):
                raise ValueError('Polygon holes may not overlap, nest or touch.')
    return outer, rings


def geometry_bounds(item):
    """World-space bounds; off-centre polygons use their actual rotated points."""
    if item['kind'] == 'polygon':
        angle = math.radians(item['rotation_deg']); c, s = math.cos(angle), math.sin(angle)
        xy = [(p['x'] * c - p['y'] * s, p['x'] * s + p['y'] * c) for p in item['vertices']]
        low = {'x': min(p[0] for p in xy), 'y': min(p[1] for p in xy), 'z': -item['size']['z'] / 2}
        high = {'x': max(p[0] for p in xy), 'y': max(p[1] for p in xy), 'z': item['size']['z'] / 2}
    else:
        half = {'x': item['radius'], 'y': item['radius'], 'z': item['size']['z'] / 2}
        if item['kind'] == 'box':
            angle = math.radians(item['rotation_deg']); c, s = abs(math.cos(angle)), abs(math.sin(angle))
            half.update(x=(item['size']['x'] * c + item['size']['y'] * s) / 2, y=(item['size']['x'] * s + item['size']['y'] * c) / 2)
        low, high = {a: -half[a] for a in 'xyz'}, half
    return {'min': {a: item['position'][a] + low[a] for a in 'xyz'}, 'max': {a: item['position'][a] + high[a] for a in 'xyz'}}


def validate_geometry(value):
    """Validate both draft schemas without coercion or dropping unsupported data."""
    if not isinstance(value, dict) or type(value.get('schema_version')) is not int or value['schema_version'] not in (1, 2):
        raise ValueError('Expected a version-1 or version-2 geometry draft.')
    version = value['schema_version']
    _record(value, BASE_KEYS | ({'materials', 'components'} if version == 2 else set()), label='Geometry document')
    if value['schema'] != 'photonics-workbench-geometry' or value['units'] != 'um' or value['mode'] not in ('draft-3d', 'slab-preview'):
        raise ValueError('Expected a micrometre geometry draft, not a solver configuration.')
    domain = _vector(value['domain'], .1, 1000, 'Domain')
    materials = []
    if version == 2:
        if not isinstance(value['materials'], list) or len(value['materials']) > 512:
            raise ValueError('A geometry draft may contain at most 512 materials.')
        materials = [validate_material(m) for m in value['materials']]
        if len({m['id'] for m in materials}) != len(materials):
            raise ValueError('Material ids must be unique.')
    material_by_id = {m['id']: m for m in materials}
    limit = 100 if version == 1 else 512
    if not isinstance(value['objects'], list) or len(value['objects']) > limit:
        raise ValueError(f'A geometry draft may contain at most {limit} objects.')
    ids, objects = set(), []
    for item in value['objects']:
        polygon = isinstance(item, dict) and item.get('kind') == 'polygon'
        _record(item, SOLID_KEYS | ({'vertices', 'holes'} if polygon and version == 2 else set()),
                ('material_id', 'component_id') if version == 2 else (), 'Geometry object')
        key = _id(item['id'], 'Object id')
        if key in ids:
            raise ValueError('Geometry object ids must be unique.')
        ids.add(key)
        if type(item['visible']) is not bool or item['kind'] not in ('box', 'cylinder', 'ring') + (('polygon',) if version == 2 else ()):
            raise ValueError('Unsupported geometry shape or visibility value.')
        solid = {'id': key, 'name': _name(item['name'], 'Object name'), 'visible': item['visible'], 'kind': item['kind'],
                 'position': _vector(item['position'], -500, 500, 'Position'), 'size': _vector(item['size'], .001, 1000, 'Size'),
                 'radius': _number(item['radius'], .001, 500, 'Radius'), 'inner_radius': _number(item['inner_radius'], 0, 500, 'Inner radius'),
                 'rotation_deg': _angle(item['rotation_deg']),
                 'refractive_index': _number(item['refractive_index'], 1 if version == 1 else (0 if 'material_id' in item else .01), 5 if version == 1 else 20, 'Refractive index', positive=True)}
        if solid['kind'] == 'ring' and (solid['inner_radius'] < .001 or solid['radius'] - solid['inner_radius'] < .001 - 1e-12):
            raise ValueError('Ring inner radius and wall thickness must each be at least .001 µm.')
        if polygon:
            solid['vertices'], solid['holes'] = validate_polygon(item['vertices'], item['holes'])
        if 'material_id' in item:
            material_id = _id(item['material_id'], 'Material reference')
            if material_id not in material_by_id:
                raise ValueError('An object refers to a missing material.')
            material = material_by_id[material_id]
            index = evaluate_material(material, material['reference_wavelength_um'])['n']
            if not math.isclose(solid['refractive_index'], index, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('Object preview index differs from its material at the reference wavelength.')
            solid['material_id'] = material_id
        if 'component_id' in item:
            solid['component_id'] = _id(item['component_id'], 'Component reference')
        bounds = geometry_bounds(solid)
        if any(bounds['min'][a] < -domain[a] / 2 - EPS or bounds['max'][a] > domain[a] / 2 + EPS for a in 'xyz'):
            raise ValueError(f'{solid["name"]} extends outside the domain.')
        objects.append(solid)
    result = {'schema': 'photonics-workbench-geometry', 'schema_version': version, 'units': 'um', 'name': _name(value['name'], 'Document name'),
              'mode': value['mode'], 'domain': domain, 'objects': objects}
    if version == 2:
        result.update(materials=materials, components=_components(value['components'], objects, material_by_id, domain))
    return result


def _components(value, objects, materials, domain):
    if not isinstance(value, list) or len(value) > 512:
        raise ValueError('A geometry draft may contain at most 512 component instances.')
    by_id = {o['id']: o for o in objects}
    ids, owned, result = set(), set(), []
    for instance in value:
        _record(instance, {'id', 'family', 'generator_version', 'parameters', 'object_ids', 'placement', 'material_id', 'ports'}, label='Component')
        key = _id(instance['id'], 'Component id')
        if key in ids:
            raise ValueError('Component ids must be unique.')
        ids.add(key)
        if type(instance['generator_version']) is not int or instance['generator_version'] != 1:
            raise ValueError('Unsupported component generator version.')
        parameters = instance['parameters']
        if not isinstance(parameters, dict) or len(parameters) > 64:
            raise ValueError('Component parameters must contain at most 64 named finite numbers.')
        parameters = {_id(k, 'Parameter key'): _number(v, label='Component parameter') for k, v in parameters.items()}
        members = instance['object_ids']
        if not isinstance(members, list) or not 1 <= len(members) <= 512 or any(not isinstance(m, str) for m in members) or len(set(members)) != len(members):
            raise ValueError('Component object ids must be a nonempty unique array.')
        if any(m not in by_id or m in owned or by_id[m].get('component_id') != key for m in members):
            raise ValueError('Component membership does not match the actual scene objects.')
        owned.update(members)
        material_id = _id(instance['material_id'], 'Component material')
        if material_id not in materials:
            raise ValueError('Component material is missing.')
        placement = instance['placement']
        _record(placement, {'x', 'y', 'z', 'rotation_deg'}, label='Component placement')
        placement = {a: _number(placement[a], -500, 500, 'Component placement') for a in 'xyz'} | {'rotation_deg': _angle(instance['placement']['rotation_deg'])}
        ports = instance['ports']
        if not isinstance(ports, list) or len(ports) > 64:
            raise ValueError('A component may contain at most 64 named layout ports.')
        port_names, normalized_ports = set(), []
        for port in ports:
            _record(port, {'name', 'x', 'y', 'z', 'angle_deg', 'width_um'}, label='Layout port')
            name = _name(port['name'], 'Port name')
            if name in port_names:
                raise ValueError('Layout port names must be unique within a component.')
            port_names.add(name)
            normalized = {'name': name, **{a: _number(port[a], -500, 500, 'Layout port coordinate') for a in 'xyz'},
                          'angle_deg': _angle(port['angle_deg'], 'Port direction'), 'width_um': _number(port['width_um'], 0, 1000, 'Port width', positive=True)}
            if any(abs(normalized[a]) > domain[a] / 2 + EPS for a in 'xyz'):
                raise ValueError('A layout port extends outside the domain.')
            normalized_ports.append(normalized)
        result.append({'id': key, 'family': _id(instance['family'], 'Component family'), 'generator_version': 1, 'parameters': parameters,
                       'object_ids': list(members), 'placement': placement, 'material_id': material_id, 'ports': normalized_ports})
    if any(o.get('component_id') not in ids for o in objects if 'component_id' in o) or {o['id'] for o in objects if 'component_id' in o} != owned:
        raise ValueError('An object has a missing or inconsistent component instance.')
    return result
