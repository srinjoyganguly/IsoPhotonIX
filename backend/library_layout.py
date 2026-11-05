"""Atomic layout composition for parametric geometry drafts, without a solver."""
from __future__ import annotations

import copy
import math
import uuid

from .component_library import build_component, get_catalog
from .geometry_draft import validate_geometry, evaluate_material, geometry_bounds


def upgrade_geometry(value):
    geometry = validate_geometry(value)
    if geometry['schema_version'] == 1:
        geometry.update(schema_version=2, materials=[], components=[])
    geometry['mode'] = 'draft-3d'
    return geometry


def _material(identifier, geometry=None, supplied=None):
    candidates = (geometry or {}).get('materials', []) + get_catalog()['materials']
    if supplied is not None:
        if supplied.get('id') != identifier:
            raise ValueError('The supplied material id must match the selected material.')
        candidates = [supplied] + candidates
    material = next((item for item in candidates if item['id'] == identifier), None)
    if material is None:
        raise ValueError('Choose a catalog material or a material defined in this design.')
    return copy.deepcopy(material)


def _instance(component_id, parameters, placement, material):
    # Generate shape independently of the material approximation. A custom
    # optical record is preserved, never silently reduced for a solver.
    generated = build_component(component_id, parameters, placement, 'generic-core')
    geometry = generated['geometry']
    if len(geometry['components']) != 1:
        raise ValueError('A catalog recipe must generate one explicit component instance.')
    instance = copy.deepcopy(geometry['components'][0])
    identifier = 'component-' + uuid.uuid4().hex[:16]
    ids = {item['id']: f'{identifier}-{index + 1}' for index, item in enumerate(geometry['objects'])}
    n = evaluate_material(material, material['reference_wavelength_um'])['n']
    objects = []
    for original in geometry['objects']:
        item = copy.deepcopy(original)
        item.update(id=ids[original['id']], component_id=identifier,
                    material_id=material['id'], refractive_index=n)
        objects.append(item)
    instance.update(id=identifier, object_ids=[ids[value] for value in instance['object_ids']], material_id=material['id'])
    return geometry, objects, instance


def build_design(component_id, parameters=None, placement=None, material_id='generic-core', material=None):
    selected = _material(material_id, supplied=material)
    source, objects, instance = _instance(component_id, parameters, placement, selected)
    geometry = {**source, 'objects': objects, 'components': [instance], 'materials': [selected]}
    geometry = validate_geometry(geometry)
    return {'geometry': geometry, 'component': geometry['components'][0]}



def place_component(geometry, component_id, parameters=None, placement=None,
                    material_id=None, replace_component_id=None, fit_domain=False):
    original = upgrade_geometry(geometry)
    removed = set()
    old = None
    if replace_component_id is not None:
        old = next((item for item in original['components'] if item['id'] == replace_component_id), None)
        if old is None:
            raise ValueError('The selected component no longer exists; select its current instance again.')
        removed = set(old['object_ids'])
        # Partial regeneration edits only the requested values. Do not reset a
        # positioned instance or its custom material to new-component defaults.
        parameters = {**(old['parameters'] if old['family'] == component_id else {}), **(parameters or {})}
        placement = {**old['placement'], **(placement or {})}
        if material_id is None:
            material_id = old['material_id']
    selected = _material(material_id if material_id is not None else 'generic-core', original)
    _, objects, instance = _instance(component_id, parameters, placement, selected)
    if old is not None:
        # Keep the instance identity stable for selection and future layout
        # references; regenerated solids receive fresh, unambiguous ids.
        instance['id'] = old['id']
        for item in objects:
            item['component_id'] = old['id']
    candidate = copy.deepcopy(original)
    if removed:
        # The EM raster uses scene order for material overlaps. Replace old
        # member slots in place, including in imported interleaved layouts.
        # Additional members belong at the final old slot, below later solids.
        last = max(i for i, item in enumerate(candidate['objects']) if item['id'] in removed)
        replacements = iter(objects)
        ordered = []
        for i, item in enumerate(candidate['objects']):
            if item['id'] not in removed:
                ordered.append(item)
            elif i == last:
                ordered.extend(replacements)
            else:
                replacement = next(replacements, None)
                if replacement is not None:
                    ordered.append(replacement)
        candidate['objects'] = ordered
    else:
        candidate['objects'].extend(objects)
    candidate['components'] = [item for item in candidate['components'] if item['id'] != replace_component_id] + [instance]
    if not any(item['id'] == selected['id'] for item in candidate['materials']):
        candidate['materials'].append(selected)
    if fit_domain:
        _fit_domain(candidate)
    candidate = validate_geometry(candidate)
    return {'geometry': candidate, 'component': candidate['components'][-1]}


def _fit_domain(geometry):
    bounds = [geometry_bounds(item) for item in geometry['objects']]
    ports = [port for component in geometry['components'] for port in component['ports']]
    for axis in 'xyz':
        extents = [abs(box[edge][axis]) for box in bounds for edge in ('min', 'max')]
        extents.extend(abs(port[axis]) for port in ports)
        geometry['domain'][axis] = max(geometry['domain'][axis], 2 * max(extents, default=0) + .5)


def _same_endpoint(first, second):
    return math.dist([first[a] for a in 'xyz'], [second[a] for a in 'xyz']) <= 1e-8


def _facing(first, second):
    return abs((first['angle_deg'] - second['angle_deg']) % 360 - 180) <= 1e-7


def align_ports(geometry, moving_component, moving_port, fixed_component, fixed_port, fit_domain=False):
    """Rigidly place one component against a named port. Dimensions are in µm."""
    candidate = upgrade_geometry(geometry)
    if moving_component == fixed_component:
        raise ValueError('Choose ports on two different components.')

    def endpoint(component_id, port_name):
        component = next((c for c in candidate['components'] if c['id'] == component_id), None)
        if component is None:
            raise ValueError('Choose a component that is still in the layout.')
        port = next((p for p in component['ports'] if p['name'] == port_name), None)
        if port is None:
            raise ValueError('Choose an existing named port on this component.')
        return component, port

    moving, source = endpoint(moving_component, moving_port)
    _, target = endpoint(fixed_component, fixed_port)
    if not math.isclose(source['width_um'], target['width_um'], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('Port widths differ. Choose equal-width ports or insert a taper.')
    for neighbor in candidate['components']:
        if neighbor['id'] in (moving_component, fixed_component):
            continue
        if any(_same_endpoint(p, target) for p in neighbor['ports']):
            raise ValueError('That fixed port already meets another component.')
    if _same_endpoint(source, target) and _facing(source, target):
        return candidate
    for neighbor in candidate['components']:
        if neighbor['id'] != moving_component and any(
            _same_endpoint(p, q) and _facing(p, q) for p in moving['ports'] for q in neighbor['ports']
        ):
            raise ValueError('Moving this component would break an existing port alignment. Move it apart first.')

    angle = (target['angle_deg'] + 180 - source['angle_deg']) % 360
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    # Keep the original pivot until every solid, port and generator origin has moved.
    pivot = {axis: source[axis] for axis in 'xyz'}

    def move(point):
        dx, dy = point['x'] - pivot['x'], point['y'] - pivot['y']
        point.update(x=target['x'] + cos * dx - sin * dy,
                     y=target['y'] + sin * dx + cos * dy,
                     z=target['z'] + (point['z'] - pivot['z']))

    for solid in candidate['objects']:
        if solid.get('component_id') == moving_component:
            move(solid['position'])
            solid['rotation_deg'] = (solid['rotation_deg'] + angle) % 360
    move(moving['placement'])
    moving['placement']['rotation_deg'] = (moving['placement']['rotation_deg'] + angle) % 360
    for port in moving['ports']:
        move(port)
        port['angle_deg'] = (port['angle_deg'] + angle) % 360
    if fit_domain:
        _fit_domain(candidate)
    return validate_geometry(candidate)
