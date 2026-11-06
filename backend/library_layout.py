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


def _endpoint(geometry, component_id, port_name):
    component=next((c for c in geometry['components'] if c['id']==component_id),None)
    if component is None:
        raise ValueError('Choose a component that is still in the layout.')
    port=next((p for p in component['ports'] if p['name']==port_name),None)
    if port is None:
        raise ValueError('Choose an existing named port on this component.')
    return component,port


def _existing_connection(geometry, a, b):
    for edge in geometry.get('connections',[]):
        ends=(edge['a'],edge['b'])
        if a in ends and b in ends:
            return edge
        if a in ends or b in ends:
            raise ValueError('That port already belongs to another connection. Disconnect it first.')
    return None


def connection_status(geometry):
    """Evaluate geometry at saved port endpoints without inferring guided power."""
    checked=validate_geometry(geometry)
    ports={(c['id'],p['name']):p for c in checked.get('components',[]) for p in c['ports']}
    rows=[]
    for edge in checked.get('connections',[]):
        a,b=(ports[(end['component_id'],end['port'])] for end in (edge['a'],edge['b']))
        separation=math.dist([a[k] for k in 'xyz'],[b[k] for k in 'xyz'])
        angle_error=abs((a['angle_deg']-b['angle_deg'])%360-180)
        widths_match=math.isclose(a['width_um'],b['width_um'],rel_tol=1e-9,abs_tol=1e-9)
        aligned=separation<=1e-8 and angle_error<=1e-7 and widths_match
        rows.append({**edge,'state':'aligned' if aligned else 'needs_route','distance_um':separation,
                     'angle_error_deg':angle_error,'width_difference_um':abs(a['width_um']-b['width_um']),
                     'widths_match':widths_match})
    return rows


def connect_ports(geometry, moving_component, moving_port, fixed_component, fixed_port, align=False, fit_domain=False):
    """Save design intent, optionally applying the normal rigid port alignment first."""
    if moving_component==fixed_component:
        raise ValueError('Choose ports on two different components.')
    candidate=upgrade_geometry(geometry)
    _endpoint(candidate,moving_component,moving_port)
    _endpoint(candidate,fixed_component,fixed_port)
    a={'component_id':moving_component,'port':moving_port}
    b={'component_id':fixed_component,'port':fixed_port}
    existing=_existing_connection(candidate,a,b)
    if align:
        candidate=align_ports(candidate,moving_component,moving_port,fixed_component,fixed_port,fit_domain)
    if existing is not None:
        return {'geometry':candidate,'connection':existing}
    edge={'id':'connection-'+uuid.uuid4().hex[:16],'a':a,'b':b}
    candidate.setdefault('connections',[]).append(edge)
    checked=validate_geometry(candidate)
    return {'geometry':checked,'connection':checked['connections'][-1]}


def disconnect_ports(geometry, connection_id):
    candidate=validate_geometry(geometry)
    edges=candidate.get('connections',[])
    if not any(e['id']==connection_id for e in edges):
        raise ValueError('Choose a saved connection in this design.')
    candidate['connections']=[e for e in edges if e['id']!=connection_id]
    return candidate


def align_ports(geometry, moving_component, moving_port, fixed_component, fixed_port, fit_domain=False):
    """Rigidly place one component against a named port. Dimensions are in µm."""
    candidate = upgrade_geometry(geometry)
    if moving_component == fixed_component:
        raise ValueError('Choose ports on two different components.')

    moving, source = _endpoint(candidate,moving_component,moving_port)
    _, target = _endpoint(candidate,fixed_component,fixed_port)
    _existing_connection(candidate,{'component_id':moving_component,'port':moving_port},
                         {'component_id':fixed_component,'port':fixed_port})
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
