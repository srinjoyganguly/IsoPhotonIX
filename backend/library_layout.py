"""Atomic layout composition for parametric geometry drafts, without a solver."""
from __future__ import annotations

import copy
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
    candidate['objects'] = [item for item in candidate['objects'] if item['id'] not in removed] + objects
    candidate['components'] = [item for item in candidate['components'] if item['id'] != replace_component_id] + [instance]
    if not any(item['id'] == selected['id'] for item in candidate['materials']):
        candidate['materials'].append(selected)
    if fit_domain:
        extents = [geometry_bounds(item) for item in candidate['objects']]
        for axis in 'xyz':
            needed = 2*max((max(abs(item['min'][axis]), abs(item['max'][axis])) for item in extents), default=0) + .5
            candidate['domain'][axis] = max(candidate['domain'][axis], needed)
    candidate = validate_geometry(candidate)
    return {'geometry': candidate, 'component': candidate['components'][-1]}
