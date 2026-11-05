"""Reconstruct measured catalog configurations for the local editor."""
import json

from .component_library import build_component
from .layout_study import LayoutStudy
from .runtime import ROOT


def qualified_layout_example(family):
    report = json.loads((ROOT/'docs/verification/v09-family-fields.json').read_text())
    case = next((entry for entry in report['cases'] if entry['family'] == family), None)
    if case is None:
        raise KeyError('No measured field example is recorded for this family.')
    geometry = build_component(family,case['overrides'])['geometry']
    config = LayoutStudy(geometry=geometry, solver='meep', device='cpu',
        source_um=case['source_um'], resolution=case['resolution'], duration_fs=case['duration_fs'],
        pml_um=case['pml_um'], wavelength_um=case['wavelength_um'], cladding_index=case['cladding_index'])
    if config.layout()['epsilon_xyz_sha256'] != case['compiled_material_sha256']:
        raise ValueError('The compiled example differs from its measured material grid.')
    return config
