"""Bounded 3D field study of one intact finite-height library taper."""
import math
from typing import Literal

from pydantic import Field, model_validator

from .geometry_draft import validate_geometry
from .study3d import StrictModel
from .taper_geometry import taper_geometry


class TaperStudy(StrictModel):
    schema_version: Literal[5] = 5
    study_type: Literal['taper-3d'] = 'taper-3d'
    geometry: dict
    wavelength_um: float = Field(1.55, ge=.5, le=3.)
    cladding_index: float = Field(1., ge=1., le=2.5)
    resolution: int = Field(20, ge=10, le=32)
    duration_fs: float = Field(60., ge=20., le=150.)
    pml_um: float = Field(.75, ge=.5, le=1.5)
    solver: Literal['meep', 'fdtdx']
    device: Literal['cpu', 'gpu']

    def layout(self):
        return taper_geometry(self.model_dump())

    def estimate(self, layout=None):
        domain = (layout or self.layout())['domain']
        grid = [round(domain[a] * self.resolution) for a in 'xyz']
        cells = math.prod(grid)
        courant = .5 if self.solver == 'meep' else .99 / math.sqrt(3)
        steps = math.ceil(self.duration_fs * .299792458 * self.resolution / courant)
        return {'grid': grid, 'cells': cells, 'estimated_steps': steps, 'cell_updates': cells * steps,
                'grid_spacing_um': 1 / self.resolution, 'planning_memory_mib': round(cells * 1024 / 2**20),
                'field_arrays_mib': round(cells * 6 * (8 if self.solver == 'meep' else 4) / 2**20, 1),
                'note': 'Memory is a planning allowance. Solver allocation and compilation depend on the backend.'}

    @model_validator(mode='after')
    def qualify(self):
        self.geometry = validate_geometry(self.geometry)
        if self.solver == 'meep' and self.device != 'cpu':
            raise ValueError('Select Meep CPU or FDTDX with an explicit CPU or NVIDIA target.')
        layout = self.layout()
        estimate = self.estimate(layout)
        if estimate['cells'] > 1_200_000 or estimate['cell_updates'] > 1_200_000_000:
            raise ValueError('Taper exceeds the local limit of 1.2 million cells or 1.2 billion cell updates. Reduce the design domain, resolution or duration.')
        return self

    def warnings(self):
        return ['The original polygon is extruded to its saved height. Straight input/output leads extend from its end faces through the X PML.',
                'The computational domain includes extra cladding and both access leads. The editor design is retained in the configuration.',
                'Ey is driven by a ramped single-frequency source. The final field is an instantaneous sample in arbitrary source units.',
                'Meep uses subpixel averaging and a current sheet. FDTDX samples the material on voxels and uses directed plane excitation. Compare material grids and recorded sampling before interpreting amplitudes.']


def example_taper():
    from .component_library import build_component
    geometry = build_component('taper')['geometry']
    return TaperStudy(geometry=geometry, solver='meep', device='cpu')
