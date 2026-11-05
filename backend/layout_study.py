"""Finite passive structures driven by a local Ey point current."""
from typing import Literal
from pydantic import Field, model_validator

from .geometry_draft import validate_geometry
from .study3d import StrictModel, Vec3
from .layout_grid import rasterize


class LayoutStudy(StrictModel):
    schema_version: Literal[6] = 6
    study_type: Literal['layout-3d'] = 'layout-3d'
    geometry: dict
    source_um: Vec3
    wavelength_um: float = Field(1.55, ge=.5, le=3.)
    cladding_index: float = Field(1., ge=1., le=2.5)
    resolution: int = Field(16, ge=10, le=40)
    duration_fs: float = Field(35., ge=5., le=120.)
    pml_um: float = Field(.75, ge=.5, le=1.5)
    solver: Literal['meep','fdtdx']
    device: Literal['cpu','gpu']

    @model_validator(mode='after')
    def check_layout(self):
        self.geometry = validate_geometry(self.geometry)
        if self.solver == 'meep' and self.device != 'cpu':
            raise ValueError('Meep uses CPU. Choose FDTDX for NVIDIA execution.')
        self._spec, _, _ = rasterize(self.model_dump())
        return self

    def layout(self):
        return self._spec


def example_layout():
    from .library_layout import build_design
    geometry = build_design('directional_coupler', {'length_um':3.,'width_um':.5,'height_um':.35,'gap_um':.25})['geometry']
    return LayoutStudy(geometry=geometry, source_um=Vec3(x=-1.,y=-.375,z=0.),solver='meep',device='cpu')
