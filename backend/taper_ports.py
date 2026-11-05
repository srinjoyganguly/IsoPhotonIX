"""Four-directional solves for the guided two-port finite-height taper."""
import math
from typing import Literal
from pydantic import Field, model_validator
from .study3d import StrictModel
from .geometry_draft import validate_geometry
from .taper_port_geometry import port_layout


class TaperPortStudy(StrictModel):
    schema_version: Literal[7] = 7
    study_type: Literal['taper-ports-3d'] = 'taper-ports-3d'
    geometry: dict
    cladding_index: float = Field(1.,ge=1.,le=2.5)
    wavelength_min_um: float = Field(1.45,ge=.8,le=2.8)
    wavelength_max_um: float = Field(1.65,ge=.8,le=2.8)
    frequency_points: int = Field(5,ge=3,le=9)
    resolution: int = Field(16,ge=12,le=32)
    pml_um: float = Field(.75,ge=.5,le=1.5)
    max_after_sources_fs: float = Field(600.,ge=150.,le=1200.)
    decay_by: float = Field(1e-7,ge=1e-9,le=1e-4)
    spectrum_tolerance: float = Field(1e-4,ge=1e-6,le=1e-2)
    solver: Literal['meep','fdtdx']
    device: Literal['cpu','gpu']

    @model_validator(mode='after')
    def physical(self):
        self.geometry = validate_geometry(self.geometry)
        fmin,fmax = 1/self.wavelength_max_um,1/self.wavelength_min_um
        if not .04 <= (fmax-fmin)/((fmax+fmin)/2) <= .20:
            raise ValueError('Use a 4–20% frequency span with increasing wavelength bounds.')
        if self.solver=='meep' and self.device!='cpu':
            raise ValueError('Meep modal measurements require CPU.')
        estimate=self.estimate()
        if estimate['cells']>1_200_000 or estimate['maximum_cell_updates']>48_000_000_000:
            raise ValueError('The taper modal study exceeds the bounded four-stage work or memory limit.')
        return self

    def layout(self):
        return port_layout(self.model_dump())

    def estimate(self):
        domain=self.layout()['domain']
        grid=[round(domain[a]*self.resolution) for a in 'xyz']
        source_fs=6/(1/self.wavelength_min_um-1/self.wavelength_max_um)/.299792458
        courant=.5 if self.solver=='meep' else .99/math.sqrt(3)
        steps=math.ceil((source_fs+self.max_after_sources_fs)*.299792458*self.resolution/courant)
        cells=math.prod(grid)
        return {'grid':grid,'cells':cells,'stage_count':4,'estimated_source_duration_fs':source_fs,
                'maximum_steps_per_stage':steps,'maximum_cell_updates':4*cells*steps,
                'planning_memory_mib':round(cells*1024/2**20),'worker_timeout_seconds':1200,
                'note':'Two uniform references and two taper solves. Budgets use the requested time cap.'}

    def warnings(self):
        return ['Each excitation uses a uniform reference matching its own input end.',
            'The fundamental modes at the two port widths are solved separately.',
            'Complex S retains propagation phase between the recorded reference planes.',
            'Use measured settling, residual power and mesh/plane sensitivity when interpreting the fundamental-channel result.']


def example_taper_ports():
    from .component_library import build_component
    g=build_component('taper',{'length_um':2.,'input_width_um':.75,'output_width_um':1.,'height_um':.5})['geometry']
    return TaperPortStudy(geometry=g,solver='meep',device='cpu')
