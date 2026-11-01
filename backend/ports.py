"""Bounded, versioned studies of incident-normalized waveguide power flux."""
import math
from typing import Literal
from pydantic import Field, model_validator
from .study3d import StrictModel


class PortStudy(StrictModel):
    schema_version: Literal[3] = 3
    study_type: Literal['waveguide-ports-2d'] = 'waveguide-ports-2d'
    device_kind: Literal['straight', 'gap'] = 'gap'
    gap_um: float = Field(.25, ge=0, le=2)
    core_width_um: float = Field(.625, ge=.2, le=1.5)
    core_index: float = Field(2., ge=1.01, le=3.5)
    cladding_index: float = Field(1., ge=1., le=2.5)
    wavelength_min_um: float = Field(1.45, ge=.8, le=2.8)
    wavelength_max_um: float = Field(1.65, ge=.8, le=2.8)
    frequency_points: int = Field(21, ge=3, le=61)
    length_um: float = Field(12., ge=10, le=16)
    height_um: float = Field(6., ge=4, le=8)
    pml_um: float = Field(1., ge=.75, le=1.5)
    resolution: int = Field(32, ge=20, le=48)
    max_after_sources_fs: float = Field(1000., ge=300, le=1500)
    decay_by: float = Field(1e-8, ge=1e-8, le=1e-4)
    spectrum_tolerance: float = Field(1e-4, ge=1e-6, le=1e-2)
    solver: Literal['meep', 'fdtdx'] = 'meep'
    device: Literal['cpu', 'gpu'] = 'cpu'

    @model_validator(mode='after')
    def physical_study(self):
        if self.core_index <= self.cladding_index:
            raise ValueError('The core index must exceed the cladding index.')
        if self.wavelength_min_um >= self.wavelength_max_um:
            raise ValueError('Minimum wavelength must be below maximum wavelength.')
        fmin, fmax = 1/self.wavelength_max_um, 1/self.wavelength_min_um
        bandwidth = (fmax-fmin)/((fmax+fmin)/2)
        if not .04 <= bandwidth <= .2:
            raise ValueError('Use a frequency span of 4–20% of the center frequency for this fixed-profile mode pulse.')
        if self.core_width_um*self.resolution < 6:
            raise ValueError('The core must span at least six grid cells.')
        if self.wavelength_min_um*self.resolution/self.core_index < 12:
            raise ValueError('Use at least 12 cells per shortest wavelength inside the core.')
        if self.pml_um*self.resolution < 12:
            raise ValueError('Each PML must span at least twelve grid cells.')
        if self.height_um-2*self.pml_um-self.core_width_um < 2:
            raise ValueError('Leave at least 1 µm between each core side and the transverse PML.')
        if self.length_um-2*self.pml_um < 7:
            raise ValueError('Leave at least 7 µm between the longitudinal PML layers for source and monitors.')
        if self.device_kind == 'straight' and self.gap_um != 0:
            raise ValueError('A straight-reference check must use gap_um=0; no gap is silently ignored.')
        if self.device_kind == 'gap' and self.gap_um*self.resolution < 4:
            raise ValueError('The cladding gap must span at least four grid cells.')
        if self.solver == 'meep' and self.device != 'cpu':
            raise ValueError('Meep uses CPU only. FDTDX exposes the separately verified NVIDIA route.')
        plan=self.estimate()
        if plan['xy_cells'] > 180_000 or plan['maximum_xy_cell_updates'] > 12_000_000_000:
            raise ValueError('Requested reference and device studies exceed this bounded local resource budget. Reduce the grid, domain or time cap.')
        return self

    def estimate(self):
        nx,ny=round(self.length_um*self.resolution),round(self.height_um*self.resolution)
        fmin,fmax=1/self.wavelength_max_um,1/self.wavelength_min_um
        # Meep Gaussian cutoff 6 gives support 12/fwidth; each solver records its exact profile.
        source_duration_fs=12/(2*(fmax-fmin))/.299792458
        courant=.5 if self.solver=='meep' else .99/math.sqrt(3)
        steps=math.ceil((source_duration_fs+self.max_after_sources_fs)*.299792458*self.resolution/courant)
        return {'grid':[nx,ny,1 if self.solver=='meep' else 3], 'xy_cells':nx*ny,
                'stage_count':2, 'estimated_source_duration_fs':source_duration_fs,
                'maximum_steps_per_stage':steps,'maximum_xy_cell_updates':2*nx*ny*steps,
                'grid_spacing_um':1/self.resolution,
                'planning_memory_mib':round(nx*ny*(1 if self.solver=='meep' else 3)*1024/2**20),
                'note':'Two local solves: straight incident reference, then requested device. Memory is a planning allowance, not measured peak. Pulse support and actual settling are recorded by each solver; five-minute worker limit.'}

    def layout(self):
        source=-self.length_um/2+self.pml_um+.5
        return {'source_x_um':source,'reflection_x_um':source+1,
                'transmission_x_um':self.length_um/2-self.pml_um-1.5,
                'monitor_aperture_um':self.height_um-2*self.pml_um,
                'gap_center_x_um':0.,'field_component':'Ez','propagation_axis':'x',
                'dimensions':'2D invariant z; FDTDX uses three periodic z cells'}

    def warnings(self):
        return ['A fundamental TE-like mode profile excites a finite-band pulse; a separate straight-guide run measures incident power.',
                'R and T describe total flux through finite monitor apertures, not mode-resolved complex S-parameters.',
                '1 − R − T is uncollected flux. It includes lateral radiation and numerical/monitor error; these media are lossless.',
                'Pulse decay is a settling diagnostic, not proof of mesh, PML or domain convergence.',
                'Meep and FDTDX have different discretizations and source implementations; recorded normalization does not imply exact cross-solver agreement.',
                'FDTDX rounds geometry and monitor positions to its grid; exported metadata records requested and actual bounds. Fractional-cell inputs can shift interfaces.']


def example_ports(kind='gap'):
    return PortStudy(device_kind=kind,gap_um=.25 if kind=='gap' else 0.)
