"""Finite-height, two-port fundamental-mode studies with explicit scope."""
import math
from typing import Literal

from pydantic import Field, model_validator

from .study3d import Geometry, StrictModel, Waveguide3DConfig, example3d

STAGES = ('reference_forward', 'device_forward', 'reference_reverse', 'device_reverse')


class ModalPortStudy(StrictModel):
    schema_version: Literal[4] = 4
    study_type: Literal['waveguide-ports-3d'] = 'waveguide-ports-3d'
    geometry: Geometry
    device_kind: Literal['straight', 'gap'] = 'gap'
    gap_um: float = Field(.25, ge=0, le=1)
    cladding_index: float = Field(1., ge=1, le=2.5)
    wavelength_min_um: float = Field(1.45, ge=.8, le=2.8)
    wavelength_max_um: float = Field(1.65, ge=.8, le=2.8)
    frequency_points: int = Field(5, ge=3, le=9)
    resolution: int = Field(16, ge=12, le=32)
    pml_um: float = Field(.75, ge=.5, le=1.5)
    max_after_sources_fs: float = Field(600., ge=150, le=1200)
    decay_by: float = Field(1e-7, ge=1e-9, le=1e-4)
    spectrum_tolerance: float = Field(1e-4, ge=1e-6, le=1e-2)
    solver: Literal['meep', 'fdtdx'] = 'meep'
    device: Literal['cpu', 'gpu'] = 'cpu'

    @property
    def core(self):
        return next(item for item in self.geometry.objects if item.id == 'core')

    @property
    def substrate(self):
        return next((item for item in self.geometry.objects if item.id == 'substrate'), None)

    @model_validator(mode='after')
    def physical_study(self):
        # Reuse the established exact solid/substrate contract, with a small
        # dummy duration; this study's four-stage budget is checked separately.
        Waveguide3DConfig(geometry=self.geometry, wavelength_um=self.wavelength_min_um,
            cladding_index=self.cladding_index, resolution=self.resolution, duration_fs=20.,
            pml_um=self.pml_um, solver=self.solver, device=self.device)
        if self.wavelength_min_um >= self.wavelength_max_um:
            raise ValueError('Minimum wavelength must be below maximum wavelength.')
        fmin, fmax = 1/self.wavelength_max_um, 1/self.wavelength_min_um
        if not .04 <= (fmax-fmin)/((fmax+fmin)/2) <= .2:
            raise ValueError('Use a 4–20% frequency span for this fixed-profile mode pulse.')
        if min(self.core.size.y, self.core.size.z)*self.resolution < 6:
            raise ValueError('The finite core width and height must each span at least six cells.')
        if self.device_kind == 'straight' and self.gap_um != 0:
            raise ValueError('A straight validation study requires gap_um=0.')
        if self.device_kind == 'gap' and self.gap_um*self.resolution < 3:
            raise ValueError('The cladding gap must span at least three cells.')
        if self.geometry.domain.x-2*self.pml_um < 5:
            raise ValueError('Leave at least 5 µm between the longitudinal PML layers for both sources and ports.')
        if self.layout()['port_2_x_um']-self.gap_um/2 < .75:
            raise ValueError('Leave at least 0.75 µm of uniform guide between the gap and each port.')
        if self.estimate()['maximum_cell_updates'] > 48_000_000_000:
            raise ValueError('The four-stage study exceeds the local work budget. Reduce grid, domain or time cap.')
        return self

    def layout(self):
        source = self.geometry.domain.x/2-self.pml_um-.5
        port = source-1.
        return {'source_forward_x_um':-source, 'source_reverse_x_um':source,
                'port_1_x_um':-port, 'port_2_x_um':port,
                'aperture_y_um':self.geometry.domain.y-2*self.pml_um,
                'aperture_z_um':self.geometry.domain.z-2*self.pml_um,
                'gap_center_x_um':0., 'mode':'Fundamental Ey-like guided mode',
                'reference_planes':'Physical port monitor centres; no automatic de-embedding.'}

    def estimate(self):
        grid = [round(getattr(self.geometry.domain,a)*self.resolution) for a in 'xyz']
        fmin, fmax = 1/self.wavelength_max_um, 1/self.wavelength_min_um
        source_fs = 6/(fmax-fmin)/.299792458
        courant = .5 if self.solver == 'meep' else .99/math.sqrt(3)
        steps = math.ceil((source_fs+self.max_after_sources_fs)*.299792458*self.resolution/courant)
        cells = math.prod(grid)
        return {'grid':grid, 'cells':cells, 'stage_count':4, 'estimated_source_duration_fs':source_fs,
                'maximum_steps_per_stage':steps, 'maximum_cell_updates':4*cells*steps,
                'planning_memory_mib':round(cells*1024/2**20), 'worker_timeout_seconds':1200,
                'note':'Four local solves; measured settling may stop each early. Memory is a planning allowance, not measured peak. Twenty-minute worker limit; complex modes are prepared locally on CPU.'}

    def warnings(self):
        return ['Finite-height lossless, isotropic, nondispersive guide; all six domain faces absorb with PML.',
                'One Ey-like fundamental channel per port. Other modes and radiation are outside this 2×2 S-matrix.',
                'S rows are outgoing ports, columns are excited ports. Complex phase is defined at the recorded physical port planes.',
                'The gap replaces core material with cladding; an optional substrate stays continuous below the core.',
                'The fixed source profile is computed at the centre frequency; monitor modes are resolved across the band.',
                'FDTDX rounds geometry and planes to its Yee grid; actual coordinates and mode conventions are exported.',
                'Settling, reciprocity and CPU/GPU consistency are checks, not proof of mesh convergence or commercial agreement.',
                'Four reference/device solves can take several minutes on CPU; cancellation stops the whole job.']


def example_ports3d(kind='gap'):
    geometry = example3d().geometry.model_dump()
    geometry['name'] = 'Finite-height modal two-port guide'
    geometry['objects'] = [obj for obj in geometry['objects'] if obj['id'] == 'core']
    return ModalPortStudy(geometry=Geometry.model_validate(geometry), device_kind=kind,
                          gap_um=.25 if kind == 'gap' else 0.)
