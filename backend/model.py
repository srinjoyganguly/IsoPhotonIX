"""Small, shared physical model. All lengths are micrometres, times femtoseconds."""
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WaveguideConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    wavelength_um: float = Field(1.55, ge=0.5, le=3.0)
    core_width_um: float = Field(0.6, ge=0.2, le=3.0)
    core_index: float = Field(2.0, ge=1.01, le=3.5)
    cladding_index: float = Field(1.0, ge=1.0, le=2.5)
    length_um: float = Field(10.0, ge=6.0, le=20.0)
    height_um: float = Field(5.0, ge=3.0, le=12.0)
    resolution: int = Field(20, ge=10, le=60)
    duration_fs: float = Field(80.0, ge=20.0, le=250.0)
    pml_um: float = Field(1.0, ge=0.5, le=2.0)
    solver: Literal["meep", "fdtdx"] = "meep"
    device: Literal["cpu", "gpu"] = "cpu"

    @model_validator(mode="after")
    def physical_constraints(self):
        if self.core_index <= self.cladding_index:
            raise ValueError("Core index must exceed cladding index for this guiding example.")
        if self.height_um - 2*self.pml_um - self.core_width_um < 0.8:
            raise ValueError("Leave at least 0.4 µm of cladding above and below the core, inside the PML.")
        if self.length_um - 2*self.pml_um < 4:
            raise ValueError("Propagation region inside the PML must be at least 4 µm long.")
        if self.wavelength_um*self.resolution/self.core_index < 10:
            raise ValueError("Use at least 10 cells per wavelength in the core: increase resolution or wavelength.")
        if self.core_width_um*self.resolution < 4:
            raise ValueError("Core width must span at least four grid cells.")
        if self.pml_um*self.resolution < 8:
            raise ValueError("PML must span at least eight grid cells.")
        if self.length_um*self.height_um*self.resolution**2 > 180_000:
            raise ValueError("This demo is limited to 180,000 cells in the x-y plane; reduce domain or resolution.")
        if self.solver == "meep" and self.device != "cpu":
            raise ValueError("Meep supports CPU execution only in this workbench. Choose FDTDX for NVIDIA GPU.")
        return self

    def estimate(self):
        nx, ny = round(self.length_um*self.resolution), round(self.height_um*self.resolution)
        nz = 3 if self.solver == "fdtdx" else 1
        return {"grid": [nx, ny, nz], "cells": nx*ny*nz,
                "cells_per_core_wavelength": round(self.wavelength_um*self.resolution/self.core_index, 2),
                "grid_spacing_um": 1/self.resolution,
                "estimated_steps": math.ceil(self.duration_fs*0.299792458*self.resolution/(0.99/math.sqrt(3) if nz == 3 else 0.5)),
                "note": "Grid estimate only. Compilation memory and solver boundary discretization vary."}

    def warnings(self):
        return ["Educational, lossless, nondispersive 2D slab model; no convergence study or normalized transmission.",
                "Meep electric line and FDTDX directed plane sources differ; field amplitudes are not cross-solver calibrated."]
