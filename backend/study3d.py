"""Versioned, bounded finite-height waveguide studies in micrometres/fs.

Geometry drafts and executable studies are separate formats. No arbitrary draft
is projected into a simpler model: unsupported solids/transforms are rejected.
"""
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True, serialize_by_alias=True)


class Vec3(StrictModel):
    x: float
    y: float
    z: float


class Solid(StrictModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=80)
    visible: bool
    kind: Literal["box", "cylinder", "ring"]
    position: Vec3
    size: Vec3
    radius: float = Field(ge=.001, le=500)
    inner_radius: float = Field(ge=0, le=500)
    rotation_deg: float = Field(ge=-360, le=360)
    refractive_index: float = Field(ge=1, le=5)


class Geometry(StrictModel):
    schema_name: Literal["photonics-workbench-geometry"] = Field(alias="schema")
    schema_version: Literal[1]
    units: Literal["um"]
    name: str = Field(min_length=1, max_length=80)
    mode: Literal["draft-3d"]
    domain: Vec3
    objects: list[Solid] = Field(min_length=1, max_length=2)


class Waveguide3DConfig(StrictModel):
    schema_version: Literal[2] = 2
    study_type: Literal["waveguide-3d"] = "waveguide-3d"
    geometry: Geometry
    wavelength_um: float = Field(1.55, ge=.5, le=3)
    cladding_index: float = Field(1, ge=1, le=2.5)
    resolution: int = Field(16, ge=10, le=32)
    duration_fs: float = Field(60, ge=20, le=150)
    pml_um: float = Field(.75, ge=.5, le=1.5)
    solver: Literal["meep", "fdtdx"] = "meep"
    device: Literal["cpu", "gpu"] = "cpu"

    @property
    def core(self):
        return next(item for item in self.geometry.objects if item.id == "core")

    @property
    def substrate(self):
        return next((item for item in self.geometry.objects if item.id == "substrate"), None)

    @model_validator(mode="after")
    def supported_study(self):
        domain = self.geometry.domain
        if not (6 <= domain.x <= 16 and 3 <= domain.y <= 8 and 3 <= domain.z <= 8):
            raise ValueError("3D domain must be X 6–16 µm, Y/Z 3–8 µm. Open the runnable 3D waveguide example.")
        ids = [item.id for item in self.geometry.objects]
        if len(set(ids)) != len(ids) or "core" not in ids or any(i not in {"core", "substrate"} for i in ids):
            raise ValueError("This study supports one box named by id 'core' and an optional 'substrate' box. Rings, couplers and arbitrary drafts are not yet executable.")
        for solid in self.geometry.objects:
            if solid.kind != "box" or abs(solid.rotation_deg % 360) > 1e-8:
                raise ValueError("The finite-height waveguide study supports axis-aligned boxes only; rotated solids, rings and cylinders cannot run yet.")
            if not solid.name.strip() or any(ord(c) < 32 or ord(c) == 127 for c in solid.name):
                raise ValueError("Object names must contain printable characters.")
            for axis in ("x", "y", "z"):
                if getattr(solid.size, axis) < .001:
                    raise ValueError("Every solid dimension must be positive.")
                if abs(getattr(solid.position, axis)) + getattr(solid.size, axis) / 2 > getattr(domain, axis) / 2 + 1e-8:
                    raise ValueError(f"{solid.name} extends outside the domain along {axis}.")
        core = self.core
        if any(abs(getattr(core.position, a)) > 1e-8 for a in ("x", "y", "z")):
            raise ValueError("The supported straight core must be centered at X/Y/Z = 0.")
        if not math.isclose(core.size.x, domain.x, abs_tol=1e-8):
            raise ValueError("Core length must equal domain X so the straight guide continues through both X PML regions.")
        if core.size.y * self.resolution < 4 or core.size.z * self.resolution < 4:
            raise ValueError("Core width and finite height must each span at least four grid cells.")
        for axis in ("y", "z"):
            if (getattr(domain, axis) - getattr(core.size, axis)) / 2 - self.pml_um < .4:
                raise ValueError("Leave at least 0.4 µm between the core and each Y/Z absorbing layer.")
        substrate = self.substrate
        if substrate:
            expected_height = (domain.z - core.size.z) / 2
            expected_z = -domain.z / 2 + expected_height / 2
            if not all(math.isclose(a, b, abs_tol=1e-8) for a, b in (
                (substrate.size.x, domain.x), (substrate.size.y, domain.y),
                (substrate.size.z, expected_height), (substrate.position.x, 0),
                (substrate.position.y, 0), (substrate.position.z, expected_z))):
                raise ValueError("Substrate must fill domain X/Y from the bottom boundary up to the core bottom. Use 'Fit substrate to core' after changing height or domain.")
        highest_cladding = max(self.cladding_index, substrate.refractive_index if substrate else 1)
        if core.refractive_index <= highest_cladding:
            raise ValueError("Core index must exceed both substrate and background index for this guiding study.")
        max_index = max([self.cladding_index] + [o.refractive_index for o in self.geometry.objects])
        if self.wavelength_um * self.resolution / max_index < 10:
            raise ValueError("Use at least 10 cells per wavelength in the highest-index material: increase resolution or wavelength.")
        if self.pml_um * self.resolution < 8:
            raise ValueError("Each 3D PML must span at least eight grid cells.")
        if domain.x - 2 * self.pml_um < 4:
            raise ValueError("Leave at least 4 µm of propagation length between X PML layers.")
        estimate = self.estimate()
        if estimate["cells"] > 1_200_000:
            raise ValueError("This 3D milestone is limited to 1,200,000 cells. Reduce domain or resolution.")
        if estimate["cell_updates"] > 1_200_000_000:
            raise ValueError("The requested grid × time steps exceeds the bounded local study budget. Reduce resolution, domain or duration.")
        if self.solver == "meep" and self.device != "cpu":
            raise ValueError("Meep uses CPU only here. Select FDTDX for the NVIDIA GPU route.")
        return self

    def estimate(self):
        grid = [round(getattr(self.geometry.domain, a) * self.resolution) for a in ("x", "y", "z")]
        cells = math.prod(grid)
        courant = .5 if self.solver == "meep" else .99 / math.sqrt(3)
        steps = math.ceil(self.duration_fs * .299792458 * self.resolution / courant)
        return {"grid": grid, "cells": cells, "estimated_steps": steps, "cell_updates": cells * steps,
                "grid_spacing_um": 1 / self.resolution,
                "field_arrays_mib": round(cells * 6 * (8 if self.solver == "meep" else 4) / 2**20, 1),
                "planning_memory_mib": round(cells * 1024 / 2**20),
                "note": "Planning allowance only, not a measured peak: includes a conservative per-cell allowance for fields, PML, material and compilation buffers; backend compilation can allocate additional memory. One worker at a time; five-minute limit."}

    def warnings(self):
        return ["Full 3D finite-height, lossless nondispersive straight guide; all six faces use PML.",
                "Ey finite sheet/plane excitation, not an eigenmode. No normalized transmission or mode-purity result.",
                "Meep current-sheet and FDTDX directed E/H-plane injection are different; amplitudes are not calibrated across solvers.",
                "FDTDX rounds solids and boundaries to its voxel grid; Meep uses subpixel material averaging. Exports describe actual sampling and placements.",
                "Scene visibility is presentation only: hidden solids still participate in the simulation.",
                "Grid/PML sensitivity must be studied for quantitative results; a finite nonzero field is not an accuracy certificate."]


def example3d():
    def solid(id, name, size, position, index):
        return dict(id=id, name=name, visible=True, kind="box", size=dict(zip("xyz", size)),
                    position=dict(zip("xyz", position)), radius=1., inner_radius=.6,
                    rotation_deg=0., refractive_index=index)
    return Waveguide3DConfig(geometry=Geometry.model_validate({
        "schema": "photonics-workbench-geometry", "schema_version": 1, "units": "um",
        "name": "Runnable finite-height waveguide", "mode": "draft-3d",
        "domain": {"x": 8., "y": 4., "z": 4.},
        "objects": [solid("substrate", "Substrate", (8., 4., 1.75), (0., 0., -1.125), 1.44),
                    solid("core", "Finite-height core", (8., .75, .5), (0., 0., 0.), 2.)],
    }))


def parse_study(value):
    # Preserve the v1 slab format and reject unknown versions, never downgrade.
    from .model import WaveguideConfig
    if value.get("schema_version") == 4:
        from .ports3d import ModalPortStudy
        return ModalPortStudy.model_validate(value)
    if value.get("schema_version") == 3:
        from .ports import PortStudy
        return PortStudy.model_validate(value)
    if value.get("schema_version", 1) == 2:
        return Waveguide3DConfig.model_validate(value)
    return WaveguideConfig.model_validate(value)
