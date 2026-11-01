"""Finite-height, fully three-dimensional Meep CPU waveguide adapter.

The study schema owns supported-geometry validation. This adapter translates
the saved box geometry, not an illustrative extrusion of a 2D calculation.
Official API: https://meep.readthedocs.io/en/latest/Python_User_Interface/
"""
from __future__ import annotations

import math
import platform
from pathlib import Path
import time

C_UM_PER_FS = 0.299792458


def run(config: dict, output_dir: Path) -> dict:
    if config.get("device") != "cpu":
        raise ValueError("Meep supports CPU only in this application; select CPU explicitly.")
    if config.get("study_type") != "waveguide-3d" or config.get("schema_version") != 2:
        raise ValueError("Expected a schema-version-2 finite-height waveguide study.")
    geometry = config["geometry"]
    objects = geometry["objects"]
    if any(obj["kind"] != "box" or obj["rotation_deg"] % 360 != 0 for obj in objects):
        raise ValueError("This 3D study supports axis-aligned boxes only.")
    by_id = {obj["id"]: obj for obj in objects}
    if "core" not in by_id or set(by_id) - {"core", "substrate"}:
        raise ValueError("This 3D study requires one core and an optional substrate.")

    started = time.perf_counter()
    import meep as mp
    import numpy as np

    mp.verbosity(0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    domain = geometry["domain"]
    core = by_id["core"]
    wavelength = float(config["wavelength_um"])
    resolution = int(config["resolution"])
    pml = float(config["pml_um"])
    duration = float(config["duration_fs"]) * C_UM_PER_FS
    source_x = -domain["x"] / 2 + pml + 0.5
    cell = mp.Vector3(domain["x"], domain["y"], domain["z"])

    def source_profile(t):
        return -math.cos(2 * math.pi * t / wavelength) * min(1.0, max(0.0, t / (4 * wavelength)))

    # Material precedence is explicit and independent of editor list order.
    # Visibility is presentation state and does not exclude physical material.
    material_objects = []
    for object_id in ("substrate", "core"):
        if object_id not in by_id:
            continue
        obj = by_id[object_id]
        material_objects.append(mp.Block(
            size=mp.Vector3(*(obj["size"][axis] for axis in "xyz")),
            center=mp.Vector3(*(obj["position"][axis] for axis in "xyz")),
            material=mp.Medium(index=float(obj["refractive_index"])),
        ))
    print("Meep 3D: initializing finite-height guide with PML on all six faces", flush=True)
    sim = mp.Simulation(
        cell_size=cell, dimensions=3, resolution=resolution, Courant=0.5,
        default_material=mp.Medium(index=float(config["cladding_index"])),
        geometry=material_objects, boundary_layers=[mp.PML(pml)],
        sources=[mp.Source(
            mp.CustomSource(source_profile, start_time=0, center_frequency=1 / wavelength),
            component=mp.Ey,
            center=mp.Vector3(source_x, core["position"]["y"], core["position"]["z"]),
            size=mp.Vector3(0, core["size"]["y"], core["size"]["z"]),
        )],
    )
    sim.init_sim()
    initialized = time.perf_counter()
    print("Meep 3D: evolving the full vector Maxwell fields", flush=True)
    sim.run(until=duration)
    solved = time.perf_counter()
    volume = mp.Volume(center=mp.Vector3(), size=cell)
    fields = np.stack([
        np.asarray(sim.get_array(component=component, vol=volume), dtype=np.float32).transpose(2, 1, 0)
        for component in (mp.Ex, mp.Ey, mp.Ez)
    ])
    epsilon = np.asarray(sim.get_array(component=mp.Dielectric, vol=volume), dtype=np.float32).transpose(2, 1, 0)
    x_um, y_um, z_um, _ = sim.get_array_metadata(vol=volume)
    x_um, y_um, z_um = map(np.asarray, (x_um, y_um, z_um))
    shape = (len(z_um), len(y_um), len(x_um))
    if fields.shape != (3, *shape) or epsilon.shape != shape:
        raise RuntimeError("Meep 3D field/material arrays do not match their coordinate axes.")
    if not all(np.isfinite(array).all() for array in (fields, epsilon, x_um, y_um, z_um)):
        raise RuntimeError("Meep 3D returned non-finite numerical arrays.")
    if not np.any(fields[1]):
        raise RuntimeError("Meep 3D returned an all-zero Ey field.")
    actual_time = float(sim.meep_time() / C_UM_PER_FS)
    time_steps = int(sim.timestep())
    np.savez_compressed(output_dir / "arrays.npz", field=fields[1], fields=fields,
                        epsilon=epsilon, x_um=x_um, y_um=y_um, z_um=z_um)
    sim.reset_meep()
    finished = time.perf_counter()
    peak = float(np.max(np.abs(fields[1])))
    print(f"Meep 3D: saved {shape} [z,y,x] samples; max |Ey| = {peak:.6g}", flush=True)
    return {
        "solver": "meep", "device": "cpu", "actual_device": "CPU", "device_name": "CPU",
        "versions": {"meep": mp.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "dimensions": "3D", "field_component": "Ey", "field_components": ["Ex", "Ey", "Ez"],
        "array_order": "fields[component,z,y,x]; field and epsilon[z,y,x]",
        "field_units": "arbitrary electric-current source amplitude",
        "field_shape": list(shape), "grid_shape": [len(x_um), len(y_um), len(z_um)],
        "field_max_abs": peak, "frame_count": 0,
        "actual_duration_fs": actual_time, "time_steps": time_steps,
        "time_step_fs": 0.5 / resolution / C_UM_PER_FS, "courant_factor": 0.5,
        "grid_spacing_um": 1 / resolution,
        "precision": "Meep float64 evolution; exported electric fields and epsilon float32",
        "timings": {"initialization_s": initialized - started, "compute_s": solved - initialized,
                    "export_s": finished - solved, "total_s": finished - started},
        "source": {"type": "uniform electric current sheet; not an eigenmode; bidirectional radiation",
                   "component": "Ey", "x_um": source_x, "y_um": core["position"]["y"],
                   "z_um": core["position"]["z"], "width_um": core["size"]["y"],
                   "height_um": core["size"]["z"], "wavelength_um": wavelength,
                   "amplitude": 1, "startup_periods": 4,
                   "temporal_profile": "-cos(2*pi*t/T) * clip(t/(4*T),0,1)"},
        "boundaries": {"x": "PML both faces", "y": "PML both faces", "z": "PML both faces", "pml_um": pml},
        "material_sampling": "Meep subpixel smoothing enabled; Dielectric output is the harmonic mean of the permittivity eigenvalues; fields interpolated onto common output grid by get_array.",
        "field_sampling": "Meep get_array linearly interpolates all three electric components from the Yee lattice onto common cell-center coordinates; exported fields are float32 samples of the final solver state.",
        "material_precedence": [object_id for object_id in ("substrate", "core") if object_id in by_id],
        "assumptions": [
            "Finite-height, full 3D, lossless nondispersive isotropic dielectric guide.",
            "Core extends to the domain edges and through the x PML; no periodic or invariant direction.",
            "Final signed electric-field components are actual sampled solver output; no volume time animation is stored.",
            "Material visibility is a preview setting and does not remove a solid from the solve.",
            "The Meep current sheet and FDTDX directed plane are different injections; cross-solver field amplitudes are not calibrated.",
            "A finite-source demonstration, not a pure eigenmode, normalized transmission, or convergence certificate.",
            "Exported field and permittivity volumes include the absorbing PML region.",
        ],
    }
