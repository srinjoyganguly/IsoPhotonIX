"""Actual Meep CPU FDTD adapter for the bundled two-dimensional slab.

API references: https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/
and https://meep.readthedocs.io/en/latest/Python_User_Interface/#array-slices
No mode solve, power normalization, or transmission calculation is performed.
"""
from __future__ import annotations

import platform
import math
from pathlib import Path
import time


# Meep uses c=1 and one spatial unit = 1 micrometre in this adapter.
C_UM_PER_FS = 0.299792458


def run(config: dict, output_dir: Path) -> dict:
    """Run Meep and save actual Ez/material samples in ``arrays.npz``.

    The caller owns schema validation and process isolation. The device guard is
    also enforced here so a direct adapter invocation cannot silently use CPU
    for a GPU request. Arrays use image order [y, x], coordinates are micrometres.
    """
    if config.get("device") != "cpu":
        raise ValueError("Meep supports CPU only in this application; select CPU explicitly.")

    started = time.perf_counter()
    import meep as mp
    import numpy as np

    mp.verbosity(0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    length = float(config["length_um"])
    height = float(config["height_um"])
    width = float(config["core_width_um"])
    wavelength = float(config["wavelength_um"])
    pml = float(config["pml_um"])
    resolution = int(config["resolution"])
    duration = float(config["duration_fs"]) * C_UM_PER_FS
    source_x = -length / 2 + pml + 0.5
    cell = mp.Vector3(length, height, 0)

    def source_profile(t):
        # Match FDTDX SingleFrequencyProfile's scalar four-period startup.
        # The physical injections still differ: current line versus E/H plane.
        ramp = min(1.0, max(0.0, t / (4 * wavelength)))
        return -math.cos(2 * math.pi * t / wavelength) * ramp

    print("Meep: initializing 2D Ez slab on explicit CPU target", flush=True)
    sim = mp.Simulation(
        cell_size=cell,
        dimensions=2,
        resolution=resolution,
        default_material=mp.Medium(index=float(config["cladding_index"])),
        geometry=[mp.Block(size=mp.Vector3(mp.inf, width, mp.inf),
                           center=mp.Vector3(),
                           material=mp.Medium(index=float(config["core_index"])))],
        boundary_layers=[mp.PML(pml)],
        sources=[mp.Source(mp.CustomSource(source_profile, start_time=0,
                                          center_frequency=1 / wavelength),
                           component=mp.Ez,
                           center=mp.Vector3(source_x, 0, 0),
                           size=mp.Vector3(0, width, 0))],
    )
    sim.init_sim()
    initialized = time.perf_counter()
    volume = mp.Volume(center=mp.Vector3(), size=cell)
    frames: list = []
    times_fs: list[float] = []

    def sample(simulation):
        frames.append(np.asarray(simulation.get_array(component=mp.Ez, vol=volume),
                                 dtype=np.float32).T.copy())
        times_fs.append(float(simulation.meep_time() / C_UM_PER_FS))

    sample(sim)
    print("Meep: evolving Maxwell fields and recording 24 time intervals", flush=True)
    sim.run(mp.at_every(duration / 24, sample), until=duration)
    # Retain the exact final solver state even if sampling times round to steps.
    if not times_fs or abs(times_fs[-1] - sim.meep_time() / C_UM_PER_FS) > 1e-6:
        sample(sim)
    solved = time.perf_counter()
    field = np.asarray(sim.get_array(component=mp.Ez, vol=volume), dtype=np.float32).T
    epsilon = np.asarray(sim.get_array(component=mp.Dielectric, vol=volume), dtype=np.float32).T
    x_um, y_um, _, _ = sim.get_array_metadata(vol=volume)
    x_um, y_um = np.asarray(x_um), np.asarray(y_um)
    if field.shape != (len(y_um), len(x_um)) or epsilon.shape != field.shape:
        raise RuntimeError("Meep field/material coordinates do not match the exported arrays.")
    if not np.isfinite(field).all() or not np.isfinite(epsilon).all():
        raise RuntimeError("Meep returned non-finite field or material samples.")
    if not np.any(field):
        raise RuntimeError("Meep returned an all-zero field; no result will be claimed.")
    np.savez_compressed(output_dir / "arrays.npz", field=field, epsilon=epsilon,
                        x_um=x_um, y_um=y_um, frames=np.stack(frames),
                        times_fs=np.asarray(times_fs))
    finished = time.perf_counter()
    print(f"Meep: complete; {field.shape[1]} x {field.shape[0]} field samples; "
          f"max |Ez| = {np.max(np.abs(field)):.6g}", flush=True)
    return {
        "solver": "meep",
        "device": "cpu",
        "actual_device": "CPU",
        "versions": {"meep": mp.__version__, "numpy": np.__version__,
                     "python": platform.python_version()},
        "timings": {"initialization_s": initialized - started,
                    "compute_s": solved - initialized,
                    "export_s": finished - solved,
                    "total_s": finished - started},
        "field_component": "Ez",
        "field_units": "arbitrary source amplitude",
        "field_shape": list(field.shape),
        "field_min": float(field.min()),
        "field_max": float(field.max()),
        "field_max_abs": float(np.max(np.abs(field))),
        "frame_count": len(frames),
        "actual_duration_fs": float(sim.meep_time() / C_UM_PER_FS),
        "grid_spacing_um": 1 / resolution,
        "precision": "Meep float64; exported field snapshots float32",
        "dimensions": "2D (invariant z)",
        "source": {"type": "uniform electric current line; continuous sinusoid",
                   "component": "Ez", "x_um": source_x, "y_um": 0,
                   "width_um": width, "wavelength_um": wavelength,
                   "amplitude": 1, "eigenmode": False,
                   "startup_periods": 4,
                   "temporal_profile": "-cos(2*pi*t/T) * clip(t/(4*T), 0, 1)"},
        "boundaries": {"x": "PML", "y": "PML", "z": "invariant", "pml_um": pml},
        "notes": ["Nondispersive, lossless isotropic dielectric slab.",
                  "Finite-time field snapshot; no convergence or transmission claim.",
                  "Meep injects an electric current line, unlike FDTDX's directed E/H plane source.",
                  "Absolute field amplitudes across the two solvers are not directly comparable.",
                  "The field map includes the absorbing PML region."],
    }
