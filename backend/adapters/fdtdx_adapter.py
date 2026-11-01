"""FDTDX slab-waveguide adapter; imports are deferred to isolated workers.

The thin, periodic z extrusion follows the official FDTDX 2D tutorial.
This is a driven finite uniform plane source, not a solved eigenmode source.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import sys
import time


def _runtime(device: str):
    if device not in {"cpu", "gpu"}:
        raise ValueError("FDTDX requires an explicit cpu or gpu target")
    platform = "cuda" if device == "gpu" else "cpu"
    if "jax" in sys.modules:
        raise RuntimeError("FDTDX must run in a fresh worker before JAX is imported")
    os.environ["JAX_PLATFORMS"] = platform
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    import jax.numpy as jnp

    try:
        devices = jax.devices(platform)
        if not devices:
            raise RuntimeError("no devices found")
        target = devices[0]
        jax.config.update("jax_default_device", target)
        probe = jax.jit(lambda x: x * x)(jax.device_put(jnp.arange(4), target))
        probe.block_until_ready()
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise RuntimeError(
            f"Requested {platform} execution could not initialize: {detail}. "
            "CPU fallback is disabled. Check the selected Python environment, "
            "CUDA libraries, NVIDIA driver, and WSL GPU access."
        ) from exc
    if probe.device.platform != ("gpu" if device == "gpu" else "cpu"):
        raise RuntimeError(f"Requested {device}, but JAX executed on {probe.device}")
    import fdtdx
    return jax, jnp, fdtdx, target


def probe(device: str) -> dict:
    jax, _, _, target = _runtime(device)
    return {
        "available": True,
        "device": device,
        "actual_device": str(target),
        "device_name": target.device_kind,
        "versions": {"fdtdx": importlib.metadata.version("fdtdx"), "jax": jax.__version__,
                     "jaxlib": importlib.metadata.version("jaxlib")},
    }


def run(config: dict, output_dir: Path) -> dict:
    import numpy as np

    started = time.perf_counter()
    requested = config["device"]
    jax, jnp, fdtdx, target = _runtime(requested)
    print(f"FDTDX: verified {target.device_kind} ({target}); building model", flush=True)
    dx_um = 1.0 / config["resolution"]
    nx = round(config["length_um"] / dx_um)
    ny = round(config["height_um"] / dx_um)
    core_cells = round(config["core_width_um"] / dx_um)
    pml_cells = round(config["pml_um"] / dx_um)
    sim = fdtdx.SimulationConfig(time=config["duration_fs"] * 1e-15,
        resolution=dx_um * 1e-6, backend=requested, dtype=jnp.float32, courant_factor=0.99)
    if sim.backend != requested:
        raise RuntimeError(f"FDTDX changed {requested} to {sim.backend}; refusing fallback")
    volume = fdtdx.SimulationVolume(name="domain", partial_grid_shape=(nx, ny, 3),
        material=fdtdx.Material(permittivity=config["cladding_index"] ** 2))
    core = fdtdx.UniformMaterialObject(name="waveguide", partial_grid_shape=(nx, core_cells, 3),
        material=fdtdx.Material(permittivity=config["core_index"] ** 2))
    boundaries, constraints = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig(
        boundary_type_minz="periodic", boundary_type_maxz="periodic",
        thickness_grid_minx=pml_cells, thickness_grid_maxx=pml_cells,
        thickness_grid_miny=pml_cells, thickness_grid_maxy=pml_cells,
        thickness_grid_minz=1, thickness_grid_maxz=1), volume)
    source_x_um = -config["length_um"] / 2 + config["pml_um"] + 0.5
    source = fdtdx.UniformPlaneSource(name="excitation", partial_grid_shape=(1, core_cells, 3),
        fixed_E_polarization_vector=(0, 0, 1), direction="+", normalize_by_energy=False,
        temporal_profile=fdtdx.SingleFrequencyProfile(num_startup_periods=4),
        wave_character=fdtdx.WaveCharacter(wavelength=config["wavelength_um"] * 1e-6))
    interval = max(1, sim.time_steps_total // 24)
    detector = fdtdx.FieldDetector(name="field_monitor", partial_grid_shape=(nx, ny, 1),
        components=("Ez",), switch=fdtdx.OnOffSwitch(interval=interval))
    constraints = list(constraints)
    constraints.extend([core.place_at_center(volume), detector.place_at_center(volume),
        source.place_at_center(volume, margins=(source_x_um * 1e-6, 0, 0))])
    key = jax.random.PRNGKey(0)
    objects, arrays, params, sim, _ = fdtdx.place_objects(
        object_list=[volume, core, source, detector, *boundaries.values()],
        config=sim, constraints=constraints, key=key)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    jax.block_until_ready(arrays)
    if arrays.fields.E.device != target:
        raise RuntimeError("FDTDX arrays were not placed on the explicitly requested device")
    setup_s = time.perf_counter() - started
    print(f"FDTDX: compiling {nx} × {ny} × 3 grid, {sim.time_steps_total} steps", flush=True)
    compile_started = time.perf_counter()
    compiled = jax.jit(lambda a: fdtdx.run_fdtd(arrays=a, objects=objects,
        config=sim, key=key, show_progress=False)).lower(arrays).compile()
    compile_s = time.perf_counter() - compile_started
    print("FDTDX: running compiled electromagnetic solve", flush=True)
    compute_started = time.perf_counter()
    final_step, final_arrays = compiled(arrays)
    jax.block_until_ready(final_arrays)
    compute_s = time.perf_counter() - compute_started
    actual = final_arrays.fields.E.device
    if actual != target:
        raise RuntimeError("FDTDX output device does not match the requested execution target")
    field = np.asarray(final_arrays.fields.E[2, :, :, 1]).T
    epsilon = np.asarray(1.0 / final_arrays.inv_permittivities[2, :, :, 1]).T
    recorded = np.asarray(final_arrays.detector_states["field_monitor"]["fields"])
    frames = recorded[:, 0, :, :, 0].transpose(0, 2, 1)
    times_fs = np.arange(len(frames)) * interval * sim.time_step_duration * 1e15
    if not np.isfinite(field).all() or not np.isfinite(frames).all():
        raise RuntimeError("FDTDX produced non-finite fields")
    if not np.any(field):
        raise RuntimeError("FDTDX returned an identically zero field")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "arrays.npz", field=field, epsilon=epsilon,
        x_um=(np.arange(nx) + 0.5) * dx_um - nx * dx_um / 2,
        y_um=(np.arange(ny) + 0.5) * dx_um - ny * dx_um / 2,
        frames=frames, times_fs=times_fs)
    print(f"FDTDX: saved finite nonzero Ez fields; max |Ez|={float(np.abs(field).max()):.6g}", flush=True)
    return {
        "solver": "fdtdx", "device": requested, "actual_device": str(actual),
        "device_name": actual.device_kind,
        "versions": {"fdtdx": importlib.metadata.version("fdtdx"), "jax": jax.__version__,
                     "jaxlib": importlib.metadata.version("jaxlib"), "numpy": np.__version__},
        "field_component": "Ez", "field_units": "arbitrary source units",
        "grid_shape": [nx, ny, 3], "field_shape": list(field.shape), "precision": "float32",
        "time_steps": int(final_step), "time_step_fs": sim.time_step_duration * 1e15,
        "simulated_time_fs": int(final_step) * sim.time_step_duration * 1e15,
        "setup_seconds": setup_s, "compile_seconds": compile_s, "compute_seconds": compute_s,
        "total_seconds": time.perf_counter() - started,
        "source": {"type": "finite uniform plane, directed +x, not an eigenmode",
                   "x_um": source_x_um, "width_um": core_cells * dx_um,
                   "temporal_profile": "continuous sinusoid, linear turn-on over four periods",
                   "normalization": "unit electric polarization; no energy normalization"},
        "boundaries": "PML on x/y; three-cell periodic extrusion on z",
        "assumptions": ["2D invariant dielectric slab via periodic-z extrusion",
            "Lossless nondispersive isotropic indices", "No flux normalization or transmission claim",
            "FDTDX plane-source injection differs from Meep electric-current line",
            "Grid and PML convergence have not been established"],
        "max_abs_field": float(np.abs(field).max()),
    }
