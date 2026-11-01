"""Finite-height 3D FDTDX FDTD on an explicitly verified CPU or NVIDIA target.

Uses the documented SimulationVolume/UniformMaterialObject/UniformPlaneSource
and six-face PML APIs, checked against installed FDTDX 0.6.2. Placement is
quantized explicitly and the actual material grid is checked before evolution.
"""
from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path
import time

from .fdtdx_adapter import _runtime


def cell_center_fields(raw_fields):
    """Average actual Yee samples to shared cell centres without edge extrapolation.

    Input/output order is [component,z,y,x]. FDTDX's documented Yee offsets are
    Ex=(.5,0,0), Ey=(0,.5,0), Ez=(0,0,.5). Each component is averaged across the
    other two axes. The positive outermost cell is omitted on all spatial axes.
    """
    import numpy as np

    values = np.asarray(raw_fields)
    if values.ndim != 4 or values.shape[0] != 3 or min(values.shape[1:]) < 2:
        raise ValueError("Expected three Yee components with at least two samples on every axis.")
    ex, ey, ez = values
    return np.stack([
        (ex[:-1, :-1, :-1] + ex[1:, :-1, :-1] + ex[:-1, 1:, :-1] + ex[1:, 1:, :-1]) * .25,
        (ey[:-1, :-1, :-1] + ey[1:, :-1, :-1] + ey[:-1, :-1, 1:] + ey[1:, :-1, 1:]) * .25,
        (ez[:-1, :-1, :-1] + ez[:-1, 1:, :-1] + ez[:-1, :-1, 1:] + ez[:-1, 1:, 1:]) * .25,
    ])


def grid_layout(config: dict) -> dict:
    """Translate the supported straight guide into explicit integer cell bounds."""
    if config.get("schema_version") != 2 or config.get("study_type") != "waveguide-3d":
        raise ValueError("Expected a version 2 finite-height waveguide study.")
    geometry = config["geometry"]
    objects = geometry["objects"]
    by_id = {obj["id"]: obj for obj in objects}
    if len(by_id) != len(objects) or "core" not in by_id or set(by_id) - {"core", "substrate"}:
        raise ValueError("The 3D adapter supports one core box and an optional substrate box.")
    if any(obj["kind"] != "box" or abs(obj["rotation_deg"] % 360) > 1e-8 for obj in objects):
        raise ValueError("This study requires axis-aligned boxes without rotation.")
    resolution = config["resolution"]
    domain = [float(geometry["domain"][axis]) for axis in "xyz"]
    if not all(math.isfinite(value) and value > 0 for value in domain) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("Domain and resolution must be finite and positive.")
    shape = [round(value * resolution) for value in domain]
    dx = 1.0 / resolution
    core = by_id["core"]
    if any(abs(core["position"][axis]) > 1e-8 for axis in "xyz") or not math.isclose(core["size"]["x"], domain[0], abs_tol=1e-8):
        raise ValueError("The core must be centered and span the full domain along x.")
    core_shape = [shape[0], round(core["size"]["y"] * resolution), round(core["size"]["z"] * resolution)]
    core_start = [0, (shape[1] - core_shape[1]) // 2, (shape[2] - core_shape[2]) // 2]
    pml_cells = round(config["pml_um"] * resolution)
    if min(core_shape[1:]) < 4 or pml_cells < 8:
        raise ValueError("Core width/height need at least four cells and PML at least eight cells.")
    if any(core_start[axis] - pml_cells < 4 or shape[axis] - core_start[axis] - core_shape[axis] - pml_cells < 4 for axis in (1, 2)):
        raise ValueError("The core needs at least four interior cladding cells on each transverse side.")
    placement = {"core": {"start": core_start, "shape": core_shape}}
    if "substrate" in by_id:
        substrate = by_id["substrate"]
        expected_height = (domain[2] - core["size"]["z"]) / 2
        expected_center = -domain[2] / 2 + expected_height / 2
        if any(not math.isclose(substrate["size"][axis], domain[index], abs_tol=1e-8) or abs(substrate["position"][axis]) > 1e-8 for index, axis in enumerate("xy")) or not math.isclose(substrate["size"]["z"], expected_height, abs_tol=1e-8) or not math.isclose(substrate["position"]["z"], expected_center, abs_tol=1e-8):
            raise ValueError("Substrate must fill domain x/y and touch the core bottom from the domain's lower z face.")
        # One shared interface prevents rounding the two touching solids into a gap.
        placement["substrate"] = {"start": [0, 0, 0], "shape": [shape[0], shape[1], core_start[2]]}
    source_x = -domain[0] / 2 + config["pml_um"] + .5
    source_index = round((source_x + shape[0] * dx / 2) / dx)
    if source_index <= pml_cells or source_index >= shape[0] - pml_cells - 1:
        raise ValueError("Source must be inside the propagation domain, outside the PML.")
    return {"shape": shape, "dx_um": dx, "pml_cells": pml_cells,
            "placement": placement, "source_start": [source_index, core_start[1], core_start[2]],
            "source_shape": [1, core_shape[1], core_shape[2]], "requested_source_x_um": source_x}


def run(config: dict, output_dir: Path) -> dict:
    import numpy as np

    layout = grid_layout(config)
    started = time.perf_counter()
    requested = config["device"]
    jax, jnp, fdtdx, target = _runtime(requested)
    print(f"FDTDX 3D: verified {target.device_kind} ({target}); constructing finite-height guide", flush=True)
    shape = tuple(layout["shape"])
    dx = layout["dx_um"]
    sim = fdtdx.SimulationConfig(time=config["duration_fs"] * 1e-15,
        resolution=dx * 1e-6, backend=requested, dtype=jnp.float32, courant_factor=.99)
    if sim.backend != requested:
        raise RuntimeError(f"FDTDX changed requested {requested} to {sim.backend}; fallback refused.")
    volume = fdtdx.SimulationVolume(name="domain", partial_grid_shape=shape,
        material=fdtdx.Material(permittivity=config["cladding_index"] ** 2))
    pml = layout["pml_cells"]
    boundary_config = fdtdx.BoundaryConfig(**{f"thickness_grid_{side}{axis}": pml for side in ("min", "max") for axis in "xyz"},
        **{f"boundary_type_{side}{axis}": "pml" for side in ("min", "max") for axis in "xyz"})
    boundaries, constraints = fdtdx.boundary_objects_from_config(boundary_config, volume)
    constraints = list(constraints)
    material_objects = []
    by_id = {obj["id"]: obj for obj in config["geometry"]["objects"]}
    object_records = []
    # Explicit precedence, though the two supported boxes do not overlap.
    for order, name in enumerate(("substrate", "core")):
        if name not in by_id:
            continue
        spec, placed = by_id[name], layout["placement"][name]
        obj = fdtdx.UniformMaterialObject(name=name, partial_grid_shape=tuple(placed["shape"]),
            placement_order=order, material=fdtdx.Material(permittivity=spec["refractive_index"] ** 2))
        material_objects.append(obj)
        constraints.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=tuple(placed["start"])))
        low = [(placed["start"][i] - shape[i] / 2) * dx for i in range(3)]
        high = [(placed["start"][i] + placed["shape"][i] - shape[i] / 2) * dx for i in range(3)]
        object_records.append({"id": name, "requested_size_um": spec["size"], "requested_position_um": spec["position"],
            "grid_start_xyz": placed["start"], "grid_shape_xyz": placed["shape"], "actual_min_um": low,
            "actual_max_um": high, "refractive_index": spec["refractive_index"], "visibility_ignored_for_physics": True})
    source = fdtdx.UniformPlaneSource(name="excitation", partial_grid_shape=tuple(layout["source_shape"]),
        fixed_E_polarization_vector=(0, 1, 0), direction="+", normalize_by_energy=False,
        temporal_profile=fdtdx.SingleFrequencyProfile(num_startup_periods=4),
        wave_character=fdtdx.WaveCharacter(wavelength=config["wavelength_um"] * 1e-6))
    constraints.append(source.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=tuple(layout["source_start"])))
    key = jax.random.PRNGKey(0)
    objects, arrays, params, sim, _ = fdtdx.place_objects(
        object_list=[volume, *material_objects, source, *boundaries.values()],
        config=sim, constraints=constraints, key=key)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    jax.block_until_ready(arrays)
    if arrays.fields.E.device != target:
        raise RuntimeError("FDTDX's 3D input fields are on a different execution device.")
    epsilon_xyz = np.asarray(1.0 / arrays.inv_permittivities[0])
    expected = np.full(shape, config["cladding_index"] ** 2, dtype=np.float32)
    for name in ("substrate", "core"):
        if name in by_id:
            placed = layout["placement"][name]
            slices = tuple(slice(start, start + count) for start, count in zip(placed["start"], placed["shape"]))
            expected[slices] = by_id[name]["refractive_index"] ** 2
    material_error = float(np.max(np.abs(epsilon_xyz - expected)))
    if epsilon_xyz.shape != shape or not np.allclose(epsilon_xyz, expected, rtol=1e-6, atol=1e-6):
        raise RuntimeError("The initialized FDTDX material grid differs from the explicit geometry translation.")
    setup_s = time.perf_counter() - started
    print(f"FDTDX 3D: material grid verified; compiling {shape} cells, {sim.time_steps_total} time steps, six-face PML", flush=True)
    compile_started = time.perf_counter()
    compiled = jax.jit(lambda a: fdtdx.run_fdtd(arrays=a, objects=objects, config=sim, key=key, show_progress=False)).lower(arrays).compile()
    compile_s = time.perf_counter() - compile_started
    compute_started = time.perf_counter()
    print("FDTDX 3D: executing compiled Maxwell time evolution", flush=True)
    final_step, final_arrays = compiled(arrays)
    jax.block_until_ready(final_arrays)
    compute_s = time.perf_counter() - compute_started
    actual = final_arrays.fields.E.device
    if actual != target:
        raise RuntimeError("FDTDX's 3D result fields are on a different execution device; fallback refused.")
    raw_fields = np.asarray(final_arrays.fields.E).transpose(0, 3, 2, 1).copy()
    raw_epsilon = np.asarray(1.0 / final_arrays.inv_permittivities[0]).transpose(2, 1, 0).copy()
    if not np.isfinite(raw_fields).all() or not np.isfinite(raw_epsilon).all():
        raise RuntimeError("FDTDX returned non-finite raw 3D samples, including the outer grid cells.")
    fields = cell_center_fields(raw_fields)
    epsilon = raw_epsilon[:-1, :-1, :-1].copy()
    if not np.isfinite(fields).all() or not np.isfinite(epsilon).all() or not np.any(fields[1]):
        raise RuntimeError("FDTDX returned invalid or zero 3D electric fields.")
    axes = {f"{axis}_um": (np.arange(count - 1) + .5) * dx - count * dx / 2 for axis, count in zip("xyz", shape)}
    raw_axes = {f"raw_{axis}_um": np.arange(count) * dx - count * dx / 2 for axis, count in zip("xyz", shape)}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "arrays.npz", field=fields[1], fields=fields, epsilon=epsilon,
        raw_fields=raw_fields, raw_epsilon=raw_epsilon, **axes, **raw_axes)
    peaks = {name: float(np.max(np.abs(values))) for name, values in zip(("Ex", "Ey", "Ez"), fields)}
    print(f"FDTDX 3D: saved actual volumes {fields.shape[1:]}; max |Ey|={peaks['Ey']:.6g}", flush=True)
    return {"solver": "fdtdx", "device": requested, "actual_device": str(actual), "device_name": actual.device_kind,
        "versions": {"fdtdx": importlib.metadata.version("fdtdx"), "jax": jax.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"), "numpy": np.__version__},
        "dimensions": "3D", "field_component": "Ey", "field_components": ["Ex", "Ey", "Ez"],
        "field_units": "arbitrary source units", "array_order": "fields[component,z,y,x]; field and epsilon[z,y,x]",
        "grid_shape": list(shape), "field_shape": list(fields.shape[1:]), "precision": "float32",
        "field_sampling": "Four neighbouring Yee samples averaged spatially to each cell centre; positive outermost x/y/z cell omitted to avoid extrapolation. Raw staggered fields and material grid are preserved in the export. No temporal interpolation.",
        "material_sampling": "Scalar isotropic cell material values, cropped to the same common-centre grid as fields.",
        "grid_spacing_um": dx, "field_peaks": peaks, "time_steps": int(final_step),
        "time_step_fs": sim.time_step_duration * 1e15, "simulated_time_fs": int(final_step) * sim.time_step_duration * 1e15,
        "setup_seconds": setup_s, "compile_seconds": compile_s, "compute_seconds": compute_s,
        "total_seconds": time.perf_counter() - started,
        "source": {"type": "finite uniform E/H plane directed +x; not an eigenmode", "component": "Ey",
            "requested_x_um": layout["requested_source_x_um"], "x_um": (layout["source_start"][0] - shape[0] / 2) * dx,
            "grid_start_xyz": layout["source_start"],
            "grid_shape_xyz": layout["source_shape"], "wavelength_um": config["wavelength_um"],
            "temporal_profile": "continuous sinusoid, linear turn-on over four periods",
            "normalization": "unit electric polarization; no energy normalization"},
        "boundaries": {"x": "PML", "y": "PML", "z": "PML", "pml_cells": pml, "actual_pml_um": pml * dx},
        "geometry_translation": object_records, "material_grid_check": {"passed": True, "maximum_absolute_error": material_error},
        "assumptions": ["Finite-height rectangular dielectric guide; full 3D Maxwell time evolution.",
            "Isotropic, lossless nondispersive media. Grid-quantized material boxes, no subpixel smoothing.",
            "Source injects a finite uniform E/H plane; it is not a solved waveguide eigenmode.",
            "Field amplitudes are not calibrated against Meep's electric-current sheet.",
            "No normalized transmission, mode purity, or scientific convergence claim.",
            "Displayed fields are explicit spatial averages to common cell centres; raw fields retain staggered Yee locations in the export.",
            "The saved volume includes PML cells; visibility flags do not omit physical materials."],
        "yee_component_offsets_cells": {"Ex": [.5, 0, 0], "Ey": [0, .5, 0], "Ez": [0, 0, .5]},
        "raw_coordinate_convention": "raw_x/y/z_um are voxel-origin coordinates; add component-specific Yee offsets times grid_spacing_um to locate raw field samples."}
