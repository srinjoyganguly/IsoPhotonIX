"""Compare a fresh monitored CUDA notebook with the previous acquisition at the identical mesh."""
import argparse
import json
from pathlib import Path

import numpy as np

from backend.calibration_replay import verify_source
from scripts.calibrate_finite_grating import file_hash
from scripts.lumerical_grating3d import reflected_fiber_amplitude
from scripts.verify_finite_grating import verify
from scripts.verify_vendor_grating_notebooks import check_executed_notebook


LIMITS = {
    "field_relative_l2": 5e-5,
    "coupling_percentage_points": .001,
    "complex_amplitude": 1e-4,
    "phase_rad": .001,
    "phase_amplitude_floor": .005,
}


def field_errors(old, new, component_axis):
    """Compare E and H independently, retaining the worst single wavelength."""
    a = np.moveaxis(old, component_axis, 0)
    b = np.moveaxis(new, component_axis, 0)
    if a.shape != b.shape or a.shape[0] != 6:
        raise ValueError("Field extents or vector components differ.")
    errors = {}
    for label, components in (("E", slice(0, 3)), ("H", slice(3, 6))):
        first, second = a[components], b[components]
        if component_axis == 1:
            first, second = first.swapaxes(0, 1), second.swapaxes(0, 1)
        else:
            first, second = first[None], second[None]
        relative = []
        for baseline, current in zip(first, second):
            baseline = baseline.astype(np.complex128, copy=False)
            current = current.astype(np.complex128, copy=False)
            denominator = np.linalg.norm(baseline)
            if denominator == 0:
                if np.any(current):
                    raise ValueError(f"An originally zero {label} field became nonzero.")
                relative.append(0.)
            else:
                relative.append(float(np.linalg.norm(current - baseline) / denominator))
        errors[label] = relative
    return errors


def channel_change(a, b):
    valid = (abs(a) >= LIMITS["phase_amplitude_floor"]) & (abs(b) >= LIMITS["phase_amplitude_floor"])
    phase = np.angle(b * a.conj())
    return {"max_complex_change": float(np.max(abs(b-a))),
            "phase_reportable": valid.tolist(),
            "phase_difference_rad": [float(value) if ok else None for ok, value in zip(valid, phase)],
            "max_reportable_phase_change_rad": float(np.max(abs(phase[valid]))) if np.any(valid) else None}


def compare_notebook(notebook_folder, baseline, sampled_memory=None):
    check_executed_notebook(notebook_folder)
    manifest = verify_source(notebook_folder)
    run = notebook_folder / "notebook-result"
    original, current = verify(baseline), verify(run)
    if current["settings"] != original["settings"]:
        raise ValueError("The physical settings differ between acquisitions.")
    model = current["settings"]
    performance = {}
    for label, stage in current["stages"].items():
        previous = original["stages"][label]
        recorded = stage["performance"]
        if stage["acquisition"]["status"] != "fresh" or recorded["platform"] != "gpu":
            raise ValueError("This proof requires a fresh CUDA acquisition for both stages.")
        if any(previous["performance"][key] != recorded[key] for key in
               ("cells", "grid_shape", "spacing_um", "pml_cells", "steps", "dt_seconds", "device_kind")):
            raise ValueError("Timing cannot compare different native grids, integration times or devices.")
        if stage["native_worker_sha256"] != manifest["files"]["source/scripts/calibrate_finite_grating.py"]:
            raise ValueError("The acquisition worker differs from the exported program.")
        if (stage["signature"] == previous["signature"] or
                recorded["acquisition_id"] == previous["performance"]["acquisition_id"]):
            raise ValueError("Expected independently acquired, source-bound stages.")
        performance[label] = {"old_acquisition_id": previous["performance"]["acquisition_id"],
                              "previous_stage_sha256": previous["arrays_sha256"],
                              "new_stage_sha256": stage["arrays_sha256"],
                              "acquisition": stage["acquisition"], "performance": recorded,
                              "memory_admission": stage["memory_admission"]}
    field_difference = {}
    with np.load(baseline / "arrays.npz", allow_pickle=False) as old, np.load(run / "arrays.npz", allow_pickle=False) as new:
        if not set(old.files) <= set(new.files) or "dev_reciprocal_return_history" not in new:
            raise ValueError("Expected all historical fields plus complex return time samples.")
        for key in old.files:
            if any(token in key for token in ("_grid_", "_material_", "_wavelength_", "_upper_x_um", "_upper_y_um")) or key == "valid":
                np.testing.assert_array_equal(old[key], new[key])
        for key, axis in (("ref_upper", 1), ("dev_upper", 1), ("dev_output", 1), ("dev_center_fields", 0)):
            field_difference[key] = field_errors(old[key], new[key], axis)
        coupling_change = 100 * (new["coupling"] - old["coupling"])
        guide = channel_change(old["output_mode"] / np.sqrt(old["incident_power"]),
                               new["output_mode"] / np.sqrt(new["incident_power"]))
        incident = old["ref_upper"]
        old_return = reflected_fiber_amplitude(old["dev_upper"] - incident, incident, old["incident_power"], 1/model["resolution"])
        samples = new["dev_reciprocal_return_history"]
        reflected = channel_change(old_return, samples[-1])
        time_trace = {"device_time_fs": [row["time_fs"] for row in current["stages"]["device"]["history"]],
                      "real": samples.real.tolist(), "imag": samples.imag.tolist(),
                      "qualification": current["radiation"]["reciprocal_return_settling"]}
        max_field = max(max(errors) for group in field_difference.values() for errors in group.values())
    channel_match = all(ch["max_complex_change"] <= LIMITS["complex_amplitude"] and
                        (ch["max_reportable_phase_change_rad"] is None or ch["max_reportable_phase_change_rad"] <= LIMITS["phase_rad"])
                        for ch in (guide, reflected))
    stable = bool(max_field <= LIMITS["field_relative_l2"] and np.max(abs(coupling_change)) <= LIMITS["coupling_percentage_points"] and channel_match)
    memory = {}
    if sampled_memory:
        for label in performance:
            with sampled_memory.open() as stream:
                rows = [r for line in stream if (r := json.loads(line)) and
                        [model["geometry"], model["resolution"], label] in r["jobs"]]
            memory[label] = {"samples": len(rows), "peak_observed_global_used_mib": max((r["used_mib"] for r in rows), default=None),
                             "scope": "2 s read-only GPU-wide samples, including other graphics allocations."}
    return {
        "geometry": model["geometry"], "domain": model["grating_domain"], "resolution": model["resolution"],
        "wavelength_um": current["result"]["wavelength_um"], "native_qualified": current["passed"],
        "same_mesh_regression_passed": stable, "cross_run_limits": LIMITS,
        "original_arrays_sha256": original["arrays_sha256"], "new_arrays_sha256": current["arrays_sha256"],
        "manifest_sha256": file_hash(notebook_folder/"manifest.json"),
        "worker_sha256": manifest["files"]["source/scripts/calibrate_finite_grating.py"],
        "unchanged_notebook_sha256": file_hash(notebook_folder/"calibration.ipynb"),
        "native_stages": performance, "sampled_memory": memory, "fields_relative_l2_by_wavelength": field_difference,
        "coupling_original": original["result"]["beam_to_guide"], "coupling_current": current["result"]["beam_to_guide"],
        "coupling_difference_percentage_points": coupling_change.tolist(), "guide_channel": guide,
        "reciprocal_return_channel": reflected, "return_time_history": time_trace,
        "scope": "Matched independent acquisitions on one physical mesh. Return settling is qualified per wavelength. Computational repeatability and temporal settling are separate from mesh and commercial accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("notebook", "original", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--sampled-memory", type=Path)
    args = parser.parse_args()
    proof = compare_notebook(args.notebook, args.original, args.sampled_memory)
    args.output.write_text(json.dumps(proof, indent=2, allow_nan=False)+"\n")
    if not proof["native_qualified"] or not proof["same_mesh_regression_passed"]:
        raise SystemExit("Inspect the measured native or cross-run regression gate.")
