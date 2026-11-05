"""Measure one independent input on a finer grid and compare with saved full device matrices."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "backend").is_dir():
    sys.path.insert(0, str(ROOT))
from scripts.calibrate_fdtdx_multiport import (
    _ports_runtime, build_plan, complex_components, compute_stage, grid, multiport_model, raster, save_record,
)


def first_column(model, ref, dev):
    """Apply the reference subtraction and outward directions to the single measured launch."""
    frequencies = model["frequency_per_um"]
    if not all(np.array_equal(data["frequency_per_um"], frequencies) for data in (ref, dev)):
        raise ValueError("The recorded frequencies differ from the selected model.")
    source_difference = max(float(np.linalg.norm(ref[name] - dev[name]) / np.linalg.norm(ref[name]))
                            for name in ("source_e", "source_h"))
    if not np.isfinite(source_difference) or source_difference > 1e-5:
        raise ValueError("The device and reference have different incident fields.")
    ports = model["ports"]
    incoming = 0 if ports[0]["inward_sign"] == 1 else 1
    ref_amplitudes = ref["amplitudes"]
    incident = ref_amplitudes[0, :, incoming]
    if np.any(abs(incident)**2 <= np.max(abs(incident)**2) * 1e-6):
        raise ValueError("A spectral sample lacks sufficient incident amplitude.")
    result = []
    for i, port in enumerate(ports):
        outward = 1 if port["inward_sign"] == 1 else 0
        outgoing = dev["amplitudes"][i, :, outward].copy()
        if i == 0:
            outgoing -= ref_amplitudes[0, :, outward]
        result.append(outgoing / incident)
    column = np.stack(result, axis=-1)
    through = ref_amplitudes[1, :, incoming] / incident
    if not (np.isfinite(column).all() and np.isfinite(through).all()):
        raise ValueError("Nonfinite guided-port projection.")
    return column, incident, through, source_difference


def compare_baseline(folder, plan, column):
    other = json.loads((folder / "report.json").read_text())
    if not (
        other["physical_model"] == plan["model"] and other["material_model"] == plan["material_model"]
        and other["actual_device"] == plan["actual_device"] and other["numerical_sha256"] == plan["numerical_sha256"]
        and other["sampled_setup_checks_passed"] and other["resolution"] != plan["resolution"]
    ):
        raise ValueError("Each baseline must be a passing complete matrix for this circuit, target, material model and another mesh.")
    with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
        previous = archive["s"][:, :, 0]
        if previous.shape != column.shape or not archive["valid_column"].all():
            raise ValueError("The baseline lacks the requested complete column.")
        delta = column - previous
        return {
            "baseline_resolution": other["resolution"],
            "max_complex_difference": float(np.max(abs(delta))),
            "complex_relative_l2_to_baseline": float(np.linalg.norm(delta) / np.linalg.norm(previous)),
            "max_port_power_difference": float(np.max(abs(abs(column)**2 - abs(previous)**2))),
            "baseline_center_guided_power": (abs(previous[1])**2).tolist(),
            "baseline_arrays_sha256": hashlib.sha256((folder / "arrays.npz").read_bytes()).hexdigest(),
        }


def measure(case, device, resolution, destination, baselines):
    runtime = _ports_runtime(device)
    target = runtime[-1]
    model = multiport_model(case)
    plan = build_plan(model, device, resolution, 80., target, "yee-aligned")
    destination.mkdir(parents=True, exist_ok=False)
    save_record(destination / "run.json", plan)
    bases, stages = {}, {}
    port = model["ports"][0]["name"]
    for role in ("reference", "device"):
        name = f"{port}-{role}"
        archive = destination / (name + ".npz")
        print(f"Starting {case} {resolution} {name}", flush=True)
        stages[role] = compute_stage(runtime, plan, 0, role == "reference", archive, bases)
        save_record(destination / (name + ".json"), stages[role])
        print(name, stages[role]["time_steps"], stages[role]["settled"], flush=True)
        runtime[0].clear_caches()
    with np.load(destination / (port + "-reference.npz")) as ref, np.load(destination / (port + "-device.npz")) as dev:
        column, incident, through, source_diff = first_column(model, ref, dev)
    power = abs(column)**2
    if not np.allclose(stages["reference"]["mode_neff"][port], stages["device"]["mode_neff"][port], rtol=1e-6, atol=1e-6):
        raise ValueError("Reference and device projected different input modes.")
    mask = raster(model, resolution)
    _, _, _, sampled_ports = grid(model, resolution)
    checks = {
        "both_stages_settled": all(stage["settled"] for stage in stages.values()),
        "straight_reference_power_within_0_03": bool(np.max(abs(abs(through)**2-1)) <= .03),
        "measured_column_total_guided_power_at_most_1_05": bool(power.sum(axis=1).max() <= 1.05),
        "requested_native_device": all(stage["field_device"] == str(target) for stage in stages.values()),
    }
    report = {
        "case": case, "resolution": resolution, "material_model": plan["material_model"],
        "actual_device": str(target), "device_name": target.device_kind, "native_versions": plan["versions"],
        "frequency_per_um": model["frequency_per_um"], "wavelength_um": (1/np.array(model["frequency_per_um"])).tolist(),
        "launch": port, "port_order": [item["name"] for item in model["ports"]], "s": complex_components(column),
        "matrix_axes": ["frequency", "outgoing_port"], "full_matrix": False, "modal_power": power.tolist(),
        "sum_observed_power": power.sum(axis=1).tolist(), "reference_through_power": (abs(through)**2).tolist(),
        "checks": checks, "passed": all(checks.values()), "sampled_ports": sampled_ports,
        "scalar_mask_mirror_disagreeing_voxels": int(np.count_nonzero(mask != mask[:, ::-1, :])),
        "reference_source_relative_difference": source_diff, "numerical_sha256": plan["numerical_sha256"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stage_sha256": {role: stage["archive_sha256"] for role, stage in stages.items()},
        "stage_step_counts": {role: stage["time_steps"] for role, stage in stages.items()},
        "stage_final_convergence": {role: stage["history"][-2:] for role, stage in stages.items()},
        "mesh_comparisons": [compare_baseline(folder, plan, column) for folder in baselines],
        "scope": "Independently measured first-input column; other input ports were not launched in this mesh control. Reciprocity and full-matrix passivity are outside this column's scope.",
    }
    np.savez_compressed(destination / "arrays.npz", s_column=column, incident=incident, reference_through=through,
                        frequency_per_um=model["frequency_per_um"])
    report["arrays_sha256"] = hashlib.sha256((destination / "arrays.npz").read_bytes()).hexdigest()
    save_record(destination / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("coupler", "splitter"), required=True)
    parser.add_argument("--resolution", type=int, choices=(12, 16, 20, 24), default=24)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--compare", action="append", default=[], type=Path, help="Complete differently sampled FDTDX study directory.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.case, args.device, args.resolution, args.output, args.compare)
    if not result["passed"]:
        raise SystemExit("The measured column failed its sampled checks.")


if __name__ == "__main__":
    main()
