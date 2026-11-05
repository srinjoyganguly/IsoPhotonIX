"""Compare full CPU and CUDA field archives from the same FDTDX multiport study."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_report(folder, target):
    with (folder / "report.json").open() as source:
        data = json.load(source)
    if data["requested_device"] != target:
        raise ValueError(f"The {target} input selects a different device.")
    expected = "cpu:" if target == "cpu" else "cuda:"
    if not data["actual_device"].startswith(expected) or not data["sampled_setup_checks_passed"]:
        raise ValueError(f"The {target} study requires the selected hardware and passing sampled checks.")
    return data


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def aggregate(reference, candidate, names):
    """Combine entire field planes, using the CPU record's field scale."""
    differences, squared_norms, largest, peak = [], [], 0., 0.
    for name in names:
        cpu = reference[name].astype(np.complex128)
        gpu = candidate[name].astype(np.complex128)
        if cpu.shape != gpu.shape or not (np.isfinite(cpu).all() and np.isfinite(gpu).all()):
            raise ValueError(f"Invalid native array {name}.")
        delta = gpu - cpu
        differences.append(np.vdot(delta.ravel(), delta.ravel()).real)
        squared_norms.append(np.vdot(cpu.ravel(), cpu.ravel()).real)
        largest = max(largest, float(np.max(abs(delta))))
        peak = max(peak, float(np.max(abs(cpu))))
    if not names or sum(squared_norms) == 0 or peak == 0:
        raise ValueError("The selected native fields contain no signal.")
    return {
        "relative_l2": float(np.sqrt(sum(differences) / sum(squared_norms))),
        "max_difference_over_cpu_peak": largest / peak,
    }


def compare(cpu_folder, gpu_folder):
    cpu = load_report(cpu_folder, "cpu")
    gpu = load_report(gpu_folder, "gpu")
    for key in ("case", "physical_model", "resolution", "grid_shape", "port_order", "frequency_per_um",
                "sampled_ports", "numerical_sha256", "after_source_cap_um_over_c", "versions"):
        if cpu[key] != gpu[key]:
            raise ValueError(f"The CPU and GPU studies differ in {key}.")
    material_model = cpu.get("material_model", "legacy-x")
    if material_model != gpu.get("material_model", "legacy-x"):
        raise ValueError("The material models differ.")
    if cpu["stages"].keys() != gpu["stages"].keys():
        raise ValueError("The two targets have different launches.")
    matrix_paths = [folder / "arrays.npz" for folder in (cpu_folder, gpu_folder)]
    with np.load(matrix_paths[0], allow_pickle=False) as a, np.load(matrix_paths[1], allow_pickle=False) as b:
        if not (a["valid_column"].all() and b["valid_column"].all()):
            raise ValueError("The measured matrix contains an unqualified launch.")
        cpu_s, gpu_s = a["s"], b["s"]
        shape = (len(cpu["frequency_per_um"]), len(cpu["port_order"]), len(cpu["port_order"]))
        if cpu_s.shape != shape or gpu_s.shape != shape or not (np.isfinite(cpu_s).all() and np.isfinite(gpu_s).all()):
            raise ValueError("The measured matrix is incomplete.")
        matrix = {
            "complex_s_relative_l2": float(np.linalg.norm(gpu_s-cpu_s) / np.linalg.norm(cpu_s)),
            "max_complex_s_difference": float(np.max(abs(gpu_s-cpu_s))),
            "max_modal_power_difference": float(np.max(abs(abs(gpu_s)**2 - abs(cpu_s)**2))),
        }
    checks = {key: value <= 2e-5 for key, value in matrix.items()}
    stages = {}
    for name, cpu_stage in cpu["stages"].items():
        gpu_stage = gpu["stages"][name]
        if not (cpu_stage["settled"] and gpu_stage["settled"]):
            raise ValueError("An input stage failed its convergence limits.")
        if cpu_stage["field_device"] != cpu["actual_device"] or gpu_stage["field_device"] != gpu["actual_device"]:
            raise ValueError(f"The native fields are on an unexpected device at {name}.")
        for key in ("material_grid_sha256", "yee_component_grid_sha256"):
            if cpu_stage.get(key) != gpu_stage.get(key):
                raise ValueError(f"Different stepped materials at {name}.")
        archives = [folder / (name + ".npz") for folder in (cpu_folder, gpu_folder)]
        hashes = [digest(path) for path in archives]
        if hashes != [stage["archive_sha256"] for stage in (cpu_stage, gpu_stage)]:
            raise ValueError(f"Native field archive checksum mismatch at {name}.")
        with np.load(archives[0], allow_pickle=False) as a, np.load(archives[1], allow_pickle=False) as b:
            if set(a.files) != set(b.files) or not np.array_equal(a["frequency_per_um"], b["frequency_per_um"]):
                raise ValueError(f"The two spectral records differ in structure at {name}.")
            source = aggregate(a, b, ["source_e", "source_h"])
            fields = aggregate(a, b, [field for field in a.files if field.startswith("phasors_")])
        stages[name] = {
            "cpu_raw_sha256": hashes[0], "gpu_raw_sha256": hashes[1],
            "field_device_cpu": cpu_stage["field_device"], "field_device_gpu": gpu_stage["field_device"],
            "cpu_steps": cpu_stage["time_steps"], "gpu_steps": gpu_stage["time_steps"],
            "source_profile": source, "native_six_component_monitor_planes": fields,
        }
    checks["same_time_steps"] = all(s["cpu_steps"] == s["gpu_steps"] for s in stages.values())
    checks["source_profiles"] = all(max(s["source_profile"].values()) <= 1e-5 for s in stages.values())
    checks["native_monitor_planes"] = all(max(s["native_six_component_monitor_planes"].values()) <= 1e-4
                                          for s in stages.values())
    return {
        "case": cpu["case"], "resolution": cpu["resolution"], "material_model": material_model,
        "port_order": cpu["port_order"], "matrix_order": cpu["matrix_axes"], "frequency_per_um": cpu["frequency_per_um"],
        "native_versions": cpu["versions"], "numerical_sha256": cpu["numerical_sha256"],
        "cuda_device": gpu["device_name"], "cpu_array_sha256": digest(matrix_paths[0]),
        "gpu_array_sha256": digest(matrix_paths[1]), **matrix, "stages": stages,
        "limits": {"matrix_metrics": 2e-5, "source_metrics": 1e-5, "native_field_metrics": 1e-4},
        "checks": checks, "passed": all(checks.values()),
        "scope": "Execution-target repeatability at the same mesh, materials, terminal times and source. The limits are tolerances on this hardware comparison; they do not measure mesh convergence or agreement with commercial results.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output file.")
    result = compare(args.cpu, args.gpu)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    if not result["passed"]:
        raise SystemExit("The targets failed their matched numerical comparison.")


if __name__ == "__main__":
    main()
