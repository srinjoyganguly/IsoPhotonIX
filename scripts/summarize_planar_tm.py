"""Summarize native multi-mesh TM matrices without rerunning the field solver."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def phase_statistics(a, b, reflected, ceiling=None):
    """Use a declared signal floor when reporting phase of a complex coefficient."""
    strong = np.minimum(abs(a),abs(b)) >= .01
    # Rows share a physical side for the reflected blocks.
    sides = np.arange(4)//2
    same_side = sides[:,None] == sides[None,:]
    visible = strong & (same_side if reflected else ~same_side)
    if ceiling is not None:
        visible &= np.maximum(abs(a), abs(b)) < ceiling
    delta = np.abs(np.angle(a*b.conj()))
    return {"included":int(visible.sum()),"signal_amplitude_floor":.01, "signal_amplitude_ceiling":ceiling,
            "max_phase_difference_rad":float(delta[visible].max()) if visible.any() else None}


def native_stages(folder, report, data):
    stages = []
    for path in sorted(folder.glob("*/receipt.json")):
        receipt = json.loads(path.read_text())
        spec = receipt["specification"]
        if spec["settings"] != report["parameters"]:
            continue
        label = spec["label"]
        samples = path.with_name("arrays.npz")
        if digest(samples) != receipt["arrays_sha256"]:
            raise ValueError(f"Invalid native sample digest for {label}.")
        with np.load(samples, allow_pickle=False) as recorded:
            for key in recorded.files:
                saved_key = key if key in ("x_um", "y_um") else label+"_"+key
                np.testing.assert_array_equal(recorded[key], data[saved_key])
        stages.append({"launch_stage":label, "settings_hash":receipt["signature"],
                       "arrays_sha256":receipt["arrays_sha256"], "source_sha256":spec["provenance"]["worker_sha256"]})
    expected = {f"{i}-{kind}" for i in range(4) for kind in ("reference", "device")}
    if {stage["launch_stage"] for stage in stages} != expected or len(stages) != len(expected):
        raise ValueError("Missing a unique native reference/device stage for a guided mode launch.")
    return stages


def summarize(root, output):
    rows, matrices, archive = [], [], {}
    for mesh in (48,64,80):
        folder = root/f"v012-tm{mesh}"/"result"
        report_file, samples, reference = (folder/n for n in ("report.json","arrays.npz","reference.npz"))
        report = json.loads(report_file.read_text())
        if report["parameters"]["resolution"] != mesh or report["actual_device"] != "CPU":
            raise ValueError("Unexpected native model or execution target.")
        with np.load(samples,allow_pickle=False) as data, np.load(reference,allow_pickle=False) as independent:
            matrix = data["s"][0]
            shifted = data["s"][1]
            np.testing.assert_allclose(matrix.real,report["s_real"],rtol=0,atol=1e-15)
            np.testing.assert_allclose(matrix.imag,report["s_imag"],rtol=0,atol=1e-15)
            archive[f"mesh{mesh}_s"] = data["s"]
            archive[f"mesh{mesh}_reference_s"] = independent["s"]
            archive[f"mesh{mesh}_phase_prediction"] = independent["plane_shift_prediction"]
            for key in data.files:
                if key.endswith(("_coefficients","_neff")):
                    archive[f"mesh{mesh}_{key}"] = data[key]
            if mesh == 64:
                for key in ("coarse", "wider", "analytic_neff"):
                    archive["independent_"+key] = independent[key]
                for key in data.files:
                    if key in ("0-device_final_Hz","x_um","y_um","wavelength_um","planes_um"):
                        archive[key] = data[key]
            rows.append({"mesh":mesh,"passed":report["passed"],"parameters":report["parameters"],
                "stage_source_and_profile_digests":native_stages(folder.parent/"ckpt", report, data),
                "complex_reciprocity_residual_max":float(abs(matrix-matrix.transpose(0,2,1)).max()),
                "reflection_phase_vs_independent":phase_statistics(matrix, independent["s"], True),
                "weak_reflection_phase_vs_independent":phase_statistics(matrix, independent["s"], True, .1),
                "forward_phase_vs_independent":phase_statistics(matrix, independent["s"], False),
                "plane_shift_measured_power_difference":float(abs(abs(matrix)**2-abs(shifted)**2).max()),
                "raw_report_sha256":digest(report_file),"raw_native_arrays_sha256":digest(samples),
                "reference_arrays_sha256":digest(reference),"checks":report["checks"],"limits":report["limits"]})
            matrices.append(matrix.copy())
    comparisons = []
    for index in (0,1):
        a, b = matrices[index:index+2]
        comparisons.append({"meshes":[rows[index]["mesh"],rows[index+1]["mesh"]],
            "power_fraction_change_max":float(abs(abs(a)**2-abs(b)**2).max()),
            "complex_amplitude_change_max":float(abs(a-b).max()),
            "reflection_phase":phase_statistics(a,b,True),"transmission_phase":phase_statistics(a,b,False)})

    filename = "v012-planar-tm-arrays.npz"
    np.savez_compressed(output/filename, **archive)
    reference = json.loads((root/"v012-tm64"/"result"/"report.json").read_text())
    record = {**reference,"passed":all(row["passed"] for row in rows),"meshes":rows,"mesh_sensitivity":comparisons,
              "field_preview":{"array":"0-device_final_Hz","index":[],"axis_order":"xy","component":"Hz",
                  "horizontal_axis":"x","vertical_axis":"y","horizontal_um":archive["x_um"].tolist(),
                  "vertical_um":archive["y_um"].tolist(),
                  "description":"Final-time Hz from the left TM0 launch on the 64-cell/µm offset joint after the source and 100 µm/c of observation."},
              "source_arrays":{"file":filename,"sha256":digest(output/filename)},
              "source_arrays_scope":"All native modal coefficients and indices on three meshes and two planes; the full left-TM0 device Hz field on the default 64-cell mesh. Full per-mesh native files can be regenerated using the recorded settings.",
              "source_worker_sha256":digest(Path(__file__).with_name("calibrate_planar_tm.py"))}
    (output/"v012-planar-tm.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=Path("runs"))
    parser.add_argument("--output",type=Path,default=Path("docs/verification"))
    args = parser.parse_args()
    result = summarize(args.runs,args.output)
    print("Default TM case pass:",result["passed"])
    print("Mesh changes:",result["mesh_sensitivity"])
