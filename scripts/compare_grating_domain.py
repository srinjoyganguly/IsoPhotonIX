"""Check same-mesh linear observables before qualifying a reduced-domain or allocator control."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_finite_grating import file_hash


# Absolute fractional power, complex unit-incident field amplitude and radians.
# Use amplitude error for a weak return so a nearly zero phase cannot pass or fail a mesh.
LIMITS = {"power": 0.002, "complex_s21": 0.01, "phase_s21_rad": 0.025,
          "complex_return": 0.005, "angular_bin_power": 0.002, "incident_relative": 0.005, "upper_relative_l2": 0.02}


def read(path):
    if path.is_dir():
        receipt, arrays = path/"report.json", path/"arrays.npz"
    else:
        receipt, arrays = path, path.with_name(path.stem+"-arrays.npz")
    report = json.loads(receipt.read_text())
    if not report["passed"] or file_hash(arrays) != report["arrays_sha256"]:
        raise ValueError(f"Unqualified grating receipt or changed fields: {receipt}")
    return report, arrays


def compare(baseline, trial):
    left, a_path = read(baseline)
    right, b_path = read(trial)
    ignored = {"grating_domain", "grating_allocator"}
    for key in (left["settings"].keys() | right["settings"].keys()) - ignored:
        if left["settings"].get(key) != right["settings"].get(key):
            raise ValueError(f"Full and trial configurations differ in {key}.")
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        wave = a["ref_wavelength_um"]
        np.testing.assert_array_equal(wave, b["ref_wavelength_um"])
        for key in ("grid_x_um", "grid_y_um", "grid_z_um"):
            np.testing.assert_array_equal(a["ref_" + key], b["ref_" + key])
        inc_a, inc_b = a["incident_power"], b["incident_power"]
        sa, sb = (n["output_mode"]/np.sqrt(n["incident_power"]) for n in (a, b))
        diff = abs(sb-sa)
        phase = np.angle(sb*sa.conj())
        upper_error = []
        for stage in ("ref_", "dev_"):
            ua, ub = a[stage+"upper"], b[stage+"upper"]
            axes=tuple(range(1, ua.ndim))
            upper_error.append(np.sqrt(np.sum(abs(ub-ua)**2, axis=axes) / np.sum(abs(ua)**2, axis=axes)))
        stats = {"power":abs(b["coupling"]-a["coupling"]), "complex_s21":diff, "phase_s21_rad":abs(phase),
                 "incident_relative":abs(inc_b/inc_a-1), "upper_relative_l2":np.max(upper_error,axis=0)}
        if "reflected_fiber_mode" in a:
            stats["complex_return"] = abs(b["reflected_fiber_mode"]-a["reflected_fiber_mode"])
        else:
            delta = b["reflected_angular_fraction"] - a["reflected_angular_fraction"]
            stats["angular_bin_power"] = np.max(abs(delta), axis=1)
        exact_source = all(np.array_equal(a["ref_"+k], b["ref_"+k])
                           for k in ("source_E", "source_H", "source_time_offset_E", "source_time_offset_H"))
    match = exact_source and all(np.all(v <= LIMITS[k]) and np.all(np.isfinite(v)) for k,v in stats.items())
    return {"geometry":right["settings"]["geometry"], "resolution":right["settings"]["resolution"],
            "wavelength_um":wave.tolist(), "baseline_arrays_sha256":left["arrays_sha256"],
            "trial_arrays_sha256":right["arrays_sha256"], "identical_full_source":exact_source,
            "limits":{k:LIMITS[k] for k in stats}, "absolute_errors":{k:v.tolist() for k,v in stats.items()},
            "qualified":bool(match), "performance":{stage:rec.get("performance") for stage,rec in right["stages"].items()},
            "scope":"Same physical planes, unit-incident normalization, identical full source and output mode gauge. Fields include every stored upper-plane vector component."}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--trial",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    options=parser.parse_args()
    result=compare(options.baseline,options.trial)
    options.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("qualified =",result["qualified"])
