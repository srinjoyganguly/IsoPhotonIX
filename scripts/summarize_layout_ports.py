"""Collect native polygon-port matrices and compare them with independent measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


RUN_NAMES = ("elbow32", "elbow40", "shifted32", "straight-x", "straight-y", "tee32", "cross32")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def comparison(measured, reference):
    if measured.shape != reference.shape:
        raise ValueError("Cannot compare matrices with different dimensions.")
    visible = np.minimum(abs(measured), abs(reference)) >= .01
    phase = abs(np.angle(measured * reference.conj()))
    return {
        "complex_amplitude_error_max": float(abs(measured-reference).max()),
        "power_fraction_error_max": float(abs(abs(measured)**2-abs(reference)**2).max()),
        "phase_amplitude_floor": .01, "phase_coefficients_included": int(visible.sum()),
        "phase_error_rad_max": float(phase[visible].max()) if visible.any() else None,
    }


def fundamental_slab_index(wavelength, width):
    k0 = 2*np.pi/wavelength
    v = k0*width*np.sqrt(3.48**2-1.44**2)/2
    u = brentq(lambda u: u*np.tan(u)-np.sqrt(v*v-u*u), 1e-10, min(v, np.pi/2)-1e-10)
    return np.sqrt(3.48**2-(2*u/(k0*width))**2)


def load_native(folder, name, archive):
    report_path = folder/"result/report.json"
    report = json.loads(report_path.read_text())
    if report["case"] != "layout-ports-2d" or report["actual_device"] != "CPU":
        raise ValueError("Expected a native Meep CPU polygon-port result.")
    array_path = folder/"result/arrays.npz"
    with np.load(array_path, allow_pickle=False) as data:
        matrix = data["s"]
        reported = np.array(report["s_real"]) + 1j*np.array(report["s_imag"])
        np.testing.assert_array_equal(matrix, reported)
        np.testing.assert_allclose(abs(matrix)**2, report["power"], rtol=1e-14)
        count = len(report["channels"])
        if matrix.shape != (len(report["wavelength_um"]), count, count):
            raise ValueError("The archived matrix must contain one launch per port.")
        prefix = "" if name == "elbow32" else name+"_"
        for key in data.files:
            if key in ("s", "s_early", "wavelength_um") or key.endswith(("_coefficients", "_neff")):
                archive[prefix+key] = data[key]
        if not prefix:
            for key in ("port0-device_final_Ez", "port0-device_x_um", "port0-device_y_um"):
                archive[key] = data[key]
        stages = []
        for receipt_path in sorted((folder/"ckpt").glob("*/receipt.json")):
            receipt = json.loads(receipt_path.read_text())
            spec = receipt["specification"]
            if spec["settings"] != report["parameters"]:
                continue
            label = spec["label"]
            stage_path = receipt_path.with_name("arrays.npz")
            if receipt["arrays_sha256"] != digest(stage_path):
                raise ValueError(f"Invalid checkpoint digest for {name}/{label}.")
            with np.load(stage_path, allow_pickle=False) as stage:
                for key in stage.files:
                    np.testing.assert_array_equal(data[label+"_"+key], stage[key])
            stages.append({"launch_stage": label, "settings_hash": receipt["signature"],
                           "arrays_sha256": receipt["arrays_sha256"],
                           "source_sha256": spec["provenance"]["worker_sha256"]})
        expected = {f"port{i}-{kind}" for i in range(count) for kind in ("reference", "device")}
        if {stage["launch_stage"] for stage in stages} != expected or len(stages) != len(expected):
            raise ValueError(f"Missing a unique reference/device launch at {name}.")
    return {**report, "native_report_sha256": digest(report_path), "native_arrays_sha256": digest(array_path),
            "stage_digests": stages}


def plane_control(nominal, shifted, archive):
    distances = []
    for first, second in zip(nominal["parameters"]["design"]["ports"], shifted["parameters"]["design"]["ports"]):
        axis = 0 if first["outward"].endswith("x") else 1
        sign = 1 if first["outward"].startswith("+") else -1
        distances.append(sign*(second["center_um"][axis]-first["center_um"][axis]))
    waves = np.array(nominal["wavelength_um"])
    indices = archive["port0-device_neff"][:, 0, :].T
    phase = np.exp(2j*np.pi*indices*np.array(distances)/waves[:, None])
    predicted = archive["s"]*phase[:, :, None]*phase[:, None, :]
    archive["shifted32_prediction"] = predicted
    return {"outward_displacement_um": distances,
            "model": "Multiply by exp[i(beta_i delta_i + beta_j delta_j)] using native straight-lead eigenmode indices.",
            **comparison(archive["shifted32_s"], predicted)}


def straight_control(archive, nominal):
    report = {}
    for axis in ("x", "y"):
        prefix = "straight-"+axis+"_"
        s = archive[prefix+"s"]
        neff = archive[prefix+"port0-device_neff"][0, 0, :]
        ports = nominal["straight-"+axis]["parameters"]["design"]["ports"]
        distance = np.linalg.norm(np.subtract(ports[0]["center_um"], ports[1]["center_um"]))
        wavelengths = archive[prefix+"wavelength_um"]
        reference = np.exp(2j*np.pi*distance*neff/wavelengths)
        report[axis] = {"plane_separation_um": float(distance),
                        "min_transmitted_power": float(min(abs(s[:, 1, 0])**2)),
                        "max_reflected_power": float(np.max(abs(np.diagonal(s, axis1=1, axis2=2))**2)),
                        "phase_residual_vs_native_modal_index_rad": np.angle(s[:, 1, 0]*reference.conj()).tolist()}
    return {**report, "rotation_equivalence": comparison(archive["straight-x_s"], archive["straight-y_s"])}


def summarize(args):
    archive = {}
    records = {name: load_native(args.runs/f"v012-layout-{name}-final", name, archive) for name in RUN_NAMES}
    nominal = records["elbow32"]
    independent = {}
    vendor = json.loads(args.tidy_modes.read_text())
    vendor["sampling"] = "Uniform mesh, local mode solver, local subpixel averaging disabled."
    vendor["native_neff_by_mesh_and_axis"] = {}
    vendor["max_same_mesh_neff_error"] = {}
    vendor["analytic_isolated_slab_neff"] = [
        float(fundamental_slab_index(wave, vendor["width_um"])) for wave in nominal["wavelength_um"]]
    for mesh, path in [(32, args.reference32), (40, args.reference40)]:
        ref = json.loads(path.read_text())
        if (ref["geometry"] != nominal["parameters"]["design"] or ref["cells_per_um"] != mesh or
                ref["wavelength_um"] != nominal["wavelength_um"]):
            raise ValueError("The frequency-domain reference uses different physical inputs.")
        prefix = "" if mesh == 32 else "elbow40_"
        s = archive[prefix+"s"]
        reference = np.array(ref["s_real"])+1j*np.array(ref["s_imag"])
        archive[f"independent_elbow{mesh}_s"] = reference
        independent[str(mesh)] = {**comparison(s, reference), "solver": ref["solver"],
                                  "source_json_sha256": digest(path)}
        indices = archive[prefix+"port0-device_neff"][:, 0, :]
        vendor["native_neff_by_mesh_and_axis"][str(mesh)] = dict(zip(("x", "y"), indices.tolist()))
        other = vendor["neff_by_mesh_and_axis"][str(mesh)]
        vendor["max_same_mesh_neff_error"][str(mesh)] = float(abs(indices-np.array([other["x"],other["y"]])).max())
    controls = {"same_layout_mesh_32_to_40": comparison(archive["s"], archive["elbow40_s"]),
                "independent_helmholtz": independent,
                "independent_mesh_32_to_40": comparison(archive["independent_elbow32_s"],archive["independent_elbow40_s"]),
                "shifted_planes": plane_control(nominal, records["shifted32"], archive),
                "straight_lines": straight_control(archive, records)}
    filename = "v012-layout-ports-arrays.npz"
    np.savez_compressed(args.output/filename, **archive)
    report = {**nominal, "passed": all(r["passed"] for r in records.values()), "native_runs": records,
              "controls": controls, "tidy3d_straight_port_comparison": vendor,
              "reference_worker_sha256": digest(Path(__file__).with_name("reference_layout_ports.py")),
              "source_arrays": {"file": filename, "sha256": digest(args.output/filename)},
              "source_arrays_scope": "All native modal projections, indices and early/late complex matrices on all seven runs; one full final-time nominal device Ez plane. Prefix-free keys describe the default 32-cell elbow.",
              "field_preview": {"array": "port0-device_final_Ez", "index": [], "axis_order": "xy",
                                "component": "Ez", "horizontal_axis": "x", "vertical_axis": "y",
                                "horizontal_um": archive["port0-device_x_um"].tolist(),
                                "vertical_um": archive["port0-device_y_um"].tolist(),
                                "description": "Final-time Ez from the west-port launch on the 32-cell/µm elbow after the source and 100 µm/c of observation."}}
    (args.output/"v012-layout-ports.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("docs/verification"))
    parser.add_argument("--reference32", type=Path, required=True)
    parser.add_argument("--reference40", type=Path, required=True)
    parser.add_argument("--tidy-modes", type=Path, required=True)
    summarize(parser.parse_args())
