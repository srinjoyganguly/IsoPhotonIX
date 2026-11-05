"""Measure slab propagation phase from unchanged product FDTD worker outputs.

This is a diagnostic on sparse saved frames at a known source frequency, not
an eigensolver, source normalization, broadband spectrum, or accuracy certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.model import WaveguideConfig
from backend.mode_benchmarks import analytic_te0
from backend.runtime import ROOT, clean_log, interpreter, worker_env

SOURCE_URL = "https://doc.comsol.com/6.3/doc/com.comsol.help.models.woptics.dielectric_slab_waveguide/dielectric_slab_waveguide.html"
C_UM_PER_FS = 0.299792458


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harmonic_phase_fit(times_fs, frames, x_um, *, wavelength_um=1.55,
                       start_fs=100., x_min_um=-1., x_max_um=2.):
    """Fit actual samples to a known frequency, then the unwrapped spatial phase.

    ``frames`` is [time,x] on one saved transverse coordinate. Nothing is
    resampled. The phasor convention is Re(F exp(+i omega t)), so a forward
    traveling wave has arg(F) = -beta*x + constant.
    """
    times = np.asarray(times_fs, dtype=np.float64)
    values = np.asarray(frames, dtype=np.float64)
    x = np.asarray(x_um, dtype=np.float64)
    if (times.ndim != 1 or x.ndim != 1 or values.shape != (len(times), len(x))
            or not all(np.isfinite(a).all() for a in (times, values, x))):
        raise ValueError("Finite time/x coordinates and matching [time,x] samples are required.")
    if not np.all(np.diff(times) > 0) or not np.all(np.diff(x) > 0):
        raise ValueError("Time and x coordinates must be strictly increasing.")
    if not np.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("Wavelength must be positive and finite.")
    ti = np.flatnonzero(times >= start_fs)
    xi = np.flatnonzero((x >= x_min_um) & (x <= x_max_um))
    if len(ti) < 6 or len(xi) < 8:
        raise ValueError("At least six late-time frames and eight spatial samples are required.")
    t, xx = times[ti], x[xi]
    omega = 2*np.pi*C_UM_PER_FS/wavelength_um
    if t[-1]-t[0] < 2*wavelength_um/C_UM_PER_FS:
        raise ValueError("Selected time coverage must span at least two optical periods.")
    data = values[np.ix_(ti, xi)]
    design = np.column_stack((np.cos(omega*t), np.sin(omega*t), np.ones(len(t))))
    condition = float(np.linalg.cond(design))
    if np.linalg.matrix_rank(design) != 3 or condition > 1e6:
        raise ValueError("Saved times cannot identify independent cosine, sine and DC coefficients.")
    coefficients = np.linalg.lstsq(design, data, rcond=None)[0]
    phasor = coefficients[0]-1j*coefficients[1]
    magnitude = np.abs(phasor)
    norm = float(np.linalg.norm(data))
    if norm == 0 or float(np.max(magnitude)) <= 1e-12*float(np.max(np.abs(data))):
        raise ValueError("No nonzero oscillating field is available for phase measurement.")
    if float(np.min(magnitude)) <= 1e-12*float(np.max(magnitude)):
        raise ValueError("An oscillating-field node makes spatial phase undefined.")
    phase = np.unwrap(np.angle(phasor))
    slope, intercept = np.polyfit(xx, phase, 1)
    fitted_phase = slope*xx+intercept
    temporal_residual = data-design@coefficients
    beta = float(-slope)
    metrics = {
        "requested_time_start_fs": start_fs,
        "actual_time_window_fs": [float(t[0]), float(t[-1])],
        "selected_frame_indices": ti.tolist(), "selected_times_fs": t.tolist(),
        "selected_x_indices": xi.tolist(),
        "requested_x_window_um": [x_min_um, x_max_um],
        "actual_x_window_um": [float(xx[0]), float(xx[-1])],
        "time_sample_count": len(t), "spatial_sample_count": len(xx),
        "known_angular_frequency_rad_per_fs": float(omega),
        "temporal_design_condition_number": condition,
        "temporal_relative_l2_residual": float(np.linalg.norm(temporal_residual)/norm),
        "maximum_pointwise_temporal_relative_l2_residual": float(np.max(
            np.linalg.norm(temporal_residual, axis=0)/np.linalg.norm(data, axis=0))),
        "dc_coefficient_peak_abs": float(np.max(np.abs(coefficients[2]))),
        "phasor_peak_abs": float(np.max(magnitude)),
        "phasor_min_abs": float(np.min(magnitude)),
        "phase_slope_rad_per_um": float(slope), "phase_intercept_rad": float(intercept),
        "phase_rmse_rad": float(np.sqrt(np.mean((phase-fitted_phase)**2))),
        "phase_max_abs_residual_rad": float(np.max(np.abs(phase-fitted_phase))),
        "beta_rad_per_um": beta, "effective_index": beta*wavelength_um/(2*np.pi),
        "phase_convention": "Re(F exp(+i omega t)); forward phase slope is -beta",
        "metric_scope": "Known-frequency driven centerline phase diagnostic; no mode decomposition or accuracy pass threshold",
    }
    arrays = {"x_um": xx, "times_fs": t, "samples": data,
              "phasor": phasor, "dc": coefficients[2], "phase_rad": phase,
              "phase_fit_rad": fitted_phase, "temporal_fit": design@coefficients}
    return metrics, arrays


def make_config(solver, device, resolution, duration_fs):
    return WaveguideConfig(wavelength_um=1.55, core_width_um=1., core_index=1.5,
        cladding_index=1., length_um=8., height_um=9., resolution=resolution,
        duration_fs=duration_fs, pml_um=1., solver=solver, device=device).model_dump()


def transverse_projection(times_fs, frames, x_um, y_um):
    """Scalar overlap with a prescribed analytic TE0 profile, not modal power.

    The template uses the reference index only in the transverse profile. The
    longitudinal phase slope remains a measurement of the actual solver field.
    """
    reference = analytic_te0(1.5, 1., 1.)
    k0 = 2*np.pi/1.55
    h = k0*np.sqrt(1.5**2-reference**2)
    alpha = k0*np.sqrt(reference**2-1.)
    y = np.asarray(y_um)
    profile = np.where(abs(y) <= .5, np.cos(h*y), np.cos(h*.5)*np.exp(-alpha*(abs(y)-.5)))
    yi = np.flatnonzero((y >= -3.5)&(y <= 3.5))
    if len(yi) < 8:
        raise ValueError("Too few non-PML transverse samples for the profile projection.")
    yy, phi = y[yi], profile[yi]
    normalization = np.trapezoid(phi**2, yy)
    projected_frames = np.trapezoid(frames[:, yi, :]*phi[None, :, None], yy, axis=1)/normalization
    measured, derived = harmonic_phase_fit(times_fs, projected_frames, x_um)
    ti = np.asarray(measured["selected_frame_indices"])
    t = np.asarray(times_fs)[ti]
    omega = measured["known_angular_frequency_rad_per_fs"]
    design = np.column_stack((np.cos(omega*t), np.sin(omega*t), np.ones(len(t))))
    ny, nx = frames.shape[1:]
    coefficients = np.linalg.lstsq(design, frames[ti].reshape(len(t), -1), rcond=None)[0].reshape(3, ny, nx)
    full_phasor = coefficients[0]-1j*coefficients[1]
    projected_phasor = np.trapezoid(full_phasor[yi]*phi[:, None], yy, axis=0)/normalization
    xi = np.asarray(measured["selected_x_indices"])
    projection_fit_difference = float(np.max(abs(projected_phasor[xi]-derived["phasor"])))
    if not np.allclose(projected_phasor[xi], derived["phasor"], rtol=1e-10, atol=1e-12):
        raise ValueError("Projecting the harmonic phasor disagrees with fitting the projected raw samples.")
    measured.update({"observable": "analytic-TE0-template scalar transverse overlap phase",
        "reference_neff": reference,
        "relative_difference_from_reference": abs(measured["effective_index"]-reference)/reference,
        "requested_y_window_um": [-3.5, 3.5],
        "actual_y_window_um": [float(yy[0]), float(yy[-1])], "selected_y_indices": yi.tolist(),
        "quadrature": "Trapezoidal integration on actual saved y coordinates; divide by integral(template squared). No resampling.",
        "template": {"kind": "even symmetric-slab analytic TE0", "core_width_um": 1.,
            "reference_neff_used_for_transverse_shape": reference,
            "core_transverse_wavenumber_per_um": float(h), "cladding_decay_per_um": float(alpha)},
        "projection_vs_temporal_fit_max_abs_difference": projection_fit_difference,
        "limitation": "Uses a known analytic transverse template; does not independently identify the mode or calculate E/H-normalized modal power. Finite transverse window and radiation/reflection contamination remain."})
    derived.update(full_phasor=full_phasor, full_dc=coefficients[2],
        full_x_um=np.asarray(x_um), full_y_um=y,
        transverse_template=profile, projected_phasor_all_x=projected_phasor)
    return measured, derived


def inspect_run(directory, config):
    saved = json.loads((directory/"config.json").read_text())
    metadata = json.loads((directory/"metadata.json").read_text())
    if saved != config or metadata.get("config") != config:
        raise ValueError("Saved configuration differs from this benchmark case.")
    if config["device"] == "gpu" and not metadata["actual_device"].startswith("cuda:"):
        raise ValueError("Requested GPU run does not record actual CUDA execution.")
    with np.load(directory/"arrays.npz", allow_pickle=False) as arrays:
        y = arrays["y_um"]
        yi = int(np.argmin(abs(y)))
        primary, derived = harmonic_phase_fit(arrays["times_fs"], arrays["frames"][:, yi, :], arrays["x_um"])
        later, _ = harmonic_phase_fit(arrays["times_fs"], arrays["frames"][:, yi, :], arrays["x_um"], start_fs=125.)
        narrower, _ = harmonic_phase_fit(arrays["times_fs"], arrays["frames"][:, yi, :], arrays["x_um"], x_min_um=-.5)
        primary.update({"selected_y_index": yi, "selected_y_um": float(y[yi]),
            "recorded_frame_count": len(arrays["times_fs"]),
            "core_epsilon_along_fit_region": np.unique(arrays["epsilon"][yi, primary["selected_x_indices"]]).tolist()})
        if not np.allclose(primary["core_epsilon_along_fit_region"], 2.25):
            raise ValueError("Centerline fit region is not inside the specified dielectric core.")
        projected, projected_arrays = transverse_projection(arrays["times_fs"], arrays["frames"], arrays["x_um"], arrays["y_um"])
    reference = analytic_te0(1.5, 1., 1.)
    primary["reference_neff"] = reference
    primary["absolute_difference_from_reference"] = abs(primary["effective_index"]-reference)
    primary["relative_difference_from_reference"] = abs(primary["effective_index"]-reference)/reference
    derived_path = directory/"propagation-diagnostic.npz"
    # Analysis may be regenerated, but original solver arrays are never altered.
    np.savez_compressed(derived_path, **derived)
    np.savez_compressed(directory/"transverse-projection.npz", **projected_arrays)
    return {"status": "measured-not-certified", "config": config,
        "actual_device": metadata["actual_device"], "device_name": metadata.get("device_name"),
        "versions": metadata["versions"], "source": metadata["source"],
        "boundaries": metadata["boundaries"],
        "actual_final_time_fs": metadata.get("actual_duration_fs", metadata.get("simulated_time_fs")),
        "time_step_fs": metadata["time_step_fs"], "time_steps": metadata["time_steps"],
        "fit": primary, "late_window_check_125fs": later,
        "spatial_window_check_minus_0p5_to_2um": narrower,
        "analytic_transverse_projection": projected,
        "file_sha256": {name: sha256(directory/name) for name in
                        ("config.json", "metadata.json", "arrays.npz", "propagation-diagnostic.npz", "transverse-projection.npz")},
        "adapter_sha256": sha256(ROOT/"backend/adapters"/f"{config['solver']}_adapter.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[30, 40, 45])
    parser.add_argument("--targets", nargs="+", choices=["meep-cpu", "fdtdx-cpu", "fdtdx-gpu"], default=["meep-cpu", "fdtdx-gpu"])
    parser.add_argument("--duration-fs", type=float, default=180.)
    parser.add_argument("--reuse", action="store_true", help="Analyze existing completed case directories only when their exact configuration matches.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "case": "comsol-slab-product-fdtd-propagation",
        "generated_at": datetime.now(timezone.utc).isoformat(), "source_url": SOURCE_URL,
        "reference_kind": "analytic TE0 slab dispersion described by COMSOL; no numerical COMSOL runtime",
        "reference_neff": analytic_te0(1.5, 1., 1.), "runs": [],
        "reproduction_command": ".venv/bin/python scripts/benchmark_slab_propagation.py --output .local/slab-propagation-rerun --report .local/slab-propagation-rerun/report.json --resolutions 30 40 45 --targets meep-cpu fdtdx-gpu",
        "method": "Known-frequency cosine+sine+DC least squares on actual saved frames at nearest y=0; unweighted linear fit to unwrapped spatial phasor phase.",
        "limitations": [
            "Uses existing schema-version-1 product workers and adapters without altering their source or field sampling.",
            "COMSOL's example uses numerical ports; these adapters use a finite uniform current line (Meep) or directed E/H plane (FDTDX), neither a solved eigenmode.",
            "The 1 µm symmetric slab supports additional polarization/parity families; Ez, symmetric excitation preferentially excites the even TE fundamental, but no modal projection proves purity.",
            "Radiation, reflections and other mode content can bias the centerline phase; small temporal fit residual does not establish spatial mode purity.",
            "The additional transverse projection uses a fixed analytic TE0 shape over non-PML y; its dependence on the reference is explicit, and it is not a power decomposition or independent identification of a mode.",
            "Sparse saved snapshots do not resolve an unconstrained temporal spectrum. The fit prescribes the independently known drive frequency and uses actual recorded times.",
            "Meep is native 2D; FDTDX is a three-cell periodic-z extrusion. This does not validate the finite-height 3D adapters.",
            "No amplitude matching, source-power calibration, power or transmission claim, pass threshold, or commercial parity claim.",
            "Resolution, temporal-window and spatial-window sensitivities are retained; a finite sweep is not a rigorous error bound."]}
    failures = 0
    for resolution in args.resolutions:
        for target in args.targets:
            solver, device = target.split("-")
            config = make_config(solver, device, resolution, args.duration_fs)
            directory = args.output/f"{target}-r{resolution}"
            try:
                if directory.exists():
                    if not args.reuse:
                        raise ValueError("Case output already exists; choose a new output or explicitly --reuse matching completed runs.")
                else:
                    directory.mkdir()
                    (directory/"config.json").write_text(json.dumps(config, indent=2)+"\n")
                    result = subprocess.run([interpreter(solver, device), "-m", "backend.worker",
                        str((directory/"config.json").resolve()), str(directory.resolve())],
                        cwd=ROOT, env=worker_env(solver, device), capture_output=True, text=True, timeout=300)
                    (directory/"worker.log").write_text(clean_log(result.stdout+"\n"+result.stderr))
                    if result.returncode:
                        raise RuntimeError(f"Solver worker failed with exit {result.returncode}; inspect worker.log.")
                measured = inspect_run(directory, config)
                measured["id"] = directory.name
                report["runs"].append(measured)
                print(f"{directory.name}: neff={measured['fit']['effective_index']:.10f}; relative discrepancy={measured['fit']['relative_difference_from_reference']:.6g}", flush=True)
            except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
                failures += 1
                report["runs"].append({"id": directory.name, "status": "failed", "config": config, "error": clean_log(str(error))})
                print(f"{directory.name}: failed ({type(error).__name__})", flush=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
