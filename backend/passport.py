"""Inspectable run records and stored-frame access, without recomputing physics."""
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np


class FramesUnavailable(LookupError):
    pass


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return {"name": path.name, "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


@lru_cache(maxsize=2)
def _stored_frames(filename, modified_ns, size):
    # NPZ members are compressed: cache two immutable frame arrays so scrubbing
    # does not repeatedly decompress every saved frame on each request.
    del modified_ns, size
    with np.load(filename, allow_pickle=False) as arrays:
        if "frames" not in arrays or "times_fs" not in arrays:
            raise FramesUnavailable("This run did not record time frames; its final snapshot remains available.")
        frames, times = arrays["frames"], arrays["times_fs"]
        shape = arrays["field"].shape
    if frames.ndim != 3 or frames.shape[1:] != shape or times.ndim != 1 or len(times) != len(frames):
        raise ValueError("Stored frame dimensions or times are inconsistent with the final field.")
    if not len(frames):
        raise FramesUnavailable("This run has no recorded time frames.")
    if not np.isfinite(frames).all() or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Stored frames or recorded times contain invalid values.")
    frames.setflags(write=False)
    times.setflags(write=False)
    return frames, times, float(np.max(np.abs(frames)))


def stored_frames(output):
    path = Path(output) / "arrays.npz"
    stat = path.stat()
    return _stored_frames(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def build_passport(job, output):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    is3d = config.get("study_type") == "waveguide-3d"
    isports = config.get("study_type") == "waveguide-ports-2d"
    snapshot_ok = metadata.get("config") == config and job["config"] == config
    if (isports or job["config"].get("study_type") == "waveguide-ports-2d") and not snapshot_ok:
        raise ValueError("Port result configuration differs between saved files and the run snapshot.")
    component = "Ey" if is3d else "Ez"
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        field, epsilon = arrays["field"], arrays["epsilon"]
        x, y = arrays["x_um"], arrays["y_um"]
        finite = bool(all(np.isfinite(a).all() for a in (field, epsilon, x, y)))
        shape_ok = bool(field.ndim == 2 and epsilon.shape == field.shape and x.ndim == y.ndim == 1
                        and field.shape == (len(y), len(x)))
        if is3d:
            from .volume import validate_volume
            validate_volume(arrays)
            shape_ok = True
            finite = True
        nonzero = bool(np.any(field)) if np.isfinite(field).all() else False
        frame_count = int(len(arrays["frames"])) if "frames" in arrays else 0
        coordinate_detail = f"{len(x)} x and {len(y)} y samples" if not is3d else f"{len(x)} x, {len(y)} y and {len(arrays['z_um'])} z samples"
    files = [file_record(output / name) for name in ("config.json", "metadata.json", "arrays.npz")]
    if isports:
        from .spectra import stored_spectrum, spectrum_csv
        spectra = stored_spectrum(output,metadata)
        if (output/'spectrum.csv').read_bytes() != spectrum_csv(spectra).encode():
            raise ValueError('Saved CSV differs from the recorded spectrum.')
        files.extend(file_record(output/name) for name in ('spectra.npz','spectrum.csv'))
        files.extend(file_record(output/name) for name in ('reference-monitors.npz','device-monitors.npz') if (output/name).exists())
    checks = [
        {"label": "Finite numerical values", "passed": finite,
         "detail": f"Final {component}, permittivity, and coordinate arrays contain no NaN or infinite values."},
        {"label": "Nonzero final field", "passed": nonzero,
         "detail": f"At least one final {component} sample is nonzero; this does not establish convergence."},
        {"label": "Matching field and material grids", "passed": shape_ok,
         "detail": f"Field shape {list(field.shape)}; coordinates contain {coordinate_detail}."},
        {"label": "Consistent saved configuration", "passed": snapshot_ok,
         "detail": "Config file, saved worker metadata, and run snapshot agree."},
    ]
    limitations = [
        "Educational lossless, nondispersive slab; grid and PML convergence have not been established.",
        "Ez uses arbitrary source units. No normalized transmission, S-parameters, or mode-purity claim.",
        "Meep current-line and FDTDX directed-plane source amplitudes are not cross-solver calibrated.",
        "SHA256 identifies these exported file bytes; it is not a signature or an accuracy certificate.",
    ]
    if is3d:
        limitations = [
            "Commercial solver agreement has not been validated. No matched Lumerical, Tidy3D or COMSOL reference run is attached.",
            "Finite-height lossless, nondispersive straight waveguide. Resolution/PML sensitivity is not a convergence certificate.",
            "Ex/Ey/Ez are final-time electric fields in arbitrary source units; no normalized transmission, S-parameters or mode-purity result.",
            "Meep current-sheet and FDTDX directed-plane sources are not cross-solver amplitude calibrated.",
            str(metadata.get("field_sampling", "Read metadata for component sampling and material-grid conventions.")),
            "SHA256 identifies exported bytes; it is not a signature or an accuracy certificate.",
        ]
    if isports:
        checks[1] = {'label':'Measured nonzero incident reference','passed':bool(np.any(spectra['incident_flux']>0)),
                     'detail':'The straight-guide pulse produces measured power. A zero final ringdown field is allowed.'}
        checks.append({'label':'Reference and device settling','passed':all(metadata['settlement'][s]['settled'] for s in ('reference','device')),
                       'detail':'Each stage must satisfy its requested energy and spectrum criteria; a failed check leaves its spectrum unqualified. This is not mesh convergence.'})
        checks.append({'label':'Qualified spectral bins','passed':bool(np.all(spectra['valid_bin'])),
                       'detail':f"{int(np.sum(spectra['valid_bin']))} of {len(spectra['valid_bin'])} bins have settled runs and sufficient incident power. Unqualified bins remain marked."})
        limitations = ['2D invariant-z waveguide, lossless nondispersive materials. This does not qualify finite-height 3D port measurements.',
                       'R and T are total power through finite monitor apertures normalized to a separate straight-guide incident run; they are not complex modal S-parameters.',
                       '1-R-T is uncollected flux, including lateral radiation and numerical/monitor error; it is not absorption.',
                       'Final Ez ringdown samples use arbitrary source units. Their amplitude is not a transmission measurement.',
                       'The CSV retains raw ratios with valid_bin flags; unqualified bins must not be treated as settled measurements.',
                       'CPU source-mode preparation may accompany GPU FDTD; execution metadata identifies both.',
                       'SHA256 identifies exported bytes, not accuracy. Grid, boundary and source sensitivity require independent checks.']
    assumptions = metadata.get("assumptions", metadata.get("notes", metadata.get('limitations',[])))
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    command = ".venv/bin/python scripts/run_simulation.py config.json --output runs/reproduced-run"
    notes = ["Run from the repository root after copying config.json from this export.",
             "Use a new output directory and install the recorded solver/device environment first.",
             "The command preserves the explicit solver and device; it never switches GPU to CPU."]
    timings = metadata.get("timings") or {key: metadata[key] for key in
        ("setup_seconds", "compile_seconds", "compute_seconds", "total_seconds") if key in metadata}
    passport = {
        "schema_version": 1, "title": "Waveguide run passport", "run_id": job["id"],
        "created_at": job.get("created_at"), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "solver": metadata.get("solver", config["solver"]),
        "requested_device": config["device"], "actual_device": metadata.get("actual_device", "Not recorded"),
        "device_name": metadata.get("device_name", metadata.get("actual_device", "Not recorded")),
        "versions": metadata.get("versions", {}), "source": metadata.get("source", {}),
        "boundaries": metadata.get("boundaries", {}), "source_assumptions": assumptions,
        "numerical": {"field_component": metadata.get("field_component", "Ez"),
            "field_units": metadata.get("field_units", "Not recorded"), "field_shape": list(field.shape),
            "frame_count": frame_count, "requested_duration_fs": config.get("duration_fs"),
            "actual_duration_fs": metadata.get("simulated_time_fs", metadata.get("actual_duration_fs")),
            "precision": metadata.get("precision", "Not recorded"), "timings": timings},
        "checks": checks, "files": files,
        "array_sha256": next(record["sha256"] for record in files if record["name"] == "arrays.npz"),
        "reproduce": {"command": command, "notes": notes}, "reproduce_command": command,
        "reproduction_notes": notes, "limitations": limitations,
    }
    if is3d:
        passport["numerical"].update(dimensions="3D", components=["Ex", "Ey", "Ez"],
                                      array_order="component,z,y,x", field_sampling=metadata.get("field_sampling"))
    if isports:
        passport['numerical'].update(study_type='waveguide-ports-2d',maximum_after_source_fs=config['max_after_sources_fs'],
            field_time_fs=metadata.get('field_time_fs'), frequency_bins=len(spectra['valid_bin']),
            valid_frequency_bins=int(np.sum(spectra['valid_bin'])), settlement=metadata['settlement'],
            normalization=metadata.get('normalization',metadata.get('monitor',{})),
            array_order='Final field[y,x]; spectra indexed by increasing wavelength; native complex monitor streams are solver-specific.')
    manifest = {"schema_version": 1, "algorithm": "sha256", "run_id": job["id"],
                "files": {record["name"]: {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}
                          for record in files}}
    return passport, manifest


def passport_html(passport):
    """Standalone, escaped HTML; no network, scripts, fonts, or external assets."""
    def rendered(value):
        return escape(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False))

    checks = "".join(f"<li><strong>{'PASS' if check['passed'] else 'FAIL'} · {rendered(check['label'])}</strong>"
                     f"<br>{rendered(check['detail'])}</li>" for check in passport["checks"])
    files = "".join(f"<tr><td>{rendered(item['name'])}</td><td>{item['size_bytes']:,}</td>"
                    f"<td><code>{rendered(item['sha256'])}</code></td></tr>" for item in passport["files"])
    limits = "".join(f"<li>{rendered(value)}</li>" for value in passport["limitations"])
    sections = "".join(f"<h2>{title}</h2><pre>{rendered(passport[key])}</pre>" for title, key in
                        [("Physical configuration · µm and fs", "config"), ("Numerical record", "numerical"),
                         ("Solver and runtime versions", "versions"), ("Source", "source"),
                         ("Boundaries", "boundaries"), ("Recorded source assumptions", "source_assumptions")])
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Waveguide run passport</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:980px;margin:40px auto;padding:0 24px;color:#172b3a;background:#f7fafb}}h1{{line-height:1.15}}h2{{margin-top:30px}}.tag{{color:#176854;font-weight:bold}}pre,code{{font-family:ui-monospace,monospace;overflow-wrap:anywhere;white-space:pre-wrap}}pre{{background:white;border:1px solid #d7e2e7;border-radius:8px;padding:16px}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #d7e2e7}}li{{margin-bottom:10px}}small{{color:#4c6473}}</style>
<p class="tag">PHOTONICS WORKBENCH · LOCAL COMPUTE</p><h1>Waveguide run passport</h1>
<p>An inspectable record of actual saved solver output. These checks describe the files, not scientific convergence.</p>
<p><strong>Run:</strong> {rendered(passport['run_id'])}<br><strong>Solver:</strong> {rendered(passport['solver'])}<br>
<strong>Requested target:</strong> {rendered(passport['requested_device'])}<br><strong>Recorded actual device:</strong> {rendered(passport['actual_device'])} · {rendered(passport['device_name'])}<br>
<strong>Created:</strong> {rendered(passport['created_at'])}</p>
<h2>Observed checks</h2><ul>{checks}</ul>{sections}
<h2>File manifest · SHA256</h2><p>These hashes identify the original result files included with this passport.</p>
<table><thead><tr><th>File</th><th>Bytes</th><th>SHA256</th></tr></thead><tbody>{files}</tbody></table>
<h2>Reproduce this configuration</h2><pre>{rendered(passport['reproduce_command'])}</pre>
<ul>{''.join(f'<li>{rendered(note)}</li>' for note in passport['reproduction_notes'])}</ul>
<h2>Limitations</h2><ul>{limits}</ul><small>Generated {rendered(passport['generated_at_utc'])}. Open this file locally; no internet connection is needed.</small></html>"""
