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
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        field, epsilon = arrays["field"], arrays["epsilon"]
        x, y = arrays["x_um"], arrays["y_um"]
        finite = bool(all(np.isfinite(a).all() for a in (field, epsilon, x, y)))
        shape_ok = bool(field.ndim == 2 and epsilon.shape == field.shape and x.ndim == y.ndim == 1
                        and field.shape == (len(y), len(x)))
        nonzero = bool(np.any(field)) if np.isfinite(field).all() else False
        frame_count = int(len(arrays["frames"])) if "frames" in arrays else 0
    files = [file_record(output / name) for name in ("config.json", "metadata.json", "arrays.npz")]
    snapshot_ok = metadata.get("config") == config and job["config"] == config
    checks = [
        {"label": "Finite numerical values", "passed": finite,
         "detail": "Final Ez, permittivity, and coordinate arrays contain no NaN or infinite values."},
        {"label": "Nonzero final field", "passed": nonzero,
         "detail": "At least one final Ez sample is nonzero; this does not establish convergence."},
        {"label": "Matching field and material grids", "passed": shape_ok,
         "detail": f"Field shape {list(field.shape)}; coordinates contain {len(x)} x and {len(y)} y samples."},
        {"label": "Consistent saved configuration", "passed": snapshot_ok,
         "detail": "Config file, saved worker metadata, and run snapshot agree."},
    ]
    limitations = [
        "Educational lossless, nondispersive slab; grid and PML convergence have not been established.",
        "Ez uses arbitrary source units. No normalized transmission, S-parameters, or mode-purity claim.",
        "Meep current-line and FDTDX directed-plane source amplitudes are not cross-solver calibrated.",
        "SHA256 identifies these exported file bytes; it is not a signature or an accuracy certificate.",
    ]
    assumptions = metadata.get("assumptions", metadata.get("notes", []))
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
            "frame_count": frame_count, "requested_duration_fs": config["duration_fs"],
            "actual_duration_fs": metadata.get("simulated_time_fs", metadata.get("actual_duration_fs")),
            "precision": metadata.get("precision", "Not recorded"), "timings": timings},
        "checks": checks, "files": files,
        "array_sha256": next(record["sha256"] for record in files if record["name"] == "arrays.npz"),
        "reproduce": {"command": command, "notes": notes}, "reproduce_command": command,
        "reproduction_notes": notes, "limitations": limitations,
    }
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
<h2>File manifest · SHA256</h2><p>These hashes identify the three original result files included with this passport.</p>
<table><thead><tr><th>File</th><th>Bytes</th><th>SHA256</th></tr></thead><tbody>{files}</tbody></table>
<h2>Reproduce this configuration</h2><pre>{rendered(passport['reproduce_command'])}</pre>
<ul>{''.join(f'<li>{rendered(note)}</li>' for note in passport['reproduction_notes'])}</ul>
<h2>Limitations</h2><ul>{limits}</ul><small>Generated {rendered(passport['generated_at_utc'])}. Open this file locally; no internet connection is needed.</small></html>"""
