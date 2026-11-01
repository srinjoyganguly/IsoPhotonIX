import io
import json
import zipfile
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .capabilities import capabilities
from .jobs import JobManager
from .model import WaveguideConfig
from .passport import FramesUnavailable, build_passport, passport_html, stored_frames
from .runtime import ROOT
from .study3d import Waveguide3DConfig, example3d
from .volume import stored_volume, volume_info, volume_slice
from .compare3d import compare_runs
from .benchmarks import benchmark_report
from pydantic import BaseModel, ConfigDict, Field

manager = JobManager()


@asynccontextmanager
async def lifespan(app):
    yield
    manager.shutdown()


app = FastAPI(title="Photonics Workbench", version="0.4.0-preview.1", lifespan=lifespan)


@app.middleware("http")
async def local_browser_only(request: Request, call_next):
    # Local service: no remote origin may launch jobs on the user's machine.
    host = request.url.hostname
    if host not in ("localhost", "127.0.0.1", "::1", "testserver"):
        return JSONResponse({"detail": "Use this local app through localhost."}, status_code=403)
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).hostname not in ("localhost", "127.0.0.1", "::1"):
        return JSONResponse({"detail": "Remote browser origins are not allowed."}, status_code=403)
    return await call_next(request)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.4.0-preview.1", "compute": "local"}


@app.get("/api/benchmarks")
def public_benchmarks():
    try:
        return benchmark_report()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(409, "Recorded benchmark evidence is incomplete or invalid. Rerun the local benchmark command.") from exc


@app.get("/api/example")
def example():
    return WaveguideConfig().model_dump()


@app.get("/api/capabilities")
def get_capabilities(refresh: bool = False):
    return capabilities(refresh)


@app.post("/api/validate")
def validate(config: WaveguideConfig):
    return {"valid": True, "estimate": config.estimate(), "warnings": config.warnings()}


@app.get("/api/example3d")
def get_example3d():
    return example3d().model_dump()


@app.post("/api/validate3d")
def validate3d(config: Waveguide3DConfig):
    return {"valid": True, "estimate": config.estimate(), "warnings": config.warnings(),
            "source": {"component": "Ey", "x_um": -config.geometry.domain.x / 2 + config.pml_um + .5,
                       "width_um": config.core.size.y, "height_um": config.core.size.z},
            "boundaries": "PML on all six faces; finite height along Z"}


@app.post("/api/jobs3d", status_code=202)
def create_job3d(config: Waveguide3DConfig):
    return submit_checked(config)


class Compare3DRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_run_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    candidate_run_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")


@app.post("/api/compare3d")
def compare3d(config: Compare3DRequest):
    reference = get_job(config.reference_run_id)
    candidate = get_job(config.candidate_run_id)
    try:
        return compare_runs(reference, candidate, manager.directory)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (OSError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise HTTPException(409, "Saved result files are incomplete; comparison is unavailable.") from exc


@app.get("/api/jobs")
def jobs():
    return [manager.get(j) for j in reversed(list(manager.jobs))][:30]


@app.post("/api/jobs", status_code=202)
def create_job(config: WaveguideConfig):
    return submit_checked(config)


def submit_checked(config):
    target = next((t for t in capabilities()["targets"] if t["solver"] == config.solver and t["device"] == config.device), None)
    if not target or not target["available"]:
        raise HTTPException(409, (target or {}).get("reason", "Requested execution target is unavailable."))
    try:
        return manager.submit(config.model_dump())
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


def get_job(job_id):
    try:
        return manager.get(job_id)
    except KeyError:
        raise HTTPException(404, "Run not found")


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return get_job(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    get_job(job_id)
    return manager.cancel(job_id)


@app.get("/api/jobs/{job_id}/field")
def field(job_id: str):
    job = get_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(409, "Results are available only after successful completion.")
    if job["config"].get("study_type") == "waveguide-3d":
        raise HTTPException(409, "This is a 3D run. Use the volume and slice endpoints with explicit component and plane.")
    with np.load(manager.directory/job_id/"arrays.npz") as arrays:
        data = {key: arrays[key].tolist() for key in ("field", "epsilon", "x_um", "y_um")}
    return {**data, "metadata": job["result"]}


def completed_output(job_id):
    job = get_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(409, "Results are available only after successful completion.")
    return job, manager.directory / job_id


def load_volume(job_id):
    job, output = completed_output(job_id)
    if job["config"].get("study_type") != "waveguide-3d":
        raise HTTPException(409, "This run is a 2D slab, not a recorded 3D volume.")
    try:
        return job, stored_volume(output)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise HTTPException(409, "Saved 3D fields are missing or invalid.") from exc


@app.get("/api/jobs/{job_id}/volume")
def volume(job_id: str):
    job, data = load_volume(job_id)
    return {**volume_info(data), "metadata": job["result"]}


@app.get("/api/jobs/{job_id}/slice")
def slice3d(job_id: str, plane: str = "xy", index: int | None = None, component: str = "Ey"):
    job, data = load_volume(job_id)
    try:
        return {**volume_slice(data, plane, index, component), "metadata": job["result"]}
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/jobs/{job_id}/passport")
def passport(job_id: str):
    job, output = completed_output(job_id)
    try:
        result, _ = build_passport(job, output)
        return result
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise HTTPException(409, "Saved result files are missing or invalid; a passport could not be produced.") from exc


def frame_data(job_id):
    _, output = completed_output(job_id)
    try:
        return stored_frames(output)
    except FramesUnavailable as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise HTTPException(409, "Saved frame data is missing or invalid.") from exc


@app.get("/api/jobs/{job_id}/frames")
def frames(job_id: str):
    values, times, peak = frame_data(job_id)
    return {"count": len(values), "times_fs": times.tolist(), "global_peak_abs": peak}


@app.get("/api/jobs/{job_id}/frames/{index}")
def frame(job_id: str, index: int):
    values, times, _ = frame_data(job_id)
    if index < 0 or index >= len(values):
        raise HTTPException(404, "Frame index is outside this run's recorded range.")
    return {"field": values[index].tolist(), "time_fs": float(times[index]), "index": index}


@app.get("/api/jobs/{job_id}/export")
def export(job_id: str):
    job = get_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(409, "Only completed runs can be exported as results.")
    stream = io.BytesIO()
    output = manager.directory/job_id
    try:
        record, manifest = build_passport(job, output)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise HTTPException(409, "Saved result files are missing or invalid; export could not be produced.") from exc
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("config.json", "metadata.json", "arrays.npz"):
            archive.write(output/name, name)
        archive.writestr("run-log.txt", job["logs"])
        archive.writestr("passport.html", passport_html(record))
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        array_note = ("3D arrays: fields[component,z,y,x] in Ex/Ey/Ez order, field=Ey[z,y,x], epsilon[z,y,x], and x_um/y_um/z_um display coordinates. Final-time snapshots only. Read metadata field_sampling before interpreting interpolated Yee-grid samples. FDTDX also retains raw_fields and their coordinate convention. "
                      if job["config"].get("study_type") == "waveguide-3d" else
                      "2D arrays: field and epsilon[y,x], x_um/y_um, plus any frames/times_fs. Field is signed Ez. The final snapshot may be later than the last recorded frame. ")
        archive.writestr("README.txt", "Actual local solver output. Open passport.html for configuration, recorded execution, file checks, limitations and reproduction. manifest.json contains SHA256 hashes for config.json, metadata.json, arrays.npz. " + array_note + "Fields use arbitrary source amplitude, not normalized transmission. Reproduce from the repository with .venv/bin/python scripts/run_simulation.py config.json --output runs/reproduced-run (new output directory).\n")
    stream.seek(0)
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="waveguide-{job_id[:8]}.zip"'})


dist = ROOT/"frontend/dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
