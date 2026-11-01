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
from .runtime import ROOT

manager = JobManager()


@asynccontextmanager
async def lifespan(app):
    yield
    manager.shutdown()


app = FastAPI(title="Photonics Workbench", version="0.1.0", lifespan=lifespan)


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
    return {"status": "ok", "version": "0.1.0", "compute": "local"}


@app.get("/api/example")
def example():
    return WaveguideConfig().model_dump()


@app.get("/api/capabilities")
def get_capabilities(refresh: bool = False):
    return capabilities(refresh)


@app.post("/api/validate")
def validate(config: WaveguideConfig):
    return {"valid": True, "estimate": config.estimate(), "warnings": config.warnings()}


@app.get("/api/jobs")
def jobs():
    return [manager.get(j) for j in reversed(list(manager.jobs))][:30]


@app.post("/api/jobs", status_code=202)
def create_job(config: WaveguideConfig):
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
    with np.load(manager.directory/job_id/"arrays.npz") as arrays:
        data = {key: arrays[key].tolist() for key in ("field", "epsilon", "x_um", "y_um")}
    return {**data, "metadata": job["result"]}


@app.get("/api/jobs/{job_id}/export")
def export(job_id: str):
    job = get_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(409, "Only completed runs can be exported as results.")
    stream = io.BytesIO()
    output = manager.directory/job_id
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("config.json", "metadata.json", "arrays.npz"):
            archive.write(output/name, name)
        archive.writestr("run-log.txt", job["logs"])
        archive.writestr("README.txt", "Actual local solver output. arrays.npz contains field and epsilon indexed [y,x], x_um and y_um coordinates, plus any recorded frames/times_fs. Field is signed Ez in arbitrary source units, not normalized transmission. Metadata records the immutable input and actual execution target. Reproduce with scripts/run_simulation.py config.json --output <new-directory>.\n")
    stream.seek(0)
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="waveguide-{job_id[:8]}.zip"'})


dist = ROOT/"frontend/dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
