import io
import json
import zipfile
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from .capabilities import capabilities
from .jobs import JobManager
from .model import WaveguideConfig
from .passport import FramesUnavailable, build_passport, passport_html, stored_frames
from .runtime import ROOT
from .study3d import Waveguide3DConfig, example3d, parse_study
from .layout_study import LayoutStudy, example_layout
from .taper_ports import TaperPortStudy, example_taper_ports
from .taper_study import TaperStudy, example_taper
from .volume import stored_volume, volume_info, volume_slice
from .compare3d import compare_runs
from .benchmarks import benchmark_report, component_benchmark_report
from .ports import PortStudy, example_ports
from .ports_capabilities import port_capabilities
from .spectra import stored_spectrum, spectrum_response, spectrum_csv
from .ports3d import ModalPortStudy, example_ports3d
from .sparameters import stored_sparameters, sparameters_response, sparameters_csv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

manager = JobManager(defer_recovery=True)


@asynccontextmanager
async def lifespan(app):
    if isinstance(manager, JobManager):
        manager.start()
    try:
        yield
    finally:
        manager.shutdown()


app = FastAPI(title="Photonics Workbench", version="0.9.0-preview.1", lifespan=lifespan)


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
    return {"status": "ok", "version": "0.9.0-preview.1", "compute": "local"}


class LibraryPlacement(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    x: float = Field(0, ge=-500, le=500)
    y: float = Field(0, ge=-500, le=500)
    z: float = Field(0, ge=-500, le=500)
    rotation_deg: float = Field(0, ge=-360, le=360)


class LibraryBuild(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    component_id: str = Field(min_length=1, max_length=80)
    parameters: dict[str, float] = Field(default_factory=dict, max_length=64)
    placement: LibraryPlacement = Field(default_factory=LibraryPlacement)
    material_id: str = Field('generic-core', min_length=1, max_length=80)
    material: dict | None = None


class LibraryPlace(LibraryBuild):
    geometry: dict
    replace_component_id: str | None = Field(None, min_length=1, max_length=80)
    fit_domain: bool = False


@app.get('/api/library')
def library_catalog():
    from .component_library import get_catalog
    return get_catalog()


@app.post('/api/library/build')
def library_build(request: LibraryBuild):
    from .library_layout import build_design
    try:
        return build_design(**request.model_dump())
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/library/place')
def library_place(request: LibraryPlace):
    from .library_layout import place_component
    if request.material is not None:
        raise HTTPException(422, 'Save a custom material in the design before placing a component with it.')
    try:
        return place_component(**request.model_dump(exclude={'material'}, exclude_unset=True))
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/geometry/validate')
def validate_draft(geometry: dict):
    from .geometry_draft import validate_geometry
    try:
        return {'valid': True, 'geometry': validate_geometry(geometry),
                'simulation_support': 'Design validation does not qualify an electromagnetic study; use an explicitly supported study.'}
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/api/ports3d/example')
def modal_example(kind: str = 'gap'):
    if kind not in ('straight','gap'):
        raise HTTPException(422,'Choose straight or gap.')
    return example_ports3d(kind).model_dump()


@app.get('/api/ports3d/capabilities')
def modal_targets(refresh: bool = False):
    capability = port_capabilities(refresh)
    recommendation = capability.get('recommendation')
    if recommendation:
        recommendation = {**recommendation,'reason':'Meep provides an independent CPU modal-decomposition baseline. FDTDX can evolve the finite-height fields on CPU or NVIDIA GPU; its local mode preparation uses CPU. Review four-stage cost and measured numerical sensitivities.'}
    return {**capability,'recommendation':recommendation,'note':'Local mode preparation and device execution are probed. Each 3D S-matrix requires four measured directional solves; readiness alone is not numerical qualification.'}


@app.post('/api/ports3d/validate')
def validate_modal(config: ModalPortStudy):
    return {'valid':True,'config':config.model_dump(),'estimate':config.estimate(),'layout':config.layout(),'warnings':config.warnings()}


@app.post('/api/ports3d/jobs',status_code=202)
def submit_modal(config: ModalPortStudy):
    return submit_modal_checked(config)


@app.get('/api/taper-ports/example')
def taper_port_example():
    return example_taper_ports().model_dump()


@app.post('/api/taper-ports/validate')
def validate_taper_ports(config: TaperPortStudy):
    return {'valid':True,'config':config.model_dump(),'estimate':config.estimate(),'layout':config.layout(),'warnings':config.warnings()}


@app.post('/api/taper-ports/jobs',status_code=202)
def submit_taper_ports(config: TaperPortStudy):
    return submit_modal_checked(config)


def submit_modal_checked(config):
    target = next((t for t in port_capabilities()['targets'] if t['solver']==config.solver and t['device']==config.device),None)
    if not target or not target['available']:
        raise HTTPException(409,(target or {}).get('reason','Requested modal-study target is unavailable.'))
    try:
        return manager.submit(config.model_dump())
    except RuntimeError as exc:
        raise HTTPException(409,str(exc)) from exc


def code_archive(data,filename):
    return Response(data,media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="{filename}"'})


@app.post('/api/reproduce/study')
def reproduce_study(config: dict):
    from .reproduce import study_bundle
    try:
        study = parse_study(config)
        return code_archive(study_bundle(study.model_dump()),'photonics-study-python-notebook.zip')
    except ValidationError as exc:
        raise HTTPException(422,json.loads(exc.json())) from exc
    except (ValueError,TypeError,KeyError) as exc:
        raise HTTPException(422,str(exc)) from exc


@app.post('/api/reproduce/geometry')
def reproduce_geometry(geometry: dict):
    from .reproduce import geometry_bundle
    try:
        return code_archive(geometry_bundle(geometry),'photonics-design-python-notebook.zip')
    except (ValueError,TypeError,KeyError) as exc:
        raise HTTPException(422,str(exc)) from exc


@app.get('/api/ports/example')
def port_example(kind: str = 'gap'):
    if kind not in ('straight','gap'):
        raise HTTPException(422,'Choose straight or gap.')
    return example_ports(kind).model_dump()


@app.get('/api/ports/capabilities')
def ports_targets(refresh: bool = False):
    return port_capabilities(refresh)


@app.post('/api/ports/validate')
def validate_ports(config: PortStudy):
    return {'valid':True,'estimate':config.estimate(),'layout':config.layout(),'warnings':config.warnings()}


@app.post('/api/ports/jobs',status_code=202)
def submit_port(config: PortStudy):
    target=next((target for target in port_capabilities()['targets'] if target['solver']==config.solver and target['device']==config.device),None)
    if not target or not target['available']:
        raise HTTPException(409,(target or {}).get('reason','Requested port-study target is unavailable.'))
    try:
        return manager.submit(config.model_dump())
    except RuntimeError as exc:
        raise HTTPException(409,str(exc)) from exc


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


@app.get('/api/taper/example')
def get_taper_example():
    return example_taper().model_dump()


@app.post('/api/taper/validate')
def validate_taper(config: TaperStudy):
    layout = config.layout()
    return {'valid': True, 'estimate': config.estimate(layout), 'warnings': config.warnings(),
            'physical_layout': layout, 'boundaries': 'PML on all six faces',
            'source': {'component': 'Ey', 'x_um': layout['source_x'],
                       'width_um': layout['width_in'], 'height_um': layout['height']}}


@app.post('/api/taper/jobs', status_code=202)
def submit_taper(config: TaperStudy):
    return submit_checked(config)


@app.get('/api/layout/example')
def get_layout_example(family: str | None = None):
    if family is None:
        return example_layout().model_dump()
    from .layout_examples import qualified_layout_example
    try:
        return qualified_layout_example(family).model_dump()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(409, 'The measured field configuration is unavailable or differs from its recorded grid.') from exc


@app.post('/api/layout/validate')
def validate_layout(config: LayoutStudy):
    spec = config.layout()
    return {'valid':True,'estimate':spec,'scene_layout':spec,
            'warnings':['Finite as-drawn structures in homogeneous cladding with six PML faces.',
                'A local Ey point dipole excites all available channels. Interpret the final field in arbitrary source units.',
                'Shapes use a staircase grid. Refine the grid and absorbing boundaries before interpreting quantitative changes.',
                spec['precedence']]}


@app.post('/api/layout/jobs', status_code=202)
def submit_layout(config: LayoutStudy):
    return submit_checked(config)


@app.get('/api/component-benchmarks')
def public_component_benchmarks():
    try:
        return component_benchmark_report()
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise HTTPException(409, 'The recorded component comparison is missing or inconsistent.') from exc


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
    if job["config"].get("study_type") in ("waveguide-3d","waveguide-ports-3d","taper-3d","layout-3d","taper-ports-3d"):
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
    if job["config"].get("study_type") not in ("waveguide-3d","waveguide-ports-3d","taper-3d","layout-3d","taper-ports-3d"):
        raise HTTPException(409, "This run is a 2D slab, not a recorded 3D volume.")
    try:
        return job, stored_volume(output,allow_zero=job['config'].get('study_type') in ('waveguide-ports-3d','taper-ports-3d'))
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


def port_output(job_id):
    job,output=completed_output(job_id)
    if job['config'].get('study_type') != 'waveguide-ports-2d':
        raise HTTPException(409,'This run did not measure calibrated port spectra.')
    try:
        metadata=json.loads((output/'metadata.json').read_text())
        config=json.loads((output/'config.json').read_text())
        if metadata.get('config') != config or config != job['config']:
            raise ValueError('Saved configuration mismatch.')
        return stored_spectrum(output,metadata),metadata
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        raise HTTPException(409,'Saved power spectra are missing, inconsistent or invalid.') from exc


@app.get('/api/jobs/{job_id}/spectrum')
def spectrum(job_id: str):
    data,metadata=port_output(job_id)
    return spectrum_response(data,metadata)


@app.get('/api/jobs/{job_id}/spectrum.csv')
def spectrum_download(job_id: str):
    data,_=port_output(job_id)
    return Response(spectrum_csv(data),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="ports-{job_id[:8]}-spectrum.csv"'})


def modal_output(job_id):
    job,output = completed_output(job_id)
    if job['config'].get('study_type') not in ('waveguide-ports-3d','taper-ports-3d'):
        raise HTTPException(409,'This run did not measure a finite-height modal S-matrix.')
    try:
        metadata = json.loads((output/'metadata.json').read_text())
        config = json.loads((output/'config.json').read_text())
        if metadata.get('config') != config or config != job['config']:
            raise ValueError('Saved configuration mismatch.')
        return stored_sparameters(output,metadata),metadata
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        raise HTTPException(409,'Saved complex S-parameters are missing, inconsistent or invalid.') from exc


@app.get('/api/jobs/{job_id}/sparameters')
def modal_spectrum(job_id: str):
    values,metadata = modal_output(job_id)
    return sparameters_response(values,metadata)


@app.get('/api/jobs/{job_id}/sparameters.csv')
def modal_csv(job_id: str):
    values,_ = modal_output(job_id)
    return Response(sparameters_csv(values),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="sparameters-{job_id[:8]}.csv"'})


@app.get('/api/jobs/{job_id}/reproduce')
def reproduce_run(job_id: str):
    from .reproduce import study_bundle
    job,output = completed_output(job_id)
    try:
        build_passport(job,output)
        config = json.loads((output/'config.json').read_text())
        if config != job['config'] or json.loads((output/'metadata.json').read_text()).get('config') != config:
            raise ValueError('Saved configuration differs from the immutable run.')
        data = study_bundle(config,output)
        return code_archive(data,f'photonics-{job_id[:8]}-python-notebook.zip')
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        raise HTTPException(409,'Saved result files are inconsistent; reproducible code export is unavailable.') from exc


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
        for name in manifest['files']:
            archive.write(output/name, name)
        archive.writestr("run-log.txt", job["logs"])
        archive.writestr("passport.html", passport_html(record))
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        array_note = ("3D arrays: fields[component,z,y,x] in Ex/Ey/Ez order, field=Ey[z,y,x], epsilon[z,y,x], and x_um/y_um/z_um display coordinates. Final-time snapshots only. Read metadata field_sampling before interpreting interpolated Yee-grid samples. FDTDX also retains raw_fields and their coordinate convention. "
                      if job["config"].get("study_type") in ("waveguide-3d","waveguide-ports-3d","taper-3d","layout-3d","taper-ports-3d") else
                      "2D arrays: field and epsilon[y,x], x_um/y_um, plus any frames/times_fs. Field is signed Ez. The final snapshot may be later than the last recorded frame. ")
        measurement_note = 'Fields use arbitrary source amplitude, not normalized transmission. '
        if job['config'].get('study_type') == 'waveguide-ports-2d':
            array_note = 'arrays.npz contains actual final-time Ez and material samples after the pulse. spectra.npz preserves signed measured powers and unclipped ratios. spectrum.csv retains raw values and valid_bin flags; ignore unqualified bins as measurements. Supplemental monitor files retain solver-specific provenance. '
            measurement_note = 'R=-reflected_flux/incident_flux and T=transmitted_flux/incident_flux use the measured straight reference. 1-R-T is uncollected flux, not absorption. These are finite-aperture powers, not complex modal S-parameters. '
        if job['config'].get('study_type') in ('waveguide-ports-3d','taper-ports-3d'):
            measurement_note = 'sparameters.npz preserves complex unit-power modal amplitudes and the 2x2 S-matrix in [frequency,outgoing,incoming] order. Read physical reference planes, phase gauge and four-stage settlement in metadata.json. CSV retains raw complex values and validity flags. Residual fundamental-channel power includes radiation/other modes/error, not absorption. arrays.npz is the actual forward-device late-time volume, not a monochromatic field synthesized from S. '
        archive.writestr("README.txt", "Actual local solver output. Open passport.html for configuration, recorded execution, file checks, limitations and reproduction. manifest.json identifies all original result files with SHA256. " + array_note + measurement_note + f"Reproduce with {record['reproduce_command']} (choose a new output directory).\n")
    stream.seek(0)
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="waveguide-{job_id[:8]}.zip"'})


dist = ROOT/"frontend/dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
