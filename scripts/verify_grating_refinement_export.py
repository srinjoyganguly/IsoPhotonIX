"""Package a measured grating and audit Python and Jupyter replay from its verified native cache."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from backend.calibrations import Calibration, create_bundle, grating_meshes, result_zip
from backend.calibration_replay import verify_source
from scripts.verify_calibration_replays import replay
from scripts.verify_finite_grating import verify
from scripts.verify_vendor_grating_notebooks import check_executed_notebook, publish_notebook


def sha(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_export(run, checkpoint, output, worker_source=None):
    measured = verify(run)
    if not measured["passed"]:
        raise ValueError("Only a qualified grating can be exported as a benchmark.")
    m = measured["settings"]
    if worker_source is not None:
        # A concurrently completed benchmark is published before its GUI/API route
        # becomes selectable. Export the verified catalogue route, never an override.
        deadline=time.monotonic()+120
        while m["resolution"] not in grating_meshes(m["geometry"],m.get("grating_domain","full"),
                                                    m.get("grating_material_model","volume-average")):
            if time.monotonic()>=deadline:
                raise ValueError("The measured grating has not been published to the verified route catalogue.")
            time.sleep(1)
    config = Calibration(case="grating-3d", solver="fdtdx", device="gpu", resolution=m["resolution"],
                         geometry_3d=m["geometry"], grating_material_model=m.get("grating_material_model"),
                         grating_domain=m.get("grating_domain", "full"))
    output.mkdir(parents=True, exist_ok=False)
    case=output/"case"
    case.mkdir()
    create_bundle(config.model_dump(mode="json"),case/"replay")
    manifest=verify_source(case/"replay")
    frozen={p.name:p for p in (case/"replay/source").rglob("*") if p.is_file()}
    native_hashes={}
    for label,stage in measured["stages"].items():
        if stage.get("acquisition",{}).get("status") not in ("fresh","cache-hit"):
            raise ValueError("The native stage origin must be explicit before exporting a benchmark.")
        if stage.get("performance",{}).get("steps",0)<=0 or stage["performance"].get("platform")!="gpu":
            raise ValueError("Expected a completed measured CUDA stage.")
        cached=checkpoint/(label+"-"+stage["signature"][:20])
        saved=json.loads((cached/"receipt.json").read_text())
        if (saved["signature"]!=stage["signature"] or saved["arrays_sha256"]!=stage["arrays_sha256"]
                or saved["identity"]!=stage["identity"] or sha(cached/"arrays.npz")!=stage["arrays_sha256"]):
            raise ValueError("The exact original stage cache is required for the zero-step replay.")
        for name,digest in stage["identity"]["sources"].items():
            if name not in frozen:
                raise ValueError(f"Required native source is missing from the export: {name}")
            target=frozen[name]
            if sha(target)!=digest and worker_source is not None:
                original=worker_source/target.relative_to(case/"replay/source")
                if not original.is_file() or sha(original)!=digest:
                    raise ValueError(f"Specified acquisition snapshot differs for {name}.")
                shutil.copyfile(original,target)
                manifest["files"][str(target.relative_to(case/"replay"))]=digest
            if sha(target)!=digest:
                raise ValueError(f"The exported worker differs from the original acquisition: {name}")
            native_hashes[name]=digest
    (case/"replay/manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    verify_source(case/"replay")
    results=case/"calculation"
    results.mkdir()
    for file in ("report.json","arrays.npz"):
        shutil.copyfile(run/file,results/file)
    stem=f"grating-{m['geometry']}-{m.get('grating_domain','full')}-r{m['resolution']}"
    archive=output/f"{stem}.zip"
    result_zip(case,archive)
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise ValueError("Result export archive is damaged.")
        notice="docs/licenses/FDTDX-MIT.txt"
        license_sha256=hashlib.sha256(handle.read(notice)).hexdigest()
        if license_sha256!=sha(Path(__file__).resolve().parents[1]/notice):
            raise ValueError("Exported FDTDX license differs from the distribution notice.")
        hashes=json.loads(handle.read("reference-sha256.json"))
        for name,digest in hashes.items():
            if hashlib.sha256(handle.read(f"reference/{name}")).hexdigest()!=digest:
                raise ValueError(f"Reference export changed: {name}")
        for entry in handle.namelist():
            if entry.startswith("/") or ".." in Path(entry).parts:
                raise ValueError("Export has an unsafe path.")
            if not entry.endswith(".npz"):
                content=handle.read(entry)
                if any(s in content for s in (b"/home/srinjoy/",b"/mnt/c/Users/",b"@gmail.com",b"PRIVATE KEY-----")):
                    raise ValueError("Export includes a machine-specific path or personal content.")
    os.environ["PHOTONICS_CHECKPOINT"]=str(checkpoint.resolve())
    checks={}
    for runner, folder in (("python","script-result"),("notebook","notebook-result")):
        reproduction=replay(runner,archive,output)
        if not reproduction.get("passed") or not reproduction.get("all_array_bytes_equal"):
            raise ValueError(f"{runner} did not reproduce every numerical array byte-for-byte.")
        result=output/runner/folder
        report=verify(result)
        stages={s:v["acquisition"]["status"] for s,v in report["stages"].items()}
        if set(stages.values())!={"cache-hit"}:
            raise ValueError("An exported replay performed new native work.")
        with ExitStack() as stack:
            original=stack.enter_context(np.load(run/"arrays.npz",allow_pickle=False))
            current=stack.enter_context(np.load(result/"arrays.npz",allow_pickle=False))
            if set(original.files)!=set(current.files):
                raise ValueError("Replay array keys differ.")
            for key in original.files:
                np.testing.assert_array_equal(original[key],current[key])
            count=len(original.files)
        checks[runner]={"stage_acquisition":stages,"matched_field_and_result_arrays":count,
                        "archive_sha256":sha(result/"arrays.npz"),"reproduction":reproduction}
    check_executed_notebook(output/"notebook")
    executed=publish_notebook(output/"notebook", output/f"{stem}-executed.ipynb")
    return {"passed":True,"geometry":m["geometry"],"domain":m.get("grating_domain","full"),
            "mesh":m["resolution"],"archive":archive.name,"size_bytes":archive.stat().st_size,
            "sha256":sha(archive),"fdtdx_license_sha256":license_sha256,"reference_arrays_sha256":measured["arrays_sha256"],
            "reference_report_sha256":sha(run/"report.json"),"source_sha256":native_hashes,"executed_notebook":executed,
            "original_stages":{k:v["performance"] for k,v in measured["stages"].items()},
            "original_stage_acquisition":{k:v["acquisition"] for k,v in measured["stages"].items()},
            "replays":checks,"scope":"All native arrays and timings originate from verified completed GPU checkpoints. Original-stage acquisition flags identify reuse, including resumed runs. Python and unmodified Jupyter replay validate the exact source, archived arrays, field planes and native cache. Zero new electromagnetic steps were requested by the replays."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("run","checkpoint","output","receipt"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--worker-source",type=Path,help="Frozen checkout of the original numerical worker when packaging a measured benchmark.")
    args=parser.parse_args()
    record=verify_export(args.run,args.checkpoint,args.output,args.worker_source)
    args.receipt.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
