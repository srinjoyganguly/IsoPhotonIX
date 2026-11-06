"""Package a measured grating and audit Python and Jupyter replay from its verified native cache."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np

from backend.calibrations import Calibration, create_bundle, result_zip
from backend.calibration_replay import verify_source
from scripts.verify_calibration_replays import replay
from scripts.verify_finite_grating import verify
from scripts.verify_vendor_grating_notebooks import check_executed_notebook


def sha(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_export(run, checkpoint, output):
    measured = verify(run)
    if not measured["passed"]:
        raise ValueError("Only a qualified grating can be exported as a benchmark.")
    m = measured["settings"]
    config = Calibration(case="grating-3d", solver="fdtdx", device="gpu", resolution=m["resolution"],
                         geometry_3d=m["geometry"], grating_material_model=m.get("grating_material_model"),
                         grating_domain=m.get("grating_domain", "full"))
    output.mkdir(parents=True, exist_ok=False)
    case=output/"case"
    case.mkdir()
    create_bundle(config.model_dump(mode="json"),case/"replay")
    verify_source(case/"replay")
    frozen={p.name:p for p in (case/"replay/source").rglob("*") if p.is_file()}
    native_hashes={}
    for stage in measured["stages"].values():
        if stage.get("acquisition",{}).get("status")!="fresh":
            raise ValueError("The exported reference requires fresh native acquisitions.")
        for name,digest in stage["identity"]["sources"].items():
            if name not in frozen or sha(frozen[name])!=digest:
                raise ValueError(f"The exported worker differs from the original acquisition: {name}")
            native_hashes[name]=digest
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
        if not reproduction.get("passed"):
            raise ValueError(f"{runner} reported a reproduction mismatch.")
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
    return {"passed":True,"geometry":m["geometry"],"domain":m.get("grating_domain","full"),
            "mesh":m["resolution"],"archive":archive.name,"size_bytes":archive.stat().st_size,
            "sha256":sha(archive),"reference_arrays_sha256":measured["arrays_sha256"],
            "reference_report_sha256":sha(run/"report.json"),"source_sha256":native_hashes,
            "original_stages":{k:v["performance"] for k,v in measured["stages"].items()},
            "replays":checks,"scope":"Original measurement was fresh. Python and unmodified Jupyter replay validate the exact source, archived arrays, field planes and native cache. Zero new electromagnetic steps were requested by the replays."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("run","checkpoint","output","receipt"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    record=verify_export(args.run,args.checkpoint,args.output)
    args.receipt.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
