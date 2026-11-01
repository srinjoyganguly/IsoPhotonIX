"""Real HTTP end-to-end check, including numerical output and export integrity."""
import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--solver", choices=["meep", "fdtdx"], required=True)
parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--evidence", type=Path)
args = parser.parse_args()
with httpx.Client(base_url=args.url, timeout=120) as client:
    config = client.get("/api/example").json()
    config.update(solver=args.solver, device=args.device)
    response = client.post("/api/jobs", json=config)
    response.raise_for_status()
    job_id = response.json()["id"]
    deadline = time.monotonic()+310
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(.5)
    assert job["status"] == "completed", job
    field = client.get(f"/api/jobs/{job_id}/field")
    field.raise_for_status()
    data = field.json()
    array = np.array(data["field"])
    assert np.isfinite(array).all() and np.max(np.abs(array)) > 0
    assert array.shape == (len(data["y_um"]), len(data["x_um"]))
    export = client.get(f"/api/jobs/{job_id}/export")
    export.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
        assert json.loads(archive.read("config.json")) == config
        with np.load(io.BytesIO(archive.read("arrays.npz"))) as arrays:
            np.testing.assert_array_equal(arrays["field"], array)
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["requested_device"] == args.device
        assert metadata["requested_solver"] == args.solver
    evidence = {"job_id": job_id, "status": job["status"], "shape": list(array.shape),
                "max_abs": float(np.max(np.abs(array))), "export_bytes": len(export.content),
                "metadata": metadata}
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, indent=2))
    print(json.dumps({k:v for k,v in evidence.items() if k != "metadata"}, indent=2))
