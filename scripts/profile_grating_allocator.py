"""Measure full-domain grating CUDA memory for JAX's on-demand platform allocator."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from scripts.grating_memory_budget import estimate


def smi_memory():
    binary = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    values = subprocess.check_output(
        [binary, "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    if len(values) != 1:
        raise RuntimeError("A single unambiguous CUDA device is required.")
    return tuple(int(value) for value in values[0].split(","))


def measure(geometry, model, role, mesh, observe, domain):
    lock_path = os.environ.get("PHOTONICS_GPU_LOCK", "/tmp/photonics-v012-gpu.lock")
    with tempfile.TemporaryDirectory() as directory, open(lock_path, "a") as lock:
        path = Path(directory) / "plan.json"
        flags = ["--reference"] if role == "reference" else []
        if observe:
            flags.append("--observe-one-step")
        command = [sys.executable, "-m", "scripts.plan_grating_memory", "--geometry", geometry,
                   "--resolution", str(mesh), "--grating-material-model", model, "--grating-domain", domain, "--output", str(path), *flags]
        start = time.monotonic()
        fcntl.flock(lock, fcntl.LOCK_EX)
        waited = time.monotonic() - start
        idle, free, total = smi_memory()
        samples = [idle]
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_ALLOCATOR="platform")
        with subprocess.Popen(command, env=env, pass_fds=(lock.fileno(),)) as worker:
            while worker.poll() is None:
                used, _, capacity = smi_memory()
                if capacity != total:
                    raise RuntimeError("CUDA device capacity changed during memory measurement.")
                samples.append(used)
                time.sleep(0.15)
            if worker.returncode:
                raise subprocess.CalledProcessError(worker.returncode, command)
        detail = json.loads(path.read_text())
    return detail, dict(idle_used_mib=idle, free_before_mib=free, total_mib=total,
                        sampled_peak_used_mib=max(samples), samples=len(samples), lock_wait_seconds=waited)


def profile(geometry, mesh, base_mesh, domain):
    model = "direct-yee" if geometry == "lumerical" else "volume-average"
    roles = {}
    for role in ("reference", "device"):
        base, _ = measure(geometry, model, role, base_mesh, False, domain)
        actual, observation = measure(geometry, model, role, base_mesh, True, domain)
        candidate, _ = measure(geometry, model, role, mesh, False, domain)
        budget = estimate(base, actual, candidate, observation["idle_used_mib"],
                          observation["free_before_mib"], observation["sampled_peak_used_mib"])
        roles[role] = dict(base=base, actual=actual, observation=observation, candidate=candidate, estimate=budget)
    return dict(schema_version=1, geometry=geometry, resolution=mesh, material_model=model, domain=domain, baseline_resolution=base_mesh,
                grating_spectrum="standard", allocator="platform", device=roles["device"]["base"]["gpu"],
                policy="Higher of compiled-buffer and cell-count scaling from same-role measured baseline peak, plus 20% and 1024 MiB. Both reference and device must fit.",
                required_free_with_reserve_mib=max(role["estimate"]["required_free_with_reserve_mib"]
                                                   for role in roles.values()), roles=roles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("lumerical", "tidy3d"), required=True)
    parser.add_argument("--resolution", type=int, choices=(15,16,18,20), required=True)
    parser.add_argument("--baseline-resolution", type=int, choices=(12, 14), default=14)
    parser.add_argument("--domain", choices=("full","mirror-y"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = profile(args.geometry, args.resolution, args.baseline_resolution, args.domain)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
