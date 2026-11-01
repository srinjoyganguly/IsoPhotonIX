import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def interpreter(solver):
    override = os.environ.get(f"{solver.upper()}_PYTHON")
    local = ROOT / ".local/runtime.json"
    settings = json.loads(local.read_text()) if local.exists() else {}
    if override:
        return override
    if settings.get(f"{solver}_python"):
        return settings[f"{solver}_python"]
    candidate = ROOT / ".local" / ("meep-conda" if solver == "meep" else "fdtdx-env") / "bin/python"
    return str(candidate) if candidate.exists() else sys.executable


def worker_env(solver, device):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLCONFIGDIR=str(ROOT/".local/matplotlib"))
    if solver == "fdtdx":
        env["JAX_PLATFORMS"] = "cuda" if device == "gpu" else "cpu"
    return env


def clean_log(text):
    return text.replace(str(ROOT), "<project>").replace(str(Path.home()), "<home>")
