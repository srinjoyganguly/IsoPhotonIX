import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def data_root():
    """Installed launchers set an explicit per-user location for mutable data."""
    selected = os.environ.get("PHOTONICS_DATA_DIR")
    return Path(selected).expanduser().resolve() if selected else ROOT


def interpreter(solver):
    override = os.environ.get(f"{solver.upper()}_PYTHON")
    if override:
        return override
    state = data_root() / ".local"
    local = state / "runtime.json"
    settings = json.loads(local.read_text()) if local.exists() else {}
    if settings.get(f"{solver}_python"):
        return settings[f"{solver}_python"]
    candidate = state / ("meep-conda" if solver == "meep" else "fdtdx-env") / "bin/python"
    return str(candidate) if candidate.exists() else sys.executable


def worker_env(solver, device):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLCONFIGDIR=str(data_root()/".local/matplotlib"))
    if solver == "fdtdx":
        env["JAX_PLATFORMS"] = "cuda" if device == "gpu" else "cpu"
    return env


def clean_log(text):
    return text.replace(str(ROOT), "<project>").replace(str(data_root()), "<state>").replace(str(Path.home()), "<home>")
