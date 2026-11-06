import os
from pathlib import Path

from .solver_environment import solver_interpreter

ROOT = Path(__file__).resolve().parents[1]

def data_root():
    """Installed launchers set an explicit per-user location for mutable data."""
    selected = os.environ.get("ISOPHOTONIX_DATA_DIR") or os.environ.get("PHOTONICS_DATA_DIR")
    return Path(selected).expanduser().resolve() if selected else ROOT


def interpreter(solver, device=None):
    return solver_interpreter(solver, device, data_root())


def worker_env(solver, device):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLCONFIGDIR=str(data_root()/".local/matplotlib"))
    if solver == "fdtdx":
        env["JAX_PLATFORMS"] = "cuda" if device == "gpu" else "cpu"
    return env


def clean_log(text):
    return text.replace(str(ROOT), "<project>").replace(str(data_root()), "<state>").replace(str(Path.home()), "<home>")
