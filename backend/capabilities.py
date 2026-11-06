import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .runtime import ROOT, clean_log, interpreter, worker_env

_cache = None
_checked = 0.0
_lock = threading.Lock()
_TARGETS=(("meep","cpu"),("fdtdx","cpu"),("fdtdx","gpu"))
_target_locks={key:threading.Lock() for key in _TARGETS}
_target_cache={}



def _probe(target):
    solver, device = target
    base = {"solver": solver, "device": device, "available": False}
    # JAX and FDTDX can take more than a minute to initialize when the CPU is
    # simultaneously running simulations. This probe executes one scalar op.
    timeout = 180 if solver == "fdtdx" else 50
    env = {**worker_env(solver, device), "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1"}
    try:
        run = subprocess.run([interpreter(solver, device), "-m", "backend.probe", solver, device],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        lines = [s[11:] for s in run.stdout.splitlines() if s.startswith("PROBE_JSON=")]
        if run.returncode or not lines:
            return {**base, "reason": clean_log((run.stderr or run.stdout)[-1800:]) or "Solver probe failed."}
        return {**base, **json.loads(lines[-1])}
    except subprocess.TimeoutExpired:
        return {**base, "reason": f"The local {solver}/{device} probe did not finish within {timeout} seconds. Retry when the workers are less busy."}
    except FileNotFoundError:
        return {**base, "reason": f"{solver.title()} interpreter not found. Install its documented environment or set {solver.upper()}_PYTHON to its Python executable."}
    except Exception as exc:
        return {**base, "reason": clean_log(str(exc))}


def _target(key, refresh=False):
    with _target_locks[key]:
        cached=_target_cache.get(key)
        if cached and not refresh and time.monotonic()-cached[0]<60:
            return cached[1]
        measured=_probe(key)
        _target_cache[key]=(time.monotonic(),measured)
        return measured


def capabilities(refresh=False, target=None):
    global _cache, _checked
    if target is not None:
        return {"targets":[_target(target,refresh)]}
    with _lock:
        if _cache is not None and not refresh and time.monotonic()-_checked < 60:
            return _cache
        with ThreadPoolExecutor(max_workers=3) as pool:
            targets = list(pool.map(lambda key:_target(key,refresh),_TARGETS))
        ready = [t for t in targets if t["available"]]
        preferred = next((t for t in ready if t["solver"] == "meep"), next(iter(ready), None))
        recommendation = None
        if preferred:
            fdtdx_ready = any(t["solver"] == "fdtdx" for t in ready)
            alternative = (" FDTDX is an available alternative for explicit JAX execution; select a verified CPU or GPU target below. GPU compilation may outweigh compute savings for this small grid."
                           if fdtdx_ready else " FDTDX is currently unavailable; see its readiness error below.")
            recommendation = {"solver": preferred["solver"], "device": preferred["device"],
                "reason": ("Meep CPU is recommended for this small lossless 2D slab: native 2D FDTD, low startup overhead, and an installed CPU backend." + alternative
                if preferred["solver"] == "meep" else "FDTDX is the installed backend for this lossless slab. It uses a three-cell periodic extrusion to model 2D. Select CPU or an individually verified GPU explicitly; compilation can dominate a small run.")}
        _cache = {"targets": targets, "recommendation": recommendation,
                  "note": "Availability is verified by imports and, for JAX, a synchronized computation. A solver run is separate evidence."}
        _checked = time.monotonic()
        return _cache
