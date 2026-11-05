import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .runtime import ROOT, clean_log, interpreter, worker_env

_cache = None
_checked = 0.0
_lock = threading.Lock()


def _probe(target):
    solver, device = target
    base = {"solver": solver, "device": device, "available": False}
    try:
        run = subprocess.run([interpreter(solver, device), "-m", "backend.probe", solver, device],
                             cwd=ROOT, env=worker_env(solver, device), capture_output=True, text=True, timeout=50)
        lines = [s[11:] for s in run.stdout.splitlines() if s.startswith("PROBE_JSON=")]
        if run.returncode or not lines:
            return {**base, "reason": clean_log((run.stderr or run.stdout)[-1800:]) or "Solver probe failed."}
        return {**base, **json.loads(lines[-1])}
    except FileNotFoundError:
        return {**base, "reason": f"{solver.title()} interpreter not found. Install its documented environment or set {solver.upper()}_PYTHON to its Python executable."}
    except Exception as exc:
        return {**base, "reason": clean_log(str(exc))}


def capabilities(refresh=False):
    global _cache, _checked
    with _lock:
        if _cache is not None and not refresh and time.monotonic()-_checked < 60:
            return _cache
        with ThreadPoolExecutor(max_workers=3) as pool:
            targets = list(pool.map(_probe, [("meep", "cpu"), ("fdtdx", "cpu"), ("fdtdx", "gpu")]))
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
