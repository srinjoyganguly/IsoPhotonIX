"""Independent readiness manifest for eigenmode and phasor requirements."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import threading
import time

from .runtime import ROOT, clean_log, interpreter, worker_env

_cache = None
_checked = 0.
_lock = threading.Lock()


def _probe(target):
    solver, device = target
    base = {'solver':solver,'device':device,'available':False}
    timeout=180 if solver=='fdtdx' else 55
    env={**worker_env(solver,device),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    try:
        response = subprocess.run([interpreter(solver, device),'-m','backend.ports_probe',solver,device],cwd=ROOT,
                                  env=env,capture_output=True,text=True,timeout=timeout)
        records=[line[11:] for line in response.stdout.splitlines() if line.startswith('PROBE_JSON=')]
        if response.returncode or not records:
            return {**base,'reason':clean_log((response.stderr or response.stdout)[-1600:]) or 'Port runtime probe failed.'}
        result=json.loads(records[-1])
        if result.get('reason'): result['reason']=clean_log(result['reason'])
        return {**base,**result}
    except subprocess.TimeoutExpired:
        return {**base,'reason':f'The {solver}/{device} port probe did not finish within {timeout} seconds.'}
    except Exception as exc:
        return {**base,'reason':clean_log(str(exc))}


def port_capabilities(refresh=False):
    global _cache,_checked
    with _lock:
        if _cache is not None and not refresh and time.monotonic()-_checked < 60:
            return _cache
        with ThreadPoolExecutor(max_workers=3) as pool:
            targets=list(pool.map(_probe,[('meep','cpu'),('fdtdx','cpu'),('fdtdx','gpu')]))
        preferred=next((t for t in targets if t['available'] and t['solver']=='meep'),next((t for t in targets if t['available']),None))
        recommendation=None if preferred is None else {'solver':preferred['solver'],'device':preferred['device'],
            'reason':'Meep CPU is a useful baseline for this small native 2D mode/flux study. FDTDX targets use the same input contract with their own mode source, voxel grid and measured incident normalization.' if preferred['solver']=='meep' else 'Use an available FDTDX target explicitly. Source-mode preparation is local on CPU; electromagnetic time evolution uses the selected device.'}
        _cache={'targets':targets,'recommendation':recommendation,
                'note':'Readiness checks imports and device execution. A completed reference/device pair with recorded settling provides separate measurement evidence.'}
        _checked=time.monotonic()
        return _cache
