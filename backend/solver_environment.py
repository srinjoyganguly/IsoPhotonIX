"""Select an explicitly configured native worker in the local IsoPhotonIX state."""
import json
import os
from pathlib import Path
import sys


def default_user_state():
    base = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local/share')
    selected = os.environ.get('ISOPHOTONIX_DATA_DIR') or os.environ.get('PHOTONICS_DATA_DIR')
    if selected:
        return Path(selected).expanduser()
    current, previous = base / 'isophotonix', base / 'photonics-workbench'
    return previous if previous.is_dir() and not current.exists() else current


def solver_interpreter(solver, device, data_dir):
    override = os.environ.get(f'{solver.upper()}_PYTHON')
    if override:
        return override
    state = Path(data_dir) / '.local'
    registry = state / 'runtime.json'
    settings = json.loads(registry.read_text()) if registry.exists() else {}
    candidates = []
    if solver in ('fdtdx', 'beamz') and device in ('cpu', 'gpu'):
        candidates.append(f'{solver}_{device}_python')
    candidates.append(f'{solver}_python')
    if solver == 'fdtdx' and device == 'cpu':
        # A managed CUDA installation also proves its CPU execution path.
        candidates.append('fdtdx_gpu_python')
    for key in candidates:
        if settings.get(key):
            return settings[key]
    if solver == 'beamz':
        return sys.executable
    legacy = state / ('meep-conda' if solver == 'meep' else 'fdtdx-env') / 'bin/python'
    return str(legacy) if legacy.exists() else sys.executable
