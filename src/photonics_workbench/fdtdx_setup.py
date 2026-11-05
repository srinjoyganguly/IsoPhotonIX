"""Provision the workbench's explicit CPU and CUDA12 FDTDX profiles."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

from .managed_setup import _save_settings

_TARGETS = {'fdtdx-cpu': ('cpu',), 'fdtdx-cuda12': ('gpu', 'cpu')}
_CHECK = """
import json
from pathlib import Path
import sys

root, target = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root))
from backend.model import WaveguideConfig
from backend.adapters.fdtdx_adapter import run
import numpy as np
settings = json.loads((root/'examples/slab-waveguide.json').read_text())
config = WaveguideConfig.model_validate(dict(settings, solver='fdtdx', device=target))
result = run(config.model_dump(), Path('solve'))
with np.load('solve/arrays.npz', allow_pickle=False) as arrays:
    field, epsilon = arrays['field'], arrays['epsilon']
    sample = arrays['x_um'] > result['source']['x_um'] + 1
    peak = float(np.max(np.abs(field[:, sample])))
    if not np.isfinite(peak) or peak <= 1e-12 or not np.all(np.isfinite(epsilon)):
        raise ValueError('Missing finite propagated Ez/materials in the managed readiness run.')
    if not np.allclose([epsilon.min(), epsilon.max()], [config.cladding_index**2, config.core_index**2]):
        raise ValueError('Native material map disagrees with slab indices.')
    expected = 'cuda' if target == 'gpu' else 'cpu'
    if result['actual_device'].split(':')[0] != expected:
        raise ValueError('Explicit device mismatch in the native solve.')
    print('WORKBENCH_CHECK='+json.dumps({key:result[key] for key in
        ('actual_device','device_name','versions','time_steps','field_shape','max_abs_field')} |
        {'propagated_Ez_peak':peak,'epsilon_range':[float(epsilon.min()),float(epsilon.max())]}))
"""


def _locked_versions(lock):
    expected = {}
    for line in lock.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9._-]+)(?:\[cuda12\])?==([^\s]+) --hash=sha256:[a-f0-9]{64}', line)
        if not match:
            raise ValueError('Invalid entry in the managed wheel lock.')
        key = re.sub(r'[-_.]+', '-', match.group(1)).lower()
        if key in expected:
            raise ValueError(f'Duplicate managed package {key}.')
        expected[key] = match.group(2)
    if not {'fdtdx', 'jax', 'jaxlib', 'numpy'} <= expected.keys():
        raise ValueError('Incomplete FDTDX wheel lock.')
    return expected


def _verify(executable, profile, lock, resources):
    expected = _locked_versions(lock)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    pip = [str(executable), '-I', '-m', 'pip', '--disable-pip-version-check']
    subprocess.run([*pip, 'check'], check=True, env=env)
    result = subprocess.run([*pip, 'list', '--format=json'], check=True, capture_output=True, text=True, env=env)
    installed = {re.sub(r'[-_.]+', '-', entry['name']).lower(): entry['version'] for entry in json.loads(result.stdout)}
    if any(installed.get(name) != version for name, version in expected.items()) or set(installed) - expected.keys() - {'pip', 'setuptools', 'wheel'}:
        raise RuntimeError('Installed package versions differ from the FDTDX wheel lock.')
    checks = {}
    for target in _TARGETS[profile]:
        with tempfile.TemporaryDirectory(prefix='fdtdx-ready-') as scratch:
            env.update(JAX_PLATFORMS='cuda' if target == 'gpu' else 'cpu', MPLCONFIGDIR=scratch)
            run = subprocess.run([str(executable), '-I', '-c', _CHECK, str(resources), target], cwd=scratch,
                                 env=env, capture_output=True, text=True, timeout=180)
            lines = [line.removeprefix('WORKBENCH_CHECK=') for line in run.stdout.splitlines() if line.startswith('WORKBENCH_CHECK=')]
            if run.returncode or len(lines) != 1:
                raise RuntimeError(f'FDTDX {target} readiness solve failed: ' + (run.stderr or run.stdout)[-1500:])
            checks[target] = json.loads(lines[0])
    return {'locked_wheels':len(expected), 'targets':checks}


def setup_fdtdx(*, profile, offline, wheelhouse, data_dir, resources):
    if profile not in _TARGETS:
        raise ValueError('Choose fdtdx-cpu or fdtdx-cuda12.')
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64') or sys.version_info[:2] != (3, 11) or sys.implementation.name != 'cpython':
        raise RuntimeError('The managed FDTDX profiles require CPython 3.11 on Linux or WSL2 x86-64.')
    import fcntl
    resources = Path(resources).resolve()
    lock = resources / f'environment-{profile}-linux-64.lock'
    checksum = hashlib.sha256(lock.read_bytes()).hexdigest()
    _locked_versions(lock)
    state = Path(data_dir).expanduser().resolve() / '.local'
    state.mkdir(parents=True, exist_ok=True)
    path = state / 'runtime.json'
    target_key = 'fdtdx_gpu_python' if profile == 'fdtdx-cuda12' else 'fdtdx_cpu_python'
    with (state/'setup.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        settings = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(settings, dict) or not isinstance(settings.get('managed_fdtdx', {}), dict):
            raise ValueError('Malformed FDTDX runtime settings. The file was left intact.')
        profiles = settings.get('managed_fdtdx', {})
        previous = profiles.get(profile, {})
        runtimes = state / 'runtimes'
        if (isinstance(previous, dict) and isinstance(previous.get('python'), str)
                and previous.get('lock_sha256') == checksum and previous['python'] == settings.get(target_key)):
            executable = Path(previous['python'])
            if executable.is_file() and executable.is_relative_to(runtimes):
                try:
                    check = _verify(executable, profile, lock, resources)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                    pass
                else:
                    return {'status':'already-ready','python':str(executable),'profile':profile,'lock_sha256':checksum,'check':check}
        if wheelhouse is not None:
            wheelhouse = Path(wheelhouse).expanduser().resolve()
            if not wheelhouse.is_dir():
                raise ValueError('Pass an existing local wheelhouse directory.')
        if offline and wheelhouse is None:
            raise ValueError('Fresh offline setup requires --wheelhouse with every wheel in the profile lock.')
        runtimes.mkdir(exist_ok=True)
        prefix = runtimes / (profile + '-' + checksum[:10] + '-' + uuid.uuid4().hex[:12])
        executable = prefix / 'bin/python'
        try:
            subprocess.run([sys.executable, '-I', '-m', 'venv', str(prefix)], check=True)
            # --isolated excludes user pip settings; explicitly disable global configuration as well.
            pip = [str(executable), '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install',
                   '--require-hashes', '--only-binary=:all:', '-r', str(lock)]
            if offline:
                pip.append('--no-index')
            if wheelhouse is not None:
                pip.extend(['--find-links', str(wheelhouse)])
            env = {**os.environ, 'PIP_CONFIG_FILE':os.devnull}
            subprocess.run(pip, check=True, env=env)
            check = _verify(executable, profile, lock, resources)
            profiles = {**profiles, profile:{'python':str(executable),'lock_sha256':checksum}}
            _save_settings(path, {**settings, target_key:str(executable), 'managed_fdtdx':profiles})
        except BaseException:
            active = json.loads(path.read_text()) if path.exists() else {}
            if active.get(target_key) != str(executable):
                shutil.rmtree(prefix, ignore_errors=True)
            raise
        return {'status':'installed','python':str(executable),'profile':profile,'lock_sha256':checksum,'check':check}
