"""Provision a hash-locked BeamZ CPU reference runtime alongside other solvers."""
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


def _check(executable, lock, script, settings_env):
    pip = [str(executable), '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check']
    subprocess.run([*pip, 'check'], check=True, env=settings_env)
    listed = subprocess.run([*pip, 'list', '--format=json'], capture_output=True, text=True,
                            check=True, env=settings_env)
    normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
    installed = {normalize(p['name']):p['version'] for p in json.loads(listed.stdout)
                 if p['name'] not in ('pip', 'setuptools', 'wheel')}
    versions = {}
    for line in lock.read_text().splitlines():
        if line and not line.startswith('#'):
            package, version = line.split()[0].split('==')
            versions[normalize(package)] = version
    if installed != versions:
        raise RuntimeError('Installed packages differ from the BeamZ CPU wheel lock.')
    with tempfile.TemporaryDirectory(prefix='beamz-ready-') as scratch:
        result = Path(scratch)/'result'
        subprocess.run([str(executable), '-I', str(script), '--device', 'cpu', '--output', str(result),
                        '--no-export'], cwd=scratch, env={**settings_env, 'MPLCONFIGDIR':scratch},
                       timeout=180, check=True)
        report = json.loads((result/'report.json').read_text())
        if not report['passed'] or report['device_requested'] != 'cpu' or not all(
                r['actual_platform'] == 'cpu' for r in report['propagation']):
            raise RuntimeError('BeamZ CPU numerical readiness check failed.')
        return {'locked_wheels':len(versions), 'actual_device':report['actual_device'], 'beamz':report['release'],
                'finest_phase_relative_error':report['propagation'][-1]['continuum_relative_error'],
                'slab_neff':report['slab_modes_cpu'][-2]['neff'], 'passed':True}


def setup_beamz(*, offline, wheelhouse, data_dir, resources):
    if (platform.system() != 'Linux' or platform.machine() not in ('x86_64','amd64')
            or sys.version_info[:2] != (3,11) or sys.implementation.name != 'cpython'):
        raise RuntimeError('Managed BeamZ requires CPython 3.11 on Linux or WSL2 x86-64.')
    import fcntl

    resources = Path(resources).resolve()
    lock = resources/'environment-beamz-cpu-linux-64.lock'
    script = resources/'scripts/benchmark_beamz_reference.py'
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    state = Path(data_dir).expanduser().resolve()/'.local'
    state.mkdir(parents=True, exist_ok=True)
    settings_path, runtimes = state/'runtime.json', state/'runtimes'
    env = {**os.environ,'PIP_CONFIG_FILE':os.devnull,'PYTHONNOUSERSITE':'1'}
    env.pop('PYTHONPATH',None)
    env.pop('PYTHONHOME',None)
    with (state/'setup.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        if not isinstance(settings,dict) or not isinstance(settings.get('managed_beamz',{}),dict):
            raise ValueError('Malformed BeamZ runtime settings. The file was left intact.')
        previous = settings.get('managed_beamz',{})
        if (previous.get('lock_sha256') == digest and isinstance(previous.get('python'),str)
                and settings.get('beamz_cpu_python') == previous['python']):
            current = Path(previous['python'])
            if current.is_file() and current.parent.parent.parent.resolve() == runtimes:
                try:
                    check = _check(current,lock,script,env)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                    pass
                else:
                    return {'profile':'beamz-cpu','status':'already-ready',**previous,'check':check}
        if wheelhouse is not None:
            wheelhouse = Path(wheelhouse).expanduser().resolve()
            if not wheelhouse.is_dir():
                raise ValueError('Pass an existing local wheelhouse directory.')
        if offline and wheelhouse is None:
            raise ValueError('Fresh offline BeamZ setup requires a wheelhouse containing the locked wheels.')
        runtimes.mkdir(exist_ok=True)
        prefix = runtimes/f'beamz-cpu-{digest[:10]}-{uuid.uuid4().hex[:12]}'
        executable = prefix/'bin/python'
        try:
            subprocess.run([sys.executable, '-I', '-m', 'venv', str(prefix)], check=True, env=env)
            install = [str(executable),'-I','-m','pip','--isolated','--disable-pip-version-check','install',
                       '--require-hashes','--only-binary=:all:','-r',str(lock)]
            if offline:
                install.append('--no-index')
            if wheelhouse is not None:
                install.extend(['--find-links',str(wheelhouse)])
            subprocess.run(install, check=True, env=env)
            check = _check(executable,lock,script,env)
            record = {'python':str(executable),'lock_sha256':digest}
            _save_settings(settings_path,{**settings,'beamz_cpu_python':str(executable),'managed_beamz':record})
        except BaseException:
            active = json.loads(settings_path.read_text()) if settings_path.exists() else {}
            if active.get('beamz_cpu_python') != str(executable):
                shutil.rmtree(prefix,ignore_errors=True)
            raise
        return {'profile':'beamz-cpu','status':'installed',**record,'check':check}
