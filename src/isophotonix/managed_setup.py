"""Install and verify the locked Linux Meep/MPB environment on explicit request."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit
import uuid


def _locked_packages(executable, lock):
    requested = {}
    for line in lock.read_text().splitlines():
        if line.startswith('https://'):
            url = urlsplit(line)
            requested[Path(url.path).name] = url.fragment
    records = [json.loads(path.read_text()) for path in
               (executable.parent.parent/'conda-meta').glob('*.json')]
    installed = {record['fn']: record.get('md5') for record in records}
    if not requested or installed != requested:
        raise RuntimeError('Installed Conda artifact records differ from the pinned Meep/MPB lock.')
    versions = {record['name']: record['version'] for record in records}
    return {'locked_artifacts': len(records), 'mpb_version': versions['mpb']}


def _check_runtime(executable):
    """A propagated electric field and the known homogeneous MPB band at k=.25."""
    program = """
import json
import meep as mp
import numpy as np
from meep import mpb
mp.verbosity(0)
sim = mp.Simulation(dimensions=1, cell_size=mp.Vector3(z=3), resolution=16,
    boundary_layers=[mp.PML(.4)], sources=[mp.Source(mp.GaussianSource(1, fwidth=.5),
    component=mp.Ex, center=mp.Vector3(z=-.6))])
sim.run(until=6)
field = sim.get_array(component=mp.Ex, center=mp.Vector3(z=.4), size=mp.Vector3(z=.3))
peak = float(np.max(np.abs(field)))
sim.reset_meep()
mode = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(1,1,0)),
    default_material=mp.Medium(index=2), k_points=[mp.Vector3(.25)], resolution=8, num_bands=1)
mode.run_te()
frequency = float(mode.all_freqs[0][0])
if not np.isfinite(peak) or peak <= 1e-12 or not np.isfinite(frequency) or abs(frequency - .125) > 1e-5:
    raise RuntimeError('Native Meep/MPB numerical readiness check failed.')
print('WORKBENCH_CHECK='+json.dumps({'meep':mp.__version__, 'field_peak':peak,
    'mpb_frequency':frequency,'expected_mpb_frequency':.125}))
"""
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'PYTHONNOUSERSITE':'1'}
    env.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory(prefix='meep-check-') as scratch:
        env['MPLCONFIGDIR'] = scratch
        run = subprocess.run([str(executable), '-I', '-c', program], cwd=scratch, env=env,
                             capture_output=True, text=True, timeout=90)
    records = [line.removeprefix('WORKBENCH_CHECK=') for line in run.stdout.splitlines() if line.startswith('WORKBENCH_CHECK=')]
    if run.returncode or len(records) != 1:
        raise RuntimeError('Native readiness check failed: ' + (run.stderr or run.stdout)[-1200:])
    return json.loads(records[0])


def _save_settings(path, settings):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, prefix='runtime-', suffix='.tmp') as stream:
        temporary = Path(stream.name)
        json.dump(settings, stream, indent=2)
        stream.write('\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def setup_meep(*, conda: str | None, offline: bool, data_dir: Path, resources: Path):
    """Publish a new interpreter only after the locked environment passes both solvers."""
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
        raise RuntimeError('The managed Meep/MPB profile requires Linux or WSL2 on x86-64.')
    import fcntl
    lock = Path(resources) / 'environment-meep-linux-64.lock'
    checksum = hashlib.sha256(lock.read_bytes()).hexdigest()
    state = Path(data_dir).expanduser().resolve() / '.local'
    state.mkdir(parents=True, exist_ok=True)
    settings_path = state / 'runtime.json'
    with (state/'setup.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        if not isinstance(settings, dict):
            raise ValueError('Expected a settings object in runtime.json. The file was left intact.')
        previously = settings.get('managed_meep', {})
        runtimes = state / 'runtimes'
        if previously.get('lock_sha256') == checksum and settings.get('meep_python') == previously.get('python'):
            existing = Path(settings['meep_python'])
            if existing.is_file() and existing.is_relative_to(runtimes):
                try:
                    check = _check_runtime(existing)
                    check.update(_locked_packages(existing, lock))
                except (OSError, KeyError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                    pass  # Install a fresh prefix before switching from the previous one.
                else:
                    return {'status':'already-ready', 'python':str(existing), 'lock_sha256':checksum, 'check':check}
        command = shutil.which(conda or os.environ.get('CONDA_EXE') or 'conda')
        if not command:
            raise RuntimeError('Install Conda or pass the path to its executable with --conda.')
        runtimes.mkdir(exist_ok=True)
        # Conda embeds its installation prefix. Use the final location and switch
        # the small runtime pointer atomically after successful verification.
        prefix = runtimes / ('meep-' + checksum[:10] + '-' + uuid.uuid4().hex[:12])
        executable = prefix / 'bin/python'
        args = [command, 'create', '--yes', '--prefix', str(prefix), '--file', str(lock)]
        if offline:
            args.append('--offline')
        env = {**os.environ, 'CONDA_REGISTER_ENVS':'false', 'CONDA_AUTO_UPDATE_CONDA':'false'}
        try:
            subprocess.run(args, env=env, check=True)
            check = _check_runtime(executable)
            check.update(_locked_packages(executable, lock))
            settings.update(meep_python=str(executable),
                managed_meep={'python':str(executable),'lock_sha256':checksum})
            _save_settings(settings_path, settings)
        except BaseException:
            # An interrupt may arrive just after atomic publication. Preserve
            # the new interpreter if readers can already see that pointer.
            active = json.loads(settings_path.read_text()) if settings_path.exists() else {}
            if active.get('meep_python') != str(executable):
                shutil.rmtree(prefix, ignore_errors=True)
            raise
        return {'status':'installed', 'python':str(executable), 'lock_sha256':checksum, 'check':check}
