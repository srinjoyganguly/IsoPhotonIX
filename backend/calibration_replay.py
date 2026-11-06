"""Portable controller for calibrated component simulations, copied into each export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile


def gpu_lock_path():
    """Use one private per-user lock for calibrated GPU programs and notebooks."""
    override=os.environ.get('PHOTONICS_GPU_LOCK')
    if override:
        return Path(override).expanduser()
    import stat
    uid=os.getuid()
    fallback=Path(tempfile.gettempdir())/f'photonics-workbench-{uid}'
    runtime=os.environ.get('XDG_RUNTIME_DIR')
    for directory in ([Path(runtime),fallback] if runtime else [fallback]):
        if directory==fallback:
            directory.mkdir(mode=0o700,exist_ok=True)
        try:
            info=directory.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(info.st_mode) and info.st_uid==uid and info.st_mode & 0o077==0:
            return directory/'photonics-workbench-gpu.lock'
    raise PermissionError('A private per-user directory is required for the local GPU lock.')


def verify_source(root, *, editable_notebook=False):
    manifest = json.loads((root/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if editable_notebook and name=='calibration.ipynb':
            # Jupyter persists outputs, execution counts and user-chosen result paths.
            continue
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Export source or settings changed: {name}')
    return manifest


def measured_observables(report):
    """Numerical fields which the GUI renders; wall times and run dates vary."""
    result = {key: report[key] for key in ('wavelength_um','s','s_real','s_imag','power','modal_power','ports','port_order','channels')
              if key in report}
    if report.get('case_id') in ('grating-gaussian-profile','finite-grating-3d'):
        beam=report['result']
        for key in ('wavelength_um','beam_to_guide','valid_beam_bins','direct_output_fraction','valid_direct_bins','guide_to_beam','guide_radiation_up','valid_energy_bins'):
            if key in beam:
                result['beam/'+key]=beam[key]
    for key,value in report.get('radiation',{}).items():
        if isinstance(value,list):
            result['radiation/'+key]=value
    for key in ('array','index','axis_order','component','horizontal_axis','vertical_axis','horizontal_um','vertical_um','wavelength_um','time_um_c'):
        if key in report.get('field_preview',{}):
            result['field/'+key]=report['field_preview'][key]
    if report.get('measured_drop_fsr_nm') is not None:
        result['measured_drop_fsr_nm']=report['measured_drop_fsr_nm']
    for i,peak in enumerate(report.get('loaded_loss',{}).get('peaks',[])):
        result[f'pole/{i}/qualified']=bool(peak['qualified'])
        if peak['qualified']:
            result[f'pole/{i}/wavelength_um']=peak['sampled_drop_peak_um']
            for key,value in peak['qualified_equal_bus_loss'].items():
                if value is not None:
                    result[f'pole/{i}/'+key]=value
    for name, data in report.get('series', {}).items():
        for key in ('wavelength_um','through','drop','reflection','add_side','valid_bins'):
            if key in data:
                result[f'{name}/{key}'] = data[key]
    for name, meshes in report.get('studies', {}).items():
        for mesh, data in meshes.items():
            for key in ('wavelength_um','beam_to_guide','guide_radiation_up','guide_radiation_down','power_sum','valid_beam_bins','valid_energy_bins'):
                if key in data:
                    result[f'{name}/{mesh}/{key}'] = data[key]
    for key in ('wavelength_um', 'beam_to_guide', 'guide_radiation_up', 'valid_beam_bins', 'valid_energy_bins'):
        compact = report.get('compact_tidy_domain', {})
        if key in compact:
            result[f'compact/{key}'] = compact[key]
    for name, data in report.get('isolated_finite_height_ring', {}).items():
        if isinstance(data,dict):
            for field in ('measured_bare_ring_fsr_nm','published_loaded_ring_fsr_nm'):
                if field in data:
                    result[f'bare_ring/{name}/{field}'] = data[field]
    return result


def compare(original, output):
    import numpy as np

    def within_precision(left, right):
        # Dense near-zero samples need an absolute bound relative to the
        # measured array's own peak. Each array is checked independently.
        if not np.array_equal(np.isfinite(left), np.isfinite(right)):
            return False
        finite = np.isfinite(left) & np.isfinite(right)
        peak = float(np.max(abs(left[finite]), initial=0))
        return np.allclose(left, right, rtol=2e-6, atol=1e-10 + 2e-6*peak, equal_nan=True)

    differences = {}
    raw_files = {p.relative_to(original) for p in original.rglob('*.npz')}
    if not raw_files or raw_files != {p.relative_to(output) for p in output.rglob('*.npz')}:
        raise ValueError('The replay has different numerical files.')
    for path in sorted(original.rglob('*.npz')):
        relative = path.relative_to(original)
        with np.load(path,allow_pickle=False) as a, np.load(output/relative,allow_pickle=False) as b:
            if set(a.files)!=set(b.files):
                raise ValueError(f'Different result array names in {relative}.')
            for key in a.files:
                left,right=a[key],b[key]
                if left.shape!=right.shape or left.dtype!=right.dtype:
                    raise ValueError(f'Different array type/shape for {relative}:{key}.')
                if left.tobytes()==right.tobytes(): continue
                if left.dtype.kind not in 'fc' or not within_precision(left, right):
                    raise ValueError(f'Replay differs from saved samples in {relative}:{key}.')
                finite = np.isfinite(left) & np.isfinite(right)
                differences[f'{relative}:{key}'] = float(np.max(abs(left[finite]-right[finite]),initial=0))
    first=measured_observables(json.loads((original/'report.json').read_text()))
    fresh=measured_observables(json.loads((output/'report.json').read_text()))
    if not first or first.keys()!=fresh.keys():
        raise ValueError('The replay has different reported observables.')
    for key,value in first.items():
        if key in ('ports','port_order','channels') or isinstance(value,str):
            if value!=fresh[key]:
                raise ValueError(f'Displayed label or channel order changed in {key}.')
            continue
        a,b=np.asarray(value,dtype=float),np.asarray(fresh[key],dtype=float)
        if a.shape!=b.shape or not within_precision(a, b):
            raise ValueError(f'Displayed observable changed in {key}.')
    return {'passed':True, 'all_array_bytes_equal': not differences, 'within_tolerance':differences,
            'precision':{'relative':2e-6,'absolute':1e-10,'per_array_peak_fraction':2e-6},
            'reported_observables_checked':list(first)}


def run(root, output, checkpoint=None):
    if checkpoint is None:
        checkpoint = os.environ.get('PHOTONICS_CHECKPOINT') or None
    root,output = Path(root).resolve(),Path(output).resolve()
    manifest=verify_source(root,editable_notebook=True)
    config=json.loads((root/'config.json').read_text())
    from workbench_solver_environment import default_user_state, solver_interpreter
    solver=solver_interpreter(config['solver'],config['device'],default_user_state())
    if output.exists():
        raise FileExistsError('Choose a fresh result directory.')
    output.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1',
         'XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
    if config['solver']=='fdtdx':
        env['JAX_PLATFORMS']='cuda' if config['device']=='gpu' else 'cpu'
    env.pop('PYTHONPATH',None)
    env.pop('PYTHONHOME',None)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staged=Path(tmp)/'result'
        work=staged
        arguments=list(manifest['arguments'])
        if checkpoint is not None:
            cache=Path(checkpoint).resolve()
            if cache==output or output in cache.parents:
                raise ValueError('Choose a checkpoint folder outside the result directory.')
            option={'scripts.calibrate_ring':'--checkpoint','scripts.calibrate_grating':'--checkpoint',
                    'scripts.calibrate_multimode':'--checkpoint-dir','scripts.calibrate_planar_tm':'--checkpoint-dir',
                    'scripts.calibrate_layout_ports':'--checkpoint-dir','scripts.calibrate_layout_ports3d':'--checkpoint',
                    'scripts.calibrate_loaded_ring3d':'--checkpoint','scripts.calibrate_finite_grating':'--checkpoint'}.get(manifest['worker'])
            if option:
                arguments += [option,str(cache)]
            elif manifest['worker'] in ('scripts.calibrate_multiport','scripts.calibrate_fdtdx_multiport'):
                cache.mkdir(parents=True,exist_ok=True)
                work=cache/'native-multiport'
                if work.exists():
                    arguments.append('--resume')
            else:
                raise ValueError('No checkpoint format is defined for this worker.')
        command=[solver,'-m',manifest['worker'],*arguments,'--output',str(work)]
        if config['device']=='gpu':
            import fcntl
            lock_path=gpu_lock_path()
            env['PHOTONICS_GPU_LOCK']=str(lock_path)
            # These two native workers acquire the same lock for each separate
            # reference/device child. Their portable controller must not hold it.
            if manifest['worker'] in ('scripts.calibrate_finite_grating','scripts.calibrate_layout_ports3d'):
                process=subprocess.run(command,cwd=root/'source',env=env)
            else:
                with lock_path.open('a') as guard:
                    fcntl.flock(guard,fcntl.LOCK_EX)
                    process=subprocess.run(command,cwd=root/'source',env=env)
        else:
            process=subprocess.run(command,cwd=root/'source',env=env)
        # The shared checkpoint may already contain another completed report.
        # On rejection leave that cache intact and do not publish it as this job.
        if process.returncode and work!=staged:
            process.check_returncode()
        # A fresh worker can emit a useful unqualified report before exiting.
        if (work/'report.json').is_file():
            if work!=staged:
                shutil.copytree(work,staged)
            staged.rename(output)
        process.check_returncode()
    report=json.loads((output/'report.json').read_text())
    if not (output/'arrays.npz').is_file():
        raise ValueError('No numerical arrays were produced.')
    if config['solver']=='fdtdx':
        platform=str(report.get('actual_device','')).lower().split(':')[0]
        allowed={'gpu':{'cuda','gpu'},'cpu':{'cpu'}}[config['device']]
        if platform not in allowed:
            raise ValueError('The measured field device differs from the requested execution target.')
    reference=root/'reference'
    for key in ('passed','all_stages_settled','settled'):
        if key in report and report[key] is False:
            raise ValueError(f'The numerical report failed {key}; inspect its measurements.')
    if reference.is_dir():
        hashes=json.loads((root/'reference-sha256.json').read_text())
        for name,digest in hashes.items():
            with (reference/name).open('rb') as stream:
                measured = hashlib.file_digest(stream,'sha256').hexdigest()
            if measured!=digest:
                raise ValueError(f'Saved reference changed: {name}')
        checks=compare(reference,output)
        (output/'reproduction.json').write_text(json.dumps(checks,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,help='persistent, settings-verified native stage cache')
    args=parser.parse_args()
    run(Path(__file__).resolve().parent,args.output,args.checkpoint)
