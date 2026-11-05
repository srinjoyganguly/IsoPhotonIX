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


def verify_source(root):
    manifest = json.loads((root/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Export source or settings changed: {name}')
    return manifest


def measured_observables(report):
    """Numerical fields which the GUI renders; wall times and run dates vary."""
    result = {key: report[key] for key in ('wavelength_um','s','s_real','s_imag','power','modal_power')
              if key in report}
    if report.get('case_id')=='grating-gaussian-profile':
        beam=report['result']
        for key in ('wavelength_um','beam_to_guide','valid_beam_bins'):
            if key in beam:
                result['beam/'+key]=beam[key]
    for name, data in report.get('series', {}).items():
        for key in ('wavelength_um','through','drop','reflection','add_side','valid_bins'):
            if key in data:
                result[f'{name}/{key}'] = data[key]
    for name, meshes in report.get('studies', {}).items():
        for mesh, data in meshes.items():
            for key in ('wavelength_um','beam_to_guide','guide_radiation_up','guide_radiation_down','power_sum','valid_beam_bins','valid_energy_bins'):
                if key in data:
                    result[f'{name}/{mesh}/{key}'] = data[key]
    for key in ('wavelength_um', 'beam_to_guide', 'guide_radiation_up'):
        compact = report.get('compact_tidy_domain', {})
        if key in compact:
            result[f'compact/{key}'] = compact[key]
    for name, data in report.get('isolated_finite_height_ring', {}).items():
        if isinstance(data, dict) and 'measured_bare_ring_fsr_nm' in data:
            result[f'bare_ring/{name}/fsr_nm'] = data['measured_bare_ring_fsr_nm']
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
        a,b=np.asarray(value,dtype=float),np.asarray(fresh[key],dtype=float)
        if a.shape!=b.shape or not within_precision(a, b):
            raise ValueError(f'Displayed observable changed in {key}.')
    return {'passed':True, 'all_array_bytes_equal': not differences, 'within_tolerance':differences,
            'precision':{'relative':2e-6,'absolute':1e-10,'per_array_peak_fraction':2e-6},
            'reported_observables_checked':list(first)}


def run(root, output, checkpoint=None):
    root,output = Path(root).resolve(),Path(output).resolve()
    manifest=verify_source(root)
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
                    'scripts.calibrate_multimode':'--checkpoint-dir'}.get(manifest['worker'])
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
        process=subprocess.run(command,cwd=root/'source',env=env)
        # A solver may emit a useful partial/unqualified result with exit code 1.
        # Preserve its diagnostics; require a successful worker exit for a qualified job.
        if (work/'report.json').is_file():
            if work!=staged:
                shutil.copytree(work,staged)
            staged.rename(output)
        process.check_returncode()
    report=json.loads((output/'report.json').read_text())
    if not (output/'arrays.npz').is_file():
        raise ValueError('No numerical arrays were produced.')
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
