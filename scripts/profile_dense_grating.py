"""Compile the four dense 12-cell CUDA stage shapes and compare them against measured memory use."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.calibrate_finite_grating import monitoring_wavelengths
from scripts.grating_cuda_identity import available_mib
from scripts.grating_memory_budget import estimate

ROOT = Path(__file__).resolve().parents[1]


def budget():
    historical = json.loads((ROOT/'docs/verification/v012-grating-memory-budget.json').read_text())
    previous = historical['reference_runs']['lumerical']
    observed = next(row for row in historical['refinements'] if row['geometry'] == 'lumerical')
    plans, estimates = {}, {}
    free = available_mib()
    with tempfile.TemporaryDirectory() as temp:
        for model in ('volume-average', 'direct-yee'):
            for role in ('reference', 'device'):
                name = f'{model}-{role}'
                output = Path(temp)/(name+'.json')
                command = [sys.executable, '-m', 'scripts.plan_grating_memory', '--geometry', 'lumerical',
                           '--resolution', '12', '--grating-spectrum', 'dense', '--grating-material-model', model,
                           '--output', str(output)]
                if role == 'reference':
                    command.append('--reference')
                subprocess.run(command, cwd=ROOT, check=True)
                plans[name] = json.loads(output.read_text())
                estimates[name] = estimate(previous['shape_only'], previous['actual_one_step'], plans[name],
                                           historical['measured_idle_used_floor_mib'], free,
                                           observed['baseline_observed_global_peak_mib'])
    return {'geometry':'lumerical', 'resolution':12,'wavelength_um':monitoring_wavelengths({'grating_spectrum':'dense'}).tolist(),
            'required_free_with_reserve_mib':max(e['required_free_with_reserve_mib'] for e in estimates.values()),
            'estimated_against_measured_free_mib':free, 'gpu':previous['shape_only']['gpu'],
            'basis':'Maximum requirement over both material samplings and native reference/device detector shapes. Uses the measured 12-cell allocation, observed peak, 20% reserve and 1 GiB reserve.',
            'baseline_evidence':'v012-grating-memory-budget.json','shape_plans':plans,'estimates':estimates}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    lock_name=os.environ.get('PHOTONICS_GPU_LOCK')
    if lock_name:
        with Path(lock_name).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            report=budget()
    else:
        report=budget()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
