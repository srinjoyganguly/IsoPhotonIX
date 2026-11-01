#!/usr/bin/env python3
"""Reproduce public mode examples with local MPB, never commercial runtimes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.mode_benchmarks import CASES, solve_case
from backend.runtime import interpreter, worker_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', *CASES], default='all')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--output', type=Path, default=ROOT/'.local/public-benchmarks')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--domain-scale', type=float, default=1., help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single:
        if args.case == 'all' or len(args.resolutions) != 1:
            parser.error('A worker needs exactly one case and resolution.')
        solve_case(args.case, args.resolutions[0], args.output, args.domain_scale)
        return
    resolutions = sorted(set(args.resolutions))
    if len(resolutions) < 2 or any(not 16 <= r <= 256 for r in resolutions):
        parser.error('Choose at least two distinct resolutions from 16 through 256.')
    # Fail before writing any result if the requested environment lacks MPB.
    executable = interpreter('meep')
    env = worker_env('meep', 'cpu')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    probe = subprocess.run([executable, '-c', 'from meep import mpb; print(mpb.__version__)'],
                           capture_output=True, text=True, env=env, timeout=30)
    if probe.returncode:
        raise RuntimeError('This interpreter does not provide Meep/MPB. Use the documented Meep environment.\n'+probe.stderr)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(CASES) if args.case == 'all' else [args.case]
    report = {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
              'solver': 'MPB', 'versions': None, 'cases': [],
              'reproduction_command': '.venv/bin/python scripts/benchmark_waveguide_modes.py --output .local/public-benchmarks-rerun --report docs/verification/public-mode-benchmarks.json',
              'notes': ['Actual local CPU eigenmode calculations; no commercial runtime, cloud solve or access token.',
                        'Effective-index checks only. These mode studies do not validate the driven Meep/FDTDX 3D adapters.',
                        'Relative error is abs(local-reference)/abs(reference). There is no automatic accuracy pass/fail threshold.',
                        'Commercial references are printed finite-grid scalar values with limited precision, not exact truth.',
                        'Each solve uses a fresh process to keep Meep/MPB global state isolated.']}

    def execute(case, resolution, scale):
        name = f'{case}-r{resolution}-domain{scale:g}'
        output = args.output/name
        if output.exists():
            raise FileExistsError(f'{output} exists; choose a new --output to preserve earlier evidence.')
        command = [executable, str(Path(__file__).resolve()), '--single', '--case', case,
                   '--resolutions', str(resolution), '--domain-scale', str(scale), '--output', str(output)]
        run = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180)
        if output.exists():
            (output/'solver.log').write_text(run.stdout+run.stderr)
        if run.returncode:
            raise RuntimeError(f'{name} failed; no result is claimed.\n'+run.stderr[-3000:])
        result = json.loads((output/'result.json').read_text())
        print(f"{name}: neff={result['computed_neff']:.10f}, relative discrepancy={result['relative_error']:.6g}", flush=True)
        return result

    for case in cases:
        refinements = [execute(case, resolution, 1.) for resolution in resolutions]
        fine = dict(refinements[-1])
        enlarged = execute(case, resolutions[-1], 1.5)
        fine['refinements'] = [{key: r[key] for key in
            ('resolution_cells_per_um', 'computed_neff', 'relative_error', 'absolute_error',
             'grid_shape', 'elapsed_s', 'array_sha256', 'analytic_relative_error')} for r in refinements]
        fine['last_refinement_relative_change'] = abs(refinements[-1]['computed_neff']-refinements[-2]['computed_neff'])/abs(refinements[-1]['computed_neff'])
        fine['domain_check'] = {key: enlarged[key] for key in ('domain_scale', 'computed_neff', 'grid_shape', 'array_sha256')}
        fine['domain_check']['relative_change'] = abs(enlarged['computed_neff']-fine['computed_neff'])/abs(fine['computed_neff'])
        fine['domain_check']['note'] = 'Increasing domain also changes optimized FFT grid spacing; this is combined domain/discretization sensitivity, not a pure boundary error bound.'
        fine['reproduction_command'] = f'.venv/bin/python scripts/benchmark_waveguide_modes.py --case {case} --resolutions '+ ' '.join(map(str,resolutions)) + f' --output .local/{case}-rerun'
        report['cases'].append(fine)
        report['versions'] = fine['versions']
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Saved {len(cases)} benchmark cases. Numerical fields and logs remain under the selected output directory.')


if __name__ == '__main__':
    main()
