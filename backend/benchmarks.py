"""Expose recorded local benchmarks without importing or starting solver runtimes."""
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from .runtime import ROOT

REPORT = ROOT / 'docs/verification/public-mode-benchmarks.json'
NOTES = [
    'These are cross-sectional eigenmode benchmarks. They do not certify driven 3D FDTD fields, transmission, ports or arbitrary geometries.',
    'Displayed values come from a saved report in the project checkout. Loading or refreshing this panel does not run benchmarks or verify this machine; use the reproduction command for a fresh local run.',
    'Published solver values and analytic references are identified separately. No commercial runtime or cloud solve is required.',
    'Numerical differences are reported without an accuracy pass/fail badge. Refinement and reference limitations remain part of the evidence.',
]


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def benchmark_report(path=None):
    path = Path(path) if path is not None else REPORT
    if not path.exists():
        return {'schema_version': 1, 'scope': 'mode-benchmarks', 'status': 'not_run',
                'cases': [], 'notes': NOTES,
                'reproduction_command': '.venv/bin/python scripts/benchmark_waveguide_modes.py'}
    raw = path.read_bytes()
    report = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Non-finite report value: {value}')))
    if not isinstance(report, dict) or report.get('schema_version') != 1 or not isinstance(report.get('cases'), list) or not report['cases']:
        raise ValueError('Benchmark report is incomplete or has an unsupported schema.')
    if not report.get('solver') or not report.get('versions') or not report.get('generated_at'):
        raise ValueError('Benchmark report lacks recorded solver provenance.')
    ids = set()
    for case in report['cases']:
        if not isinstance(case, dict) or not case.get('id') or case['id'] in ids:
            raise ValueError('Benchmark case identity is missing or duplicated.')
        ids.add(case['id'])
        if case.get('reference_kind') not in ('analytic_reference', 'published_solver_value'):
            raise ValueError('Benchmark must distinguish analytic and published solver references.')
        url = urlsplit(case.get('source_url', ''))
        if url.scheme != 'https' or not url.hostname:
            raise ValueError('Benchmark source must have an explicit HTTPS reference.')
        reference, computed, relative = (case.get(key) for key in ('reference_neff', 'computed_neff', 'relative_error'))
        if not all(_finite(value) for value in (reference, computed, relative)) or reference <= 0 or computed <= 0 or relative < 0:
            raise ValueError('Benchmark report has missing or invalid numerical values.')
        expected = abs(computed-reference)/reference
        if not math.isclose(expected, relative, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError('Recorded benchmark discrepancy does not match the saved numerical values.')
    return {**report, 'scope': 'mode-benchmarks', 'status': 'available',
            'report_sha256': hashlib.sha256(raw).hexdigest(),
            'notes': [*NOTES, *report.get('notes', [])],
            'reproduction_command': report.get('reproduction_command', '.venv/bin/python scripts/benchmark_waveguide_modes.py')}
