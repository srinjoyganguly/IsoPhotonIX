"""Measure the four published COMSOL Fabry–Perot poles with a local Meep ringdown."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

# The Hankel matrices are small. A single BLAS thread also makes notebook replays deterministic.
if __name__ == '__main__':
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

C_MHZ = 299.792458  # One length unit is one metre. One time unit is 1 m/c.
PUBLISHED = [(37.5, 3.08), (74.9, 6.15), (112.4, 9.22), (149.9, 12.3)]
SOURCE_URL = ('https://doc.comsol.com/5.5/doc/com.comsol.help.models.woptics.'
              'fabry_perot/models.woptics.fabry_perot.pdf')
# Offsets from the actual source shutoff, in 1 m/c units.
FITS = [(3, 55, 16), (1, 45, 16), (7, 61, 16), (3, 55, 12), (3, 55, 20)]
CASES = [(32, 3, 2), (64, 3, 2), (128, 3, 2), (256, 3, 2), (256, 6, 2), (256, 3, 3)]


def measured_modes(times, field, rank):
    """Estimate freely decaying frequencies using a truncated Hankel matrix pencil."""
    times = np.asarray(times, dtype=float)
    field = np.asarray(field, dtype=float)
    if (times.ndim != 1 or field.shape != times.shape or times.size < 280
            or not np.isfinite(times).all() or not np.isfinite(field).all()):
        raise ValueError('Need at least 280 finite scalar samples and times.')
    dt = times[1] - times[0]
    if dt <= 0 or not np.allclose(np.diff(times), dt, atol=1e-10, rtol=0):
        raise ValueError('The ringdown requires increasing, equally spaced samples.')
    if np.max(np.abs(field)) == 0:
        raise ValueError('No field at the observation point.')
    rows = 128
    h = np.lib.stride_tricks.sliding_window_view(field, rows).T.copy()
    u, singular, vh = np.linalg.svd(h[:, :-1], full_matrices=False)
    if rank < 8 or rank >= len(singular) or singular[rank-1] <= 1e-13 * singular[0]:
        raise ValueError('Insufficient numerical rank to resolve the requested pole fit.')
    reduced = (u[:, :rank].T @ h[:, 1:] @ vh[:rank].T) / singular[:rank]
    roots = np.linalg.eigvals(reduced).astype(complex)
    powers = roots[None, :] ** np.arange(len(field))[:, None]
    amplitudes = np.linalg.lstsq(powers, field, rcond=1e-12)[0]
    relative_residual = float(np.linalg.norm(field - powers @ amplitudes) / np.linalg.norm(field))
    modes = []
    for root, amplitude in zip(roots, amplitudes):
        if abs(root) == 0:
            continue
        pole = np.log(root) / dt
        frequency = float(pole.imag / (2 * np.pi) * C_MHZ)
        # Same fixed search band for every fit; no pole initializers or vendor frequencies.
        if 20 < frequency < 166 and pole.real < 0 and abs(amplitude) > 1e-5 * max(abs(amplitudes)):
            modes.append({'frequency_MHz': frequency, 'quality_factor': float(-pole.imag/(2*pole.real)),
                          'amplitude_decay_per_time_unit': float(-pole.real),
                          'fit_amplitude_abs': float(abs(amplitude))})
    modes.sort(key=lambda x: x['frequency_MHz'])
    if len(modes) != 4:
        raise ValueError(f'Found {len(modes)} resolvable decaying poles in 20–166 MHz; expected four.')
    return {'modes': modes, 'relative_field_fit_residual': relative_residual}


def ringdown(resolution, pml_m, gap_m):
    import meep as mp

    source = mp.GaussianSource(frequency=.295, fwidth=.50, cutoff=5)
    geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, 1), material=mp.Medium(index=4))]
    sim = mp.Simulation(
        cell_size=mp.Vector3(z=1 + 2 * (pml_m + gap_m)), dimensions=1,
        geometry=geometry, resolution=resolution, Courant=.5,
        boundary_layers=[mp.PML(pml_m)],
        sources=[mp.Source(source, component=mp.Ex, center=mp.Vector3(z=-.29))],
    )
    times, field = [], []

    def sample(sim):
        times.append(sim.meep_time())
        field.append(sim.get_field_point(mp.Ex, mp.Vector3(z=.13)).real)

    try:
        sim.run(mp.at_every(.125, sample), until_after_sources=84)
        source_off = float(sim.fields.last_source_time())
    finally:
        sim.reset_meep()
    return np.array(times), np.array(field), source_off


def max_changes(base_modes, other_modes):
    return {key: max(abs(a[key] - b[key]) / a[key] for a, b in zip(base_modes, other_modes))
            for key in ('frequency_MHz', 'quality_factor')}


def write_notebook(source, output):
    # The solver can use its managed interpreter even if the Jupyter kernel is elsewhere.
    launcher = """import os, sys, subprocess, tempfile
from pathlib import Path
meep_python = os.environ.get('MEEP_PYTHON', sys.executable)
output = Path('cavity-result').resolve()
with tempfile.TemporaryDirectory() as tmp:
    script = Path(tmp) / 'cavity.py'
    script.write_text(solver_source, encoding='utf-8')
    solver_env = os.environ.copy()
    solver_env.pop('PYTHONPATH', None)
    solver_env.pop('PYTHONHOME', None)
    solver_env.update(MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1')
    subprocess.run([meep_python, str(script), '--output', str(output), '--no-export'],
                   env=solver_env, check=True)
import json
report = json.loads((output / 'report.json').read_text())
[(row['frequency_MHz'], row['quality_factor']) for row in report['reference_comparison']]"""
    intro = ("# Passive slab ringdown\nRun with a Meep-enabled kernel or set `MEEP_PYTHON` to its interpreter. "
             "The notebook embeds the complete solver and writes fresh measured traces and results to "
             "`cavity-result`. Choose a new output directory for each execution.\n")
    cells = [{'cell_type': 'markdown', 'id': 'instructions', 'metadata': {}, 'source': intro}]
    for name, code in [('source', 'solver_source = ' + repr(source)), ('measure', launcher)]:
        cells.append({'cell_type': 'code', 'id': name, 'metadata': {}, 'outputs': [],
                      'execution_count': None, 'source': code})
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': cells}
    output.write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')


def run(out, export):
    import meep as mp

    mp.verbosity(0)
    traces, cases = {}, []
    for res, pml, gap in CASES:
        times, field, source_off = ringdown(res, pml, gap)
        key = f'r{res}_pml{pml}_gap{gap}'
        traces[key+'_time'] = times
        traces[key+'_Ex'] = field
        fits = []
        for after, end, rank in FITS:
            start_time, end_time = source_off + after, source_off + end
            mask = (times >= start_time) & (times < end_time)
            fits.append({'rank': rank, 'window_time_units': [start_time, end_time],
                         **measured_modes(times[mask], field[mask], rank)})
        cases.append({'trace_key': key, 'resolution_cells_per_m': res, 'air_gap_m': gap,
                      'pml_m': pml, 'source_off_time_units': source_off, 'recorded_samples': len(times),
                      'peak_Ex': float(max(abs(field))), 'fits': fits})
    base = cases[3]['fits'][0]['modes']
    metrics = []
    for mode, (frequency, q), row in zip(range(1, 5), PUBLISHED, base):
        exact_f = mode * C_MHZ / 8
        exact_q = -mode * np.pi / (2 * np.log(.6))
        metrics.append({'longitudinal_order': mode, **row, 'published_frequency_MHz': frequency,
                        'published_quality_factor': q, 'analytic_frequency_MHz': exact_f,
                        'analytic_quality_factor': exact_q,
                        'frequency_relative_error_published': abs(row['frequency_MHz']/frequency - 1),
                        'q_relative_error_published': abs(row['quality_factor']/q - 1),
                        'frequency_relative_error_analytic': abs(row['frequency_MHz']/exact_f - 1),
                        'q_relative_error_analytic': abs(row['quality_factor']/exact_q - 1)})
    changes = {key:max_changes(base, cases[i]['fits'][0]['modes']) for key, i in
               [('mesh_128_to_256', 2), ('pml_3_to_6_m', 4), ('airgap_2_to_3_m', 5)]}
    fit_changes = [max_changes(base, alt['modes']) for alt in cases[3]['fits'][1:]]
    changes['window_and_rank'] = {k:max(v[k] for v in fit_changes) for k in fit_changes[0]}
    checks = {
        'finest_within_0_3_percent_of_both_references':
            all(row[k] <= .003 for row in metrics for k in row if '_relative_error_' in k),
        'finest_mesh_change_within_0_2_percent': max(changes['mesh_128_to_256'].values()) <= .002,
        'boundary_and_fit_changes_within_0_12_percent':
            all(max(v.values()) <= .0012 for k,v in changes.items() if k != 'mesh_128_to_256'),
        'all_fits_relative_residual_within_0_005':
            all(f['relative_field_fit_residual'] <= .005 for c in cases for f in c['fits']),
    }
    traces_file = out/'traces.npz'
    np.savez_compressed(traces_file, **traces)
    source = Path(__file__).read_bytes()
    report = {'schema_version': 1, 'passed': all(checks.values()), 'checks': checks,
              'case_id': 'comsol-fabry-perot', 'source_url': SOURCE_URL, 'published_table': 1,
              'physical_model': {'slab_m': 1, 'slab_index': 4, 'exterior_index': 1,
                 'source_Ex_z_m': -.29, 'probe_Ex_z_m': .13, 'time_unit_seconds': 1/299792458.,
                 'one_dimensional_normal_incidence': True},
              'solver': 'Meep CPU', 'versions': {'meep': mp.__version__, 'numpy': np.__version__,
                  'python': sys.version.split()[0]},
              'reference_comparison': metrics, 'max_relative_sensitivity': changes, 'cases': cases,
              'fit_method': {'type': 'truncated Hankel matrix pencil', 'search_band_MHz': [20,166],
                 'primary_window_after_source_off': list(FITS[0][:2]), 'primary_rank': FITS[0][2],
                 'hankel_rows': 128, 'minimum_relative_mode_amplitude': 1e-5},
              'sampling': {'interval_time_units': .125, 'Courant': .5, 'source_frequency': .295,
                 'source_frequency_width': .50, 'source_cutoff': 5, 'post_source_duration': 84},
              'trace_file': 'traces.npz', 'trace_sha256': hashlib.sha256(traces_file.read_bytes()).hexdigest(),
              'solver_source_sha256': hashlib.sha256(source).hexdigest(),
              'scope': 'Four passive longitudinal poles of a lossless dielectric slab at normal incidence. Q is the amplitude-decay pole Q. Electrical pumping, gain, lasing and finite-width photonic-crystal or ring cavities require separate models.'}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if export:
        (out/'cavity.py').write_bytes(source)
        write_notebook(source.decode('utf-8'), out/'cavity.ipynb')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='fresh directory for report, traces and replay files')
    parser.add_argument('--no-export', action='store_true', help='write only measured traces and report')
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        parser.error('output exists; choose a fresh directory')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staging = Path(tmp)/'completed'
        staging.mkdir()
        report = run(staging, not args.no_export)
        staging.rename(dest)
    print(dest/'report.json', 'pass' if report['passed'] else 'outside benchmark tolerances')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
