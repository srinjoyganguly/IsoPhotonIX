"""Execute actual design-export scripts/notebooks outside the source checkout.

Requires the pinned controller/notebook packages described in exported bundles.
This verifies geometric reproduction, not electromagnetic performance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def execute_notebook(folder, name='geometry.ipynb', timeout=120):
    import nbformat
    from nbclient import NotebookClient
    # Resolve the kernel's "python" inside the same explicitly chosen QA env.
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ['IPYTHONDIR'] = str(folder/'.ipython')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(folder/'.jupyter-runtime')
    os.environ['MPLCONFIGDIR'] = str(folder/'.matplotlib')
    os.environ.pop('PYTHONPATH', None)
    os.environ.pop('MPLBACKEND', None)
    notebook = nbformat.read(folder/name, as_version=4)
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=timeout, kernel_name='python3', resources={'metadata': {'path': str(folder)}}).execute()
    nbformat.write(notebook, folder/('executed-' + name))
    assert not any(output.output_type == 'error' for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs)
    assert any('image/png' in output.get('data', {}) for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs)


def verify_case(record):
    name, geometry, folder = record
    from backend.reproduce import geometry_bundle
    started = time.perf_counter()
    folder.mkdir()
    archive_bytes = geometry_bundle(geometry)
    (folder/'geometry-export.zip').write_bytes(archive_bytes)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        archive.extractall(folder)
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(folder/'.matplotlib'))
    env.pop('PYTHONPATH', None)
    with (folder/'python.log').open('w') as log:
        subprocess.run([sys.executable, 'geometry.py'], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    assert json.loads((folder/'geometry.recreated.json').read_text()) == geometry
    assert (folder/'geometry.png').stat().st_size > 1000
    (folder/'geometry.recreated.json').unlink()
    with (folder/'notebook.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--execute-notebook', str(folder)],
                       cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    assert json.loads((folder/'geometry.recreated.json').read_text()) == geometry
    result = {'name': name, 'passed': True, 'script_exact_geometry': True, 'notebook_exact_geometry': True,
              'script_png': True, 'notebook_png': True, 'objects': len(geometry['objects']),
              'polygon_holes': sum(len(o.get('holes', [])) for o in geometry['objects']),
              'materials': len(geometry.get('materials', [])), 'component_instances': len(geometry.get('components', [])),
              'export_sha256': hashlib.sha256(archive_bytes).hexdigest(),
              'geometry_sha256': hashlib.sha256((folder/'geometry.json').read_bytes()).hexdigest(),
              'wall_seconds': round(time.perf_counter()-started, 3)}
    print(json.dumps({'case': name, 'passed': True}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute-notebook', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute_notebook:
        execute_notebook(args.execute_notebook.resolve()); return
    if args.output is None: parser.error('--output must be a new directory outside the checkout')
    folder = args.output.resolve()
    if folder == ROOT or ROOT in folder.parents: parser.error('Choose an output directory outside the checkout.')
    folder.mkdir(parents=True, exist_ok=False)
    from backend.component_library import get_catalog
    from backend.library_layout import build_design, place_component
    cases = [(c['id'], build_design(c['id'])['geometry'], folder/c['id']) for c in get_catalog()['components']]
    table = {'id': 'custom-film', 'name': 'Tabulated demonstration', 'model': 'tabulated-nk',
             'reference_wavelength_um': 1.55, 'wavelength_um': [1.4, 1.55, 1.7], 'n': [2.2, 2.1, 2.],
             'k': [.01, .02, .03], 'source': 'Synthetic software fixture; not measured optical data.', 'notes': 'Geometry-export validation only.'}
    custom = build_design('nanobeam_cavity', placement={'x': 1., 'y': -1., 'z': .2, 'rotation_deg': 25.}, material_id=table['id'], material=table)['geometry']
    custom = place_component(custom, 'taper', placement={'x': -10., 'y': 3., 'z': 0., 'rotation_deg': -15.}, material_id=table['id'], fit_domain=True)['geometry']
    cases.append(('custom-material-layout', custom, folder/'custom-material-layout'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(verify_case, cases))
    report = {'schema_version': 1, 'checked_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'Actual standalone geometry Python and Jupyter executions; no electromagnetic solver or laser calculation.',
              'passed': all(c['passed'] for c in results), 'cases': results,
              'executions': len(results)*2, 'environment': {'python': sys.version.split()[0]},
              'limitations': ['Visualizations tessellate curves; saved physical contours and material records remain exact.',
                             'Design ports are geometry metadata, not verified eigenmode measurements.']}
    (folder/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'executions': report['executions']}), flush=True)


if __name__ == '__main__': main()
