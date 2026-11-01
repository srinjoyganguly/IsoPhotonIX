"""Build standalone, editable Python and notebook exports from exact studies.

Only first-party numerical source and explicitly named result artifacts enter
the archive. No local runtime configuration, logs, environments or credentials.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pprint
import re
import subprocess
import zipfile

from .reproduction_runtime import validate_geometry
from .source_snapshot import FORMAT as SOURCE_FORMAT, FILENAME as SOURCE_FILENAME, numerical_source_files, read_implementation

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(__file__).with_name('reproduction_runtime.py')
CONTROLLER_REQUIREMENTS = 'numpy==2.4.6\npydantic==2.13.4\n'
NOTEBOOK_REQUIREMENTS = '-r requirements-controller.txt\nmatplotlib==3.10.8\nnbformat==5.10.4\nnbclient==0.10.2\nipykernel==7.1.0\njupyterlab==4.4.10\n'


def _json(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def _source_snapshot():
    """Use the same numerical closure recorded by an isolated solver worker."""
    return numerical_source_files(ROOT)


def _cell(kind, source, identifier):
    value = {'cell_type': kind, 'id': identifier, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code': value.update(execution_count=None, outputs=[])
    return value


def _notebook(cells):
    return _json({'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    }, 'cells': cells})


def _configuration_code(config, variable='CONFIG'):
    return f'{variable} = {pprint.pformat(config, width=100, sort_dicts=False)}\n'


def _base_files():
    return {'workbench_runtime.py': RUNTIME.read_bytes(),
            'requirements-controller.txt': CONTROLLER_REQUIREMENTS.encode(),
            'requirements-notebook.txt': NOTEBOOK_REQUIREMENTS.encode(),
            'LICENSE': (ROOT / 'LICENSE').read_bytes(), 'CREDITS.md': (ROOT / 'CREDITS.md').read_bytes()}


def _archive(files, kind, config, *, versions=None, snapshot_origin='not_applicable', implementation=None):
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {'schema_version': 1, 'kind': kind, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'base_revision': revision, 'revision_meaning': 'Checkout revision at export time; file hashes identify the actual numerical implementation.',
                'snapshot_origin': snapshot_origin, 'implementation_snapshot': implementation,
                'source_identity': 'Run-time source comes from the verified worker archive when available; older results use an explicitly labeled export-time compatibility snapshot.',
                'study_schema_version': config['schema_version'], 'recorded_versions': versions or {},
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)} for name, data in files.items()}}
    files['bundle-manifest.json'] = _json(manifest)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return stream.getvalue()


def study_bundle(config, output=None):
    """Export an editable study, optionally with its saved numerical reference."""
    from .study3d import parse_study
    config = parse_study(config).model_dump(by_alias=True)
    files = _base_files()
    files['config.json'] = _json(config)
    versions = {}
    implementation = None
    snapshot_origin = 'export_time'
    numerical_source = None
    if output is not None:
        output = Path(output)
        original = json.loads((output / 'config.json').read_text())
        metadata = json.loads((output / 'metadata.json').read_text())
        if original != config or metadata.get('config') != config:
            raise ValueError('Saved result configuration does not match the study export.')
        versions = metadata.get('versions', {})
        implementation = metadata.get('implementation_snapshot')
        if implementation is not None:
            if not isinstance(implementation, dict) or implementation.get('format') != SOURCE_FORMAT or implementation.get('filename') != SOURCE_FILENAME:
                raise ValueError('Unknown recorded numerical implementation format.')
            numerical_source = read_implementation(output / SOURCE_FILENAME, implementation.get('sha256'))
            if implementation.get('file_count') != len(numerical_source):
                raise ValueError('Recorded numerical source file count differs from its archive.')
            files['reference/' + SOURCE_FILENAME] = (output / SOURCE_FILENAME).read_bytes()
            snapshot_origin = 'run_time'
        else:
            snapshot_origin = 'export_time_compatibility'
        files['reference/config.json'] = _json(original)
        # Metadata contains physics provenance, never copy worker logs or runtime settings.
        files['reference/metadata.json'] = _json(metadata)
        arrays = sorted(output.glob('*.npz'))
        if not any(path.name == 'arrays.npz' for path in arrays):
            raise ValueError('The saved study has no numerical arrays to reproduce.')
        for path in arrays:
            if not re.fullmatch(r'[a-z][a-z0-9_-]*\.npz', path.name):
                raise ValueError('Unexpected numerical artifact filename.')
            files['reference/' + path.name] = path.read_bytes()
        for name in ('spectrum.csv', 'sparameters.csv'):
            if (output / name).is_file(): files['reference/' + name] = (output / name).read_bytes()
    files.update(numerical_source if numerical_source is not None else _source_snapshot())
    for name in ('environment-meep.yml', 'environment-meep-linux-64.lock', 'requirements-fdtdx.txt', 'requirements-fdtdx-cuda.txt'):
        if (ROOT / name).is_file(): files[name] = (ROOT / name).read_bytes()
    pip_names = {'fdtdx', 'jax', 'jaxlib', 'numpy', 'tidy3d', 'equinox'}
    observed = ['# Install in the selected solver environment after its base requirements.',
                '# Only package versions actually recorded by the GUI worker appear here.']
    for name, version in sorted(versions.items()):
        if name in pip_names and re.fullmatch(r'[A-Za-z0-9.+_-]+', str(version)):
            observed.append(f'{name}=={version}')
        else:
            observed.append(f'# Native/runtime version: {name} {version}')
    files['requirements-observed.txt'] = ('\n'.join(observed) + '\n').encode()
    config_code = _configuration_code(config)
    files['study.py'] = ('"""Editable local study. Lengths are µm; times fs. See README.md for setup.\n'
                         f'Numerical source provenance: {snapshot_origin}; see bundle-manifest.json.\n'
                         'A saved reference is compared after execution; changed settings are a new experiment.\n"""\n'
                         'from workbench_runtime import main\n\n' + config_code + '\nif __name__ == "__main__":\n    main(CONFIG)\n').encode()
    files['study.ipynb'] = _notebook([
        _cell('markdown', '# Reproducible local photonics study\n\nThe exact physical configuration and numerical implementation are included in this bundle. '
              f'Source provenance: **{snapshot_origin}**. Run-time archives identify original solver bytes; export-time compatibility snapshots cannot make that historical claim. '
              'The notebook uses no running workbench server, cloud solver, or OpenAI API. Install the recorded environments using `README.md` before Run All. '
              'The source under `source/backend/` is readable and retained with its license.\n\n'
              'Changing the settings below creates a new experiment. A saved GUI result is compared only when its configuration is unchanged. '
              'Matching a replay is a software reproducibility check, not a commercial-accuracy or convergence certificate. '
              'The comparison separately reports physical_results_passed, all_arrays_pointwise_passed and diagnostic_variations. '
              'Only unused Meep incoming modal residual channels may use the documented amplitude-scale criterion; physical fields, S and raw E/H stay strict.', 'introduction'),
        _cell('code', 'from pathlib import Path\nfrom workbench_runtime import run, summarize, compare_results, plot_results\nimport matplotlib.pyplot as plt\n', 'imports'),
        _cell('markdown', '## Physical settings and selected hardware\n\nAll dimensions use µm, wavelengths µm and times fs. '
              'Geometry objects, source/boundary settings, monitor conventions and stopping tolerances are preserved. '
              'The solver/device selection remains explicit; GPU requests cannot retry on CPU.', 'settings-help'),
        _cell('code', config_code, 'configuration'),
        _cell('code', '# Use None when the current kernel contains the solver or MEEP_PYTHON/FDTDX_PYTHON is set.\n'
              '# Otherwise set this to your local solver environment Python executable. No machine-specific path is embedded.\n'
              'SOLVER_PYTHON = None\nOUTPUT = run(CONFIG, solver_python=SOLVER_PYTHON)\n', 'run-real-solver'),
        _cell('code', 'METADATA = summarize(OUTPUT)\nMETADATA\n', 'actual-provenance'),
        _cell('markdown', '## Inspect actual saved samples\n\nThe field is the stored final-time snapshot; a 3D study shows its central XY plane. '
              'The material panel shows the actual sampled permittivity. Port plots use saved masks, normalization and complex amplitudes. '
              'Near-zero S-parameter phases (`|S| < 1e-8`) are undefined and masked. Read the exported physical reference planes, '
              'mode gauge and phase convention before interpreting phase. Uncollected power is not absorption.', 'inspection-help'),
        _cell('code', 'plot_results(OUTPUT)\nplt.show()\n', 'plot-saved-data'),
        _cell('code', 'REPRODUCTION = compare_results(OUTPUT)\nREPRODUCTION\n', 'compare-gui-reference'),
    ])
    files['README.md'] = _study_readme(config, output is not None, snapshot_origin).encode()
    return _archive(files, 'executable-study', config, versions=versions,
                    snapshot_origin=snapshot_origin, implementation=implementation)


def _study_readme(config, has_reference, snapshot_origin):
    return f'''# Standalone Photonics Workbench study

This bundle contains editable `study.py` and `study.ipynb`, the exact JSON study,
and a snapshot of the first-party numerical implementation.
It needs no running workbench, repository checkout, network solver or API key.
Selected execution: **{config['solver']} / {config['device']}**. No GPU-to-CPU retry.
Source provenance: **{snapshot_origin}**. `run_time` uses the verified source
archive captured before the original solve; its checksum must match the replay.
`export_time_compatibility` means an older result did not capture its original
source. Its bundle uses current code, and a successful replay establishes
compatibility only, without claiming those bytes produced the original result.
`export_time` is an unrun study using the implementation present when exported.

## Setup once on Linux / WSL2

Use Python 3.11 for the controller and install its two lightweight dependencies:

```bash
python -m pip install -r requirements-controller.txt
```

Install the selected numerical engine separately. For Meep use the provided
official conda-forge `environment-meep.yml` (the PyPI package named `meep` is
unrelated). For FDTDX use `requirements-fdtdx.txt` for CPU or
`requirements-fdtdx-cuda.txt` for NVIDIA, with a compatible host driver. The
source snapshot and `bundle-manifest.json` record the exact implementation and
versions observed in the saved GUI run. Setup on arbitrary new machines is not
guaranteed: missing dependencies or unavailable GPU hardware produce an error.
No dependency installation or network request happens during a solve.
For saved runs, apply `requirements-observed.txt` in the selected solver
environment to pin recorded Python dependencies such as the local Tidy3D mode
solver. Meep/MPB are native Conda packages, not pip substitutions; the included
Linux lockfile records the tested native environment.

## Run the Python study

```bash
python study.py --solver-python /path/to/selected/solver/python
```

That path is a placeholder: use your installed solver interpreter. If the
current Python already contains it, omit `--solver-python`. Alternatively set
`MEEP_PYTHON` or `FDTDX_PYTHON`. These variables choose an interpreter, not a
different solver or device. Edit the visible `CONFIG` dictionary in `study.py`
to create a new experiment. Each run writes a new output directory. Optional
`--output NEW_DIRECTORY` never overwrites an earlier run. `--timeout` controls
only the wall-clock limit (default 1800 s); physical time is in CONFIG.

## Run the Jupyter notebook

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab study.ipynb
```

Use a kernel with these controller/notebook requirements. Set `SOLVER_PYTHON`
in the notebook when the solver has its own interpreter, then Run All. The
notebook has its own editable CONFIG initialized to the same exact study as
the script. It runs the same worker, displays actual fields/materials and
port curves, and compares saved numerical arrays. Restarting or rerunning it
creates a new output directory.

## Reproduction contract

Saved GUI numerical reference included: **{has_reference}**.
For unchanged settings, every original NPZ array is compared. Masks, integer
data, coordinate/time axes, array shapes and dtypes must match exactly. Other
real/complex arrays use the predeclared `rtol=1e-6`, `atol=1e-12`; the report
also records bitwise numerical equality, maximum absolute error and relative
L2 error. Undefined-value masks remain explicit. Wall-clock timings, human
labels and compressed NPZ bytes are excluded. A failed comparison fails loudly;
no fitted scale, phase adjustment, clipping or manufactured result is used.
One explicit diagnostic exception is recorded separately: Meep's device-run
incoming `port1_alpha[:,0]` / `port2_alpha[:,1]` and their raw-alpha counterparts
do not contribute to S, outgoing amplitudes or reference normalization. Their
native iterative mode projections can vary slightly even with fixed seeds.
The original pointwise pass/fail is preserved. These unused residuals may also
pass a bound using the same rtol/atol against the original full two-direction
coefficient array's peak and L2 norm. Outgoing coefficients, fields, S and raw
E/H never use this exception. Reports separately expose
`physical_results_passed`, `all_arrays_pointwise_passed` and
`diagnostic_variations`; `passed` requires every physical check and any declared
diagnostic bound. Recorded solver/runtime versions must match.
Changing CONFIG disables the original-result equality claim and reports a new
experiment. Different library versions/hardware can change numerical results;
actual device and version differences are recorded. All runtime checks and
scientific limitations from the original study still apply.

`bundle-manifest.json` hashes the source and original reference files. Hashes
identify bytes; they are not signatures or an accuracy certificate. Editing
the solver source or reference invalidates the original reproduction bundle.
Keep an original copy before developing a different implementation. The entry
script/notebook CONFIG remains editable without changing solver source.

The export includes only first-party source. Solver binaries, local runtime
paths, environments, worker logs, private handoffs and credentials are excluded.
Application source is MIT; dependency notices are in LICENSE and CREDITS.md.
Meep/MPB, FDTDX, JAX, the local Tidy3D mode dependency and NVIDIA runtimes retain
their respective licenses. No solver-library source or binaries are bundled.
'''


def geometry_bundle(geometry):
    """Export all supported editor solids as code without implying a solver."""
    geometry = validate_geometry(geometry)
    files = _base_files()
    files['geometry.json'] = _json(geometry)
    code = _configuration_code(geometry, 'GEOMETRY')
    files['geometry.py'] = ('"""Exact editable geometry draft; this is not an executable electromagnetic study."""\n'
                            'import json\nfrom pathlib import Path\nfrom workbench_runtime import check_bundle, validate_geometry, plot_geometry\n\n' + code +
                            '\nif __name__ == "__main__":\n    check_bundle()\n    geometry = validate_geometry(GEOMETRY)\n'
                            '    Path("geometry.recreated.json").write_text(json.dumps(geometry, indent=2) + "\\n")\n'
                            '    plot_geometry(geometry).savefig("geometry.png", dpi=160)\n'
                            '    print("Saved exact geometry.recreated.json and geometry.png; no simulation was run.")\n').encode()
    files['geometry.ipynb'] = _notebook([
        _cell('markdown', '# Exact local geometry draft\n\nThis notebook preserves every box, cylinder, ring, position, dimension, material index, '
              'rotation and visibility flag from the editor. Curved surfaces are tessellated only for display; JSON retains their exact radii. '
              'It is a design export, not a solver result. Unsupported CAD is never silently reduced to a straight guide. '
              'Choose a supported executable study in the workbench to obtain a solver notebook. Install `requirements-notebook.txt` before Run All.', 'introduction'),
        _cell('code', 'import json\nfrom pathlib import Path\nimport matplotlib.pyplot as plt\nfrom workbench_runtime import check_bundle, validate_geometry, plot_geometry\ncheck_bundle()\n', 'imports'),
        _cell('code', code, 'geometry'),
        _cell('code', 'geometry = validate_geometry(GEOMETRY)\nPath("geometry.recreated.json").write_text(json.dumps(geometry, indent=2) + "\\n")\ngeometry\n', 'validate-exact-design'),
        _cell('code', 'plot_geometry(geometry)\nplt.show()\n', 'inspect-solids'),
    ])
    files['README.md'] = b'''# Exact geometry Python and notebook export

Install Python 3.11 and `python -m pip install -r requirements-notebook.txt`.
Run `python geometry.py` to write the exact validated design JSON and an
original 3D plot, or `jupyter lab geometry.ipynb` and Run All. Edit GEOMETRY in
either file to develop a new design. All objects and hidden-state flags are
preserved; display visibility does not remove a material from a future study.

This exports design geometry only. It does not claim that rings, couplers,
rotated solids or every draft can be simulated. Unsupported geometry is not
replaced by a simpler physical model. The original geometric radii and sizes
remain exact in JSON; curved display surfaces use a 64-segment mesh.

No server, cloud account, solver license or API key is used. Source hashes,
application MIT license and dependency credits are included. No local runtime
paths, private files or credentials are embedded.
'''
    return _archive(files, 'geometry-draft', geometry)
