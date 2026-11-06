"""Audit all five grating study payloads for v0.12.0a2 without running a solver."""
import argparse
import ast
import hashlib
import io
import json
import re
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from scripts.reference_finite_grating import lumerical
from scripts.verify_finite_grating import verify

EVIDENCE = Path(__file__).resolve().parents[1] / 'docs/verification'
TAG = 'v0.12.0a2'
BASE_URL = f'https://github.com/srinjoyganguly/IsoPhotonIX/releases/download/{TAG}/'
COMMON = {
    'CREDITS.md', 'LICENSE', 'README.md', 'config.json', 'run.py', 'calibration.ipynb',
    'workbench_solver_environment.py', 'environment-fdtdx-cuda12-linux-64.lock',
    'requirements-fdtdx-cuda.txt', 'requirements-fdtdx.txt', 'source/backend/__init__.py',
    'source/docs/verification/v012-grating-commercial-references.json',
    'source/docs/verification/v012-grating-memory-budget.json',
}
for worker in ('calibrate_finite_grating', 'grating_angular_spectrum', 'grating_cuda_identity',
               'grating_emission', 'lumerical_grating3d'):
    COMMON.add(f'source/scripts/{worker}.py')
for adapter in ('fdtdx3d', 'fdtdx', 'fdtdx_ports', 'fdtdx_ports3d'):
    COMMON.add(f'source/backend/adapters/{adapter}_adapter.py')
DENSE = {'source/docs/verification/v013-grating-dense-reference.json',
         'source/docs/verification/v013-grating-dense-memory.json'}
REFERENCE = {'arrays.npz', 'report.json'}
SAMPLES = {'upper', 'upper_x_um', 'upper_y_um', 'wavelength_um', 'history_observable',
           'source_E', 'source_H', 'source_time_offset_E', 'source_time_offset_H', 'energy',
           'energy_time_fs', 'material_yz', 'material_xy', 'material_xz', 'grid_x_um',
           'grid_y_um', 'grid_z_um'}
ARRAYS = {f'{prefix}_{k}' for prefix in ('dev', 'ref') for k in SAMPLES}
ARRAYS |= {'dev_output', 'dev_mode_e', 'dev_mode_h', 'dev_mode_neff', 'dev_output_amplitudes',
           'dev_center_fields', 'incident_power', 'output_mode', 'coupling', 'valid', 'reflected_fiber_mode'}
UNSAFE_MARKERS = (b'/home/', b'/users/', b'/mnt/c/', b'\\users\\', b'/tmp/pw',
                  b'/tmp/photonics', b'.codex/', b'codex-clipboard-', b'.private.md',
                  b'submission-private', b'photonics-handoff', b'@gmail.com', b'@outlook.com')
CREDENTIALS = (
    (b'private key', rb'-----BEGIN [ A-Z]*PRIVATE KEY-----'),
    (b'gh', rb'(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{25,}'),
    (b'sk-', rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}'),
    (b'akia', rb'\bAKIA[0-9A-Z]{16}\b'), (b'asia', rb'\bASIA[0-9A-Z]{16}\b'),
    (b'xox', rb'\bxox[baprs]-[A-Za-z0-9-]{16,}'),
    (b'eyj', rb'\beyJ[\w-]{18,}\.eyJ[\w-]{18,}\.[\w-]{16,}'),
)
LITERAL_SECRET = re.compile(
    rb'''(?:password|passwd|access[_-]?token|api[_-]?key|client[_-]?secret)["']?\s*[:=]\s*["'][A-Za-z0-9._+/=~-]{12,}["']''', re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n').encode()


def scan(name, data, binary=False):
    lower = data.lower()
    markers = any(marker in lower for marker in UNSAFE_MARKERS)
    credentials = any(prefix in lower and re.search(pattern, data, re.I) for prefix, pattern in CREDENTIALS)
    if markers or credentials or (not binary and LITERAL_SECRET.search(data)):
        raise ValueError(f'Private path, contact or recognizable credential in {name}')


def check_members(z, expected):
    require(len(z.namelist()) == len(expected) and set(z.namelist()) == expected, 'Archive allowlist mismatch')
    require(not z.comment, 'Unexpected ZIP comment')
    for item in z.infolist():
        path = PurePosixPath(item.filename)
        require(not path.is_absolute() and '..' not in path.parts and str(path) == item.filename
                and not item.flag_bits & 1 and not stat.S_ISLNK(item.external_attr >> 16)
                and not item.comment and not item.extra, 'Unexpected ZIP member metadata')
    require(z.testzip() is None, 'ZIP CRC error')


def inspect_numbers(raw, companion, dense):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        check_members(z, {k + '.npy' for k in ARRAYS})
        for name in z.namelist():
            data = z.read(name)
            scan(name, data, binary=True)
            stream = io.BytesIO(data)
            array = np.load(stream, allow_pickle=False)
            require(array.dtype.kind in 'biufc' and np.isfinite(array).all() and stream.read() == b'',
                    f'Invalid numerical array {name}')
    if dense:
        with np.load(io.BytesIO(raw), allow_pickle=False) as full, np.load(companion, allow_pickle=False) as display:
            slices = {'ref_upper', 'dev_upper'}
            require(set(display.files) == (ARRAYS - slices) | {k + '_1550nm' for k in slices},
                    'Unexpected display array list')
            center = full['ref_wavelength_um'].tolist().index(1.55)
            for key in display.files:
                expected = full[key[:-7]][center] if key.endswith('_1550nm') else full[key]
                actual = display[key]
                require(actual.shape == expected.shape and actual.dtype == expected.dtype
                        and actual.tobytes() == expected.tobytes(), 'Display sample mismatch: ' + key)
    else:
        require(file_sha(companion) == sha(raw), 'Companion must contain complete original arrays')


def inspect_program(z, frozen, dense):
    expected = COMMON | (DENSE if dense else set())
    check_members(z, expected | {'manifest.json', 'reference-sha256.json'} | {'reference/' + k for k in REFERENCE})
    manifest = json.loads(z.read('manifest.json'))
    refs = json.loads(z.read('reference-sha256.json'))
    require(set(manifest['files']) == expected and set(refs) == REFERENCE, 'Incomplete internal manifests')
    hashes = {}
    for name in sorted(expected | {'manifest.json', 'reference-sha256.json', 'reference/report.json'}):
        raw = z.read(name)
        hashes[name] = sha(raw)
        scan(name, raw)
        raw.decode('utf8')
        if name.endswith(('.json', '.ipynb')):
            scan(name, encoded(json.loads(raw)))
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
        if name.startswith('source/'):
            require(hashes[name] == file_sha(frozen / name.removeprefix('source/')), 'Frozen source mismatch: ' + name)
        if name.startswith('source/docs/verification/'):
            require(hashes[name] == file_sha(EVIDENCE / Path(name).name), 'Public reference mismatch: ' + name)
    require(all(hashes[name] == h for name, h in manifest['files'].items()), 'Source manifest mismatch')
    book = json.loads(z.read('calibration.ipynb'))
    cells = [cell for cell in book['cells'] if cell['cell_type'] == 'code']
    require(book['nbformat'] == 4 and cells, 'Missing Jupyter program')
    code = '\n'.join(''.join(cell['source']) for cell in cells)
    require('from run import run' in code and 'run(Path.cwd(), output)' in code and
            all(not c.get('outputs') for c in cells), 'Unexpected notebook entry point')
    for cell in cells:
        ast.parse(''.join(cell['source']))
    return manifest, refs, hashes


def verify_stage(stage, role, hashes, settings):
    identity = stage['identity']
    native_metadata = {k:v for k,v in stage.items() if k not in {'metadata_sha256', 'identity', 'signature', 'arrays_sha256'}}
    for obj, digest in [(identity, stage['signature']), (native_metadata, stage['metadata_sha256'])]:
        require(sha(json.dumps(obj, sort_keys=True, allow_nan=False).encode()) == digest, 'Stage identity mismatch')
    require(identity['settings'] == settings and identity['stage'] == role and stage['settled'] and
            stage['actual_device'] == 'cuda:0', 'Unexpected native acquisition')
    expected = {PurePosixPath(k).name:v for k,v in hashes.items()
                if k.startswith('source/') and k.endswith('.py') and not k.endswith('__init__.py')}
    require(identity['sources'] == expected and stage['native_worker_sha256'] == expected['calibrate_finite_grating.py'],
            'Stage sources differ from exported workers')


def verify_case(args, model, mesh):
    dense = mesh == 12
    label = f'dense-{model}' if dense else 'direct'
    receipt = f'v013-grating3d-lumerical-{label}-r{mesh}.json'
    report = json.loads((EVIDENCE / receipt).read_text())
    name = f'grating-lumerical-{label}-3d-gpu-r{mesh}-{TAG}'
    archive = (args.dense_dir if dense else args.direct_dir) / (name + '.zip')
    frozen = args.dense_source if dense else args.direct_source
    assets = [archive]
    with zipfile.ZipFile(archive) as z:
        manifest, refs, hashes = inspect_program(z, frozen, dense)
        config = json.loads(z.read('config.json'))
        native = json.loads(z.read('reference/report.json'))
        raw = z.read('reference/arrays.npz')
        require(refs['arrays.npz'] == sha(raw) == native['arrays_sha256'] and
                refs['report.json'] == hashes['reference/report.json'], 'Measured array or report mismatch')
        for k,v in native.items():
            if k != 'arrays_sha256':
                require(report[k] == v, 'Native and public report differ: ' + k)
        require(config['case'] == 'grating-3d' and config['geometry_3d'] == 'lumerical' and
                config['resolution'] == mesh and config.get('grating_material_model', 'volume-average') == model and
                config.get('grating_spectrum') == ('dense' if dense else None) and
                (config['solver'], config['device']) == ('fdtdx', 'gpu'), 'Unexpected requested configuration')
        arguments = ['--device', 'gpu', '--geometry', 'lumerical', '--resolution', str(mesh)]
        if model == 'direct-yee':
            arguments += ['--grating-material-model', 'direct-yee']
        if dense:
            arguments += ['--grating-spectrum', 'dense']
        require(manifest['worker'] == 'scripts.calibrate_finite_grating' and manifest['arguments'] == arguments,
                'Unexpected worker or command')
        require(report['passed'] and report['actual_device'] == 'cuda:0' and
                set(native['stages']) == {'reference', 'device'}, 'Unexpected measured device or stage')
        for role, stage in native['stages'].items():
            verify_stage(stage, role, hashes, native['settings'])
        require(native['settings']['geometry'] == 'lumerical' and native['settings']['resolution'] == mesh and
                native['settings'].get('grating_material_model', 'volume-average') == model and
                native['settings'].get('grating_spectrum') == ('dense' if dense else None), 'Wrong measured settings')
        wave = np.array(report['result']['wavelength_um'])
        expected_wave = np.arange(1500, 1601, 5 if dense else 25) / 1000.0
        np.testing.assert_array_equal(wave, expected_wave)
        require(report['result']['valid_beam_bins'] == [True] * len(expected_wave), 'Unqualified wavelength bins')
        commercial = lumerical(args.vendor_archive, expected_wave)
        row = next(r for r in report['comparisons'] if r['vendor'] == 'Lumerical')
        captured = json.loads(z.read('source/docs/verification/v012-grating-commercial-references.json'))['lumerical']
        require(row['vendor_file_sha256'] == commercial['s_file_sha256'] == captured['s_file_sha256'] and
                commercial['archive_sha256'] == captured['archive_sha256'] and
                commercial['project_sha256'] == captured['project_sha256'], 'Published comparison input changed')
        np.testing.assert_array_equal(row['published_s21_power'], commercial['samples']['S21']['power'])
        if dense:
            dense_ref = json.loads(z.read('source/docs/verification/v013-grating-dense-reference.json'))
            require(dense_ref['s_file_sha256'] == commercial['s_file_sha256'] and
                    dense_ref['published_s21_power'] == row['published_s21_power'], 'Dense comparison changed')
        companion = EVIDENCE / (receipt[:-5] + '-arrays.npz')
        require(report['arrays_file'] == companion.name and file_sha(companion) == report['arrays_sha256'],
                'Public companion hash mismatch')
        inspect_numbers(raw, companion, dense)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'arrays.npz').write_bytes(raw)
            (directory / 'report.json').write_bytes(z.read('reference/report.json'))
            checked = verify(directory)
            require(checked['verification'] == report['verification'], 'Saved-field reconstruction disagrees')
    if dense:
        standalone = args.dense_dir / (name + '.npz')
        require(file_sha(standalone) == sha(raw) == report['native_arrays_sha256'] and
                report['native_report_sha256'] == refs['report.json'], 'Original dense result hash mismatch')
        for key, path in [('raw_arrays', standalone), ('python_notebook_and_results', archive)]:
            require(report['native_artifacts'][key] == {'asset':path.name, 'bytes':path.stat().st_size,
                    'sha256':file_sha(path), 'url':BASE_URL+path.name}, 'Public download metadata mismatch')
        assets.append(standalone)
    else:
        export = json.loads((EVIDENCE / 'v013-grating-direct-export.json').read_text())
        [entry] = export['examples']
        require(export['passed'] and entry['sha256'] == file_sha(archive) and entry['file'] == archive.name and
                entry['bytes'] == archive.stat().st_size and entry['native_receipt'] == receipt and
                entry['raw_result_sha256'] == refs['arrays.npz'] and entry['original_report_sha256'] == refs['report.json']
                and entry['exact_native_source_hashes'] ==
                {n.removeprefix('source/'):v for n,v in hashes.items() if n.startswith('source/')},
                'Direct-14 public export receipt mismatch')
    hashes['reference/arrays.npz'] = refs['arrays.npz']
    summary = {'native_receipt':receipt, 'receipt_sha256':file_sha(EVIDENCE / receipt),
               'source_manifest_sha256':hashes['manifest.json'], 'reference_sha256':refs,
               'all_zip_members_sha256':hashes, 'frozen_source_files':sum(k.startswith('source/') for k in hashes),
               'measured_wavelengths':len(wave), 'numeric_arrays':len(ARRAYS),
               'published_s21_file_sha256':commercial['s_file_sha256'],
               'measured_cuda_device':native['actual_device'], 'companion_sha256':report['arrays_sha256']}
    return assets, summary


def write_json(path, data, public=False):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(encoded(data))
        temporary = Path(stream.name)
    if public:
        temporary.chmod(0o644)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('direct-dir', 'dense-dir', 'direct-source', 'dense-source', 'vendor-archive'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--staging-list', type=Path, help='Private local file with full input paths.')
    parser.add_argument('--available', action='store_true', help='Inspect available cases without writing receipts.')
    args = parser.parse_args()
    assets, studies, private = {}, {}, []
    for model, mesh in [('direct-yee', 14), ('volume-average', 12), ('direct-yee', 12)]:
        receipt = EVIDENCE / f"v013-grating3d-lumerical-{'dense-'+model if mesh==12 else 'direct'}-r{mesh}.json"
        if args.available and not receipt.exists():
            print('Awaiting', receipt.name, flush=True)
            continue
        paths, record = verify_case(args, model, mesh)
        studies[paths[0].name] = record
        for path in paths:
            digest = file_sha(path)
            assets[path.name] = {'sha256':digest, 'bytes':path.stat().st_size}
            private.append({'input':str(path.resolve()), 'sha256':digest})
            print(path.name, digest, path.stat().st_size, flush=True)
    if args.available:
        return
    require(len(assets) == 5 and len(studies) == 3, 'Need all five original payloads')
    result = {'passed':True, 'release':TAG, 'assets':assets, 'studies':studies,
              'checks':['Exact program and array member allowlists and ZIP CRCs',
                        'Worker source files compared with frozen acquisition directories and native stage identities',
                        'Python sources and Jupyter code cells parsed; source and reference manifests rehashed',
                        'Recognizable machine paths, contact information and credential patterns scanned in text and decompressed arrays',
                        'Full native arrays decoded without pickle; complex fields, mode powers and qualification flags checked by the grating verifier',
                        'Dense display companions compared byte-for-byte per array with the original samples',
                        'Published Lumerical reference archive, project, and S21 file verified; S21 samples re-derived'],
              'scope':'Offline checks of the acquired data and frozen program. This audit launches no solver and makes no independent replay or commercial-agreement claim.'}
    scan('public receipt', encoded(result))
    if args.receipt:
        write_json(args.receipt, result, public=True)
    if args.staging_list:
        write_json(args.staging_list, private)


if __name__ == '__main__':
    main()
