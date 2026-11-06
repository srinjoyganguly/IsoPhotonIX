"""Execute the two released v0.12.0a1 grating notebooks and audit their fresh native fields."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from backend.calibration_replay import compare
from scripts.verify_calibration_replays import replay
from scripts.verify_finite_grating import verify

EVIDENCE = Path(__file__).resolve().parents[1] / 'docs/verification'
RELEASE = 'https://github.com/srinjoyganguly/IsoPhotonIX/releases/download/v0.12.0a1/'


def read_json(path):
    return json.loads(path.read_text())


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_export(archive, expected):
    require(archive.stat().st_size == expected['size_bytes'] and
            file_sha(archive) == expected['sha256'], f'Unexpected release file: {archive.name}')


def check_executed_notebook(folder):
    original = read_json(folder/'calibration.ipynb')['cells']
    executed = read_json(folder/'executed.ipynb')['cells']
    require(len(original) == len(executed), 'Executed notebook has a different cell count.')
    for before, after in zip(original, executed):
        require(before['cell_type'] == after['cell_type'] and ''.join(before['source']) == ''.join(after['source']),
                'The notebook cells changed during execution.')
        if after['cell_type'] == 'code':
            require(after['execution_count'] is not None and
                    not any(o['output_type'] == 'error' for o in after['outputs']),
                    'A code cell did not execute successfully.')


def publish_notebook(folder, path):
    # The notebook prints the absolute result directory once. Preserve all cells,
    # execution counts, numerical outputs, kernel messages and their ordering.
    original = (folder/'executed.ipynb').read_bytes()
    local_result = str((folder/'notebook-result').resolve()).encode()
    require(original.count(local_result) == 1, 'Unexpected number of local output paths.')
    public = original.replace(local_result, b'notebook-result')
    require(not any(prefix in public for prefix in (b'/home/', b'/mnt/', b'/tmp/')),
            'The executed notebook still contains a machine-specific path.')
    path.write_bytes(public)
    return {'file':path.name, 'sha256':file_sha(path),
            'change':'The printed absolute result directory was shortened to notebook-result.'}


def audit_case(work, archive, expected):
    check_export(archive, expected)
    folder = work/expected['geometry']/'notebook'
    fresh = folder/'notebook-result'
    # Check the complete archived run, all frozen helpers and the original notebook.
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            require((folder/name).read_bytes() == bundle.read(name), f'Extracted file changed: {name}')
    check_executed_notebook(folder)
    reference = read_json(folder/'reference/report.json')
    require(reference['arrays_sha256'] == expected['native_arrays_sha256'] ==
            file_sha(folder/'reference/arrays.npz'), 'Original numerical archive changed.')
    report = verify(fresh)
    require(report['passed'] and report['actual_device'].startswith('cuda:') and
            report['settings']['geometry'] == expected['geometry'] and report['settings']['resolution'] == 14,
            'The rerun did not produce a qualified 14-cell CUDA calculation for this geometry.')
    measured = compare(folder/'reference', fresh)
    require(measured['passed'] and measured == read_json(fresh/'reproduction.json'),
            'The frozen controller and the independent replay comparison disagree.')
    stages = {}
    for role in ('reference', 'device'):
        new, orig = report['stages'][role], reference['stages'][role]
        require(new['settled'] and new['actual_device'] == report['actual_device'] and
                new['device_name'] == report['device_name'] and new['versions'] == orig['versions'] and
                new['identity']['sources'] == orig['identity']['sources'] and
                new['source_eh_sha256'] == orig['source_eh_sha256'] and
                new['source_time_offsets_sha256'] == orig['source_time_offsets_sha256'] and
                new['scalar_permittivity_sha256'] == orig['scalar_permittivity_sha256'] and
                new['time_steps'] == orig['time_steps'], 'Device, source, grid or time stepping changed.')
        require(new['cuda_identity']['per_cache_device_sha256'] != orig['cuda_identity']['per_cache_device_sha256']
                and new['signature'] != orig['signature'], 'The notebook did not use fresh native stages.')
        stages[role] = {key:new[key] for key in ('shape','time_steps','settled','source_eh_sha256',
                        'source_time_offsets_sha256','scalar_permittivity_sha256','native_worker_sha256',
                        'compile_seconds','stepping_seconds','versions')}
    with np.load(folder/'reference/arrays.npz', allow_pickle=False) as old, np.load(fresh/'arrays.npz', allow_pickle=False) as new:
        preview = report['field_preview']
        view = tuple(preview['index'])
        sample = new[preview['array']][view]
        field_delta = float(np.max(abs(sample - old[preview['array']][view])))
        count = len(new.files)
        # With the original public arrays, these fresh arrays recover every replay value.
        changed = {k:new[k] for k in new.files if old[k].tobytes() != new[k].tobytes()}
        peak_relative = {}
        for key, values in changed.items():
            peak = float(np.max(abs(old[key])))
            peak_relative[key] = float(np.max(abs(values-old[key])))/peak if peak else None
    spectrum = report['result']
    comparison = next(c for c in report['comparisons'] if c['geometry_matched'])
    wavelengths = spectrum['wavelength_um']
    require(wavelengths == comparison['wavelength_um'] and all(spectrum['valid_beam_bins']),
            'A vendor wavelength or qualification flag differs.')
    published = comparison.get('published_power', comparison.get('published_s21_power'))
    local = np.asarray(spectrum['beam_to_guide'])
    residual = 100*(local - published)
    np.testing.assert_allclose(residual, comparison['local_minus_published_percentage_points'])
    np.testing.assert_allclose(np.abs(residual).mean(), comparison['mean_absolute_difference_percentage_points'])

    return {
        'passed':True, 'asset':archive.name, 'release_url':RELEASE+archive.name,
        'export_sha256':file_sha(archive),'original_notebook_sha256':file_sha(folder/'calibration.ipynb'),
        'executed_notebook_sha256':file_sha(folder/'executed.ipynb'), 'native_receipt':expected['native_receipt'],
        'reference_arrays_sha256':reference['arrays_sha256'],'replay_arrays_sha256':report['arrays_sha256'],
        'replay_report_sha256':file_sha(fresh/'report.json'),'actual_device':report['actual_device'],
        'device_name':report['device_name'],'stages':stages, 'independent_stage_signatures':True,
        'numerical_array_count':count,'native_reconstruction':report['verification'],
        'comparison':measured,'changed_arrays_max_difference_over_reference_peak':peak_relative,
        'preview_field_max_absolute_difference':field_delta,
        'coupling_fraction_max_absolute_difference':float(np.max(abs(local-reference['result']['beam_to_guide']))),
        'wavelength_nm':(np.asarray(wavelengths)*1000).tolist(),'coupling_fraction':local.tolist(),
        'published_fraction':published,'vendor_source':comparison['url'],
        'local_minus_published_percentage_points':residual.tolist(),
        'five_wavelength_mean_absolute_difference_percentage_points':float(np.abs(residual).mean())
    }, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True, help='New directory for two independent notebook executions')
    parser.add_argument('--report', type=Path, required=True, help='Report JSON; changed arrays are saved beside it')
    parser.add_argument('--archives', type=Path, help='Local copies of the named public ZIPs; default downloads anonymously')
    parser.add_argument('--verify-existing', action='store_true', help='Audit prior executions from this work directory')
    args = parser.parse_args()
    work = args.work.resolve()
    manifests = read_json(EVIDENCE/'v012-vendor-grating-exports.json')['assets']
    release = read_json(EVIDENCE/'v012-release.json')
    for name, expected in manifests.items():
        require(release['verified_assets'][name] == expected['sha256'], 'The release and export hashes differ.')
    if not args.verify_existing:
        work.mkdir(parents=True, exist_ok=False)
        for name, expected in manifests.items():
            archive = work/name
            if args.archives:
                shutil.copyfile(args.archives/name, archive)
            else:
                with urlopen(Request(RELEASE+name, headers={'User-Agent':'PhotonicsWorkbench-notebook-replay'}), timeout=180) as source:
                    with archive.open('xb') as destination:
                        shutil.copyfileobj(source, destination)
            check_export(archive, expected)
            case_dir = work/expected['geometry']
            case_dir.mkdir()
            replay('notebook', archive, case_dir)
    cases, differences = {}, {}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for name, expected in manifests.items():
        case = expected['geometry']
        result, changes = audit_case(work, work/name, expected)
        copy = args.report.with_name(args.report.stem+'-'+case+'.ipynb')
        result['published_executed_notebook'] = publish_notebook(work/case/'notebook', copy)
        cases[case] = result
        differences.update({case+'__'+key:array for key,array in changes.items()})
    result = {'passed':True,'verified_at':datetime.now(timezone.utc).isoformat(),
              'target':'FDTDX running locally on NVIDIA','program':'Jupyter, unchanged v0.12.0a1 release notebooks',
              'release_source_commit':release['public_tag_commit'],
              'release_checksums_sha256':release['sha256_list_sha256'],
              'scope':'Each notebook performed separate full reference and device simulations without a shared checkpoint.',
              'cases':cases}
    if differences:
        patch = args.report.with_name(args.report.stem+'-changed-arrays.npz')
        np.savez_compressed(patch, **differences)
        result['fresh_replacement_arrays'] = {'file':patch.name, 'sha256':file_sha(patch),'keys':sorted(differences),
            'meaning':'Keys use geometry__array. Each contains the complete fresh array where any byte differed from the original native array.'}
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Verified both GPU notebooks:',args.report)


if __name__ == '__main__':
    main()
