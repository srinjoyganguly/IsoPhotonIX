"""Exercise a built wheel in a new offline Python 3.11 environment outside the checkout."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--wheelhouse', type=Path, required=True)
    parser.add_argument('--meep-python', type=Path, required=True)
    parser.add_argument('--fdtdx-gpu-python',type=Path,help='also run the installed calibrated splitter on this native CUDA worker')
    parser.add_argument('--output', type=Path, required=True, help='new directory outside the source checkout')
    args = parser.parse_args()
    target = args.output.resolve()
    if target == ROOT or ROOT in target.parents:
        parser.error('Choose a new directory outside the checkout.')
    target.mkdir(parents=True, exist_ok=False)
    wheel = args.wheel.resolve()
    meep = args.meep_python.absolute()
    state = target / 'state'
    env = dict(os.environ, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
               PHOTONICS_DATA_DIR=str(state), MEEP_PYTHON=str(meep), PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    env.pop('FDTDX_PYTHON', None)
    venv.EnvBuilder(with_pip=True).create(target / 'venv')
    python = target / 'venv/bin/python'
    cli = target / 'venv/bin/photonics-workbench'

    def command(argv, log, timeout=300, **kw):
        with (target / log).open('w') as stream:
            subprocess.run(argv, cwd=target, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout, **kw)

    command([python, '-m', 'pip', 'install', '--no-index', '--find-links', args.wheelhouse.resolve(), wheel], 'install.log')
    command([python, '-m', 'pip', 'check'], 'dependency-check.log')
    command([cli, 'doctor', '--json'], 'doctor.json')
    command([cli,'compare-examples','--case','lumerical-coupler','--resolutions','24','32',
             '--output',target/'component-comparison'],'benchmark-cli.log')
    comparison_output = json.loads((target/'component-comparison/report.json').read_text())
    assert comparison_output['cases'][0]['id'] == 'lumerical-coupler'
    assert len(comparison_output['cases'][0]['metrics']) == 4
    capabilities = json.loads((target / 'doctor.json').read_text())
    assert any(t['solver'] == 'meep' and t['device'] == 'cpu' and t['available'] for t in capabilities['targets'])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'

    def get(route, payload=None, raw=False):
        req = Request(base + route, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=65) as response:
            body = response.read()
        return body if raw else json.loads(body)

    @contextmanager
    def server(name):
        with (target / name).open('w') as log:
            proc = subprocess.Popen([cli, 'web', '--port', str(port)], cwd=target, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(100):
                    if proc.poll() is not None:
                        raise RuntimeError(f'Installed server exited. See {name}.')
                    try:
                        if get('/api/health').get('status') == 'ok':
                            break
                    except OSError:
                        time.sleep(.1)
                else:
                    raise RuntimeError('The installed service did not become ready.')
                yield
            finally:
                proc.terminate()
                proc.wait(timeout=30)

    with server('web.log'):
        html = get('/', raw=True).decode()
        assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html)
        assert any(a.endswith('.js') for a in assets) and any(a.endswith('.css') for a in assets)
        with zipfile.ZipFile(wheel) as archive:
            for asset in assets:
                assert get(asset, raw=True) == archive.read('photonics_workbench/_app/frontend/dist' + asset)
            wheel_names = set(archive.namelist())
            assert all(name.startswith('photonics_workbench/') or '.dist-info/' in name for name in wheel_names)
            assert not any(set(name.split('/')) & {'.local', 'runs', '.git', '.env', 'node_modules', '__pycache__'} for name in wheel_names)
        assert get('/api/benchmarks')['status'] == 'available'
        presets = get('/api/calibrations')
        assert len(presets)==11 and {p['case'] for p in presets}=={'coupler','splitter','ring','grating','multimode','multimode-3d','ring-3d','grating-3d','multimode-tm','layout-ports-2d','layout-ports-3d'}
        for preset in presets:
            solver, device = preset['routes'][0].split(':')
            config={'case':preset['case'],'solver':solver,'device':device}
            with zipfile.ZipFile(io.BytesIO(get('/api/reproduce/calibration',config,raw=True))) as source:
                assert {'config.json','calibration.ipynb','run.py'} <= set(source.namelist())
        command([cli,'calibrate','coupler','--solver','fdtdx','--device','gpu','--export-only',
                 '--output',target/'calibrated-gpu-program'],'calibrated-cli.log')
        with zipfile.ZipFile(target/'calibrated-gpu-program/python-notebook.zip') as archive:
            gpu_config=json.loads(archive.read('config.json'))
            assert gpu_config['device']=='gpu' and gpu_config['material_model']=='yee-aligned'
            assert gpu_config['revision']==2 and gpu_config['resolution']==16
        cavity = get('/api/cavity-benchmark')
        assert cavity['passed'] and len(cavity['reference_comparison']) == 4
        command([cli, 'benchmark-cavity', '--output', target/'cavity-study'], 'cavity.log')
        measured = json.loads((target/'cavity-study/report.json').read_text())
        assert measured['reference_comparison'] == cavity['reference_comparison']
        assert (target/'cavity-study/cavity.py').is_file() and (target/'cavity-study/cavity.ipynb').is_file()
        fixed = get('/api/library/build', {'component_id':'strip'})
        pair = get('/api/library/place', {'geometry':fixed['geometry'], 'component_id':'strip'})
        moved = get('/api/library/align', {'geometry':pair['geometry'], 'moving_component':pair['component']['id'],
            'moving_port':'in', 'fixed_component':fixed['component']['id'], 'fixed_port':'out', 'fit_domain':True})
        assert moved['geometry']['components'][1]['ports'][0]['x'] == fixed['component']['ports'][1]['x']
        assert len(get('/api/library')['components']) == 27
        comparisons = get('/api/component-benchmarks')
        assert len(comparisons['cases']) == 2 and len(comparisons['report_sha256']) == 64
        assert sum(len(case['metrics']) for case in comparisons['cases']) == 10
        port_study = get('/api/taper-ports/example')
        assert get('/api/taper-ports/validate', port_study)['layout']['width_in'] != get('/api/taper-ports/validate', port_study)['layout']['width_out']
        config = get('/api/layout/example?family=phc_line_defect')
        assert get('/api/layout/validate',config)['valid']
        job = get('/api/layout/jobs',config)
        jid = job['id']
        for _ in range(350):
            job = get(f'/api/jobs/{jid}')
            if job['status'] in ('failed', 'cancelled', 'completed'):
                break
            time.sleep(1)
        assert job['status'] == 'completed', job.get('error')
        assert job['result']['actual_device'] == 'CPU'
        volume = get(f'/api/jobs/{jid}/volume')
        sample = get(f'/api/jobs/{jid}/slice?plane=xy&component=Ey')
        assert max(abs(value) for row in sample['field'] for value in row) > 0
        result_store = state / 'runs' / jid
        assert (result_store / 'arrays.npz').is_file()
        with zipfile.ZipFile(io.BytesIO(get(f'/api/jobs/{jid}/export', raw=True))) as result:
            assert result.read('arrays.npz') == (result_store / 'arrays.npz').read_bytes()
        with zipfile.ZipFile(io.BytesIO(get(f'/api/jobs/{jid}/reproduce', raw=True))) as exported:
            assert json.loads(exported.read('bundle-manifest.json'))['base_revision'] is None
            assert 'study.ipynb' in exported.namelist()
            exported.extractall(target / 'export')
        # The same data directory cannot be claimed concurrently by two CLI servers.
        with socket.socket() as alternate:
            alternate.bind(('127.0.0.1', 0))
            alternate_port = alternate.getsockname()[1]
        second = subprocess.run([cli, 'web', '--port', str(alternate_port)], cwd=target, env=env,
                                capture_output=True, text=True, timeout=10)
        assert second.returncode and 'already using' in second.stderr
    with server('offline-restart.log'):
        assert get(f'/api/jobs/{jid}')['status'] == 'completed'
    command([cli, 'run', result_store / 'config.json', '--output', target / 'cli-run'], 'cli-run.log')
    # Mimic a registered external worker: standalone reproduction must find it from the same state as the app.
    registry = state / '.local/runtime.json'
    workers={'meep_python':str(meep)}
    if args.fdtdx_gpu_python:
        workers['fdtdx_gpu_python']=str(args.fdtdx_gpu_python.absolute())
    registry.write_text(json.dumps(workers))
    env.pop('MEEP_PYTHON')
    command([python, target / 'export/study.py',
             '--output', target / 'replayed'], 'replay.log')
    result = json.loads((target / 'replayed/reproduction-check.json').read_text())
    assert result['passed'] and result['all_arrays_exact']
    # Compare array contents, not NPZ container metadata, for CLI equivalence.
    command([python, '-c', 'import sys,numpy as n; a=n.load(sys.argv[1]); b=n.load(sys.argv[2]); '
             'assert set(a.files)==set(b.files) and all(n.array_equal(a[k],b[k]) for k in a.files)',
             result_store / 'arrays.npz', target / 'cli-run/arrays.npz'], 'cli-comparison.log')
    command([cli,'calibrate','grating','--solver','meep','--device','cpu','--grating-scope','tidy3d-compact',
             '--resolution','40','--output',target/'calibrated-grating'], 'calibrated-grating.log', timeout=7200)
    run_folder=target/'calibrated-grating'
    beam=json.loads((run_folder/'calculation/report.json').read_text())
    assert beam['passed'] and beam['vendor']=='tidy3d'
    observed=beam['result']['center_beam_to_guide']
    grating=next(p for p in presets if p['case']=='grating')
    reference=next(row['report']['compact_tidy_domain'] for row in grating['evidence']['recorded']
                   if 'compact_tidy_domain' in row['report'])
    idx=min(range(len(reference['wavelength_um'])),key=lambda i:abs(reference['wavelength_um'][i]-1.55))
    assert abs(observed-reference['beam_to_guide'][idx])<1e-6
    with zipfile.ZipFile(run_folder/'python-notebook-results.zip') as archive:
        assert {'run.py','calibration.ipynb','reference/arrays.npz','reference/report.json'}<=set(archive.namelist())
    command([cli,'run',run_folder/'config.json','--output',target/'calibrated-config-replay'],
            'calibrated-config-replay.log',timeout=7200)
    repeated=json.loads((target/'calibrated-config-replay/calculation/report.json').read_text())
    assert repeated['passed'] and abs(repeated['result']['center_beam_to_guide']-observed)<1e-6
    spatial_folder=target/'planar-layout-cli'
    command([cli,'calibrate','layout-ports-2d','--solver','meep','--device','cpu','--output',spatial_folder],
            'planar-layout-cli.log', timeout=7200)
    spatial=json.loads((spatial_folder/'calculation/report.json').read_text())
    saved_planar=next(p for p in presets if p['case']=='layout-ports-2d')['evidence']['recorded'][0]['report']
    assert spatial['passed'] and spatial['channels']==saved_planar['channels'] and spatial['wavelength_um']==saved_planar['wavelength_um']
    def matrix_difference(first,other):
        import numpy as np
        def matrix(record):
            return np.asarray(record['s_real'])+1j*np.asarray(record['s_imag'])
        return float(np.max(np.abs(matrix(first)-matrix(other))))
    layout_delta=matrix_difference(spatial,saved_planar)
    assert layout_delta<2e-6
    cuda_check=None
    spatial_cuda=None
    if args.fdtdx_gpu_python:
        folder=target/'calibrated-cuda-splitter'
        command([cli,'calibrate','splitter','--solver','fdtdx','--device','gpu','--resolution','12',
                 '--output',folder],'calibrated-cuda.log',timeout=7200)
        cuda=json.loads((folder/'calculation/report.json').read_text())
        cfg=json.loads((folder/'config.json').read_text())
        assert cfg['revision']==2 and cfg['material_model']=='yee-aligned' and cfg['device']=='gpu'
        assert cuda['settled'] and cuda['actual_device'].lower().startswith(('cuda','gpu'))
        splitter=next(p for p in presets if p['case']=='splitter')
        saved=next(row['report'] for row in splitter['evidence']['recorded'] if row['receipt']=='v011-fdtdx-yee-splitter-gpu12.json')
        assert saved['port_order']==cuda['port_order'] and saved['wavelength_um']==cuda['wavelength_um']
        def coefficients(report):
            return [complex(*z) for at_frequency in report['s'] for outputs in at_frequency for z in outputs]
        original,measured=coefficients(saved),coefficients(cuda)
        assert len(measured)==len(original)==27
        maximum=max(abs(a-b) for a,b in zip(original,measured))
        assert maximum<2e-5 and (folder/'python-notebook-results.zip').is_file()
        cuda_check={'case':'splitter','actual_device':cuda['actual_device'],'resolution':12,'material_model':cfg['material_model'],
                    'compared_complex_coefficients':len(measured),'maximum_complex_difference':maximum}
    if args.fdtdx_gpu_python:
        folder=target/'finite-grating-cuda'
        command([cli,'calibrate','grating-3d','--solver','fdtdx','--device','gpu','--output',folder],
                'finite-grating-cuda.log',timeout=10800)
        cuda=json.loads((folder/'calculation/report.json').read_text())
        assert cuda['passed'] and cuda['actual_device'].lower().startswith(('cuda','gpu'))
        grating=next(p for p in presets if p['case']=='grating-3d')
        archived=next(row['report'] for row in grating['evidence']['recorded']
                      if row['report']['settings']['geometry']=='compact' and row['report']['settings']['resolution']==12)
        native=cuda['result']['beam_to_guide']
        expected=archived['result']['beam_to_guide']
        assert len(native)==len(expected)==5
        maximum=max(abs(a-b) for a,b in zip(native,expected))
        assert maximum<2e-6 and (folder/'python-notebook-results.zip').is_file()
        spatial_cuda={'geometry':'compact','actual_device':cuda['actual_device'],'device_name':cuda['device_name'],
                      'wavelengths':5,'max_coupling_fraction_difference':maximum}
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'passed': True, 'wheel': wheel.name,
              'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(), 'wheel_bytes': wheel.stat().st_size,
              'installed_from_offline_wheelhouse': True, 'calibrated_program_exports':len(presets),'installed_cuda_study':cuda_check,'installed_spatial_cuda':spatial_cuda,'installed_layout_max_complex_difference':layout_delta,
              'installed_grating_coupling':observed,'recorded_same_mesh_coupling':reference['beam_to_guide'][idx],
              'calibrated_cli_result_bundle_verified':True,
              'generic_calibration_config_coupling':repeated['result']['center_beam_to_guide'], 'export_found_registered_interpreter': True,
              'cavity_four_modes_equal_to_packaged_reference': True, 'installed_port_alignment': True, 'pip_check': True, 'doctor_meep_ready': True,
              'served_frontend_assets': assets, 'catalog_families': 27, 'benchmark_data': True,
              'job_id': jid, 'actual_device': job['result']['actual_device'], 'field_shape_zyx': volume['shape'],
              'state_separate_from_package': True, 'exclusive_store': True, 'saved_run_after_restart': True,
              'cli_run_all_arrays_exact': True, 'standalone_python_all_arrays_exact': True,
              'solver_versions': job['result']['versions'], 'state': 'Fresh disposable user state and virtual environment.'}
    (target / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
