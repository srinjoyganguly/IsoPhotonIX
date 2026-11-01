"""Check completed real 3D HTTP runs against their exported numerical arrays.

This checks file/API identity and numerical sanity, not commercial agreement.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.volume import COMPONENTS, validate_volume
from backend.adapters.fdtdx3d_adapter import cell_center_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_ids', nargs='+')
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    def fetch(route):
        with urllib.request.urlopen(args.url.rstrip('/') + route, timeout=60) as response:
            return response.read()
    reports = []
    for run_id in args.run_ids:
        job = json.loads(fetch(f'/api/jobs/{run_id}'))
        assert job['status'] == 'completed' and job['config']['study_type'] == 'waveguide-3d'
        exported = fetch(f'/api/jobs/{run_id}/export')
        with zipfile.ZipFile(io.BytesIO(exported)) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            config = json.loads(archive.read('config.json'))
            metadata = json.loads(archive.read('metadata.json'))
            assert config == job['config'] == metadata['config']
            for name, spec in manifest['files'].items():
                payload = archive.read(name)
                assert hashlib.sha256(payload).hexdigest() == spec['sha256']
                assert len(payload) == spec['size_bytes']
            with np.load(io.BytesIO(archive.read('arrays.npz')), allow_pickle=False) as arrays:
                data = validate_volume(arrays)
                assert all(np.any(component) for component in data['fields'])
                if config['solver'] == 'fdtdx':
                    np.testing.assert_array_equal(data['fields'], cell_center_fields(arrays['raw_fields']))
                    np.testing.assert_array_equal(data['epsilon'], arrays['raw_epsilon'][:-1, :-1, :-1])
                checks = 0
                for plane, axis in [('xy', 'z'), ('xz', 'y'), ('yz', 'x')]:
                    for index in [0, len(data[f'{axis}_um'])//2, len(data[f'{axis}_um'])-1]:
                        selection = {'xy': (index, slice(None), slice(None)), 'xz': (slice(None), index, slice(None)), 'yz': (slice(None), slice(None), index)}[plane]
                        for ci, component in enumerate(COMPONENTS):
                            result = json.loads(fetch(f'/api/jobs/{run_id}/slice?plane={plane}&index={index}&component={component}'))
                            np.testing.assert_array_equal(result['field'], data['fields'][ci][selection])
                            np.testing.assert_array_equal(result['epsilon'], data['epsilon'][selection])
                            assert result['coordinate_um'] == data[f'{axis}_um'][index]
                            np.testing.assert_array_equal(result['u_um'], data[result['u_axis'] + '_um'])
                            np.testing.assert_array_equal(result['v_um'], data[result['v_axis'] + '_um'])
                            checks += 1
                record = dict(run_id=run_id, solver=config['solver'], requested_device=config['device'], actual_device=metadata['actual_device'],
                              shape=list(data['field'].shape), all_components_nonzero=True, finite=True,
                              exact_api_slices_checked=checks, manifest_verified=True, saved_config_immutable=True,
                              field_sampling=metadata.get('field_sampling'), export_sha256=hashlib.sha256(exported).hexdigest())
                reports.append(record)
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'passed': True, 'mock_numerical_data': False,
              'scope': 'Actual completed 3D runs: API/export sample identity, provenance hashes and finite fields. Not an external accuracy benchmark.', 'runs': reports}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
