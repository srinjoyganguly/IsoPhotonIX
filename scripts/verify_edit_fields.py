"""Compare the actual arrays from the three browser-joined-fields.cjs journeys."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprint(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def difference(reference, candidate):
    reference = reference.astype(np.float64)
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def compare_edits(runs, report_file):
    browser = json.loads(report_file.read_text())
    assert browser['passed'] and [r['label'] for r in browser['runs']] == ['joined', 'deleted', 'restored']
    result, saved = [], []
    for row in browser['runs']:
        folder = runs / row['id']
        config = json.loads((folder / 'config.json').read_text())
        meta = json.loads((folder / 'metadata.json').read_text())
        assert (config['solver'] + ':' + config['device']) == browser['target']
        assert meta['actual_device'] == row['actual_device']
        expected = 'cuda' if config['device'] == 'gpu' else 'cpu'
        assert expected in meta['actual_device'].lower()
        with np.load(folder / 'arrays.npz', allow_pickle=False) as archive:
            fields = archive['fields']
            material = archive['epsilon']
            assert np.isfinite(fields).all() and np.isfinite(material).all()
            assert np.linalg.norm(fields) > 0
            axes = [archive[a + '_um'] for a in 'xyz']
            saved.append((config, fields, material, axes))
        result.append({**row, 'shape_zyx': list(material.shape), 'fields_sha256': fingerprint(fields),
                       'native_epsilon_sha256': fingerprint(material)})
    joined, deleted, restored = saved
    assert joined[0] == restored[0]
    assert {k:v for k,v in joined[0].items() if k != 'geometry'} == {
        k:v for k,v in deleted[0].items() if k != 'geometry'}
    assert len(joined[0]['geometry']['objects']) == 3 and len(deleted[0]['geometry']['objects']) == 1
    for a, b, c in zip(joined[3], deleted[3], restored[3]):
        assert np.array_equal(a, b) and np.array_equal(a, c)
    changed_cells = np.count_nonzero(joined[2] != deleted[2])
    assert changed_cells > 0 and np.array_equal(joined[2], restored[2])
    delta = difference(joined[1], deleted[1])
    repeated = difference(joined[1], restored[1])
    assert delta > 1e-3
    np.testing.assert_allclose(joined[1], restored[1], rtol=1e-6, atol=1e-12)
    return {'target': browser['target'], 'actual_runs': result, 'same_grid_source_and_settings': True,
            'native_changed_material_cells': int(changed_cells), 'deleted_vs_joined_field_relative_l2': delta,
            'undo_vs_joined_field_relative_l2': repeated,
            'undo_restored_all_field_bytes': joined[1].tobytes() == restored[1].tobytes(),
            'undo_restored_all_native_material_bytes': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--browser-reports', nargs=3, type=Path, required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cases = [compare_edits(args.runs, path) for path in args.browser_reports]
    assert {case['target'] for case in cases} == {'meep:cpu', 'fdtdx:cpu', 'fdtdx:gpu'}
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'study': 'A strip aligned to lower_in of a two-guide directional coupler, before deletion, after deletion and after Undo. Same physical grid and dipole throughout each run triplet.',
              'cases': cases}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
