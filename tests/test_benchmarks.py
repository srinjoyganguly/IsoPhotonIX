"""Report-integrity checks; fixtures are not numerical validation evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from backend.benchmarks import benchmark_report


class BenchmarkEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'report.json'
        self.report = {'schema_version': 1, 'solver': 'fixture-only', 'versions': {'fixture': '1'},
                       'generated_at': '2026-11-01T00:00:00Z', 'cases': [
                           {'id': 'fixture', 'reference_kind': 'published_solver_value',
                            'source_url': 'https://example.org/reference', 'reference_neff': 2.,
                            'computed_neff': 2.002, 'relative_error': .001}]}

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.report))

    def test_missing_report_does_not_invent_results(self):
        actual = benchmark_report(self.path)
        self.assertEqual(actual['status'], 'not_run')
        self.assertEqual(actual['cases'], [])

    def test_exact_numerical_payload_and_source_identity_are_retained(self):
        self.save()
        actual = benchmark_report(self.path)
        self.assertEqual(actual['cases'], self.report['cases'])
        self.assertEqual(actual['scope'], 'mode-benchmarks')
        self.assertEqual(actual['report_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertIn('do not certify driven 3D', actual['notes'][0])
        self.assertNotIn('passed', actual)

    def test_inconsistent_discrepancy_or_unknown_reference_kind_is_rejected(self):
        for key, value in [('relative_error', 0), ('reference_kind', 'verified-commercial'), ('source_url', 'javascript:alert(1)'), ('computed_neff', float('nan')), ('computed_neff', True)]:
            original = self.report['cases'][0][key]
            self.report['cases'][0][key] = value; self.save()
            with self.subTest(key=key), self.assertRaises(ValueError): benchmark_report(self.path)
            self.report['cases'][0][key] = original

    def test_missing_provenance_duplicate_case_and_invalid_shape_are_rejected(self):
        self.report['versions'] = {}; self.save()
        with self.assertRaises(ValueError): benchmark_report(self.path)
        self.report['versions'] = {'fixture': '1'}
        self.report['cases'].append(dict(self.report['cases'][0])); self.save()
        with self.assertRaises(ValueError): benchmark_report(self.path)
        self.path.write_text('[]')
        with self.assertRaises(ValueError): benchmark_report(self.path)
