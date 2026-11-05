"""Spot-check the published observable definitions in the actual measured record."""
import json
from pathlib import Path
import unittest

RECORD = Path(__file__).resolve().parents[1]/'docs/verification/public-component-comparisons.json'


class ComponentReferenceEvidence(unittest.TestCase):
    def setUp(self):
        self.cases = {case['id']:case for case in json.loads(RECORD.read_text())['cases']}

    def test_transfer_length_uses_half_a_beat_period_and_preserves_gap(self):
        case = self.cases['lumerical-coupler']
        self.assertEqual(case['physics']['edge_gap_um'], .05)
        for row in [*case['refinement'], case['domain_check']]:
            q = row['quantities']
            self.assertGreater(q['even_neff'], q['odd_neff'])
            self.assertAlmostEqual(q['transfer_length_um']*(q['even_neff']-q['odd_neff']), .775)
            self.assertGreater(row['modes']['even']['dominant_field_lateral_mirror_overlap'], .97)
            self.assertLess(row['modes']['odd']['dominant_field_lateral_mirror_overlap'], -.97)

    def test_group_index_includes_frequency_derivative_of_the_material_inputs(self):
        case = self.cases['tidy3d-strip-plugin']
        step = case['physics']['group_frequency_step_fraction']
        samples = list(case['physics']['eps_samples'].values())
        self.assertGreater(samples[-1]['core_real'], samples[0]['core_real'])
        self.assertGreater(samples[1]['clad_imaginary'], 0)
        for row in [*case['refinement'], case['domain_check']]:
            for pol in ('te','tm'):
                modes = row['modes'][pol]
                self.assertEqual([mode['frequency_ratio'] for mode in modes],[1-step,1.,1+step])
                expected = modes[1]['neff'] + (modes[2]['neff']-modes[0]['neff'])/(2*step)
                self.assertAlmostEqual(row['quantities'][pol+'_group'], expected)

    def test_report_compares_the_finest_standard_mesh_and_exposes_both_sensitivities(self):
        for case in self.cases.values():
            previous, fine = case['refinement'][-2:]
            self.assertEqual(fine['resolution_cells_per_um'],case['domain_check']['resolution_cells_per_um'])
            self.assertTrue(all(b>a for a,b in zip(fine['transverse_span_um'],case['domain_check']['transverse_span_um'])))
            for metric in case['metrics']:
                key = metric['key']
                self.assertEqual(metric['computed'],fine['quantities'][key])
                self.assertAlmostEqual(metric['relative_error'],abs(metric['computed']/metric['reference']-1))
                self.assertAlmostEqual(metric['last_mesh_change'],abs(metric['computed']-previous['quantities'][key]))
                self.assertAlmostEqual(metric['larger_domain_change'],abs(metric['computed']-case['domain_check']['quantities'][key]))


class ServedBenchmarkChecks(unittest.TestCase):
    def test_changed_displayed_difference_is_rejected(self):
        import tempfile
        from backend.benchmarks import component_benchmark_report
        report = component_benchmark_report()
        report['cases'][0]['metrics'][0]['relative_error'] = 1.0
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder)/'report.json'
            bad.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'differences disagree'):
                component_benchmark_report(bad)
