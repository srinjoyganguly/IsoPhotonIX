"""Independent analytic contract checks; no Meep/MPB or API import required."""
import math
import unittest

from backend.mode_benchmarks import analytic_te0, CASES


class AnalyticModeTests(unittest.TestCase):
    def test_symmetric_root_matches_independent_phase_sum_values(self):
        # Independently calculated using the asymmetric phase-sum equation;
        # provenance is in docs/verification/public-reference-analytic.json.
        self.assertAlmostEqual(analytic_te0(3.48, 1.44, .22), 2.851356345104066, places=12)
        self.assertAlmostEqual(analytic_te0(3.48, 1.44, .11), 2.2715310649710316, places=12)

    def test_solution_satisfies_interface_phase_and_scale_invariance(self):
        neff = analytic_te0(1.5, 1., 1., 1.55)
        k0 = 2*math.pi/1.55
        transverse_k = k0*math.sqrt(1.5**2-neff**2)
        decay = k0*math.sqrt(neff**2-1.)
        self.assertAlmostEqual(transverse_k, 2*math.atan(decay/transverse_k), places=12)
        self.assertAlmostEqual(neff, analytic_te0(1.5, 1., 10., 15.5), places=12)

    def test_guidance_limits_and_invalid_physical_inputs(self):
        self.assertLess(abs(analytic_te0(1.5, 1., 1e-5)-1.), 1e-8)
        self.assertLess(abs(analytic_te0(1.5, 1., 100.)-1.5), 1e-4)
        for args in ((1., 1.5, 1.), (1.5, 1., 0.), (float('nan'), 1., 1.),
                     (float('inf'), 1., 1.)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                analytic_te0(*args)

    def test_conflicting_official_material_indices_are_preserved_as_separate_cases(self):
        text_case, script_case = CASES['lumerical-slab'], CASES['lumerical-slab-script']
        self.assertEqual(text_case['core_index'], 3.48)
        self.assertEqual(script_case['core_index'], 3.47668)
        self.assertEqual(text_case['reference_neff'], script_case['reference_neff'])
        # Do not silently replace the analytic solution with the published,
        # rounded finite-grid scalar merely to force numerical agreement.
        self.assertGreater(abs(analytic_te0(3.48, 1.44, .22)-text_case['reference_neff']), .003)
