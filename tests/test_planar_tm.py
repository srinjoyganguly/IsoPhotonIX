"""Qualification of physical TM reference planes and comparable cross-sectional modes."""
import importlib.util
import unittest

import numpy as np

from scripts.calibrate_planar_tm import interface_matrix, slab_indices


@unittest.skipUnless(importlib.util.find_spec("scipy"), "The independent TM solve uses SciPy.")
class PlanarTMReferenceTests(unittest.TestCase):
    def test_exact_symmetric_slab_modes(self):
        np.testing.assert_allclose(slab_indices(1/1.55), [2.9162637564, 1.5416718866], rtol=1e-9)

    def test_straight_tm_phase_and_normalization(self):
        distance = .75
        matrix, indices = interface_matrix(1/1.55,0,distance,1/32,7.)
        expected = np.zeros((4,4), complex)
        for m, index in enumerate(indices):
            expected[m,m+2] = expected[m+2,m] = np.exp(4j*np.pi*index*distance/1.55)
        np.testing.assert_allclose(matrix,expected,atol=1e-8)

    def test_variable_planes_rotate_phase_without_changing_power(self):
        near, indices = interface_matrix(1/1.55,.125,.75,1/32,7.)
        far, _ = interface_matrix(1/1.55,.125,1.5,1/32,7.)
        phase = np.exp(2j*np.pi*np.tile(indices,2)*.75/1.55)
        np.testing.assert_allclose(far,near*phase[:,None]*phase[None,:],atol=1e-9)
        self.assertLess(float(np.max(np.abs(far-far.T))), .001)
