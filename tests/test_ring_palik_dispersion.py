import unittest

import numpy as np

from scripts import ring_palik_dispersion as material


class RingPalikFit(unittest.TestCase):
    def test_held_out_values_and_passivity_on_supported_steps(self):
        for mesh in (12,16,20,24,48,96):
            dt=.99e-6/(np.sqrt(3)*mesh*material.LIGHT)
            for samples in material.material_samples().values():
                high,c3=material.fit_effective_cells(samples[material.TRAIN],dt)
                c1,c2,k=material.recurrence(dt)
                self.assertGreater(float(high),1.)
                self.assertGreater(float(c3),0.)
                self.assertLess(k*(1+c3/(k*high)),4)
                eps=material.epsilon_at(c1,c2,c3,high,material.WAVELENGTHS,dt)
                np.testing.assert_allclose(np.sqrt(eps[material.HOLDOUT]),np.sqrt(samples[material.HOLDOUT]),rtol=0,atol=.001)

    def test_cellwise_fit_preserves_air_and_refuses_active_data(self):
        samples=next(iter(material.material_samples().values()))[material.TRAIN]
        dt=1e-16
        high,c3=material.fit_effective_cells(np.column_stack((np.ones(3),samples)),dt)
        self.assertAlmostEqual(high[0],1.)
        self.assertAlmostEqual(c3[0],0.)
        with self.assertRaises(ValueError):
            material.fit_effective_cells(samples[::-1],dt)
        with self.assertRaises(ValueError):
            material.recurrence(2.01/material.OMEGA_0)
