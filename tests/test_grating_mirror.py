import unittest
import numpy as np

from scripts.calibrate_finite_grating import settings
from scripts.grating_mirror import PARITY, full_plane, positive_count


class GratingMirror(unittest.TestCase):
    def test_reconstructs_every_component_on_original_even_aperture(self):
        # Four physical samples j=-2,-1,0,1 require three positive samples j=0,1,2.
        native=np.ones((1,6,1,3,1),dtype=np.complex64)*(2+3j)
        native[:,[0,2,4],:,0,:]=0
        native[:,[1,3,5],:,0,:]*=.5
        native[:,:,:,2,:]*=2
        self.assertEqual(positive_count(8,4,10),3)
        a=full_plane(native,3,1,8,4,10)[0,:,0,:,0]
        np.testing.assert_array_equal(a[:,0],2*(2+3j)*PARITY)
        np.testing.assert_array_equal(a[:,1],(2+3j)*PARITY)
        np.testing.assert_array_equal(a[:,2],(2+3j)*(PARITY==1))
        np.testing.assert_array_equal(a[:,3],(2+3j)*np.ones(6))
        self.assertEqual(a.dtype,np.complex64)

    def test_scope_and_missing_edge_rejected(self):
        with self.assertRaisesRegex(ValueError,'aperture'):
            full_plane(np.zeros((6,2)),1,0,8,4,10)
        with self.assertRaisesRegex(ValueError,'direct Yee'):
            settings('lumerical',12,grating_domain='mirror-y')
        with self.assertRaisesRegex(ValueError,'symmetric'):
            settings('compact',12,grating_domain='mirror-y')
