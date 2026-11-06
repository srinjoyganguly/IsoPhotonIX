import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.phase_component_comparison import compare, fixed_mode_ring_spectrum, measured_beta, native_data


class PhaseRegistrationTests(unittest.TestCase):
    def test_measured_straight_lead_resolves_many_phase_wraps(self):
        wave = np.linspace(1.50,1.60,51)
        length = 7.75
        beta = 2*np.pi*1.97/wave
        through = .99*np.exp(1j*beta*length)
        found = measured_beta(through,wave,np.full_like(wave,1.96),length)
        np.testing.assert_allclose(found,beta,atol=1e-13)
        native = np.exp(1j*found*length)
        np.testing.assert_allclose(native*np.exp(1j*found*(8.4-length)),np.exp(1j*beta*8.4))

    def test_phase_increment_is_gauge_independent_and_rejects_weak_return(self):
        wave = np.linspace(1.50,1.60,10)
        a = np.stack([np.full_like(wave,.005),.5*np.exp(1j*wave*4)],axis=1)
        b = np.exp(.8j)*a
        ok = np.ones_like(a,dtype=bool)
        result = compare('grating',a,b,np.ones(len(wave),bool),ok,vendor=True)
        self.assertEqual(result['reflection']['strong_samples'],0)
        self.assertEqual(result['reflection']['phase_increment_delta_rad']['count'],0)
        self.assertEqual(result['guide']['phase_increment_delta_rad']['count'],9)
        self.assertLess(result['guide']['phase_increment_delta_rad']['max_abs'],1e-12)
        self.assertNotIn('same_gauge_phase_delta_rad',result['guide'])

    def test_fixed_profile_interpolates_once_before_transform(self):
        t=np.arange(151)*.1
        profiles=np.zeros((len(t),1,3,2))
        profiles[:,0,:,0]=np.cos(2*np.pi*t[:,None]/1.55)*np.array([1,3,5])[None,:]
        w=np.array([1.51,1.57])
        fixed=fixed_mode_ring_spectrum(t,profiles,np.array([1.5,1.55,1.6]),1.525,w)
        expected=.1*np.exp(2j*np.pi*t[None,:]/w[:,None])@(2*np.cos(2*np.pi*t/1.55))
        np.testing.assert_allclose(fixed[:,0,0],expected,rtol=1e-12)
        np.testing.assert_array_equal(fixed[:,0,1],0)

    def test_partial_ring_requires_explicit_verified_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/'arrays.npz'
            np.savez(data,valid_bins=[True,False,True])
            record={'case_id':'loaded-ring-3d','passed':False,'measurement_complete':True,'actual_device':'cuda:0',
                    'arrays_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'total_bins':3,'valid_bins':2,
                    'checkpoint_identity':{'persistent_reuse_eligible':True},
                    'stages':{'reference':{'device':'cuda:0'},'device':{'device':'cuda:0'}}}
            path=root/'report.json';path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                native_data(path)
            report,archive=native_data(path,allow_partial_ring=True)
            with archive:
                np.testing.assert_array_equal(archive['valid_bins'],[True,False,True])
            self.assertFalse(report['passed'])
            record['valid_bins']=3;path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                native_data(path,allow_partial_ring=True)

    def test_drops_intervals_spanning_invalid_bins(self):
        values = np.ones((7,2),dtype=complex)
        valid = np.array([False,True,True,False,True,True,False])
        report = compare('grating',values,values,valid,np.ones_like(values,dtype=bool))
        self.assertEqual(report['guide']['phase_increment_delta_rad']['count'],2)


if __name__=='__main__':
    unittest.main()
