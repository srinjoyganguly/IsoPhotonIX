import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.phase_component_comparison import compare, fixed_mode_ring_spectrum, measured_beta, native_data, sampled_mean_power, tidy_grating


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
            np.savez(data,valid_bins=[True,False,True],device_ring_Hz=np.ones((12,3)),device_time_um_c=np.arange(12.))
            record={'case_id':'loaded-ring-3d','passed':False,'measurement_complete':True,'actual_device':'cuda:0',
                    'arrays_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'max_collected_port_power':1.,'total_bins':3,'valid_bins':2,
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

    def test_runaway_field_acquisition_with_nominal_valid_bin_is_rejected(self):
        from scripts.phase_stability_check import growth_diagnostic
        field=np.repeat([.2,.02,.5,16.],3)[:,None]*np.ones((1,3))
        record={'max_collected_port_power':1.74}
        arrays={'device_ring_Hz':field,'device_time_um_c':np.arange(12.)}
        self.assertTrue(growth_diagnostic(record,arrays)['reject_scattering'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            archive=root/'arrays.npz';np.savez(archive,**arrays,valid_bins=[True,False,False])
            record.update(case_id='loaded-ring-3d',passed=False,actual_device='cuda:0',measurement_complete=True,
                          checkpoint_identity={'persistent_reuse_eligible':True},stages={'reference':{'device':'cuda:0'},'device':{'device':'cuda:0'}},
                          valid_bins=1,total_bins=3,arrays_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            path=root/'report.json';path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'runaway late fields'):
                native_data(path,allow_partial_ring=True)

    def test_sparse_return_power_integral_skips_unqualified_gaps(self):
        w=np.array([1.5,1.51,1.52,1.55,1.60])
        valid=np.array([True,True,False,True,True])
        a=np.array([1.,2.,90.,3.,4.])
        result=sampled_mean_power(a,a,w,valid)
        self.assertEqual(result['adjacent_intervals'],2)
        df=abs(np.diff(299792458e6/w))[[0,3]]
        expected=np.dot(df,[2.5,12.5])/df.sum()
        self.assertAlmostEqual(result['mean_power_a'],expected)

    def test_grating_return_and_guide_remove_arbitrary_impressed_pulse_phase(self):
        n,res=16,10
        xy=(np.arange(n)-n/2+.5)/res
        power=np.full(2,.5*(n/res)**2)
        wave=np.array([1.5,1.6])
        settings={'resolution':res,'geometry':'tidy3d','taper':4.,'buffer':1.,'beam_x':0.}
        details={'settings':settings,'arrays_sha256':'synthetic',
                 'stages':{'reference':{'sample_plane_z_um':.93},'device':{'output_plane_x_um':-4.45,'history':[{'time_fs':10.}],'time_steps':100}}}
        expected=.08+.02j
        for phase in (0.,.78,-1.3):
            rotation=np.exp(1j*phase)
            incident=np.zeros((2,6,n,n),complex)
            incident[:,1]=rotation;incident[:,3]=rotation
            scatter=np.zeros_like(incident)
            scatter[:,1]=expected*rotation;scatter[:,3]=-expected*rotation
            guided=.4*np.sqrt(power)*rotation
            data={'ref_upper':incident,'dev_upper':incident+scatter,'ref_wavelength_um':wave,
                  'ref_upper_x_um':xy,'ref_upper_y_um':xy,'incident_power':power,'output_mode':guided,
                  'dev_mode_neff':np.full(2,2.4),'coupling':np.full(2,.16),'valid':np.ones(2,bool),
                  'dev_history_observable':np.stack([guided,guided,guided])}
            record=tidy_grating(details,data)
            np.testing.assert_allclose(record['s'][:,0],expected,rtol=1e-12)
            np.testing.assert_allclose(record['s'][:,1],.4,rtol=1e-12)

    def test_drops_intervals_spanning_invalid_bins(self):
        values = np.ones((7,2),dtype=complex)
        valid = np.array([False,True,True,False,True,True,False])
        report = compare('grating',values,values,valid,np.ones_like(values,dtype=bool))
        self.assertEqual(report['guide']['phase_increment_delta_rad']['count'],2)


if __name__=='__main__':
    unittest.main()
