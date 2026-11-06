import csv
import hashlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.calibration_performance import current_totals, stage_rows, timing_csv
from scripts.gpu_timing import StageTiming


class TimingReports(unittest.TestCase):
    def test_synchronized_cpu_receipt_does_not_claim_cuda_measurement(self):
        device=SimpleNamespace(platform='cpu',device_kind='CPU')
        with patch('scripts.gpu_timing.perf_counter', side_effect=[1.,2.,2.5,3.,4.,5.]):
            t=StageTiming(device)
            with t.compile(): pass
            t.begin_steps()
            with t.step(): pass
            t.end_steps()
        measurement=t.report(shape=(4,5,6),steps=2,dt_seconds=1e-16,spacing_um=.1,pml_cells=1)
        self.assertEqual(measurement['compile_seconds'],.5)
        self.assertEqual(measurement['synchronized_step_seconds'],1)
        self.assertEqual(measurement['stage_through_steps_seconds'],4)
        self.assertEqual(measurement['cells'],120)
        self.assertIsNone(measurement['cuda_event_elapsed_seconds'])

    def test_reuse_is_a_use_and_not_a_second_acquisition(self):
        case={'actual_device':'cuda:0','case_id':'finite-grating-3d','settings':{'resolution':16,'geometry':'lumerical'}}
        stage={'performance':{'acquisition_id':'single','grid_shape':[2,3,4],'steps':88,'dt_seconds':1e-17,
                             'synchronized_step_seconds':5.,'cuda_event_elapsed_seconds':5.002},
               'arrays_sha256':hashlib.sha256(b'example').hexdigest(),'acquisition':{'status':'fresh'}}
        first=stage_rows({**case,'stages':{'device':stage}},'fresh')
        second=stage_rows({**case,'stages':{'device':{**stage,'acquisition':{'status':'cache-hit'}}}},'cached')
        self.assertEqual([first[0]['origin'],second[0]['origin']],['computed_this_invocation','reused_verified_receipt'])
        totals=current_totals(first+second)
        self.assertEqual(totals['synchronized_step_seconds'],5.)
        self.assertEqual(totals['available_cuda_event_elapsed_seconds'],5.002)
        self.assertIsNone(totals['available_kernel_active_seconds'])
        csv_rows=list(csv.DictReader(io.StringIO(timing_csv(first+second))))
        self.assertEqual(json.loads(csv_rows[0]['grid_shape']),[2,3,4])
        self.assertEqual(csv_rows[1]['kernel_active_seconds'],'')
        with self.assertRaises(ValueError):
            current_totals(first+[dict(second[0],steps=87)])

    def test_historical_acquisition_and_missing_costs_stay_separate(self):
        raw={'actual_device':'cuda:0','case_id':'loaded-ring-3d','physical_model':{'shape':[3,4,5],'resolution':16},
             'stages':{'device':{'time_steps':42,'time_step_um_c':.04,'elapsed_s':1.25}}}
        row=stage_rows(raw,'archive')[0]
        self.assertEqual(row['historical_elapsed_seconds'],1.25)
        self.assertEqual(row['cells'],60)
        self.assertAlmostEqual(row['dt_seconds'],.04e-6/299792458)
        self.assertIsNone(row['synchronized_step_seconds'])
        self.assertIsNone(current_totals([row])['synchronized_step_seconds'])
        self.assertFalse(stage_rows({**raw,'actual_device':'cpu:0'},'archive'))
