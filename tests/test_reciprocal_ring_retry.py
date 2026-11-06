from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from scripts.calibration_checkpoint import MeasurementCheckpoints
from scripts.run_reciprocal_ring3d import child,run,new_verified_reference


class RingAllocatorRetry(unittest.TestCase):
    def test_real_child_only_labels_specific_post_reference_memory_refusal(self):
        import sys
        line='MemoryError: Fresh ring stage requires 8372 MiB free including reserve; measured 8235 MiB.'
        for messages,expected in [(['reference 16/700 samples','reference complete cuda:0',line], True),
                                 (['reference complete cuda:0',line], False),
                                 (['reference 16/700 samples','reference complete cuda:0','MemoryError: unrelated'],False)]:
            code,flag=child([sys.executable,'-c','import sys;print(sys.argv[1]);sys.exit(1)','\n'.join(messages)])
            self.assertEqual((code,flag),(1,expected))

    def test_retry_preserves_and_verifies_new_reference_without_user_checkpoint(self):
        calls=[]
        def fake_child(command):
            folder=Path(command[command.index('--checkpoint')+1])
            if not calls:
                c=MeasurementCheckpoints.__new__(MeasurementCheckpoints); c.root=folder;c.provenance={'test':True}
                c.acquire('reference',{'test':1},lambda:({'time_steps':2},{'unit':np.ones(1)}))
            else:
                self.assertTrue(new_verified_reference(folder,set()))
            calls.append(folder)
            return (1,True) if len(calls)==1 else (0,False)
        with patch('scripts.run_reciprocal_ring3d.child',side_effect=fake_child):
            self.assertEqual(run(['--device','gpu']),0)
        self.assertEqual(calls[0],calls[1])
        self.assertFalse(calls[0].exists())

    def test_different_error_or_missing_reference_is_never_retried(self):
        with TemporaryDirectory() as tmp:
            for flag in (True,False):
                with patch('scripts.run_reciprocal_ring3d.child',return_value=(2,flag)) as launch:
                    self.assertEqual(run(['--checkpoint',tmp]),2)
                    self.assertEqual(launch.call_count,1)


    def test_corrupted_prior_array_is_never_used_as_recovery_evidence(self):
        with TemporaryDirectory() as tmp:
            c=MeasurementCheckpoints.__new__(MeasurementCheckpoints); c.root=Path(tmp);c.provenance={'test':True}
            c.acquire('reference',{'test':1},lambda:({'time_steps':1},{'unit':np.ones(1)}))
            self.assertIsNotNone(new_verified_reference(Path(tmp),set()))
            next(Path(tmp).glob('reference-*/arrays.npz')).write_bytes(b'corrupt fixture')
            self.assertIsNone(new_verified_reference(Path(tmp),set()))
