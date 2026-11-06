"""Cache identity changes must recompute a stage without exposing a physical-device fingerprint."""
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.calibration_checkpoint import MeasurementCheckpoints
from scripts import ring_stage_provenance as provenance


class StageIdentity(unittest.TestCase):
    def test_every_transported_source_participates_in_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in provenance.RING_SOURCE_FILES:
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(name)
            original = provenance.source_fingerprint(root)
            for name in provenance.RING_SOURCE_FILES:
                file = root / name
                file.write_text(name + ' changed')
                changed = provenance.source_fingerprint(root)
                self.assertNotEqual(changed[name], original[name])
                file.write_text(name)

    @unittest.skipUnless(find_spec('packaging'), 'Runtime dependency manifests are evaluated in the numerical environment.')
    def test_real_dependency_payload_change_invalidates_an_unchanged_version(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            distributions = []
            names = ('fdtdx', 'jax', 'jaxlib', 'numpy', 'scipy', 'tidy3d', 'shared-mode-lib')
            for name in names:
                file = root / (name + '.py')
                file.write_text('original')
                distributions.append(SimpleNamespace(metadata={'Name':name}, version='1.0', files=[Path(file.name)],
                    requires=['shared-mode-lib'] if name == 'tidy3d' else [],
                    locate_file=lambda member: root / member, read_text=lambda filename: None))
            with patch.object(provenance.metadata,'distributions',return_value=distributions):
                first=provenance.software_fingerprint()
                (root/'shared-mode-lib.py').write_text('changed local mode solver')
                newer=provenance.software_fingerprint()
            self.assertNotEqual(first['shared-mode-lib']['sha256'],newer['shared-mode-lib']['sha256'])
            self.assertEqual(first['fdtdx'],newer['fdtdx'])

    def test_same_card_reuses_stage_and_another_same_model_must_calculate(self):
        with TemporaryDirectory() as tmp:
            cache = MeasurementCheckpoints.__new__(MeasurementCheckpoints)
            cache.root = Path(tmp)
            cache.provenance = {'schema':provenance.SCHEMA,'hardware':{'model':'same model','private_device_key':'A','driver_build':'tested-1'}}
            called=[]
            def measure():
                called.append(True)
                return {'device':'cuda:0'}, {'real_test_vector':np.array([1,2])}
            cache.acquire('reference',{'resolution':16},measure)
            cache.acquire('reference',{'resolution':16},measure)
            self.assertEqual(len(called),1)
            cache.provenance['hardware']['driver_build']='tested-2'
            cache.acquire('reference',{'resolution':16},measure)
            self.assertEqual(len(called),2)
            cache.provenance['hardware']['driver_build']='tested-1'
            cache.provenance['hardware']['private_device_key']='B'
            cache.acquire('reference',{'resolution':16},measure)
            self.assertEqual(len(called),3)
            public = provenance.public_fingerprint(cache.provenance)
            self.assertNotIn('private_device_key',public['hardware'])
            self.assertIn('private_device_key',cache.provenance['hardware'])

    def test_unknown_card_disables_persistent_cache(self):
        config=SimpleNamespace(jax_enable_x64=False,jax_disable_jit=False,jax_default_matmul_precision=None)
        runtime=(SimpleNamespace(config=config),None,None,object())
        with patch.object(provenance,'source_fingerprint',return_value={}), \
             patch.object(provenance,'physical_gpu',side_effect=ValueError('private diagnostic text')):
            record, reusable = provenance.fingerprint(Path('.'),runtime,{})
        self.assertFalse(reusable)
        self.assertNotIn('private diagnostic text',record['reuse_disabled_reason'])


if __name__ == '__main__':
    unittest.main()
