"""Check the native checkpoint and resume boundaries without running FDTD."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from scripts import calibrate_multiport as study


class MeepMultiportResume(unittest.TestCase):
    def test_archive_marker_binds_arrays_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); metadata=root/'reference.json'; archive=root/'reference.npz'
            study.atomic_archive(archive,E=np.array([1+2j]),H=np.array([3-4j]))
            self.assertIsNone(study.read_checkpoint(metadata,archive,'key'))
            digest=hashlib.sha256(archive.read_bytes()).hexdigest()
            study.atomic_json(metadata,{'checkpoint_key':'key','archive_sha256':digest})
            self.assertEqual(study.read_checkpoint(metadata,archive,'key')['archive_sha256'],digest)
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                study.read_checkpoint(metadata,archive,'different-key')
            study.atomic_archive(archive,E=np.array([8j]),H=np.array([9j]))
            with self.assertRaisesRegex(ValueError,'archive'):
                study.read_checkpoint(metadata,archive,'key')

    def test_resume_refuses_changed_input_mesh_or_solver(self):
        module=types.ModuleType('meep');module.__version__='1.35.0'
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,{'meep':module}), patch.object(study,'measure_locked') as measured:
            path=Path(directory)/'material-study'
            args=dict(case='splitter',output=path,launch_port='left_center')
            study.measure(**args,stage='reference')
            study.measure(**args,resume=True,stage='device')
            self.assertEqual(measured.call_count,2)
            self.assertEqual(measured.call_args.args[-1],measured.call_args_list[0].args[-1])
            for change in ({'resolution':20},{'launch_port':'right_lower'},{'aperture_um':2.25}):
                with self.assertRaisesRegex(ValueError,'Resume refused'):
                    study.measure(**(args|change),resume=True)
            module.__version__='changed'
            with self.assertRaisesRegex(ValueError,'Resume refused'):
                study.measure(**args,resume=True)
            self.assertEqual(json.loads((path/'config.json').read_text())['phase_convention_version'],2)
            self.assertEqual(measured.call_count,2)
