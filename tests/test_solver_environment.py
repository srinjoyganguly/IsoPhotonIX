import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from backend.solver_environment import solver_interpreter


class SolverSelection(unittest.TestCase):
    def test_cpu_and_cuda_are_independent_and_explicit_override_wins(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            state = Path(tmp) / '.local'
            state.mkdir()
            registry = state / 'runtime.json'
            registry.write_text(json.dumps({'fdtdx_cpu_python': 'cpu-env', 'fdtdx_gpu_python': 'cuda-env',
                                            'fdtdx_python': 'legacy', 'meep_python': 'meep-env'}))
            self.assertEqual(solver_interpreter('fdtdx', 'gpu', tmp), 'cuda-env')
            self.assertEqual(solver_interpreter('fdtdx', 'cpu', tmp), 'cpu-env')
            self.assertEqual(solver_interpreter('meep', 'cpu', tmp), 'meep-env')
            registry.write_text(json.dumps({'fdtdx_gpu_python': 'cuda-env', 'fdtdx_python': 'legacy'}))
            self.assertEqual(solver_interpreter('fdtdx', 'cpu', tmp), 'legacy')
            registry.write_text(json.dumps({'fdtdx_gpu_python': 'cuda-env'}))
            self.assertEqual(solver_interpreter('fdtdx', 'cpu', tmp), 'cuda-env')
            with patch.dict(os.environ, {'FDTDX_PYTHON': 'override'}):
                self.assertEqual(solver_interpreter('fdtdx', 'gpu', tmp), 'override')
                self.assertEqual(solver_interpreter('fdtdx', 'cpu', tmp), 'override')

    def test_gpu_request_does_not_select_managed_cpu_profile(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            state = Path(tmp) / '.local'
            state.mkdir()
            (state / 'runtime.json').write_text(json.dumps({'fdtdx_cpu_python': 'cpu-env'}))
            self.assertNotEqual(solver_interpreter('fdtdx', 'gpu', tmp), 'cpu-env')

    def test_beamz_cpu_install_does_not_service_cuda_or_select_an_fdtdx_environment(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            state = Path(tmp) / '.local'
            (state/'fdtdx-env/bin').mkdir(parents=True)
            (state/'fdtdx-env/bin/python').touch()
            (state/'runtime.json').write_text(json.dumps({'beamz_cpu_python':'beamz-cpu-env'}))
            self.assertEqual(solver_interpreter('beamz', 'cpu', tmp), 'beamz-cpu-env')
            import sys
            self.assertEqual(solver_interpreter('beamz', 'gpu', tmp), sys.executable)
