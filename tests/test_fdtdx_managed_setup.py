"""Managed FDTDX settings are published only after a verified native solve."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from photonics_workbench import fdtdx_setup


class FDTDXManagedSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        self.state = self.data/'.local'
        self.state.mkdir()
        self.path = self.state/'runtime.json'
        self.initial = {'meep_python':'keep-meep','fdtdx_python':'keep-legacy','unrelated':True}
        self.path.write_text(json.dumps(self.initial))

    def setup(self, profile, **kwargs):
        return fdtdx_setup.setup_fdtdx(profile=profile, offline=True, wheelhouse=kwargs.get('wheelhouse', self.data),
                                      data_dir=self.data, resources=ROOT)

    @staticmethod
    def create_venv(command, **kwargs):
        if 'venv' in command:
            executable = Path(command[-1])/'bin/python'
            executable.parent.mkdir(parents=True)
            executable.touch()
        elif 'install' in command:
            assert '--require-hashes' in command and '--no-index' in command and '--only-binary=:all:' in command
        else:
            raise AssertionError(f'Unexpected command: {command}')

    def test_lock_profile_and_independent_runtime_pointers(self):
        cpu = fdtdx_setup._locked_versions(ROOT/'environment-fdtdx-cpu-linux-64.lock')
        cuda = fdtdx_setup._locked_versions(ROOT/'environment-fdtdx-cuda12-linux-64.lock')
        self.assertEqual(len(cpu), 77)
        self.assertEqual(len(cuda), 92)
        self.assertEqual({key:cuda[key] for key in cpu}, cpu)
        self.assertNotIn('jax-cuda12-plugin', cpu)
        self.assertEqual(cuda['jax-cuda12-plugin'], cuda['jax'])
        with patch.object(subprocess,'run',side_effect=self.create_venv) as run, patch.object(fdtdx_setup,'_verify',return_value={'targets':{}}):
            gpu = self.setup('fdtdx-cuda12')
            cpu = self.setup('fdtdx-cpu')
            again = self.setup('fdtdx-cuda12', wheelhouse=None)
        settings = json.loads(self.path.read_text())
        self.assertEqual(run.call_count, 4)
        self.assertEqual(again['status'], 'already-ready')
        self.assertEqual(gpu['python'], again['python'])
        self.assertEqual(settings['fdtdx_cpu_python'], cpu['python'])
        self.assertEqual(settings['fdtdx_gpu_python'], gpu['python'])
        self.assertNotEqual(cpu['python'], gpu['python'])
        for key, value in self.initial.items():
            self.assertEqual(settings[key], value)

    def test_missing_offline_wheelhouse_and_gpu_probe_failure_preserve_settings(self):
        with self.assertRaisesRegex(ValueError, 'wheelhouse'):
            self.setup('fdtdx-cpu', wheelhouse=None)
        with patch.object(subprocess,'run',side_effect=self.create_venv), patch.object(fdtdx_setup,'_verify',side_effect=RuntimeError('GPU unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'GPU unavailable'):
                self.setup('fdtdx-cuda12')
        self.assertEqual(json.loads(self.path.read_text()), self.initial)
        self.assertEqual(list((self.state/'runtimes').iterdir()), [])

    def test_failure_when_repairing_existing_profile_keeps_prior_prefix_and_alias(self):
        with patch.object(subprocess,'run',side_effect=self.create_venv), patch.object(fdtdx_setup,'_verify',return_value={}):
            previous = self.setup('fdtdx-cpu')
        before = json.loads(self.path.read_text())
        with patch.object(subprocess,'run',side_effect=subprocess.CalledProcessError(1, 'venv')), patch.object(fdtdx_setup,'_verify',side_effect=RuntimeError('damaged')):
            with self.assertRaises(subprocess.CalledProcessError):
                self.setup('fdtdx-cpu')
        self.assertEqual(json.loads(self.path.read_text()), before)
        self.assertTrue(Path(previous['python']).is_file())

    def test_invalid_settings_survive_a_failed_setup(self):
        self.path.write_text('{"managed_fdtdx":[]}')
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            self.setup('fdtdx-cpu')
        self.assertEqual(self.path.read_text(), '{"managed_fdtdx":[]}')

    def test_unsupported_python_and_platform_rejected_before_install(self):
        with patch.object(fdtdx_setup.platform, 'system',return_value='Windows'):
            with self.assertRaisesRegex(RuntimeError, 'Linux'):
                self.setup('fdtdx-cpu')
        with patch.object(fdtdx_setup.sys, 'version_info',(3, 12)):
            with self.assertRaisesRegex(RuntimeError, '3.11'):
                self.setup('fdtdx-cpu')
        self.assertEqual(json.loads(self.path.read_text()), self.initial)
