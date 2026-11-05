"""Setup failure must preserve a usable runtime and independent solver settings."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from photonics_workbench import managed_setup


class ManagedSetupTests(unittest.TestCase):
    def test_failed_install_does_not_switch_the_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/'environment-meep-linux-64.lock').write_text('@EXPLICIT\n')
            state = directory/'.local'
            state.mkdir()
            settings = {'meep_python':'previous-working-python', 'fdtdx_python':'independent-runtime'}
            config = state/'runtime.json'
            config.write_text(json.dumps(settings))
            def fail(command, **kwargs):
                prefix = Path(command[command.index('--prefix')+1])
                prefix.mkdir()
                (prefix/'partial').write_text('partial attempt')
                raise subprocess.CalledProcessError(1, command)
            with patch.object(managed_setup.shutil,'which',return_value='conda'), patch.object(managed_setup.subprocess,'run',side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):
                    managed_setup.setup_meep(conda='conda',offline=True,data_dir=directory,resources=directory)
            self.assertEqual(json.loads(config.read_text()),settings)
            self.assertEqual(list((state/'runtimes').iterdir()),[])

    def test_failed_native_check_preserves_unrelated_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/'environment-meep-linux-64.lock').write_text('@EXPLICIT\n')
            state = directory/'.local'
            state.mkdir()
            settings = {'fdtdx_python':'independent-runtime'}
            (state/'runtime.json').write_text(json.dumps(settings))
            with patch.object(managed_setup.shutil,'which',return_value='conda'), patch.object(managed_setup.subprocess,'run'), patch.object(managed_setup,'_check_runtime',side_effect=RuntimeError('native smoke failed')):
                with self.assertRaisesRegex(RuntimeError, 'native smoke failed'):
                    managed_setup.setup_meep(conda='conda',offline=True,data_dir=directory,resources=directory)
            self.assertEqual(json.loads((state/'runtime.json').read_text()),settings)
