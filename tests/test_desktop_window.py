"""Desktop launcher ownership and persistent profile selection."""
import fcntl
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from isophotonix import desktop


class DesktopWindowTests(unittest.TestCase):
    def test_invalid_port_and_missing_browser_reject_before_launch(self):
        with tempfile.TemporaryDirectory() as root, patch.object(desktop.subprocess, 'Popen') as popen:
            for port in (0, 65536):
                with self.assertRaisesRegex(ValueError, 'port'):
                    desktop.launch_desktop(root, port)
            with self.assertRaisesRegex(ValueError, 'browser executable'):
                desktop.launch_desktop(root, 8765, Path(root) / 'missing-chrome')
            popen.assert_not_called()

    def test_native_profile_belongs_to_its_own_store(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root)
            first = desktop._profile(data / 'first', Path('/opt/chrome'))
            second = desktop._profile(data / 'second', Path('/opt/chrome'))
            self.assertEqual(first, desktop._profile(data / 'first', Path('/opt/chrome')))
            self.assertNotEqual(first, second)
            self.assertTrue(Path(first).is_dir())

    def test_wsl_profile_is_stable_and_reuses_existing_browser_drafts(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, WSL_DISTRO_NAME='test-distribution'), patch.object(
                desktop.subprocess, 'check_output', side_effect=[r'\\wsldistro\root\data', r'C:\Users\Example\AppData\Local', root] * 4):
            args = (Path('/meep/data'), Path('/mnt/c/chrome.exe'))
            profile = desktop._profile(*args)
            self.assertEqual(desktop._profile(*args), profile)
            self.assertTrue(profile.startswith(r'C:\Users\Example\AppData\Local\IsoPhotonIX\Profiles' + '\\'))
            key = profile.rsplit('\\', 1)[-1]
            self.assertEqual(len(key), 24)
            (Path(root) / 'Photonics Workbench' / 'Profiles' / key).mkdir(parents=True)
            self.assertEqual(desktop._profile(*args), profile.replace('IsoPhotonIX', 'Photonics Workbench'))
            (Path(root) / 'IsoPhotonIX' / 'Profiles' / key).mkdir(parents=True)
            self.assertEqual(desktop._profile(*args), profile)

    def test_occupied_store_is_left_alone(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / '.local'
            state.mkdir()
            with (state / 'web.lock').open('w') as lock, patch.object(desktop.subprocess, 'Popen') as popen:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, 'owns this data directory'):
                    desktop.launch_desktop(root, 8765, sys.executable)
                popen.assert_not_called()

    def test_occupied_port_is_left_alone(self):
        with tempfile.TemporaryDirectory() as root, socket.socket() as neighbor:
            neighbor.bind(('127.0.0.1', 0))
            neighbor.listen()
            port = neighbor.getsockname()[1]
            with patch.object(desktop.subprocess, 'Popen') as popen, self.assertRaisesRegex(RuntimeError, 'unavailable'):
                desktop.launch_desktop(root, port, sys.executable)
            popen.assert_not_called()
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                pass

    def test_normal_closure_never_terminates_another_process(self):
        window = Mock()
        window.poll.return_value = 0
        with patch.object(desktop.subprocess, 'run') as control:
            desktop._close_window(window, Path('/mnt/c/chrome.exe'), 'http://127.0.0.1:8765', 'test')
            control.assert_not_called()
        window.terminate.assert_not_called()
