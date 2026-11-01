import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.jobs import JobManager
from backend.model import WaveguideConfig


class JobLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = JobManager(Path(self.temp.name))

    def tearDown(self):
        self.manager.shutdown()
        self.temp.cleanup()

    def wait_terminal(self, job_id):
        for _ in range(100):
            job = self.manager.get(job_id)
            if job["status"] not in ("queued", "running"):
                return job
            time.sleep(0.02)
        self.fail("Worker did not finish")

    def test_crashed_worker_is_failed_without_result(self):
        with patch.object(self.manager, "_command", return_value=[sys.executable, "-c", "raise RuntimeError('deliberate worker failure')"]):
            job = self.manager.submit(WaveguideConfig().model_dump())
            final = self.wait_terminal(job["id"])
        self.assertEqual(final["status"], "failed")
        self.assertNotIn("result", final)
        self.assertIn("deliberate worker failure", final["logs"])
        self.assertIn("no fallback", final["error"])

    def test_zero_exit_without_output_is_not_success(self):
        with patch.object(self.manager, "_command", return_value=[sys.executable, "-c", "pass"]):
            job = self.manager.submit(WaveguideConfig().model_dump())
            final = self.wait_terminal(job["id"])
        self.assertEqual(final["status"], "failed")
        self.assertIn("without numerical", final["error"])

    def test_cancel_reaps_worker_and_stays_cancelled(self):
        with patch.object(self.manager, "_command", return_value=[sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"]):
            job = self.manager.submit(WaveguideConfig().model_dump())
            for _ in range(100):
                if self.manager.get(job["id"])["status"] == "running":
                    break
                time.sleep(.01)
            with self.assertRaisesRegex(RuntimeError, "already active"):
                self.manager.submit(WaveguideConfig().model_dump())
            cancelled = self.manager.cancel(job["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertNotIn("result", cancelled)
        time.sleep(.05)
        self.assertEqual(self.manager.get(job["id"])["status"], "cancelled")

    def test_restart_does_not_claim_running_job_completed(self):
        directory = Path(self.temp.name)/"a"
        directory.mkdir()
        (directory/"job.json").write_text('{"id":"a","status":"running"}')
        restored = JobManager(self.temp.name)
        self.assertEqual(restored.get("a")["status"], "failed")
