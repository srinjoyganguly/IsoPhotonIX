"""Contract/integrity tests with explicit small synthetic fixtures, not solver evidence."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
import numpy as np

from backend.app import app
from backend.jobs import JobManager
from backend.model import WaveguideConfig


class PassportAndFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.manager = JobManager(self.directory)
        self.output = self.directory / "fixture"
        self.output.mkdir()
        self.config = WaveguideConfig().model_dump()
        self.metadata = {"solver": "meep", "actual_device": "CPU", "config": copy.deepcopy(self.config),
                         "versions": {"meep": "test-fixture"}, "source": {"type": "<script>unsafe()</script>"},
                         "actual_duration_fs": 10.0, "field_units": "test units"}
        (self.output / "config.json").write_text(json.dumps(self.config))
        (self.output / "metadata.json").write_text(json.dumps(self.metadata))
        self.values = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.frame_values = np.stack([self.values * 0, -self.values * 2, self.values])
        self.save_arrays()
        self.manager.jobs["fixture"] = {"id": "fixture", "status": "completed", "created_at": "test time",
                                        "config": copy.deepcopy(self.config), "result": copy.deepcopy(self.metadata)}
        self.patcher = patch("backend.app.manager", self.manager)
        self.patcher.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.patcher.stop()
        self.temp.cleanup()

    def save_arrays(self, include_frames=True, frames=None):
        data = dict(field=self.values, epsilon=np.ones_like(self.values), x_um=np.arange(3), y_um=np.arange(2))
        if include_frames:
            data.update(frames=self.frame_values if frames is None else frames, times_fs=np.array([0., 5., 10.]))
        np.savez_compressed(self.output / "arrays.npz", **data)

    def test_passport_hashes_saved_snapshot_and_escaped_offline_html(self):
        response = self.client.get("/api/jobs/fixture/passport")
        self.assertEqual(response.status_code, 200)
        passport = response.json()
        self.assertEqual(passport["config"], self.config)
        self.assertEqual(passport["actual_device"], "CPU")
        self.assertTrue(all(check["passed"] for check in passport["checks"]))
        exported = self.client.get("/api/jobs/fixture/export")
        self.assertEqual(exported.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            for name, item in manifest["files"].items():
                raw = archive.read(name)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])
                self.assertEqual(len(raw), item["size_bytes"])
            self.assertEqual(passport["array_sha256"], manifest["files"]["arrays.npz"]["sha256"])
            html = archive.read("passport.html").decode()
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("No normalized transmission", html)

    def test_snapshot_disagreement_is_visible_without_relabeling_saved_config(self):
        self.manager.jobs["fixture"]["config"]["core_width_um"] = 0.9
        passport = self.client.get("/api/jobs/fixture/passport").json()
        self.assertEqual(passport["config"]["core_width_um"], 0.6)
        check = next(item for item in passport["checks"] if item["label"] == "Consistent saved configuration")
        self.assertFalse(check["passed"])

    def test_frame_values_and_times_are_exact_and_final_snapshot_is_unchanged(self):
        summary = self.client.get("/api/jobs/fixture/frames").json()
        self.assertEqual(summary, {"count": 3, "times_fs": [0., 5., 10.], "global_peak_abs": 10.})
        frame = self.client.get("/api/jobs/fixture/frames/1").json()
        np.testing.assert_array_equal(frame["field"], self.frame_values[1])
        self.assertEqual(frame["time_fs"], 5.)
        np.testing.assert_array_equal(self.client.get("/api/jobs/fixture/field").json()["field"], self.values)
        for index in (-1, 3):
            self.assertEqual(self.client.get(f"/api/jobs/fixture/frames/{index}").status_code, 404)

    def test_changed_frame_file_invalidates_cache(self):
        self.client.get("/api/jobs/fixture/frames")
        changed = self.frame_values * 3
        self.save_arrays(frames=changed)
        self.assertEqual(self.client.get("/api/jobs/fixture/frames").json()["global_peak_abs"], 30.)

    def test_missing_frames_preserve_final_snapshot_and_passport(self):
        self.save_arrays(include_frames=False)
        self.assertEqual(self.client.get("/api/jobs/fixture/frames").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/fixture/frames/0").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/fixture/field").status_code, 200)
        self.assertEqual(self.client.get("/api/jobs/fixture/passport").json()["numerical"]["frame_count"], 0)

    def test_invalid_frames_are_rejected_and_zero_fields_are_reported(self):
        invalid = self.frame_values.copy()
        invalid[0, 0, 0] = np.nan
        self.save_arrays(frames=invalid)
        self.assertEqual(self.client.get("/api/jobs/fixture/frames").status_code, 409)
        self.values[:] = 0
        self.save_arrays()
        passport = self.client.get("/api/jobs/fixture/passport").json()
        check = next(item for item in passport["checks"] if item["label"] == "Nonzero final field")
        self.assertFalse(check["passed"])

    def test_corrupt_result_archive_returns_friendly_error(self):
        (self.output / "arrays.npz").write_bytes(b"corrupt archive")
        for endpoint in ("frames", "passport", "export"):
            response = self.client.get(f"/api/jobs/fixture/{endpoint}")
            self.assertEqual(response.status_code, 409)
            self.assertIn("invalid", response.json()["detail"])

    def test_noncompleted_jobs_cannot_present_passports_frames_or_exports(self):
        for status in ("running", "failed", "cancelled"):
            self.manager.jobs["fixture"]["status"] = status
            for endpoint in ("passport", "frames", "frames/0", "export"):
                with self.subTest(status=status, endpoint=endpoint):
                    self.assertEqual(self.client.get(f"/api/jobs/fixture/{endpoint}").status_code, 409)
