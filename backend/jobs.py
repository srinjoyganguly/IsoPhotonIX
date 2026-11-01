import json
import os
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .runtime import ROOT, clean_log, interpreter, worker_env


class JobManager:
    def __init__(self, directory=None):
        self.directory = Path(directory or ROOT/"runs")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.processes = {}
        for path in self.directory.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if job["status"] in ("queued", "running"):
                    job.update(status="failed", error="Service restarted before completion. Run again.")
                self.jobs[job["id"]] = job
                self._save(job)
            except (ValueError, KeyError, OSError):
                continue

    def _save(self, job):
        path = self.directory/job["id"]/"job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2, allow_nan=False))
        temporary.replace(path)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = dict(self.jobs[job_id])
            log = self.directory/job_id/"worker.log"
            job["logs"] = clean_log(log.read_text(errors="replace")[-18000:]) if log.exists() else ""
            if job.get('config',{}).get('study_type') == 'waveguide-ports-2d':
                stages = [line[11:].strip() for line in job['logs'].splitlines() if line.startswith('PORT_STAGE=')]
                stage = 'complete' if job['status'] == 'completed' else stages[-1] if stages else 'reference'
                if stage in ('reference','device','normalization','complete'):
                    job['port_progress'] = {'stage':stage,'message':{'reference':'Measuring straight-guide incident reference','device':'Simulating the requested device','normalization':'Checking settling and normalizing signed power','complete':'Reference and device outputs recorded'}[stage]}
            if job.get('config',{}).get('study_type') == 'waveguide-ports-3d':
                stages = [line[11:].strip() for line in job['logs'].splitlines() if line.startswith('PORT_STAGE=')]
                stage = 'complete' if job['status'] == 'completed' else stages[-1] if stages else 'reference_forward'
                labels = {'reference_forward':'Reference excitation from port 1', 'device_forward':'Device excitation from port 1',
                          'reference_reverse':'Reference excitation from port 2', 'device_reverse':'Device excitation from port 2',
                          'normalization':'Auditing complex modal amplitudes', 'complete':'Four directional runs recorded'}
                if stage in labels:
                    job['port_progress'] = {'stage':stage,'message':labels[stage]}
            return job

    def submit(self, config):
        with self.lock:
            config = json.loads(json.dumps(config))
            if any(j["status"] in ("queued", "running") for j in self.jobs.values()):
                raise RuntimeError("A simulation is already active. Wait for it or cancel it before starting another.")
            job_id = uuid.uuid4().hex
            output = self.directory/job_id
            output.mkdir()
            (output/"config.json").write_text(json.dumps(config, indent=2))
            job = {"id": job_id, "status": "queued", "config": config,
                   "created_at": datetime.now(timezone.utc).isoformat(), "error": None}
            self.jobs[job_id] = job
            self._save(job)
            threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
            return dict(job)

    def _command(self, config, output):
        return [interpreter(config["solver"]), "-m", "backend.worker", str(output/"config.json"), str(output)]

    def _run(self, job_id):
        started = time.monotonic()
        output = self.directory/job_id
        try:
            with self.lock:
                job = self.jobs[job_id]
                if job["status"] == "cancelled":
                    return
                config = job["config"]
                log = (output/"worker.log").open("w")
                try:
                    process = subprocess.Popen(self._command(config, output), cwd=ROOT,
                        env=worker_env(config["solver"], config["device"]), stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                finally:
                    log.close()
                self.processes[job_id] = process
                job["status"] = "running"
                self._save(job)
            try:
                limit = 1200 if config.get('study_type') == 'waveguide-ports-3d' else 300
                code = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError(f"Simulation exceeded its {limit//60}-minute local worker limit and was stopped.")
            with self.lock:
                if job["status"] == "cancelled":
                    return
                if code != 0:
                    raise RuntimeError(f"Solver worker exited with code {code}. See the run log for the cause; no fallback was attempted.")
                if not (output/"arrays.npz").exists() or not (output/"metadata.json").exists():
                    raise RuntimeError("Solver exited without numerical results.")
                job.update(status="completed", result=json.loads((output/"metadata.json").read_text()),
                           elapsed_s=round(time.monotonic()-started, 3))
                self._save(job)
        except Exception as exc:
            with self.lock:
                job = self.jobs[job_id]
                if job["status"] != "cancelled":
                    job.update(status="failed", error=clean_log(str(exc)), elapsed_s=round(time.monotonic()-started, 3))
                    self._save(job)
        finally:
            with self.lock:
                self.processes.pop(job_id, None)

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] not in ("queued", "running"):
                return self.get(job_id)
            process = self.processes.get(job_id)
            if process and process.poll() is None:
                # Kill the isolated process group, including any solver children.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            job.update(status="cancelled", error="Cancelled by user. Partial arrays are not a completed result.")
            self._save(job)
            return self.get(job_id)

    def shutdown(self):
        for job_id in list(self.jobs):
            if self.jobs[job_id]["status"] in ("queued", "running"):
                self.cancel(job_id)
