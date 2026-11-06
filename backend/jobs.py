import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .runtime import ROOT, clean_log, data_root, interpreter, worker_env


class JobManager:
    def __init__(self, directory=None, *, defer_recovery=False):
        self.directory = Path(directory or data_root()/"runs")
        self.lock = threading.RLock()
        self.jobs = {}
        self.processes = {}
        self._deferred = defer_recovery
        self._store_lock = None
        if defer_recovery:
            self._reload(recover=False)
        else:
            # Standalone callers historically recover on construction.
            with self._claim_store():
                self._reload(recover=True)

    def _claim_store(self):
        import fcntl
        self.directory.mkdir(parents=True, exist_ok=True)
        guard = (self.directory / ".service.lock").open("a")
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            guard.close()
            raise RuntimeError(f"Another workbench service is using the run store {self.directory}.") from exc
        except BaseException:
            guard.close()
            raise
        return guard

    def _reload(self, *, recover):
        jobs = {}
        for path in self.directory.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if job["id"] != path.parent.name:
                    continue
                interrupted = job["status"] in ("queued", "running")
            except (ValueError, KeyError, OSError, TypeError):
                continue
            if recover and interrupted:
                job.update(status="failed", error="Service restarted before completion. Run again.")
                self._save(job)
            jobs[job["id"]] = job
        self.jobs = jobs

    def start(self):
        """Claim the store before treating any queued or running record as interrupted."""
        with self.lock:
            if self._store_lock is not None:
                raise RuntimeError("This workbench service already owns the run store.")
            guard = self._claim_store()
            try:
                self._reload(recover=True)
            except BaseException:
                guard.close()
                raise
            self._store_lock = guard

    def _require_active(self):
        if self._deferred and self._store_lock is None:
            raise RuntimeError("The workbench service must own the run store before changing jobs.")

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
            if job.get('config',{}).get('study_type') in ('waveguide-ports-3d','taper-ports-3d'):
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
            self._require_active()
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

    def calibration_can_resume(self, job):
        if job.get('config',{}).get('study_type')!='calibrated-component':
            return False
        if job['status'] not in ('cancelled','failed') or job['id'] in self.processes:
            return False
        output=self.directory/job['id']
        return ((output/'replay/manifest.json').is_file() and (output/'checkpoints').is_dir()
                and not (output/'calculation').exists())

    def resume_calibration(self, job_id):
        with self.lock:
            self._require_active()
            job=self.jobs[job_id]
            if not self.calibration_can_resume(job):
                raise RuntimeError('There is no interrupted calibrated study to resume.')
            if any(record['status'] in ('queued','running') for record in self.jobs.values()):
                raise RuntimeError('Another simulation is active.')
            job.update(status='queued',error=None,attempt=job.get('attempt',1)+1)
            self._save(job)
            threading.Thread(target=self._run,args=(job_id,),daemon=True).start()
            return dict(job)

    def _command(self, config, output):
        if config.get('study_type') == 'calibrated-component':
            return [sys.executable, '-m', 'backend.calibrations', str(output)]
        return [interpreter(config["solver"], config["device"]), "-m", "backend.worker", str(output/"config.json"), str(output)]

    def _run(self, job_id):
        started = time.monotonic()
        output = self.directory/job_id
        try:
            with self.lock:
                job = self.jobs[job_id]
                if job["status"] == "cancelled":
                    return
                config = job["config"]
                resumed = job.get('attempt',1)>1
                log = (output/"worker.log").open("a" if resumed else "w")
                if resumed:
                    log.write(f"\nResuming saved calibration, attempt {job['attempt']}.\n")
                    log.flush()
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
                limit = 1200 if config.get('study_type') in ('waveguide-ports-3d','taper-ports-3d') else 300
                if config.get('study_type') == 'calibrated-component':
                    limit = 6*60*60
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
                if config.get('study_type') == 'calibrated-component':
                    numerical = output/'calculation'
                    if not (numerical/'arrays.npz').is_file() or not (numerical/'report.json').is_file():
                        raise RuntimeError('Calibration ended without the measured report and arrays.')
                    report=json.loads((numerical/'report.json').read_text())
                    actual=report.get('actual_device')
                    if actual is None and report.get('solver')=='Meep CPU':
                        actual='CPU'
                    result={'case':config['case'],'solver':config['solver'],'device':config['device'],
                            'actual_device':actual,'device_name':report.get('device_name')}
                else:
                    if not (output/"arrays.npz").exists() or not (output/"metadata.json").exists():
                        raise RuntimeError("Solver exited without numerical results.")
                    result = json.loads((output/"metadata.json").read_text())
                job.update(status="completed", result=result, elapsed_s=round(time.monotonic()-started, 3))
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
            self._require_active()
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
        with self.lock:
            if self._deferred and self._store_lock is None:
                return
            try:
                for job_id, job in self.jobs.items():
                    if job["status"] in ("queued", "running"):
                        self.cancel(job_id)
            finally:
                if self._store_lock is not None:
                    self._store_lock.close()
                    self._store_lock = None
