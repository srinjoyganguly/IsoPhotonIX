"""Reproduce a configuration in an isolated, explicitly selected solver worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.study3d import parse_study
from backend.runtime import ROOT, interpreter, worker_env

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
settings = json.loads(args.config.read_text())
if isinstance(settings, dict) and settings.get('study_type') == 'calibrated-component':
    from backend.calibrations import Calibration, run_config
    calibration = Calibration.model_validate(settings)
    raise SystemExit(run_config(calibration.model_dump(), args.output))
config = parse_study(settings).model_dump()
args.output.mkdir(parents=True, exist_ok=False)
output = args.output.resolve()
(output/"config.json").write_text(json.dumps(config, indent=2))
run = subprocess.run([interpreter(config["solver"], config["device"]), "-m", "backend.worker", str(output/"config.json"), str(output)],
                     cwd=ROOT, env=worker_env(config["solver"], config["device"]))
raise SystemExit(run.returncode)
