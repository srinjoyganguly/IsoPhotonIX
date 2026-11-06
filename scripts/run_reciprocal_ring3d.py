"""Run the reciprocal loaded ring with a shared checkpoint and one narrowly scoped allocator retry."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from scripts.calibration_checkpoint import MeasurementCheckpoints


def new_verified_reference(cache, before):
    for path in set(cache.glob('reference-*/receipt.json'))-before:
        receipt=json.loads(path.read_text())
        spec=receipt['specification']
        if (spec['label']=='reference' and path.parent.name=='reference-'+receipt['signature'][:20]
                and receipt['report']['time_steps']>0
                and MeasurementCheckpoints.json_hash(spec)==receipt['signature']
                and MeasurementCheckpoints.json_hash(receipt['report'])==receipt['report_sha256']
                and MeasurementCheckpoints.file_hash(path.parent/'arrays.npz')==receipt['arrays_sha256']):
            performance=receipt['report'].get('performance',{})
            return dict(checkpoint_signature=receipt['signature'],arrays_sha256=receipt['arrays_sha256'],
                        acquisition_id=performance.get('acquisition_id'),synchronized_step_seconds=performance.get('synchronized_step_seconds'))
    return None


def child(command):
    fresh=False
    complete=False
    refused=False
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as proc:
        try:
            for line in proc.stdout:
                print(line,end='',flush=True)
                fresh |= bool(re.fullmatch(r'reference \d+/\d+ samples',line.strip()))
                complete |= bool(re.fullmatch(r'reference complete cuda:\d+',line.strip()))
                refused |= bool(re.fullmatch(r'MemoryError: Fresh ring stage requires \d+ MiB free including reserve; measured \d+ MiB\.',line.strip()))
            code=proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    return code,fresh and complete and refused


def run(argv):
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--ring-interface')
    known,_=parser.parse_known_args(argv)
    if known.ring_interface not in (None,'reciprocal-wall'):
        raise ValueError('This orchestration entry is for the reciprocal-wall ring model.')
    args=list(argv)
    if known.ring_interface is None:
        args+=['--ring-interface','reciprocal-wall']
    workspace=TemporaryDirectory(prefix='photonics-ring-stages-') if known.checkpoint is None else nullcontext(known.checkpoint)
    with workspace as selected:
        cache=Path(selected)
        if known.checkpoint is None:
            args+=['--checkpoint',str(cache)]
        before=set(cache.glob('reference-*/receipt.json'))
        command=[sys.executable,'-m','scripts.calibrate_loaded_ring3d',*args]
        code,recoverable=child(command)
        original=new_verified_reference(cache,before) if code and recoverable else None
        if original:
            print('The reference was saved. Releasing its GPU allocator and rechecking memory in a fresh process.',flush=True)
            code,_=child(command)
            if not code and known.output is not None and (known.output/'report.json').is_file():
                info=dict(child_processes=2,first_attempt='Fresh reference finished; native memory guard refused device allocation.',
                          second_attempt='Fresh CUDA process rechecked admission and verified the reference checkpoint.',
                          reference_computed_before_retry=original)
                (known.output/'ring-stage-recovery.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
        return code


if __name__=='__main__':
    raise SystemExit(run(sys.argv[1:]))
