"""Build the public GPU timing ledger from the checked-in stage receipts."""
import hashlib
import json
from pathlib import Path

from backend.calibration_performance import current_totals, stage_rows, timing_csv

BASE=Path(__file__).resolve().parents[1]/'docs/verification'


def collect(base):
    rows=[]
    hashes={}
    for path in sorted(base.glob('v0*.json')):
        if 'index' in path.name or 'performance' in path.name:
            continue
        data=path.read_bytes()
        report=json.loads(data)
        if not isinstance(report,dict):
            continue
        record=stage_rows(report,path.name)
        if record:
            hashes[path.name]=hashlib.sha256(data).hexdigest()
            rows.extend(record)
    return {'schema':1,'source_sha256':hashes,'totals':current_totals(rows),'rows':rows}


if __name__=='__main__':
    ledger=collect(BASE)
    (BASE/'v014-gpu-performance.json').write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
    (BASE/'v014-gpu-performance.csv').write_text(timing_csv(ledger['rows']))
    print(len(ledger['source_sha256']), 'reports;', len(ledger['rows']), 'stage uses;', ledger['totals'])
