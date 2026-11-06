"""Measure one finite-height bare-ring order on meshes aligned to both silicon faces."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .calibrate_ring import CASES, isolated_pole
else:
    from calibrate_ring import CASES, isolated_pole


def main():
    import meep as mp
    mp.verbosity(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=tuple(CASES),required=True)
    parser.add_argument('--order',type=int,required=True)
    parser.add_argument('--resolution',type=int,choices=(50,100),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.order<=80:
        parser.error('choose a positive azimuthal order no greater than 80')
    if args.output.exists():
        parser.error('choose a fresh output directory')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    case=CASES[args.case]
    arrays={}
    row=isolated_pole(case,args.order,args.resolution,.8,.8,arrays,'pole',args.case=='tidy3d')
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        stage=Path(temp)/'result';stage.mkdir()
        raw=stage/'arrays.npz'
        np.savez_compressed(raw,**arrays)
        record=dict(case=args.case,solver='Meep CPU',version=mp.__version__,result=row,
            passed=row['pole_repeat_passed'],angular_order=args.order,physical_geometry=case,
            vertical_core_cells=round(case['height']*args.resolution),
            vertical_domain_cells=round((case['height']+3.2)*args.resolution),
            arrays_file='arrays.npz',arrays_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            scope='One bare-ring pole with 0.8 µm radial/vertical clearance and 0.8 µm PML. The vertical cell contains an integer number of grid cells between the two physical silicon faces.')
        (stage/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        stage.rename(args.output)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
