"""Compile the photonic GPU timestep on abstract shapes to inspect memory without running it."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.calibrate_finite_grating import native_stage,settings
from scripts.grating_cuda_identity import cache_salt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',choices=['compact','tidy3d','lumerical'],required=True)
    p.add_argument('--resolution',type=int,choices=(12,13,14,15,16,18,20),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grating-spectrum',choices=('standard','dense'),default='standard')
    p.add_argument('--grating-material-model',choices=('volume-average','direct-yee'),default='volume-average')
    p.add_argument('--reference',action='store_true',help='Compile the incident simulation instead of the device.')
    p.add_argument('--grating-domain',choices=('full','mirror-y'),default='full')
    p.add_argument('--observe-one-step',action='store_true',help='measure runtime allocation on an already executed mesh')
    args=p.parse_args()
    if args.observe_one_step and args.resolution not in (12, 14):
        p.error('One-step observations require an already executed mesh.')
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['PHOTONICS_GRATING_DEVICE_SALT']=cache_salt(Path(tmp)).hex()
        if args.grating_spectrum=='dense' and args.resolution!=12:
            p.error('The dense monitor grid requires a dedicated 12-cell memory admission record.')
        m=settings(args.geometry,12,args.grating_material_model,args.grating_spectrum,args.grating_domain)
        m['resolution']=args.resolution
        if args.observe_one_step and args.resolution == 14 and args.grating_domain == 'full':
            sampling = '-direct' if args.grating_material_model == 'direct-yee' else ''
            version = 'v013' if sampling else 'v012'
            receipt = ROOT/'docs/verification'/f'{version}-grating3d-{args.geometry}{sampling}-r14.json'
            if not receipt.exists() or not json.loads(receipt.read_text())['passed']:
                p.error('14-cell resource observation requires its prior qualified full-run receipt.')
        report=native_stage(m,args.reference,memory_plan=not args.observe_one_step,observe_one_step=args.observe_one_step)
        report.update(grating_domain=args.grating_domain,grating_spectrum=args.grating_spectrum,grating_material_model=args.grating_material_model,
                      stage='reference' if args.reference else 'device')
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))
