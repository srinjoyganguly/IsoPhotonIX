"""Independent straight, mesh, aperture, boundary and duration controls for one launch."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_multimode import FREQUENCIES, OFFSET, PHASE_VERSION, checkpointed_stage


def main():
    import meep as mp
    mp.verbosity(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finite',action='store_true')
    parser.add_argument('--control',choices=['straight','aperture','pml','time','mesh'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():
        parser.error('The finished control already exists.')
    aperture=3. if args.finite else 4.
    mesh=16 if args.finite else 48
    if args.control=='mesh':mesh=20 if args.finite else 64
    offset=0. if args.control=='straight' else OFFSET
    if args.control=='aperture':aperture-=.5 if args.finite else .75
    pml=1.25 if args.control=='pml' else 1.
    after=110. if args.control=='time' else 80.
    stages=[]
    for ref in (True,False):
        stages.append(checkpointed_stage(mp,(mesh,offset,aperture,pml,after,args.finite),0,1,ref,args.output/'stages'))
    ref,dev=stages[0][0],stages[1][0]
    incident=ref[0,0,0]
    s=np.vstack([(dev[0,:,1]-ref[0,:,1])/incident,dev[1,:,0]/incident]).T
    require_finite = np.isfinite(s).all() and np.isfinite(incident).all()
    if not require_finite:raise ValueError('Invalid measured column.')
    report={'case':'multimode-3d' if args.finite else 'multimode','control':args.control,'phase_convention_version':PHASE_VERSION,
            'resolution':mesh,'pml_um':pml,'aperture_um':aperture,'after_source_um_over_c':after,
            'offset_um':offset,'input':'left TE0','solver':'Meep CPU','version':mp.__version__,
            'wavelength_um':(1/FREQUENCIES).tolist(),'power':(abs(s)**2).tolist(),
            's_real':s.real.tolist(),'s_imag':s.imag.tolist()}
    np.savez_compressed(args.output/'arrays.npz',s=s,reference=ref,device=dev,incident=incident)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
