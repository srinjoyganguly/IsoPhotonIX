"""Compare the two multiport material translations on the identical straight-guide mode."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT/'backend').is_dir():
    sys.path.insert(0,str(ROOT))
from scripts.calibrate_fdtdx_multiport import _ports_runtime, multiport_model, port_basis, yee_material


def measure():
    """Use the same port solver, finite aperture, wavelength and phase gauge for each translation."""
    _,jnp,fdtdx,target=_ports_runtime('cpu')
    model=multiport_model('splitter')
    frequencies=np.array([1/1.55])
    rows=[]
    for resolution in (12,16,20,24,32):
        ny,nz=[round(size*resolution) for size in model['aperture']]
        dx=1/resolution
        y=(np.arange(ny)+.5-ny/2)*dx
        z=(np.arange(nz)+.5-nz/2)*dx
        mask=(y[:,None]>=-model['width']/2)&(y[:,None]<model['width']/2)&(z>=-model['height']/2)&(z<model['height']/2)
        epsilon=np.where(mask,model['core_index']**2,model['cladding_index']**2).astype(np.float32)[None,:,:]
        sim=fdtdx.SimulationConfig(time=1e-15,resolution=dx*1e-6,backend='cpu',dtype=jnp.float32,courant_factor=.99)
        row={'resolution':resolution}
        for name in ('legacy-x','yee-aligned'):
            eps=yee_material(epsilon,name)
            basis,_=port_basis(jnp.asarray(1/eps),frequencies,frequencies*299792458e6,dx,sim.time_step_duration,
                               model['cladding_index'],model['core_index'])
            row[name]=float(basis['mode_neff'][0].real)
        rows.append(row)
    return {'actual_device':str(target),'wavelength_um':1.55,'width_um':model['width'],'height_um':model['height'],
            'core_index':model['core_index'],'cladding_index':model['cladding_index'],'aperture_um':model['aperture'],
            'rows':rows,'scope':'FDTDX local straight-guide mode basis. A source-independent time-domain phase measurement is a separate check.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='new JSON report file')
    args=parser.parse_args()
    data=measure()
    with args.output.open('x') as out:
        json.dump(data,out,indent=2,allow_nan=False)
        out.write('\n')
