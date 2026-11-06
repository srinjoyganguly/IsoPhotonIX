"""Check both signs of the physical quasi-TE source and projection frames on the native port grids."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.adapters.fdtdx_ports3d_adapter import collocate_mode, require_quasi_te_mode
from scripts.calibrate_layout_ports3d import (
    FREQ, LIGHT, OXIDE, SILICON, WAVELENGTHS, canonical_source_profile, file_digest, frame, grid,
    layout, local_phasors, modal_amplitudes, permittivity, port_basis, _ports_runtime,
)


def measure():
    _,jnp,_,target = _ports_runtime('cpu')
    from fdtdx.core.physics.modes import compute_mode
    design = layout.default_planar_port_layout()
    rows = []
    for mesh in (12,16):
        _,ports = grid(design,mesh)
        eps = permittivity(design,mesh,0,False)
        dt = .99/(np.sqrt(3)*mesh*LIGHT*1e6)
        for p in ports:
            axis = p['axis']
            vol = tuple(slice(s,s+n) for s,n in zip(p['start'],p['shape']))
            section = eps[(slice(None),*vol)]
            basis,details = port_basis(jnp.asarray(1/frame(section,axis)),FREQ,FREQ*LIGHT*1e6,1/mesh,dt,OXIDE,SILICON)
            for sign in (1,-1):
                native_e,native_h,n = compute_mode(frequency=LIGHT*1e6/1.55,inv_permittivities=jnp.asarray(1/section),
                    inv_permeabilities=1.,resolution=1e-6/mesh,direction='+' if sign==1 else '-',filter_pol='te')
                e,h,_ = canonical_source_profile(frame(np.asarray(native_e),axis),frame(np.asarray(native_h),axis,True))
                flux = float(.5*(e[1]*h[2]-e[2]*h[1]).sum())
                if flux*sign<=0:
                    raise ValueError('Modal Poynting vector disagrees with the requested native source direction.')
                ec,hc = collocate_mode(e,h,2*np.pi*n.real/(mesh*1.55),2*np.pi*LIGHT*1e6*dt/1.55)
                global_e = frame(ec[:,None],axis)
                global_h = frame(hc[:,None],axis,True)
                native_phasor = np.concatenate((global_e,global_h))[None]
                overlaps,power = modal_amplitudes(local_phasors(native_phasor,axis),basis['mode_e'][1:2],
                                                   basis['mode_h'][1:2],1/mesh**2)
                signal = overlaps[0,int(sign<0)]
                leakage = abs(overlaps[0,int(sign>0)])/abs(signal)
                if leakage>1e-5 or abs(signal/np.sqrt(power[0])-1)>1e-5:
                    raise ValueError('Source profile projects onto the wrong direction or phase.')
                rows.append({'mesh':mesh,'axis':'xy'[axis],'direction':sign,'basis_neff':details['neff_real'],
                             'source_neff':float(np.real(n)),'signed_source_flux_native_units':flux,
                             'source_transverse_fraction':require_quasi_te_mode(e),
                             'reciprocal_projection_wrong_direction_ratio':float(leakage),
                             'max_basis_edge_fraction':max(details['edge_electric_l2_fraction'])})
    return {'study':'FDTDX signed transverse source and reciprocal-overlap checks','version':importlib.metadata.version('fdtdx'),
            'device':str(target),'wavelength_um':WAVELENGTHS.tolist(),'rows':rows,
            'scope':'Native CPU cross-section eigensolves and projections only. Whole-device spectra and time evolution use separately injected NVIDIA fields.',
            'sources':{f:file_digest(ROOT/f) for f in ('scripts/verify_layout_ports3d_frames.py','scripts/calibrate_layout_ports3d.py',
                'backend/adapters/fdtdx_ports3d_adapter.py','backend/adapters/fdtdx_ports_adapter.py')}}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output path.')
    args.output.write_text(json.dumps(measure(),indent=2,allow_nan=False)+'\n')
