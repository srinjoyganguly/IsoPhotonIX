"""Compare physical S31 with the ring-induced excess phase over its measured straight bus transit."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.phase_component_comparison import distribution, file_sha, fixed_mode_ring_spectrum, measured_beta, native_data


def measure(directory):
    registered_path=directory/'v014-ring-phase-reciprocal.json'
    dataset=json.loads(registered_path.read_text())
    rows=dataset['native_samples']
    if [r['resolution'] for r in rows] != [20,24]:
        raise ValueError('This diagnostic requires the matched 20- and 24-cell reciprocal-wall acquisitions.')
    runs=[]
    for row in rows:
        receipt,arrays=native_data(directory/row['native_receipt'])
        with arrays:
            if receipt['arrays_sha256']!=row['arrays_sha256'] or receipt['physical_model']['ring_interface']!='reciprocal-wall':
                raise ValueError('Registered spectrum and physical reciprocal-wall acquisition must match.')
            w=arrays['wavelength_um']
            central=row['central_vendor_mode_check']['port_overlaps'][0]['wavelength_um']
            fixed=row['single_central_basis_band']
            s=np.asarray(fixed['registered_native_real'])+1j*np.asarray(fixed['registered_native_imag'])
            waves=arrays['basis_wavelength_um']
            result=fixed_mode_ring_spectrum(arrays['reference_time_um_c'],arrays['reference_projections'],waves,central,w)
            straight=result[:,1,0]/result[:,0,0]
            p0,p1=row['planes']['native_port_x_um'][:2]
            neff=np.interp(w,waves,arrays['reference_mode_neff'][0])
            beta=measured_beta(straight,w,neff,p1-p0)
            target=row['planes']['target_abs_x_um']
            straight*=np.exp(1j*beta*(2*target-(p1-p0)))
            np.testing.assert_allclose(np.angle(straight*np.exp(-2j*target*beta)),0,atol=2e-13)
            stable=np.asarray(fixed['native_valid_bins']) & np.asarray(fixed['native_phase_ok'])[:,1]
            runs.append({'w':w,'s':s,'straight':straight,'stable':stable,'sha256':receipt['arrays_sha256']})
    before,after=runs
    w=before['w']
    np.testing.assert_array_equal(w,after['w'])
    a,b=before['s'],after['s']
    mask=before['stable'] & after['stable'] & (w>=1.5) & (w<=1.6)
    passband=mask & (abs(a[:,1])**2>=.9) & (abs(b[:,1])**2>=.9) & (abs(a[:,2])**2<=.05) & (abs(b[:,2])**2<=.05)
    transit=before['straight']*after['straight'].conj()
    total=a[:,1]*b[:,1].conj()
    excess=(a[:,1]/before['straight'])*(b[:,1]/after['straight']).conj()
    np.testing.assert_allclose(np.angle(excess*(transit/abs(transit))*total.conj()),0,atol=1e-12)
    channels={'physical_S31_rad':np.angle(total),'straight_between_port_planes_rad':np.angle(transit),
              'ring_excess_over_straight_rad':np.angle(excess)}
    center=int(np.argmin(abs(w-1.55)))
    return {'native_mesh_order':[20,24],'native_arrays_sha256':[r['sha256'] for r in runs],
            'registered_source_sha256':file_sha(registered_path),'signed_convention':'Principal angle of 20-cell quantity times conjugate of 24-cell quantity.',
            'target_port_planes_um':[-4.2,4.2],'wavelength_um':w.tolist(),'qualified_bins':mask.tolist(),
            'passband_bins':passband.tolist(),'passband_rule':'On both meshes: through power >=0.90 and drop power <=0.05; retain the independent partial-time and signal gates.',
            'all_qualified':{k:distribution(v[mask]) for k,v in channels.items()},
            'passband_only':{k:distribution(v[passband]) for k,v in channels.items()},
            'at_1550_nm':{k:float(v[center]) if mask[center] else None for k,v in channels.items()},
            'signed_radians_by_bin':{k:[float(x) if ok else None for x,ok in zip(v,mask,strict=True)] for k,v in channels.items()},
            'scope':'S31 is always the physical through coefficient at the filed planes. Dividing by the independently measured straight-guide transfer defines a separate ring-induced excess phase for this mesh diagnostic. It neither changes the reported S31 nor registers the commercial poles.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('docs/verification'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.write_text(json.dumps(measure(args.directory),indent=2,allow_nan=False)+'\n')
