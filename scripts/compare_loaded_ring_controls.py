"""Compare qualified native spectra at fixed wavelengths and common physical port planes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    r=json.loads(path.read_text())
    a=path.parent/r['arrays_file']
    if hashlib.sha256(a.read_bytes()).hexdigest()!=r['arrays_sha256']:
        raise ValueError(f'Archive hash mismatch for {path.name}.')
    with np.load(a,allow_pickle=False) as data:
        wl=data['wavelength_um']
        index=data['device_mode_neff']
        basis=data['basis_wavelength_um']
        native=data['s']
    g=r['physical_model']
    snapped=(g['shape'][0]/2-g['ports'][0][0])/g['resolution']
    desired=g['radius_um']+.76
    # Both distances in each optical path extend when the nominal planes move farther out.
    modal=np.stack([np.interp(wl,basis,row) for row in index],axis=1)
    phase=2*np.pi*(desired-snapped)*(modal[:,0,None]+modal)/wl[:,None]
    return r,wl,native*np.exp(1j*phase),desired,snapped


def compare(a,b):
    old,wl,s1,plane,old_plane=read(a)
    new,wl2,s2,new_plane,new_snapped=read(b)
    if old['vendor']!=new['vendor'] or plane!=new_plane or not np.array_equal(wl,wl2):
        raise ValueError('Cases, desired planes and wavelength sampling must agree.')
    vendor=old['vendor']
    mask=np.array(old['series'][vendor]['valid_bins']) & np.array(new['series'][vendor]['valid_bins'])
    power1=abs(s1)**2;power2=abs(s2)**2
    rows={}
    for i,port in enumerate(('reflection','through','drop','add_side')):
        diff=(power2-power1)[mask,i]
        if not len(diff):
            raise ValueError('No jointly qualified wavelengths.')
        phase_mask=mask & (np.minimum(power1[:,i],power2[:,i])>=.01)
        angle=np.angle(s2[phase_mask,i]*s1[phase_mask,i].conj())
        rows[port]=dict(max_power_difference=float(np.max(abs(diff))),mean_absolute_power_difference=float(np.mean(abs(diff))),
                        max_complex_amplitude_difference=float(np.max(abs(s2[mask,i]-s1[mask,i]))),
                        phase_bins_with_both_powers_above_0_01=int(phase_mask.sum()),
                        max_wrapped_phase_difference_rad=float(np.max(abs(angle))) if len(angle) else None)
    a_fsr,b_fsr=old['measured_drop_fsr_nm'],new['measured_drop_fsr_nm']
    return dict(vendor=vendor,first=a.name,second=b.name,jointly_qualified_wavelengths=int(mask.sum()),
                first_resolution=old['physical_model']['resolution'],second_resolution=new['physical_model']['resolution'],
                first_duration_um_c=old['duration_um_c'],second_duration_um_c=new['duration_um_c'],
                first_fsr_nm=a_fsr,second_fsr_nm=b_fsr,fsr_change_nm=b_fsr-a_fsr if a_fsr and b_fsr else None,
                nominal_plane_distance_from_center_um=plane,first_snapped_plane_um=old_plane,second_snapped_plane_um=new_snapped,
                port_differences=rows,phase_correction='Known guided propagation distance from each sampled plane to its common physical plane. Effective indices from the five independently solved lossless port modes, linearly interpolated in wavelength. No fitted phase or spectral registration.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first',type=Path)
    parser.add_argument('second',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare(args.first,args.second),indent=2,allow_nan=False)+'\n')
