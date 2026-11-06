"""Compare qualified native spectra at fixed wavelengths and common physical port planes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.calibrate_loaded_ring3d import directional_spectrum


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
        reference={key.removeprefix('reference_'):data[key] for key in data.files if key.startswith('reference_')}
        straight=directional_spectrum(reference,1.)
        measured_through=straight[:,1,0]/straight[:,0,0]
    g=r['physical_model']
    snapped=(g['shape'][0]/2-g['ports'][0][0])/g['resolution']
    desired=g['radius_um']+.76
    # Both distances in each optical path extend when the nominal planes move farther out.
    modal=np.stack([np.interp(wl,basis,row) for row in index],axis=1)
    dx=1/g['resolution']
    dt=r['stages']['reference']['time_step_um_c']
    beta=2*np.pi*modal/wl[:,None]
    # A uniform lossless port on this Yee grid advances with a discrete longitudinal wavenumber.
    yee_beta=2/dx*np.arcsin(.5*dx*beta*np.sinc(dt/wl)[:,None])
    length=2*snapped
    residual=np.angle(measured_through*np.exp(-1j*yee_beta[:,0]*length))
    continuum_residual=np.angle(measured_through*np.exp(-1j*beta[:,0]*length))
    # The lower bus is independently excited in the straight reference. Its measured residual
    # sets the propagation constant for that bus; only the discrete modal estimate is used above.
    yee_beta[:,:2]+=residual[:,None]/length
    phase=(desired-snapped)*(yee_beta[:,0,None]+yee_beta)
    control={'reference_separation_um':length,
             'max_wrapped_continuous_mode_phase_error_rad':float(np.max(abs(continuum_residual))),
             'max_wrapped_discrete_mode_phase_error_rad':float(np.max(abs(residual))),
             'at_1550nm_continuous_error_rad':float(continuum_residual[np.argmin(abs(wl-1.55))]),
             'at_1550nm_discrete_error_rad':float(residual[np.argmin(abs(wl-1.55))])}
    return r,wl,native*np.exp(1j*phase),desired,snapped,control


def compare(a,b):
    old,wl,s1,plane,old_plane,first_control=read(a)
    new,wl2,s2,new_plane,new_snapped,second_control=read(b)
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
                port_differences=rows,straight_phase_controls={'first':first_control,'second':second_control},
                phase_correction='Known distance to common physical planes. The lower bus uses its measured complex straight transmission with phase branch anchored to discrete modal propagation. The unexcited upper bus uses its own discrete modal propagation estimate. No fitted phase or wavelength registration.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first',type=Path)
    parser.add_argument('second',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare(args.first,args.second),indent=2,allow_nan=False)+'\n')
