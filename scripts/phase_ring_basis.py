"""Check the single saved ring-port mode gauge against the native modal fields."""
import struct

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scripts.audit_ring_vendor_project import FSP_SHA, scalar, verified
from scripts.compare_grating_port_basis import read_matrix, tag_offsets


def project_modes(path):
    raw = verified(path,FSP_SHA)
    if b'abs(S11), unwrap(angle(S11))' not in raw:
        raise ValueError('Expected the filed magnitude/unwrapped-radian export.')
    modes = []
    for j in range(4):
        marker = raw.index(f'::port {j+1} Mode Expansion Monitor'.encode())
        at = tag_offsets(raw[:marker],'modeCurrentlyValid')[-1]
        if struct.unpack_from('<III',raw,at)!=(2,1,1):
            raise ValueError('The saved modal field must be valid.')
        start = raw.rfind(b'freqNumberOfModes\0',0,marker)
        freq = scalar(raw,'modeFrequency',1,start)
        coords = [read_matrix(raw,k+'Matrix',j+1,5)*1e6 for k in 'xyz']
        fields = np.array([[read_matrix(raw,f'freqEHn{k}{eh}',j,4) for k in 'xyz'] for eh in (0,1)])
        # Lumerical stores SI H; the native update uses impedance-scaled H.
        fields[1] *= 376.730313668
        if fields.shape != (2,3,len(coords[1]),len(coords[2])):
            raise ValueError('Filed port coordinates and modal fields differ.')
        modes.append((freq,coords,fields))
    return modes


def overlap_profiles(native_e,native_h,file_e,file_h,dx):
    def cross(e,h):
        return ((e[1]*h[2].conj()-e[2]*h[1].conj())*dx**2).sum()
    local_power = .5*cross(native_e,native_h).real
    file_power = .5*cross(file_e,file_h).real
    if min(file_power,local_power)<=0:
        raise ValueError('The native and filed forward modes must both carry positive-x power.')
    return (cross(native_e,file_h)+cross(file_e,native_h).conj())/(4*np.sqrt(local_power*file_power))


def central_overlaps(modes, model, arrays):
    shape,res = model['shape'],model['resolution']
    # Workbench orders reflection, through, drop, add-side. The FSP numbers
    # reflection 1, drop 2, through 3, add-side 4.
    result=[]
    for port,j in enumerate((0,2,1,3)):
        freq,coords,fields = modes[j]
        wl=299792458e6/freq
        bw=arrays['basis_wavelength_um']
        i = np.searchsorted(bw,wl)-1
        if not 0<=i<len(bw)-1:
            raise ValueError('The filed mode is outside the measured native basis band.')
        mix=(wl-bw[i])/(bw[i+1]-bw[i])
        local = [(1-mix)*arrays[f'device_basis_{kind}_{port}'][i]+mix*arrays[f'device_basis_{kind}_{port}'][i+1] for kind in ('E','H')]
        _,y0,z0 = model['ports'][port]
        ny,nz=local[0].shape[-2:]
        # Reflect the upper-lower bus layout about y=0, then translate the
        # native core centre z=0 to z=0.09 µm in the filed project.
        y = -(y0+np.arange(ny)-shape[1]/2)/res
        z = (z0+np.arange(nz)+.5-shape[2]/2)/res + .09
        query=np.stack(np.meshgrid(y,z,indexing='ij'),axis=-1)
        vv=np.array([[RegularGridInterpolator((coords[1],coords[2]),component)(query)
                      for component in field] for field in fields])
        # Polar E and axial H transform differently under reflection. An
        # additional common minus sign restores the positive-Ey gauge at every
        # port; it cancels between numerator and denominator of all launch ratios.
        vv[0] *= np.array([-1,1,-1])[:,None,None]
        vv[1] *= np.array([1,-1,1])[:,None,None]
        overlap = overlap_profiles(*local,*vv,1/res)
        def flux(e,h):
            return .5*(e[1]*h[2].conj()-e[2]*h[1].conj()).real
        full_density=flux(*fields)
        # Integrate on the original nonuniform port grid. The resampled native
        # grid uses its uniform dual-cell midpoint area.
        filed_power=np.trapezoid(np.trapezoid(full_density,coords[2],axis=1),coords[1],axis=0)
        local_aperture_power=flux(*vv).sum()/res**2
        if filed_power<=0:
            raise ValueError('The full filed aperture must carry positive-x power.')
        aperture_fraction=local_aperture_power/filed_power
        peak = np.unravel_index(np.argmax(abs(vv[0,1])),(ny,nz))
        result.append({'file_port':j+1,'wavelength_um':wl,'port_x_um':float(coords[0]),
                       'native_aperture_to_full_filed_power':float(aperture_fraction),'overlap_magnitude':float(abs(overlap)),'overlap_phase_rad':float(np.angle(overlap)),
                       'file_Ey_peak_phase_rad':float(np.angle(vv[0,1][peak]))})
    return result
