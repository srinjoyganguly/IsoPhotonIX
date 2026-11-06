"""Predict straight-lead propagation using the spatial and temporal symbols of the Yee stencil."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_fdtdx_multiport import grid
from backend.calibrated_geometry import straight_polygon
from scripts.phase_polygon_material import extruded_yee
from scripts.qualify_port_phase import native_column

C_UM_S=299792458e6


def section(model,r,port=0):
    shape,pml,axes,planes=grid(model,r)
    m=planes[port]
    i,y,z=m['start']
    _,ny,nz=m['shape']
    # Include a halo for the lateral normal estimate. Keep the exact port
    # location and aperture used when the native 3D amplitudes were projected.
    x_axis=axes[0][i-1:i+2]
    y_axis=axes[1][y-1:y+ny+1]
    z_axis=axes[2][z:z+nz]
    maps=extruded_yee(model,r,(x_axis,y_axis,z_axis),[straight_polygon(model,port)])
    return maps[:,1:2,1:-1,:]


def mode_index(section, f, dx):
    import jax
    import jax.numpy as jnp
    from fdtdx.core.physics.modes import compute_mode
    if jax.default_backend()!='cpu':
        raise RuntimeError('This separate 2D transverse control requires the CPU mode solver.')
    _,_,n=compute_mode(frequency=float(f*C_UM_S),inv_permittivities=1/section,
                        inv_permeabilities=1.,resolution=dx*1e-6,direction='+',mode_index=0,
                        filter_pol='te',dtype=jnp.float32)
    if abs(complex(n).imag)>1e-5:
        raise ValueError('Expected a bound lossless quasi-TE mode.')
    return float(complex(n).real)


def predict(folder):
    plan,report,_,through,hashes=native_column(folder)
    if plan['material_model']!='polygon-area':
        raise ValueError('This control reconstructs the polygon-area straight lead.')
    m,r=plan['model'],plan['resolution']
    src=m['ports'][0]['name']
    s=json.loads((folder/f'{src}-reference.json').read_text())
    f=np.array(m['frequency_per_um'])
    dx=1/r
    eps=section(m,r)
    n=np.array([mode_index(eps,k,dx) for k in f])
    np.testing.assert_allclose(n,s['mode_neff'][src],rtol=2e-6,atol=1e-6)
    dt=s['performance']['dt_seconds']
    eff_f=f*np.sinc(f*C_UM_S*dt)
    eff_n=np.array([mode_index(eps,k,dx) for k in eff_f])
    beta_mode=2*np.pi*f*n
    argument=np.pi*eff_f*eff_n*dx
    if np.any(abs(argument)>=1):
        raise ValueError('The longitudinal mode lies above the propagating Yee branch.')
    beta_yee=2/dx*np.arcsin(argument)
    x=[p['actual_x'] for p in report['sampled_ports']]
    length=max(x)-min(x)
    delta_mode=np.angle(through*np.exp(-1j*beta_mode*length))
    delta_yee=np.angle(through*np.exp(-1j*beta_yee*length))
    return {'case':m['case'],'resolution':r,'straight_plane_distance_um':length,'dt_seconds':dt,
            'frequency_per_um':f.tolist(),'effective_temporal_frequency_per_um':eff_f.tolist(),
            'continuous_longitudinal_mode_neff':n.tolist(),'effective_frequency_mode_neff':eff_n.tolist(),
            'yee_longitudinal_neff':(beta_yee/(2*np.pi*f)).tolist(),'measured_straight_transfer_real':through.real.tolist(),
            'measured_straight_transfer_imag':through.imag.tolist(),'measured_minus_continuous_phase_rad':delta_mode.tolist(),
            'measured_minus_yee_phase_rad':delta_yee.tolist(),'native_reference_sha256':hashes[f'{src}-reference.npz']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    rows=[predict(path) for path in args.directories]
    result={'method':'Independent transverse eigenmodes on the identical port section. For f in cycles/µm use f_eff = f sinc(f c Δt), solve n_eff(f_eff), and β_Yee = 2 asin(π f_eff n_eff Δx)/Δx. Compare the measured forward transfer with exp(iβL). No fitted constant or change to the device scattering matrix.',
            'transverse_execution':'FDTDX local CPU 2D mode calculation','native_execution':'Fully time-stepped 3D FDTDX on NVIDIA',
            'scope':'Uniform nondispersive straight leads. This control does not estimate junction radiation or resonant scattering.',
            'samples':rows}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
