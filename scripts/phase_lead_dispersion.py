"""Predict straight-lead propagation using the spatial and temporal symbols of the Yee stencil."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_fdtdx_multiport import grid
from backend.calibrated_geometry import straight_polygon
from scripts.phase_polygon_material import extruded_yee
from scripts.qualify_port_phase import native_column

C_UM_S=299792458e6


def section(model,r,port=0,origin_shift_cells=(0.,0.),extra_aperture_cells=0):
    shape,pml,axes,planes=grid(model,r)
    m=planes[port]
    i,y,z=m['start']
    _,ny,nz=m['shape']
    # Include a halo for the lateral normal estimate. Keep the exact port
    # location and aperture used when the native 3D amplitudes were projected.
    x_axis=axes[0][i-1:i+2]
    y_axis=axes[1][y-1-extra_aperture_cells:y+ny+1+extra_aperture_cells]+origin_shift_cells[0]/r
    z_axis=axes[2][z-extra_aperture_cells:z+nz+extra_aperture_cells]+origin_shift_cells[1]/r
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


def registration_controls(folder):
    plan=json.loads((folder/'run.json').read_text())
    model,r=plan['model'],plan['resolution']
    f=1/1.55
    rows=[]
    for y in (0.,.5):
        for z in (0.,.25,.5):
            value=mode_index(section(model,r,origin_shift_cells=(y,z)),f,1/r)
            rows.append({'y_shift_cells':y,'z_shift_cells':z,'neff':value})
    # Independently bound the small motion of the aperture edge in the shifted
    # grid by extending each edge 0.25 µm while preserving the origin.
    extra=round(.25*r)
    larger=mode_index(section(model,r,origin_shift_cells=(.5,.5),extra_aperture_cells=extra),f,1/r)
    return {'resolution':r,'nominal_guide_width_height_um':[model['width'],model['height']],
            'wavelength_um':1.55,'registration':rows,'max_minus_min_neff':max(x['neff'] for x in rows)-min(x['neff'] for x in rows),
            'larger_aperture_edge_extension_um':extra/r,'largest_translation_larger_aperture_neff':larger,
            'largest_translation_aperture_change':larger-rows[-1]['neff']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    rows=[predict(path) for path in args.directories]
    unique={}
    for path in args.directories:
        plan=json.loads((path/'run.json').read_text())
        unique.setdefault(plan['resolution'],path)
    result={'method':'Independent transverse eigenmodes on the identical port section. For f in cycles/µm use f_eff = f sinc(f c Δt), solve n_eff(f_eff), and β_Yee = 2 asin(π f_eff n_eff Δx)/Δx. Compare the measured forward transfer with exp(iβL). No fitted constant or change to the device scattering matrix.',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'transverse_versions':{p:metadata.version(p) for p in ('fdtdx','jax','numpy','tidy3d')},
            'transverse_execution':'FDTDX local CPU 2D mode calculation','native_execution':'Fully time-stepped 3D FDTDX on NVIDIA',
            'scope':'Uniform nondispersive straight leads. This control does not estimate junction radiation or resonant scattering.',
            'samples':rows,'grid_registration':[registration_controls(path) for path in unique.values()]}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
