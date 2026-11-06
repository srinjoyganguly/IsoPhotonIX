"""Measure the isolated straight-port response to lateral quadrature and a normal-wall control."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_loaded_ring3d import LIGHT, OXIDE, SILICON, geometry, lateral_fraction, permittivity

BASE=Path(__file__).resolve().parents[1]/'docs/verification'


def plane(g, method):
    mesh=g['resolution']
    px,py,pz=g['ports'][0]
    aperture=2*g['half_ap']
    ys=(np.arange(g['shape'][1])-g['shape'][1]/2+.5)/mesh
    straight=lateral_fraction(np.array([0.]),ys,1/mesh,g['radius_um'],g['width_um'],g['bus_y_um'],True,
                              'legacy' if method=='legacy' else 'exact-area')[0]
    eps=permittivity(g,True)[:,px,:, :].copy()
    if method=='normal-wall':
        z=(np.arange(g['shape'][2])-g['shape'][2]/2)/mesh
        bottom=z-.5/mesh
        top=z+.5/mesh
        height=g['height_um']
        substrate=np.clip((-height/2-bottom)*mesh,0,1)
        slab=np.clip((np.minimum(top,height/2)-np.maximum(bottom,-height/2))*mesh,0,1)
        # Ey is normal to a perfectly straight vertical wall. Within the Si/air
        # layer the two lateral dielectric fractions act in series for Ey.
        lateral=1/(straight/SILICON**2+(1-straight))
        eps[1]=substrate*OXIDE**2+slab*lateral[:,None]+1-substrate-slab
    return eps[:,None,py:py+aperture,pz:pz+aperture], straight.sum()/(2*mesh)


def study():
    import jax
    import jax.numpy as jnp
    from fdtdx.core.physics.modes import compute_mode
    vendor=json.loads((BASE/'v013-ring-project-audit.json').read_text())['port_modes'][0]
    result={'scope':'Local two-dimensional transverse mode solves through the actual FDTDX/Tidy3D port routine. The normal-wall column is a straight-lead experiment only; curved off-diagonal tensors, triple-interface corners and 3D device propagation require separate validation.',
            'vendor_mode_wavelength_um':vendor['mode_wavelength_um'],'vendor_neff':vendor['neff'],'rows':[]}
    for mesh in (16,20,24):
        baseline=json.loads((BASE/f'v012-ring3d-lumerical-r{mesh}-t3200.json').read_text())
        for method in ('legacy','exact-area','normal-wall'):
            interface='legacy' if method=='legacy' else 'exact-area'
            g=geometry('lumerical',mesh,.65,1.8,interface)
            eps,width=plane(g,method)
            measured={}
            for wave in (vendor['mode_wavelength_um'],1.55):
                _,_,n=compute_mode(frequency=LIGHT/(wave*1e-6),inv_permittivities=jnp.asarray(1/eps),inv_permeabilities=1.,
                                   resolution=1e-6/mesh,direction='+',mode_index=0,filter_pol='te',dtype=jnp.float32)
                measured[str(wave)]=float(np.real(n))
            if method=='legacy':
                np.testing.assert_allclose(measured[str(1.55)],baseline['stages']['reference']['source_mode_neff'],atol=2e-6,rtol=0)
            result['rows'].append({'mesh':mesh,'method':method,'wavelength_neff':measured,
                                   'equivalent_strip_width_um':float(width),'cpu_platform':jax.devices()[0].platform})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(study(),indent=2,allow_nan=False)+'\n')
