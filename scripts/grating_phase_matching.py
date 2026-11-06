"""Screen vertical mesh dispersion in the Lumerical grating's two silicon slab sections."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.calibrate_finite_grating import grid,settings
from scripts.lumerical_grating3d import lp01


def continuous_te(m,height,wavelength):
    k=2*np.pi/wavelength
    v=k*height/2*math.sqrt(m['n_si']**2-m['n_oxide']**2)
    u=brentq(lambda t:t*math.tan(t)-math.sqrt(v*v-t*t),0.,min(v,np.pi/2)-1e-10)
    return math.sqrt(m['n_si']**2-(2*u/(height*k))**2)


def sampled_te(m,height,wavelength,mesh, material_model='volume-average'):
    """Infinite symmetric oxide cladding, using the device's actual z-grid origin."""
    dx=1/mesh
    zs=grid(dict(m,resolution=mesh))[2][2]
    z=zs[0]+np.arange(-5*mesh,len(zs)+5*mesh)*dx
    if material_model=='direct-yee':
        z-=dx/2  # Ey's vertical half step relative to the supplied scalar-cell centers.
    elif material_model!='volume-average':
        raise ValueError('Unknown slab sampling model.')
    fraction=np.clip((np.minimum(height,z+dx/2)-np.maximum(0.,z-dx/2))/dx,0.,1.)
    epsilon=m['n_oxide']**2+fraction*(m['n_si']**2-m['n_oxide']**2)
    if material_model=='volume-average':
        epsilon=.5*(epsilon+np.r_[epsilon[0],epsilon[:-1]])
    k=2*np.pi/wavelength
    cdt=.99*dx/math.sqrt(3.)
    k_time=2*math.sin(k*cdt/2)/cdt
    diag=k_time**2*epsilon-2/dx**2
    off=np.ones(len(z)-1)/dx**2
    beta_grid_sq=eigh_tridiagonal(diag,off,eigvals_only=True,select='i',
                                  select_range=(len(z)-1,len(z)-1))[0]
    return 2*math.asin(math.sqrt(beta_grid_sq)*dx/2)/(k*dx)


def screen(mesh,material_model='volume-average', material_samples=None):
    m=settings('lumerical',12)
    def index(height,wavelength):
        medium=m
        if material_samples is not None:
            waves=material_samples['wavelength_um']
            if not waves[0]<=wavelength<=waves[-1]:
                raise ValueError('Wavelength is outside the published Palik samples.')
            perms=material_samples['material_epsilon']
            medium={**m,**{name:math.sqrt(np.interp(wavelength,waves,perms[key]['real'])) for name,key in
                          (('n_si','Si (Silicon) - Palik'),('n_oxide','SiO2 (Glass) - Palik'))}}
        if mesh is None: return continuous_te(medium,height,wavelength)
        return sampled_te(medium,height,wavelength,mesh,material_model)
    def equation(wavelength):
        rib=index(m['height'],wavelength)
        film=index(m['height']-m['etch'],wavelength)
        fiber=lp01(m['fiber_radius'],m['fiber_core_n'],m['fiber_clad_n'],wavelength)[2]
        return m['fill']*rib+(1-m['fill'])*film-fiber*math.sin(math.radians(m['angle_deg']))-wavelength/m['period']
    bracket=(1.5,1.6) if material_samples is not None else (1.2,1.9)
    return {'mesh_cells_per_um':mesh,'rib_te_neff_1550':index(.22,1.55),'film_te_neff_1550':index(.12,1.55),
            'estimated_first_order_wavelength_um':brentq(equation,*bracket)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();continuous=screen(None);rows=[screen(mesh) for mesh in (12,14,16,24,48,96)]
    if abs(rows[-1]['estimated_first_order_wavelength_um']-continuous['estimated_first_order_wavelength_um'])>.001:
        raise ValueError('Finest sampled slab solution must approach the independent continuous root.')
    measured=json.loads((ROOT/'docs/verification/v012-grating3d-lumerical-r12.json').read_text())['result']
    native=json.loads((ROOT/'docs/verification/v012-grating-commercial-references.json').read_text())['lumerical']['samples']['S21']
    peak=int(np.argmax(measured['beam_to_guide']))
    study={'scope':'Approximate first-order phase matching of two symmetric oxide-supported TE slab sections, including the Yee temporal and longitudinal dispersion relations. The model omits curved-tooth scattering, the 3D taper, fiber overlap, finite substrate and transverse radiation. Its output is the reduced slab phase-matching wavelength.',
           'physical_continuous_slabs':continuous,'sampled_slabs':rows,'measured_12_cell_device':{'strongest_tested_wavelength_um':measured['wavelength_um'][peak],'power':measured['beam_to_guide'][peak]},
           'published_lumerical_peak_um':native['peak_wavelength_um'],'fine_root_check_passed':True}
    args.output.write_text(json.dumps(study,indent=2,allow_nan=False)+'\n')
