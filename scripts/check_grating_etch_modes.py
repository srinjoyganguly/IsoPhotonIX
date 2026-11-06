"""Measure the Tidy3D design-period observable from the Workbench etched and unetched sections."""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal

from scripts.calibrate_finite_grating import TIDY_URL, aligned, grid, material, settings


def estimate(mesh, verify_native=False, span=1.3):
    model=settings('tidy3d',mesh if mesh in (12,14,16) else 12) | {'resolution':mesh}
    _,_,(x,y,z)=grid(model)
    dx=1/mesh
    # Use the same staggered transverse-E coefficient as the finite grating.
    # These two straight sections are uniform in x,y across the mode plane.
    records=[]
    native=None
    if verify_native:
        base=material(model,False)
        native=aligned(base)[1]
        del base
    for thickness,plane_x in ((model['height'],-.1),(model['height']-model['etch'],model['teeth']*model['period']+.1)):
        cover=np.clip((np.minimum(z+dx/2,thickness)-np.maximum(z-dx/2,0))/dx,0,1)
        scalar=np.where(z<model['oxide_top'],model['n_oxide']**2,1.)
        scalar[z < -model['oxide_bottom']]=model['n_si']**2
        scalar+=cover*(model['n_si']**2-model['n_oxide']**2)
        eps=.5*(scalar+np.r_[scalar[0],scalar[:-1]])
        if native is not None:
            sampled=native[np.argmin(abs(x-plane_x)),np.argmin(abs(y)),:]
            np.testing.assert_allclose(eps,sampled,rtol=1e-6,atol=1e-6)
        # A discrete 1-D transverse TE solve at fixed frequency, with zero-E
        # boundaries on the cropped monitor aperture. No time stepping here.
        where=np.flatnonzero(abs(z-dx/2)<span/2)
        k0=2*np.pi/1.55
        diag=k0**2*eps[where]-2/dx**2
        beta_sq,field=eigh_tridiagonal(diag,np.full(len(diag)-1,1/dx**2),select='i',select_range=(len(diag)-1,len(diag)-1))
        mode=field[:,0]
        neff=float(np.sqrt(beta_sq[0])/k0)
        if not model['n_oxide']<neff<model['n_si'] or np.any(mode*mode[len(mode)//2]<0):
            raise ValueError('The transverse solution must be a node-free guided TE mode.')
        records.append({'thickness_nm':round(thickness*1000),'effective_index':neff,
                        'aperture_grid_points':len(where),'edge_intensity_fraction':float(np.sum(mode[[0,-1]]**2))})
    angle=np.deg2rad(model['angle_deg'])
    weighted=model['fill']*records[0]['effective_index']+(1-model['fill'])*records[1]['effective_index']
    period=1550/(weighted-model['n_oxide']*np.sin(np.sin(angle)/model['n_oxide']))
    return {'mesh_cells_per_um':mesh,'aperture_um':span,'sections':records,'period_nm':period,
            'difference_from_notebook_nm':period-608,'native_field_grid_coefficients_checked':verify_native}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--notebook',type=Path,required=True,help='Pinned Tidy3D GratingCoupler.ipynb')
    args=parser.parse_args()
    vendor=json.loads((Path(__file__).resolve().parents[1]/'docs/verification/v012-grating-commercial-references.json').read_text())['tidy3d']
    source=args.notebook.read_bytes()
    if hashlib.sha256(source).hexdigest()!=vendor['notebook_sha256']:
        raise ValueError('Expected the pinned Tidy3D notebook.')
    outputs=[''.join(output.get('text',[])).strip() for cell in json.loads(source)['cells'] for output in cell.get('outputs',[])]
    if 'Calculated grating period is 608 nm' not in outputs:
        raise ValueError('Pinned notebook does not contain the stated design-period result.')
    rows=[estimate(mesh,mesh in (12,14)) for mesh in (12,14,16,20,24,48,96,192,384)]
    rows.append(estimate(192,False,2.0))
    result={'source':TIDY_URL,'notebook':vendor['notebook'],'notebook_sha256':vendor['notebook_sha256'],'published_design_period_nm':608,'published_optimized_period_nm':620,
            'etch_nm':160,'unetched_height_nm':260,'residual_silicon_nm':100,'rows':rows,
            'scope':'Local transverse TE calculation on a common input stack. The 608 nm quantity is the published design-period estimate, distinct from the 620 nm optimized pitch used in the 3D power comparison.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
