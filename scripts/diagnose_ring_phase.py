"""Estimate the isolated round-trip phase of the filed Lumerical ring using local bent-guide modes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.ring_palik_dispersion import LIGHT, WAVELENGTHS, material_samples


def boundaries(mesh, aperture, half_size):
    cuts=(-aperture/2,-half_size,half_size,aperture/2)
    segments=[np.linspace(a,b,round(mesh*(b-a))+1)[:-1] for a,b in zip(cuts,cuts[1:])]
    return np.concatenate([*segments,[cuts[-1]]])


def mode(mesh, aperture, wavelength, si, oxide, radius, aligned):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    # The bend lies in xy; z is the second tangent coordinate of this yz mode plane.
    sub=td.Structure(geometry=td.Box.from_bounds((-td.inf,-td.inf,-aperture),(td.inf,td.inf,-.09)),
                     medium=td.Medium(permittivity=oxide))
    core=td.Structure(geometry=td.Box(size=(td.inf,.40,.18)),medium=td.Medium(permittivity=si))
    grid=td.GridSpec.uniform(dl=1/mesh)
    if aligned:
        grid=td.GridSpec(grid_x=td.UniformGrid(dl=1/mesh),
                         grid_y=td.CustomGridBoundaries(coords=boundaries(mesh,aperture,.20)),
                         grid_z=td.CustomGridBoundaries(coords=boundaries(mesh,aperture,.09)))
    sim=td.Simulation(size=(.2,aperture,aperture),structures=[sub,core],run_time=1e-12,grid_spec=grid)
    spec=td.ModeSpec(num_modes=4,target_neff=1.94,num_pml=(round(.20*mesh),round(.20*mesh)),
                     bend_radius=radius,bend_axis=1,precision='double')
    data=ModeSolver(simulation=sim,plane=td.Box(size=(0,aperture,aperture)),
                    mode_spec=spec,freqs=[LIGHT*1e6/wavelength]).data
    ns=np.asarray(data.n_complex).reshape(-1)
    frac=np.asarray(data.pol_fraction.te).reshape(-1)
    candidates=np.flatnonzero((ns.real>1.75)&(ns.real<2.25)&(frac>.8)&(abs(ns.imag)<.01))
    if len(candidates)!=1:
        raise ValueError(f'A unique quasi-TE bend mode is required. n={ns}, TE fractions={frac}')
    i=candidates[0]
    field=data.Ey.isel(x=0,f=0,mode_index=int(i)).transpose('y','z')
    y,z=np.asarray(field.coords['y']),np.asarray(field.coords['z'])
    weight=abs(np.asarray(field))**2*np.gradient(y)[:,None]*np.gradient(z)[None,:]
    inside=(abs(y[:,None])<.20)&(abs(z[None,:])<.09)
    core_share=float(weight[inside].sum()/weight.sum())
    if core_share<.10:
        raise ValueError('Insufficient Ey localization in the silicon core for mode tracking.')
    return {'wavelength_um':float(wavelength),'n_real':float(ns[i].real),'n_imag':float(ns[i].imag),
            'te_fraction':float(frac[i]),'core_ey_intensity_fraction':core_share,
            'mode_index':int(i),'source_grid_interface_registered':aligned,
            'round_trip_cycles':float(2*np.pi*radius*ns[i].real/wavelength)}


def estimate(mesh, aperture, indices, aligned=False):
    radius=3.10
    samples=list(material_samples().values())
    rows=[mode(mesh,aperture,WAVELENGTHS[k],samples[0][k],samples[1][k],radius,aligned) for k in indices]
    result={'mesh_cells_per_um':mesh,'aperture_um':aperture,'interface_registered':aligned,'modes':rows}
    if len(rows)>1:
        phase=np.array([r['round_trip_cycles'] for r in rows])
        wave=np.array([r['wavelength_um'] for r in rows])
        if not np.all(np.diff(phase)<0):
            raise ValueError('Round-trip mode tracking is not monotonic.')
        result['isolated_ring_estimates']=[
            {'integer_order':m,'wavelength_um':float(np.interp(m,phase[::-1],wave[::-1]))}
            for m in range(int(np.ceil(phase.min())),int(np.floor(phase.max()))+1)]
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import tidy3d as td
    td.config.simulation.use_local_subpixel=False
    checks=[estimate(64,2.6,range(5)),estimate(96,2.6,range(5)),estimate(96,3.2,[2])]
    checks.extend(estimate(mesh,2.6,range(5),True) for mesh in (64,96,128,160))
    checks.append(estimate(128,3.2,[2],True))
    aligned={c['mesh_cells_per_um']:c for c in checks if c['interface_registered'] and c['aperture_um']==2.6}
    poles={mesh:next(row['wavelength_um'] for row in model['isolated_ring_estimates'] if row['integer_order']==24)
           for mesh,model in aligned.items()}
    extrapolations=[]
    for first,second,holdout in ((64,96,128),(96,128,160)):
        slope=(poles[first]-poles[second])/(1/first-1/second)
        limit=poles[second]-slope/second
        prediction=limit+slope/holdout
        extrapolations.append({'coarse_mesh':first,'fine_mesh':second,'heldout_mesh':holdout,
            'order':24,'linear_spacing_limit_um':limit,'heldout_prediction_um':prediction,
            'heldout_measured_um':poles[holdout],'heldout_difference_nm':1000*(prediction-poles[holdout])})
    report={'method':'Local Tidy3D transverse mode solver, curved coordinate system, no time-domain or cloud solve.',
            'version':td.__version__,'bend_radius_um':3.10,'material_input':'v013-ring-project-audit.json',
            'scope':'Isolated bend phase model. Sampling and aperture sensitivity are shown; the two buses, coupler phase, backscatter and loaded lineshapes require full time evolution.',
            'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'controls':checks,
            'conditional_mesh_extrapolation':extrapolations}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
