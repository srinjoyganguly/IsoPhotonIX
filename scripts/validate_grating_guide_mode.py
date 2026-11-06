"""Compare independent local strip modes to the guide mode stored in the Ansys grating model."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scripts.compare_grating_port_basis import BASE, project_basis


def solve(mesh, wavelength, medium, filed, axis, aperture=(2.5,2.25)):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    td.config.simulation.use_local_subpixel=False
    y,z=axis
    # Match the finite port aperture; centering on the physical core is distinct
    # from centering the numerical window on the mode maximum.
    sy,sz=aperture; z0=.05
    silicon=td.Medium(permittivity=medium[0]); oxide=td.Medium(permittivity=medium[1])
    core=td.Structure(geometry=td.Box(center=(0,0,.11),size=(td.inf,.5,.22)),medium=silicon)
    buried=td.Structure(geometry=td.Box.from_bounds((-td.inf,-td.inf,-1.),(td.inf,td.inf,.7)),medium=oxide)
    substrate=td.Structure(geometry=td.Box.from_bounds((-td.inf,-td.inf,-5.),(td.inf,td.inf,-1.)),medium=silicon)
    sim=td.Simulation(center=(0,0,z0),size=(.2,sy,sz),structures=[substrate,buried,core],
                      run_time=1e-13,grid_spec=td.GridSpec.uniform(dl=1/mesh),
                      boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()))
    data=ModeSolver(simulation=sim,plane=td.Box(center=(0,0,z0),size=(0,sy,sz)),freqs=[td.C_0/wavelength],
                    mode_spec=td.ModeSpec(num_modes=3,precision='double',target_neff=2.5)).solve()
    scores=[]
    for i in range(3):
        local=data.Ey.isel(f=0,mode_index=i).squeeze().transpose('y','z')
        ys,zs=np.asarray(local.y),np.asarray(local.z)
        yy,zz=np.meshgrid(ys,zs,indexing='ij')
        external=RegularGridInterpolator(axis,filed,bounds_error=False,fill_value=0.)((yy,zz))
        local=np.asarray(local)
        power=np.abs(local)**2
        frac=power[(abs(yy)<.40)&(zz>-.1)&(zz<.32)].sum()/power.sum()
        overlap=abs(np.vdot(local,external))**2/(np.vdot(external,external).real*np.vdot(local,local).real)
        edge=power.copy();edge[2:-2,2:-2]=0
        scores.append((float(frac),float(overlap),float(edge.sum()/power.sum())))
    i=int(np.argmax([s[0] for s in scores]))
    neff=float(data.n_eff.values.ravel()[i].real)
    if scores[i][0]<.45 or scores[i][1]<.8:
        raise RuntimeError('Failed to identify the core-localized Ey strip mode.')
    return {'cells_per_um':mesh,'guide_neff':neff,'squared_Ey_overlap':scores[i][1],
            'core_window_Ey_fraction':scores[i][0],'edge_Ey_fraction':scores[i][2],'aperture_um':list(aperture),'mode_index':i}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    vendor,axis,fields=project_basis(args.project)
    row=json.loads((BASE/'v012-grating-commercial-references.json').read_text())['lumerical']
    wave=vendor['wavelength_um']
    eps=tuple(float(np.interp(1/wave,(1/np.array(row['wavelength_um']))[::-1],
                  row['material_epsilon'][m]['real'][::-1])) for m in ('Si (Silicon) - Palik','SiO2 (Glass) - Palik'))
    results=[]
    for mesh in (32,48,72,96,128):
        result=solve(mesh,wave,eps,fields[0,1],axis)
        result['neff_minus_filed']=result['guide_neff']-vendor['guide_neff']
        results.append(result);print(result,flush=True)
    enlarged=solve(96,wave,eps,fields[0,1],axis,aperture=(3.0,2.7))
    enlarged['neff_minus_filed']=enlarged['guide_neff']-vendor['guide_neff']
    report={'solver':'Local Tidy3D ModeSolver','version':importlib.metadata.version('tidy3d'),
            'geometry':'Straight output strip, 500 × 220 nm; 1 µm buried oxide over substrate, 0.70 µm top oxide, air cover.',
            'scope':'Independent continuous geometry and meshes with local subpixel averaging disabled. FDTDX also calls Tidy3D for transverse port modes. The mode window is finite; this experiment does not propagate the curved 3D device.',
            'wavelength_um':wave,'filed_neff':vendor['guide_neff'],'filed_project_sha256':vendor['project_sha256'],
            'medium_permittivities':list(eps),'aperture_um':[2.5,2.25],'rows':results,'larger_window':enlarged}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
