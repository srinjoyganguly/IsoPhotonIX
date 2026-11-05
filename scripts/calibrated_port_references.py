"""Local Tidy3D mode solves for identical Lumerical and workbench port sections.

No cloud simulation, API key or commercial installation is used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LUMERICAL_COUPLER = 'https://optics.ansys.com/hc/en-us/articles/360042304694-Evanescent-waveguide-couplers'
LUMERICAL_SLAB = 'https://optics.ansys.com/hc/en-us/articles/1500007378182-Inverse-design-of-an-efficient-narrow-band-grating-coupler'
TIDY3D = 'https://www.flexcompute.com/tidy3d/examples/notebooks/ModalSourcesMonitors/'


def solve(td, mode_solver, case, mesh):
    slab = case in ('lumerical-slab','multimode')
    vendor = case.startswith('lumerical')
    core, clad = (3.5,1.5) if case == 'lumerical-coupler' else (3.48,1.44 if slab else 1.45)
    width = (.22 if vendor else .375) if slab else (.875 if case=='multimode-3d' else .5)
    gap = .05 if vendor else .125
    height = .20 if vendor else .25
    sy, sz = (6.,0.) if slab else (3.,3.)
    mat = td.Medium(permittivity=core**2)
    if slab:
        parts = [td.Structure(geometry=td.Box(size=(td.inf,width,td.inf)),medium=mat)]
    else:
        parts = [td.Structure(geometry=td.Box(center=(0,sign*(width+gap)/2,0),size=(td.inf,width,height)),medium=mat)
                 for sign in (-1,1)]
    if case=='multimode-3d':
        parts=[td.Structure(geometry=td.Box(size=(td.inf,width,height)),medium=mat)]
        clad=1.44
    if case == 'lumerical-coupler':
        parts.append(td.Structure(geometry=td.Box(center=(0,0,-sz/2-height/2),
                        size=(td.inf,td.inf,sz)),medium=td.Medium(permittivity=clad**2)))
    background = 1. if case=='lumerical-coupler' else clad
    simulation = td.Simulation(size=(1.,sy,sz),medium=td.Medium(permittivity=background**2),
                               structures=parts,grid_spec=td.GridSpec.uniform(dl=mesh),
                               run_time=1e-12,boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()))
    plane=td.Box(size=(0,sy,td.inf if slab else sz))
    data=mode_solver(simulation=simulation,plane=plane,freqs=[td.C_0/1.55],
                     mode_spec=td.ModeSpec(num_modes=6,target_neff=core,precision='double')).solve()
    found={}
    for idx, neff in enumerate(data.n_eff.values.ravel()):
        if neff <= clad + .001:continue
        fields=[np.squeeze(getattr(data,c).isel(mode_index=idx,f=0).values) for c in ('Ex','Ey','Ez')]
        powers=np.array([np.linalg.norm(v)**2 for v in fields])
        field=fields[2 if slab else 1]
        fraction=powers[2 if slab else 1]/powers.sum()
        if fraction<.5:continue
        # Tidy3D E components occupy staggered locations. Interpolate on
        # mirrored physical y coordinates before measuring scalar parity.
        a=getattr(data,'Ez' if slab else 'Ey').isel(mode_index=idx,f=0).squeeze(drop=True)
        y=a.coords['y'].values
        sample=y[(y>0)&(y<min(sy/2-.1, 1.5))]
        left=np.asarray(a.interp(y=-sample)).ravel()
        right=np.asarray(a.interp(y=sample)).ravel()
        overlap=float(np.vdot(right,left).real/(np.linalg.norm(right)*np.linalg.norm(left)))
        label='even' if overlap>=0 else 'odd'
        if abs(overlap)<.95 or label in found:continue
        found[label]={'neff':float(neff),'polarization_fraction':float(fraction),'parity_overlap':overlap}
    required={'even'} if case=='lumerical-slab' else {'even','odd'}
    if not required.issubset(found):raise ValueError(f'Could not identify {required} modes in {case}: {found}')
    row={'mesh_um':mesh,'modes':found}
    if case.endswith('coupler'):row['transfer_length_um']=1.55/(2*(found['even']['neff']-found['odd']['neff']))
    return row


def published_comparisons(report):
    published = {'lumerical-slab':{'even':2.84825},
                 'lumerical-coupler':{'even':2.378022,'odd':2.317864,'transfer_length_um':12.8827}}
    comparisons={}
    for name, values in published.items():
        row=report['cases'][name][-1]
        comparisons[name]={}
        for mode,reference in values.items():
            measured=row['transfer_length_um'] if mode=='transfer_length_um' else row['modes'][mode]['neff']
            comparisons[name][mode]={'published_Lumerical':reference,'measured_Tidy3D_local':measured,
                                     'relative_difference':(measured-reference)/reference}
    report.update(direct_published_comparisons=comparisons,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  qualification='Local double-precision mode solver on uniform grids without subpixel smoothing; parity is checked at mirrored physical coordinates. These are cross-section measurements.')


def main():
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refine-couplers',action='store_true',help='also solve both coupled cross sections with 6.25 nm cells')
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a fresh JSON file.')
    td.config.simulation.use_local_subpixel=False
    report={'solver':'Tidy3D installed local CPU mode solver','version':td.__version__,'wavelength_um':1.55,
            'references':{'lumerical_coupler':LUMERICAL_COUPLER,'lumerical_slab':LUMERICAL_SLAB,'tidy3d':TIDY3D},'cases':{}}
    for name in ('lumerical-slab','multimode','lumerical-coupler','workbench-coupler','multimode-3d'):
        meshes=(.0125,.00625,.003125) if name in ('lumerical-slab','multimode') else (.025,.0125)
        if args.refine_couplers and name.endswith('coupler'):
            meshes=(*meshes,.00625)
        rows=[]
        for mesh in meshes:
            row=solve(td,ModeSolver,name,mesh)
            print(name,row,flush=True)
            rows.append(row)
        report['cases'][name]=rows
    published_comparisons(report)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    main()
