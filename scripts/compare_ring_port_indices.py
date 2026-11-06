"""Independent straight-lead eigensolves for the asymmetric Lumerical ring, outside its FDTD run."""
import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]/'docs/verification'
WIDTH,HEIGHT,APERTURE = .4,.18,3.


def permittivities(audit, wavelength, material):
    if material=='constant':
        return 3.48**2,1.44**2
    grid=1/np.asarray(audit['material_wavelengths_um'])
    return tuple(float(np.interp(1/wavelength,grid[::-1],audit['material_tables_epsilon'][name]['real'][::-1]))
                 for name in ('Si (Silicon) - Palik','SiO2 (Glass) - Palik'))


def tidy_index(mesh,wave,si,oxide):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    td.config.simulation.use_local_subpixel = False
    sub=td.Structure(geometry=td.Box.from_bounds((-td.inf,-td.inf,-APERTURE), (td.inf,td.inf,-HEIGHT/2)),
                     medium=td.Medium(permittivity=oxide))
    core=td.Structure(geometry=td.Box(size=(td.inf,WIDTH,HEIGHT)),medium=td.Medium(permittivity=si))
    sim=td.Simulation(size=(.2,APERTURE,APERTURE),structures=[sub,core],run_time=1e-12,
                      boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),grid_spec=td.GridSpec.uniform(dl=1/mesh))
    data=ModeSolver(simulation=sim,plane=td.Box(size=(0,APERTURE,APERTURE)),freqs=[td.C_0/wave],
                    mode_spec=td.ModeSpec(num_modes=2,target_neff=2.1,precision='double')).solve()
    en=np.array([np.sum(abs(getattr(data,c).isel(f=0).values)**2, axis=(0,1,2)) for c in ('Ex','Ey','Ez')])
    frac=en[1]/(en[1]+en[2])
    ns=np.asarray(data.n_eff.isel(f=0))
    valid=np.flatnonzero((frac>.5)&(ns>np.sqrt(oxide)))
    if len(valid)!=1:
        raise ValueError('Expected a single fundamental quasi-TE mode.')
    i=valid[0]
    return float(ns[i]),float(frac[i])


def mpb_index(mesh,wave,si,oxide):
    import meep as mp
    from meep import mpb
    mp.verbosity(0)
    sub_height=(APERTURE-HEIGHT)/2
    sub=mp.Block(size=mp.Vector3(mp.inf,mp.inf,sub_height),center=mp.Vector3(z=-HEIGHT/2-sub_height/2),
                 material=mp.Medium(epsilon=oxide))
    core=mp.Block(size=mp.Vector3(mp.inf,WIDTH,HEIGHT),material=mp.Medium(epsilon=si))
    solver=mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0,APERTURE,APERTURE)),resolution=mesh,
                          geometry=[sub,core],num_bands=1,deterministic=True,tolerance=1e-9,mesh_size=5)
    f=1/wave
    fractions=[]
    def sample(model,band):
        e=np.asarray(model.get_efield(band,bloch_phase=False))
        en=np.sum(abs(e)**2,axis=tuple(range(e.ndim-1)))
        fractions.append(float(en[1]/(en[1]+en[2])))
    k=solver.find_k(mp.NO_PARITY,f,1,1,mp.Vector3(1,0,0),1e-8,1.93*f,np.sqrt(oxide)*f*1.001,np.sqrt(si)*f,sample)[0]
    if len(fractions)!=1 or fractions[0]<=.5:
        raise ValueError('The highest-index guided MPB mode was not quasi-TE.')
    return float(k/f),fractions[0]


def solve(engine,output):
    audit=json.loads((BASE/'v013-ring-project-audit.json').read_text())
    vendor=audit['port_modes'][0]
    waves=[1.525,vendor['mode_wavelength_um'],1.55,1.575]
    if engine=='mpb':
        from meep import __version__ as version
        compute=mpb_index
    else:
        from tidy3d import __version__ as version
        compute=tidy_index
    result={'engine':engine,'version':version,'source_project_sha256':audit['sources']['project']['sha256'],
            'design':{'width_um':WIDTH,'height_um':HEIGHT,'port_span_um':APERTURE,'top':'air','bottom':'oxide'},
            'rows':[],'scope':'Local transverse eigensolves only. Tidy3D uses a local uniform grid without subpixel averaging; FDTDX also uses Tidy3D internally for transverse modes. MPB is a separate frequency-domain eigensolver with dielectric averaging.'}
    for mesh in (40,64,96,128):
        for wave in waves:
            if mesh not in (64,96) and wave!=vendor['mode_wavelength_um']:
                continue
            for material in ('constant','palik-table'):
                si,oxide=permittivities(audit,wave,material)
                neff,te=compute(mesh,wave,si,oxide)
                result['rows'].append(dict(resolution=mesh,wavelength_um=wave,material=material,neff=neff,
                                           transverse_Ey_fraction=te,epsilon_si=si,epsilon_oxide=oxide))
                print(engine,mesh,wave,material,neff,flush=True)
    for mesh in (64,96):
        result[f'group_index_{mesh}']={}
        for material in ('constant','palik-table'):
            cells={r['wavelength_um']:r['neff'] for r in result['rows'] if r['resolution']==mesh and r['material']==material}
            result[f'group_index_{mesh}'][material]=cells[1.55]-1.55*(cells[1.575]-cells[1.525])/.05
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=('tidy3d','mpb'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    solve(args.engine,args.output)
