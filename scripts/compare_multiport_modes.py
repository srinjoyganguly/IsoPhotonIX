"""Local full-vector transverse comparisons for the calibrated directional coupler.

Run each engine in its own Python environment. Tidy3D uses its local CPU mode
solver only. No Tidy3D service or commercial license is needed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.calibrated_geometry import CORE_WIDTH, CORE_HEIGHT, INNER_OFFSET, OUTER_OFFSET

WAVELENGTH = 1.55
ANSYS = 'https://optics.ansys.com/hc/en-us/articles/360042304694-Evanescent-waveguide-couplers'
TIDY = 'https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/faq/how-do-i-run-the-mode-solver-locally.html'


def section(name, centers, width, height, core, upper, lower):
    return {'name':name, 'centers':centers, 'width':width, 'height':height, 'core':core,
            'upper':upper, 'lower':lower, 'span_yz':[6., 3.]}


def sections():
    models = [section('lumerical-soi', [-.275,.275], .5, .2, 3.5, 1., 1.5),
              section('isolated-lead', [0.], CORE_WIDTH, CORE_HEIGHT, 3.48, 1.45, 1.45)]
    for i in range(9):
        distance = INNER_OFFSET + (OUTER_OFFSET-INNER_OFFSET)*(1-np.cos(np.pi*i/8))/2
        models.append(section(f'coupler-section-{i}', [-distance,distance], CORE_WIDTH, CORE_HEIGHT, 3.48, 1.45, 1.45))
    return models


def mpb_modes(model, resolution):
    import meep as mp
    from meep import mpb
    mp.verbosity(0)
    ly,lz = model['span_yz']
    w,h=model['width'],model['height']
    f = 1/WAVELENGTH
    geometry = [mp.Block(center=mp.Vector3(0,y,0), size=mp.Vector3(mp.inf,w,h),
                         material=mp.Medium(index=model['core'])) for y in model['centers']]
    if model['upper'] != model['lower']:
        geometry.append(mp.Block(center=mp.Vector3(0,0,-(lz+h)/4), size=mp.Vector3(mp.inf,mp.inf,(lz-h)/2),
                                 material=mp.Medium(index=model['lower'])))
    modes = {}
    for name,parity in [('even',mp.ODD_Y), ('odd',mp.EVEN_Y)][:len(model['centers'])]:
        if model['upper'] == model['lower']:
            parity += mp.EVEN_Z
        ms=mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0,ly,lz)), geometry=geometry,
            default_material=mp.Medium(index=model['upper']), resolution=resolution, num_bands=1,
            deterministic=True, tolerance=1e-10, mesh_size=3)
        samples={}
        def capture(solver,band):
            field=np.asarray(solver.get_efield(band,bloch_phase=False)).reshape(ms.get_epsilon().shape+(3,))
            power=np.sum(abs(field)**2,axis=(0,1))
            ey=field[:,:,1]
            reflected=ey[(-np.arange(len(ey))) % len(ey)]
            samples.update(ey_fraction=float(power[1]/power.sum()),
                           mirror_overlap=float(np.vdot(ey,reflected).real/np.vdot(ey,ey).real),
                           residual=abs(float(solver.freqs[band-1])-f)/f)
        k=ms.find_k(parity,f,1,1,mp.Vector3(1),1e-9,2.5*f,max(model['lower'],model['upper'])*f*1.000001,
                    model['core']*f,capture)[0]
        modes[name]={'neff':float(k/f),**samples}
    return modes


def tidy_modes(model, resolution):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    td.config.simulation.use_local_subpixel = False
    ly,lz=model['span_yz']
    w,h=model['width'],model['height']
    waveguides=[td.Structure(geometry=td.Box(size=(td.inf,w,h),center=(0,y,0)), medium=td.Medium(permittivity=model['core']**2))
                for y in model['centers']]
    if model['lower']!=model['upper']:
        waveguides.append(td.Structure(geometry=td.Box(size=(td.inf,td.inf,(lz-h)/2),center=(0,0,-(lz+h)/4)),
                          medium=td.Medium(permittivity=model['lower']**2)))
    sim=td.Simulation(size=(.5,ly,lz),run_time=1e-12, medium=td.Medium(permittivity=model['upper']**2),
                      structures=waveguides,grid_spec=td.GridSpec.uniform(dl=1/resolution),
                      boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()))
    solver=ModeSolver(simulation=sim, plane=td.Box(size=(0,ly,lz)), freqs=[td.C_0/WAVELENGTH],
                      mode_spec=td.ModeSpec(num_modes=4, target_neff=model['core']-.01,precision='double',filter_pol='te'))
    data=solver.solve()
    modes={}
    for i, neff in enumerate(np.real(data.n_eff.values[0])):
        fields=[getattr(data,k).isel(mode_index=i,x=0,f=0) for k in ('Ex','Ey','Ez')]
        power = np.array([np.sum(abs(v.values)**2) for v in fields])
        ey=fields[1]
        q=ey.interp(y=-ey.y.values).values
        mask=np.isfinite(q)
        overlap=float(np.vdot(ey.values[mask],q[mask]).real/np.vdot(ey.values[mask],ey.values[mask]).real)
        parity='even' if overlap>0 else 'odd'
        if power[1]/power.sum()>.5 and abs(overlap)>.95 and neff>max(model['upper'],model['lower']) and parity not in modes:
            modes[parity]={'neff':float(neff), 'ey_fraction':float(power[1]/power.sum()), 'mirror_overlap':overlap}
    return {k:modes[k] for k in ('even','odd')[:len(model['centers'])]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=('mpb','tidy3d'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolutions',nargs='+',type=int,choices=(32,48,64,80,128),default=[32,64])
    args=parser.parse_args()
    rows=[]
    solver=mpb_modes if args.engine=='mpb' else tidy_modes
    for mesh in args.resolutions:
        cases={}
        for m in sections():
            modes=solver(m,mesh)
            if any(x['ey_fraction']<=.5 or abs(x['mirror_overlap'])<.95 for x in modes.values()):
                raise ValueError('Transverse mode identification failed.')
            cases[m['name']]={'section':m,'modes':modes}
            if 'odd' in modes:
                cases[m['name']]['delta_neff']=modes['even']['neff']-modes['odd']['neff']
            print(args.engine,mesh,m['name'],[round(v['neff'],7) for v in modes.values()],flush=True)
        splits=np.array([cases[f'coupler-section-{i}']['delta_neff'] for i in range(9)])
        # Central parallel straight plus two 3 µm smooth approaches; trapezoidal
        # and Simpson convergence belongs to this approximation, separately from FDTD.
        integrated=2*splits[0]+2*(.375/3)*(splits[0]+splits[8]+4*sum(splits[1:8:2])+2*sum(splits[2:8:2]))
        coarse=2*splits[0]+2*(.75/3)*(splits[0]+splits[8]+4*(splits[2]+splits[6])+2*splits[4])
        # The remaining parallel access leads occupy 1.5 µm between both port planes.
        integrated += 1.5*splits[8]
        coarse += 1.5*splits[8]
        rows.append({'cells_per_um':mesh,'sections':cases,'central_uniform_transfer_length_um':WAVELENGTH/(2*splits[0]),
                     'adiabatic_cross_fraction':float(np.sin(np.pi/WAVELENGTH*integrated)**2),
                     'adiabatic_delta_neff_integral_um':float(integrated),
                     'adiabatic_cross_fraction_coarser_quadrature':float(np.sin(np.pi/WAVELENGTH*coarse)**2)})
    with args.output.open('x') as out:
        json.dump({'engine':args.engine,'wavelength_um':WAVELENGTH,'meshes':rows,'lumerical_reference':ANSYS,'tidy3d_local_reference':TIDY},out,indent=2)
        out.write('\n')


if __name__=='__main__':
    main()
