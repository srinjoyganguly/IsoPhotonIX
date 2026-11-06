"""Independently resolve the 500 × 250 nm straight-port index using local transverse solvers."""
import argparse
import hashlib
import sys
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np

from scripts.compare_multiport_modes import sections, mpb_modes


def registered_tidy(model,r):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    from scripts.diagnose_ring_phase import boundaries
    td.config.simulation.use_local_subpixel=False
    ly,lz=model['span_yz']
    y=boundaries(r,ly,model['width']/2)
    z=boundaries(r,lz,model['height']/2)
    grid=td.GridSpec(grid_x=td.UniformGrid(dl=1/r),grid_y=td.CustomGridBoundaries(coords=y),
                     grid_z=td.CustomGridBoundaries(coords=z))
    guide=td.Structure(geometry=td.Box(size=(td.inf,model['width'],model['height'])),
                       medium=td.Medium(permittivity=model['core']**2))
    sim=td.Simulation(size=(.25,ly,lz),run_time=1e-12,medium=td.Medium(permittivity=model['upper']**2),
                      structures=(guide,),grid_spec=grid)
    solver=ModeSolver(simulation=sim,plane=td.Box(size=(0,ly,lz)),freqs=[td.C_0/1.55],
        mode_spec=td.ModeSpec(num_modes=1,target_neff=2.55,num_pml=(round(.2*r),round(.2*r)),precision='double'))
    data=solver.solve()
    n=np.asarray(data.n_complex).item()
    te=np.asarray(data.pol_fraction.te).item()
    if not model['upper']<n.real<model['core'] or abs(n.imag)>1e-7 or te<.75:
        raise ValueError('Failed to track a guided fundamental quasi-TE mode.')
    return {'neff':float(n.real),'te_fraction':float(te),'max_abs_imag_neff':float(abs(n.imag)),
            'section_cells_yz':[len(y)-1,len(z)-1]}


def measure(engine,resolutions):
    model=next(s for s in sections() if s['name']=='isolated-lead')
    rows=[]
    for r in resolutions:
        if engine=='tidy3d':
            result=registered_tidy(model,r)
        else:
            result=mpb_modes(model,r)['even']
        rows.append({'mesh':r,**result})
    if engine=='tidy3d':
        versions={'tidy3d':metadata.version('tidy3d')}
    else:
        import meep
        versions={'meep':meep.__version__}
        for entry in (Path(sys.prefix)/'conda-meta').glob('mpb-*.json'):
            package=json.loads(entry.read_text())
            if package['name']=='mpb':
                versions['mpb']=package['version']
    files=[Path(__file__),Path(__file__).with_name('compare_multiport_modes.py')]
    if engine=='tidy3d':
        files.append(Path(__file__).with_name('diagnose_ring_phase.py'))
    return {'engine':engine,'versions':versions,'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            'wavelength_um':1.55,'geometry':model,'transverse_execution':'local CPU eigensolve','measurements':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=('tidy3d','mpb'),required=True)
    p.add_argument('--meshes',nargs='+',required=True,type=int);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();args.output.write_text(json.dumps(measure(args.engine,args.meshes),indent=2,allow_nan=False)+'\n')
