"""Compare the LP01 launch to independent local full-vector Tidy3D cross-sectional modes."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.lumerical_grating3d import fiber_radial_field,lp01,model


def solve(mesh,wavelength):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver

    m=model(12); span=24.
    medium=td.Medium(permittivity=m['fiber_clad_n']**2)
    core=td.Structure(geometry=td.Cylinder(radius=m['fiber_radius'],length=td.inf,axis=2),
                      medium=td.Medium(permittivity=m['fiber_core_n']**2))
    sim=td.Simulation(size=(span,span,1.),medium=medium,structures=[core],
                      grid_spec=td.GridSpec.uniform(dl=1/mesh),run_time=1e-13,
                      boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()))
    solver=ModeSolver(simulation=sim,plane=td.Box(size=(span,span,0)),freqs=[td.C_0/wavelength],
                      mode_spec=td.ModeSpec(num_modes=2,precision='double',target_neff=m['fiber_core_n']))
    data=solver.solve()
    neffs=data.n_eff.values.ravel()
    power=[]
    for i in range(2):
        ex=np.asarray(data.Ex.isel(f=0,mode_index=i));ey=np.asarray(data.Ey.isel(f=0,mode_index=i))
        power.append(float(np.linalg.norm(ey)**2/(np.linalg.norm(ex)**2+np.linalg.norm(ey)**2)))
    idx=int(np.argmax(power))
    ey=data.Ey.isel(f=0,mode_index=idx).squeeze().transpose('x','y')
    u,w,expected=lp01(m['fiber_radius'],m['fiber_core_n'],m['fiber_clad_n'],wavelength)
    rho=np.hypot(np.asarray(ey.x)[:,None],np.asarray(ey.y)[None,:])/m['fiber_radius']
    reference=fiber_radial_field(rho,u,w)
    sampled=np.asarray(ey)
    overlap=float(abs(np.vdot(reference,sampled))**2/(np.vdot(reference,reference).real*np.vdot(sampled,sampled).real))
    return {'cells_per_um':mesh,'span_um':span,'wavelength_um':wavelength,'vector_tidy3d_neff':neffs.tolist(),
            'selected_mode':idx,'ey_transverse_fraction':power[idx],'weak_guidance_lp01_neff':expected,
            'vector_minus_scalar_neff':float(neffs[idx]-expected),'ey_squared_shape_overlap':overlap,
            'scope':'Normalized Ey shape on the native component coordinates. This is a transverse field-shape comparison, distinct from full device coupling.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); rows=[solve(mesh,1.55) for mesh in (8,12)]
    result={'solver':'Local Tidy3D full-vector ModeSolver','version':importlib.metadata.version('tidy3d'),
            'source':'https://www.flexcompute.com/tidy3d/examples/notebooks/ModalSourcesMonitors/',
            'geometry':'Circular 9 µm diameter step-index fiber in a transverse box, using the filed Lumerical material indices.',
            'rows':rows,'passed':all(abs(r['vector_minus_scalar_neff'])<.001 and r['ey_squared_shape_overlap']>.995 for r in rows)}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(result)
