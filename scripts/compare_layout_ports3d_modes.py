"""Local continuous-geometry lead-mode controls for the finite-height default polygon elbow."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WAVELENGTHS = np.array([1.60,1.55,1.50])
WIDTH, HEIGHT, APERTURE = .375, .25, 2.
CORE, CLAD = 3.48, 1.44


def tidy(axis, mesh):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver

    td.config.simulation.use_local_subpixel = False
    cell = [APERTURE]*3
    cell[axis] = .5
    guide = [WIDTH, WIDTH, HEIGHT]
    guide[axis] = td.inf
    plane = [APERTURE]*3
    plane[axis] = 0.
    sim = td.Simulation(size=cell,medium=td.Medium(permittivity=CLAD**2),run_time=1e-12,
        structures=[td.Structure(geometry=td.Box(size=guide),medium=td.Medium(permittivity=CORE**2))],
        grid_spec=td.GridSpec.uniform(dl=1/mesh),boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()))
    frequencies = td.C_0/WAVELENGTHS
    data = ModeSolver(simulation=sim,plane=td.Box(size=plane),freqs=frequencies.tolist(),
        mode_spec=td.ModeSpec(num_modes=3,target_neff=CORE-.01,precision='double')).solve()
    indices, fractions = [], []
    for frequency in frequencies:
        energy = np.stack([np.sum(abs(getattr(data,c).sel(f=frequency).values)**2,axis=(0,1,2)) for c in ('Ex','Ey','Ez')])
        fraction = energy[1-axis]/(energy[1-axis]+energy[2])
        ns = np.asarray(data.n_eff.sel(f=frequency))
        valid = np.flatnonzero((ns>CLAD)&(fraction>.5))
        if not valid.size:
            raise ValueError('No guided quasi-TE mode on this local Tidy3D plane.')
        i = valid[np.argmax(ns[valid])]
        indices.append(float(ns[i]))
        fractions.append(float(fraction[i]))
    return {'neff':indices,'inplane_transverse_e_fraction':fractions}


def mpb(axis, mesh):
    import meep as mp
    from meep import mpb

    mp.verbosity(0)
    sizes = [APERTURE]*3
    sizes[axis] = 0.
    gs = [WIDTH, WIDTH, HEIGHT]
    gs[axis] = mp.inf
    normal = [0.,0.,0.]
    normal[axis] = 1.
    model = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(*sizes)),resolution=mesh,
        geometry=[mp.Block(size=mp.Vector3(*gs),material=mp.Medium(index=CORE))],default_material=mp.Medium(index=CLAD),
        num_bands=1,deterministic=True,tolerance=1e-10,mesh_size=5)
    indices, fractions = [], []
    for f in 1/WAVELENGTHS:
        def capture(ms, band):
            e = np.asarray(ms.get_efield(band,bloch_phase=False))
            energy = np.sum(abs(e)**2,axis=tuple(range(e.ndim-1)))
            fractions.append(float(energy[1-axis]/(energy[1-axis]+energy[2])))
        k = model.find_k(mp.EVEN_Z,float(f),1,1,mp.Vector3(*normal),1e-9,2.4*f,CLAD*f*1.00001,CORE*f,capture)[0]
        indices.append(float(k/f))
    if len(fractions)!=len(indices) or min(fractions)<.5:
        raise ValueError('The independent MPB mode has the wrong transverse polarization.')
    return {'neff':indices,'inplane_transverse_e_fraction':fractions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=('tidy3d','mpb'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--meshes',nargs='+',type=int,choices=(32,64,96),default=[32,64])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output filename.')
    if args.engine=='tidy3d':
        import tidy3d as td
        version = td.__version__
        solver = tidy
    else:
        import meep as mp
        version = mp.__version__
        solver = mpb
    rows = {}
    for mesh in args.meshes:
        rows[str(mesh)] = {}
        for axis in range(2):
            data = solver(axis,mesh)
            rows[str(mesh)]['xy'[axis]] = data
            print(args.engine,mesh,'xy'[axis],data,flush=True)
    report = {'engine':args.engine,'version':version,'device':'CPU','wavelength_um':WAVELENGTHS.tolist(),
        'width_um':WIDTH,'height_um':HEIGHT,'aperture_um':APERTURE,'core_index':CORE,'cladding_index':CLAD,
        'scope':'Bound fundamental transverse mode of two perpendicular straight leads. Tidy3D is set up independently but FDTDX also calls Tidy3D internally; MPB uses a separate plane-wave eigensolver.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
