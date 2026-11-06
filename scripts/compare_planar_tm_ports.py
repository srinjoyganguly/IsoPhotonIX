"""Measure identical TM sections in local Meep/MPB and Tidy3D and compare vendor references."""
import argparse
import json
from pathlib import Path

import numpy as np

WAVELENGTH = 1.55
LUMERICAL = "https://optics.ansys.com/hc/en-us/articles/360042304134-Asymmetric-slab-dielectric-waveguide"
INDEX_FILE = "https://optics.ansys.com/hc/en-us/article_attachments/360056405454"
GEOMETRY_FILE = "https://optics.ansys.com/hc/en-us/article_attachments/360057294573"
TIDY3D = "https://www.flexcompute.com/tidy3d/examples/notebooks/ModalSourcesMonitors/"
REFERENCE = np.array([1.433098106928464, 1.412444019217784, 1.378468071822638, 1.334814055288823])
CASES = {
    "lumerical_asymmetric": {"width":5., "core":1.44, "lower":1.33, "upper":1., "span":36., "modes":4},
    "workbench_joint": {"width":.375, "core":3.48, "lower":1.44, "upper":1.44, "span":8., "modes":2},
}


def measure_meep(case, mesh):
    import meep as mp

    mp.verbosity(0)
    span, width = case["span"], case["width"]
    core = mp.Block(size=mp.Vector3(mp.inf,width,mp.inf), material=mp.Medium(index=case["core"]))
    substrate_depth = (span-width) / 2
    substrate = mp.Block(size=mp.Vector3(mp.inf,substrate_depth,mp.inf),
        center=mp.Vector3(0, -(width+substrate_depth)/2),material=mp.Medium(index=case["lower"]))
    sim = mp.Simulation(cell_size=mp.Vector3(1,span),dimensions=2,resolution=mesh,
                        geometry=[core,substrate],default_material=mp.Medium(index=case["upper"]))
    try:
        sim.init_sim()
        values = []
        for i in range(case["modes"]):
            mp.set_random_seed(0)
            guess = REFERENCE[i] if case["modes"] == 4 else (2.92,1.54)[i]
            mode = sim.get_eigenmode(1/WAVELENGTH,mp.X,mp.Volume(center=mp.Vector3(),
                    size=mp.Vector3(0,span)),i+1,mp.Vector3(guess/WAVELENGTH),
                    parity=mp.EVEN_Z,resolution=mesh,eigensolver_tol=1e-14)
            neff = mode.k.x * WAVELENGTH
            if neff <= max(case["lower"],case["upper"]):
                raise ValueError("Requested TM mode is unbound.")
            values.append(float(neff))
        return values
    finally:
        sim.reset_meep()


def measure_tidy3d(case, mesh):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver

    td.config.simulation.use_local_subpixel = False
    width, span = case["width"], case["span"]
    substrate_depth = (span-width) / 2
    structures = [td.Structure(geometry=td.Box(size=(td.inf,width,td.inf)),
                    medium=td.Medium(permittivity=case["core"]**2)),
                  td.Structure(geometry=td.Box(size=(td.inf,substrate_depth,td.inf),
                    center=(0,-(width+substrate_depth)/2,0)),
                    medium=td.Medium(permittivity=case["lower"]**2))]
    simulation = td.Simulation(size=(1,span,0),structures=structures,medium=td.Medium(permittivity=case["upper"]**2),
            grid_spec=td.GridSpec.uniform(dl=1/mesh),run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()))
    data = ModeSolver(simulation=simulation,plane=td.Box(size=(0,span,td.inf)),
            freqs=[td.C_0/WAVELENGTH],mode_spec=td.ModeSpec(num_modes=case["modes"]*2+2,
                target_neff=case["core"],precision="double")).solve()
    values = []
    for index, neff in enumerate(data.n_eff.values.ravel()):
        powers = np.array([np.linalg.norm(getattr(data,f).isel(mode_index=index,f=0).values)**2
                           for f in ("Hx","Hy","Hz")])
        if neff > max(case["lower"],case["upper"])+1e-4 and powers[2]/powers.sum() > .999:
            values.append(float(neff))
    if len(values) != case["modes"]:
        raise ValueError(f"Unexpected guided TM mode count: {values}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver",required=True,choices=("meep","tidy3d"))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh file for the measured indices.")
    if args.solver == "meep":
        import meep
        version, measure = meep.__version__, measure_meep
    else:
        import tidy3d
        version, measure = tidy3d.__version__, measure_tidy3d
    results = {}
    for name, case in CASES.items():
        results[name] = {}
        for mesh in (64,128):
            values = measure(case,mesh)
            print(name,mesh,values,flush=True)
            results[name][mesh] = values
    args.output.write_text(json.dumps({"solver":args.solver,"version":version,"platform":"local CPU",
        "case_settings":CASES,"wavelength_um":WAVELENGTH,"indices_by_mesh":results,
        "lumerical_analytic_reference":REFERENCE.tolist(),"sources":[LUMERICAL,INDEX_FILE,GEOMETRY_FILE,TIDY3D]},
        indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
