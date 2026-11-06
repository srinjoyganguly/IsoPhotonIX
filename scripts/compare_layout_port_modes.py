"""Compare the default elbow's two lead modes with a local Tidy3D mode solve."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def index_for_axis(td, axis, mesh, wavelengths, width, aperture):
    from tidy3d.plugins.mode import ModeSolver

    # The geometry and monitor rotate together: propagation is along x or y.
    transverse = 1 - axis
    cell = [1., 1., 0.]
    cell[transverse] = aperture
    guide = [td.inf, td.inf, td.inf]
    guide[transverse] = width
    plane = [aperture, aperture, td.inf]
    plane[axis] = 0
    simulation = td.Simulation(
        size=cell, medium=td.Medium(permittivity=1.44**2),
        structures=[td.Structure(geometry=td.Box(size=guide), medium=td.Medium(permittivity=3.48**2))],
        grid_spec=td.GridSpec.uniform(dl=1/mesh), run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )
    freqs = td.C_0 / wavelengths
    data = ModeSolver(simulation=simulation, plane=td.Box(size=plane), freqs=freqs.tolist(),
                      mode_spec=td.ModeSpec(num_modes=4, target_neff=3.48, precision="double")).solve()
    indices = []
    for freq in freqs:
        neff = data.n_eff.sel(f=freq)
        energies = np.array([
            np.sum(abs(getattr(data, field).sel(f=freq).values)**2, axis=(0, 1, 2))
            for field in ("Ex", "Ey", "Ez")
        ])
        guided = np.asarray(neff)[(energies[2]/energies.sum(axis=0) > .999) & (np.asarray(neff) > 1.44)]
        if not len(guided):
            raise ValueError("The local mode solver found no bound Ez mode.")
        indices.append(float(max(guided)))
    return indices


def measure():
    import tidy3d as td
    td.config.simulation.use_local_subpixel = False
    wavelengths = np.array([1.60, 1.55, 1.50])
    width, aperture = .375, 2.
    values = {str(mesh): {name: index_for_axis(td, axis, mesh, wavelengths, width, aperture)
                          for axis, name in enumerate(("x", "y"))} for mesh in (32, 40, 128)}
    return {"solver": "Tidy3D local ModeSolver", "version": td.__version__, "actual_device": "CPU",
            "scope": "Bound fundamental Ez index of each straight elbow lead. Full polygon scattering is independently checked with a frequency-domain Helmholtz solve.",
            "width_um": width, "port_aperture_um": aperture, "core_index": 3.48, "cladding_index": 1.44,
            "wavelength_um": wavelengths.tolist(), "neff_by_mesh_and_axis": values,
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path.")
    args.output.write_text(json.dumps(measure(), indent=2, allow_nan=False) + "\n")
