"""Measure how the finite grid represents the calibrated coupler's central section."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "backend").is_dir():
    sys.path.insert(0, str(ROOT))
from scripts.calibrate_fdtdx_multiport import _ports_runtime, grid, multiport_model, raster, yee_material


def central_section(model, resolution):
    """Use the same physical raster as the finite device, restricted to the 3 µm mode window."""
    shape, *_ = grid(model, resolution)
    section = raster(model, resolution)[shape[0] // 2]
    window = 3 * resolution
    y0, z0 = ((width - window) // 2 for width in shape[1:])
    core = section[y0:y0+window, z0:z0+window]
    mask = core[:, window // 2]
    edges = np.flatnonzero(mask != np.roll(mask, 1))
    if len(edges) != 4:
        raise ValueError("Expected two distinct silicon rails at the central plane.")
    actual_gap = (edges[2] - edges[1]) / resolution
    epsilon = np.where(core, model["core_index"]**2, model["cladding_index"]**2)[None].astype(np.float32)
    return yee_material(epsilon, "yee-aligned"), float(actual_gap)


def measure(runtime, model, resolution):
    from fdtdx.core.physics.modes import compute_mode

    _, jnp, _, _ = runtime
    material, gap = central_section(model, resolution)
    inverse_epsilon = jnp.asarray(1 / material)
    indices, parities, te_fractions = {}, {}, {}
    for index in (0, 1):
        electric, _, mode_index = compute_mode(
            frequency=299792458e6 / 1.55,
            inv_permittivities=inverse_epsilon,
            inv_permeabilities=1.,
            resolution=1e-6 / resolution,
            direction="+",
            mode_index=index,
            filter_pol="te",
            dtype=jnp.float32,
        )
        electric = np.asarray(electric)[:, 0]
        ey = electric[1]
        ey_fraction = float(np.linalg.norm(ey)**2 / max(np.linalg.norm(electric)**2, 1e-18))
        parity = float(np.vdot(ey, ey[::-1]).real / np.vdot(ey, ey).real)
        label = "even" if parity > 0 else "odd"
        if label in indices or abs(parity) < 0.8 or ey_fraction < 0.5:
            raise ValueError("The two local modes must have distinct parity and quasi-TE polarization.")
        indices[label] = float(mode_index.real)
        parities[label] = parity
        te_fractions[label] = ey_fraction
    if min(indices.values()) <= model["cladding_index"] or indices["even"] <= indices["odd"]:
        raise ValueError("The section must support both guided supermodes.")
    return {
        "resolution": resolution,
        "effective_scalar_center_gap_um": gap,
        "indices": indices,
        "Ey_mirror_parity": parities,
        "Ey_fraction": te_fractions,
        "uniform_transfer_length_um": 1.55 / (2 * (indices["even"] - indices["odd"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output file.")
    runtime = _ports_runtime("cpu")
    model = multiport_model("coupler")
    rows = []
    for resolution in (12, 16, 20, 24, 32):
        row = measure(runtime, model, resolution)
        print(row, flush=True)
        rows.append(row)
    report = {
        "case": "coupler",
        "material_model": "yee-aligned",
        "solver": "FDTDX local mode preparation using its bundled Tidy3D mode solver",
        "actual_device": str(runtime[-1]),
        "nominal_gap_um": 0.125,
        "wavelength_um": 1.55,
        "mode_aperture_um": [3., 3.],
        "rows": rows,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Uniform transfer length from even/odd phase indices in the local central section. The finite coupler has tapered approaches and separate measured scattering matrices.",
    }
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
