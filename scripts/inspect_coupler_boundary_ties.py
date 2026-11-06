"""Audit material-interface ties in the measured 24-cell coupler raster without running a solver."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "backend").is_dir():
    sys.path.insert(0, str(ROOT))
from scripts.calibrate_fdtdx_multiport import digest, grid, multiport_model, raster


def audit(measured_directory=None):
    model = multiport_model("coupler")
    resolution = 24
    shape, _, (xs, ys, zs), _ = grid(model, resolution)
    original = raster(model, resolution)
    exact = np.zeros(shape[:2], dtype=bool)
    near = np.zeros_like(exact)
    for polygon in model["polygons"]:
        for (x0, y0), (x1, y1) in zip(polygon, polygon[1:]+polygon[:1]):
            crossing = (xs >= min(x0, x1)) & (xs < max(x0, x1))
            if not crossing.any():
                continue
            edge = y0+(xs[crossing]-x0)*(y1-y0)/(x1-x0)
            distances = abs(ys[None, :]-edge[:, None])
            exact[crossing] |= distances == 0.
            near[crossing] |= (distances > 0.) & (distances < 1e-13)
    zcore = (zs >= -model["height"]/2) & (zs < model["height"]/2)
    boundary = exact[:, :, None] & zcore[None, None, :]
    mismatch = original != original[:, ::-1, :]
    if np.any(mismatch & ~boundary):
        raise ValueError("A mirror difference occurs away from an exact evaluated interface.")
    closed = original | boundary
    opened = original & ~boundary
    fractional = original.astype(float)
    fractional[boundary] = 0.5
    xcenter, zcenter = shape[0]//2, shape[2]//2
    line = original[xcenter, :, zcenter]
    edges = np.flatnonzero(line != np.roll(line, 1))
    if len(edges) != 4:
        raise ValueError("Expected two separate rails at the central section.")
    rails = []
    for side, lower, upper in (("lower", edges[0], edges[1]), ("upper", edges[2], edges[3])):
        samples = ys[lower:upper]
        nominal_center = (-1 if side == "lower" else 1)*(.5+.125)/2
        rails.append({
            "side": side,
            "nominal_center_um": nominal_center,
            "sample_centroid_um": float(np.mean(samples)),
            "centroid_shift_um": float(np.mean(samples)-nominal_center),
            "half_open_sample_count": int(upper-lower),
            "half_open_width_um": float((upper-lower)/resolution),
            "closed_width_um": float(np.count_nonzero(closed[xcenter, :, zcenter] & (ys < 0 if side=="lower" else ys > 0))/resolution),
            "open_width_um": float(np.count_nonzero(opened[xcenter, :, zcenter] & (ys < 0 if side=="lower" else ys > 0))/resolution),
        })
    ix, iy, iz = np.where(mismatch)
    report = {
        "resolution": resolution,
        "grid_shape": shape,
        "raster_source_sha256": hashlib.sha256(inspect.getsource(raster).encode()).hexdigest(),
        "nominal_geometry": {"width_um":0.5, "gap_um":0.125, "height_um":0.25},
        "mirror_disagreeing_voxels": int(mismatch.sum()),
        "mismatch_only_at_exact_evaluated_boundary": True,
        "mismatched_y_um": np.unique(ys[iy]).tolist(),
        "mismatched_x_cell_count": int(len(np.unique(ix))),
        "mismatched_z_cell_count": int(len(np.unique(iz))),
        "mismatched_y_cell_count": int(len(np.unique(iy))),
        "mismatched_x_range_um": [float(xs[ix].min()), float(xs[ix].max())],
        "exact_y_interface_voxels": int(boundary.sum()),
        "near_but_unequal_interface_voxels_within_1e_minus_13_um": int(near.sum()*zcore.sum()),
        "symmetric_sample_coordinates": bool(np.array_equal(ys, -ys[::-1])),
        "central_rails": rails,
        "diagnostic_masks": {
            "half_open": {"mirror_disagreements":int(mismatch.sum()), "core_voxel_sum":float(original.sum())},
            "closed": {"mirror_disagreements":int(np.count_nonzero(closed!=closed[:, ::-1, :])), "core_voxel_sum":float(closed.sum())},
            "open": {"mirror_disagreements":int(np.count_nonzero(opened!=opened[:, ::-1, :])), "core_voxel_sum":float(opened.sum())},
            "half_weighted_ties": {"mirror_disagreements":int(np.count_nonzero(fractional!=fractional[:, ::-1, :])), "core_voxel_sum":float(fractional.sum())},
        },
        "interpretation": "All 1,152 asymmetric voxels occur at exactly equal interface/sample coordinates under the explicit half-open interval convention. Changing equality to closed or open restores mask symmetry while changing the central represented widths to 13 or 11 cells. Half-weighted interface samples preserve total material volume and symmetry, but would constitute a separately validated material representation.",
        "scope": "Static diagnostics on the existing sampled geometry. The recorded native electromagnetic fields and their source model are preserved. No electromagnetic solve used the alternative masks in this report.",
    }
    if measured_directory:
        native = json.loads((measured_directory/"left_lower-device.json").read_text())
        epsilon = np.where(original, model["core_index"]**2, model["cladding_index"]**2).astype(np.float32)
        if digest(epsilon) != native["material_grid_sha256"]:
            raise ValueError("The diagnostic mask differs from the measured material grid.")
        report["recorded_native_material_grid_sha256"] = native["material_grid_sha256"]
        report["recorded_native_field_archive_sha256"] = native["archive_sha256"]
        report["measured_mask_verified"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh diagnostic report file.")
    result = audit(args.measured_directory)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
