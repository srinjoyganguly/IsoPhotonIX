"""Bounded access to immutable numerical 3D fields, in [z,y,x] order."""
from functools import lru_cache
from pathlib import Path

import numpy as np

COMPONENTS = ("Ex", "Ey", "Ez")


def validate_volume(arrays):
    required = ("field", "fields", "epsilon", "x_um", "y_um", "z_um")
    result = {name: arrays[name] for name in required}
    if any(value.dtype.kind not in "fiu" for value in result.values()):
        raise ValueError("3D snapshots must contain real numeric samples, not complex or object values.")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("3D results contain non-finite values.")
    field, fields, epsilon = result["field"], result["fields"], result["epsilon"]
    axes = [result[f"{axis}_um"] for axis in "xyz"]
    if any(axis.ndim != 1 or len(axis) < 2 or np.any(np.diff(axis) <= 0) for axis in axes):
        raise ValueError("3D result axes must be finite, increasing one-dimensional coordinates.")
    shape = tuple(len(axis) for axis in reversed(axes))
    if field.ndim != 3 or field.shape != shape or epsilon.shape != shape or fields.shape != (3, *shape):
        raise ValueError("3D field, material and coordinate shapes do not agree.")
    if field.size > 1_200_000:
        raise ValueError("Saved 3D grid exceeds the supported output limit.")
    if not np.array_equal(field, fields[1]):
        raise ValueError("The primary 3D field must be the recorded Ey component.")
    if not np.any(field) or np.any(epsilon < 1):
        raise ValueError("Expected a nonzero Ey field and positive dielectric material grid.")
    for value in result.values():
        value.setflags(write=False)
    return result


@lru_cache(maxsize=2)
def _stored_volume(filename, modified_ns, size):
    del modified_ns, size
    with np.load(filename, allow_pickle=False) as arrays:
        return validate_volume(arrays)


def stored_volume(output):
    path = Path(output) / "arrays.npz"
    stat = path.stat()
    return _stored_volume(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def volume_info(data):
    peaks = {name: float(np.max(np.abs(data["fields"][i]))) for i, name in enumerate(COMPONENTS)}
    return {"components": list(COMPONENTS), "shape": list(data["field"].shape),
            **{f"{a}_um": data[f"{a}_um"].tolist() for a in "xyz"},
            "field_peak_abs": peaks["Ey"], "field_peaks_by_component": peaks}


def volume_slice(data, plane="xy", index=None, component="Ey"):
    if plane not in ("xy", "xz", "yz") or component not in COMPONENTS:
        raise ValueError("Choose XY, XZ or YZ and Ex, Ey or Ez.")
    u_axis, v_axis, normal = {"xy": ("x", "y", "z"), "xz": ("x", "z", "y"), "yz": ("y", "z", "x")}[plane]
    coordinate = data[f"{normal}_um"]
    if index is None:
        index = int(np.argmin(np.abs(coordinate)))
    if index < 0 or index >= len(coordinate):
        raise IndexError("Slice index is outside this run's recorded grid.")
    selection = {"xy": (index, slice(None), slice(None)),
                 "xz": (slice(None), index, slice(None)),
                 "yz": (slice(None), slice(None), index)}[plane]
    field = data["fields"][COMPONENTS.index(component)][selection]
    epsilon = data["epsilon"][selection]
    return {"field": field.tolist(), "epsilon": epsilon.tolist(),
            "u_um": data[f"{u_axis}_um"].tolist(), "v_um": data[f"{v_axis}_um"].tolist(),
            "u_axis": u_axis, "v_axis": v_axis, "normal_axis": normal,
            "coordinate_um": float(coordinate[index]), "index": index, "plane": plane, "component": component}
