"""Isolated numerical worker. Exits nonzero if output or placement is invalid."""
import importlib
import json
import sys
from pathlib import Path


def main():
    config_path, output = map(Path, sys.argv[1:3])
    config = json.loads(config_path.read_text())
    if config["solver"] not in ("meep", "fdtdx"):
        raise ValueError("Unknown solver")
    print(f"Starting {config['solver']} on explicitly requested {config['device']}", flush=True)
    adapter = importlib.import_module(f"backend.adapters.{config['solver']}_adapter")
    metadata = adapter.run(config, output)
    import numpy as np
    with np.load(output/"arrays.npz") as data:
        for name in ("field", "epsilon", "x_um", "y_um"):
            if name not in data or not np.isfinite(data[name]).all():
                raise RuntimeError(f"Invalid or missing numerical array: {name}")
        if data["field"].ndim != 2 or data["field"].shape != data["epsilon"].shape:
            raise RuntimeError("Field and permittivity must be matching 2D arrays.")
        if data["field"].shape != (len(data["y_um"]), len(data["x_um"])):
            raise RuntimeError("Array shape does not match coordinate axes.")
        if not np.any(data["field"]):
            raise RuntimeError("Simulation returned only zero fields.")
        metadata["field_peak_abs"] = float(np.max(np.abs(data["field"])))
        metadata["field_shape"] = list(data["field"].shape)
    metadata.update(config=config, requested_solver=config["solver"], requested_device=config["device"])
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print("Simulation complete; synchronized numerical arrays and provenance saved.", flush=True)


if __name__ == "__main__":
    main()
