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
    is3d = config.get("schema_version") == 2 and config.get("study_type") == "waveguide-3d"
    isports = config.get("schema_version") == 3 and config.get("study_type") == "waveguide-ports-2d"
    adapter_name = f"{config['solver']}_ports" if isports else f"{config['solver']}3d" if is3d else config['solver']
    adapter = importlib.import_module(f"backend.adapters.{adapter_name}_adapter")
    metadata = adapter.run(config, output)
    import numpy as np
    with np.load(output/"arrays.npz") as data:
        for name in ("field", "epsilon", "x_um", "y_um"):
            if name not in data or not np.isfinite(data[name]).all():
                raise RuntimeError(f"Invalid or missing numerical array: {name}")
        if is3d:
            from .volume import validate_volume
            validate_volume(data)
        elif data["field"].ndim != 2 or data["field"].shape != data["epsilon"].shape:
            raise RuntimeError("Field and permittivity must be matching 2D arrays.")
        if not is3d and data["field"].shape != (len(data["y_um"]), len(data["x_um"])):
            raise RuntimeError("Array shape does not match coordinate axes.")
        if not isports and not np.any(data["field"]):
            raise RuntimeError("Simulation returned only zero fields.")
        metadata["field_peak_abs"] = float(np.max(np.abs(data["field"])))
        metadata["field_shape"] = list(data["field"].shape)
    metadata.update(config=config, requested_solver=config["solver"], requested_device=config["device"])
    if isports:
        from .spectra import stored_spectrum, spectrum_csv
        spectra = stored_spectrum(output, metadata)
        (output/"spectrum.csv").write_text(spectrum_csv(spectra))
    if is3d:
        metadata["commercial_solver_agreement"] = {"status": "not_validated", "reference_runs": [],
            "note": "Execution and internal numerical checks do not establish agreement with Lumerical, Tidy3D or COMSOL. Matched sources, sampling, converged reference data and justified tolerances are required."}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print("Simulation complete; synchronized numerical arrays and provenance saved.", flush=True)


if __name__ == "__main__":
    main()
