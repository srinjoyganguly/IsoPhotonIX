#!/usr/bin/env python3
"""Run and verify one genuine FDTDX solve in this Python environment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.adapters.fdtdx_adapter import run
from backend.model import WaveguideConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--config", type=Path, default=Path("examples/slab-waveguide.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-width-um", type=float)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(solver="fdtdx", device=args.device)
    if args.core_width_um is not None:
        config["core_width_um"] = args.core_width_um
    config = WaveguideConfig.model_validate(config).model_dump()
    metadata = run(config, args.output)

    import numpy as np
    with np.load(args.output / "arrays.npz", allow_pickle=False) as arrays:
        field, epsilon, frames = arrays["field"], arrays["epsilon"], arrays["frames"]
        assert field.shape == epsilon.shape, "Field and material grids must align"
        assert field.shape == (len(arrays["y_um"]), len(arrays["x_um"]))
        assert np.isfinite(field).all() and np.isfinite(frames).all(), "Non-finite fields"
        assert np.max(np.abs(field)) > 1e-12, "Zero field"
        assert frames.shape[0] > 1 and frames.shape[1:] == field.shape
        assert np.all(np.diff(arrays["times_fs"]) > 0), "Nonmonotonic frame times"
        expected_epsilon = sorted([config["cladding_index"] ** 2, config["core_index"] ** 2])
        assert np.allclose(np.unique(epsilon), expected_epsilon), "Material grid is incorrect"
        assert metadata["device"] == args.device
        assert metadata["actual_device"].startswith("cuda" if args.device == "gpu" else "cpu")
    metadata["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"verified": True, **metadata}, indent=2))


if __name__ == "__main__":
    main()
