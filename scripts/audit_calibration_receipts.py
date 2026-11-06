"""Verify published calibration archives, matrix samples, paired target metrics and UI records."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.build_calibration_index import BASE, build


def read(name):
    return json.loads((BASE / name).read_text())


def verify_archives():
    archives = {}
    for file in sorted(BASE.glob("v011-*.json")):
        record = read(file.name)
        if not isinstance(record, dict):
            continue
        source = record.get("source_arrays", {})
        name = source.get("file") or record.get("arrays_file")
        expected = source.get("sha256") or record.get("arrays_sha256")
        if not name and expected and file.with_suffix(".npz").is_file():
            name = file.with_suffix(".npz").name
        if not name or not expected:
            continue
        path = BASE / name
        if not path.is_file() or path.resolve().parent != BASE.resolve():
            raise ValueError(f"Numerical archive missing for {file.name}: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Numerical archive changed: {name}")
        with np.load(path, allow_pickle=False) as data:
            if not data.files:
                raise ValueError(f"Empty numerical archive: {name}")
            for key in data.files:
                if data[key].dtype.hasobject:
                    raise ValueError(f"Unexpected object payload: {name}/{key}")
        archives[name] = actual
    unreferenced = {file.name for file in BASE.glob("v011-*.npz")} - archives.keys()
    if unreferenced:
        raise ValueError(f"Public numerical archives without a verified hash: {sorted(unreferenced)}")
    return archives


def target_matrix(case, target, resolution, ports):
    sample = read(f"v011-fdtdx-yee-{case}-{target}{resolution}.json")
    values = np.asarray(sample["s"])
    if not sample["settled"] or values.shape != (3, ports, ports, 2) or not np.isfinite(values).all():
        raise ValueError(f"Incomplete {target} matrix for {case} at {resolution} cells/µm")
    return values[..., 0] + 1j * values[..., 1]


def audit():
    arrays = verify_archives()
    index = build()
    if index != read("v011-calibrated-index.json") or len(index) != 6:
        raise ValueError("Bundled reference index differs from the verified measurements.")

    pairs = {}
    for case, ports in [("coupler", 4), ("splitter", 3)]:
        for resolution in (12, 16):
            cpu = target_matrix(case, "cpu", resolution, ports)
            gpu = target_matrix(case, "gpu", resolution, ports)
            pair = read(f"v011-fdtdx-yee-{case}-cpu-cuda{resolution}.json")
            if not pair["passed"]:
                raise ValueError(f"CPU and NVIDIA comparison failed for {case} at {resolution} cells/µm")
            max_complex = float(abs(cpu - gpu).max())
            max_power = float(abs(abs(cpu) ** 2 - abs(gpu) ** 2).max())
            np.testing.assert_allclose(max_complex, pair["max_complex_s_difference"], rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(max_power, pair["max_modal_power_difference"], rtol=1e-10, atol=1e-14)
            pairs[f"{case}-{resolution}"] = {
                "matrix_coefficients": cpu.size,
                "max_complex_difference": max_complex,
                "max_power_fraction_difference": max_power,
            }

    for filename in ("v011-multimode-planar.json", "v011-multimode-finite.json"):
        record = read(filename)
        matrix = np.array(record["s_real"]) + 1j * np.array(record["s_imag"])
        np.testing.assert_allclose(abs(matrix) ** 2, record["power"], atol=1e-14, rtol=1e-12)
        with np.load(BASE / record["source_arrays"]["file"], allow_pickle=False) as source:
            np.testing.assert_allclose(matrix, source[f"mesh{record['mesh']}_s_electric"], atol=1e-14, rtol=1e-12)

    return {
        "passed": True,
        "native_archives_verified": arrays,
        "complete_cpu_cuda_pairs": pairs,
        "recorded_presets": {name: len(case["recorded"]) for name, case in index.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(result['native_archives_verified'])} numerical archives, "
          f"8 full target matrices and 6 displayed presets.")
