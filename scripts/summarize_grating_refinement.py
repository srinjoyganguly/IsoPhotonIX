"""Compare the same five physical wavelengths across full 3D grating meshes."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_grating_domain import read


def summarize(paths):
    groups = {}
    for path in paths:
        row, arrays = read(path)
        config = row["settings"]
        geometry = config["geometry"]
        if geometry not in ("lumerical", "tidy3d"):
            raise ValueError("Only a geometry-matched published benchmark is comparable.")
        reference = next(v for v in row["comparisons"] if v["geometry_matched"])
        waves = np.array(reference["wavelength_um"])
        published = np.asarray(reference["published_s21_power"] if geometry == "lumerical" else reference["published_power"])
        with np.load(arrays, allow_pickle=False) as data:
            np.testing.assert_array_equal(waves, data["dev_wavelength_um"])
            p = np.abs(data["output_mode"])**2/data["incident_power"]
            np.testing.assert_allclose(p, row["result"]["beam_to_guide"])
            weak_return = (abs(data["reflected_fiber_mode"])**2).tolist() if geometry == "lumerical" else None
            neff = data["dev_mode_neff"].real.tolist()
        delta = 100*(p-published)
        physical_etch_nm = 1000*config["etch"]
        sample = {"mesh":config["resolution"], "spacing_nm":1000/config["resolution"],
                  "etch_nm":physical_etch_nm,"etch_cells":config["etch"]*config["resolution"],
                  "arrays_sha256":row["arrays_sha256"], "wavelength_um":waves.tolist(),
                  "power":p.tolist(), "published_power":published.tolist(), "delta_percentage_points":delta.tolist(),
                  "mean_absolute_delta_pp":float(np.mean(abs(delta))), "max_absolute_delta_pp":float(max(abs(delta))),
                  "fiber_return_power":weak_return, "port_neff":neff,
                  "stages":{k:{"shape":v["shape"],"steps":v["time_steps"],"freshness":v.get("acquisition"),
                               "performance":v.get("performance")} for k,v in row["stages"].items()}}
        groups.setdefault(geometry,[]).append(sample)
    for runs in groups.values():
        runs.sort(key=lambda r:r["mesh"])
        for r1, r2 in zip(runs,runs[1:]):
            np.testing.assert_array_equal(r1["wavelength_um"],r2["wavelength_um"])
            d=100*(np.asarray(r2["power"])-r1["power"])
            r2["previous_mesh"]=r1["mesh"]
            r2["change_vs_previous_pp"]=d.tolist()
            r2["max_absolute_change_vs_previous_pp"]=float(max(abs(d)))
    return {"schema_version":1,"observables":"Independently illuminated full-width gratings. Power is output TE0 power divided by measured incoming finite-aperture power. Each reference belongs to its own published design.",
            "tidy_readout_uncertainty_db":0.08,"groups":groups}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.receipts),indent=2,allow_nan=False)+"\n")
