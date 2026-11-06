"""Check complete component-wise Yee material symmetry before projecting fine half-grid fields."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_finite_grating import aligned, digest, grid, material, settings
from scripts.lumerical_grating3d import silicon_on_yee_grid


def check(geometry, resolution):
    model="direct-yee" if geometry=="lumerical" else "volume-average"
    m=settings(geometry,resolution,model,"standard","mirror-y")
    shape,_,axes=grid(m)
    middle=shape[1]//2
    stages={}
    for reference in (True,False):
        scalar=material(m,reference)
        components=aligned(scalar)
        if not reference and model=="direct-yee":
            silicon_on_yee_grid(m,axes,components)
        rows=[]
        for component,a in enumerate(components):
            # Ey is half a cell from y=0; Ex and Ez lie at integer y. At the
            # lower outermost integer plane the reflected peer lies beyond the array.
            far=a[:,middle-1::-1,:] if component==1 else a[:,middle:0:-1,:]
            near=a[:,middle:,:]
            delta=near-far
            rows.append({"component":"xyz"[component],"component_permittivity_sha256":digest(a),
                         "maximum_absolute_epsilon_difference":float(abs(delta).max()),
                         "relative_l2_difference":float(np.linalg.norm(delta.ravel())/np.linalg.norm(near.ravel()))})
        stages["reference" if reference else "device"]={"scalar_permittivity_sha256":digest(scalar),"components":rows}
        del scalar,components,a,far,near,delta
    return {"geometry":geometry,"mesh":resolution,"full_shape":shape,"stages":stages}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--receipts",nargs="*",type=Path,default=[])
    args=p.parse_args()
    result={"scope":"Full intended material volumes on the staggered Ex/Ey/Ez lattices, prior to selecting the positive-y half. This is a geometry audit with zero electromagnetic timesteps.",
            "examples":[check("lumerical",20),check("tidy3d",16)]}
    for path in args.receipts:
        run=json.loads(path.read_text());m=run["settings"]
        matches=[item for item in result["examples"] if (item["geometry"],item["mesh"])==(m["geometry"],m["resolution"])]
        if not run["passed"] or m.get("grating_domain")!="mirror-y" or len(matches)!=1:
            raise ValueError("Expected a qualified fine mirror acquisition.")
        target=matches[0]
        for label,expected in target["stages"].items():
            stage=run["stages"][label]
            actual=stage["electric_component_permittivity_sha256"]
            if [row["component_permittivity_sha256"] for row in expected["components"]]!=actual:
                raise ValueError("Native electric-component materials differ from the full-grid audit.")
            if expected["scalar_permittivity_sha256"]!=stage["scalar_permittivity_sha256"]:
                raise ValueError("Native scalar material differs from the full-grid audit.")
        target["matched_native_arrays_sha256"]=run["arrays_sha256"]
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
