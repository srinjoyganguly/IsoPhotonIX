"""Reproject native grating fields onto a strictly local Tidy3D model of the published receiver."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from backend.adapters.fdtdx_ports3d_adapter import modal_amplitudes
from scripts.audit_grating_vendor_mesh import tidy_simulation
from scripts.calibrate_finite_grating import WL, settings
from scripts.compare_grating_domain import read


def receiver():
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver

    td.config.simulation.use_local_subpixel=False
    model=tidy_simulation(settings("tidy3d",12))
    monitor=model.monitors[0]
    solver=ModeSolver(simulation=model,plane=td.Box(center=monitor.center,size=monitor.size),
                      freqs=td.C_0/WL,mode_spec=monitor.mode_spec,direction="+")
    result=solver.solve()
    neff=np.asarray(result.n_eff).ravel()
    if not np.all((1.44<neff) & (neff<3.47)):
        raise ValueError("Local published receiver mode lies outside the guided interval.")
    return result, neff


def measure(path, modes, neff):
    import tidy3d as td
    receipt,archive=read(path);m=receipt["settings"]
    if m["geometry"]!="tidy3d":
        raise ValueError("Expected the finite Tidy3D grating.")
    with np.load(archive,allow_pickle=False) as d:
        np.testing.assert_array_equal(d["dev_wavelength_um"],WL)
        dx=1/m["resolution"]
        phasors=d["dev_output"]; py,pz=phasors.shape[-2:]
        y,z=d["dev_grid_y_um"],d["dev_grid_z_um"]
        sy=(len(y)-py)//2
        sz=int(np.argmin(abs(z-(m["height"]/2-pz*dx/2+dx/2))))
        yy,zz=np.meshgrid(y[sy:sy+py]-dx/2,z[sz:sz+pz],indexing="ij")
        basis=np.empty((2,len(WL),3,py,pz),complex)
        for fi,w in enumerate(WL):
            for kind,name in enumerate(("E","H")):
                for component,axis in enumerate("xyz"):
                    values=getattr(modes,name+axis).sel(f=td.C_0/w,method="nearest").isel(mode_index=0).squeeze().transpose("y","z")
                    obj=RegularGridInterpolator((np.asarray(values.y),np.asarray(values.z)),np.asarray(values),
                                                bounds_error=False,fill_value=0.)
                    basis[kind,fi,component]=obj((yy,zz))*(td.ETA_0 if kind else 1.)
        e,h=basis
        te=np.sum(abs(e[:,1])**2,axis=(1,2))/np.sum(abs(e[:,1:])**2,axis=(1,2,3))
        if np.min(te)<.5: raise ValueError("Expected dominant Ey at the strip receiver.")
        a=modal_amplitudes(phasors,e,h,dx**2)[0][:,1]
        local=d["dev_mode_e"][:,1]
        product=np.sum(local.conj()*e[:,1],axis=(1,2))
        overlap=abs(product)**2/(np.sum(abs(local)**2,axis=(1,2))*np.sum(abs(e[:,1])**2,axis=(1,2)))
        power=abs(a)**2/d["incident_power"]
        return {"mesh":m["resolution"],"domain":m.get("grating_domain","full"),"arrays_sha256":receipt["arrays_sha256"],
                "native_neff":d["dev_mode_neff"].real.tolist(),"sdk_neff":neff.tolist(),
                "native_power":d["coupling"].tolist(),"reprojected_power":power.tolist(),"ey_shape_overlap":overlap.tolist(),
                "reprojection_minus_native_pp":(100*(power-d["coupling"])).tolist()}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipts",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    modes,indices=receiver()
    record={"sdk_version":importlib.metadata.version("tidy3d"),"source":"https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/",
            "wavelength_um":WL.tolist(),"scope":"Client-side CPU ModeSolver on the reconstructed published monitor aperture and automatic grid, with local subpixel extension disabled. This is an independent decomposition of actual FDTDX E/H fields, distinct from the remote published 3D solve.",
            "examples":[measure(path,modes,indices) for path in args.receipts]}
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
