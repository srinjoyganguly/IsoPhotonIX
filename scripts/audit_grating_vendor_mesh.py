"""Extract the filed Ansys grating lattice and reconstruct the published Tidy3D automatic grid locally."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from scripts.audit_ring_vendor_project import array
from scripts.calibrate_finite_grating import settings


def at(axis, coordinate):
    i=int(np.searchsorted(axis,coordinate,side="right"))-1
    if i<0 or i+1>=len(axis):
        raise ValueError("Coordinate outside the filed mesh.")
    return float(1000*(axis[i+1]-axis[i]))


def section(axis, lo, hi):
    delta=np.diff(axis)
    overlap=np.maximum(0,np.minimum(axis[1:],hi)-np.maximum(axis[:-1],lo))
    if np.any(delta<=0) or not np.isclose(sum(overlap),hi-lo):
        raise ValueError("Grid does not cover the requested layer.")
    cells=overlap>1e-12
    return {"physical_bounds_um":[lo,hi],"fractional_cell_count":float(sum(overlap/delta)),
            "intersected_steps_nm":(1000*delta[cells]).tolist()}


def geometry_steps(axes, m):
    x,y,z=(np.asarray(a,dtype=float) for a in axes)
    return {"grid_boundary_bounds_um":[[float(a[0]),float(a[-1])] for a in (x,y,z)],
            "sampled_cell_counts":[len(a)-1 for a in (x,y,z)],
            "step_nm_at_beam_x":at(x,m["beam_x"]),"step_nm_at_center_y":at(y,0.),
            "ridge_section":section(z,0.,m["height"]),
            "etched_part":section(z,m["height"]-m["etch"],m["height"])}


def tidy_grid(m):
    import tidy3d as td
    si=td.Medium(permittivity=m["n_si"]**2);ox=td.Medium(permittivity=m["n_oxide"]**2)
    h,etch,width=m["height"],m["etch"],m["width"]
    buf,taper,guide=m["buffer"],m["taper"],m["guide_width"]
    def box(lo,hi,medium):
        return td.Structure(geometry=td.Box.from_bounds(rmin=lo,rmax=hi),medium=medium)
    teeth=[td.Box(center=((i+m["fill"]/2)*m["period"],0,h-etch/2),
                  size=(m["period"]*m["fill"],width,etch)) for i in range(m["teeth"])]
    vertices=((0,width/2),(0,-width/2),(-taper,-guide/2),(-taper-2*buf,-guide/2),
              (-taper-2*buf,guide/2),(-taper,guide/2))
    structures=[box((-1000,-1000,0),(1000,1000,m["oxide_top"]),ox),
                td.Structure(geometry=td.GeometryGroup(geometries=tuple(teeth)),medium=si),
                box((0,-1000,0),(1000,1000,h-etch),si),
                td.Structure(geometry=td.PolySlab(vertices=vertices,axis=2,slab_bounds=(0,h)),medium=si),
                box((-1000,-1000,-m["oxide_bottom"]),(1000,1000,0),ox),
                box((-1000,-1000,-1000),(1000,1000,-m["oxide_bottom"]),si)]
    domain=td.Box.from_bounds(rmin=(-taper-buf,-width/2-buf,-m["oxide_bottom"]-buf),
                              rmax=(m["teeth"]*m["period"]+buf,width/2+buf,h+m["oxide_top"]+buf))
    beam=td.GaussianBeam(size=(2*width,2*width,0),center=(m["beam_x"],0,m["source_z"]),
                         source_time=td.GaussianPulse(freq0=td.C_0/1.55,fwidth=.5*td.C_0*(1/1.5-1/1.6)),
                         angle_theta=np.deg2rad(m["angle_deg"]),direction="-",waist_radius=m["waist"],pol_angle=np.pi/2)
    monitor=td.ModeMonitor(center=(-taper-buf/2,0,h/2),size=(0,4*guide,6*h),
                           freqs=td.C_0/np.linspace(1.5,1.6,101),mode_spec=td.ModeSpec(num_modes=1,target_neff=m["n_si"]),name="mode")
    sim=td.Simulation(center=domain.center,size=domain.size,symmetry=(0,-1,0),
                      grid_spec=td.GridSpec.auto(min_steps_per_wvl=15,wavelength=1.55),
                      structures=structures,sources=[beam],monitors=[monitor],run_time=2e-12)
    b=sim.grid.boundaries
    return geometry_steps((b.x,b.y,b.z),m)


def audit(archive, notebook):
    refs=json.loads((Path(__file__).resolve().parents[1]/"docs/verification/v012-grating-commercial-references.json").read_text())
    with ZipFile(archive) as z:
        fsp=z.read("grating_coupler_3D.fsp")
    digest=hashlib.sha256(fsp).hexdigest()
    book_sha=hashlib.sha256(notebook.read_bytes()).hexdigest()
    if digest!=refs["lumerical"]["project_sha256"] or book_sha!=refs["tidy3d"]["notebook_sha256"]:
        raise ValueError("Input differs from the pinned public example.")
    position=fsp.index(b"GUIboundary\0")
    axes=[array(fsp,a+"Grid",position).real*1e6 for a in "xyz"]
    return {"lumerical":{"source":refs["lumerical"]["source"],"project_sha256":digest,
                         "grid":"Stored FDTD simulation grid after GUIboundary",**geometry_steps(axes,settings("lumerical",12))},
            "tidy3d":{"source":refs["tidy3d"]["notebook"],"notebook_sha256":book_sha,
                      "grid":"Local reconstruction of the 3D example with symmetry (0,-1,0) and 15 minimum steps per wavelength; no cloud solve",
                      "sdk_version":importlib.metadata.version("tidy3d"), **tidy_grid(settings("tidy3d",12))}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("archive","notebook","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(audit(args.archive,args.notebook),indent=2,allow_nan=False)+"\n")
