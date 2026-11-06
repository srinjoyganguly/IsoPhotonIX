"""Compare measured FDTDX grating illumination to Tidy3D's analytic vector Gaussian beam."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from scripts.compare_grating_domain import read


def inspect(receipt):
    import tidy3d as td

    row, path=read(receipt)
    m=row["settings"]
    if m["geometry"]!="tidy3d":
        raise ValueError("This diagnostic applies to the air-side Tidy3D Gaussian grating.")
    dx=1/m["resolution"]
    with np.load(path,allow_pickle=False) as data:
        wave=data["ref_wavelength_um"]
        # FDTDX exact-interpolation phase planes are co-located on Ez (i,j,k+½).
        xx=data["ref_upper_x_um"]-dx/2-m["beam_x"]
        yy=data["ref_upper_y_um"]-dx/2
        z=row["stages"]["reference"]["sample_plane_z_um"]
        frequencies=td.C_0/wave
        beam=td.GaussianBeamProfile(center=(m["beam_x"],0,m["source_z"]),size=(4*m["waist"],4*m["waist"],0),
                                    freqs=frequencies,angle_theta=np.deg2rad(m["angle_deg"]),pol_angle=np.pi/2,
                                    direction="-",waist_radius=m["waist"])
        def evaluate(axis_x, axis_y, field):
            values=beam.analytic_beam(x=axis_x,y=axis_y,z=np.array([z-m["source_z"]]),
                                      background_n=np.ones(len(wave),dtype=complex),field=field)
            return values[:,:,:,0,:].transpose(3,0,1,2)
        def incident(e,h,spacing):
            return .5*spacing**2*np.real(e[:,1]*h[:,0].conj()-e[:,0]*h[:,1].conj()).sum(axis=(1,2))
        analytic_fields={field:evaluate(xx,yy,field) for field in ("E","H")}
        footprint=4*m["waist"]
        n=int(np.ceil(footprint/dx))
        broad_axis=-footprint/2+(np.arange(n)+.5)*footprint/n
        nominal=incident(evaluate(broad_axis,broad_axis,"E"),evaluate(broad_axis,broad_axis,"H"),footprint/n)
        finite=incident(analytic_fields["E"],analytic_fields["H"],dx)
        overlaps={}
        for field, index in (("E",0),("H",3)):
            analytic=analytic_fields[field]
            measured=data["ref_upper"][:,index:index+3]
            axes=(1,2,3)
            product=np.sum(np.conj(analytic)*measured,axis=axes,dtype=np.complex128)
            norm=np.sum(abs(analytic)**2,axis=axes,dtype=np.float64)*np.sum(abs(measured)**2,axis=axes,dtype=np.float64)
            overlaps[field]=(abs(product)**2/norm).tolist()
    return {"mesh":m["resolution"],"domain":m.get("grating_domain","full"),"arrays_sha256":row["arrays_sha256"],
            "wavelength_um":wave.tolist(),"sample_z_um":z,"vector_shape_overlap":overlaps,
            "analytic_native_aperture_to_nominal_square_power":(finite/nominal).tolist(),
            "nominal_square_width_um":footprint}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipts",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    record={"sdk_version":importlib.metadata.version("tidy3d"),
            "scope":"Finite-aperture squared normalized vector overlaps with the SDK analytic Gaussian on the actual recorded air plane. Independent global amplitude and phase cancel. This isolates incident profile shape. The full-device measurements supply the coupled power.",
            "definition":"abs(sum(conj(analytic)*measured))²/(sum(abs(analytic)²)*sum(abs(measured)²)), separately for E and H.",
            "source":"https://docs.flexcompute.com/projects/tidy3d/en/v2.12.0/api/analytic_beams.html",
            "examples":[inspect(row) for row in args.receipts]}
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
