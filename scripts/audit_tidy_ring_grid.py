"""Reconstruct the public add/drop ring's automatic grid locally, without submitting a job."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from scripts.compare_loaded_ring3d import TIDY_SHA


def occupied(boundaries, start, stop):
    values=np.asarray(boundaries,dtype=float)
    widths=np.diff(values)
    overlaps=np.minimum(stop,values[1:])-np.maximum(start,values[:-1])
    hit=overlaps>1e-12
    complete=hit & (overlaps >= widths-1e-12)
    if not hit.any() or not np.isclose(overlaps[hit].sum(),stop-start,rtol=1e-9):
        raise ValueError('The reconstructed grid does not cover the feature.')
    return dict(interval_um=[start,stop],intersecting_cells=int(hit.sum()),fully_inside_cells=int(complete.sum()),
                equivalent_cells=float((overlaps[hit]/widths[hit]).sum()),widths_nm=(1000*widths[hit]).tolist())


def audit(path):
    import tidy3d as td
    if hashlib.sha256(path.read_bytes()).hexdigest()!=TIDY_SHA:
        raise ValueError('Unexpected public ring notebook revision.')
    wavelength=1.55
    r,w,h,gap=5.,.43,.22,.10
    center=r+w+gap
    si=td.material_library['cSi']['Palik_Lossless']
    oxide=td.material_library['SiO2']['Palik_Lossless']
    buses=[td.Structure(geometry=td.Box(center=(0,y,0),size=(td.inf,w,h)),medium=si) for y in (center,-center)]
    outer,inner=[td.Cylinder(center=(0,0,0),axis=2,radius=rad,length=h) for rad in (r+w/2,r-w/2)]
    ring=td.Structure(geometry=outer-inner,medium=si)
    freq0=td.C_0/wavelength
    freqs=td.C_0/np.linspace(1.5,1.6,1001)
    mode=td.ModeSpec(num_modes=1,target_neff=3.47)
    aperture=(0,4*w,6*h)
    source=td.ModeSource(center=(-r,center,0),size=aperture,source_time=td.GaussianPulse(freq0=freq0,
        fwidth=.5*(freqs.max()-freqs.min())),direction='+',mode_spec=mode,mode_index=0,num_freqs=7)
    monitors=[td.FieldMonitor(center=(0,0,0),size=(td.inf,td.inf,0),freqs=[freq0],name='field')]
    for name,position in [('through',(r,center,0)),('drop',(-r,-center,0))]:
        monitors.append(td.ModeMonitor(center=position,size=aperture,freqs=freqs,mode_spec=mode,name=name))
    buffer=.7*wavelength
    lx=2*r+w+2*buffer
    sim=td.Simulation(center=(0,0,0),size=(lx,lx+2*gap+2*w,h+2*buffer),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=15,wavelength=wavelength),structures=[*buses,ring],
        sources=[source],monitors=monitors,run_time=3e-11,symmetry=(0,0,1),medium=oxide,
        boundary_spec=td.BoundarySpec(x=td.Boundary(plus=td.Absorber(num_layers=60),minus=td.Absorber(num_layers=60)),
                                      y=td.Boundary.absorber(),z=td.Boundary.absorber()))
    bounds=sim.grid.boundaries
    return dict(scope='Offline grid and central material readout using the installed Tidy3D version and the inputs from the pinned notebook. This reconstruction is not a measured copy of the historical cloud solver grid or complex fields.',
                notebook_sha256=TIDY_SHA,tidy3d=version('tidy3d'),solver_jobs_submitted=0,
                shape=list(sim.grid.num_cells),symmetry=sim.symmetry,source_aperture_yz_um=aperture[1:],
                nominal_input_through_drop_plane_x_um=[-r,r,-r],central_wavelength_um=wavelength,
                silicon_n=float(np.sqrt(si.eps_model(freq0)).real),oxide_n=float(np.sqrt(oxide.eps_model(freq0)).real),
                gap_at_closest_approach=occupied(bounds.y,r+w/2,center-w/2),
                core_height=occupied(bounds.z,-h/2,h/2),straight_width=occupied(bounds.y,center-w/2,center+w/2),
                source='https://github.com/flexcompute/tidy3d-notebooks/blob/18fd47f8bd87ad89582b507c20bf4997f79d55c0/RingResonator.ipynb')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.notebook)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
