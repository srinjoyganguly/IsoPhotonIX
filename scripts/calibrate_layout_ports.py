"""Measure the complex fundamental Ez matrix of bounded polygon layouts with X/Y ports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import planar_port_layout as layout
from scripts.calibration_checkpoint import MeasurementCheckpoints

WAVELENGTHS = np.array([1.60,1.55,1.50])
FREQUENCIES = 1/WAVELENGTHS
CORE, CLAD = 3.48, 1.44


def block(mp, bounds, material):
    x0,x1,y0,y1 = bounds
    return mp.Block(center=mp.Vector3((x0+x1)/2,(y0+y1)/2),
                    size=mp.Vector3(x1-x0,y1-y0,mp.inf),material=material)


def geometry(mp, design, launch, reference):
    material = mp.Medium(index=CORE)
    if reference:
        port = design["ports"][launch]
        axis, _ = layout.OUTWARD[port["outward"]]
        transverse = port["center_um"][1-axis]
        half_width = port["width_um"]/2
        transverse_bounds = [transverse-half_width,transverse+half_width]
        longitudinal = [-layout.HALF_CELL,layout.HALF_CELL]
        return [block(mp,longitudinal+transverse_bounds if axis==0 else transverse_bounds+longitudinal,material)]
    polygons = [mp.Prism(vertices=[mp.Vector3(*point) for point in outline],height=mp.inf,material=material)
                for outline in design["polygons"]]
    return polygons+[block(mp,layout.lead_bounds(port),material) for port in design["ports"]]


def plane(mp, port, shift):
    axis, sign = layout.OUTWARD[port["outward"]]
    center = list(port["center_um"])
    center[axis] += sign*shift
    size = [0.,0.]
    size[1-axis] = layout.APERTURE
    return mp.Volume(center=mp.Vector3(*center),size=mp.Vector3(*size))


def coefficients(mp, sim, geometry_plane, monitor, axis, mesh):
    normal = (mp.X,mp.Y)[axis]
    result = np.empty((2,len(FREQUENCIES)),complex)
    indices = np.empty(result.shape)
    for branch, sign in enumerate((1,-1)):
        def k_guess(f, band):
            coordinate=[0.,0.]
            coordinate[axis]=sign*f*2.8
            return mp.Vector3(*coordinate)
        mp.set_random_seed(0)
        projection = sim.get_eigenmode_coefficients(monitor,[1],eig_parity=mp.ODD_Z,
            eig_resolution=mesh,eig_tolerance=1e-14,direction=normal,kpoint_func=k_guess)
        for i,(freq,k) in enumerate(zip(FREQUENCIES,projection.kpoints)):
            mp.set_random_seed(0)
            mode = sim.get_eigenmode(freq,normal,geometry_plane,1,k,parity=mp.ODD_Z,
                                     resolution=mesh,eigensolver_tol=1e-14)
            anchor = mode.amplitude(geometry_plane.center,mp.Ez)
            neff = sign*(k.x,k.y)[axis]/freq
            if neff<=CLAD or sign*projection.vgrp[i]<=0 or abs(anchor)<1e-8:
                raise ValueError("Fundamental Ez direction, guidance or electric phase anchor failed.")
            result[branch,i] = projection.alpha[0,i,branch]*np.sign(projection.vgrp[i])*anchor/abs(anchor)
            indices[branch,i] = neff
    return result,indices


def simulate(mp, design, launch, reference, resolution, after):
    port = design["ports"][launch]
    axis, sign = layout.OUTWARD[port["outward"]]
    source_plane = plane(mp,port,.5)
    launch_k = [0.,0.]
    launch_k[axis] = -sign*2.
    source = mp.EigenModeSource(mp.GaussianSource(1/1.55,fwidth=.16,cutoff=5),
        center=source_plane.center,size=source_plane.size,direction=(mp.X,mp.Y)[axis],
        eig_band=1,eig_kpoint=mp.Vector3(*launch_k),eig_parity=mp.ODD_Z,eig_match_freq=True,
        eig_resolution=resolution,eig_tolerance=1e-14)
    sim = mp.Simulation(cell_size=mp.Vector3(2*layout.HALF_CELL,2*layout.HALF_CELL),
        dimensions=2,boundary_layers=[mp.PML(layout.PML)],resolution=resolution,
        default_material=mp.Medium(index=CLAD),geometry=geometry(mp,design,launch,reference),sources=[source])
    try:
        mp.set_random_seed(0)
        sim.init_sim()
        observed = [launch] if reference else range(len(design["ports"]))
        monitors = []
        for p in observed:
            observation = design["ports"][p]
            port_axis,_ = layout.OUTWARD[observation["outward"]]
            volume = plane(mp,observation,0.)
            monitor = sim.add_mode_monitor(FREQUENCIES.tolist(),mp.ModeRegion(center=volume.center,size=volume.size))
            monitors.append((p,volume,monitor,port_axis))
        sim.run(until_after_sources=after*.75)
        def measure():
            samples = np.zeros((len(design["ports"]),2,len(FREQUENCIES)),complex)
            indices = np.zeros(samples.shape)
            for p,volume,monitor,port_axis in monitors:
                samples[p],indices[p]=coefficients(mp,sim,volume,monitor,port_axis,resolution)
            return samples,indices
        early,_ = measure()
        sim.run(until=after*.25)
        final,indices = measure()
        x,y,_,_ = sim.get_array_metadata()
        arrays = {"coefficients":final,"early_coefficients":early,"neff":indices,
                  "final_Ez":sim.get_array(component=mp.Ez),"x_um":x,"y_um":y}
        if not reference:
            arrays["epsilon"] = sim.get_array(component=mp.Dielectric)
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Nonfinite native field or projection.")
        return {"observed_port_indices":list(observed)},arrays
    finally:
        sim.reset_meep()


def build_matrix(design, data, key):
    ports = design["ports"]
    matrix = np.empty((len(FREQUENCIES),len(ports),len(ports)),complex)
    for launch, (ref,device) in enumerate(data):
        _, sign = layout.OUTWARD[ports[launch]["outward"]]
        incident = ref[key][launch,int(sign>0)]
        if np.max(abs(incident)) == 0 or np.min(abs(incident))<1e-6*np.max(abs(incident)):
            raise ValueError("Incident mode is insufficiently excited.")
        for output, port in enumerate(ports):
            _, outgoing = layout.OUTWARD[port["outward"]]
            branch = int(outgoing<0)
            amplitude = device[key][output,branch].copy()
            if output == launch:
                amplitude -= ref[key][output,branch]
            matrix[:,output,launch] = amplitude/incident
    return matrix


def run(design, mesh, after, dest, checkpoint):
    import meep as mp
    mp.verbosity(0)
    for port in design["ports"]:
        axis,_ = layout.OUTWARD[port["outward"]]
        native = port["center_um"][axis]*mesh
        if abs(native-round(native))>1e-9:
            raise ValueError("A port's normal coordinate times mesh resolution must be an integer.")
    stage = MeasurementCheckpoints(checkpoint,__file__)
    settings = {"design":design,"resolution":mesh,"post_source_um_c":after,
        "geometry_helper_sha256":hashlib.sha256(Path(layout.__file__).read_bytes()).hexdigest()}
    raw, pairs = {}, []
    for launch, port in enumerate(design["ports"]):
        stages = []
        for reference in (True,False):
            category = "reference" if reference else "device"
            label = f"port{launch}-{category}"
            print(f"{port['name']} {category}",flush=True)
            _, arrays = stage.acquire(label,settings,lambda:simulate(mp,design,launch,reference,mesh,after))
            stages.append(arrays)
            for key,value in arrays.items():
                raw[label+"_"+key] = value
        pairs.append(stages)
    matrix = build_matrix(design,pairs,"coefficients")
    early = build_matrix(design,pairs,"early_coefficients")
    power = abs(matrix)**2
    checks = {"max_collected_power":float(power.sum(axis=1).max()),
              "power_reciprocity_residual_max":float(abs(power-power.transpose(0,2,1)).max()),
              "complex_reciprocity_residual_max":float(abs(matrix-matrix.transpose(0,2,1)).max()),
              "early_late_complex_change_max":float(abs(early-matrix).max())}
    limits = {"max_collected_power":1.03,"power_reciprocity_residual_max":.04,
              "complex_reciprocity_residual_max":.10,"early_late_complex_change_max":.025}
    passed = all(checks[k]<v for k,v in limits.items())
    np.savez_compressed(dest/"arrays.npz",s=matrix,s_early=early,wavelength_um=WAVELENGTHS,**raw)
    report = {"schema_version":1,"case":"layout-ports-2d","solver":"Meep","actual_device":"CPU","version":mp.__version__,
              "passed":passed,"parameters":settings,"channels":[p["name"] for p in design["ports"]],
              "wavelength_um":WAVELENGTHS.tolist(),"s_real":matrix.real.tolist(),"s_imag":matrix.imag.tolist(),
              "power":power.tolist(),"checks":checks,"limits":limits,"core_index":CORE,"cladding_index":CLAD,
              "port_aperture_um":layout.APERTURE,"scope":"Finite monitor-aperture, fundamental Ez modes. Positive-real Ez gauge at each physical plane. Each input has an independent native straight reference. Uncollected power includes radiation, unmeasured modes and finite-domain error."}
    (dest/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--resolution",type=int,choices=(24,32,40),default=32)
    parser.add_argument("--after",type=float,default=100.)
    parser.add_argument("--checkpoint-dir",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not 60<=args.after<=180 or args.output.exists():
        parser.error("Choose a fresh output and a post-source time between 60 and 180 µm/c.")
    design=layout.validate_planar_port_layout(json.loads(args.design.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        dest=Path(tmp)/"result"
        dest.mkdir()
        passed=run(design,args.resolution,args.after,dest,args.checkpoint_dir or Path(tmp)/"stages")
        dest.rename(args.output)
    if not passed:
        raise SystemExit("The numerical qualification limits were exceeded. Inspect report.json.")


if __name__ == "__main__":
    main()
