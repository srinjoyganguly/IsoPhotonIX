"""Measure a planar offset slab's TM scattering at two physical reference planes."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

try:
    from .calibration_checkpoint import MeasurementCheckpoints
except ImportError:  # Direct execution of this file.
    from calibration_checkpoint import MeasurementCheckpoints

WIDTH = .375
CORE, CLAD = 3.48, 1.44
WAVELENGTHS = np.array([1.60, 1.55, 1.50])
FREQUENCIES = 1 / WAVELENGTHS
CHANNELS = ["left TM0", "left TM1", "right TM0", "right TM1"]
LUMERICAL = "https://optics.ansys.com/hc/en-us/articles/360042304134-Asymmetric-slab-dielectric-waveguide"
TIDY3D = "https://www.flexcompute.com/tidy3d/examples/notebooks/EMESolver/"


def slab_indices(frequency):
    """Even and odd TM roots of the symmetric three-layer dispersion relation."""
    from scipy.optimize import brentq

    k = 2 * np.pi * frequency
    v = WIDTH * k * np.sqrt(CORE**2 - CLAD**2) / 2
    roots = []
    for m in range(2):
        def equation(u):
            return u * np.tan(u - m * np.pi / 2) - (CORE / CLAD)**2 * np.sqrt(v*v - u*u)
        u = brentq(equation, m*np.pi/2 + 1e-10, min(v - 1e-10, (m+1)*np.pi/2 - 1e-10))
        roots.append(np.sqrt(CORE**2 - (2*u/(k*WIDTH))**2))
    return np.array(roots)


def interface_matrix(frequency, offset, distance, spacing, span):
    """Match Hz and (1/epsilon) dHz/dx at x=0, with an outgoing transverse PML."""
    from scipy.linalg import eig, solve, sqrtm

    count = round(span / spacing)
    y = (np.arange(count) + .5 - count/2) * spacing
    k0 = 2*np.pi*frequency
    def stretch(x):
        return 1 + 2j * np.maximum((np.abs(x) - (span/2 - 2))/2, 0)**3

    def fraction(x):
        lower = np.maximum(x - spacing/2, -offset/2 - WIDTH/2)
        upper = np.minimum(x + spacing/2, -offset/2 + WIDTH/2)
        return np.maximum(upper-lower, 0) / spacing

    # Hz is cell-centred. Use volume-averaged inverse epsilon for the x-flux
    # mass, and integrate epsilon across adjacent centres for the y face flux.
    inv_eps = fraction(y)/CORE**2 + (1-fraction(y))/CLAD**2
    def face_coefficient(x):
        eps = CLAD**2 + fraction(x) * (CORE**2 - CLAD**2)
        return 1 / (eps * stretch(x))

    plus, minus = face_coefficient(y+spacing/2), face_coefficient(y-spacing/2)
    cell = stretch(y)
    diagonal = (k0*k0 - (plus+minus)/(spacing**2*cell))/inv_eps
    up = plus[:-1]/(spacing**2*cell[:-1]*inv_eps[:-1])
    down = minus[1:]/(spacing**2*cell[1:]*inv_eps[1:])
    operator = np.diag(diagonal) + np.diag(up, 1) + np.diag(down, -1)
    eigenvalues, vectors = eig(operator, check_finite=False)
    beta = np.sqrt(eigenvalues)
    guided = np.flatnonzero((beta.real > k0*CLAD) & (np.abs(beta.imag) < 2e-5))
    if len(guided) != 2:
        raise ValueError(f"Expected two bound TM modes, measured {len(guided)}.")
    order = guided[np.argsort(-beta.real[guided])]
    beta = beta[order].real
    basis = vectors[:, order]
    root = np.exp(1j*np.pi/4)*sqrtm(-1j*operator)
    if np.linalg.norm(root@root-operator)/np.linalg.norm(operator) > 1e-8:
        raise ValueError("TM longitudinal operator failed its square-root residual.")
    admittance = inv_eps[:, None] * root
    admittances = [admittance, admittance[::-1, ::-1]]
    modes, left_vectors = [], []
    for side, center in enumerate((-offset/2, offset/2)):
        phi = basis.copy() if side == 0 else basis[::-1].copy()
        weight = inv_eps if side == 0 else inv_eps[::-1]
        for m in range(2):
            anchor = np.interp(center + WIDTH/4, y, phi[:, m])
            phi[:, m] *= np.conj(anchor)/abs(anchor)
            interior = abs(y) < span/2 - 2
            # The flux of an Hz mode is proportional to beta times this integral.
            phi[:, m] /= np.sqrt(beta[m] * np.sum(weight[interior]*abs(phi[interior, m])**2))
        dual = (phi * (cell*weight)[:, None]).T
        dual /= np.diag(dual@phi)[:, None]
        modes.append(phi)
        left_vectors.append(dual)
    right_hand_side = np.hstack([a@m for a, m in zip(admittances, modes)]) * 2
    interface = solve(admittances[0] + admittances[1], right_hand_side)
    matrix = np.vstack([v@interface for v in left_vectors]) - np.eye(4)
    # A positive Ey anchor changes the sign of Hz for waves travelling along -x.
    outgoing_sign = np.array([-1, -1, 1, 1])
    matrix *= np.outer(outgoing_sign, -outgoing_sign)
    phase = np.exp(1j*np.tile(beta, 2)*distance)
    return matrix*phase[:, None]*phase[None, :], beta/k0


def acquire_stage(mp, settings, side, mode, straight):
    mesh = settings["resolution"]
    centers = np.array([-.5, .5]) * settings["offset_um"]
    if straight:
        centers[:] = centers[side]
    signs = (-1, 1)
    cell = mp.Vector3(8, 10)
    direction = -signs[side]
    aperture = mp.Vector3(0, 6.)
    silicon = mp.Medium(index=CORE)
    layout = [mp.Block(center=mp.Vector3(sign*cell.x/4, cy),
                       size=mp.Vector3(cell.x/2, WIDTH, mp.inf), material=silicon)
              for sign, cy in zip(signs, centers)]
    source = mp.EigenModeSource(mp.GaussianSource(1/1.55, fwidth=.16, cutoff=5),
             center=mp.Vector3(signs[side]*2.5, centers[side]), size=aperture,
             direction=mp.X, eig_parity=mp.EVEN_Z, eig_band=mode+1,
             eig_kpoint=mp.Vector3(direction*slab_indices(1/1.55)[mode]/1.55),
             eig_match_freq=True, eig_resolution=mesh, eig_tolerance=1e-14)
    sim = mp.Simulation(cell_size=cell, dimensions=2, boundary_layers=[mp.PML(1.)],
            default_material=mp.Medium(index=CLAD), geometry=layout, resolution=mesh, sources=[source])
    try:
        mp.set_random_seed(0)
        sim.init_sim()
        planes, monitors = [], []
        for distance in settings["planes_um"]:
            for port in range(2):
                center = mp.Vector3(signs[port]*distance, centers[port])
                planes.append(mp.Volume(center=center, size=aperture))
                monitors.append(sim.add_mode_monitor(FREQUENCIES.tolist(), mp.ModeRegion(center=center,size=aperture)))
        sim.run(until_after_sources=settings["after_um_c"])
        coeff = np.empty((2, 2, 2, 2, 3), complex)
        neff = np.empty(coeff.shape)
        for index, (plane, monitor) in enumerate(zip(planes, monitors)):
            plane_idx, port = divmod(index, 2)
            for branch, sign in enumerate((1, -1)):
                for m in range(2):
                    mp.set_random_seed(0)
                    data = sim.get_eigenmode_coefficients(monitor,[m+1],eig_parity=mp.EVEN_Z,
                            eig_resolution=mesh,eig_tolerance=1e-14,direction=mp.X,
                            kpoint_func=lambda f,b:mp.Vector3(sign*f*slab_indices(f)[b-1]))
                    for i, (freq, k) in enumerate(zip(FREQUENCIES, data.kpoints)):
                        mp.set_random_seed(0)
                        eigenmode = sim.get_eigenmode(freq,mp.X,plane,m+1,k,parity=mp.EVEN_Z,
                                                      resolution=mesh,eigensolver_tol=1e-14)
                        anchor = eigenmode.amplitude(mp.Vector3(plane.center.x, centers[port]+WIDTH/4), mp.Ey)
                        mirror = eigenmode.amplitude(mp.Vector3(plane.center.x, centers[port]-WIDTH/4), mp.Ey)
                        correlation = ((-1)**m * (mirror*anchor.conjugate()).real)
                        if abs(anchor)<1e-8 or correlation < .9*abs(mirror*anchor):
                            raise ValueError("Projected Ey does not have the intended mode parity.")
                        if sign*data.vgrp[i] <= 0 or sign*k.x/freq <= CLAD:
                            raise ValueError("Projected TM mode has an invalid propagation direction or index.")
                        coeff[plane_idx,port,m,branch,i] = (
                            data.alpha[0,i,branch] * anchor/abs(anchor) * np.sign(data.vgrp[i]))
                        neff[plane_idx,port,m,branch,i] = abs(k.x)/freq
        x, y, _, _ = sim.get_array_metadata()
        arrays = {"coefficients":coeff,"neff":neff,"final_Hz":sim.get_array(component=mp.Hz),
                  "x_um":x,"y_um":y}
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Nonfinite modal amplitude or field sample.")
        return {"launch":CHANNELS[side*2+mode],"straight_reference":straight}, arrays
    finally:
        sim.reset_meep()


def measured_columns(mp, stage, settings, dest):
    matrices = np.empty((2, 3, 4, 4), complex)
    saved = {}
    all_indices = []
    for side in range(2):
        for mode in range(2):
            column = side*2+mode
            raw = []
            for reference in (True, False):
                kind = "reference" if reference else "device"
                label = f"{column}-{kind}"
                print(f"{CHANNELS[column]} {kind}", flush=True)
                _, data = stage.acquire(label, settings, lambda: acquire_stage(mp, settings, side, mode, reference))
                raw.append(data["coefficients"])
                saved[label+"_coefficients"] = data["coefficients"]
                saved[label+"_neff"] = data["neff"]
                saved[label+"_final_Hz"] = data["final_Hz"]
                saved["x_um"], saved["y_um"] = data["x_um"], data["y_um"]
                all_indices.append(data["neff"])
            ref, device = raw
            incident = ref[:,side,mode,side,:]
            if np.min(abs(incident)) < 1e-6*np.max(abs(incident)) or np.max(abs(incident)) == 0:
                raise ValueError("Incident fundamental or higher-order mode is too weak.")
            matrices[:,:,side*2:side*2+2,column] = (
                    (device[:,side,:,1-side]-ref[:,side,:,1-side])/incident[:,None,:]).transpose(0,2,1)
            other = 1-side
            matrices[:,:,other*2:other*2+2,column] = (
                    device[:,other,:,side]/incident[:,None,:]).transpose(0,2,1)
    saved.update({"s":matrices,"wavelength_um":WAVELENGTHS,"planes_um":settings["planes_um"]})
    np.savez_compressed(dest/"arrays.npz", **saved)
    return matrices, np.stack(all_indices)


def run(args, dest, checkpoint):
    import meep as mp
    mp.verbosity(0)
    other_distance = .75 if args.port_distance > .875 else 1.25
    distances = [args.port_distance, other_distance]
    settings = {"resolution":args.resolution,"offset_um":args.offset,"planes_um":distances,
                "after_um_c":args.after,"aperture_um":6.0,"cell_um":[8.,10.],"pml_um":1.}
    store = MeasurementCheckpoints(checkpoint, __file__)
    matrices, all_indices = measured_columns(mp, store, settings, dest)
    matrix, control = matrices
    refs, coarse, wider, analytic_indices = [], [], [], []
    for f in FREQUENCIES:
        spectrum, _ = interface_matrix(f,args.offset,args.port_distance,1/96,8.)
        comparison, _ = interface_matrix(f,args.offset,args.port_distance,1/64,8.)
        outer, _ = interface_matrix(f,args.offset,args.port_distance,1/64,10.)
        refs.append(spectrum)
        coarse.append(comparison)
        wider.append(outer)
        analytic_indices.append(slab_indices(f))
    refs, coarse, wider = map(np.stack, (refs, coarse, wider))
    indices = np.stack(analytic_indices)
    phase = np.exp(2j*np.pi * FREQUENCIES[:,None] * np.tile(indices, (1,2)) * (distances[0]-distances[1]))
    predicted = control * phase[:,:,None] * phase[:,None,:]
    np.savez_compressed(dest/"reference.npz",s=refs,coarse=coarse,wider=wider,analytic_neff=indices,
                        plane_shift_prediction=predicted)
    power = abs(matrix)**2
    checks = {"independent_complex_error_max":float(abs(matrix-refs).max()),
              "independent_power_error_max":float(abs(power-abs(refs)**2).max()),
              "reference_mesh_complex_change_max":float(abs(refs-coarse).max()),
              "reference_span_complex_change_max":float(abs(wider-coarse).max()),
              "reference_mesh_power_change_max":float(abs(abs(refs)**2-abs(coarse)**2).max()),
              "plane_translation_complex_error_max":float(abs(matrix-predicted).max()),
              "plane_translation_power_change_max":float(abs(power-abs(control)**2).max()),
              "reciprocity_power_error_max":float(abs(power-power.transpose(0,2,1)).max()),
              "max_collected_power":float(power.sum(axis=1).max()),
              "port_neff_abs_error_max":float(abs(all_indices-indices.T[:,None,:]).max())}
    limits = {"independent_complex_error_max":.15,"independent_power_error_max":.04,
              "reference_mesh_complex_change_max":.06,"reference_span_complex_change_max":.04,
              "reference_mesh_power_change_max":.01,"plane_translation_complex_error_max":.06,
              "plane_translation_power_change_max":.02,"reciprocity_power_error_max":.025,
              "max_collected_power":1.03,"port_neff_abs_error_max":.04}
    passed = all(checks[k] < threshold for k, threshold in limits.items())
    report = {"schema_version":1,"case":"multimode-tm","solver":"Meep","version":mp.__version__,
              "actual_device":"CPU","passed":passed,"parameters":settings,"channels":CHANNELS,
              "wavelength_um":WAVELENGTHS.tolist(),"s_real":matrix.real.tolist(),"s_imag":matrix.imag.tolist(),
              "power":power.tolist(),"independent_power":(abs(refs)**2).tolist(),"analytic_port_neff":indices.tolist(),
              "checks":checks,"limits":limits,"vendor_sources":{"lumerical":LUMERICAL,"tidy3d":TIDY3D},
              "scope":"Hz slab modes with positive Ey gauge at centre + one-quarter width, both propagation directions. Two physical plane sets are monitored in the same native reference and device runs. Guided-power sums exclude radiation."}
    (dest/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution",type=int,choices=(48,64,80),default=64)
    parser.add_argument("--offset",type=float,default=.125)
    parser.add_argument("--port-distance",type=float,default=1.25)
    parser.add_argument("--after",type=float,default=100.)
    parser.add_argument("--checkpoint-dir",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if not 0 <= args.offset <= .25 or not .75 <= args.port_distance <= 1.75 or not 60 <= args.after <= 150:
        parser.error("Offset must be 0–0.25 µm, port distance 0.75–1.75 µm, and post-source time 60–150 µm/c.")
    if abs(args.port_distance*args.resolution-round(args.port_distance*args.resolution)) > 1e-9:
        parser.error("Place the physical port on a mesh plane (distance times resolution must be integral).")
    if args.output.exists():
        parser.error("Choose a fresh output directory.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        dest = Path(tmp)/"result"
        dest.mkdir()
        passed = run(args,dest,args.checkpoint_dir or Path(tmp)/"stages")
        dest.rename(args.output)
    if not passed:
        raise SystemExit("Numerical qualification limits exceeded. Inspect report.json.")


if __name__ == "__main__":
    main()
