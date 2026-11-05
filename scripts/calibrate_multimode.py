"""Measure all four guided channels of a transversely offset planar slab joint."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

CORE, CLADDING = 3.48, 1.44
WIDTH, OFFSET = .375, .125
PORT_X, SOURCE_X = 1.25, 2.3
FREQUENCIES = 1 / np.array([1.60, 1.55, 1.50])
PHASE_VERSION = 2
CHANNELS = ['left TE0', 'left TE1', 'right TE0', 'right TE1']
TIDY = 'https://www.flexcompute.com/tidy3d/examples/notebooks/EMESolver/'
ANSYS = 'https://optics.ansys.com/hc/en-us/articles/360034396614-MODE-EigenMode-Expansion-EME-solver-introduction'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interface_reference(frequency, offset, spacing, span):
    """Solve the scalar Helmholtz interface problem in a complete transverse basis.

    Complex coordinate stretching absorbs outgoing radiation at the transverse
    edges. Guiding-region fields use the ordinary power integral. Both electric
    field and its x derivative are continuous across the abrupt interface.
    """
    from scipy.linalg import eig, solve, sqrtm

    count = round(span / spacing)
    y = (np.arange(count) + .5 - count/2) * spacing
    k0 = 2*np.pi*frequency

    def stretch(position):
        return 1 + 2j*np.maximum((abs(position) - (span/2-2.0))/2.0, 0.)**3

    # Cell-centred, flux-conservative transverse second derivative in a PML.
    cell = stretch(y)
    plus, minus = 1/stretch(y+spacing/2), 1/stretch(y-spacing/2)
    length = np.maximum(0, np.minimum(y + spacing/2, -offset/2 + WIDTH/2)
                        - np.maximum(y - spacing/2, -offset/2 - WIDTH/2))
    eps = CLADDING**2 + (CORE**2 - CLADDING**2)*length/spacing
    op = np.diag(k0*k0*eps - (plus+minus)/(spacing**2*cell))
    op += np.diag(plus[:-1]/(spacing**2*cell[:-1]), 1)
    op += np.diag(minus[1:]/(spacing**2*cell[1:]), -1)
    vals, basis = eig(op, check_finite=False)
    beta = np.sqrt(vals)
    # Forward branches decay towards +x. Guided modes may have roundoff-level
    # imaginary parts, so retain their positive propagation direction.
    beta = np.where(beta.imag < -1e-8, -beta, beta)
    guided = np.flatnonzero((beta.real > k0*CLADDING) & (abs(beta.imag)<1e-5))
    require(len(guided)==2, 'Expected precisely two bound TE modes.')
    order = guided[np.argsort(-beta.real[guided])]
    basis, beta = basis[:,order], beta[order]
    # The Schur square root is well-conditioned here even when individual
    # high-order PML eigenvectors nearly coincide. Rotate the branch cut off
    # the negative real axis so evanescent modes decay away from the joint.
    dtn = np.exp(1j*np.pi/4) * sqrtm(-1j*op)
    residual = np.linalg.norm(dtn@dtn-op) / np.linalg.norm(op)
    require(residual < 1e-8, 'Open-boundary derivative operator failed its residual check.')
    operators = [dtn,dtn[::-1,::-1]]
    bases, projections = [], []
    for center, modes in zip((-offset/2,offset/2),(basis.copy(),basis[::-1].copy())):
        for m in range(2):
            anchor = np.interp(center+WIDTH/4, y, modes[:,m])
            modes[:,m] *= np.conj(anchor)/abs(anchor)
            modes[:,m] /= np.linalg.norm(modes[abs(y) < span/2-2.,m])
        # The stretch-weighted operator is symmetric under transpose. Its
        # bilinear left modes project guided amplitudes without inverting a
        # possibly ill-conditioned entire radiation eigenbasis.
        dual = (modes*cell[:,None]).T
        dual /= np.diag(dual@modes)[:,None]
        bases.append(modes)
        projections.append(dual)
    face = solve(operators[0]+operators[1], np.hstack([2*modes*beta for modes in bases]))
    s = np.vstack([dual@face for dual in projections]) - np.eye(4)
    ks = np.tile(beta.real,2)
    phase = np.exp(1j*ks*PORT_X)
    s *= np.sqrt(ks[:,None]/ks[None,:]) * phase[:,None]*phase[None,:]
    return s, ks/k0


def measure_stage(mp, resolution, side, band, offset, reference, aperture, pml, after, finite=False):
    width = .875 if finite else WIDTH
    height = .25 if finite else mp.inf
    parity = mp.EVEN_Z if finite else mp.ODD_Z
    component = mp.Ey if finite else mp.Ez
    outward = (-1, 1)
    centers = [-offset/2, offset/2]
    source_center = centers[side]
    if reference:
        centers = [source_center, source_center]
    cell = mp.Vector3(6 + 2*pml, (3.5 if finite else 5.5) + 2*pml, 2 + 2*pml if finite else 0)
    medium = mp.Medium(index=CORE)
    geometry = [mp.Block(center=mp.Vector3(sign*cell.x/4, centers[i]),
                         size=mp.Vector3(cell.x/2, width, height), material=medium)
                for i, sign in enumerate(outward)]
    direction = -outward[side]
    size = mp.Vector3(0, aperture, 2 if finite else 0)
    src = mp.EigenModeSource(mp.GaussianSource(1/1.55, fwidth=.16, cutoff=5),
                             center=mp.Vector3(outward[side]*SOURCE_X, source_center),
                             size=size, direction=mp.X, eig_band=band,
                             eig_parity=parity, eig_kpoint=mp.Vector3(direction * 1.6),
                             eig_match_freq=True, eig_resolution=resolution, eig_tolerance=1e-14)
    sim = mp.Simulation(cell_size=cell, dimensions=3 if finite else 2, symmetries=[mp.Mirror(mp.Z)] if finite else [], boundary_layers=[mp.PML(pml)],
                        default_material=mp.Medium(index=CLADDING), geometry=geometry,
                        resolution=resolution, sources=[src])
    try:
        mp.set_random_seed(0)
        sim.init_sim()
        monitors = []
        for i, sign in enumerate(outward):
            point = mp.Vector3(sign*PORT_X, centers[i])
            monitors.append(sim.add_mode_monitor(FREQUENCIES.tolist(), mp.ModeRegion(center=point,size=size)))
        sim.run(until_after_sources=after)
        coeff = np.empty((2, 2, 2, len(FREQUENCIES)),complex)
        neff = np.empty(coeff.shape)
        for i, monitor in enumerate(monitors):
            plane = mp.Volume(center=mp.Vector3(outward[i]*PORT_X, centers[i]), size=size)
            for branch, sign in enumerate((1,-1)):
                for mode_idx in range(2):
                    band_num = mode_idx+1
                    mp.set_random_seed(0)
                    measures = sim.get_eigenmode_coefficients(monitor, [band_num], eig_parity=parity,
                        eig_resolution=resolution, eig_tolerance=1e-14, direction=mp.X,
                        kpoint_func=lambda f,b:mp.Vector3(sign*f*(2.8 if b==1 else 1.9)))
                    for f_idx, (freq,k) in enumerate(zip(FREQUENCIES,measures.kpoints)):
                        mp.set_random_seed(0)
                        mode = sim.get_eigenmode(freq,mp.X,plane,band_num,k,parity=parity,
                                                 resolution=resolution,eigensolver_tol=1e-14)
                        anchor = mode.amplitude(mp.Vector3(plane.center.x,centers[i]+width/4),component)
                        mirror = mode.amplitude(mp.Vector3(plane.center.x,centers[i]-width/4),component)
                        parity_sign = 1 if mode_idx==0 else -1
                        require(abs(mirror)>1e-7 and (mirror*anchor.conjugate()).real*parity_sign >
                                .9*abs(mirror*anchor), 'Mode order does not match the expected even/odd field.')
                        require(abs(anchor)>1e-7 and sign*k.x/freq>CLADDING, 'Port mode failed its gauge or guidance check.')
                        velocity = measures.vgrp[f_idx]
                        require(sign * velocity > 0, 'Port mode propagates in the wrong direction.')
                        # Meep's signed flux overlap carries the group-velocity sign.
                        coeff[i,mode_idx,branch,f_idx] = (measures.alpha[0,f_idx,branch]
                                                          * anchor/abs(anchor) * np.sign(velocity))
                        neff[i,mode_idx,branch,f_idx] = abs(k.x)/freq
        coordinates = sim.get_array_metadata()
        field = sim.get_array(component=component)
        require(np.isfinite(coeff).all() and np.isfinite(field).all(), 'Nonfinite Meep fields or amplitudes.')
        return coeff, neff, field, coordinates[:3]
    finally:
        sim.reset_meep()


def measure(resolution, offset, aperture, pml, after, dest, finite=False):
    import meep as mp
    mp.verbosity(0)
    output = {}
    matrix = np.empty((len(FREQUENCIES),4,4), complex)
    for side in range(2):
        incoming, reflected = (0,1) if side==0 else (1,0)
        for mode in range(2):
            col=side*2+mode
            print(f'Mesh {resolution}: {CHANNELS[col]} reference and device',flush=True)
            ref, ref_index, _, _ = measure_stage(mp,resolution,side,mode+1,offset,True,aperture,pml,after,finite)
            dev, dev_index, field, axes = measure_stage(mp,resolution,side,mode+1,offset,False,aperture,pml,after,finite)
            incident=ref[side,mode,incoming]
            require(np.min(abs(incident)) > 1e-6*np.max(abs(incident)) > 0, 'Weak incident reference.')
            matrix[:,side*2:side*2+2,col] = ((dev[side,:,reflected]-ref[side,:,reflected])/incident).T
            matrix[:,(1-side)*2:(1-side)*2+2,col] = (dev[1-side,:,incoming]/incident).T
            output[f'incident_{col}']=incident
            output[f'reference_{col}']=ref
            output[f'device_{col}']=dev
            output[f'reference_neff_{col}']=ref_index
            output[f'device_neff_{col}']=dev_index
            field_name = 'Ey' if finite else 'Ez'
            output[f'final_{field_name}_{col}']=field
            output['x_um'],output['y_um'],output['z_um']=axes
    np.savez_compressed(dest/'arrays.npz',s=matrix, phase_convention_version=PHASE_VERSION, frequency_per_um=FREQUENCIES, **output)
    return matrix


def run(args,dest):
    import meep as mp
    if args.finite:
        return finite_height(args,dest)
    reference, indices = zip(*(interface_reference(f,args.offset,1/64,8) for f in FREQUENCIES))
    refined = np.stack([interface_reference(f,args.offset,1/96,8)[0] for f in FREQUENCIES])
    wider = np.stack([interface_reference(f,args.offset,1/64,10)[0] for f in FREQUENCIES])
    reference = np.stack(reference)
    matrix = measure(args.resolution,args.offset,args.aperture,args.pml,args.after,dest)
    ref_power, power = abs(refined)**2, abs(matrix)**2
    differences = {'power_vs_independent_reference_max':float(np.max(abs(power-ref_power))),
                   'complex_vs_independent_reference_max':float(np.max(abs(matrix-refined))),
                   'reference_mesh_power_change_max':float(np.max(abs(ref_power-abs(reference)**2))),
                   'reference_box_power_change_max':float(np.max(abs(abs(reference)**2-abs(wider)**2))),
                   'power_reciprocity_max':float(np.max(abs(power-power.transpose(0,2,1)))),
                   'largest_guided_output_sum':float(np.max(power.sum(axis=1)))}
    np.savez_compressed(dest/'reference.npz',reference=refined,coarse_reference=reference,
                        wide_reference=wider,neff=indices)
    passed = (differences['power_vs_independent_reference_max'] < .04 and
              differences['complex_vs_independent_reference_max'] < .15 and
              differences['reference_mesh_power_change_max'] < .01 and
              differences['reference_box_power_change_max'] < .01 and
              differences['power_reciprocity_max'] < .03 and differences['largest_guided_output_sum'] < 1.03)
    report={'schema_version':1,'phase_convention_version':PHASE_VERSION,'case':'multimode','solver':'Meep','version':mp.__version__,
            'actual_device':'CPU','dimension':'2D planar, invariant in z. Ez, two guided TE modes per side.',
            'passed':bool(passed),'parameters':{'resolution':args.resolution,'offset_um':args.offset,
              'width_um':WIDTH,'core_index':CORE,'cladding_index':CLADDING,'port_x_um':[-PORT_X,PORT_X],
              'aperture_um':args.aperture,'pml_um':args.pml,'post_source_time_um_over_c':args.after},
            'channels':CHANNELS,'wavelength_um':(1/FREQUENCIES).tolist(),'power':power.tolist(),
            's_real':matrix.real.tolist(),'s_imag':matrix.imag.tolist(),
            'independent_power':ref_power.tolist(),'independent_port_neff':np.asarray(indices).tolist(),
            'checks':differences,'vendor_sources':{'lumerical':ANSYS,'tidy3d':TIDY},
            'scope':'Signed port modes have positive-real Ez at center plus one quarter-width. Complex S includes the straight access lengths. Guided output excludes radiated power. The independent calculation matches E and dE/dx and uses an outgoing transverse PML. Signed MPB overlaps are converted to electric amplitudes using their measured group-velocity signs.'}
    (dest/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return passed


def finite_height(args,dest):
    import meep as mp
    matrix = measure(args.resolution,args.offset,args.aperture,args.pml,args.after,dest,True)
    power = abs(matrix)**2
    with np.load(dest/'arrays.npz') as raw:
        indices = raw['device_neff_0'][0,:,0,:]
        crossfeed = max(float(np.max(abs(raw[f'reference_{col}'][col//2, 1-col%2, col//2] /
                                          raw[f'incident_{col}'])**2)) for col in range(4))
    differences = {'power_reciprocity_max':float(np.max(abs(power-power.transpose(0,2,1)))),
                   'largest_guided_output_sum':float(np.max(power.sum(axis=1))),
                   'straight_reference_offmode_input_power_max':crossfeed}
    passed = (np.isfinite(matrix).all() and differences['power_reciprocity_max']<.03
              and differences['largest_guided_output_sum']<1.03 and crossfeed<.01)
    report={'schema_version':1,'phase_convention_version':PHASE_VERSION,'case':'multimode-3d','solver':'Meep','version':mp.__version__,'actual_device':'CPU',
            'dimension':'3D finite-height silicon in oxide. Two quasi-TE guided channels per port.',
            'passed':bool(passed),'parameters':{'resolution':args.resolution,'offset_um':args.offset,
                'width_um':.875,'height_um':.25,'core_index':CORE,'cladding_index':CLADDING,
                'port_x_um':[-PORT_X,PORT_X],'aperture_um':[args.aperture,2.], 'pml_um':args.pml,
                'post_source_time_um_over_c':args.after},
            'channels':CHANNELS,'wavelength_um':(1/FREQUENCIES).tolist(),'power':power.tolist(),
            's_real':matrix.real.tolist(),'s_imag':matrix.imag.tolist(),'port_neff':indices.tolist(),
            'checks':differences,'vendor_sources':{'lumerical':ANSYS,'tidy3d':TIDY},
            'scope':'Two guided quasi-TE channels per side in the even-Z symmetry sector. Ey gauge at one quarter width on each port. Radiation is excluded; report absolute errors and mesh sensitivity for weak amplitudes.'}
    (dest/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finite',action='store_true',help='use a finite-height strip instead of the planar slab')
    parser.add_argument('--resolution',type=int,choices=range(16,97))
    parser.add_argument('--offset',type=float,default=OFFSET)
    parser.add_argument('--aperture',type=float,default=None)
    parser.add_argument('--pml',type=float,default=1.)
    parser.add_argument('--after',type=float,default=80.)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.aperture is None: args.aperture = 3. if args.finite else 4.
    if args.resolution is None: args.resolution = 20 if args.finite else 48
    if args.finite and (args.aperture > 3.5 or args.resolution > 32):
        parser.error('Finite-height example requires aperture ≤3.5 µm and resolution ≤32.')
    if not (0 <= args.offset <= .25 and 3 <= args.aperture <= 5 and .75 <= args.pml <= 1.5 and 50 <= args.after <= 180):
        parser.error('Offset, aperture, PML or post-source time outside the qualified study range.')
    if args.output.exists():parser.error('Choose a fresh output directory.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        dest=Path(tmp)/'run'
        dest.mkdir()
        passed=run(args,dest)
        dest.rename(args.output)
    if not passed:raise SystemExit('Numerical limits exceeded. Inspect report.json.')

if __name__=='__main__':
    main()
