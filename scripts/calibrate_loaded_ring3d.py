"""Full-vector 3D FDTDX simulation of a finite-height ring between two bus guides."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "backend").is_dir():
    sys.path.insert(0, str(ROOT))

from backend.adapters.fdtdx_ports_adapter import _ports_runtime
from backend.adapters.fdtdx_ports3d_adapter import canonical_source_profile, port_basis
from scripts.calibrate_ring import CASES, characterize
from scripts.measure_loaded_ring_losses import measure as measure_loaded_loss
from scripts.ring_stage_provenance import fingerprint as stage_fingerprint, public_fingerprint

LIGHT = 299792458.
SILICON, OXIDE = 3.48, 1.44
WAVELENGTHS = np.linspace(1.49, 1.61, 241)
BASIS_WAVELENGTHS = np.array([1.49, 1.52, 1.55, 1.58, 1.61])
PULSE_END = 2 * 4 / .14
PORTS = ("reflection", "through", "drop", "add_side")


def geometry(vendor, resolution, pml, aperture, interface='legacy'):
    case = CASES[vendor]
    radius, width = case['radius'], case['width']
    center = radius + width + case['gap']
    spans = (2*(radius+2.15+pml), 2*(center+aperture/2+.28+pml), 2*(aperture/2+.16+pml))
    # Even dimensions position the physical origin exactly on a cell boundary.
    shape = tuple(2*math.ceil(v*resolution/2) for v in spans)
    half_ap = round(aperture * resolution / 2)
    px = round((radius+.76)*resolution)
    sy = round(center*resolution)
    sx = round((-radius-1.55)*resolution+shape[0]/2)
    ports = [(shape[0]//2+sign*px, shape[1]//2+y*sy-half_ap, shape[2]//2-half_ap)
             for sign,y in [(-1,-1),(1,-1),(-1,1),(1,1)]]
    return dict(shape=shape, pml=round(pml*resolution), ports=ports, source_x=sx, half_ap=half_ap,
                vendor=vendor, resolution=resolution, radius_um=radius, width_um=width,
                height_um=case['height'], edge_gap_um=case['gap'], bus_y_um=center,
                top_index=1. if vendor=='lumerical' else OXIDE, substrate_index=OXIDE,
                silicon_index=SILICON, aperture_um=2*half_ap/resolution, pml_um=pml,
                ring_interface=interface)


def disk_fractions(xs, ys, dx, radius):
    """Area of a disk inside each square, from its analytic quadrant integral."""
    def integral(x, y):
        a = np.minimum(abs(x), radius)
        b = np.minimum(abs(y), radius)
        corner = np.minimum(a, np.sqrt(np.maximum(0., radius**2-b*b)))
        def circular_segment(t):
            return .5*(t*np.sqrt(np.maximum(0., radius**2-t*t)) + radius**2*np.arcsin(t/radius))
        return np.sign(x)*np.sign(y)*(b*corner + circular_segment(a)-circular_segment(corner))

    left = xs[:, None]-dx/2
    right = xs[:, None]+dx/2
    bottom = ys[None, :]-dx/2
    top = ys[None, :]+dx/2
    area = integral(right,top)-integral(left,top)-integral(right,bottom)+integral(left,bottom)
    return np.clip(area/(dx*dx), 0., 1.)


def lateral_fraction(xs, ys, dx, radius, width, bus, reference, interface):
    if interface=='exact-area':
        lo, hi = ys-dx/2, ys+dx/2
        straight = np.zeros_like(ys)
        for center in (-bus, bus):
            straight += np.maximum(0., np.minimum(hi, center+width/2)-np.maximum(lo, center-width/2))/dx
        fill = np.broadcast_to(straight, (len(xs),len(ys)))
        if not reference:
            outer = disk_fractions(xs, ys, dx, radius+width/2)
            inner = disk_fractions(xs, ys, dx, radius-width/2)
            fill = fill+outer-inner
        # The disjoint waveguides can share a cell. Roundoff on fully covered
        # or empty cells is the only reason to clamp their summed area.
        return np.clip(fill, 0., 1.)
    if interface!='legacy':
        raise ValueError('Unknown ring interface sampling method.')
    offsets = (np.arange(6)+.5)/6-.5
    fill = np.zeros((len(xs),len(ys)))
    for a in offsets:
        x = xs[:,None]+a*dx
        for b in offsets:
            y = ys[None,:]+b*dx
            inside = (abs(abs(y)-bus)<=width/2)
            if not reference:
                rho2 = x*x+y*y
                inside = inside | ((rho2 >= (radius-width/2)**2) & (rho2 <= (radius+width/2)**2))
            fill += inside
    return fill / len(offsets)**2


def permittivity(g, reference, silicon_epsilon=SILICON**2, oxide_epsilon=OXIDE**2, dtype=np.float32):
    """Sample actual component-centred Yee cells, including the substrate under the Si.

    Lateral area fractions use the selected 6×6 quadrature or analytic disk/strip intersection.
    Vertical series averaging for Ez and parallel averaging for Ex/Ey preserve planar interfaces.
    Lateral curved interfaces use scalar area averaging and remain mesh sensitive.
    """
    dx = 1/g['resolution']
    nx, ny, nz = g['shape']
    radius,width,height,bus = [g[k] for k in ('radius_um','width_um','height_um','bus_y_um')]
    eps = np.empty((3,nx,ny,nz), dtype)
    for component in range(3):
        xs = (np.arange(nx)-nx/2+.5*(component==0))*dx
        ys = (np.arange(ny)-ny/2+.5*(component==1))*dx
        zs = (np.arange(nz)-nz/2+.5*(component==2))*dx
        fill = lateral_fraction(xs,ys,dx,radius,width,bus,reference,g.get('ring_interface','legacy'))
        z0,z1 = zs-dx/2,zs+dx/2
        slab = (np.minimum(z1,height/2)-np.maximum(z0,-height/2)).clip(0,dx)/dx
        below = ((-height/2-z0)/dx).clip(0,1)
        background = g['top_index']**2
        slices = np.empty((nx,ny,nz),dtype)
        for k in range(nz):
            oxide, core = below[k], slab[k] * fill
            outer = 1-oxide-core
            if component==2:
                # Area average the two columns after harmonic z mixing within each.
                column = 1/(oxide/oxide_epsilon+slab[k]/silicon_epsilon+(1-oxide-slab[k])/background)
                outside = 1/(oxide/oxide_epsilon+(1-oxide)/background)
                slices[:,:,k] = fill*column+(1-fill)*outside
            else:
                slices[:,:,k] = oxide*oxide_epsilon+core*silicon_epsilon+outer*background
        eps[component] = slices
    return eps


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def stage(runtime, g, duration, reference, frozen_source=None):
    jax,jnp,fdtdx,target = runtime
    from fdtdx.core.physics.modes import compute_mode
    from fdtdx.core.physics.curl import interpolate_fields
    from fdtdx.fdtd.forward import forward
    from scripts.gpu_timing import StageTiming
    stage_started=time.monotonic()
    timer=StageTiming(target)

    dx=1/g['resolution']
    sim=fdtdx.SimulationConfig(time=duration*1e-6/LIGHT, resolution=dx*1e-6,
                               backend='gpu',dtype=jnp.float32,courant_factor=.99)
    if target.platform!='gpu' or sim.backend!='gpu':
        raise RuntimeError('A real NVIDIA GPU is required.')
    shape, pml = g['shape'], g['pml']
    volume=fdtdx.SimulationVolume(name='domain',partial_grid_shape=shape,material=fdtdx.Material(permittivity=OXIDE**2))
    boundary=fdtdx.BoundaryConfig(**{f'thickness_grid_{s}{a}':pml for s in ('min','max') for a in 'xyz'},
                                  **{f'boundary_type_{s}{a}':'pml' for s in ('min','max') for a in 'xyz'})
    bounds,constraints=fdtdx.boundary_objects_from_config(boundary,volume)
    aperture=2*g['half_ap']
    source=fdtdx.ModePlaneSource(name='launch',partial_grid_shape=(1,aperture,aperture),
        mode_index=0,filter_pol='te',direction='+',wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6),
        temporal_profile=fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(frequency=.14*LIGHT*1e6/(2*np.pi)),
            center_wave=fdtdx.WaveCharacter(wavelength=1.55e-6)),
        switch=fdtdx.OnOffSwitch(end_time=PULSE_END*1e-6/LIGHT))
    constraints=list(constraints)+[source.set_grid_coordinates(axes=(0,1,2),sides=('-','-','-'),
        coordinates=(g['source_x'],*g['ports'][0][1:]))]
    key=jax.random.PRNGKey(0)
    objects, arrays, params, sim, _ = fdtdx.place_objects(object_list=[volume,*bounds.values(),source],
        config=sim,constraints=constraints,key=key)
    ade=g.get('ring_material')=='palik-ade'
    basis_wavelengths=BASIS_WAVELENGTHS
    reciprocal=g.get('ring_interface')=='reciprocal-wall'
    wall=g.get('ring_interface') in ('wall-tensor','reciprocal-wall')
    links=link_info=None
    full_inverse=None
    if wall:
        from scripts.ring_wall_tensor import forward_lossless_xy, wall_inverse
        full_inverse=wall_inverse(g,reference,disk_fractions,lateral_fraction,permittivity)
        for x in (g['source_x'], *(p[0] for p in g['ports'])):
            if np.any(full_inverse[1,x]) or np.any(full_inverse[3,x]):
                raise ValueError('A requested source or port intersects a curved sidewall.')
        if reciprocal and not reference:
            from scripts.reciprocal_ring_wall import reciprocal_links
            links,link_info=reciprocal_links(full_inverse,1/g['silicon_index']**2,1/min(g['top_index'],g['substrate_index'])**2)
        diagonal=full_inverse[[0,4,8]]
        eps=1/diagonal
    elif ade:
        from scripts import ring_ade_stage as dispersion
        coefficients,eps,material_metadata=dispersion.build(g,reference,sim.time_step_duration,permittivity)
        basis_wavelengths=dispersion.material.WAVELENGTHS
    else:
        eps=permittivity(g,reference)
    material_sha=fingerprint(full_inverse if wall else eps)
    arrays=arrays.aset('inv_permittivities',jax.device_put(diagonal if wall else 1/eps,target))
    arrays, objects, _ = fdtdx.apply_params(arrays,objects,params,key)
    if arrays.fields.E.device!=target or not np.allclose(np.asarray(1/arrays.inv_permittivities),eps,rtol=1e-6):
        raise RuntimeError('Native FDTDX material or target differs from the requested 3D grid.')
    native=objects['launch']
    e,h,n=compute_mode(frequency=LIGHT/1.55e-6,inv_permittivities=native._inv_permittivity,
                       inv_permeabilities=native._inv_permeability,resolution=dx*1e-6,
                       direction='+',mode_index=0,filter_pol='te',dtype=jnp.float32)
    e,h,_=canonical_source_profile(e,h)
    if not OXIDE<float(np.real(n))<SILICON:
        raise RuntimeError('Unguided launch mode.')
    if (wall or ade) and frozen_source is not None:
        for current,key in [(e,'source_E'),(h,'source_H'),(native._time_offset_E,'source_offset_E'),(native._time_offset_H,'source_offset_H')]:
            np.testing.assert_allclose(np.asarray(current),frozen_source[key],rtol=2e-6,atol=1e-8)
        e,h=frozen_source['source_E'],frozen_source['source_H']
        native=native.aset('_time_offset_E',jax.device_put(frozen_source['source_offset_E'],target))
        native=native.aset('_time_offset_H',jax.device_put(frozen_source['source_offset_H'],target))
    native=native.aset('_E',jax.device_put(e,target)).aset('_H',jax.device_put(h,target))
    objects=objects.aset('object_list',[native if obj.name=='launch' else obj for obj in objects.object_list])
    weights=[]
    bases={}
    neff=[]
    edge=[]
    raw = {'source_E':e,'source_H':h,
           'permittivity_midplane':eps[1,:,:,shape[2]//2].copy(),
           'permittivity_cross_section':eps[:,g['source_x']].copy()}
    if links is not None:
        raw['reciprocal_links_midplane']=links[:,:,:,shape[2]//2].copy()
        link_info['links_sha256']=fingerprint(links)
        links=jax.device_put(links,target)
    if wall:
        raw.update(inverse_xy_midplane=full_inverse[[1,3],:,:,shape[2]//2],
                   source_offset_E=np.asarray(native._time_offset_E),source_offset_H=np.asarray(native._time_offset_H))
    if ade:
        raw.update(projection_wavelengths_um=basis_wavelengths,source_offset_E=np.asarray(native._time_offset_E),
                   source_offset_H=np.asarray(native._time_offset_H))
        source_drive=dispersion.audit_waveform(native)
        arrays=dispersion.attach(arrays,coefficients,target)
    for i,(x,y,z) in enumerate(g['ports']):
        section=arrays.inv_permittivities[:,x:x+1,y:y+aperture,z:z+aperture]
        digest=fingerprint(np.asarray(section))
        if ade:
            where=(slice(None),slice(x,x+1),slice(y,y+aperture),slice(z,z+aperture))
            digest += fingerprint(coefficients['c3'][where])
        if digest not in bases:
            if ade:
                bases[digest]=dispersion.port_profiles(coefficients,where,dx,sim.time_step_duration,OXIDE,SILICON)
            else:
                bases[digest]=port_basis(section,1/BASIS_WAVELENGTHS, LIGHT*1e6/BASIS_WAVELENGTHS,
                                          dx,sim.time_step_duration,OXIDE,SILICON)
        basis,details=bases[digest]
        mode_e,mode_h=basis['mode_e'],basis['mode_h']
        p=.5*np.real((mode_e[:,1]*mode_h[:,2].conj()-mode_e[:,2]*mode_h[:,1].conj()).sum(axis=(1,2)))*dx**2
        # Reciprocal overlap on the local positive-Ey gauge. Transverse lossless bases are real.
        terms=np.stack([mode_h[:,2],-mode_h[:,1],-mode_e[:,2],mode_e[:,1]],axis=1)
        if np.max(abs(terms.imag))>1e-4*np.max(abs(terms.real)) or np.any(p<=0):
            raise RuntimeError('Unsupported complex transverse basis or nonpositive modal power.')
        weights.append(jax.device_put((terms.real*dx**2/(4*np.sqrt(p)[:,None,None,None])).astype(np.float32),target))
        neff.append(details['neff_real']);edge.append(details['edge_electric_l2_fraction'])
        raw[f'basis_E_{i}']=mode_e
        raw[f'basis_H_{i}']=mode_h
    if wall and not reference:
        # All source/monitor cross-sections were checked to be exactly diagonal.
        # Evolve the device with all nine coefficients; the reference is exactly diagonal everywhere.
        arrays=arrays.aset('inv_permittivities',jax.device_put(full_inverse,target))
    dt=sim.time_step_duration*LIGHT*1e6
    stride=max(1,int(.20/dt))
    chunk=16 if wall else 64
    size=sim.time_steps_total//stride
    points=[(round(x/dx+shape[0]/2),round(y/dx+shape[1]/2),shape[2]//2)
            for x,y in ((g['radius_um'],0.),(0.,-g['radius_um']),(0.,g['radius_um']))]

    def sample(carry, _, material_links=None):
        def step(_,state):
            if wall and not reference:
                return forward_lossless_xy(state,sim,objects,material_links)
            return forward(state=state,config=sim,objects=objects,key=key,
                           record_detectors=False,record_boundaries=False,simulate_boundaries=True)
        prior=jax.lax.fori_loop(0,stride-1,step,carry)
        following=step(0,prior)
        new=following[1].fields
        old=prior[1].fields
        result=[]
        for (x,y,z),w in zip(g['ports'],weights):
            # Crop one halo on every side before calling FDTDX's native Ez-point collocation.
            sl=(slice(None),slice(x-1,x+2),slice(y-1,y+aperture+1),slice(z-1,z+aperture+1))
            ec,hc=interpolate_fields(new.E[sl],(old.H[sl]+new.H[sl])/2)
            coeff=w*jnp.stack([ec[1,0],ec[2,0],hc[1,0],hc[2,0]])
            a=coeff[:,:2].sum(axis=(1,2,3))
            b=coeff[:,2:].sum(axis=(1,2,3))
            result.append(jnp.stack([a+b,a-b],axis=-1))
        probes=jnp.stack([new.H[2,x,y,z] for x,y,z in points])
        return following,(jnp.stack(result),probes)
    state=(jnp.asarray(0,dtype=jnp.int32),arrays)
    # Fixed sample count per execution amortizes dispatch. The last chunk uses its actual length.
    jax.block_until_ready((state, weights, links))
    setup_seconds=time.monotonic()-stage_started
    start=time.monotonic()
    compiled={}
    for n_samples in {min(chunk,size),size % chunk} - {0}:
        with timer.compile():
            def advance(st,material_links):
                return jax.lax.scan(lambda carry,it:sample(carry,it,material_links),st,None,length=n_samples)
            compiled[n_samples]=jax.jit(advance,donate_argnums=(0,)).lower(state,links).compile()
    traces=[];probes=[];snapshots=[];snapshot_times=[]
    timer.begin_steps()
    for first in range(0,size,chunk):
        n_samples=min(chunk,size-first)
        with timer.step():
            state,(signal,probe)=compiled[n_samples](state,links)
            jax.block_until_ready((state,signal,probe))
        signal_np, probe_np=np.asarray(signal),np.asarray(probe)
        if not np.isfinite(signal_np).all() or not np.isfinite(probe_np).all():
            raise ValueError(f'Nonfinite native fields at step {int(state[0])}.')
        traces.append(signal_np);probes.append(probe_np)
        if not snapshot_times or (snapshot_times[-1]<PULSE_END+40<=float(state[0])*dt) or first+chunk>=size:
            snapshot_times.append(float(state[0])*dt)
            snapshots.append(np.asarray(state[1].fields.E[:,:,:,shape[2]//2]))
        if first%max(chunk,(size//8//chunk)*chunk)==0:
            print(f"{'reference' if reference else 'device'} {min(first+chunk,size)}/{size} samples",flush=True)
    timer.end_steps()
    if state[1].fields.E.device!=target:
        raise RuntimeError('Final fields left the requested GPU.')
    raw.update(time_um_c=np.arange(1,size+1)*stride*dt,projections=np.concatenate(traces),
               ring_Hz=np.concatenate(probes),probe_xyz_grid=np.asarray(points),
               E_midplane_snapshots=np.asarray(snapshots),snapshot_times_um_c=snapshot_times,
               mode_neff=np.array(neff),mode_edge_fraction=np.array(edge))
    if not all(np.isfinite(v).all() for v in raw.values()):
        raise RuntimeError('Nonfinite fields or modal data.')
    info=dict(device=str(target),device_name=target.device_kind,time_steps=int(state[0]),time_step_um_c=dt,
              elapsed_s=time.monotonic()-start,material_sha256=material_sha,source_mode_neff=float(np.real(n)),
              sampled_mode_edge_fraction_max=float(np.max(edge)),source_E_sha256=fingerprint(e),source_H_sha256=fingerprint(h))
    info['performance']=timer.report(shape=shape,steps=int(state[0]),dt_seconds=sim.time_step_duration,
                                     spacing_um=dx,pml_cells={f'{s}_{a}':int(pml) for a in 'xyz' for s in ('min','max')})
    interior=math.prod(n-2*pml for n in shape)
    info['performance'].update(setup_seconds=setup_seconds, interior_non_pml_cells=interior,
                               cells_touching_pml=math.prod(shape)-interior,
                               exclusions='Step scopes exclude compilation, host trace transfers, snapshots, Fourier analysis and checkpoint I/O; all six PML faces are included in the field evolution.')
    info['requested_time_steps']=int(sim.time_steps_total)
    info['unsampled_tail_steps']=int(sim.time_steps_total-state[0])
    if wall:
        info.update(evolution_tensor_components=arrays.inv_permittivities.shape[0],
                    curved_inverse_xy_nonzero_cells=int(np.count_nonzero(full_inverse[[1,3]])),
                    model='Sequential vertical-wall lamination, local radial normal, frozen positive-Ey source',
                    electric_update='Native diagonal' if reference else 'FDTDX lossless xy block, native CPML and H update')
    if link_info is not None:
        info.update(reciprocal_operator=link_info,electric_update='Reciprocal bounded edge coupling, native CPML and H update')
    if ade:
        info.update(material_validation=material_metadata,source_drive=source_drive)
    return raw,info


def directional_spectrum(data, fraction):
    """Fourier transform the synchronized native modes, with wavelength-resolved basis interpolation."""
    t=data['time_um_c']
    stop=max(1,int(len(t)*fraction))
    t=t[:stop]
    dft=np.exp(2j*np.pi*t[None,:]/WAVELENGTHS[:,None])
    traces=data['projections'][:stop]
    transformed=(dft @ traces.reshape(stop,-1)).reshape(len(WAVELENGTHS),*traces.shape[1:])*(t[1]-t[0])
    # Interpolate between lossless, identically gauged mode weights, never across optical frequencies.
    basis=data.get('projection_wavelengths_um',BASIS_WAVELENGTHS)
    i=np.searchsorted(basis,WAVELENGTHS,side='right')-1
    i=i.clip(0,len(basis)-2)
    a=(WAVELENGTHS-basis[i])/(basis[i+1]-basis[i])
    rows=np.arange(len(i))
    return (1-a[:,None,None])*transformed[rows,:,i,:]+a[:,None,None]*transformed[rows,:,i+1,:]


def qualified_spacing(peaks, valid):
    """Use consecutive detected peaks whose entire wavelength interval passes the gate."""
    adjacent=[]
    for left,right in zip(peaks,peaks[1:]):
        interval=(WAVELENGTHS>=left['wavelength_um']) & (WAVELENGTHS<=right['wavelength_um'])
        if interval.any() and valid[interval].all():
            adjacent.append((left,right))
    return min(adjacent,key=lambda pair:abs(sum(p['wavelength_um'] for p in pair)/2-1.55)) if adjacent else None


def field_stability(probes, collected_power, duration):
    peaks=[np.max(abs(chunk),axis=0) for chunk in np.array_split(probes,4)]
    growth=peaks[-1]/np.maximum(np.maximum(peaks[0],peaks[1]),np.finfo(float).tiny)
    late_checked=duration>2*PULSE_END
    severe_growth=late_checked and np.all(peaks[3]>peaks[2]) and np.all(peaks[2]>peaks[1]) and np.all(growth>8)
    passivity_failures=int(np.sum(collected_power>1.05))
    rejected=bool(severe_growth and passivity_failures>=len(collected_power)/10)
    return dict(usable_spectrum=not rejected,screen_checked=late_checked,
                reason=('After source shutoff all three ring probes grow by over eight times their first-half peaks, and collected power exceeds the passivity limit across over 10% of the band.' if rejected
                        else 'No joint severe post-pulse growth and widespread passivity failure.'),
                first_to_fourth_quarter_probe_peak_abs_Hz=[p.tolist() for p in peaks],
                fourth_to_first_half_peak_per_probe=growth.tolist(),passivity_failed_bins=passivity_failures,
                max_collected_power=float(np.max(collected_power)))


def assemble(g, vendor, stages, duration, output, versions, checkpoint_identity):
    ref,dev = [stages[k][1] for k in ('reference','device')]
    ade=g.get('ring_material')=='palik-ade'
    if g.get('ring_interface') in ('wall-tensor','reciprocal-wall'):
        for key in ('source_E','source_H','source_offset_E','source_offset_H'):
            np.testing.assert_array_equal(ref[key],dev[key])
    if ade:
        for key in ('source_E','source_H','source_offset_E','source_offset_H','projection_wavelengths_um'):
            if not np.array_equal(ref[key],dev[key]):
                raise ValueError(f'Unequal impressed source or port frequency input: {key}.')
        if stages['reference'][0]['source_drive']!=stages['device'][0]['source_drive']:
            raise ValueError('Impressed E/H source pulses differ.')
    for k in ('source_E','source_H'):
        if not np.allclose(ref[k],dev[k],rtol=1e-6,atol=1e-8):
            raise ValueError('Launch profiles differ between reference and device.')
    r=directional_spectrum(ref,1)
    incident=r[:,0,0]
    if not np.isfinite(incident).all() or np.any(abs(incident)<1e-14):
        raise ValueError('No broadband guided incident field.')
    def normalized(a):
        return np.stack([a[:,0,1]-r[:,0,1],a[:,1,0],a[:,2,1],a[:,3,0]],axis=1)/incident[:,None]
    full=directional_spectrum(dev,1)
    s=normalized(full); early=normalized(directional_spectrum(dev,.8))
    power=abs(s)**2; difference=np.max(abs(power-abs(early)**2),axis=1)
    straight=abs(r[:,1,0]/incident)**2
    ratio=full[:,0,0]/incident
    peaks=characterize(WAVELENGTHS,power[:,2],duration-PULSE_END)
    valid=(abs(incident)>abs(incident).max()*.03)&(abs(straight-1)<.04)&(power.sum(axis=1)<=1.05)&(difference<.05)
    valid &= (abs(abs(ratio)-1)<.05)&(abs(np.angle(ratio))<.06)
    # Never report a finite-window resonance from an isolated reassuring wavelength sample.
    in_band=np.ones(len(WAVELENGTHS),dtype=bool)
    if ade:
        in_band=(WAVELENGTHS>=1.5)&(WAVELENGTHS<=1.6)
        valid &= in_band
    stability=field_stability(dev['ring_Hz'],power.sum(axis=1),duration)
    local_valid=valid.copy()
    stability['sample_gates_passed_before_stability']=int(local_valid.sum())
    if not stability['usable_spectrum']:
        valid[:]=False
    reliable=[p for p in peaks if valid[np.argmin(abs(WAVELENGTHS-p['wavelength_um']))]]
    pair=qualified_spacing(peaks,valid)
    fsr=(pair[1]['wavelength_um']-pair[0]['wavelength_um'])*1000 if pair else None
    published=CASES[vendor]['reference_peaks_um']
    external_pair=min(zip(published,published[1:]),key=lambda p:abs(sum(p)/2-1.55))
    external_fsr=1000*(external_pair[1]-external_pair[0])
    # Peak spacings are meaningful only when adjacent peaks and their neighborhood are settled.
    passed=bool(valid[in_band].all() and pair is not None)
    row=dict(wavelength_um=WAVELENGTHS.tolist(),valid_bins=valid.tolist(),**{k:power[:,i].tolist() for i,k in enumerate(PORTS)})
    data={'s':s,'s_early':early,'incident':incident,'valid_bins':valid,'local_valid_bins':local_valid,'wavelength_um':WAVELENGTHS,
          'basis_wavelength_um':ref.get('projection_wavelengths_um',BASIS_WAVELENGTHS),'reference_straight_power':straight,'incident_device_over_reference':ratio}
    for label,(info,arrays) in stages.items():
        data.update({label+'_'+key:value for key,value in arrays.items()})
    np.savez_compressed(output/'arrays.npz',**data)
    report=dict(case_id='loaded-ring-3d',vendor=vendor,solver='fdtdx',device='gpu',
        actual_device=stages['device'][0]['device'],device_name=stages['device'][0]['device_name'],
        passed=passed,measurement_complete=True,physical_model=g,versions=versions,checkpoint_identity=checkpoint_identity,
        stages={k:v[0] for k,v in stages.items()},series={vendor:row},stability=stability,
        duration_um_c=duration,early_fraction=.8,peaks=peaks,qualified_peaks=reliable,
        measured_drop_fsr_nm=fsr,published_drop_fsr_nm=external_fsr,
        published_peaks_um=published,source_url=CASES[vendor]['source'],reference_spacing_pair_um=list(external_pair),
        measured_spacing_pair_um=[p['wavelength_um'] for p in pair] if pair else [],
        valid_bins=int(valid.sum()),total_bins=len(WAVELENGTHS),
        max_power_change_early_to_final=float(difference.max()),straight_reference_power_range=[float(straight.min()),float(straight.max())],
        max_collected_port_power=float(power.sum(axis=1).max()),incident_ratio_magnitude_range=[float(abs(ratio).min()),float(abs(ratio).max())],
        arrays_file='arrays.npz',arrays_sha256=hashlib.sha256((output/'arrays.npz').read_bytes()).hexdigest(),
        limits='Lossless, fixed central indices. Modal E/H overlaps interpolate five independently solved mode profiles across the band. FSR is loaded drop-peak spacing. Intrinsic/external rate estimates require the independent spectral/temporal gates in loaded_loss.')
    if g.get('ring_interface') in ('wall-tensor','reciprocal-wall'):
        report['limits']+=' The rotated sidewall model uses native off-diagonal electric updates; subcell corners use sequential mixing. The midplane display stores the reciprocal inverse yy coefficient.'
        if g['ring_interface']=='reciprocal-wall':
            report['limits']+=' The evolution uses shared Ex/Ey edge-pair coefficients bounded by the material eigenvalue range. Accuracy requires mesh and vendor comparisons.'
        report['physical_port_x_um']=[(x-g['shape'][0]/2)/g['resolution'] for x,y,z in g['ports']]
    if ade:
        report['material_band_bins']=int(in_band.sum())
        report['qualified_material_band_um']=[1.5,1.6]
        report['limits']='Undamped one-pole fit to the 1500–1600 nm public samples, diagonal Yee coefficients, fixed impressed carrier mode and five frequency-specific port bases. Frequency margins are diagnostic only. Loss separation assumes equal buses.'
    nx,ny,_=g['shape']
    snapshot=int(np.argmin(abs(dev['snapshot_times_um_c']-(PULSE_END+40))))
    report['field_preview']=dict(array='device_E_midplane_snapshots',index=[snapshot,1],axis_order='xy',
        horizontal_axis='x',vertical_axis='y',component='Ey',
        horizontal_um=((np.arange(nx)-nx/2)/g['resolution']).tolist(),
        vertical_um=((np.arange(ny)-ny/2+.5)/g['resolution']).tolist(),
        time_um_c=float(dev['snapshot_times_um_c'][snapshot]),
        description='Instantaneous Ey electric field at z = 0 µm')
    report['loaded_loss']=measure_loaded_loss(output,report)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    from scripts.calibration_checkpoint import MeasurementCheckpoints
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor',choices=tuple(CASES),required=True)
    parser.add_argument('--device',choices=['gpu'],required=True)
    parser.add_argument('--resolution',type=int,choices=(12,16,20,24),default=16)
    parser.add_argument('--ring-interface', choices=('legacy','exact-area','wall-tensor','reciprocal-wall'), default='legacy',
                        help='How the ring and straight guides occupy the lateral Yee cells.')
    parser.add_argument('--ring-material',choices=('constant','palik-ade'),default='constant')
    parser.add_argument('--duration-um-c',type=float,help='Total recording in µm/c (Lumerical 3200, Tidy3D 4800).')
    parser.add_argument('--pml-um',type=float,default=.65)
    parser.add_argument('--aperture-um',type=float,default=1.8)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.duration_um_c is None:
        args.duration_um_c=3200. if args.vendor=='lumerical' else 4800.
    if not 100<=args.duration_um_c<=6000 or not .5<=args.pml_um<=1.2 or not 1.5<=args.aperture_um<=2.5:
        parser.error('Require duration 100–6000 µm/c, PML 0.5–1.2 µm and aperture 1.5–2.5 µm.')
    if args.ring_interface in ('wall-tensor','reciprocal-wall') and args.resolution<16:
        parser.error('The separated-wall tensor control requires at least 16 cells/µm.')
    if args.ring_material=='palik-ade' and (args.vendor!='lumerical' or args.ring_interface!='exact-area'):
        parser.error('The Palik material control requires the exact-area Lumerical ring.')
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Choose an empty result directory.')
    runtime=_ports_runtime(args.device)
    versions={k:importlib.metadata.version(k) for k in ('jax','jaxlib','fdtdx','scipy','numpy','tidy3d')}
    g=geometry(args.vendor,args.resolution,args.pml_um,args.aperture_um,args.ring_interface)
    provenance, reusable = stage_fingerprint(ROOT, runtime, versions)
    if args.ring_interface=='reciprocal-wall':
        from scripts.reciprocal_ring_wall import fresh_admission,worker_hashes
        provenance['source_files'].update(worker_hashes(ROOT))
    if args.ring_material=='palik-ade':
        from scripts.ring_palik_dispersion import MATERIAL_FILE, MATERIAL_SOURCE_SHA256
        g['ring_material']=args.ring_material
        g['material_band_um']=[1.5,1.6]
        g['material_sha256']=MATERIAL_SOURCE_SHA256
        for name in ('scripts/ring_ade_stage.py','scripts/ring_palik_dispersion.py',MATERIAL_FILE):
            provenance['source_files'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if provenance['source_files'][MATERIAL_FILE]!=MATERIAL_SOURCE_SHA256:
            raise ValueError('The captured Palik input differs from the fitted model.')
    if args.checkpoint and not reusable:
        print('Persistent checkpoint reuse disabled. ' + provenance['reuse_disabled_reason'], flush=True)
    class StageCache(MeasurementCheckpoints):
        def __init__(self,directory):
            self.root=Path(directory)
            self.root.mkdir(exist_ok=True,parents=True)
            self.provenance=provenance
    with tempfile.TemporaryDirectory() as tmp:
        cache=StageCache(args.checkpoint if args.checkpoint and reusable else tmp)
        stages={}
        acquisitions={}
        for label in ('reference','device'):
            is_reference=label=='reference'
            interval=150. if is_reference else args.duration_um_c
            settings=json.loads(json.dumps({'geometry':g,'duration_um_c':interval,'basis':BASIS_WAVELENGTHS.tolist()}))
            frozen_source=None
            if args.ring_material=='palik-ade':
                settings['basis']=[1.5,1.525,1.55,1.575,1.6]
            if not is_reference and (args.ring_material=='palik-ade' or args.ring_interface in ('wall-tensor','reciprocal-wall')):
                frozen_source=stages['reference'][1]
                settings['frozen_source_sha256']={key:fingerprint(frozen_source[key]) for key in
                    ('source_E','source_H','source_offset_E','source_offset_H')}
            calculated_here=False
            def calculate():
                nonlocal calculated_here
                calculated_here=True
                admission=fresh_admission(ROOT,g,runtime[3],provenance) if args.ring_interface=='reciprocal-wall' else None
                raw,info=stage(runtime,g,interval,is_reference,frozen_source)
                if admission:
                    info['fresh_memory_admission']=admission
                return info,raw
            stages[label]=cache.acquire(label,settings,calculate)
            acquisitions[label]='computed_this_invocation' if calculated_here else 'reused_verified_receipt'
            print(label,'complete',stages[label][0]['device'],flush=True)
        identity = {'persistent_reuse_eligible': reusable, 'provenance': public_fingerprint(provenance), 'invocation_acquisition': acquisitions}
        result=assemble(g,args.vendor,stages,args.duration_um_c,args.output,versions,identity)
        print('Measured',result['valid_bins'],'of',result['total_bins'],'qualified bins; FSR (nm)',result['measured_drop_fsr_nm'],flush=True)


if __name__=='__main__':
    main()
