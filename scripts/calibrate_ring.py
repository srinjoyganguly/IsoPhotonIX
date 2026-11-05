"""Two-bus Meep ring spectra with calibrated fundamental-mode ports in a planar guide."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks

ANSYS_URL = 'https://optics.ansys.com/hc/en-us/articles/360042800213-Ring-resonator-getting-started-Final-parameter-extraction'
ANSYS_DATA = 'https://optics.ansys.com/hc/en-us/article_attachments/50334805245587'
TIDY_URL = 'https://www.flexcompute.com/tidy3d/examples/notebooks/RingResonator/'
TIDY_NOTEBOOK = 'https://github.com/flexcompute/tidy3d-notebooks/blob/18fd47f8bd87ad89582b507c20bf4997f79d55c0/RingResonator.ipynb'
CASES = {
    'lumerical': dict(radius=3.1, width=.4, height=.18, gap=.10,
                      reference_peaks_um=[1.52463718,1.55009208,1.57724075], source=ANSYS_URL),
    'tidy3d': dict(radius=5., width=.43, height=.22, gap=.10,
                   reference_peaks_um=[1.51242,1.52927,1.54701,1.56475,1.58293], source=TIDY_URL),
}
WAVELENGTHS = np.linspace(1.48,1.62,561)
SAMPLE_DT = .25
CLADDING = 1.44
SILICON = 3.48


def slab_mode(core, width, wavelength, magnetic):
    """Symmetric slab fundamental from transverse phase continuity (µm)."""
    k = 2 * math.pi/wavelength
    v = k*width/2 * math.sqrt(core**2 - CLADDING**2)
    ratio = (core/CLADDING)**2 if magnetic else 1
    def characteristic(u):
        return u*math.tan(u) - ratio*math.sqrt(max(0.,v*v-u*u))
    u = brentq(characteristic,1e-12,min(v,math.pi/2)-1e-12)
    return math.sqrt(core**2 - (2*u/(k*width))**2)


def finite_height_fsr(case, top_index, resolution, span):
    """Straight finite-height TE-like strip mode and its local group-delay estimate."""
    import meep as mp
    from meep import mpb
    mpb.verbosity(0)
    half_height = case['height']/2
    substrate_thickness = span/2-half_height
    geometry = [mp.Block(size=mp.Vector3(mp.inf,mp.inf,substrate_thickness),
                         center=mp.Vector3(z=-half_height-substrate_thickness/2),
                         material=mp.Medium(index=CLADDING)),
                mp.Block(size=mp.Vector3(mp.inf,case['width'],case['height']),
                         material=mp.Medium(index=SILICON))]
    samples = []
    for wl in (1.545,1.55,1.555):
        f = 1/wl
        modes = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0,span,span)),
                  geometry=geometry,default_material=mp.Medium(index=top_index),
                  resolution=resolution,num_bands=1,deterministic=True,tolerance=1e-9)
        capture = {}
        def check_mode(ms, band):
            e = np.asarray(ms.get_efield(band,bloch_phase=False))
            magnitude = np.sum(abs(e.reshape(-1,3))**2,axis=0)
            capture.update(ey_fraction=float(magnitude[1]/magnitude.sum()),frequency=float(ms.freqs[band-1]))
        wavevector = modes.find_k(mp.ODD_Y,f,1,1,mp.Vector3(x=1),1e-8,2.4*f,
                                  CLADDING*f*1.00001,SILICON*f,check_mode)[0]
        if capture['ey_fraction'] < .5 or abs(capture['frequency']/f-1)>1e-5:
            raise RuntimeError('Incorrect parity, polarization or spectral residual in straight mode.')
        samples.append({'wavelength_um':wl,'neff':float(wavevector/f),**capture})
    ng=samples[1]['neff']-1.55*(samples[2]['neff']-samples[0]['neff'])/.01
    return dict(resolution=resolution,span_um=span,samples=samples,group_index=ng,
                predicted_fsr_nm=1000*1.55**2/(ng*2*math.pi*case['radius']))


def record(case, resolution, pml, pad, aperture, duration, with_ring):
    import meep as mp
    radius,width = case['radius'],case['width']
    y_bus = radius+width+case['gap']
    core = mp.Medium(index=slab_mode(SILICON,case['height'],1.55,False))
    clad = mp.Medium(index=CLADDING)
    geom = [mp.Block(size=(mp.inf,width,mp.inf),center=(0,y),material=core) for y in (-y_bus,y_bus)]
    if with_ring:
        geom.extend([mp.Cylinder(radius=radius+width/2,material=core),
                     mp.Cylinder(radius=radius-width/2,material=clad)])
    src = mp.EigenModeSource(src=mp.GaussianSource(1/1.55,fwidth=.14,cutoff=4),
             center=(-radius-1.85,-y_bus),size=(0,aperture),eig_band=1,eig_parity=mp.EVEN_Z)
    sim = mp.Simulation(cell_size=(2*(radius+2.4+pml+pad),2*(y_bus+1.2+pad+pml)),
              resolution=resolution,geometry=geom,default_material=clad,sources=[src],boundary_layers=[mp.PML(pml)])
    x = radius+.85
    planes = [(-x,-y_bus),(x,-y_bus),(-x,y_bus),(x,y_bus)]
    times, fields, probe = [], [], []
    def sample(s):
        times.append(s.meep_time())
        fields.append([[s.get_array(center=mp.Vector3(xx,yy),size=(0,aperture),component=c)
                        for c in (mp.Ey,mp.Hz)] for xx,yy in planes])
        probe.append(s.get_field_point(mp.Hz,mp.Vector3(radius,0)).real)
    sim.run(mp.at_every(SAMPLE_DT,mp.synchronized_magnetic(sample)),until=duration)
    axes = np.array([sim.get_array_metadata(center=mp.Vector3(xx,yy),size=(0,aperture))[1]-yy
                     for xx,yy in planes])
    sim.reset_meep()
    ts = np.array(times)
    if len(ts)<10 or not np.allclose(np.diff(ts),SAMPLE_DT,atol=1e-10,rtol=0):
        raise RuntimeError('Nonuniform port time sampling.')
    return ts, np.array(fields), axes, np.array(probe)


def project(ts, fields, axes, width, film):
    """Paired tangential Ey/Hz overlap with analytic left- and right-going modes."""
    dy = np.diff(axes,axis=1)
    if not np.allclose(dy,dy[:,0,None]):
        raise ValueError('Expected a uniform transverse port mesh.')
    projection = np.empty((len(WAVELENGTHS),4,2),complex)
    for first in range(0,len(WAVELENGTHS),64):
        band = WAVELENGTHS[first:first+64]
        dft = np.exp(2j*np.pi*ts[None,:]/band[:,None]) @ fields.reshape(len(ts),-1) * SAMPLE_DT
        dft = dft.reshape(len(band),*fields.shape[1:])
        for j,wavelength in enumerate(band):
            neff = slab_mode(film,width,wavelength,True)
            k = 2*np.pi/wavelength
            transverse = k*math.sqrt(film**2-neff**2)
            decay = k*math.sqrt(neff**2-CLADDING**2)
            distance = abs(axes)
            h = np.where(distance<=width/2,np.cos(transverse*distance),
                         np.cos(transverse*width/2)*np.exp(-decay*(distance-width/2)))
            e = neff*h/np.where(distance<=width/2,film**2,CLADDING**2)
            # All four guides use the same width and stack. Normalize before
            # forming port ratios to allow for a different sub-grid y offset.
            normal = 2*np.sqrt(np.trapezoid(e*h,axes,axis=-1))
            projection[first+j,:,0] = np.trapezoid(dft[j,:,0]*h+dft[j,:,1]*e,axes,axis=-1)/normal
            projection[first+j,:,1] = np.trapezoid(-dft[j,:,0]*h+dft[j,:,1]*e,axes,axis=-1)/normal
    return projection


def characterize(wavelengths, drop, post_source_record=None):
    peaks,_ = find_peaks(drop,prominence=.03,distance=max(1,int(.010/np.min(np.diff(wavelengths)))))
    rows=[]
    for i in peaks:
        if not 1.49 < wavelengths[i] < 1.61:
            continue
        half = drop[i]/2
        left=np.flatnonzero(drop[:i]<half); right=np.flatnonzero(drop[i+1:]<half)
        row={'wavelength_um':float(wavelengths[i]),'drop_power':float(drop[i]),'loaded_q_half_max':None}
        if len(left) and len(right):
            a,b=left[-1],i+1+right[0]
            # Require the half-maximum intersections to span four sampled
            # wavelength intervals before quantifying Q at this resolution.
            lo=np.interp(half,drop[a:a+2],wavelengths[a:a+2])
            hi=np.interp(half,drop[b-1:b+1][::-1],wavelengths[b-1:b+1][::-1])
            temporal_width=wavelengths[i]**2/post_source_record if post_source_record else 0.
            if hi-lo >= max(4*np.min(np.diff(wavelengths)),2*temporal_width):
                row['loaded_q_half_max']=float(wavelengths[i]/(hi-lo))
        rows.append(row)
    return rows


def run_case(name,setting,arrays):
    case=CASES[name]
    res,pml,pad,aperture,duration=setting
    model=slab_mode(SILICON,case['height'],1.55,False)
    ts,fields,axes,_=record(case,res,pml,pad,aperture,140,False)
    reference=project(ts,fields,axes,case['width'],model)
    ts,fields,axes,probe=record(case,res,pml,pad,aperture,duration,True)
    full=project(ts,fields,axes,case['width'],model)
    cut=ts<=duration*.8
    early=project(ts[cut],fields[cut],axes,case['width'],model)
    incident=reference[:,0,0]
    if np.min(abs(incident)) < .01*np.max(abs(incident)):
        raise RuntimeError('Insufficient incident mode in part of the requested band.')
    def normalized(coeff):
        return np.stack([coeff[:,0,1]-reference[:,0,1], coeff[:,1,0],coeff[:,2,1],coeff[:,3,0]],axis=1)/incident[:,None]
    s=normalized(full); power=abs(s)**2; prev=abs(normalized(early))**2
    straight=abs(reference[:,1,0]/incident)**2
    prefix=f'{name}_r{res}_p{pml}_pad{pad}_a{aperture}'
    arrays[prefix+'_reference_modes']=reference; arrays[prefix+'_device_modes']=full
    arrays[prefix+'_s']=s; arrays[prefix+'_s_early']=normalized(early)
    arrays[prefix+'_ring_time']=ts; arrays[prefix+'_ring_Hz']=probe
    post_source_record=duration-2*4/.14
    peaks=characterize(WAVELENGTHS,power[:,2],post_source_record)
    report=dict(resolution_cells_per_um=res,pml_um=pml,cladding_padding_um=pad,port_aperture_um=aperture,
            duration_um_c=duration,early_duration_um_c=duration*.8,array_prefix=prefix,
            post_source_record_um_c=post_source_record,
            peaks=peaks,straight_reference_power_range=[float(straight.min()),float(straight.max())],
            port_power_sum_range=[float(power.sum(axis=1).min()),float(power.sum(axis=1).max())],
            valid_bins=((np.max(abs(power-prev),axis=1)<=.05) & (power.sum(axis=1)<=1.03)).tolist(),
            max_power_change_early_to_final=float(np.max(abs(power-prev))),
            final_window_ring_Hz_peak_ratio=float(np.max(abs(probe[ts>.95*duration]))/np.max(abs(probe))))
    return report,power


def isolated_pole(case, order, resolution, radial_padding, pml, traces, key, oxide_top):
    """Full-vector 3D bare-ring pole using its exact exp(i*m*phi) symmetry."""
    import meep as mp
    r,w,h=case['radius'],case['width'],case['height']
    sr=r+w/2+radial_padding+pml
    sz=h+2*(radial_padding+pml)
    material=mp.Medium(index=SILICON)
    geometry=[mp.Block(size=mp.Vector3(mp.inf,mp.inf,sz),center=mp.Vector3(z=-h/2-sz/2),
                       material=mp.Medium(index=CLADDING)),
              mp.Block(size=mp.Vector3(w,mp.inf,h),center=mp.Vector3(r,0,0),material=material)]
    point=mp.Vector3(r+.13*w,0,0)
    pulse=mp.GaussianSource(1/1.55,fwidth=.16,cutoff=4)
    sim=mp.Simulation(cell_size=mp.Vector3(sr,0,sz),dimensions=mp.CYLINDRICAL,m=order,
                      default_material=mp.Medium(index=CLADDING if oxide_top else 1),
                      resolution=resolution,boundary_layers=[mp.PML(pml)],geometry=geometry,
                      sources=[mp.Source(pulse,component=mp.Er,center=mp.Vector3(r,0,h*.06))])
    samples,ts=[],[]
    probes=[mp.Vector3(r+dr*w,0,dz*h) for dr in (-.17,.17) for dz in (-.25,.25)]
    def observe(simulation):
        ts.append(simulation.meep_time())
        samples.append([[simulation.get_field_point(comp,v) for comp in (mp.Er,mp.Ez)]
                        for v in probes])
    def dominant(monitor):
        roots=[mode for mode in monitor.modes if abs(mode.freq-1/1.55)<.079 and abs(mode.amp)>0]
        if not roots:
            raise RuntimeError(f'No finite-height resonance resolved for angular order {order}.')
        mode=max(roots,key=lambda root:abs(root.amp))
        return dict(wavelength_um=float(1/mode.freq),frequency_per_um=float(mode.freq),
                    imaginary_frequency_per_um=float(mode.decay),field_amplitude=float(abs(mode.amp)),
                    harminv_error_magnitude=float(abs(mode.err)))
    first=mp.Harminv(mp.Er,point,1/1.55,.16)
    sim.run(mp.after_sources(first),mp.at_every(.25,observe),until_after_sources=110)
    second=mp.Harminv(mp.Er,point,1/1.55,.16)
    sim.run(second,mp.at_every(.25,observe),until=80)
    a=dominant(first);b=dominant(second)
    sim.reset_meep()
    field=np.asarray(samples)
    late=np.asarray(ts)>ts[-1]-60
    radial_power=float(np.sum(abs(field[late,:,0])**2))
    vertical_power=float(np.sum(abs(field[late,:,1])**2))
    radial_fraction=radial_power/(radial_power+vertical_power)
    traces[key+'_time']=np.asarray(ts)
    traces[key+'_Er_Ez']=field
    checked=(abs(a['wavelength_um']-b['wavelength_um'])<.001 and radial_fraction>.5
             and abs(b['imaginary_frequency_per_um'])<.001)
    return dict(angular_order=int(order),resolution=resolution,radial_padding_um=radial_padding,
                pml_um=pml,first_post_source_window=a,second_post_source_window=b,
                sample_core_radial_fraction=radial_fraction,pole_repeat_passed=bool(checked))


def pole_wavelength(row):
    return row['second_post_source_window']['wavelength_um']


def bracketed_pole_pair(rows):
    ordered=sorted(rows,key=pole_wavelength)
    for left,right in zip(ordered,ordered[1:]):
        if pole_wavelength(left)<=1.55<=pole_wavelength(right):
            if abs(left['angular_order']-right['angular_order'])!=1:
                raise RuntimeError('The ring spectrum has a missing angular order.')
            return [left,right]
    raise RuntimeError('The measured poles do not bracket 1.55 µm.')


def cover_band_center(name, resolution, orders, samples, traces):
    """Retain each measured order and extend along one physical azimuthal branch."""
    def acquire(order):
        if order not in samples:
            sample=isolated_pole(CASES[name],order,resolution,.80,.75,traces,
                                 f'{name}_r{resolution}_m{order}',name=='tidy3d')
            samples[order]=sample
            print(name,'isolated pole',order,resolution,pole_wavelength(sample),flush=True)
    for order in sorted(orders):
        acquire(order)
    for _ in range(8):
        values=[pole_wavelength(row) for row in samples.values()]
        if min(values)<=1.55<=max(values):
            return [samples[m] for m in sorted(samples)]
        acquire(max(samples)+1 if min(values)>1.55 else min(samples)-1)
    raise RuntimeError(f'{name} isolated poles do not bracket the band center at mesh {resolution}.')


def summarize_bare_ring(name, m0, straight, runs, boundary, traces):
    pair=bracketed_pole_pair(runs['r48'])
    finest=np.array([pole_wavelength(row) for row in pair])
    fsr=float((finest[1]-finest[0])*1000)
    r36={row['angular_order']:row for row in runs['r36']}
    for row in pair:
        order=row['angular_order']
        if order not in r36:
            r36[order]=isolated_pole(CASES[name],order,36,.80,.75,traces,
                                     f'{name}_r36_m{order}',name=='tidy3d')
    runs['r36']=[r36[m] for m in sorted(r36)]
    same_order=np.sort([pole_wavelength(r36[row['angular_order']]) for row in pair])
    vendor=np.array(CASES[name]['reference_peaks_um'])
    k=int(np.argmin(abs((vendor[:-1]+vendor[1:])/2-1.55)))
    published=float((vendor[k+1]-vendor[k])*1000)
    coarse={r['angular_order']:r for r in runs['r24']}
    print(name,'finest pair bracketing 1.55',finest.tolist(),flush=True)
    return dict(physical_scope='Finite-height isolated rotationally symmetric ring. No bus/loading. Outgoing cylindrical Maxwell fields in 3D.',
        order_selection_from_straight_mode=m0, straight_mode=straight, mesh_runs=runs, larger_boundary=boundary,
        measured_bare_ring_pair_um=finest.tolist(), measured_bare_ring_fsr_nm=fsr,
        measured_pair_orders=[row['angular_order'] for row in pair],
        same_order_r36_pair_um=same_order.tolist(),r36_to_r48_fsr_change_nm=fsr-1000*np.diff(same_order)[0],
        published_loaded_pair_um=vendor[k:k+2].tolist(), published_loaded_ring_fsr_nm=published,
        bare_minus_loaded_reference_nm=fsr-published,
        maximum_mesh_peak_shift_nm=max(1000*abs(pole_wavelength(coarse[m])-pole_wavelength(row))
                                       for m,row in r36.items() if m in coarse),
        boundary_peak_shift_nm=1000*abs(pole_wavelength(boundary)-pole_wavelength(coarse[m0])),
        checks_passed=all(row['pole_repeat_passed'] for rows in runs.values() for row in rows)
                      and boundary['pole_repeat_passed'])


def finite_ring_sweep(traces):
    """Measure neighboring full-vector cylindrical poles bracketing 1.55 µm at each mesh."""
    studies={}
    for name,case in CASES.items():
        top=CLADDING if name=='tidy3d' else 1.
        straight=finite_height_fsr(case,top,40,3.)
        m0=round(2*math.pi*case['radius']*straight['samples'][1]['neff']/1.55)
        runs={}
        orders=set(range(m0-1,m0+2))
        for resolution in (24,36,48):
            rows=cover_band_center(name,resolution,orders,{},traces)
            runs[f'r{resolution}']=rows
            seed=bracketed_pole_pair(rows) if resolution>=36 else rows
            orders={row['angular_order'] for row in seed}
        boundary=isolated_pole(case,m0,24,1.10,1.,traces,f'{name}_boundary_m{m0}',name=='tidy3d')
        studies[name]=summarize_bare_ring(name,m0,straight,runs,boundary,traces)
    return studies


def run(out, isolated_only=False):
    import meep as mp
    mp.verbosity(0)
    arrays={'wavelength_um':WAVELENGTHS}; series={}; sensitivity={}; comparisons=[]; checks={}
    isolated=finite_ring_sweep(arrays)
    if isolated_only:
        np.savez_compressed(out/'arrays.npz',**arrays)
        report=dict(schema_version=1,case_id='isolated-ring-poles',solver='Meep CPU',
                    passed=all(row['checks_passed'] for row in isolated.values()),cases=isolated)
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    for name,case in CASES.items():
        settings={'coarse':(20,.8,.8,2.,1000.),'fine':(28,.8,.8,2.,1000.)}
        if name=='lumerical':
            settings.update(pml=(20,1.2,.8,2.,1000.),padding=(20,.8,1.3,2.,1000.),
                            aperture=(20,.8,.8,2.4,1000.))
        rows={}; powers={}
        for tag,setting in settings.items():
            print(f'{name}: starting {tag}',flush=True)
            rows[tag],powers[tag]=run_case(name,setting,arrays)
            print(tag,rows[tag]['peaks'],rows[tag]['max_power_change_early_to_final'],flush=True)
        primary=rows['fine']
        if len(primary['peaks']) < 3:
            raise RuntimeError(f'Insufficient resolvable {name} drop peaks.')
        peak_wl=np.array([p['wavelength_um'] for p in primary['peaks']])
        center=(peak_wl[:-1]+peak_wl[1:])/2
        index=np.argmin(abs(center-1.55))
        local_fsr=float(np.diff(peak_wl)[index]*1000)
        vendor=np.array(case['reference_peaks_um'])
        index=np.argmin(abs((vendor[:-1]+vendor[1:])/2-1.55))
        vendor_fsr=float(np.diff(vendor)[index]*1000)
        top_index = 1.0 if name=='lumerical' else CLADDING
        mode_checks = [finite_height_fsr(case,top_index,mesh,span)
                       for mesh,span in [(48,3.),(80,3.),(80,4.)]]
        comparisons.append(dict(vendor='Ansys Lumerical' if name=='lumerical' else 'Tidy3D',
           observable='Adjacent drop resonance spacing with interval midpoint nearest 1.55 µm',
           local_fsr_nm=local_fsr,reference_fsr_nm=vendor_fsr,delta_nm=local_fsr-vendor_fsr,
           reference_peaks_um=vendor.tolist(), finite_height_straight_mode_estimates=mode_checks,
           finite_height_estimate_minus_vendor_nm=mode_checks[-1]['predicted_fsr_nm']-vendor_fsr, source_url=case['source'],scope=
           'Matched ring radius, width, height and gap. Local 2D vertical effective index replaces a finite, dispersive three-dimensional guide. FSR is a diagnostic, not a matched 3D equivalence test.'))
        sensitivity[name]={key:{'max_spectral_power_change_vs_coarse':float(np.max(abs(p-powers['coarse']))),
                      'max_drop_peak_shift_nm_vs_coarse':float(max(min(abs(v-q['wavelength_um']) for q in rows['coarse']['peaks'])*1e3 for v in [q['wavelength_um'] for q in rows[key]['peaks']]))}
                      for key,p in powers.items() if key!='coarse'}
        checks[name+'_straight_calibration']=all(min(row['straight_reference_power_range'])>.97 and
                              max(row['straight_reference_power_range'])<1.03 for row in rows.values())
        checks[name+'_distinct_drop_resonances']=len(primary['peaks'])>=3
        checks[name+'_finite_height_bare_poles']=isolated[name]['checks_passed']
        checks[name+'_power_finite']=np.isfinite(powers['fine']).all().item()
        checks[name+'_finite_window_power_change_le_0p05']=primary['max_power_change_early_to_final'] <= .05
        checks[name+'_fundamental_power_sum_le_1p03']=primary['port_power_sum_range'][1] <= 1.03
        series[name]=dict(cases=rows,best_peaks=primary['peaks'],wavelength_um=WAVELENGTHS.tolist(),
              reflection=powers['fine'][:,0].tolist(),through=powers['fine'][:,1].tolist(),
              drop=powers['fine'][:,2].tolist(),add_side=powers['fine'][:,3].tolist(),
              valid_bins=primary['valid_bins'],
              port_coordinates_um=[[-case['radius']-.85,-case['radius']-case['width']-case['gap']],
                                   [case['radius']+.85,-case['radius']-case['width']-case['gap']],
                                   [-case['radius']-.85,case['radius']+case['width']+case['gap']],
                                   [case['radius']+.85,case['radius']+case['width']+case['gap']]],
              vertical_effective_model={'constant_core_index':slab_mode(SILICON,case['height'],1.55,False),
                      'reference_wavelength_um':1.55})
    np.savez_compressed(out/'arrays.npz',**arrays)
    report=dict(schema_version=1,case_id='ring',solver='Meep CPU',versions={'meep':mp.__version__,'numpy':np.__version__},
      dimensionality='Planar bus-loaded Hz spectra with a symmetric vertical slab index fixed at 1.55 µm. Separate finite-height cylindrical 3D Maxwell bare-ring poles and straight-strip MPB group delay.',
      physical_model=CASES,isolated_finite_height_ring=isolated,series=series,sensitivity=sensitivity,comparisons=comparisons,checks=checks,passed=all(checks.values()),
      vendor_reference_provenance={'lumerical_data_url':ANSYS_DATA,'lumerical_file_sha256':'497d0ed797b82d5887bd67ee1630712fb5ac38de0eee6a7b99d55e63fc938bd8',
        'tidy3d_notebook_url':TIDY_NOTEBOOK,'tidy3d_file_sha256':'b1dac0b3a47518cc5461c5457511748f5435eaea53703d9fdff1236adbd24908',
        'tidy3d_readout':'Published 574×432 notebook spectrum; x=91 and 542 pixels represent 1.50 and 1.60 µm; through dips colocated with drop peaks at x=147,223,303,383,465. About ±0.00025 µm each.'},
      arrays_file='arrays.npz',arrays_sha256=hashlib.sha256((out/'arrays.npz').read_bytes()).hexdigest(),
      limitations=['One driven input column, all four output channels. Spectra and missing port power use the two-dimensional model with symmetric cladding.',
                   'Peak positions, half-height linewidths and settling deltas are mesh/window dependent. Compare sensitivities before using peak power or Q.',
                   'Finite-height radiation, anisotropic material tensors and vendor material-library dispersion require separate qualification.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--isolated-only',action='store_true',help='run only the finite-height cylindrical bare-ring pole studies')
    args=parser.parse_args()
    destination=args.output.resolve()
    if destination.exists():
        parser.error('choose a fresh result directory')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        stage=Path(tmp)/'study';stage.mkdir()
        result=run(stage,args.isolated_only);stage.rename(destination)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
