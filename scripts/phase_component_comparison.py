"""Register native complex port measurements and compare with pinned public spectra."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from scripts.compare_loaded_ring3d import ANSYS_SHA
from scripts.lumerical_grating3d import lp01, reflected_fiber_amplitude
from scripts.grating_angular_spectrum import angular_components
from scripts.phase_stability_check import growth_diagnostic, inspect as inspect_stability

CHANNELS = {'ring': ('reflection','through','drop','add_side'), 'grating': ('reflection','guide')}
PORTS = {'ring': (1,3,2,4), 'grating': (1,2)}
GRATING_SHA = '8d3c6ffc490831ddc616433a3b26a5a6fee4088840e0a81209f893a0dbc90233'
S_FLOOR = .02


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_data(path, allow_partial_ring=False):
    report = json.loads(path.read_text())
    archive = path.parent / report.get('arrays_file', 'arrays.npz')
    partial = (allow_partial_ring and report.get('case_id')=='loaded-ring-3d' and report.get('measurement_complete') and
               report.get('checkpoint_identity',{}).get('persistent_reuse_eligible') and
               set(report.get('stages',{}))=={'reference','device'} and
               all(stage.get('device')=='cuda:0' for stage in report['stages'].values()))
    if (not report['passed'] and not partial) or report['actual_device']!='cuda:0' or file_sha(archive)!=report['arrays_sha256']:
        raise ValueError('Require a source-bound native NVIDIA measurement with its original numeric archive.')
    data=np.load(archive,allow_pickle=False)
    if report.get('case_id')=='loaded-ring-3d' and growth_diagnostic(report,data)['reject_scattering']:
        data.close()
        raise ValueError('The passive ring has runaway late fields and excess collected power. Withhold all scattering, including nominally qualified bins.')
    if not report['passed']:
        mask=data['valid_bins']
        if (mask.dtype!=np.bool_ or mask.shape!=(report['total_bins'],) or not 0<mask.sum()<len(mask)
                or mask.sum()!=report['valid_bins']):
            data.close()
            raise ValueError('Partially qualified ring must include an auditable wavelength mask.')
    return report,data


def optical_sfile(path, component, wavelengths):
    """Reconstruct complex S from the known 3-column, no-group-delay files.

    Interpolate in the native frequency coordinate. Read the supplied phase
    column before interpolating. No global reference-phase fit is permitted.
    """
    expected = ANSYS_SHA if component=='ring' else GRATING_SHA
    if file_sha(path)!=expected:
        raise ValueError('Unexpected vendor S-parameter file.')
    lines = path.read_text().splitlines()
    query = 299792458e6/np.asarray(wavelengths)
    channels, valid = [], []
    for port in PORTS[component]:
        header = f'("port{port}","TE",1,"port1",1,"transmission")'
        row = next(i for i,line in enumerate(lines) if line.replace(' ','')==header)
        count = int(re.fullmatch(r'\((\d+),\s*3\)', lines[row+1]).group(1))
        f, magnitude, phase = np.loadtxt(lines[row+2:row+2+count]).T
        if not (np.all(np.diff(f)>0) and np.all(magnitude>=0) and np.isfinite(phase).all()):
            raise ValueError('Invalid filed S samples.')
        phase = np.unwrap(phase)
        fitted = np.interp(query,f,magnitude)*np.exp(1j*np.interp(query,f,phase))
        lo = (np.searchsorted(f,query)-1).clip(0,count-2)
        inside = (query>=f[0]) & (query<=f[-1])
        # Exclude intervals with weak endpoints or large native phase increments.
        stable = inside & (magnitude[lo]>=S_FLOOR) & (magnitude[lo+1]>=S_FLOOR) & (abs(phase[lo+1]-phase[lo])<np.pi/2)
        channels.append(fitted)
        valid.append(stable)
    return np.stack(channels,axis=1),np.stack(valid,axis=1),inside


def measured_beta(through, wavelength, neff, length):
    """Recover propagation from a measured straight guide, using its mode index to resolve 2π."""
    predicted = 2*np.pi*neff/wavelength*length
    raw = np.angle(through)
    turns = np.rint((predicted-raw)/(2*np.pi))
    phase = raw + 2*np.pi*turns
    # A near-half-turn ambiguity invalidates plane transport, rather than
    # choosing the branch closest to the vendor data.
    if np.max(abs(phase-predicted)) > .9*np.pi or np.max(abs(np.diff(phase))) > np.pi:
        raise ValueError('Ambiguous propagation constant in the independent straight reference.')
    return phase/length


def fixed_mode_ring_spectrum(times, projections, basis_wavelengths, central_wavelength, wavelengths, fraction=1.):
    """Use one physical central port profile at every optical frequency, as in the filed FSP."""
    hi = np.searchsorted(basis_wavelengths,central_wavelength)
    if not 0<hi<len(basis_wavelengths):
        raise ValueError('The central profile lies outside archived mode projections.')
    lo=hi-1
    mix=(central_wavelength-basis_wavelengths[lo])/(basis_wavelengths[hi]-basis_wavelengths[lo])
    end=int(len(times)*fraction)
    if end<2:
        raise ValueError('The time window is too short.')
    trace=(1-mix)*projections[:end,:,lo,:]+mix*projections[:end,:,hi,:]
    kernel=np.exp(2j*np.pi*times[None,:end]/wavelengths[:,None])
    return ((kernel@trace.reshape(end,-1))*(times[1]-times[0])).reshape(len(wavelengths),*trace.shape[1:])


def fixed_ring_acquisition(archive, wl, wavelength):
    basis=archive['basis_wavelength_um']
    tt,rr=archive['reference_time_um_c'],archive['reference_projections']
    td,dd=archive['device_time_um_c'],archive['device_projections']
    ref=fixed_mode_ring_spectrum(tt,rr,basis,wl,wavelength)
    dev=fixed_mode_ring_spectrum(td,dd,basis,wl,wavelength)
    early=fixed_mode_ring_spectrum(td,dd,basis,wl,wavelength,.8)
    incident=ref[:,0,0]
    def normalize(fields):
        return np.column_stack([fields[:,0,1]-ref[:,0,1],fields[:,1,0],fields[:,2,1],fields[:,3,0]])/incident[:,None]
    s,es=normalize(dev),normalize(early)
    through=ref[:,1,0]/incident
    ratio=dev[:,0,0]/incident
    valid=(abs(incident)>.03*max(abs(incident)))&(abs(abs(through)**2-1)<.04)&((abs(s)**2).sum(axis=1)<=1.05)
    valid&=(np.max(abs(abs(s)**2-abs(es)**2),axis=1)<.05)&(abs(abs(ratio)-1)<.05)&(abs(np.angle(ratio))<.06)
    stable=(abs(s)>=S_FLOOR)&(abs(es)>=S_FLOOR)&(abs(np.angle(s*es.conj()))<=.05)
    return s,es,through,valid,stable


def ring_measurement(path, project=None, allow_partial=False):
    from scripts.calibrate_loaded_ring3d import directional_spectrum
    r,a = native_data(path,allow_partial)
    with a:
        w, s, early = a['wavelength_um'],a['s'],a['s_early']
        model = r['physical_model']
        p = (np.asarray(model['ports'])[:,0]-model['shape'][0]/2)/model['resolution']
        if 'physical_port_x_um' in a:
            np.testing.assert_allclose(p,a['physical_port_x_um'],atol=1e-12)
        ref = {k:a['reference_'+k] for k in ('time_um_c','projections')}
        ref['projection_wavelengths_um'] = a['basis_wavelength_um']
        direction = directional_spectrum(ref,1)
        np.testing.assert_allclose(direction[:,0,0],a['incident'],rtol=1e-10)
        through = direction[:,1,0]/a['incident']
        basis = a['reference_mode_neff'][0]
        neff = np.interp(w,a['basis_wavelength_um'],basis)
        beta = measured_beta(through,w,neff,p[1]-p[0])
        # Positive extra lengths extend away from the coupler at both ends.
        # Same-side returns and opposite-side transmission both travel two leads.
        target = 4.2 if r['vendor']=='lumerical' else 5.
        lengths = 2*target - abs(p[0]) - abs(p)
        shift = np.exp(1j*beta[:,None]*lengths)
        stable = (abs(s)>=S_FLOOR) & (abs(early)>=S_FLOOR) & (abs(np.angle(s*early.conj()))<=.05)
        if project is not None:
            from scripts.phase_ring_basis import central_overlaps
            central = central_overlaps(project,model,a)
            cw=central[0]['wavelength_um']
            fs,fe,ft,fvalid,fstable=fixed_ring_acquisition(a,cw,w)
            fb=measured_beta(ft,w,neff,p[1]-p[0])
            fs *= np.exp(1j*fb[:,None]*lengths)
            fixed=(fs,fvalid,fstable)
        else:
            central,fixed=None,None
        case_key={k:model.get(k) for k in ('vendor','radius_um','width_um','height_um','edge_gap_um','bus_y_um',
                  'top_index','substrate_index','silicon_index','ring_material','ring_interface')}
        case_key['electric_update']=r['stages']['device'].get('electric_update')
        return {'central':central,'fixed':fixed,'native_passed':r['passed'],'duration':r['duration_um_c'],'case_key':case_key,
                'vendor':r['vendor'],'r':model['resolution'],'w':w,'s':s*shift,'valid':a['valid_bins'],
                'phase_ok':stable,'transient_abs':abs(s-early),'sha':r['arrays_sha256'],
                'info':{'native_port_x_um':p.tolist(),'target_abs_x_um':target,'input_plus_output_lead_um':lengths.tolist(),
                        'straight_beta_range_rad_um': [float(beta.min()),float(beta.max())],
                        'phase_convention':'Native DFT +iωt; positive propagation contributes +βL.'}}


def grating_case_key(settings):
    keys = ('geometry','period','fill','teeth','width','guide_width','height','etch','radius',
            'taper','taper_exponent','oxide_top','oxide_bottom','n_si','n_oxide','source_z','beam_x',
            'fiber_radius','fiber_cladding_radius','fiber_core_n','fiber_clad_n','fiber_center_x',
            'angle_deg','grating_material_model','physical_bounds')
    return {k:settings.get(k) for k in keys}


def grating_phase_flags(a, s):
    history = a['dev_history_observable']
    if len(history)<3 or history.shape[1]!=len(s):
        raise ValueError('Need three final guided-mode observations to test temporal phase.')
    np.testing.assert_allclose(history[-1],a['output_mode'],rtol=1e-7)
    flags = abs(s)>=S_FLOOR
    flags[:,1] &= np.all(abs(np.angle(history[-2:]*history[-3:-1].conj()))<=.05,axis=0)
    return flags


def guide_plane_sensitivity(report, indices, wavelengths, distance):
    """Estimate plane-translation sensitivity to the numerical longitudinal wavenumber.

    The retained spectrum uses its documented transverse mode propagation.
    This independent estimate interpolates the archived effective indices
    to the temporal Yee frequency; only the tiny straight port translation
    is tested, not propagation through a taper or a grating.
    """
    stage=report['stages']['device']
    dt=stage['history'][-1]['time_fs']*1e-15/stage['time_steps']
    f=1/np.asarray(wavelengths)
    order=np.argsort(f);ordered=f[order];n=np.asarray(indices).real[order]
    effective=f*np.sinc(f*299792458e6*dt)
    lo=(np.searchsorted(ordered,effective)-1).clip(0,len(f)-2)
    estimate=n[lo]+(effective-ordered[lo])*(n[lo+1]-n[lo])/(ordered[lo+1]-ordered[lo])
    dx=1/report['settings']['resolution']
    beta=2/dx*np.arcsin(np.pi*dx*effective*estimate)
    correction=(beta-2*np.pi*f*np.asarray(indices).real)*distance
    return {'guide_length_um':distance,'time_step_seconds':dt,'estimated_yee_minus_continuous_rad':correction.tolist(),
            'max_abs_rad':float(max(abs(correction)))}


def tidy_grating(r,a):
    """Reference the measured incoming Gaussian beam to one common air plane."""
    m = r['settings']; dx=1/m['resolution']
    w = a['ref_wavelength_um']; incoming=a['ref_upper']
    z = r['stages']['reference']['sample_plane_z_um']
    xx,yy = a['ref_upper_x_um']-dx/2,a['ref_upper_y_um']-dx/2
    beam_plane_z = .93
    guide_x = -m['taper']-m['buffer']/2
    x_out = r['stages']['device']['output_plane_x_um']-dx/2
    es,et,cz,tx,ty,prop,_,_ = angular_components(incoming,dx,w,upward=False)
    # Downward propagating Ey at the same physical x, y, z on every mesh;
    # reject grazing waves as in the archived air-power decomposition.
    ey_spectrum = np.where(prop,(es*tx+et*ty)*np.exp(-2j*np.pi*cz*(beam_plane_z-z)/w[:,None,None]),0)
    propagated = np.fft.ifft2(ey_spectrum,axes=(-2,-1))
    on_axis=[]
    for f in propagated:
        transverse = np.array([np.interp(0,yy,row) for row in f])
        on_axis.append(np.interp(m['beam_x'],xx,transverse))
    on_axis=np.asarray(on_axis)
    if np.any(abs(on_axis)<.02*np.max(abs(propagated),axis=(1,2))):
        raise ValueError('No stable incoming Gaussian amplitude at the common anchor.')
    chi=on_axis/abs(on_axis)
    inc=a['incident_power']
    reflected=reflected_fiber_amplitude(a['dev_upper']-incoming,incoming,inc,dx)/chi**2
    guided=a['output_mode']/np.sqrt(inc)/chi
    guided*=np.exp(2j*np.pi*a['dev_mode_neff'].real/w*(x_out-guide_x))
    s=np.stack([reflected,guided],axis=1)
    np.testing.assert_allclose(abs(guided)**2,a['coupling'],rtol=2e-6)
    return {'case_key':grating_case_key(m),'vendor':'tidy3d','r':m['resolution'],'w':w,'s':s,'valid':a['valid'],'phase_ok':grating_phase_flags(a,s),
            'sha':r['arrays_sha256'],'info':{'sample_z_um':z,'target_beam_z_um':beam_plane_z,'target_beam_x_um':m['beam_x'],
                'guide_x_um':x_out,'target_guide_x_um':guide_x,
                'guide_translation_sensitivity':guide_plane_sensitivity(r,a['dev_mode_neff'],w,x_out-guide_x),
                'phase_convention':'Downward free-space Ey reconstructed at common beam anchor from finite-aperture propagating TE/TM components below 80 degrees. Reflection observes the reciprocal Gaussian mode.'}}


def grating_measurement(path):
    r,a = native_data(path)
    with a:
        m = r['settings']; n = m['resolution']; dx=1/n
        if m['geometry']=='tidy3d':
            return tidy_grating(r,a)
        if m['geometry']!='lumerical':
            raise ValueError('Select a public vendor geometry.')
        z = r['stages']['reference']['sample_plane_z_um']
        # Detectors colocate at Ez (i,j,k+1/2); stored x,y axes label scalar cell centres.
        x_out = r['stages']['device']['output_plane_x_um'] - dx/2
        xx,yy = a['ref_upper_x_um']-dx/2,a['ref_upper_y_um']-dx/2
        theta = np.deg2rad(m['angle_deg'])
        center = m['fiber_center_x']+z*np.tan(theta)
        upper = a['ref_upper']
        # Use the measured Ey at the axis interpolated to continuous x,y for a
        # positive-Ey incoming port. A mere sqrt(power) retains pulse phase.
        ey = upper[:,1]
        on_axis = []
        for field in ey:
            y0 = np.array([np.interp(0,yy,row) for row in field])
            on_axis.append(np.interp(center,xx,y0))
        on_axis = np.asarray(on_axis)
        if np.any(abs(on_axis) < .02*np.max(abs(ey),axis=(1,2))):
            raise ValueError('The chosen incoming fiber field has no stable phase on axis.')
        chi = on_axis/abs(on_axis)
        w, power = a['ref_wavelength_um'],a['incident_power']
        outgoing = a['output_mode']/np.sqrt(power)/chi
        reflected = reflected_fiber_amplitude(a['dev_upper']-upper,upper,power,dx)
        np.testing.assert_allclose(reflected,a['reflected_fiber_mode'],rtol=2e-6,atol=1e-8)
        reflected = reflected/chi**2
        # Express the modal estimate at the two public physical planes. The
        # LP01 centerline distance is Δz/cosθ. The guide exits toward -x.
        target_z,target_x = .9901250394579352,-25.5
        length_in, length_out = (target_z-z)/np.cos(theta), x_out-target_x
        beta = np.array([2*np.pi*lp01(m['fiber_radius'],m['fiber_core_n'],m['fiber_clad_n'],wl)[2]/wl for wl in w])
        beta_out = 2*np.pi*a['dev_mode_neff'].real/w
        s = np.stack([reflected*np.exp(2j*beta*length_in),outgoing*np.exp(1j*(beta*length_in+beta_out*length_out))],axis=1)
        np.testing.assert_allclose(abs(s[:,1])**2,a['coupling'],rtol=2e-6)
        return {'case_key':grating_case_key(m),'vendor':'lumerical','r':n,'w':w,'s':s,'valid':a['valid'],'phase_ok':grating_phase_flags(a,s),
                'sha':r['arrays_sha256'],'info':{'center_x_um':center,'sample_z_um':z,'guide_x_um':x_out,
                    'target_fiber_z_um':target_z,'target_axis_x_um':float(m['fiber_center_x']+target_z*np.tan(theta)),
                    'file_port_center_x_um':2.8582013154835044,'target_guide_x_um':target_x,
                    'fiber_lead_um':float(length_in),'guide_lead_um':float(length_out),
                    'guide_translation_sensitivity':guide_plane_sensitivity(r,a['dev_mode_neff'],w,length_out),
                    'phase_convention':'Native DFT +iωt; incoming LP01 field has positive Ey on its axis. The commercial FSP marks the cached fiber basis invalid.'}}


def distribution(values):
    if values.size==0:
        return {'count':0,'rms':None,'max_abs':None}
    return {'count':values.size,'rms':float(np.sqrt(np.mean(abs(values)**2))),'max_abs':float(np.max(abs(values)))}


def sampled_mean_power(a,b,wavelengths,valid):
    # Only integrate adjacent qualified bins; never bridge a missing resonance interval.
    df=np.abs(np.diff(299792458e6/np.asarray(wavelengths)))
    selected=valid[:-1]&valid[1:]
    band=float(np.sum(df[selected]))
    def average(values):
        power=abs(values)**2
        return float(np.sum(((power[:-1]+power[1:])*.5*df)[selected])/band) if band>0 else None
    return {'covered_frequency_hz':band,'adjacent_intervals':int(selected.sum()),'mean_power_a':average(a),'mean_power_b':average(b)}


def compare(component, a, b, common, phase_ok, vendor=False, wavelengths=None):
    """Compare squared powers and finite spectral phase differences on qualified data."""
    result = {}
    for j,name in enumerate(CHANNELS[component]):
        sa,sb = a[:,j],b[:,j]
        strong = common & phase_ok[:,j] & (abs(sa)>=S_FLOOR) & (abs(sb)>=S_FLOOR)
        changes = [np.angle(s[1:]*s[:-1].conj()) for s in (sa,sb)]
        pairs = strong[:-1] & strong[1:] & (abs(changes[0])<np.pi/2) & (abs(changes[1])<np.pi/2)
        difference = np.angle(np.exp(1j*(changes[0]-changes[1])))
        row = {'power_delta':distribution((abs(sa)**2-abs(sb)**2)[common]),
               'phase_increment_delta_rad':distribution(difference[pairs]),
               'strong_samples':int(strong.sum()),'bins_below_amplitude_floor':int((common & ((abs(sa)<S_FLOOR)|(abs(sb)<S_FLOOR))).sum())}
        if not vendor:
            row['same_gauge_complex_delta'] = distribution((sa-sb)[common])
            row['same_gauge_phase_delta_rad'] = distribution(np.angle(sa*sb.conj())[strong])
        if name in ('reflection','add_side'):
            if wavelengths is not None:
                row['sampled_frequency_average']=sampled_mean_power(sa,sb,wavelengths,common)
            row['peak_power_a'] = float(max(abs(sa[common])**2)) if common.any() else None
            row['peak_power_b'] = float(max(abs(sb[common])**2)) if common.any() else None
        result[name] = row
    return result


def mode_gauge_valid(mode):
    return (abs(mode['overlap_magnitude']-1)<=.05 and abs(mode['overlap_phase_rad'])<.01
            and abs(mode['native_aperture_to_full_filed_power']-1)<=.05)


def compare_fixed_ring(row,vendor_file):
    """Compare the filed single-profile spectrum and verify the one saved mode frequency."""
    w,s=row['w'],row['s']
    modes = row['central']
    fixed_s,fixed_valid,fixed_temporal=row['fixed']
    vendor_s,vendor_stable,inside_band=optical_sfile(vendor_file,'ring',w)
    modal_ok=np.array([all(mode_gauge_valid(mode)
                           for mode in (modes[0],modes[j])) for j in range(len(modes))])
    common=row['valid']&fixed_valid&inside_band
    both=fixed_temporal&vendor_stable&modal_ok[None,:]&common[:,None]
    delta=np.angle(fixed_s*vendor_s.conj())
    diagnostics={}
    for j,name in enumerate(CHANNELS['ring']):
        strong=both[:,j]
        diagnostics[name]={'absolute_phase_residual_rad':distribution(delta[:,j][strong]),
            'native_vs_vendor_power_delta':distribution((abs(fixed_s[:,j])**2-abs(vendor_s[:,j])**2)[common]),
            'fixed_minus_frequency_specific_power_max':float(np.max(abs(abs(fixed_s[common,j])**2-abs(s[common,j])**2))),
            'fixed_minus_frequency_specific_phase_change_rad':distribution(np.angle(fixed_s[:,j]*s[:,j].conj())[strong&row['phase_ok'][:,j]])}
    band_result={'vendor_expansion':'One central TE profile at all 500 frequencies, as described by Ansys.',
         'native_method':'Time projections interpolated once at the saved central-mode wavelength; same weights then applied at every frequency. The actual vendor and local profiles have their measured overlap at that frequency.',
         'channels':diagnostics,'native_valid_bins':(row['valid']&fixed_valid).tolist(),
         'native_phase_ok':fixed_temporal.tolist(),'phase_qualified':both.tolist(),'registered_native_real':fixed_s.real.tolist(),
         'registered_native_imag':fixed_s.imag.tolist(),'signed_native_minus_vendor_phase_rad':[
             [float(v) if resolved else None for v,resolved in zip(values,flags)] for values,flags in zip(delta,both)]}
    center_wavelength = modes[0]['wavelength_um']
    if not all(mode['wavelength_um']==center_wavelength for mode in modes):
        raise ValueError('The cached port modes refer to different wavelengths.')
    v, vok, inside = optical_sfile(vendor_file,'ring',np.array([center_wavelength]))
    hi = np.searchsorted(w,center_wavelength)
    lo = hi-1
    values = {}
    for j,name in enumerate(CHANNELS['ring']):
        two = s[lo:hi+1,j]
        phase = np.unwrap(np.angle(two))
        magnitude = np.interp(center_wavelength,w[lo:hi+1],abs(two))
        native = magnitude*np.exp(1j*np.interp(center_wavelength,w[lo:hi+1],phase))
        resolved = bool(row['valid'][lo:hi+1].all() and row['phase_ok'][lo:hi+1,j].all() and
            inside[0] and vok[0,j] and abs(phase[1]-phase[0])<np.pi/2 and
            all(mode_gauge_valid(mode) for mode in (modes[0],modes[j])))
        values[name] = {'native_power':float(abs(native)**2),'vendor_power':float(abs(v[0,j])**2),
                        'phase_gates_passed':resolved,'signed_native_minus_vendor_phase_rad':
                            float(np.angle(native*v[0,j].conj())) if resolved else None}
    central_result={'native_bracket_um':w[lo:hi+1].tolist(),'port_overlaps':modes,'channels':values}
    return {'single_central_basis_band':band_result,'central_vendor_mode_check':central_result}


def compare_mesh(component,previous,row,vendor_file):
    w,s=row['w'],row['s']
    if row['case_key']!=previous['case_key'] or row['r']<=previous['r']:
        raise ValueError('A mesh comparison requires the same physical design/material model at increasing mesh resolution.')
    # Match only physically identical wavelength samples on different monitor grids.
    ia,ib=[],[]
    for i,wl in enumerate(previous['w']):
        matches=np.flatnonzero(np.isclose(w,wl,rtol=0,atol=1e-10))
        if matches.size==1:
            ia.append(i);ib.append(int(matches[0]))
    if len(ia)<2:
        raise ValueError('Insufficient common frequencies for a phase comparison.')
    shared_valid=previous['valid'][ia]&row['valid'][ib]
    if vendor_file:
        _,_,inside=optical_sfile(vendor_file,component,w[ib])
        shared_valid &= inside
    elif component=='ring' and row['vendor']=='tidy3d':
        shared_valid &= (w[ib]>=1.5)&(w[ib]<=1.6)
    return {'meshes':[previous['r'],row['r']],'shared_wavelength_um':w[ib].tolist(),'shared_valid':shared_valid.tolist(),
        'channels':compare(component,previous['s'][ia],s[ib],shared_valid,
            previous['phase_ok'][ia]&row['phase_ok'][ib],wavelengths=w[ib])}


def analyze(component, paths, vendor_file, vendor_project=None, allow_partial=False, rejected_rings=()):
    if not paths:
        raise ValueError('At least one native receipt is required.')
    if vendor_project and (component!='ring' or not vendor_file):
        raise ValueError('The filed ring modes require the matching scattering file.')
    if vendor_project:
        from scripts.phase_ring_basis import project_modes
        project = project_modes(vendor_project)
    else:
        project = None
    measurements = [ring_measurement(p,project,allow_partial) if component=='ring' else grating_measurement(p) for p in paths]
    if vendor_file and any(row['vendor']!='lumerical' for row in measurements):
        raise ValueError('The Lumerical S file applies only to that physical geometry.')
    if len({row['vendor'] for row in measurements})!=1:
        raise ValueError('Compare one physical vendor geometry per report.')
    result = {'component':component,'vendor':measurements[0]['vendor'],'amplitude_floor':S_FLOOR,
              'phase_increment_limit_rad':float(np.pi/2),'ring_80_to_100_percent_phase_limit_rad':.05,
              'native_samples':[],'mesh_pairs':[],'vendor_absolute_phase_basis':'Matched fixed central ring profile and qualified samples' if project else None,
              'vendor_sfile_sha256':file_sha(vendor_file) if vendor_file else None,
              'vendor_project_sha256':file_sha(vendor_project) if vendor_project else None}
    if component=='grating':
        result['native_phase_evidence']={
            'guide':{'minimum_amplitude':S_FLOOR,'verified_final_phase_intervals':2,'phase_change_limit_rad':.05},
            'reflection':{'minimum_amplitude':S_FLOOR,'verified_final_phase_intervals':0,
                          'scope':'Final source-subtracted E/H projection; an independent return-phase duration history is unavailable.'}}
    if rejected_rings:
        if component!='ring':
            raise ValueError('A rejected ring can only accompany a ring study.')
        result['rejected_acquisitions']=[]
        for path in rejected_rings:
            bad=inspect_stability(path)
            if not bad['diagnostic']['reject_scattering']:
                raise ValueError('Only verified unstable acquisitions can be excluded with this flag.')
            result['rejected_acquisitions'].append({'native_receipt':bad['receipt'],'native_arrays_sha256':bad['native_arrays_sha256'],
                'original_valid_wavelength_um':bad['nominal_qualified_wavelength_um'],'phase_reportable_bins':0,
                'diagnostic':bad['diagnostic'],'reason':bad['interpretation']})
    previous = None
    for source,row in zip(paths,measurements):
        w,s = row['w'],row['s']
        record = {'native_receipt':source.name,'native_run_passed':bool(row.get('native_passed',True)),'qualified_wavelengths':int(row['valid'].sum()),
                  'physical_model':row['case_key'],'total_wavelengths':len(w),'resolution':row['r'],'arrays_sha256':row['sha'],'planes':row['info'],'wavelength_um':w.tolist(),
                  'registered_complex_s_real':s.real.tolist(),'registered_complex_s_imag':s.imag.tolist(),
                  'valid_bins':row['valid'].tolist(),'phase_ok':row['phase_ok'].tolist()}
        if 'duration' in row:
            record['native_duration_um_c']=row['duration']
        if 'transient_abs' in row:
            record['early_to_final_complex_change_max_by_port']=row['transient_abs'].max(axis=0).tolist()
        if vendor_file:
            vendor, phase_ok, inside = optical_sfile(vendor_file,component,w)
            record['commercial_comparison'] = compare(component,s,vendor,row['valid']&inside,row['phase_ok']&phase_ok,True,w)
        if row['vendor']=='tidy3d':
            record['commercial_phase_data']='Pinned notebook exposes a power plot; it does not include the complex monitor data in its saved outputs.'
        if row.get('central') is not None:
            record.update(compare_fixed_ring(row,vendor_file))
        if previous:
            result['mesh_pairs'].append(compare_mesh(component,previous,row,vendor_file))
        result['native_samples'].append(record)
        previous=row
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component',choices=('ring','grating'),required=True)
    p.add_argument('--vendor-sfile',type=Path)
    p.add_argument('--vendor-ring-project',type=Path)
    p.add_argument('--include-partial-rings',action='store_true',help='Use qualified bins of source-bound, complete ring acquisitions; preserve the original run status.')
    p.add_argument('--record-rejected-ring',action='append',default=[],type=Path,help='Keep a verified runaway acquisition visible without publishing S from it.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('receipts',type=Path,nargs='+')
    args=p.parse_args()
    args.output.write_text(json.dumps(analyze(args.component,args.receipts,args.vendor_sfile,args.vendor_ring_project,args.include_partial_rings,args.record_rejected_ring),indent=2,allow_nan=False)+'\n')
