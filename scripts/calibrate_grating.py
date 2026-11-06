"""Measure upward radiation and Gaussian-to-guide coupling in 2D Meep (lengths in µm)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

TIDY_URL = 'https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/'
ANSYS_URL = 'https://optics.ansys.com/hc/en-us/articles/360042304774-Broadband-grating-coupler-2D'
ANSYS_PROJECT = 'https://optics.ansys.com/hc/en-us/article_attachments/360057295433'
ANSYS_SHA = '849ce6962c5bf6ac6afeca8e02c9a2fea21bc6d5948d1cfbc206bb1dc171c9e5'
ANSYS_SPECTRUM = 'https://optics.ansys.com/hc/article_attachments/360057322053/broadband_coupler_wavelength_results.png'
ANSYS_SPECTRUM_SHA = '3d7cb298706477113ea9a586b3d8dd6b2ffb5ee0adbdd6a2c1269a72b81d9036'

CASES = {
    'tidy3d': dict(wavelength=1.55, period=.620, fill=.80, teeth=20,
                   height=.260, etch=.160, oxide_top=.680, oxide_bottom=2.0,
                   core_n=3.47, oxide_n=1.44, beam_x=3.6, angle_deg=14.5,
                   waist_radius=5.4, beam_span=21.6, tooth_at_start=True,
                   right_guide_is_full=False, published=.5623, url=TIDY_URL),
    'lumerical': dict(wavelength=1.310, period=.500, fill=.50, teeth=100,
                      height=.200, etch=.030, oxide_top=.900, oxide_bottom=.700,
                      core_n=3.4993, oxide_n=1.447, beam_x=13.0, angle_deg=13.9,
                      waist_radius=17.3, beam_span=67.5, tooth_at_start=False,
                      right_guide_is_full=True, published=.557, url=ANSYS_URL),
}

ANSYS_FILED = dict(CASES['lumerical'], oxide_n=1.44, beam_x=14.8, beam_z=3.3,
                   waist_radius=17.288, focus_distance=3.2, exit_trench=.250)
ANSYS_OUTPUT = dict(ANSYS_FILED, output_x=-40., output_aperture=1.,
                    beam_left=-42., beam_right=62., beam_span=104.,
                    post_source_duration=300)


def phase_matching_estimate(case):
    """Symmetric TE slab estimate of the first-order grating period in air."""
    def slab(height):
        k0=2*math.pi/case['wavelength']
        half=height/2
        v=k0*half*math.sqrt(case['core_n']**2-case['oxide_n']**2)
        lo,hi=0.,min(v,math.pi/2)
        for _ in range(60):
            u=(lo+hi)/2
            if u*math.tan(u)<math.sqrt(max(0.,v*v-u*u)):
                lo=u
            else:
                hi=u
        return math.sqrt(case['core_n']**2-(((lo+hi)/2)/(half*k0))**2)
    unetched,etched=slab(case['height']),slab(case['height']-case['etch'])
    index=case['fill']*unetched+(1-case['fill'])*etched
    projected_air_index=math.sin(math.radians(case['angle_deg']))
    return dict(unetched_neff=unetched,etched_neff=etched,weighted_neff=index,
                estimated_period_nm=1000*case['wavelength']/(index-projected_air_index),
                phase_match_required_index=case['wavelength']/case['period']+projected_air_index)


def sampled_peak(wavelengths, coupling):
    """Peak sample and its neighbor spacing, without a fitted spectral shift."""
    index=int(np.argmax(coupling))
    left=max(0,index-1)
    right=min(len(wavelengths)-1,index+1)
    spacing=max(abs(wavelengths[index]-wavelengths[left]),
                abs(wavelengths[right]-wavelengths[index]))
    return dict(wavelength_um=float(wavelengths[index]), beam_to_guide=float(coupling[index]),
                largest_adjacent_bin_width_nm=float(1000*spacing), at_band_edge=index in (0,len(coupling)-1))


def valid_beam_bins(incident, coupling):
    return (incident>=.01*np.max(incident)) & np.isfinite(coupling) & (coupling>=0) & (coupling<=1.02)


def beam_band_passes(incident, coupling):
    inside=incident>=.01*np.max(incident)
    valid=valid_beam_bins(incident,coupling)
    return bool(valid[len(valid)//2] and np.all(valid[inside]) and np.mean(valid)>=.8)


def compare_ansys_spectrum(wavelengths, coupling):
    """Independent readout from the labeled broadband-direct curve in Figure 6."""
    reference_nm=np.array([1298.,1303.,1308.,1310.,1313.,1318.,1323.])
    reference_percent=np.array([10.45,35.61,54.06,55.61,51.51,29.22,6.83])
    nm=1000*np.asarray(wavelengths)
    order=np.argsort(nm)
    if reference_nm[0]<nm[order[0]] or reference_nm[-1]>nm[order[-1]]:
        raise ValueError('The measured grating band must cover the published comparison wavelengths.')
    measured=100*np.interp(reference_nm,nm[order],np.asarray(coupling)[order])
    delta=measured-reference_percent
    return dict(wavelength_nm=reference_nm.tolist(), published_broadband_direct_percent=reference_percent.tolist(),
                measured_percent=measured.tolist(), difference_percentage_points=delta.tolist(),
                mean_absolute_difference_percentage_points=float(np.mean(abs(delta))),
                reference_readout_uncertainty_percentage_points=.4, reference_fwhm_nm=16.66,
                reference_figure_url=ANSYS_SPECTRUM, reference_figure_sha256=ANSYS_SPECTRUM_SHA,
                scope='Local spectrum linearly interpolated to seven published plot coordinates; single-frequency Meep Gaussian profile compared with vendor broadband direct curve. Nominal 55.7% is reported separately by the vendor.')


def measure(case, stage, resolution, pml, margin, end, aperture, compact=False, direct_flux=None):
    import meep as mp

    period, teeth = case['period'], case['teeth']
    length = teeth * period
    top, bottom = case['oxide_top'], case['oxide_bottom']
    height, etch = case['height'], case['etch']
    beam_z = case.get('beam_z', top + .5)
    beam = stage.startswith('beam')
    beam_left=case.get('beam_left',case['beam_x']-case['beam_span']/2)
    beam_right=case.get('beam_right',case['beam_x']+case['beam_span']/2)
    xmin = min(-6., beam_left) - pml - margin
    xmax = max(length + 2., beam_right) + pml + margin
    zmin, zmax = -bottom - .65 - pml - margin, beam_z + pml + margin
    if compact:
        if not beam:
            raise ValueError('The compact source-domain comparison applies to the Gaussian measurement.')
        buffer=.6*case['wavelength']
        xmin,xmax=-buffer,length+buffer
        zmin,zmax=-bottom-buffer,height+top+buffer
    source_start=max(xmin+pml,beam_left)
    source_end=min(xmax-pml,beam_right)
    source_midpoint=(source_start+source_end)/2
    source_span=source_end-source_start
    # Meep's y is physical vertical z; all structures are invariant out of plane.
    def point(x, z):
        return mp.Vector3(x - (xmin + xmax)/2, z - (zmin + zmax)/2)

    def block(left, right, low, high, material):
        return mp.Block(size=mp.Vector3(right-left, high-low, mp.inf),
                        center=point((left+right)/2, (low+high)/2), material=material)

    core, oxide = mp.Medium(index=case['core_n']), mp.Medium(index=case['oxide_n'])
    geometry = []
    if stage != 'beam_reference':
        geometry = [block(xmin-1, xmax+1, zmin-1, -bottom, core),
                    block(xmin-1, xmax+1, -bottom, top, oxide)]
    if stage == 'guide_reference':
        geometry.append(block(xmin-1, xmax+1, 0., height, core))
    if stage in ('beam_device', 'guide_device'):
        geometry.extend([block(xmin-1, 0, 0, height, core),
                         block(0, xmax+1, 0, height-etch, core)])
        if case['right_guide_is_full']:
            geometry.append(block(length+case.get('exit_trench',0.), xmax+1, 0, height, core))
        for i in range(teeth):
            x = period * i + (0 if case['tooth_at_start'] else period*(1-case['fill']))
            geometry.append(block(x, x+period*case['fill'], height-etch, height, core))

    frequency = 1 / case['wavelength']
    bandwidth = .08 if case['wavelength'] > 1.5 else .045
    bins = 21
    pulse = mp.GaussianSource(frequency, fwidth=1.4*bandwidth, cutoff=4)
    if beam:
        angle = math.radians(case['angle_deg'])
        focus_distance = case.get('focus_distance', 0.)
        focus = mp.Vector3(case['beam_x']-source_midpoint-focus_distance*math.sin(angle),
                           -focus_distance*math.cos(angle))
        source = mp.GaussianBeam2DSource(src=pulse, center=point(source_midpoint, beam_z),
                 size=mp.Vector3(source_span, 0), beam_E0=mp.Vector3(z=1),
                 beam_w0=case['waist_radius'], beam_x0=focus,
                 beam_kdir=mp.Vector3(-math.sin(angle), -math.cos(angle)))
    else:
        source = mp.EigenModeSource(src=pulse, center=point(-4., height/2),
                                    size=mp.Vector3(0, 1.56), eig_band=1, eig_parity=mp.ODD_Z)
    sim = mp.Simulation(cell_size=(xmax-xmin, zmax-zmin), geometry=geometry,
                        resolution=resolution, sources=[source], boundary_layers=[mp.PML(pml)])

    def flux(x, z, dx, dz):
        return sim.add_flux(frequency, bandwidth, bins, mp.FluxRegion(center=point(x,z), size=(dx,dz)))

    if stage == 'beam_reference':
        monitors = [flux(source_midpoint, top+.23, source_span, 0)]
    elif stage == 'beam_device':
        output_x=case.get('output_x',-.3*case['wavelength'] if compact else -1.)
        output_aperture=case.get('output_aperture',aperture)
        mode = sim.add_mode_monitor(frequency, bandwidth, bins,
                       mp.ModeRegion(center=point(output_x,height/2), size=(0,output_aperture)))
        if direct_flux is not None:
            total_output=flux(output_x,height/2,0,output_aperture)
    else:
        left, right, low, high = -1.5, length + 1.35, -bottom - .45, top + .25
        monitors = [flux(left, (low+high)/2, 0, high-low), flux(right, (low+high)/2, 0, high-low),
                    flux((left+right)/2, high, right-left, 0), flux((left+right)/2, low, right-left,0)]
    sim.run(until_after_sources=end)
    if stage == 'beam_device':
        result = sim.get_eigenmode_coefficients(mode, [1], eig_parity=mp.ODD_Z).alpha[0].copy()
        freqs = np.array(mp.get_flux_freqs(mode))
        if direct_flux is not None:
            direct_flux.extend(mp.get_fluxes(total_output))
    else:
        result = np.array([mp.get_fluxes(monitor) for monitor in monitors])
        freqs = np.array(mp.get_flux_freqs(monitors[0]))
    sim.reset_meep()
    return freqs, result


def measured_stage(case, stage, settings, compact=False, direct=False, checkpoint=None):
    def acquire():
        output=[] if direct else None
        frequencies,values=measure(case,stage,*settings,compact,output)
        samples={'frequencies':frequencies,'values':values}
        if direct:
            samples['direct_flux']=np.array(output)
        return {},samples
    if checkpoint is None:
        return acquire()[1]
    spec={'physical_model':case,'settings':settings,'compact':compact,'direct_flux':direct}
    return checkpoint.acquire(stage,spec,acquire)[1]


def spectrum(case, label, settings, arrays, checkpoint=None):
    resolution, pml, margin, end, aperture = settings
    data = {}
    for stage in ('beam_reference','beam_device','guide_reference','guide_device'):
        captured=measured_stage(case,stage,settings,checkpoint=checkpoint)
        freq,values=captured['frequencies'],captured['values']
        arrays[f'{label}_{stage}'] = values
        data[stage] = values
    arrays[f'{label}_frequency_per_um'] = freq
    incident_beam = -data['beam_reference'][0]
    incident_guide = data['guide_reference'][0]
    if min(np.min(incident_beam), np.min(incident_guide)) <= 0:
        raise RuntimeError('A reference source had nonpositive incident flux.')
    coupled = abs(data['beam_device'][:, 1])**2 / incident_beam
    left, right, up, down = data['guide_device'] / incident_guide
    reflection, downward = 1-left, -down
    balance = reflection + right + up + downward
    zero = len(freq)//2
    result = dict(label=label, resolution_cells_per_um=resolution, pml_um=pml,
                  padding_um=margin, post_source_duration_um_c=end, port_aperture_um=aperture,
                  wavelength_um=(1/freq).tolist(), beam_to_guide=coupled.tolist(),
                  sampled_peak=sampled_peak(1/freq,coupled),
                  guide_radiation_up=up.tolist(), guide_radiation_down=downward.tolist(),
                  outlet_right=right.tolist(), outlet_left=reflection.tolist(),
                  power_sum=balance.tolist(),
                  center=dict(wavelength_um=float(1/freq[zero]), beam_to_guide=float(coupled[zero]),
                              upper_flux_fraction=float(up[zero]), lower_flux_fraction=float(downward[zero]),
                              outlet_left_fraction=float(reflection[zero]), outlet_right_fraction=float(right[zero]),
                              power_sum=float(balance[zero])),
                  valid_energy_bins=(abs(balance-1) < .04).tolist(),
                  valid_beam_bins=valid_beam_bins(incident_beam,coupled).tolist(),
                  qualified_beam_band=beam_band_passes(incident_beam,coupled),
                  max_abs_balance_error=float(np.max(abs(balance-1))))
    if case['url']==ANSYS_URL:
        result['published_spectral_comparison']=compare_ansys_spectrum(1/freq,coupled)
    print(label, result['center'], flush=True)
    return result


def gaussian_only(case, resolution, compact, arrays, prefix, checkpoint=None):
    """Matched empty/reference and device runs on the same explicit source aperture."""
    pml=.4 if compact else .8
    duration=case.get('post_source_duration',100 if case['wavelength']>1.5 else 150)
    settings=(resolution,pml,1.,duration,1.56)
    reference=measured_stage(case,'beam_reference',settings,compact,checkpoint=checkpoint)
    measured=measured_stage(case,'beam_device',settings,compact,'output_x' in case,checkpoint)
    f,ref,device=reference['frequencies'],reference['values'],measured['values']
    direct=measured.get('direct_flux')
    incident=-ref[0]
    if np.min(incident)<=0: raise RuntimeError('Nonpositive measured Gaussian input.')
    coupled=abs(device[:,1])**2/incident
    arrays[prefix+'_reference']=ref
    arrays[prefix+'_device_modes']=device
    result=dict(resolution=resolution,pml_um=pml,physical_cell='vendor box with finite source clipping' if compact else 'extended aperture',
                sampled_peak=sampled_peak(1/f,coupled),
                valid_beam_bins=valid_beam_bins(incident,coupled).tolist(),
                qualified_beam_band=beam_band_passes(incident,coupled),
                wavelength_um=(1/f).tolist(),beam_to_guide=coupled.tolist(),center_beam_to_guide=float(coupled[len(coupled)//2]))
    if case['url']==ANSYS_URL:
        result['published_spectral_comparison']=compare_ansys_spectrum(1/f,coupled)
    if direct is not None:
        measured=-np.array(direct)/incident
        arrays[prefix+'_output_flux']=np.array(direct)
        result.update(output_x_um=case['output_x'],output_aperture_um=case['output_aperture'],
                      post_source_duration_um_c=duration,direct_output_fraction=measured.tolist(),
                      center_direct_output_fraction=float(measured[len(measured)//2]),
                      valid_direct_bins=valid_beam_bins(incident,measured).tolist(),
                      qualified_direct_band=beam_band_passes(incident,measured),
                      direct_output_spectral_comparison=compare_ansys_spectrum(1/f,measured))
    return result


def guided_radiation(out,vendor,resolution,checkpoint):
    """One launched TE0 mode and an independent straight-guide normalization."""
    import meep as mp
    mp.verbosity(0)
    case=CASES[vendor]
    duration=100 if vendor=='tidy3d' else 150
    settings=(resolution,.8,1.,duration,1.56)
    reference=measured_stage(case,'guide_reference',settings,checkpoint=checkpoint)
    device=measured_stage(case,'guide_device',settings,checkpoint=checkpoint)
    incident=reference['values'][0]
    if np.min(incident)<=0:
        raise RuntimeError('The straight reference has nonpositive incident power.')
    left,right,up,down=device['values']/incident
    reflected=1-left
    total=reflected+right+up-down
    valid=(abs(total-1)<.04) & (incident>=.01*max(incident))
    center=len(incident)//2
    archive=out/'arrays.npz'
    np.savez_compressed(archive,frequency_per_um=device['frequencies'],straight_flux=reference['values'],
                        device_flux=device['values'])
    row=dict(wavelength_um=(1/device['frequencies']).tolist(),guide_radiation_up=up.tolist(),
             guide_radiation_down=(-down).tolist(),outlet_left=reflected.tolist(),outlet_right=right.tolist(),
             power_sum=total.tolist(),valid_energy_bins=valid.tolist(),center_upper_fraction=float(up[center]),
             center_lower_fraction=float(-down[center]),center_power_sum=float(total[center]))
    report=dict(schema_version=1,case_id='grating-guided-radiation',solver='Meep CPU',version=mp.__version__,
        vendor=vendor,physical_model=case,resolution=resolution,pml_um=.8,padding_um=1.,post_source_duration_um_c=duration,
        passed=bool(valid[center] and np.isfinite(total).all()),studies={vendor:{'guided':row}},
        scope='Power leaving an enclosing 2D cross-sectional contour for a guide launch. Upward flux includes all angles; the vendor percentages are for excitation by a specific Gaussian instead.',
        arrays_file=archive.name,arrays_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def run(out, checkpoint=None):
    import meep as mp
    mp.verbosity(0)
    arrays, studies = {}, {}
    # Separate one-axis sensitivities for the Tidy3D geometry. A second mesh tests
    # the shallow etch of the larger Ansys grating independently.
    sweeps = {
        'tidy3d': {'coarse':(28,.8,1.,100,1.56), 'fine':(40,.8,1.,100,1.56),
                   'long':(40,.8,1.,160,1.56), 'pml':(40,1.25,1.,100,1.56),
                   'padding':(40,.8,1.6,100,1.56), 'aperture':(40,.8,1.,100,1.95)},
        'lumerical': {'coarse':(28,.8,1.,150,1.56), 'fine':(40,.8,1.,150,1.56)},
    }
    for name, runs in sweeps.items():
        studies[name] = {key:spectrum(CASES[name],name+'_'+key,settings,arrays,checkpoint)
                         for key,settings in runs.items()}
    comparisons = []
    checks = {}
    for name, results in studies.items():
        central = results['fine']['center']
        value = central['beam_to_guide']
        reference = CASES[name]['published']
        comparisons.append(dict(vendor='Tidy3D' if name=='tidy3d' else 'Ansys Lumerical',
            case=name, observable='Incident Gaussian power coupled into left TE0 guide at nominal wavelength',
            wavelength_um=central['wavelength_um'], reference_fraction=reference, measured_fraction=value,
            difference_percentage_points=100*(value-reference), relative_difference=(value-reference)/reference,
            source_url=CASES[name]['url'], scope=(
            'Published best 2D period/beam position, layer boundaries and indices matched. Beam normalization and domain placement differ.'
            if name=='tidy3d' else
            'Article-based 100-tooth reconstruction at the published optimum angle. Oxide index, vectorial thin-lens focus, beam position and actual waist differ. The separate project-source control uses the index, waist and position from the downloadable FSP.')))
        checks[name+'_finite_reference_and_power'] = all(
            np.isfinite(row['beam_to_guide']).all() and .01 < row['center']['beam_to_guide'] < 1
            and row['center']['upper_flux_fraction'] > .01 for row in results.values())
        checks[name+'_center_energy_balance_within_0_04'] = abs(central['power_sum']-1) < .04
        checks[name+'_beam_spectrum_within_source_band_and_passivity'] = all(row['qualified_beam_band'] for row in results.values())
    sensitivity = {name:{key:dict(
        max_beam_efficiency_change=float(np.max(np.abs(np.subtract(row['beam_to_guide'],results['fine']['beam_to_guide'])))),
        max_upper_fraction_change=float(np.max(np.abs(np.subtract(row['guide_radiation_up'],results['fine']['guide_radiation_up'])))))
        for key,row in results.items() if key!='fine'} for name,results in studies.items()}
    compact=gaussian_only(CASES['tidy3d'],40,True,arrays,'tidy_compact',checkpoint)
    np.savez_compressed(out/'arrays.npz', **arrays)
    report = dict(schema_version=1,case_id='grating',solver='Meep CPU',versions={'meep':mp.__version__, 'numpy':np.__version__},
        dimensionality='2D x-z cross section, invariant lateral width; TE electric field perpendicular to the cross section',
        physical_model=CASES,compact_tidy_domain=compact,phase_matching_estimates={key:phase_matching_estimate(case) for key,case in CASES.items()}, comparisons=comparisons, sensitivity=sensitivity, studies=studies, checks=checks,
        passed=all(checks.values()), arrays_file='arrays.npz',
        arrays_sha256=hashlib.sha256((out/'arrays.npz').read_bytes()).hexdigest(),
        ansys_project_url=ANSYS_PROJECT, ansys_project_sha256=ANSYS_SHA,
        limitations=['Net upward flux across the enclosing contour and coupling to a particular Gaussian are separate quantities.',
                     'Finite lateral-width diffraction, a 3D taper, all-angle far fields, and fibers with a specified guided mode require separate models.',
                     'The left/right contour fluxes include lateral radiation. Frequency-specific power imbalance and physical sensitivity are reported unmodified.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def beam_receipt(vendor, result, arrays_path, case=None):
    if case is None:
        case=CASES[vendor]
    measured=result['center_beam_to_guide']
    return dict(schema_version=1,case_id='grating-gaussian-profile',vendor=vendor,solver='Meep CPU',
      physical_model=case,result=result,reference_coupling=case['published'],
      delta_percentage_points=100*(measured-case['published']),source_url=case['url'],
      passed=bool(result['qualified_beam_band'] and 0<measured<1),
      arrays_file='arrays.npz',arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
      scope='Actual downward Gaussian power in air and a separate excited device measured on the same source aperture; outgoing TE0 power referenced to that source.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint',type=Path,help='retain and reuse verified, completed solver stages in this directory')
    parser.add_argument('--beam-only',choices=tuple(CASES),help='repeat only one measured Gaussian-to-guide case')
    parser.add_argument('--guide-only',choices=tuple(CASES),help='measure upward and downward flux for one guide-excited grating')
    parser.add_argument('--vendor-box',action='store_true',help='use Tidy3D notebook box and monitor with an explicitly clipped beam')
    parser.add_argument('--project-source',action='store_true',help='use the published Ansys project oxide index, beam position and waist')
    parser.add_argument('--project-output',action='store_true',help='measure Ansys total output flux and TE0 at the filed far-port coordinates')
    parser.add_argument('--resolution',type=int,choices=(28,40,60),help='mesh for an individual source study (default 40)')
    args = parser.parse_args()
    if args.beam_only and args.guide_only:
        parser.error('choose a Gaussian or a guided input')
    if args.vendor_box and args.beam_only!='tidy3d':
        parser.error('--vendor-box requires --beam-only tidy3d')
    if (args.project_source or args.project_output) and args.beam_only!='lumerical':
        parser.error('--project-source and --project-output apply to --beam-only lumerical')
    if args.resolution is not None and not (args.beam_only or args.guide_only):
        parser.error('--resolution applies to individual source studies')
    destination = args.output.resolve()
    if destination.exists():
        parser.error('choose a new output directory')
    checkpoint=None
    if args.checkpoint:
        checkpoint_path=args.checkpoint.resolve()
        if checkpoint_path==destination or checkpoint_path in destination.parents or destination in checkpoint_path.parents:
            parser.error('use separate output and checkpoint directories without nesting them')
        if __package__:
            from .calibration_checkpoint import MeasurementCheckpoints
        else:
            from calibration_checkpoint import MeasurementCheckpoints
        checkpoint=MeasurementCheckpoints(args.checkpoint,__file__)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp)/'study'; stage.mkdir()
        if args.guide_only:
            result=guided_radiation(stage,args.guide_only,args.resolution or 40,checkpoint)
        elif args.beam_only:
            import meep as mp
            mp.verbosity(0)
            arrays={}
            case=ANSYS_OUTPUT if args.project_output else ANSYS_FILED if args.project_source else CASES[args.beam_only]
            row=gaussian_only(case,args.resolution or 40,args.vendor_box,arrays,
                              'tidy_compact' if args.vendor_box else args.beam_only,checkpoint)
            arrays_path=stage/'arrays.npz'
            np.savez_compressed(arrays_path,**arrays)
            result=beam_receipt(args.beam_only,row,arrays_path,case)
            if args.project_source or args.project_output:
                result.update(ansys_project_url=ANSYS_PROJECT, ansys_project_sha256=ANSYS_SHA,
                              beam_profile='2D Gaussian with downstream waist; vendor uses a vectorial thin-lens source')
            if args.project_output:
                direct=row['center_direct_output_fraction']
                result['direct_delta_percentage_points']=100*(direct-case['published'])
                result['passed'] = result['passed'] and row['qualified_direct_band'] and bool(0<direct<=1.02)
            (stage/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        else:
            result = run(stage,checkpoint)
        stage.rename(destination)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
