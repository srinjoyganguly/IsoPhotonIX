"""Compare two measured 21-wavelength Lumerical gratings and their five native control points."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from backend.adapters.fdtdx_ports3d_adapter import modal_amplitudes
from scripts.calibrate_finite_grating import monitoring_wavelengths
from scripts.verify_finite_grating import verify

EVIDENCE = Path(__file__).resolve().parents[1]/'docs/verification'
MODELS = {'volume-average':'v012-grating3d-lumerical-r12.json',
          'direct-yee':'v013-grating3d-lumerical-direct-r12.json'}
POWER_STABILITY_PP = 0.25


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def qualified_curve(path, model, vendor):
    report=verify(path)
    settings=report['settings']
    anchor_path=EVIDENCE/MODELS[model]
    original=json.loads(anchor_path.read_text())
    nominal_settings={k:v for k,v in settings.items() if k!='grating_spectrum'}
    if (not report['passed'] or not original['passed'] or
            settings.get('grating_spectrum')!='dense' or nominal_settings!=original['settings']):
        raise ValueError('Measurements do not describe the same qualified 12-cell geometry and material model.')
    wavelengths=monitoring_wavelengths(settings)
    ref=next(c for c in report['comparisons'] if c['geometry_matched'])
    if ref['vendor']!='Lumerical' or ref['vendor_file_sha256']!=vendor['s_file_sha256']:
        raise ValueError('Unexpected comparison geometry or vendor data file.')
    np.testing.assert_array_equal(report['result']['wavelength_um'],wavelengths)
    np.testing.assert_array_equal(wavelengths[::5],original['result']['wavelength_um'])
    np.testing.assert_array_equal(wavelengths,vendor['wavelength_um'])
    np.testing.assert_array_equal(ref['published_s21_power'],vendor['published_s21_power'])
    stages={}
    for role in ('reference','device'):
        current=report['stages'][role]; previous=original['stages'][role]
        keys=('actual_device','shape','source_eh_sha256','source_time_offsets_sha256','scalar_permittivity_sha256')
        for key in keys:
            if current[key]!=previous[key]:
                raise ValueError(f'{role} anchor changed {key}.')
        if 'electric_component_permittivity_sha256' in previous:
            if current['electric_component_permittivity_sha256']!=previous['electric_component_permittivity_sha256']:
                raise ValueError(f'{role} component sampling changed with the monitor grid.')
        stages[role]={'time_steps':current['time_steps'],'five_bin_time_steps':previous['time_steps'],
                      'source_eh_sha256':current['source_eh_sha256'],'source_time_offsets_sha256':current['source_time_offsets_sha256'],
                      'scalar_permittivity_sha256':current['scalar_permittivity_sha256'],
                      'electric_component_permittivity_sha256':current['electric_component_permittivity_sha256']}
    local=np.asarray(report['result']['beam_to_guide'])
    published=np.asarray(vendor['published_s21_power'])
    anchor=np.asarray(original['result']['beam_to_guide'])
    delta=100*(local-published)
    np.testing.assert_allclose(delta,ref['local_minus_published_percentage_points'],atol=1e-12)
    np.testing.assert_allclose(abs(delta).mean(),ref['mean_absolute_difference_percentage_points'],atol=1e-12)
    local_max=int(np.argmax(local)); published_max=int(np.argmax(published))
    windows=[]
    with np.load(path/'arrays.npz',allow_pickle=False) as arrays:
        incident=arrays['ref_history_observable']
        device=arrays['dev_history_observable']
        np.testing.assert_allclose(abs(device[-1])**2/incident[-1],local,rtol=1e-7)
        for offset in (1,2):
            power=abs(device[-1-offset])**2/incident[-1-offset]
            windows.append({'previous_windows':offset,'peak_sample_nm':float(1000*wavelengths[np.argmax(power)]),
                            'reference_time_fs':report['stages']['reference']['history'][-1-offset]['time_fs'],
                            'device_time_fs':report['stages']['device']['history'][-1-offset]['time_fs'],
                            'max_absolute_difference_percentage_points':float(100*np.max(abs(power-local)))})
    time_window=report['stages']['device']['history'][-1]['time_fs']
    anchor_error=float(100*np.max(abs(local[::5]-anchor)))
    peak_samples=[float(1000*wavelengths[local_max]), *(row['peak_sample_nm'] for row in windows)]
    return {'material_model':model, 'actual_device':report['actual_device'], 'device_name':report['device_name'],
            'qualified_wavelengths':sum(report['result']['valid_beam_bins']), 'wavelength_nm':(1000*wavelengths).tolist(),
            'local_coupling_fraction':local.tolist(),'published_coupling_fraction':published.tolist(),
            'local_minus_published_percentage_points':delta.tolist(),
            'all_wavelength_mean_absolute_difference_percentage_points':float(np.mean(abs(delta))),
            'five_anchor_mean_absolute_difference_percentage_points':float(np.mean(abs(delta[::5]))),
            'local_sampled_peak_nm':float(1000*wavelengths[local_max]),
            'local_sampled_peak_fraction':float(local[local_max]),
            'published_same_grid_sampled_peak_nm':float(1000*wavelengths[published_max]),
            'peak_separation_from_50_point_published_peak_nm':float(1000*(wavelengths[local_max]-vendor['peak_wavelength_um'])),
            'five_point_local_sampled_peak_nm':float(1000*wavelengths[5*int(np.argmax(anchor))]),
            'observation_windows':windows,'device_observation_duration_fs':time_window,
            'observed_peak_sample_range_across_windows_nm':[min(peak_samples),max(peak_samples)],
            'inverse_duration_scale_at_1550_nm':float(1000*1.55**2/(0.299792458*time_window)),
            'stability_target_percentage_points':POWER_STABILITY_PP,
            'within_window_target':all(w['max_absolute_difference_percentage_points']<=POWER_STABILITY_PP for w in windows),
            'within_anchor_target':anchor_error<=POWER_STABILITY_PP,
            'max_anchor_difference_fraction':float(np.max(abs(local[::5]-anchor))),
            'anchor_difference_fraction':(local[::5]-anchor).tolist(), 'five_wavelength_anchor_receipt':anchor_path.name,
            'reference_file_sha256':vendor['s_file_sha256'],'full_native_report_sha256':sha(path/'report.json'),
            'full_native_arrays_sha256':sha(path/'arrays.npz'), 'stages':stages}


def compare_output_projections(directory):
    cases={}
    with (np.load(directory/'volume-average/arrays.npz',allow_pickle=False) as baseline,
          np.load(directory/'direct-yee/arrays.npz',allow_pickle=False) as direct):
        for key in ('dev_grid_x_um','dev_grid_y_um','dev_grid_z_um','dev_wavelength_um'):
            np.testing.assert_array_equal(baseline[key],direct[key])
        center=baseline['dev_wavelength_um'].tolist().index(1.55)
        for name, fields, alternate in [('volume-average',baseline,direct),('direct-yee',direct,baseline)]:
            projected=modal_amplitudes(fields['dev_output'],alternate['dev_mode_e'],alternate['dev_mode_h'],1/12**2)[0][:,1]
            own=abs(fields['output_mode'])**2/fields['incident_power']
            counterfactual=abs(projected)**2/fields['incident_power']
            delta_pp=100*(counterfactual-own)
            cases[name]={'alternate_minus_own_percentage_points':delta_pp.tolist(),
                         'nominal_change_percentage_points':float(delta_pp[center]),
                         'maximum_absolute_change_percentage_points':float(np.max(abs(delta_pp))),
                         'native_strip_neff_at_1550_nm':float(fields['dev_mode_neff'][center].real)}
    return {'scope':'The same native six-component guide fields and incident normalization are projected onto either local material model’s outgoing mode at each native frequency. This isolates the readout-basis swap for these computed fields.',
            'fixed_fields':cases}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',required=True,type=Path,help='Contains volume-average and direct-yee native output directories.')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    vendor=json.loads((EVIDENCE/'v013-grating-dense-reference.json').read_text())
    cases={model:qualified_curve(args.runs/model,model,vendor) for model in MODELS}
    base,changed=cases.values()
    for role in ('reference','device'):
        for field in ('source_eh_sha256','source_time_offsets_sha256'):
            if base['stages'][role][field]!=changed['stages'][role][field]:
                raise ValueError('The two material models use different source amplitudes or delays.')
    result={'passed':all(c['within_window_target'] and c['within_anchor_target'] for c in cases.values()),
            'geometry':'Lumerical curved grating','resolution_cells_per_um':12,'cases':cases,
            'sample_spacing_nm':5,'published_file_samples':vendor['input_samples'],
            'published_file_peak_nm':1000*vendor['peak_wavelength_um'],
            'published_file_peak_fraction':vendor['peak_power'],
            'output_mode_projection':compare_output_projections(args.runs),
            'local_sampled_peak_change_nm':changed['local_sampled_peak_nm']-base['local_sampled_peak_nm'],
            'all_wavelength_mean_absolute_difference_change_percentage_points':
                changed['all_wavelength_mean_absolute_difference_percentage_points']-
                base['all_wavelength_mean_absolute_difference_percentage_points'],
            'scope':'All local powers are separate native phasor detector evaluations. Peaks are maxima over the measured grid; the bin spacing is not a continuous-peak error bar.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
