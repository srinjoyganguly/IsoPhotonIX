"""Compare absolute powers and resolve line position, height and power normalization separately."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.compare_loaded_ring3d import lumerical_port_spectra

TARGET_UM = 1.55
POWER_TOLERANCE = 0.05  # Workbench investigation target in incident-power fraction.
NOMINAL = [
    'v012-ring3d-lumerical-r16-t3200', 'v012-ring3d-lumerical-r20-t3200',
    'v012-ring3d-lumerical-r24-t3200', 'v012-ring3d-tidy3d-r16-t4800',
    'v012-ring3d-tidy3d-r20-t4800', 'v012-grating3d-lumerical-r12',
    'v012-grating3d-lumerical-r14', 'v012-grating3d-tidy3d-r12', 'v012-grating3d-tidy3d-r14',
]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sample(wavelengths, target):
    indices = np.flatnonzero(np.isclose(wavelengths, target, rtol=0, atol=1e-12))
    if len(indices) != 1:
        raise ValueError(f'Expected one measured sample at {target} µm.')
    return int(indices[0])


def gain_interval(local, published, tolerance):
    """All positive power multipliers that meet the absolute error bound at every sample."""
    local, published = np.asarray(local), np.asarray(published)
    if local.shape != published.shape or local.size == 0 or np.any(local <= 0):
        raise ValueError('Gain test requires paired positive measured powers.')
    low, high = (published - tolerance) / local, (published + tolerance) / local
    return {'tolerance_power_fraction': tolerance, 'minimum_required': max(0., float(low.max())),
            'maximum_allowed': float(high.min()), 'feasible': bool(max(0., low.max()) <= high.min()),
            'minimum_set_by_index': int(low.argmax()), 'maximum_set_by_index': int(high.argmin())}


def ring_details(report, raw, reference):
    comparison = report['vendor_comparison']
    assert comparison['wavelength_shift_applied_nm'] == 0
    row = report['series'][report['vendor']]
    wave = np.asarray(row['wavelength_um'])
    valid = np.asarray(row['valid_bins'])
    np.testing.assert_array_equal(raw['valid_bins'], valid)
    powers = np.abs(raw['s'])**2
    outputs = ('reflection', 'through', 'drop', 'add_side')
    for i, port in enumerate(outputs):
        np.testing.assert_allclose(powers[:, i], row[port], atol=1e-11)
    j = sample(wave, TARGET_UM)
    if not valid[j]:
        raise ValueError('The target wavelength failed the native qualification checks.')
    channels = {}
    extra = {}
    for i, port in enumerate(outputs):
        if port in comparison['channels']:
            paired = comparison['channels'][port]['paired_spectrum']
        elif reference:
            vendor_wave, vendor_ports, _ = reference
            mask = valid & (wave >= max(1.5, vendor_wave.min())) & (wave <= min(1.6, vendor_wave.max()))
            paired = {'wavelength_um': wave[mask].tolist(), 'local_power': powers[mask, i].tolist(),
                      'published_power': np.interp(wave[mask], vendor_wave, vendor_ports[port]).tolist()}
            extra[port] = paired
        else:
            continue
        k = sample(paired['wavelength_um'], TARGET_UM)
        local, published = paired['local_power'][k], paired['published_power'][k]
        np.testing.assert_allclose(local, powers[j, i])
        difference = np.asarray(paired['local_power']) - np.asarray(paired['published_power'])
        channels[port] = {'local': local, 'published': published,
                          'mean_absolute_difference': float(np.abs(difference).mean())}
    line = min(comparison['published_drop_resonances'], key=lambda p: abs(p['read_peak_um'] - TARGET_UM))
    nearby = line.get('nearest_local_peak_um')
    peak = next((p for p in report['loaded_loss']['peaks'] if p['sampled_drop_peak_um'] == nearby), None)
    resonance = {'published_sampled_center_um': line['read_peak_um'], 'published_peak_power': line['read_peak_power'],
                 'published_fwhm_nm': line.get('fwhm_nm') if line['loaded_q_half_max'] is not None else None}
    if peak is not None:
        idx = sample(wave, nearby)
        resonance.update(local_sampled_center_um=nearby, local_sampled_peak_power=float(powers[idx, 2]),
                         nearest_sampled_center_difference_nm=1000 * (nearby-line['read_peak_um']))
        if peak['qualified']:
            fit = peak['full_fit']
            resonance.update(local_fitted_center_um=fit['wavelength_um'], local_fwhm_nm=fit['linewidth_nm'],
                             local_target_distance_in_linewidths=abs(fit['wavelength_um']-TARGET_UM)*1000/fit['linewidth_nm'])
    if resonance['published_fwhm_nm']:
        resonance['published_target_distance_in_linewidths'] = abs(line['read_peak_um']-TARGET_UM)*1000/resonance['published_fwhm_nm']
    return {'observable': 'ring', 'target_um': TARGET_UM, 'channels_at_target': channels,
            'nearest_published_drop_line': resonance, 'additional_spectra': extra,
            'spacing_nm': {'local': report['measured_drop_fsr_nm'], 'published': report['published_drop_fsr_nm']}}


def grating_details(report, raw):
    ref = next(c for c in report['comparisons'] if c.get('geometry_matched'))
    wave = np.asarray(ref['wavelength_um'])
    measured = np.abs(raw['output_mode'])**2 / raw['incident_power']
    if not np.asarray(raw['valid']).all():
        raise ValueError('Grating spectrum contains an unqualified bin.')
    np.testing.assert_allclose(wave, report['result']['wavelength_um'], atol=0, rtol=0)
    np.testing.assert_allclose(measured, report['result']['beam_to_guide'], rtol=1e-11)
    published = np.asarray(ref.get('published_power', ref.get('published_s21_power')))
    delta = measured - published
    j = sample(wave, TARGET_UM)
    return {'observable': 'grating', 'target_um': TARGET_UM, 'wavelength_um': wave.tolist(),
            'local_power': measured.tolist(), 'published_power': published.tolist(),
            'local_at_target': float(measured[j]), 'published_at_target': float(published[j]),
            'mean_absolute_difference': float(np.abs(delta).mean()), 'max_absolute_difference': float(np.abs(delta).max()),
            'local_best_sample_um': float(wave[measured.argmax()]), 'published_best_sample_um': float(wave[published.argmax()]),
            'gain_only': gain_interval(measured, published, POWER_TOLERANCE)}


def diagnose(directory, lumerical_file, names):
    reference = lumerical_port_spectra(lumerical_file)
    rows = {}
    for name in names:
        path = directory / (name+'.json')
        report = json.loads(path.read_text())
        raw = directory / report['arrays_file']
        if not report['passed'] or sha(raw) != report['arrays_sha256']:
            raise ValueError(f'Unqualified result or changed arrays in {name}.')
        with np.load(raw, allow_pickle=False) as arrays:
            if report['case_id']=='loaded-ring-3d':
                row = ring_details(report, arrays, reference if report['vendor']=='lumerical' else None)
            elif report['case_id']=='finite-grating-3d':
                row = grating_details(report, arrays)
            else:
                raise ValueError(f'Unexpected component in {name}.')
        rows[path.name] = {**row, 'source_report_sha256': sha(path), 'native_arrays_sha256': report['arrays_sha256']}
    return {'target_um': TARGET_UM, 'lumerical_s_parameter_sha256': sha(lumerical_file), 'records': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1]/'docs/verification')
    parser.add_argument('--lumerical-ring', type=Path, required=True, help='Original FDTDtoINTERCONNECT.txt attachment')
    parser.add_argument('--extra-record', action='append', default=[], help='Additional verified receipt stem')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.directory, args.lumerical_ring, [*NOMINAL, *args.extra_record])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
