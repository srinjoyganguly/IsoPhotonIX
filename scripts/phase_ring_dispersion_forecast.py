"""Predict ring mesh shifts from the independently measured straight-guide dispersion."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_loaded_ring3d import BASIS_WAVELENGTHS, WAVELENGTHS, directional_spectrum
from scripts.calibration_checkpoint import MeasurementCheckpoints
from scripts.phase_component_comparison import file_sha, measured_beta, native_data


def straight(stage, resolution):
    receipt = json.loads((stage / 'receipt.json').read_text())
    spec = receipt['specification']
    geometry = spec['settings']['geometry']
    if (spec['label'] != 'reference' or geometry['resolution'] != resolution
            or geometry['vendor'] != 'lumerical' or geometry['ring_interface'] != 'reciprocal-wall'
            or receipt['report']['device'] != 'cuda:0'
            or MeasurementCheckpoints.json_hash(spec) != receipt['signature']
            or MeasurementCheckpoints.json_hash(receipt['report']) != receipt['report_sha256']
            or file_sha(stage / 'arrays.npz') != receipt['arrays_sha256']):
        raise ValueError('Expected an intact matching straight-guide GPU checkpoint.')
    length = (geometry['ports'][1][0] - geometry['ports'][0][0]) / resolution
    curves = {}
    with np.load(stage / 'arrays.npz', allow_pickle=False) as data:
        neff = np.interp(WAVELENGTHS, BASIS_WAVELENGTHS, data['mode_neff'][0])
        traces = {key:data[key] for key in ('time_um_c','projections')}
        for fraction in (.8, 1.):
            modal = directional_spectrum(traces, fraction)
            through = modal[:,1,0] / modal[:,0,0]
            beta = measured_beta(through, WAVELENGTHS, neff, length)
            if np.any(np.diff(beta) >= 0) or np.any(abs(abs(through)-1) >= .05):
                raise ValueError('Straight transfer must have a unique monotone phase branch and near-unit transmission.')
            curves[fraction] = beta
    return curves, {'resolution':resolution,'port_separation_um':length,'stage_signature':receipt['signature'],
                    'stage_arrays_sha256':receipt['arrays_sha256'],
                    'beta_full_rad_per_um':curves[1.].tolist(),'beta_early_rad_per_um':curves[.8].tolist()}


def predict(reference20, reference24, device24):
    beta20, source20 = straight(reference20, 20)
    beta24, source24 = straight(reference24, 24)
    report, archive = native_data(device24)
    with archive:
        if (report['physical_model']['ring_interface'] != 'reciprocal-wall'
                or report['physical_model']['resolution'] != 24 or not archive['valid_bins'].all()):
            raise ValueError('Require the stable 24-cell ring and its exact straight reference.')
        with np.load(reference24/'arrays.npz',allow_pickle=False) as stage:
            for key in ('time_um_c','projections','mode_neff','source_E','source_H'):
                np.testing.assert_array_equal(archive['reference_'+key],stage[key])
        w = np.array(archive['wavelength_um'])
        np.testing.assert_array_equal(w, WAVELENGTHS)
    grid = float(np.median(np.diff(w)))
    anchors = [p['wavelength_um'] for p in report['peaks'] if 1.50 <= p['wavelength_um'] <= 1.59]
    if len(anchors) != 3:
        raise ValueError('Expected the three interior predeclared drop resonances.')

    def inverse_20(position24, curve24, curve20):
        target = np.interp(position24, w, curve24)
        if not curve20[-1] < target < curve20[0]:
            raise ValueError('Predicted line is outside the measured straight dispersion band.')
        return float(np.interp(target, curve20[::-1], w[::-1]))

    rows = []
    for anchor in anchors:
        point = inverse_20(anchor, beta24[1.], beta20[1.])
        extremes = [inverse_20(position, c24, c20) for position in (anchor-grid/2,anchor+grid/2)
                    for c24 in beta24.values() for c20 in beta20.values()]
        rows.append({'anchor_r24_nm':anchor*1000,'prediction_r20_nm':point*1000,
                     'predicted_r20_minus_r24_nm':(point-anchor)*1000,
                     'sampling_and_prefix_interval_nm':[(min(extremes)-grid/2)*1000,(max(extremes)+grid/2)*1000]})
    return {'status':'prediction before the 20-cell device spectrum','target':'Three 20-cell/µm loaded-ring drop peaks',
            'hypothesis':'A mesh change that acts only through uniform straight-guide propagation preserves beta at a corresponding resonance. Solve beta20(lambda20) = beta24(lambda24) directly.',
            'line_rule':'Use the native drop-peak detector in each receipt. Label the three 24-cell peaks from 1500 through 1590 nm in increasing wavelength. At 20 cells/µm choose exactly one valid detected peak in each disjoint half-spacing window about its prediction, using half the minimum separation between predicted peaks. Zero or multiple candidates are unassigned. Correspondence is conditional; angular order is unmeasured.',
            'interval_rule':'Use both 80% and 100% prefixes of both measured straight references and +/- half a spectral sample on both the anchor and the observed peak. This brackets measurement and sampling sensitivity, not unknown curvature/coupler phase or model error. A correctly assigned peak outside this interval rejects the propagation-only prediction for that line.',
            'excluded_inputs':'The 20-cell device and commercial power or phase were not used. No phase, index or wavelength offset was fitted.',
            'spectral_grid_um':w.tolist(),'anchors_r24_report':device24.name,'anchors_r24_arrays_sha256':report['arrays_sha256'],
            'straight_reference_inputs':[source20,source24],'predictions':rows}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('reference20','reference24','device24','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    args=p.parse_args()
    result=predict(args.reference20,args.reference24,args.device24)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
