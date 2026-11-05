"""Reconstruct the taper S-matrix from its saved native port fields and modes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.adapters.fdtdx_ports3d_adapter import modal_amplitudes
from backend.sparameters import validate_sparameters


def audit(folder):
    config = json.loads((folder/'config.json').read_text())
    metadata = json.loads((folder/'metadata.json').read_text())
    if (config['solver'], config['study_type']) != ('fdtdx','taper-ports-3d') or metadata.get('config') != config:
        raise ValueError('Expected a saved FDTDX taper-port run with its exact study configuration.')
    with np.load(folder/'sparameters.npz') as arrays:
        spectral = validate_sparameters(arrays, metadata)
    if not spectral['valid_bin'].all():
        raise ValueError('Unsettled or unqualified bins cannot supply this qualification check.')
    measured = {}
    largest_amplitude_error = 0.
    with np.load(folder/'port-modes.npz') as modes:
        np.testing.assert_array_equal(spectral['mode_neff_ports'], modes['mode_neff_ports'][::-1])
        for direction in ('forward','reverse'):
            for kind in ('reference','device'):
                amplitudes = {}
                with np.load(folder/f'{kind}-{direction}-monitors.npz') as monitor:
                    for end in ('left','right'):
                        # Both ends of a reference use the launched width. The
                        # device has distinct physical port profiles.
                        basis = end if kind == 'device' else ('left' if direction == 'forward' else 'right')
                        computed, _ = modal_amplitudes(monitor[end+'_phasors'],modes[basis+'_mode_e'],
                                                      modes[basis+'_mode_h'],1/config['resolution']**2)
                        saved = monitor[end+'_amplitudes']
                        np.testing.assert_allclose(computed,saved,atol=1e-12,rtol=1e-7)
                        largest_amplitude_error = max(largest_amplitude_error, float(np.max(abs(computed-saved))))
                        amplitudes[end] = computed
                measured[kind,direction] = amplitudes
    forward, reverse = measured['device','forward'], measured['device','reverse']
    ref_f, ref_r = measured['reference','forward'], measured['reference','reverse']
    inc_f, inc_r = ref_f['left'][:,0], ref_r['right'][:,1]
    matrix = spectral['s']
    check = np.empty_like(matrix)
    check[:,0,0] = (forward['left'][:,1]-ref_f['left'][:,1])/inc_f
    check[:,1,0] = forward['right'][:,0]/inc_f
    check[:,0,1] = reverse['left'][:,1]/inc_r
    check[:,1,1] = (reverse['right'][:,0]-ref_r['right'][:,0])/inc_r
    check = check[::-1]  # native monitors are ordered by increasing frequency
    np.testing.assert_allclose(matrix,check,atol=1e-9,rtol=1e-7)
    with np.load(folder/'arrays.npz') as arrays:
        field_shape = list(arrays['field'].shape)
    stages = metadata['settlement']
    return {
        'case':folder.name, 'config':config, 'actual_device':metadata['actual_device'],
        'device_name':metadata['device_name'], 'versions':metadata['versions'],
        'run_time_s':metadata['total_seconds'], 'valid_bins':spectral['valid_bin'].tolist(),
        'wavelength_um':spectral['wavelength_um'].tolist(),'n_eff_ports':spectral['mode_neff_ports'].real.tolist(),
        'power':{key:(abs(matrix[:,i,j])**2).tolist() for key,i,j in
                 [('S11',0,0),('S21',1,0),('S12',0,1),('S22',1,1)]},
        'complex_s':{'real':matrix.real.tolist(),'imag':matrix.imag.tolist()},
        'max_abs_reciprocity_difference':float(abs(matrix[:,1,0]-matrix[:,0,1]).max()),
        'column_modal_power':(abs(matrix)**2).sum(axis=1).tolist(),
        'measured_planes_um':metadata['ports']['actual_x_um'],
        'stage_actual_device':{name:s['actual_device'] for name,s in stages.items()},
        'stage_settled_fs':{name:s['elapsed_fs'] for name,s in stages.items()},
        'energy_ratio':{name:s['energy_ratio'] for name,s in stages.items()},
        'spectral_change':{name:s['spectral_relative_change'] for name,s in stages.items()},
        'maximum_stored_amplitude_reprojection_error':largest_amplitude_error,
        'maximum_s_reconstruction_error':float(abs(matrix-check).max()),
        'array_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.glob('*.npz'))},
        'late_time_field_shape_zyx':field_shape,
        'material_grid_verified':metadata['geometry_translation']['material_grid_verified'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir',type=Path)
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    result = {'checked_at':datetime.now(timezone.utc).isoformat(), **audit(args.result_dir)}
    text = json.dumps(result,indent=2,allow_nan=False)
    if args.report:
        args.report.write_text(text+'\n')
    else:
        print(text)
