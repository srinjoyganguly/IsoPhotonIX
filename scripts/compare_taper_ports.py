"""Compare measured complex S arrays from matched local taper runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def load_run(folder, solver, device):
    job = json.loads((folder / 'job.json').read_text())
    metadata = json.loads((folder / 'metadata.json').read_text())
    config = json.loads((folder / 'config.json').read_text())
    if job['status'] != 'completed' or metadata['config'] != config or job['config'] != config:
        raise ValueError('Choose a completed run with its exact recorded configuration.')
    if (config['study_type'], config['solver'], config['device']) != ('taper-ports-3d', solver, device):
        raise ValueError(f'Expected a {solver} {device} taper-port study.')
    if not all(metadata['settlement'][name]['settled'] for name in (
            'reference_forward', 'device_forward', 'reference_reverse', 'device_reverse')):
        raise ValueError('Each of the four stages must have met its settling criteria.')
    source = folder / 'sparameters.npz'
    with np.load(source, allow_pickle=False) as arrays:
        data = {key: arrays[key].copy() for key in ['wavelength_um', 's', 'mode_neff_ports', 'valid_bin']}
    if not data['valid_bin'].all():
        raise ValueError('At least one spectral bin was unqualified. Inspect its saved valid_bin flag.')
    ports = metadata['ports']
    info = {'id': job['id'], 'actual_device': metadata['actual_device'], 'device_name': metadata['device_name'],
            'reference_planes_um': [ports['port_1_x_um'], ports['port_2_x_um']],
            'npz_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'implementation_sha256': metadata['implementation_snapshot']['sha256']}
    physical = {key:value for key,value in config.items() if key not in ('solver','device')}
    return physical, data, info


def differences(left, right):
    a, b = left['s'], right['s']
    delta = b - a
    denominator = np.linalg.norm(a)
    out = {'overall_complex_relative_l2': float(np.linalg.norm(delta)/denominator) if denominator else None,
           'maximum_absolute_complex_difference': float(np.max(np.abs(delta))),
           'maximum_wavelength_difference_um': float(np.max(np.abs(left['wavelength_um']-right['wavelength_um']))),
           'maximum_port_neff_difference': float(np.max(np.abs(left['mode_neff_ports']-right['mode_neff_ports']))),
           'entries': {}}
    for name, i, j in [('S11',0,0),('S21',1,0),('S12',0,1),('S22',1,1)]:
        u, v = a[:,i,j], b[:,i,j]
        phase = (abs(u)>1e-8) & (abs(v)>1e-8)
        out['entries'][name] = {'max_abs_complex_difference': float(np.max(abs(u-v))),
            'max_abs_modal_power_difference': float(np.max(abs(abs(u)**2-abs(v)**2))),
            'max_abs_principal_phase_difference_rad': float(np.max(abs(np.angle(v[phase]*u[phase].conj())))) if phase.any() else None}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meep-cpu', type=Path, required=True)
    parser.add_argument('--fdtdx-cpu', type=Path, required=True)
    parser.add_argument('--fdtdx-gpu', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configs, data, info = {}, {}, {}
    for key in ('meep_cpu', 'fdtdx_cpu', 'fdtdx_gpu'):
        configs[key], data[key], info[key] = load_run(getattr(args,key), *key.split('_'))
        if configs[key] != configs['meep_cpu'] or info[key]['reference_planes_um'] != info['meep_cpu']['reference_planes_um']:
            raise ValueError('Study settings, geometry and physical reference planes must match.')
        if not np.allclose(data[key]['wavelength_um'], data['meep_cpu']['wavelength_um'], rtol=0, atol=1e-14):
            raise ValueError('Wavelength arrays differ beyond floating-point roundoff. No interpolation is performed.')
    if not np.array_equal(data['fdtdx_cpu']['wavelength_um'], data['fdtdx_gpu']['wavelength_um']):
        raise ValueError('CPU/GPU repeatability needs identical frequency samples.')
    report = {'schema_version':1, 'checked_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Direct differences between actual taper S matrices from matched GUI studies. CPU/GPU repeatability and cross-solver differences are separate measurements.',
        'cases':info, 'same_physical_configuration':True, 'all_four_stages_settled_per_case':True,
        'wavelength_um':data['meep_cpu']['wavelength_um'].tolist(), 'valid_bins_per_case':len(data['meep_cpu']['wavelength_um']),
        'fdtdx_cpu_gpu':differences(data['fdtdx_cpu'],data['fdtdx_gpu']),
        'meep_vs_fdtdx_cpu':differences(data['meep_cpu'],data['fdtdx_cpu']),
        'formulas':'Relative L2 uses all complex bins and entries. Power differences use raw |S|². Phase difference is the principal argument of candidate times conjugate(reference), with amplitudes above 1e-8. No phase fitting, interpolation or amplitude normalization.',
        'limits':'One finite-height taper on one mesh. Meep and FDTDX have different material/source/mode sampling. External benchmark agreement and convergence require separate checks.'}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
