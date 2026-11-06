"""Test physical-plane phase and weak return sensitivity using independent native CUDA meshes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_fdtdx_multiport import load_stage
from scripts.measure_fdtdx_mesh_column import project_column


def native_column(directory):
    plan = json.loads((directory/'run.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    if plan['device'] != 'gpu' or plan['actual_device'] != report['actual_device'] or not (
            report.get('passed') or report.get('sampled_setup_checks_passed')):
        raise ValueError(f'Target or numerical checks failed in {directory.name}')
    model = plan['model']
    names = [p['name'] for p in model['ports']]
    port = report.get('launch',names[0])
    launch = names.index(port)
    files = {}
    measurements = []
    for kind in ('reference','device'):
        path = directory/f'{port}-{kind}.npz'
        stage = load_stage(directory/f'{port}-{kind}.json',path)
        if not stage['settled'] or stage['field_device'] != plan['actual_device']:
            raise ValueError('A stage is unqualified or was run on a different device.')
        files[path.name] = stage['archive_sha256']
        with np.load(path,allow_pickle=False) as data:
            measurements.append({k:data[k] for k in ('frequency_per_um','source_e','source_h','amplitudes')})
    column, _, through, _ = project_column(model,launch,*measurements)
    with np.load(directory/'arrays.npz',allow_pickle=False) as raw:
        archived = raw['s_column'] if 's_column' in raw else raw['s'][:,:,launch]
        np.testing.assert_allclose(column,archived,rtol=1e-12,atol=1e-13)
    files['arrays.npz'] = hashlib.sha256((directory/'arrays.npz').read_bytes()).hexdigest()
    return plan, report, column, through, files


def qualify(directories):
    studies = sorted((native_column(path) for path in directories),key=lambda item:item[0]['resolution'])
    if len(studies)<2 or len({row[0]['resolution'] for row in studies}) != len(studies):
        raise ValueError('Provide at least two distinct meshes.')
    first = studies[0][0]
    ports = first['model']['ports']
    name=studies[0][1].get('launch',ports[0]['name'])
    launch=next(port for port in ports if port['name']==name)
    records = []
    comparisons = []
    for i,(plan,report,column,through,files) in enumerate(studies):
        if report.get('launch',ports[0]['name'])!=name:
            raise ValueError('All meshes must excite the same incoming port.')
        for key in ('model','actual_device','material_model','numerical_sha256','versions'):
            if plan[key] != first[key]:
                raise ValueError(f'The meshes differ in {key}.')
        physical_planes = [(p['name'],p['actual_x'],p['actual_y']) for p in report['sampled_ports']]
        if i and physical_planes != records[0]['planes_um']:
            raise ValueError('S coefficients refer to different physical planes.')
        records.append({'resolution':plan['resolution'],'native_sha256':files,'planes_um':physical_planes,
                        'straight_control_phase_rad':np.angle(through).tolist()})
        if i == 0:
            continue
        previous, previous_ref = studies[i-1][2:4]
        outputs = []
        for j,port in enumerate(ports):
            before,after = previous[:,j],column[:,j]
            phase = np.angle(after*before.conj())
            qualified_phase = (abs(after)>=.02) & (abs(before)>=.02)
            power_delta = abs(abs(after)**2-abs(before)**2)
            change = abs(after-before)
            result = {'port':port['name'],'power_change_max':float(power_delta.max()),
                      'absolute_complex_change_max':float(change.max()),
                      'before_power':(abs(before)**2).tolist(),'after_power':(abs(after)**2).tolist(),
                      'physical_phase_change_rad':[float(v) if valid else None for v,valid in zip(phase,qualified_phase)],
                      'phase_resolved_at_both_meshes':qualified_phase.tolist()}
            if port['inward_sign'] != launch['inward_sign']:
                # Both transfers share the same input plane and the straight control's output x plane.
                # This quotient isolates excess device phase relative to a measured straight guide.
                relative = np.angle(after/through * (before/previous_ref).conj())
                result['straight_relative_phase_change_rad'] = [float(v) if valid else None for v,valid in zip(relative,qualified_phase)]
                result['physical_phase_within_0_10_rad'] = bool(qualified_phase.all() and max(abs(phase))<=.10)
            else:
                result['return_within_0_001_power_and_0_01_amplitude'] = bool(power_delta.max()<=.001 and change.max()<=.01)
            outputs.append(result)
        comparisons.append({'lower_mesh':studies[i-1][0]['resolution'],'higher_mesh':plan['resolution'],'ports':outputs})
    return {'case':first['model']['case'],'actual_device':first['actual_device'],'device_name':first['device_name'],
            'launch':launch['name'],'frequency_per_um':first['model']['frequency_per_um'],
            'thresholds':{'compare_every_frequency':True,'min_phase_amplitude':.02,'physical_phase_rad':.10,
                          'return_power_fraction':.001,'return_complex_amplitude':.01},
            'meshes':records,'measurements':comparisons,
            'scope':'One independently excited column per device. Agreement of successive meshes bounds observed sensitivity; it alone cannot establish continuum accuracy.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(qualify(args.directories),indent=2,allow_nan=False)+'\n')
