"""Audit native four-launch results across meshes, preserving both original and electric bases."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(runs, references, controls, dest):
    from scripts.calibrate_multimode import CHANNELS, FREQUENCIES
    matrices, notes, samples = {}, [], {}
    for directory in runs:
        path = directory / 'arrays.npz'
        original = json.loads((directory/'report.json').read_text())
        mesh = original['parameters']['resolution']
        with np.load(path, allow_pickle=False) as native:
            phase = int(native['phase_convention_version']) if 'phase_convention_version' in native else 1
            if phase not in (1, 2):
                raise ValueError('Unknown native phase convention')
            matrix = np.empty_like(native['s'])
            for column in range(4):
                side, mode = divmod(column, 2)
                ref = native[f'reference_{column}'].copy()
                dev = native[f'device_{column}'].copy()
                for prefix, array in (('reference',ref), ('device',dev)):
                    samples[f'mesh{mesh}_{prefix}_{column}_as_measured'] = array.copy()
                    if phase == 1:
                        array[:,:,1,:] *= -1
                mat = np.empty((4,len(FREQUENCIES)),complex)
                incident = ref[side,mode,side]
                mat[2*side:2*side+2] = (dev[side,:,1-side] - ref[side,:,1-side]) / incident
                mat[2*(1-side):2*(1-side)+2] = dev[1-side,:,side] / incident
                matrix[:,:,column] = mat.T
            np.testing.assert_allclose(abs(matrix)**2,abs(native['s'])**2,rtol=1e-13,atol=1e-16)
            indices = native['device_neff_0'][0,:,0,:].copy()
            samples[f'mesh{mesh}_s_electric'] = matrix
            samples[f'mesh{mesh}_original_phase_version'] = phase
        notes.append({'mesh':mesh, 'native_arrays_sha256':digest(path),'native_phase_version':phase,
                      'phase_conversion':'Signed flux overlap to electric amplitudes' if phase==1 else 'Recorded electric amplitudes',
                      'left_port_neff':indices.tolist()})
        matrices[mesh] = matrix
    meshes = sorted(matrices)
    mesh = meshes[-1]
    matrix = matrices[mesh]
    power = abs(matrix)**2
    differences = {}
    for lower,higher in zip(meshes,meshes[1:]):
        differences[f'{lower}_to_{higher}'] = {
            'max_power_fraction_change':float(np.max(abs(abs(matrices[higher])**2-abs(matrices[lower])**2))),
            'max_complex_amplitude_change':float(np.max(abs(matrices[higher]-matrices[lower])))}
    metrics = {'largest_guided_output_sum':float(power.sum(axis=1).max()),
               'max_power_reciprocity_residual':float(np.max(abs(power-power.transpose(0,2,1)))),
               'max_complex_reciprocity_residual':float(np.max(abs(matrix-matrix.transpose(0,2,1))))}
    passed = np.isfinite(matrix).all() and metrics['largest_guided_output_sum'] < 1.03 and metrics['max_power_reciprocity_residual'] < .03
    if references:
        refs = {}
        for label,file in zip(('coarse','refined','wide'),references):
            with np.load(file,allow_pickle=False) as data:
                refs[label]=data['s'].copy()
                samples['independent_'+label] = refs[label]
        refined=refs['refined']
        errors={}
        for size,native in matrices.items():
            errors[str(size)] = {'max_power_fraction_error':float(np.max(abs(abs(native)**2-abs(refined)**2))),
                                 'max_complex_amplitude_error':float(np.max(abs(native-refined)))}
        metrics.update(power_and_complex_error_by_mesh=errors,
                       reference_mesh_power_change=float(np.max(abs(abs(refs['coarse'])**2-abs(refined)**2))),
                       reference_box_power_change=float(np.max(abs(abs(refs['coarse'])**2-abs(refs['wide'])**2))))
        passed = passed and errors[str(mesh)]['max_power_fraction_error'] < .04 and errors[str(mesh)]['max_complex_amplitude_error'] < .15 and metrics['reference_box_power_change'] < .01 and metrics['reference_mesh_power_change'] < .01
    checked_controls = []
    for folder in controls:
        row = json.loads((folder/'report.json').read_text())
        cfgmesh = row['resolution']
        with np.load(folder/'arrays.npz') as saved:
            measured = saved['s']
        baseline=matrices[cfgmesh][:,:,0]
        entry = {'control':row['control'], 'resolution':cfgmesh,
                 'pml_um':row['pml_um'],'aperture_um':row['aperture_um'],'after_source_um_over_c':row['after_source_um_over_c'],
                 'original_sha256':digest(folder/'arrays.npz'),'native_phase_version':row.get('phase_convention_version',1),
                 'scope':'One separately measured left TE0 input column.'}
        samples['control_'+row['control']+'_original_s'] = measured
        if row['control']=='straight':
            entry['max_spurious_mode_power'] = float(np.max(abs(measured[:,[0,1,3]])**2))
            entry['max_through_power_error'] = float(np.max(abs(abs(measured[:,2])**2-1)))
        else:
            entry['max_power_fraction_change'] = float(np.max(abs(abs(measured)**2-abs(baseline)**2)))
            entry['max_complex_through_change'] = float(np.max(abs(measured[:,2:]-baseline[:,2:])))
        checked_controls.append(entry)
    case = original['case']
    output = dest.with_suffix('.npz')
    np.savez_compressed(output, **samples)
    report = dict(schema_version=1,case=case,solver='Meep',version=original['version'],actual_device='CPU',
                  phase_convention_version=2, passed=bool(passed), channels=CHANNELS, wavelength_um=(1/FREQUENCIES).tolist(),
                  mesh=mesh, parameters={**original['parameters'],'resolution':mesh}, checks=metrics,
                  mesh_sensitivity=differences,native_runs=notes,controls=checked_controls,
                  s_real=matrix.real.tolist(),s_imag=matrix.imag.tolist(),power=power.tolist(),
                  source_arrays={'file':output.name,'sha256':digest(output),
                                 'contents':'Original measured modal arrays, corrected electric-basis S, available single-launch controls and independent matrices. All four input columns are independently simulated.'},
                  vendor_sources=original['vendor_sources'])
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+',type=Path,required=True)
    parser.add_argument('--references',nargs=3,type=Path)
    parser.add_argument('--controls',nargs='*',type=Path,default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=assess(args.runs,args.references,args.controls,args.output)
    print(result['case'],result['passed'],result['checks'])
    if not result['passed']:raise SystemExit(1)
