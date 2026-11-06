"""Check the Tidy grating's extended notebook against the complete shorter time trace."""
import argparse
import json
from pathlib import Path

import numpy as np

from backend.calibration_replay import verify_source
from scripts.calibrate_finite_grating import file_hash
from scripts.verify_finite_grating import verify
from scripts.verify_grating_temporal_notebook import channel_change, field_errors
from scripts.verify_vendor_grating_notebooks import check_executed_notebook


def audit(notebook, baseline, observed_windows=None, sampled_memory=None):
    check_executed_notebook(notebook)
    manifest=verify_source(notebook)
    if '--resolve-weak-return' not in manifest['arguments']:
        raise ValueError('The export did not request a reciprocal-return stop.')
    new_dir=notebook/'notebook-result'
    old,current=verify(baseline),verify(new_dir)
    if current['settings']!={**old['settings'],'resolve_weak_return':True}:
        raise ValueError('The duration control differs in more than its early-stop policy.')
    if old['settings']['max_time_fs']!=2000 or current['settings']['geometry']!='tidy3d':
        raise ValueError('This experiment uses the 2,000 fs Tidy3D-design ceiling.')
    stages=current['stages']
    for label, stage in stages.items():
        a,b=old['stages'][label]['performance'],stage['performance']
        if stage['acquisition']['status']!='fresh' or b['platform']!='gpu' or a['acquisition_id']==b['acquisition_id']:
            raise ValueError('Independent, fresh CUDA stages are required.')
        if any(a[key]!=b[key] for key in ('cells','grid_shape','dt_seconds','spacing_um','pml_cells','device_kind')):
            raise ValueError('Shape, medium, timestep or CUDA device differs from the earlier observation.')
        if stage['native_worker_sha256']!=manifest['files']['source/scripts/calibrate_finite_grating.py']:
            raise ValueError('A native stage used a different numerical worker.')
    if stages['reference']['arrays_sha256']!=old['stages']['reference']['arrays_sha256']:
        raise ValueError('The incident source or recorded reference arrays changed.')
    pre=old['stages']['device']
    post=stages['device']
    times=lambda stage: np.asarray([r['time_fs'] for r in stage['history']])
    np.testing.assert_array_equal(times(pre),times(post)[:len(pre['history'])])
    if post['time_steps']<=pre['time_steps']:
        raise ValueError('The duration control must include the original acquisition and additional physical steps.')
    with np.load(baseline/'arrays.npz',allow_pickle=False) as a,np.load(new_dir/'arrays.npz',allow_pickle=False) as b:
        prefix_entries=('dev_history_observable','dev_reciprocal_return_history','dev_energy_time_fs')
        for key in prefix_entries:
            np.testing.assert_array_equal(a[key],b[key][:len(a[key])])
        energy=a['dev_energy']
        extended_energy=b['dev_energy'][:len(energy)]
        if energy.dtype!=np.float32 or np.any(energy<0) or np.any(extended_energy<0):
            raise ValueError('Expected finite nonnegative single-precision energy reductions.')
        # The scalar detector reduces a full volume. Use the existing workbench
        # export tolerance for this reduction and report its actual float32 ULPs.
        np.testing.assert_allclose(energy,extended_energy,rtol=2e-6,atol=0)
        energy_error={'samples':len(energy),'byte_equal':np.array_equal(energy,extended_energy),
            'max_absolute_difference':float(np.max(abs(energy-extended_energy))),
            'max_ulp_difference':int(np.max(abs(energy.view(np.int32).astype(np.int64)-extended_energy.view(np.int32)))),
            'relative_l2':float(np.linalg.norm(energy-extended_energy)/np.linalg.norm(energy)),
            'relative_tolerance':2e-6}
        np.testing.assert_array_equal(a['valid'],b['valid'])
        for key in a.files:
            if key.startswith('ref_') or any(token in key for token in ('_grid_','_material_','_source_','_mode_','_wavelength_')):
                # Static material, basis, axes and source samples. Output amplitudes
                # are physical observables and can change as the pulse tail clears.
                np.testing.assert_array_equal(a[key],b[key])
        power=(100*(b['coupling']-a['coupling'])).tolist()
        guide=channel_change(a['output_mode']/np.sqrt(a['incident_power']),b['output_mode']/np.sqrt(b['incident_power']))
        s11=channel_change(a['dev_reciprocal_return_history'][-1],b['dev_reciprocal_return_history'][-1])
        fields={k:field_errors(a[k],b[k],axis) for k,axis in
                [('dev_upper',1),('dev_output',1),('dev_center_fields',0)]}
        samples=b['dev_reciprocal_return_history']
    ledger={}
    if sampled_memory:
        rows=[json.loads(line) for line in sampled_memory.read_text().splitlines() if line]
        for label in stages:
            selected=[r for r in rows if ['tidy3d',16,label] in r['jobs']]
            ledger[label]={'samples':len(selected),'peak_observed_global_used_mib':max((r['used_mib'] for r in selected),default=None)}
    extra_wall=None
    if observed_windows:
        marks=[json.loads(line) for line in observed_windows.read_text().splitlines() if line]
        marks={r['rounded_window_fs']:r['observed_monotonic_s'] for r in marks if r['stage']=='device'}
        start,end=round(times(pre)[-1]),round(times(post)[-1])
        if start in marks and end in marks:
            extra_wall=marks[end]-marks[start]
    return {'geometry':'tidy3d','domain':'mirror-y','resolution':16,'wavelength_um':current['result']['wavelength_um'],
            'manifest_sha256':file_hash(notebook/'manifest.json'),'worker_sha256':post['native_worker_sha256'],
            'notebook_sha256':file_hash(notebook/'calibration.ipynb'),'default_arrays_sha256':old['arrays_sha256'],
            'control_arrays_sha256':current['arrays_sha256'],'identical_reference_arrays_sha256':stages['reference']['arrays_sha256'],
            'exact_prefix_arrays':list(prefix_entries),'prefix_energy_reduction':energy_error,
            'exact_prefix_windows':len(pre['history']),'total_windows':len(post['history']),
            'max_time_fs':current['settings']['max_time_fs'],'old_stop_time_fs':times(pre)[-1],'new_stop_time_fs':times(post)[-1],
            'old_steps':pre['time_steps'],'new_steps':post['time_steps'],'additional_steps':post['time_steps']-pre['time_steps'],
            'old_device_stepping_s':pre['performance']['synchronized_step_seconds'],
            'additional_observed_elapsed_s':extra_wall,
            'elapsed_scope':'Window message timestamp difference sampled every 0.1 s. Includes synchronized stepping, host-side field analysis and launch gaps beyond the previous stopping point.',
            'stop':post['duration_stop'],'old_window_qualification':old['radiation']['reciprocal_return_settling'],
            'new_window_qualification':current['radiation']['reciprocal_return_settling'],
            'guide_qualified':current['result']['valid_beam_bins'],'default_guided_power':old['result']['beam_to_guide'],
            'control_guided_power':current['result']['beam_to_guide'],'guided_power_difference_pp':power,
            'guide_complex_change':guide,'return_complex_change':s11,'final_measured_field_changes_relative_l2':fields,
            'window_time_fs':times(post).tolist(),'return_real':samples.real.tolist(),'return_imag':samples.imag.tolist(),
            'stages':{k:{'performance':v['performance'],'acquisition':v['acquisition'],'memory_admission':v['memory_admission']}
                      for k,v in stages.items()},'sampled_memory':ledger,
            'scope':'One additional duration control at the unchanged mesh with an identical short-run prefix. Read the amplitude and above-floor phase masks separately.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('notebook','baseline','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--observed-windows',type=Path)
    parser.add_argument('--sampled-memory',type=Path)
    args=parser.parse_args()
    record=audit(args.notebook,args.baseline,args.observed_windows,args.sampled_memory)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
