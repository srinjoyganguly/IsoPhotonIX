"""Register an extended Tidy grating run and check its entire original detector prefix."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.phase_component_comparison import analyze, file_sha


def compare_duration(original, extended):
    reports=[json.loads(p.read_text()) for p in (original,extended)]
    old,new=reports
    settings=dict(new['settings'])
    if (settings.pop('resolve_weak_return',False) is not True or settings!=old['settings']
            or settings['geometry']!='tidy3d' or settings['resolution']!=16
            or settings['grating_domain']!='mirror-y'):
        raise ValueError('Only the weak-return stopping policy may vary in this Tidy3D-geometry control.')
    if any(stage['acquisition']['status']!='fresh' for r in reports for stage in r['stages'].values()):
        raise ValueError('Both stages in each control must be independently acquired.')
    measured=[analyze('grating',[p],None,require_return_windows=True) for p in (original,extended)]
    before,after=[r['native_samples'][0] for r in measured]
    history0,history1=[s['registered_return_time_history'] for s in (before,after)]
    if not history0 or not history1 or len(history1['time_fs'])<=len(history0['time_fs']):
        raise ValueError('The extended device must add measured reciprocal-return windows.')
    count=len(history0['time_fs'])
    prefix={}
    def check(name,a,b):
        a=np.asarray(a);b=np.asarray(b)
        if a.shape!=b.shape:
            raise ValueError(f'Incompatible prefix shape for {name}.')
        prefix[name]={'shape':list(a.shape),'bit_identical':a.dtype==b.dtype and a.tobytes()==b.tobytes(),
                      'max_absolute_difference':float(np.max(abs(a-b)))}
    check('device_sample_times_fs',history0['time_fs'],history1['time_fs'][:count])
    for key in ('registered_real','registered_imag'):
        check(key,history0[key],history1[key][:count])
    with np.load(original.parent/old.get('arrays_file','arrays.npz'),allow_pickle=False) as a, np.load(extended.parent/new.get('arrays_file','arrays.npz'),allow_pickle=False) as b:
        for key in ('ref_upper','ref_upper_x_um','ref_upper_y_um','ref_wavelength_um','incident_power','dev_upper_x_um',
                    'dev_upper_y_um','dev_grid_x_um','dev_grid_y_um','dev_grid_z_um','dev_material_yz','dev_material_xy',
                    'dev_material_xz','dev_mode_e','dev_mode_h','dev_mode_neff','ref_source_E','ref_source_H',
                    'dev_source_E','dev_source_H','ref_source_time_offset_E','ref_source_time_offset_H',
                    'dev_source_time_offset_E','dev_source_time_offset_H'):
            check(key,a[key],b[key])
        for key in ('dev_history_observable','dev_reciprocal_return_history','dev_energy','dev_energy_time_fs'):
            check(key,a[key],b[key][:len(a[key])])
        old_energy=a['dev_energy']
        new_energy=b['dev_energy'][:len(old_energy)]
        if old_energy.dtype!=np.float32 or new_energy.dtype!=np.float32 or np.any(old_energy<0) or np.any(new_energy<0):
            raise ValueError('Expected nonnegative float32 volume-energy reductions.')
        prefix['dev_energy']['max_float32_ulp_difference']=int(np.max(abs(
            old_energy.view(np.uint32).astype(np.int64)-new_energy.view(np.uint32).astype(np.int64))))
        prefix['dev_energy']['relative_l2_difference']=float(np.linalg.norm(new_energy.astype(float)-old_energy)/
                                                            np.linalg.norm(old_energy.astype(float)))
    for key in ('sample_z_um','guide_x_um','target_beam_z_um','target_beam_x_um','target_guide_x_um'):
        if before['planes'][key]!=after['planes'][key]:
            raise ValueError(f'The registered plane or beam anchor differs: {key}.')
    w=before['wavelength_um']
    np.testing.assert_array_equal(after['wavelength_um'],w)
    s0,s1=[np.asarray(s['registered_complex_s_real'])+1j*np.asarray(s['registered_complex_s_imag']) for s in (before,after)]
    joint=(np.asarray(before['phase_ok']) & np.asarray(after['phase_ok']) &
           np.asarray(before['valid_bins'])[:,None] & np.asarray(after['valid_bins'])[:,None])
    delta=np.angle(s1*s0.conj())
    result=measured[1]
    result['duration_control']={
        'original_report':original.name,'original_report_sha256':file_sha(original),
        'original_arrays_sha256':old['arrays_sha256'],'extended_report_sha256':file_sha(extended),
        'source_and_detector_prefix_bit_identical':all(r['bit_identical'] for k,r in prefix.items() if k!='dev_energy'),
        'all_checked_prefix_data_bit_identical':all(r['bit_identical'] for r in prefix.values()),
        'original_time_fs':history0['time_fs'][-1],'extended_time_fs':history1['time_fs'][-1],
        'original_windows':count,'extended_windows':len(history1['time_fs']),'prefix_checks':prefix,
        'original_return_qualification':history0['qualification'],'extended_return_qualification':history1['qualification'],
        'registered_final_change':{
            'channel_order':['reflection','guide'],'wavelength_um':w,'complex_absolute':abs(s1-s0).tolist(),
            'power_extended_minus_original_pp':(100*(abs(s1)**2-abs(s0)**2)).tolist(),
            'joint_reportable_phase':joint.tolist(),'signed_phase_change_rad':[
                [float(v) if ok else None for v,ok in zip(values,mask,strict=True)] for values,mask in zip(delta,joint,strict=True)]},
        'scope':'The source and detector prefix is checked independently from parallel volume-energy reductions. A matched detector prefix isolates the added observation time for the same registered signals. S11 phase is reportable only at a duration that independently passes the registered amplitude and temporal gates. Signed phase changes require both endpoints to qualify. Public Tidy3D comparison covers guide power.'}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original','extended','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    result=compare_duration(args.original,args.extended)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not result['duration_control']['source_and_detector_prefix_bit_identical']:
        raise SystemExit('Inspect the copied sources and measured time histories before attributing changes to duration.')
