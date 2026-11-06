"""Reconstruct duration stability separately from launch, straight-lead and passivity checks."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_loaded_ring3d import directional_spectrum, PORTS
from scripts.compare_ring_interface_models import digest


def stage_arrays(data,label):
    return {key.removeprefix(label+'_'):data[key] for key in data.files if key.startswith(label+'_')}


def prefixes(data,limit):
    use=data['time_um_c']<=limit+1e-9
    if use.sum()<2:
        raise ValueError('Requested optical interval precedes available projections.')
    return dict(data,time_um_c=data['time_um_c'][use],projections=data['projections'][use])


def check_window(reference,device,wl,end):
    r=directional_spectrum(reference,1.)
    short=prefixes(device,end)
    a=directional_spectrum(short,1.)
    earlier=directional_spectrum(short,.8)
    incident=r[:,0,0]
    def normalized(values):
        return np.stack((values[:,0,1]-r[:,0,1],values[:,1,0],values[:,2,1],values[:,3,0]),axis=1)/incident[:,None]
    s=normalized(a)
    se=normalized(earlier)
    p=abs(s)**2
    changes=abs(p-abs(se)**2)
    straight=abs(r[:,1,0]/incident)**2
    launch=a[:,0,0]/incident
    gates=dict(illuminated=abs(incident)>abs(incident).max()*.03,straight_lead=abs(straight-1)<.04,
               collected_port_power=p.sum(axis=1)<=1.05,early_window=changes.max(axis=1)<.05,
               launch_magnitude=abs(abs(launch)-1)<.05,launch_phase=abs(np.angle(launch))<.06)
    qualified=np.logical_and.reduce(list(gates.values()))
    bad=[]
    for index in np.flatnonzero(~qualified):
        bad.append(dict(wavelength_nm=float(1000*wl[index]),gates_failed=[key for key,ok in gates.items() if not ok[index]],
                        largest_power_change=float(changes[index].max()),
                        responsible_port=PORTS[int(np.argmax(changes[index]))]))
    return dict(nominal_window_um_c=end,sampled_stop_um_c=float(short['time_um_c'][-1]),total_bins=len(wl),
                qualified_bins=int(qualified.sum()),failed_bins=bad,
                maximum_80_percent_power_change=float(changes.max()),early_window_fraction=.8), qualified, s


def load(path):
    report=json.loads(path.read_text())
    archive=path.parent/report['arrays_file']
    if digest(archive)!=report['arrays_sha256']:
        raise ValueError('Unexpected ring waveform archive.')
    with np.load(archive,allow_pickle=False) as data:
        return report, stage_arrays(data,'reference'),stage_arrays(data,'device'),data['wavelength_um'],data['s'],data['valid_bins']


def diagnose(short_path,long_path=None):
    short,reference,device,wl,s,mask=load(short_path)
    summary, checked, rebuilt=check_window(reference,device,wl,short['duration_um_c'])
    np.testing.assert_array_equal(mask,checked)
    np.testing.assert_allclose(s,rebuilt,rtol=1e-9,atol=1e-11)
    result=dict(source_report=short_path.name,short_arrays_sha256=short['arrays_sha256'],short_window=summary)
    if long_path:
        long,ref2,dev2,wl2,s2,mask2=load(long_path)
        if (short['physical_model']!=long['physical_model'] or short['versions']!=long['versions'] or
                short['native_stage_provenance']['reference']['checkpoint_signature']!=long['native_stage_provenance']['reference']['checkpoint_signature']):
            raise ValueError('Compared observation windows have different physical models, environment or straight reference.')
        np.testing.assert_array_equal(wl,wl2)
        for key in ('source_E','source_H','source_offset_E','source_offset_H','permittivity_cross_section','inverse_xy_midplane'):
            np.testing.assert_array_equal(device[key],dev2[key])
        steps=len(device['time_um_c'])
        np.testing.assert_array_equal(device['time_um_c'],dev2['time_um_c'][:steps])
        short_peak=np.max(abs(device['projections']))
        overlap=np.max(abs(device['projections']-dev2['projections'][:steps]))
        result.update(long_report=long_path.name,long_arrays_sha256=long['arrays_sha256'],
                      long_acquisitions=long['checkpoint_identity']['invocation_acquisition'],
                      maximum_shared_prefix_projection_discrepancy_relative_to_peak=float(overlap/short_peak),
                      maximum_shared_prefix_probe_discrepancy=float(np.max(abs(device['ring_Hz']-dev2['ring_Hz'][:steps]))),
                      same_reused_reference=long['native_stage_provenance']['reference']['checkpoint_signature'])
        result['long_windows']=[]
        for endpoint in (4800.,long['duration_um_c']):
            row,tmask,values=check_window(ref2,dev2,wl2,endpoint)
            result['long_windows'].append(row)
        np.testing.assert_array_equal(mask2,tmask)
        np.testing.assert_allclose(s2,values,rtol=1e-9,atol=1e-11)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short',type=Path,required=True)
    parser.add_argument('--long',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(diagnose(args.short,args.long),indent=2,allow_nan=False)+'\n')
