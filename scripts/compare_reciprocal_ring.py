"""Report same-geometry stability and unshifted reference errors for the reciprocal ring control."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_ring_interface_models import BASE, check_record, digest
from scripts.calibrate_loaded_ring3d import PORTS
from scripts.diagnose_ring_stability import rejection_for


def compare(directory,mesh):
    paths=[directory/f'v014-ring3d-lumerical-r{mesh}-t3200-{interface}.json' for interface in ('wall-tensor','reciprocal-wall')]
    old,new=(check_record(p,'lumerical',mesh,3200,interface) for p,interface in zip(paths,('wall-tensor','reciprocal-wall')))
    g0,g1=old['physical_model'],new['physical_model']
    if dict(g0,ring_interface='reciprocal-wall')!=g1 or new['versions']!=old['versions']:
        raise ValueError('Matched wall controls differ in geometry, grid or runtime.')
    with np.load(directory/old['arrays_file'],allow_pickle=False) as a, np.load(directory/new['arrays_file'],allow_pickle=False) as b:
        for name in ('device_inverse_xy_midplane','device_permittivity_cross_section','reference_permittivity_cross_section'):
            np.testing.assert_array_equal(a[name],b[name])
        source_difference={}
        for name in ('reference_source_E','reference_source_H'):
            delta=float(np.max(abs(a[name]-b[name]))/np.max(abs(a[name])))
            if delta>2e-6: raise ValueError('Straight source differs between interface controls.')
            source_difference[name]=delta
        wl=b['wavelength_um']; center=int(np.argmin(abs(wl-1.55)))
        valid=b['valid_bins']
        power=abs(b['s'])**2
        center_powers={name:float(power[center,i]) for i,name in enumerate(PORTS)} if valid[center] else None
        old_rejection=rejection_for(paths[0])
        old_center_allowed=old_rejection is None and bool(a['valid_bins'][center])
        old_power={name:float(abs(a['s'][center,i])**2) for i,name in enumerate(PORTS)} if old_center_allowed else None
    channels={}
    for name,paired in new['vendor_comparison']['channels'].items():
        if not paired['qualified_sample_count']:
            channels[name]=dict(samples=0)
            continue
        spectrum=paired['paired_spectrum']
        w=np.asarray(spectrum['wavelength_um'])
        hit=np.flatnonzero(np.abs(w-1.55)<1e-12)
        channels[name]=dict(samples=len(w),mean_absolute_power_error=paired['mean_absolute_power_difference'],
                            filed_power_at_1550=float(spectrum['published_power'][hit[0]]) if len(hit) else None)
    return dict(mesh=mesh,old_report=paths[0].name,corrected_report=paths[1].name,
                report_sha256=digest(paths[1]),old_arrays_sha256=old['arrays_sha256'],corrected_arrays_sha256=new['arrays_sha256'],
                shape=g1['shape'],source_relative_max_differences=source_difference,
                raw_interface_midplane_and_source_sections_identical=True,
                old_spectrum_rejected=bool(old_rejection),old_qualified_power_at_1550=old_power,
                corrected_power_at_1550=center_powers,channels=channels,
                native_all_bin_pass=new['passed'],valid_bins=new['valid_bins'],
                stability=new['stability'],operator=new['stages']['device']['reciprocal_operator'],
                fsr_nm=new['measured_drop_fsr_nm'],published_fsr_nm=new['published_drop_fsr_nm'],
                reference_linewidths_and_local_peak_offsets=new['vendor_comparison']['published_drop_resonances'],
                acquisition=new['checkpoint_identity']['invocation_acquisition'],
                timing={k:v['performance'] for k,v in new['stages'].items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=BASE)
    p.add_argument('--mesh',action='append',type=int,choices=(20,24));p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[compare(args.directory,mesh) for mesh in (args.mesh or (20,24))]
    args.output.write_text(json.dumps({'comparisons':rows},indent=2,allow_nan=False)+'\n')
