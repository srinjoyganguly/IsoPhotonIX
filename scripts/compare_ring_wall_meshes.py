"""Compare identical finite-height ring targets on adjacent wall-tensor meshes and common planes."""
import argparse
import json
from pathlib import Path

import numpy as np
from scripts.calibrate_loaded_ring3d import PORTS
from scripts.compare_ring_interface_models import BASE, check_record
from scripts.compare_loaded_ring_controls import read


def compare(directory,vendor,first,last):
    duration=3200 if vendor=='lumerical' else 4800
    files=[directory/f'v014-ring3d-{vendor}-r{m}-t{duration}-wall-tensor.json' for m in (first,last)]
    reports=[check_record(path,vendor,m,duration,'wall-tensor') for path,m in zip(files,(first,last))]
    physical=('radius_um','width_um','height_um','edge_gap_um','bus_y_um','top_index','silicon_index','substrate_index')
    if any(reports[0]['physical_model'][k]!=reports[1]['physical_model'][k] for k in physical):
        raise ValueError('Physical ring dimensions or indices changed across meshes.')
    fields=[read(f) for f in files]
    wl=fields[0][1]
    np.testing.assert_array_equal(wl,fields[1][1])
    if fields[0][3]!=fields[1][3]:
        raise ValueError('The corrected ports must be at common physical x planes.')
    s1,s2=(v[2] for v in fields)
    qualified=[np.asarray(r['series'][vendor]['valid_bins']) for r in reports]
    mask=qualified[0] & qualified[1] & (wl>=1.5-1e-12) & (wl<=1.6+1e-12)
    if not mask.any():
        raise ValueError('No jointly qualified bins in the vendor band.')
    ports={}
    for i,name in enumerate(PORTS):
        p1,p2=abs(s1[:,i])**2,abs(s2[:,i])**2
        phase=mask & (np.minimum(p1,p2)>=.01)
        angles=np.angle(s2[phase,i]*s1[phase,i].conj())
        k=int(np.argmin(abs(wl-1.55)))
        ports[name]=dict(mean_mesh_power_change=float(abs(p2[mask]-p1[mask]).mean()),
            max_mesh_power_change=float(abs(p2[mask]-p1[mask]).max()),
            max_mesh_complex_s_change=float(abs(s2[mask,i]-s1[mask,i]).max()),
            phase_samples_at_least_one_percent_power=int(phase.sum()),
            max_wrapped_phase_change_rad=float(max(abs(angles))) if len(angles) else None,
            rms_wrapped_phase_change_rad=float(np.sqrt(np.mean(angles**2))) if len(angles) else None,
            at_1550_power=[float(p1[k]),float(p2[k])] if mask[k] else None)
        if name in reports[0]['vendor_comparison']['channels']:
            spectra=[r['vendor_comparison']['channels'][name]['paired_spectrum'] for r in reports]
            w,a,b=np.intersect1d(*[row['wavelength_um'] for row in spectra],return_indices=True)
            ref1,ref2=(np.asarray(row['published_power'])[indices] for row,indices in zip(spectra,(a,b)))
            np.testing.assert_array_equal(ref1,ref2)
            ports[name]['common_vendor_samples']=len(w)
            ports[name]['mean_vendor_absolute_power_error']=[float(np.mean(abs(np.asarray(row['local_power'])[indices]-ref1)))
                                                            for row,indices in zip(spectra,(a,b))]
            center=np.flatnonzero(abs(w-1.55)<1e-12)
            ports[name]['published_power_at_1550']=float(ref1[center[0]]) if len(center) else None
    return dict(vendor=vendor,meshes=[first,last],source_reports=[p.name for p in files],
                source_archive_sha256=[r['arrays_sha256'] for r in reports],method='Constant indices, wall-tensor Yee sampling.',
                native_all_bins_passed=[r['passed'] for r in reports],native_bins=[r['valid_bins'] for r in reports],
                jointly_qualified_bins_1500_to_1600=int(mask.sum()),joint_wavelength_um=wl[mask].tolist(),
                nominal_x_plane_absolute_um=fields[0][3],snapped_x_planes_absolute_um=[v[4] for v in fields],
                actual_pml_um=[r['physical_model']['pml']/r['physical_model']['resolution'] for r in reports],
                actual_square_port_aperture_um=[r['physical_model']['aperture_um'] for r in reports],
                field_projection='Positive Ey gauge; common nominal port planes. The launched lower bus uses measured straight transfer; the unexcited upper bus uses the discrete modal propagation estimate. Phase cut requires at least 1% power at both meshes.',
                channels=ports,fsr_nm=[r['measured_drop_fsr_nm'] for r in reports],published_fsr_nm=reports[0]['published_drop_fsr_nm'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=BASE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',action='append',choices=('l16-20','l20-24','t16-20'))
    args=parser.parse_args()
    cases=args.case or ['l16-20','l20-24','t16-20']
    comparisons=[compare(args.directory,'lumerical' if c[0]=='l' else 'tidy3d',*map(int,c[1:].split('-'))) for c in cases]
    args.output.write_text(json.dumps({'adjacent_mesh_controls':comparisons},indent=2,allow_nan=False)+'\n')
