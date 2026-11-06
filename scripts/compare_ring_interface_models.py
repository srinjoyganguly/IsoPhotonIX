"""Compare full-vector GPU measurements after changing only the ring's lateral fill rule."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_loaded_ring3d import geometry, lateral_fraction, PORTS
from scripts.compare_loaded_ring3d import ANSYS_SHA, TIDY_SHA
from scripts.compare_loaded_ring_controls import compare as compare_planes

BASE=Path(__file__).resolve().parents[1]/'docs/verification'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_record(path, vendor, mesh, duration, method):
    r=json.loads(path.read_text())
    g=r['physical_model']
    if not (r['measurement_complete'] and r['actual_device']=='cuda:0' and r['vendor']==vendor and
            g.get('ring_interface','legacy')==method):
        raise ValueError(f'Unexpected measurement or interface for {path.name}.')
    if g['resolution']!=mesh or r['duration_um_c']!=duration:
        raise ValueError('A mesh or observation window changed.')
    raw=path.parent/r['arrays_file']
    if digest(raw)!=r['arrays_sha256']:
        raise ValueError('Changed field archive.')
    source=r['vendor_comparison']['source']
    if source.get('attachment_sha256',source.get('notebook_sha256')) != (ANSYS_SHA if vendor=='lumerical' else TIDY_SHA):
        raise ValueError('Unexpected reference revision.')
    with np.load(raw,allow_pickle=False) as a:
        wl=a['wavelength_um']
        bins=a['valid_bins']
        s=a['s']
        if s.shape!=(len(wl),4) or np.sum(bins)!=r['valid_bins']:
            raise ValueError('Wrong measurement dimensions.')
        for col,key in enumerate(PORTS):
            np.testing.assert_allclose(abs(s[:,col])**2,r['series'][vendor][key],rtol=1e-11,atol=1e-12)
        np.testing.assert_array_equal(bins,r['series'][vendor]['valid_bins'])
        np.testing.assert_array_equal(wl,r['series'][vendor]['wavelength_um'])
    return r


def observed_width(vendor,mesh,method):
    g=geometry(vendor,mesh,.65,1.8,method)
    dx=1/mesh
    y=(np.arange(g['shape'][1])-g['shape'][1]/2+.5)*dx
    straight=lateral_fraction(np.array([0]),y,dx,g['radius_um'],g['width_um'],g['bus_y_um'],True,method)
    return float(straight.sum()*dx/2)


def compare_one(base, vendor, mesh):
    duration=3200 if vendor=='lumerical' else 4800
    name=f'ring3d-{vendor}-r{mesh}-t{duration}'
    first=base/f'v012-{name}.json'
    second=base/f'v013-{name}-exact-area.json'
    old=check_record(first,vendor,mesh,duration,'legacy')
    new=check_record(second,vendor,mesh,duration,'exact-area')
    a,b=dict(old['physical_model']),dict(new['physical_model'])
    a.pop('ring_interface',None); b.pop('ring_interface',None)
    if a!=b or old['versions']!=new['versions']:
        raise ValueError('A controlled geometry, runtime or grid changed.')
    power={}
    for port in ('through','drop'):
        def choose(r):
            p=r['vendor_comparison']['channels'][port]['paired_spectrum']
            w=np.array(p['wavelength_um'])
            if len(w)<2 or not np.all(np.diff(w)>0):
                raise ValueError('Invalid qualified comparison grid.')
            return w,np.array(p['published_power']),np.array(p['local_power'])
        w0,v0,l0=choose(old);w1,v1,l1=choose(new)
        common,i,j=np.intersect1d(w0,w1,return_indices=True)
        np.testing.assert_array_equal(v0[i],v1[j])
        k=np.flatnonzero(common==1.55)
        power[port]=dict(common_vendor_wavelengths=len(common),legacy_mae=float(abs(l0[i]-v0[i]).mean()),
                         exact_mae=float(abs(l1[j]-v1[j]).mean()),
                         at_1550=None if len(k)!=1 else dict(legacy=float(l0[i[k[0]]]),exact=float(l1[j[k[0]]]),
                                                            published=float(v0[i[k[0]]])))
    return dict(vendor=vendor,mesh=mesh,legacy_report=first.name,exact_report=second.name,
                legacy_arrays_sha256=old['arrays_sha256'],exact_arrays_sha256=new['arrays_sha256'],
                exact_report_sha256=digest(second),local_source_sha256=new['native_stage_provenance']['device']['provenance']['source_files'],
                versions=new['versions'],device=new['actual_device'],device_name=new['device_name'],
                legacy_and_exact_pass=[old['passed'],new['passed']],qualified_bins=[old['valid_bins'],new['valid_bins']],
                effective_guide_width_um=[observed_width(vendor,mesh,m) for m in ('legacy','exact-area')],
                fsr_nm=[old['measured_drop_fsr_nm'],new['measured_drop_fsr_nm']],published_fsr_nm=old['published_drop_fsr_nm'],
                measured_peaks_um=[old['measured_spacing_pair_um'],new['measured_spacing_pair_um']],
                neff_at_1550=[r['stages']['reference']['source_mode_neff'] for r in (old,new)],
                power=power,fixed_plane_sensitivity=compare_planes(first,second)['port_differences'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--directory',type=Path,default=BASE)
    parser.add_argument('--pair',action='append', choices=('lumerical16','lumerical20','tidy3d16'))
    args=parser.parse_args()
    runs=args.pair or ('lumerical16','lumerical20','tidy3d16')
    rows=[compare_one(args.directory,s[:-2],int(s[-2:])) for s in runs]
    args.output.write_text(json.dumps({'source':'Full-vector FDTDX on explicitly selected CUDA; fixed materials and identical geometry, aperture and duration within each pair. The Tidy3D device is a held-out example. No wavelength shifting or fitting to vendor transmission.', 'pairs':rows},indent=2,allow_nan=False)+'\n')
