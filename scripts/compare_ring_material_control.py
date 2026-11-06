"""Compare a passive ADE ring with its same-grid constant-index exact-area control."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.compare_loaded_ring3d import ANSYS_SHA


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(path):
    r=json.loads(path.read_text())
    a=path.parent/r['arrays_file']
    if digest(a)!=r['arrays_sha256'] or r['actual_device']!='cuda:0':
        raise ValueError('The native NVIDIA run or its array hash failed verification.')
    if r['vendor_comparison']['source']['attachment_sha256']!=ANSYS_SHA or r['vendor']!='lumerical':
        raise ValueError('Expected the published Lumerical S file.')
    with np.load(a,allow_pickle=False) as data:
        for i,port in enumerate(('reflection','through','drop','add_side')):
            np.testing.assert_allclose(abs(data['s'][:,i])**2,r['series']['lumerical'][port],atol=1e-12,rtol=1e-11)
        np.testing.assert_array_equal(data['valid_bins'],r['series']['lumerical']['valid_bins'])
        snapshots=data[r['field_preview']['array']]
        if not snapshots.size or not np.isfinite(snapshots).all() or np.ptp(snapshots)==0:
            raise ValueError('Missing native field snapshots.')
        if r['physical_model'].get('ring_material')=='palik-ade':
            np.testing.assert_array_equal(data['basis_wavelength_um'],[1.5,1.525,1.55,1.575,1.6])
            for key in ('source_E','source_H','source_offset_E','source_offset_H','projection_wavelengths_um'):
                np.testing.assert_array_equal(data['reference_'+key],data['device_'+key])
            w=data['wavelength_um']
            if data['valid_bins'][(w<1.5)|(w>1.6)].any():
                raise ValueError('Out-of-material-band wavelength unexpectedly qualified.')
    return r


def compare(constant, dispersive):
    old,new=read(constant),read(dispersive)
    g0,g1=dict(old['physical_model']),dict(new['physical_model'])
    if g1.pop('ring_material')!='palik-ade' or g1.pop('material_band_um')!=[1.5,1.6]:
        raise ValueError('Expected band-limited ADE model.')
    source_input=g1.pop('material_sha256')
    if g0!=g1 or g0['ring_interface']!='exact-area' or old['duration_um_c']!=new['duration_um_c']:
        raise ValueError('Expected the same ring, mesh, aperture and observation window.')
    stages=new['stages']
    if stages['reference']['source_drive']!=stages['device']['source_drive']:
        raise ValueError('Source temporal drive changed between the paired acquisitions.')
    metrics={}
    for channel in ('drop','through'):
        def values(r):
            row=r['vendor_comparison']['channels'][channel]['paired_spectrum']
            return [np.array(row[k]) for k in ('wavelength_um','published_power','local_power')]
        a,b,c=values(old);x,y,z=values(new)
        w,i,j=np.intersect1d(a,x,return_indices=True)
        np.testing.assert_array_equal(b[i],y[j])
        k=np.flatnonzero(w==1.55)
        metrics[channel]={'jointly_compared_wavelengths':len(w),'constant_mae':float(abs(c[i]-b[i]).mean()),
                          'ade_mae':float(abs(z[j]-y[j]).mean()),'at_1550':None if len(k)!=1 else
                          {'constant':float(c[i[k[0]]]),'ade':float(z[j[k[0]]]),'published':float(b[i[k[0]]])}}
    return {'constant_receipt':constant.name,'constant_receipt_sha256':digest(constant),'constant_arrays_sha256':old['arrays_sha256'],
            'ade_receipt':dispersive.name,'ade_receipt_sha256':digest(dispersive),'ade_arrays_sha256':new['arrays_sha256'],
            'material_input_sha256':source_input,'numerical_pass':[old['passed'],new['passed']],
            'ade_allowed_bins':new['material_band_bins'],'ade_qualified_bins':new['valid_bins'],'actual_device':new['actual_device'],
            'fsr_nm':[old['measured_drop_fsr_nm'],new['measured_drop_fsr_nm']],'vendor_fsr_nm':new['published_drop_fsr_nm'],
            'central_peaks_um':[old['measured_spacing_pair_um'],new['measured_spacing_pair_um']],
            'source_mode_neff_1550':[r['stages']['reference']['source_mode_neff'] for r in (old,new)],
            'power':metrics,'max_held_out_mixed_epsilon_error':max(max(s['material_validation']['maximum_cell_epsilon_errors'][i] for i in [1,3]) for s in stages.values()),
            'source_profile_and_temporal_hashes':stages['reference']['source_drive'],
            'scope':'No spectral alignment or fit to vendor transmission. The source is impressed, frozen at the fitted carrier, and calibrated with a separate dispersive straight-reference pulse. The five modal overlaps use their frequency-specific discrete ADE permittivity.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('constant',type=Path)
    p.add_argument('dispersive',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.write_text(json.dumps(compare(a.constant,a.dispersive),indent=2,allow_nan=False)+'\n')
