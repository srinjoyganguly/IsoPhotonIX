"""Isolate the wall-update effect at fixed 20-cell geometry and source registration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.phase_component_comparison import compare, file_sha, mode_gauge_valid, native_data


def control(directory):
    names=('wall-tensor','reciprocal-wall')
    paths=[directory/f'v014-ring3d-lumerical-r20-t3200-{name}.json' for name in names]
    phase_paths=[directory/'v014-ring-phase-wall.json',directory/'v014-ring-phase-reciprocal.json']
    receipts=[json.loads(p.read_text()) for p in paths]
    if dict(receipts[0]['physical_model'],ring_interface=names[1]) != receipts[1]['physical_model']:
        raise ValueError('The wall intervention requires identical physical geometry.')
    fixed={}
    _,old=native_data(paths[0])
    _,new=native_data(paths[1])
    with old,new:
        required=['reference_time_um_c','reference_projections','reference_source_E','reference_source_H',
                  'reference_mode_neff','device_source_E','device_source_H','device_inverse_xy_midplane',
                  'device_permittivity_cross_section','basis_wavelength_um']
        for field in required:
            a,b=old[field],new[field]
            first=hashlib.sha256(a.tobytes()).hexdigest()
            second=hashlib.sha256(b.tobytes()).hexdigest()
            if first != second or a.shape!=b.shape or a.dtype!=b.dtype:
                raise ValueError(f'The fixed-input comparison differs in {field}.')
            fixed[field]={'shape':list(a.shape),'dtype':a.dtype.str,'sha256':first}
    rows=[]
    for receipt,path in zip(receipts,phase_paths,strict=True):
        panel=json.loads(path.read_text())
        row=next(s for s in panel['native_samples'] if s['resolution']==20)
        if row['arrays_sha256']!=receipt['arrays_sha256']:
            raise ValueError('Registered ports do not refer to this acquisition.')
        if not all(mode_gauge_valid(m) for m in row['central_vendor_mode_check']['port_overlaps']):
            raise ValueError('A registered port has an unverified gauge.')
        rows.append(row)
    w=np.array(rows[0]['wavelength_um'])
    np.testing.assert_array_equal(w,rows[1]['wavelength_um'])
    if rows[0]['planes']!=rows[1]['planes']:
        raise ValueError('The two acquisitions require identical transport to the filed port planes.')
    a,b=[r['single_central_basis_band'] for r in rows]
    signals=[np.asarray(row['registered_native_real'])+1j*np.asarray(row['registered_native_imag']) for row in (a,b)]
    valid=np.asarray(a['native_valid_bins']) & np.asarray(b['native_valid_bins']) & (w>=1.5) & (w<=1.6)
    phases=np.asarray(a['native_phase_ok']) & np.asarray(b['native_phase_ok'])
    spectral=compare('ring',signals[0],signals[1],valid,phases,wavelengths=w)
    observed=[[p['wavelength_um']*1000 for p in r['qualified_peaks'] if 1.50<=p['wavelength_um']<=1.59]
              for r in receipts]
    if any(len(m)!=3 for m in observed):
        raise ValueError('The line-order comparison requires exactly three interior peaks on each operator.')
    return {'case':'Lumerical ring, 20 cells/µm, 3200 µm/c','interfaces':list(names),
            'source_report_sha256':[file_sha(p) for p in paths],'source_arrays_sha256':[r['arrays_sha256'] for r in receipts],
            'registered_report_sha256':[file_sha(p) for p in phase_paths],
            'byte_identical_fields':fixed,'ordered_peak_positions_nm':observed,
            'reciprocal_minus_original_peak_nm':(np.asarray(observed[1])-observed[0]).tolist(),
            'common_valid_in_band_samples':int(valid.sum()),'fixed_basis_port_comparison':spectral,
            'scope':'Both native 20-cell acquisitions pass. The raw input wall map, port sections, launching fields and straight-reference time signals are byte-identical. Peak pairing uses increasing wavelength in the interior interval. This intervention measures sensitivity to the update in the bent, coupled device. It does not separate the contributions of curvature, directional staggering and coupler phase or identify angular mode order.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('docs/verification'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(control(args.directory),indent=2,allow_nan=False)+'\n')
