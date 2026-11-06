"""Withhold ringing-device S when the archived passive time evolution has runaway fields."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def growth_diagnostic(report,archive):
    field=np.asarray(archive['device_ring_Hz'])
    time=np.asarray(archive['device_time_um_c'])
    if field.ndim!=2 or field.shape!=(len(time),3) or len(time)<8 or not np.isfinite(field).all() or not np.all(np.diff(time)>0):
        raise ValueError('Need finite three-probe records and strictly increasing native samples.')
    quarters=[]
    for selection in np.array_split(np.arange(len(time)),4):
        chunk=field[selection]
        quarters.append({'time_um_c':[float(time[selection[0]]),float(time[selection[-1]])],
                         'max_abs_hz':abs(chunk).max(axis=0).tolist(),
                         'rms_hz':np.sqrt(np.mean(np.abs(np.asarray(chunk,dtype=np.complex128))**2,axis=0)).tolist()})
    second,third,last=(np.asarray(x['rms_hz']) for x in quarters[1:])
    # A conservative rejection for a passive pulse followed by accelerating
    # multi-probe growth. Passing it by itself is not a stability proof.
    late_growth=bool(np.all(third>10*second) and np.all(last>10*third))
    excess=report['max_collected_port_power']>1.05
    return {'quarters':quarters,'all_three_probes_grow_tenfold_in_each_of_last_two_transitions':late_growth,
            'maximum_collected_port_power':report['max_collected_port_power'],
            'reject_scattering':bool(late_growth and excess)}


def inspect(path):
    r=json.loads(path.read_text())
    arrays=path.parent/r.get('arrays_file','arrays.npz')
    sha=hashlib.sha256(arrays.read_bytes()).hexdigest()
    if r['case_id']!='loaded-ring-3d' or r['actual_device']!='cuda:0' or sha!=r['arrays_sha256']:
        raise ValueError('Require a source-bound native NVIDIA ring archive.')
    with np.load(arrays,allow_pickle=False) as a:
        diagnostic=growth_diagnostic(r,a)
        return {'receipt':path.name,'native_arrays_sha256':sha,'native_run_passed':r['passed'],'duration_um_c':r['duration_um_c'],
                'nominal_valid_bins':a['valid_bins'].tolist(),'nominal_qualified_wavelength_um':a['wavelength_um'][a['valid_bins']].tolist(),
                'native_bin_count':len(a['wavelength_um']),'diagnostic':diagnostic,
                'interpretation':'Withhold all phase and weak returns from a globally unstable acquisition, including wavelengths that passed the original per-bin mask.' if diagnostic['reject_scattering'] else 'No large late multisite growth with excess collection detected; other qualification gates still apply.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();r=inspect(args.receipt)
    if args.output:
        args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print('reject' if r['diagnostic']['reject_scattering'] else 'continue')
