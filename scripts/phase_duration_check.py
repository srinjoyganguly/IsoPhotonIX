"""Check ring phase/returns versus observation time on a verified common native prefix."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.phase_component_comparison import compare, native_data, ring_measurement


def compare_duration(short,long):
    first,a=native_data(short,True);second,b=native_data(long,True)
    with a,b:
        if first['physical_model']!=second['physical_model'] or second['duration_um_c']<=first['duration_um_c']:
            raise ValueError('Need a longer acquisition of the identical ring model.')
        for key in ('time_um_c','projections','ring_Hz'):
            x=a['device_'+key];y=b['device_'+key]
            if len(y)<=len(x) or not np.array_equal(x,y[:len(x)]):
                raise ValueError('The shorter evolution is not an exact prefix of the longer native trace.')
        for key in ('reference_time_um_c','reference_projections'):
            np.testing.assert_array_equal(a[key],b[key])
        prefix_samples=len(a['device_time_um_c'])
    x,y=ring_measurement(short,allow_partial=True),ring_measurement(long,allow_partial=True)
    np.testing.assert_array_equal(x['w'],y['w'])
    band=(x['w']>=1.5)&(x['w']<=1.6)
    good=x['valid']&y['valid']&band
    result={'vendor':x['vendor'],'resolution':x['r'],'native_durations_um_c':[first['duration_um_c'],second['duration_um_c']],
            'native_run_passed':[x['native_passed'],y['native_passed']],'native_arrays_sha256':[x['sha'],y['sha']],
            'verified_exact_prefix_samples':prefix_samples,'common_qualified_inband_bins':int(good.sum()),
            'newly_qualified_wavelength_um':y['w'][~x['valid']&y['valid']].tolist(),
            'channels':compare('ring',x['s'],y['s'],good,x['phase_ok']&y['phase_ok'])}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('short',type=Path);p.add_argument('long',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.write_text(json.dumps(compare_duration(args.short,args.long),indent=2,allow_nan=False)+'\n')
