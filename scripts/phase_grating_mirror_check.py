"""Check registered reciprocal return and guide coefficients against a fresh full-domain control."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.phase_component_comparison import grating_measurement, compare


def check(full,mirror):
    records=[json.loads(path.read_text()) for path in (full,mirror)]
    first,second=[grating_measurement(path) for path in (full,mirror)]
    np.testing.assert_array_equal(first['w'],second['w'])
    if first['case_key']!=second['case_key'] or first['r']!=second['r']:
        raise ValueError('The full and reconstructed acquisitions must use the same physical model and mesh.')
    keys=('source_eh_sha256','source_time_offsets_sha256')
    fingerprints=[{k:record['stages']['reference'][k] for k in keys} for record in records]
    if fingerprints[0]!=fingerprints[1]:
        raise ValueError('Full and mirror controls require identical impressed E/H and offsets.')
    return {'metric':'Fresh full versus mirror-reconstructed solver at one mesh with identical impressed source fields/delays, registered planes and wavelengths.',
            'full_receipt':full.name,'mirror_receipt':mirror.name,'mesh':first['r'],'verified_identical_source':fingerprints[0],
            'full_arrays_sha256':first['sha'],'mirror_arrays_sha256':second['sha'],'registered_planes':second['info'],
            'channels':compare('grating',first['s'],second['s'],first['valid']&second['valid'],
                               first['phase_ok']&second['phase_ok'],wavelengths=first['w'])}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('full',type=Path);p.add_argument('mirror',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.write_text(json.dumps(check(args.full,args.mirror),indent=2,allow_nan=False)+'\n')
