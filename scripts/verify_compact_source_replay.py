"""Check the corrected exported air-side source against native GPU fields and the recorded baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]/'docs/verification'


def verify(directory):
    raw=(directory/'report.json').read_bytes()
    run=json.loads(raw)
    m=run['settings']
    name,mesh=m['geometry'],m['resolution']
    if name not in ('compact','compact-narrow') or mesh!=12:
        raise ValueError('A bundled compact 12-cell study is required.')
    reference=BASE/f'v012-grating3d-{name}-r12.json'
    original=json.loads(reference.read_text())
    if not original['passed'] or not run['passed'] or run['actual_device']!='cuda:0':
        raise ValueError('Require a complete and qualified GPU measurement.')
    expected=np.array(original['result']['beam_to_guide'])
    current=np.array(run['result']['beam_to_guide'])
    np.testing.assert_allclose(current,expected,atol=2e-6,rtol=0)
    source=run['stages']['reference']
    device=run['stages']['device']
    for key in ('source_eh_sha256','source_time_offsets_sha256'):
        if source[key]!=device[key]:
            raise ValueError(f'The two stages used different {key}.')
    before=reference.parent/original['arrays_file']
    after=directory/'arrays.npz'
    for path,sha in ((before,original['arrays_sha256']),(after,run['arrays_sha256'])):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError(f'Numerical archive digest changed: {path.name}')
    with np.load(before,allow_pickle=False) as a,np.load(after,allow_pickle=False) as b:
        np.testing.assert_array_equal(a['valid'],b['valid'])
        for k in ('source_E','source_H','source_time_offset_E','source_time_offset_H'):
            # Both independent launch arrays are part of the native evidence.
            np.testing.assert_array_equal(b['ref_'+k],b['dev_'+k])
        x,y=a['dev_center_fields'],b['dev_center_fields']
        if x.shape!=y.shape or not np.all(np.isfinite(y)):
            raise ValueError('Invalid recorded field plane.')
        field_err=float(np.linalg.norm(x-y)/np.linalg.norm(x))
        if field_err>=1e-4:
            raise ValueError('Measured field plane moved beyond the predeclared relative L2 tolerance.')
    fields=('performance','acquisition','source_eh_sha256','source_time_offsets_sha256','versions','arrays_sha256','native_worker_sha256')
    stages={role:{k:s[k] for k in fields if k in s} for role,s in run['stages'].items()}
    if any(s.get('acquisition',{}).get('status')!='fresh' or s['performance']['cuda_event_elapsed_seconds']<=0 for s in stages.values()):
        raise ValueError('Require separately timed fresh reference and device acquisitions.')
    return {'passed':True,'case':'grating-3d','actual_device':run['actual_device'],'settings':m,'stages':stages,
            'baseline':reference.name,'baseline_native_arrays_sha256':original['arrays_sha256'],'native_report_sha256':hashlib.sha256(raw).hexdigest(),
            'native_arrays_sha256':run['arrays_sha256'],'wavelength_um':run['result']['wavelength_um'],'old_power':expected.tolist(),'new_power':current.tolist(),
            'maximum_absolute_power_difference':float(max(abs(current-expected))),'field_relative_l2':field_err,
            'source_and_time_offset_hashes_match':True,'entry_point':'Extracted calibrated Python export, fresh reference and device on local NVIDIA.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(verify(args.run),indent=2,allow_nan=False)+'\n')
