"""Convert historic Meep signed-overlap multiport exports to the centre-Ey electric basis.

The original run stays intact. This operation re-expresses signed amplitudes
from measured native projections. It reruns no Maxwell timesteps and fits no
phase. The underlying E/H monitor samples are copied and verified bytewise.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs(a):
    return np.stack((a.real,a.imag),axis=-1).tolist()


def _convert(base,out):
    cfg=json.loads((base/'config.json').read_text())
    meta=json.loads((base/'report.json').read_text())
    if meta.get('phase_convention_version',1)!=1 or 'phase_convention' in meta or meta['solver']!='Meep':
        raise ValueError('Choose a historic Meep signed-overlap run. This conversion applies only once.')
    if digest(base/'arrays.npz')!=meta['arrays_sha256']:
        raise ValueError('Historic array archive differs from its report.')
    model=cfg['model']
    port_names=[p['name'] for p in model['ports']]
    if port_names!=meta['ports']:
        raise ValueError('Port order differs from the saved model.')
    inputs=meta.get('input_ports',port_names)
    signs=np.array([p['inward_sign'] for p in model['ports']])
    if not np.all(np.abs(signs)==1):
        raise ValueError('Each named port requires an unambiguous inward sign.')
    in_signs=np.array([signs[port_names.index(n)] for n in inputs])
    raw_hashes,corrected_hashes,cols={},{},[]
    (out/'config.json').write_bytes((base/'config.json').read_bytes())
    for name,inc_sign in zip(inputs,in_signs):
        old=base/(name+'-monitors.npz')
        raw_hashes[name]=digest(old)
        with np.load(old,allow_pickle=False) as archive:
            arr={k:archive[k] for k in archive}
        if not np.allclose(arr['s'],arr['outgoing']/arr['incident'][:,None],rtol=1e-13,atol=0):
            raise ValueError('Historic column differs from its measured amplitudes.')
        arr['incident']=inc_sign*arr['incident']
        arr['outgoing']=(-signs)[None,:]*arr['outgoing']
        arr['s']=arr['outgoing']/arr['incident'][:,None]
        np.savez_compressed(out/old.name,**arr)
        corrected_hashes[name]=digest(out/old.name)
        with np.load(old,allow_pickle=False) as prior,np.load(out/old.name,allow_pickle=False) as current:
            if any(prior[k].tobytes()!=current[k].tobytes() for k in prior.files if k not in ('s','incident','outgoing')):
                raise RuntimeError('Raw fields or extracted effective indices changed.')
        receipt=json.loads((base/(name+'.json')).read_text())
        receipt.update(phase_convention_version=2,incident=pairs(arr['incident']),outgoing=pairs(arr['outgoing']),s=pairs(arr['s']))
        (out/(name+'.json')).write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        cols.append(arr['s'])
    s=np.stack(cols,axis=-1)
    with np.load(base/'arrays.npz',allow_pickle=False) as old:
        arrays={k:old[k] for k in old.files}
    if not np.allclose(s,arrays['s']*(-signs)[None,:,None]*in_signs[None,None,:],rtol=1e-14,atol=0):
        raise RuntimeError('Matrix re-expression is inconsistent with the original amplitudes.')
    arrays['s']=s
    arrays['incident']=arrays['incident']*in_signs[None,:]
    np.savez_compressed(out/'arrays.npz',**arrays)
    meta.update(input_ports=inputs,full_matrix=len(inputs)==len(port_names),
       phase_convention_version=2,phase_convention='Positive-real centre Ey for each signed-k MPB mode; native overlap multiplied by sign(group velocity).',
       s=pairs(s),measured_power_fraction=(abs(s)**2).tolist(),arrays_sha256=digest(out/'arrays.npz'),
       maximum_column_power_excess=max(0.,float(np.max(meta['power_sums']))-1),
       largest_singular_value_excess=max(0.,float(np.max(meta['largest_singular_value']))-1),
       passed=bool(meta['all_stages_settled'] and all(meta['valid_bins'])),
       phase_reexpression={'method':'Sign(group velocity) conversion of stored native mode projections. Original E/H samples and powers are preserved.',
           'original_arrays_sha256':digest(base/'arrays.npz'),'original_report_sha256':digest(base/'report.json'),
           'original_native_monitor_sha256':raw_hashes,'raw_dft_bytes_unchanged':True},
       native_monitor_sha256=corrected_hashes)
    (out/'report.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')


def convert(original, output):
    original,output=Path(original),Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage=Path(temp)/'corrected'
        stage.mkdir()
        _convert(original,stage)
        if output.exists():
            raise FileExistsError('The output directory appeared during conversion.')
        stage.rename(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    convert(args.original,args.output)
