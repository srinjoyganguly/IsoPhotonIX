"""Verify reciprocity of the independently launched finite-height CUDA port pairs."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.calibrate_fdtdx_multiport import save_record
from scripts.phase_publish_columns import stage_performance
from scripts.qualify_port_phase import native_column


def assess(first,reverse_paths,destination):
    plan,original,forward,_,_=native_column(first)
    ports = [port['name'] for port in plan['model']['ports']]
    start=ports.index(original['launch'])
    result={'case':plan['model']['case'],'resolution':plan['resolution'],'native_numerical_sha256':plan['numerical_sha256'],
            'tolerances':{'reciprocity_complex_amplitude':.05,'minimum_amplitude_for_phase':.02},
            'forward_arrays_sha256':original['arrays_sha256'],'pairs':[]}
    for path in reverse_paths:
        reverse_plan,report,reverse,_,hashes=native_column(path)
        if reverse_plan!=plan or report['launch']==original['launch']:
            raise ValueError('Need an independently launched distinct port of the identical native plan.')
        end=ports.index(report['launch'])
        a,b=forward[:,end],reverse[:,start]
        strong=(abs(a)>=.02)&(abs(b)>=.02)
        delta=abs(a-b)
        result['pairs'].append({'forward_input':ports[start],'forward_output':ports[end],
            'reverse_arrays_sha256':report['arrays_sha256'],'frequency_per_um':plan['model']['frequency_per_um'],
            'complex_amplitude_delta':delta.tolist(),'power_delta':abs(abs(a)**2-abs(b)**2).tolist(),
            'signed_phase_delta_rad':[float(v) if good else None for v,good in zip(np.angle(a*b.conj()),strong)],
            'reciprocity_within_0_05':bool(np.all(delta<=.05))})
        performance,acquisition=stage_performance(path,ports[end],report)
        stem=f'v014-{result["case"]}-polygon-cuda{plan["resolution"]}-{ports[end]}'
        report.update(stage_performance=performance,stage_acquisition=acquisition,arrays_file=stem+'-arrays.npz')
        if hashes['arrays.npz']!=report['arrays_sha256']:
            raise ValueError('Reverse column data changed.')
        shutil.copyfile(path/'arrays.npz',destination/report['arrays_file'])
        save_record(destination/(stem+'.json'),report)
    save_record(destination/f'v014-{result["case"]}-polygon-reciprocity.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('forward',type=Path);p.add_argument('reverse',nargs='+',type=Path)
    p.add_argument('--destination',required=True,type=Path)
    args=p.parse_args();assess(args.forward,args.reverse,args.destination)
