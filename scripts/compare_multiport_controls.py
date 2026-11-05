"""Compare finite-height Meep modal results after changing one numerical setting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_run(directory):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    config = json.loads((directory/'config.json').read_text())
    archive = directory/'arrays.npz'
    if report['arrays_sha256'] != hashlib.sha256(archive.read_bytes()).hexdigest():
        raise ValueError('Numerical archive digest differs from the recorded measurement.')
    with np.load(archive,allow_pickle=False) as saved:
        matrix=saved['s']
        freqs=saved['frequency_per_um']
    ports=report['ports']
    incident=report.get('input_ports', ports)
    if matrix.shape != (len(freqs),len(ports),len(incident)) or len(set(incident))!=len(incident):
        raise ValueError('Stored matrix shape or input-port names are inconsistent.')
    encoded=np.array(report['s'])
    if not np.array_equal(matrix,encoded[...,0]+1j*encoded[...,1]):
        raise ValueError('Saved matrix differs from displayed report.')
    if len(report['valid_bins']) != len(freqs):
        raise ValueError('Invalid qualified-bin array.')
    return report,config,matrix,freqs,incident


def compare(baseline, control, phase_floor=.01):
    a,ac,sa,fa,ins_a = read_run(baseline)
    b,bc,sb,fb,ins_b = read_run(control)
    if (a['solver'],a['version'],a['actual_device'],a['case'],a['ports']) != (
            b['solver'],b['version'],b['actual_device'],b['case'],b['ports']):
        raise ValueError('Solver/version, target, case and physical ports must match.')
    if not np.array_equal(fa,fb) or not set(ins_b).issubset(ins_a):
        raise ValueError('Frequencies or incident-port coverage are incompatible.')
    # These quantities define the physical circuit, excitation and geometry.
    fixed=('polygons','ports','core_index','cladding_index','height','width','frequency_per_um',
           'pulse_fwidth','pulse_cutoff','straight_extent')
    if any(ac['model'][k] != bc['model'][k] for k in fixed):
        raise ValueError('Physical material, excitation or geometry changed.')
    def settings(c):
        m=c['model']
        return {'mesh':c['resolution'],'pml':m['pml_um'],'aperture':m['aperture'],
                'ringdown':[c['max_after_source_time'],c.get('minimum_ringdown',0.)]}
    before,after=settings(ac),settings(bc)
    changed=[k for k in before if before[k]!=after[k]]
    if len(changed)!=1:
        raise ValueError('A single numerical control must change per comparison.')
    va=np.asarray(a['valid_bins'],dtype=bool); vb=np.asarray(b['valid_bins'],dtype=bool)
    valid=va&vb
    if not valid.any():
        raise ValueError('No common qualified spectral bins.')
    rows=[]
    for column,name in enumerate(ins_b):
        for i,output in enumerate(a['ports']):
            base=sa[valid,i,ins_a.index(name)]; candidate=sb[valid,i,column]
            phase_mask=(np.abs(base)>=phase_floor)&(np.abs(candidate)>=phase_floor)
            phase=np.abs(np.angle(candidate[phase_mask]*base[phase_mask].conjugate()))
            rows.append({'incident':name,'outgoing':output,
                'max_absolute_complex_change':float(max(abs(candidate-base))),
                'max_absolute_power_change':float(max(abs(abs(candidate)**2-abs(base)**2))),
                'max_principal_phase_change_rad':float(phase.max()) if phase.size else None})
    return {'case':a['case'],'solver':a['solver'],'actual_device':a['actual_device'],'control':changed[0],
            'before':before[changed[0]],'after':after[changed[0]],'inputs_measured':ins_b,'valid_bins':valid.tolist(),
            'phase_amplitude_floor':phase_floor,'entries':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path)
    p.add_argument('control',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=compare(args.baseline,args.control)
    with args.output.open('x') as output:
        json.dump(result,output,indent=2,allow_nan=False)
        output.write('\n')
