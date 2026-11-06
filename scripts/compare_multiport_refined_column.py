"""Compare one genuinely excited column from Meep and Yee-aligned FDTDX at the same numerical resolution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compare_multiport_controls import compare, read_run


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def unpack(pairs):
    array=np.asarray(pairs)
    return array[...,0]+1j*array[...,1]


def encode(values):
    return np.stack([values.real,values.imag],axis=-1).tolist()


def compare_column(meep,fdtdx,incident,previous=None):
    m,mc,sm,frequency,inputs=read_run(meep)
    d=json.loads((fdtdx/'report.json').read_text());dc=json.loads((fdtdx/'run.json').read_text())
    if (m['case']!=d['case'] or m['model']!=dc['model'] or m['resolution']!=d['resolution'] or
            m['resolution']!=dc['resolution'] or m['ports']!=d['port_order']):
        raise ValueError('Native geometry, modal planes, resolution and port order must match.')
    if (m.get('phase_convention_version')!=2 or dc['material_model']!='yee-aligned' or
            dc['actual_device']!=d['actual_device'] or not all(m['valid_bins'])):
        raise ValueError('The Meep electric basis and qualified Yee-aligned FDTDX model are required.')
    if incident not in inputs or incident not in d['port_order']:
        raise ValueError('The requested incident port was not measured.')
    archive=fdtdx/'arrays.npz'
    with np.load(archive,allow_pickle=False) as raw:
        if not np.array_equal(raw['frequency_per_um'],frequency) or not np.array_equal(frequency,d['frequency_per_um']):
            raise ValueError('Measured frequencies differ.')
        if d.get('full_matrix') is False:
            if d['launch']!=incident or not d['passed'] or d['matrix_axes']!=['frequency','outgoing_port']:
                raise ValueError('FDTDX partial input identity or qualification differs.')
            if digest(archive)!=d['arrays_sha256']:
                raise ValueError('FDTDX saved column digest changed.')
            for stage,sha in d['stage_sha256'].items():
                if digest(fdtdx/f'{incident}-{stage}.npz')!=sha:
                    raise ValueError('A native FDTDX monitor archive was modified.')
            sf=raw['s_column'];ref=unpack(d['s'])
        else:
            if not np.all(d['valid_column']) or d['matrix_axes']!=['frequency','outgoing_port','incoming_port']:
                raise ValueError('FDTDX full matrix is not qualified.')
            col=d['port_order'].index(incident)
            sf=raw['s'][:,:,col];ref=unpack(d['s'])[:,:,col]
        if not np.array_equal(sf,ref):
            raise ValueError('FDTDX native columns and displayed S disagree.')
    sm=sm[:,:,inputs.index(incident)]
    if sm.shape!=sf.shape or not np.isfinite(sm).all() or not np.isfinite(sf).all():
        raise ValueError('Wrong column shape or nonfinite coefficient.')
    rows=[]
    for index,name in enumerate(m['ports']):
        a,b=sm[:,index],sf[:,index]
        phase=np.angle(b*a.conj())
        stable=(abs(a)>=.01)&(abs(b)>=.01)
        rows.append({'port':name,'meep_S':encode(a),'fdtdx_S':encode(b),'meep_power':(abs(a)**2).tolist(),
            'fdtdx_power':(abs(b)**2).tolist(),'fdtdx_minus_meep_power':(abs(b)**2-abs(a)**2).tolist(),
            'fdtdx_minus_meep_physical_phase_rad':[float(p) if ok else None for p,ok in zip(phase,stable)],
            'absolute_complex_difference':abs(b-a).tolist()})
    result={'case':m['case'],'resolution':m['resolution'],'incident':incident,'wavelength_um':m['wavelength_um'],
            'devices':[m['actual_device'],d['actual_device']],'versions':{'meep':m['version'],**dc['versions']},
            'meep_archive_sha256':m['arrays_sha256'],'fdtdx_archive_sha256':digest(archive),
            'meep_report_sha256':digest(meep/'report.json'),'fdtdx_report_sha256':digest(fdtdx/'report.json'),
            'meep_native_monitor_sha256':m['native_monitor_sha256'],'meep_valid_bins':m['valid_bins'],
            'meep_stage_checks':m['stage_checks'],'fdtdx_checks':d.get('checks',d.get('sampled_setup_checks')),
            'physical_model_sha256':hashlib.sha256(json.dumps(mc['model'],sort_keys=True).encode()).hexdigest(),
            'meep_mesh_change':compare(previous,meep) if previous is not None else None,'ports':rows,
            'scope':'One independently excited input for both solvers. Same physical layout and cell density; Meep subpixel permittivity and FDTDX component/Yee material assignment differ. Reported physical phases are at the specified planes; no fitted or removed phase. Weak amplitudes below 0.01 are excluded only from the phase statistic.'}
    if m['case']=='coupler' and m['resolution']==24:
        proof=ROOT/'docs/verification/v011-fdtdx-coupler-boundary-ties.json'
        stage=fdtdx/f'{incident}-device.json'
        if proof.is_file() and stage.is_file():
            ties=json.loads(proof.read_text())
            native=json.loads(stage.read_text())
            if ties['recorded_native_material_grid_sha256']==native['material_grid_sha256']:
                result['fdtdx_boundary_sampling']={'audit':proof.name,'audit_sha256':digest(proof),
                    'native_material_grid_sha256':native['material_grid_sha256'],
                    'mirror_disagreeing_voxels':ties['mirror_disagreeing_voxels'],
                    'affected_y_coordinates_um':ties['mismatched_y_um'],
                    'interpretation':ties['interpretation']}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('meep','fdtdx','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--previous-meep',type=Path);parser.add_argument('--incident',required=True)
    args=parser.parse_args(); result=compare_column(args.meep,args.fdtdx,args.incident,args.previous_meep)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
