"""Verify native single-column acquisitions and publish their compact mesh/timing receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.calibrate_fdtdx_multiport import load_stage,save_record
from scripts.qualify_port_phase import native_column, qualify


def independent_meep(plan, column, reference):
    data = json.loads(reference.read_text())
    if 'physical_model_sha256' in data:
        digest=hashlib.sha256(json.dumps(plan['model'],sort_keys=True).encode()).hexdigest()
        names=[p['port'] for p in data['ports']]
        if data['physical_model_sha256']!=digest or not np.all(data['meep_valid_bins']):
            raise ValueError('Require the same physical layout and qualified Meep column.')
        raw=np.asarray([p['meep_S'] for p in data['ports']]).transpose(1,0,2)
        version=data['versions']['meep']
        archive=data['meep_archive_sha256']
    else:
        if (data['model']!=plan['model'] or not data['passed'] or not data['all_stages_settled'] or not data['full_matrix']):
            raise ValueError('Require a qualified independently excited Meep matrix.')
        names=data['ports']
        raw=np.asarray(data['s'])[:,:,0]
        version=data['version']
        archive=data['arrays_sha256']
    if names!=[port['name'] for port in plan['model']['ports']]:
        raise ValueError('The Meep port planes differ.')
    np.testing.assert_allclose(data['wavelength_um'],1/np.asarray(plan['model']['frequency_per_um']),rtol=1e-12)
    meep=raw[:,:,0]+1j*raw[:,:,1]
    result={}
    for i,port in enumerate(names):
        a,b=column[:,i],meep[:,i]
        strong=(abs(a)>=.02)&(abs(b)>=.02)
        result[port]={'max_abs_power_delta':float(np.max(abs(abs(a)**2-abs(b)**2))),
                      'max_abs_complex_delta':float(np.max(abs(a-b))),
                      'phase_delta_rad':[float(x) if ok else None for x,ok in zip(np.angle(a*b.conj()),strong)]}
    return {'meep_resolution':data['resolution'],'meep_version':version,'meep_source':reference.name,
            'meep_arrays_sha256':archive,'scope':'Independent electric-gauge Meep column on the same layout and planes; this finite mesh is a control, not a continuum reference.',
            'ports':result}


def stage_performance(folder,port,report):
    performance, acquisition = {}, {}
    for role in ('reference','device'):
        stage = load_stage(folder/f'{port}-{role}.json',folder/f'{port}-{role}.npz')
        if not stage.get('performance') or stage['archive_sha256']!=report['stage_sha256'][role]:
            raise ValueError('A measured stage is missing timing or has the wrong digest.')
        performance[role]=stage['performance']
        acquisition[role]='cache' if stage.get('reuse_existing_'+role) else 'fresh'
    return performance,acquisition


def publish(paths,destination):
    if not destination.is_dir():
        raise ValueError('The verification destination must already exist.')
    if not paths:
        raise ValueError('Select at least one native column.')
    compared = qualify(paths) if len(paths)>1 else None
    for folder in paths:
        plan, report, column, _, hashes = native_column(folder)
        report = dict(report)
        case=plan['model']['case']
        port = plan['model']['ports'][0]['name']
        performance, acquisition = stage_performance(folder,port,report)
        meep=destination/f'v011-meep-multiport-{case}-electric.json'
        refined = destination/f'v011-multiport-refined-{case}24.json'
        report['independent_meep_comparisons']=[independent_meep(plan,column,ref) for ref in (meep,refined)]
        stem=f'v014-{case}-polygon-cuda{plan["resolution"]}'
        array_name=stem+'-arrays.npz'
        report.update(stage_performance=performance,stage_acquisition=acquisition,arrays_file=array_name)
        if report['arrays_sha256']!=hashes['arrays.npz']:
            raise ValueError('The native column archive changed.')
        shutil.copyfile(folder/'arrays.npz',destination/array_name)
        save_record(destination/(stem+'.json'),report)
    if compared is not None:
        # Keep all consecutive meshes as the main qualification. A supplementary
        # series fixes the fractional position of the physical top Si wall on
        # the z lattice while changing mesh spacing (e.g. 16→24 or 20→28).
        groups={}
        for folder in paths:
            plan=json.loads((folder/'run.json').read_text())
            offset=round((plan['model']['height']*plan['resolution']/2)%1,10)
            groups.setdefault(offset,[]).append(folder)
        compared['fixed_vertical_registration']=[]
        for offset,folders in groups.items():
            if len(folders)>=2:
                pair=qualify(folders)
                compared['fixed_vertical_registration'].append({'positive_wall_offset_cells':offset,
                    'measurements':pair['measurements']})
        save_record(destination/f'v014-{case}-polygon-phase.json',compared)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+',type=Path)
    p.add_argument('--destination',required=True,type=Path)
    args=p.parse_args();publish(args.directories,args.destination)
