"""Build the public GPU timing ledger from the checked-in stage receipts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from backend.calibration_performance import FIELDS, current_totals, stage_rows, timing_csv

BASE=Path(__file__).resolve().parents[1]/'docs/verification'


def extra_costs(report, name):
    """Account separately for measured diagnostics, memory planning, and incomplete stages."""
    def item(case, geometry, stage, purpose, **observed):
        return {**dict.fromkeys(FIELDS), 'receipt':name,'case':case,'geometry':geometry,'stage':stage,
                'purpose':purpose,'origin':'computed_this_invocation', **observed}

    rows=[]
    if name=='v014-reciprocal-wall-gpu.json':
        p=report['performance']
        native={'case':'ring-3d','actual_device':p['device'],'physical_model':report['source_geometry'],
                'stages':{'passive wall tile':{'performance':p}}}
        rows.extend(dict(row,purpose='bounded stability control',origin='computed_this_invocation') for row in stage_rows(native,name))
    elif name.startswith('v014-ring-memory-'):
        for role,stage in report['roles'].items():
            native={'actual_device':stage['performance']['device'],'case':'ring-3d','vendor':report['vendor'],
                    'physical_model':stage['physical_model'],'stages':{role:stage}}
            rows.extend(dict(row,purpose='memory probe',origin='computed_this_invocation') for row in stage_rows(native,name))
    elif name=='v014-ring-dense-attempt.json':
        complete={'actual_device':'cuda:0','case':'ring-3d','physical_model':report['physical_model'],
                  'stages':{'reference':report['successful_reference']},'versions':report['versions']}
        rows.extend(dict(row,purpose='reference of interrupted run',origin='computed_this_invocation',
                         array_sha256=report['reference_native_arrays_sha256']) for row in stage_rows(complete,name))
        partial=report['incomplete_device']
        model=report['physical_model']
        rows.append(item('ring-3d',model['vendor'],'device','interrupted optical simulation',
                         steps_lower_bound=partial['completed_fdtd_steps_lower_bound'],grid_shape=model['shape'],
                         cells=math.prod(model['shape']),mesh_cells_per_um=model['resolution'],
                         kernel_timing_note=partial['timing_evidence']))
    elif name=='v014-grating-memory-timed-one-step.json':
        p=report['performance']
        native={'case':'grating-3d','actual_device':p['device'],'settings':report,
                'stages':{report['stage']:{'performance':p}}}
        rows.extend(dict(row,purpose='memory probe',origin='computed_this_invocation',
                         input_free_mib=report['global_free_mib']) for row in stage_rows(native,name))
    elif name.startswith('v014-grating-memory-') and 'roles' in report:
        plans=[report]
        if report.get('more_distant_r12_budget'):
            plans.append(report['more_distant_r12_budget'])
        for plan in plans:
            for role,stage in plan['roles'].items():
                sample=stage['actual']
                if sample.get('executed_fdtd_steps')!=1:
                    continue
                rows.append(item('grating-3d',sample['geometry'],f"{role} r{sample['resolution']}",'memory probe',
                            device_kind=sample['gpu'],mesh_cells_per_um=sample['resolution'],steps=1,
                            grid_shape=sample['shape'],cells=sample['cells'],input_free_mib=stage['observation']['free_before_mib'],
                            kernel_timing_note='Synchronized one-step allocator probe. Its stepping time was not recorded.'))
    elif name=='v014-grating-profiler-failures.json':
        for trial in report['trials']:
            rows.append(item('grating-3d',trial['geometry'],trial['trial'],'interrupted profiling attempt',
                             mesh_cells_per_um=trial['resolution'],kernel_timing_note=trial['error']))
    elif name.startswith('v014-grating-') and name.endswith('-interruption.json'):
        rows.append(item('grating-3d',report['geometry'],report['stage'],'interrupted optical simulation',
                         mesh_cells_per_um=report['resolution'],grid_shape=report['grid_shape'],cells=report['cells'],
                         device_kind=report.get('device_name'),dt_seconds=report['dt_seconds'],
                         steps_lower_bound=report['completed_fdtd_steps_lower_bound'],kernel_timing_note=report['scope']))
    elif name=='v014-multiport-profiling-attempts.json':
        for trial in report['attempts']:
            case=trial['study'].split('-')[0]
            rows.append(item(case,case,trial['study'],'interrupted profiling attempt',mesh_cells_per_um=trial['resolution'],
                             device_kind=trial['device_kind'],kernel_timing_note=trial['error']))
    elif name=='v014-splitter-gui-notebook.json':
        for label in ('gui','fresh_notebook'):
            stages={role:{'performance':p} for role,p in report['gpu_stages_'+label].items()}
            sample={'case':'splitter','resolution':report['resolution'],'actual_device':report['actual_device'],'stages':stages}
            rows.extend(dict(row,purpose='GUI simulation' if label=='gui' else 'fresh notebook replay',
                             stage=label+' '+row['stage'],origin='computed_this_invocation') for row in stage_rows(sample,name))
    return rows


def collect(base):
    rows=[]
    hashes={}
    for path in sorted(base.glob('v0*.json')):
        if 'index' in path.name or 'performance' in path.name:
            continue
        data=path.read_bytes()
        report=json.loads(data)
        if not isinstance(report,dict):
            continue
        record=stage_rows(report,path.name)+extra_costs(report,path.name)
        if record:
            hashes[path.name]=hashlib.sha256(data).hexdigest()
            rows.extend(record)
    return {'schema':1,'source_sha256':hashes,'totals':current_totals(rows),'rows':rows}


def audit(ledger, base):
    if json.loads((base/'v014-gpu-performance.json').read_text())!=ledger:
        raise ValueError('Regenerate GPU timing JSON from the current stage receipts.')
    if (base/'v014-gpu-performance.csv').read_text()!=timing_csv(ledger['rows']):
        raise ValueError('GPU timing CSV differs from its source records.')
    for row in ledger['rows']:
        if not row['acquisition_id']:
            continue
        if (row['synchronized_step_seconds']<=0 or row['steps']<=0 or
                row['cells']!=math.prod(row['grid_shape']) or not math.isclose(
                    row['simulated_time_fs'],row['dt_seconds']*row['steps']*1e15)):
            raise ValueError(f"Invalid timed stage for {row['receipt']} / {row['stage']}")
        if row['cuda_event_elapsed_seconds'] is not None and row['cuda_event_elapsed_seconds']<=0:
            raise ValueError('CUDA event intervals must be positive.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',help='Verify the saved ledger against all original stage receipts.')
    args=parser.parse_args()
    ledger=collect(BASE)
    if not args.check:
        (BASE/'v014-gpu-performance.json').write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
        (BASE/'v014-gpu-performance.csv').write_text(timing_csv(ledger['rows']))
    audit(ledger,BASE)
    print(len(ledger['source_sha256']), 'reports;', len(ledger['rows']), 'stage uses;', ledger['totals'])
