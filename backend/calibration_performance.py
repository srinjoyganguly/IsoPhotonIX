"""Extract stage-level GPU costs without treating cached uses as new computation."""
import csv
import io
import json
import math

CASE_NAMES = {'loaded-ring-3d':'ring-3d', 'finite-grating-3d':'grating-3d'}
FIELDS = ('receipt','case','geometry','stage','purpose','steps_lower_bound','origin','acquisition_id','device_kind','mesh_cells_per_um',
          'grid_shape','cells','pml_cells','steps','dt_seconds','simulated_time_fs','compile_seconds',
          'synchronized_step_seconds','cuda_event_elapsed_seconds','cuda_event_note','kernel_active_seconds','kernel_count','kernel_timing_note',
          'million_cell_steps_per_second','input_free_mib','driver_version','versions',
          'stage_through_steps_seconds','historical_elapsed_seconds','historical_basis','array_sha256')


def stage_rows(report, receipt):
    if str(report.get('actual_device','')).split(':')[0] not in ('cuda','gpu'):
        return []
    case = CASE_NAMES.get(report.get('case_id'),report.get('case'))
    settings = report.get('settings',{})
    model = report.get('physical_model',{})
    resolution = settings.get('resolution',model.get('resolution',report.get('resolution',
                 report.get('parameters',{}).get('resolution'))))
    stages = report.get('stages',{})
    if isinstance(stages,list):
        stages = {s.get('name',s.get('label',str(i))):s for i,s in enumerate(stages)}
    if not stages and report.get('stage_performance'):
        stages = {k:{'performance':v} for k,v in report['stage_performance'].items()}
    rows = []
    for role,stage in stages.items():
        if not isinstance(stage,dict):
            continue
        p=stage.get('performance') or {}
        shape=p.get('grid_shape') or stage.get('shape') or model.get('shape') or report.get('grid_shape')
        steps=p.get('steps',stage.get('time_steps',stage.get('steps',stage.get('step_count'))))
        if not steps:
            continue
        historical = [(key,stage[key]) for key in ('elapsed_s','stepping_seconds','seconds') if key in stage]
        basis={'elapsed_s':'ring acquisition including compilation and host sampling',
               'stepping_seconds':'grating loop including host monitor analysis',
               'seconds':'whole multiport stage including setup and output'}
        digest=(report.get('native_stage_provenance',{}).get(role,{}).get('native_arrays_sha256') or
                report.get('stage_sha256',{}).get(role) or stage.get('arrays_sha256') or
                stage.get('raw_monitor_sha256') or stage.get('archive_sha256'))
        origin=report.get('checkpoint_identity',{}).get('invocation_acquisition',{}).get(role,
               report.get('stage_acquisition',{}).get(role,stage.get('invocation_acquisition','unspecified')))
        acquisition=stage.get('acquisition') or report.get('invocation_acquisition',{}).get(role)
        if isinstance(acquisition,dict):
            origin={'fresh':'computed_this_invocation','cache-hit':'reused_verified_receipt'}.get(acquisition.get('status'),acquisition.get('origin',origin))
        for name in ('reference','device'):
            if role.endswith(name):
                for flag in ('reuse_existing_'+name,name+'_reused_from_verified_cache'):
                    if origin=='unspecified' and flag in stage:
                        origin='reused_verified_receipt' if stage[flag] else 'computed_this_invocation'
        origin={'fresh':'computed_this_invocation','cache-hit':'reused_verified_receipt','cache':'reused_verified_receipt'}.get(origin,origin)
        dt=p.get('dt_seconds')
        if dt is None and stage.get('time_step_um_c'):
            dt=stage['time_step_um_c']*1e-6/299792458
        final_fs=p.get('simulated_time_fs')
        if final_fs is None:
            history=stage.get('history') or [stage.get('final_sample',{})]
            final_fs=history[-1].get('time_fs')
        if dt is None and final_fs is not None:
            dt=final_fs*1e-15/steps
        if final_fs is None and dt is not None:
            final_fs=dt*steps*1e15
        hardware = stage.get('cuda_identity',report.get('checkpoint_identity',{}).get('provenance',{}).get('hardware',{}))
        versions = stage.get('versions') or report.get('versions') or report.get('native_versions')
        rows.append(dict(receipt=receipt,case=case,geometry=settings.get('geometry') or report.get('vendor') or case,
            stage=role,purpose='optical simulation',steps_lower_bound=None,origin=origin,acquisition_id=p.get('acquisition_id'),device_kind=p.get('device_kind',stage.get('device_name',report.get('device_name'))),
            mesh_cells_per_um=resolution,grid_shape=shape,cells=math.prod(shape) if shape else None,
            pml_cells=p.get('pml_cells',stage.get('pml_cells',model.get('pml'))),steps=steps,
            dt_seconds=dt,simulated_time_fs=final_fs,compile_seconds=p.get('compile_seconds',stage.get('compile_seconds')),
            synchronized_step_seconds=p.get('synchronized_step_seconds'),kernel_active_seconds=p.get('kernel_active_seconds'),
            cuda_event_elapsed_seconds=p.get('cuda_event_elapsed_seconds'),cuda_event_note=p.get('cuda_event_note'),
            million_cell_steps_per_second=p.get('million_cell_steps_per_second'),
            input_free_mib=stage.get('memory_admission',{}).get('available_mib'),
            driver_version=hardware.get('driver_version',hardware.get('driver_build')) if isinstance(hardware,dict) else None,versions=versions,
            kernel_count=p.get('kernel_count'),kernel_timing_note=p.get('kernel_timing_note','Hardware intervals were not recorded.'),
            stage_through_steps_seconds=p.get('stage_through_steps_seconds'),
            historical_elapsed_seconds=historical[0][1] if historical and not p else None,
            historical_basis=basis[historical[0][0]] if historical and not p else None,array_sha256=digest))
    return rows


def timing_csv(rows):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return stream.getvalue()


def current_totals(rows):
    unique={}
    for row in rows:
        key=row['acquisition_id']
        if not key:
            continue
        times=(row['synchronized_step_seconds'],row['kernel_active_seconds'],row['steps'],row['cells'],row['cuda_event_elapsed_seconds'])
        if unique.setdefault(key,times)!=times:
            raise ValueError(f'Conflicting records for native acquisition {key}')
    return {'distinct_timed_acquisitions':len(unique),
            'distinct_acquisitions_with_kernel_time':sum(r[1] is not None for r in unique.values()),
            'synchronized_step_seconds':sum(r[0] for r in unique.values()) if unique else None,
            'distinct_acquisitions_with_cuda_event_time':sum(r[4] is not None for r in unique.values()),
            'available_cuda_event_elapsed_seconds':sum(r[4] for r in unique.values() if r[4] is not None) if any(r[4] is not None for r in unique.values()) else None,
            'available_kernel_active_seconds':sum(r[1] for r in unique.values() if r[1] is not None) if any(r[1] is not None for r in unique.values()) else None}
