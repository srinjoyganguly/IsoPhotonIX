"""Compare an abstract CUDA timestep analysis with an actual 12-cell allocation before refining."""
import argparse
import json
from pathlib import Path


MIB=2**20


def read(directory,name):
    return json.loads((directory/name).read_text())


def estimate(base,observed,refined,idle_used_mib,free_before_mib,observed_global_peak_mib=None):
    if not base['cpu_field_allocation_verified'] or not refined['cpu_field_allocation_verified']:
        raise ValueError('Memory planning arrays must have been allocated on the host.')
    if base['executed_fdtd_steps'] or refined['executed_fdtd_steps'] or observed['executed_fdtd_steps']!=1:
        raise ValueError('Expected two shape-only plans and one bounded native measurement.')
    if base['geometry']!=refined['geometry'] or observed['geometry']!=base['geometry'] or base['resolution']!=12 or observed['resolution']!=12:
        raise ValueError('Compare one physical geometry against its own measured 12-cell baseline.')
    if base['gpu']!=refined['gpu'] or observed['gpu']!=base['gpu']:
        raise ValueError('GPU device models differ.')
    reference_live=base['compiled_live_bytes']/MIB
    measured_live=observed['actual_compiled_live_bytes']/MIB
    # A shape-only plan must be close enough to the actual device executable to make a comparison.
    disagreement=abs(measured_live-reference_live)/max(measured_live,1.)
    if disagreement>.05:
        raise ValueError('The abstract and actual 12-cell executable live memory disagree by more than 5%.')
    peak=max(observed['global_used_mib'],observed_global_peak_mib or 0.)
    allocator_peak=observed['allocator_bytes'].get('peak_bytes_in_use',0)/MIB
    baseline=max(allocator_peak,peak-idle_used_mib,measured_live)
    refined_live=refined['compiled_live_bytes']/MIB
    cells=refined['cells']/base['cells']
    compiled_ratio=refined_live/reference_live
    extra=max(baseline-reference_live,0.)
    # Evaluate both scaling estimates. Reserve an additional 20% and 1 GiB for the graphics driver.
    estimated=max(baseline*max(cells,compiled_ratio),refined_live+cells*extra)
    required=1.2*estimated+1024.
    return {'geometry':base['geometry'],'resolution':refined['resolution'],'cells':refined['cells'],
            'actual_reference_cells':base['cells'],'baseline_native_peak_mib':baseline,
            'baseline_observed_global_peak_mib':peak,'idle_used_floor_mib':idle_used_mib,
            'reference_abstract_compiled_live_mib':reference_live,'reference_actual_compiled_live_mib':measured_live,
            'reference_abstract_actual_fraction_difference':disagreement,
            'refined_abstract_compiled_live_mib':refined_live,'cell_count_ratio':cells,
            'compiled_live_ratio':compiled_ratio,'estimated_runtime_mib':estimated,
            'required_free_with_reserve_mib':required,'available_before_launch_mib':free_before_mib,
            'fits_envelope':required<=free_before_mib,
            'scope':'Hardware-specific memory admission estimate, combining actual 12-cell allocator/global measurements and CUDA XLA live-buffer analysis. Runtime success remains subject to available memory at launch.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plans',type=Path,required=True);p.add_argument('--geometry',required=True)
    p.add_argument('--resolution',type=int,required=True);p.add_argument('--idle-used-mib',type=float,required=True)
    p.add_argument('--free-mib',type=float,required=True);p.add_argument('--observed-peak-mib',type=float)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    base=read(args.plans,f'{args.geometry}-12-plan.json')
    actual=read(args.plans,f'{args.geometry}-12-observe.json')
    refined=read(args.plans,f'{args.geometry}-{args.resolution}-plan.json')
    result=estimate(base,actual,refined,args.idle_used_mib,args.free_mib,args.observed_peak_mib)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
