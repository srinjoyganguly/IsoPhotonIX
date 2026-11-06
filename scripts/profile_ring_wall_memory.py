"""Measure native wall-solver memory and conservatively admit an additional same-vendor uniform mesh."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def memory():
    smi=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    lines=subprocess.check_output([smi,'--query-gpu=memory.used,memory.free,memory.total',
                                  '--format=csv,noheader,nounits'],text=True).splitlines()
    if len(lines)!=1:
        raise RuntimeError('This single-adapter memory policy requires exactly one visible NVIDIA card.')
    return [int(v) for v in lines[0].split(',')]


def numerical_probe(vendor, mesh, reference, model):
    from backend.adapters.fdtdx_ports_adapter import _ports_runtime
    from scripts.calibrate_loaded_ring3d import geometry, stage
    from scripts.ring_stage_provenance import source_fingerprint
    import numpy as np
    runtime=_ports_runtime('gpu')
    target=runtime[3]
    g=geometry(vendor,mesh,.65,1.8,model)
    # Compile and execute actual production material, six PMLs, four port projections
    # and both main and tail blocks. This short run is a footprint measurement only.
    _,info=stage(runtime,g,20.,reference)
    statistics=target.memory_stats() or {}
    hashes=source_fingerprint(Path(sys.path[0]))
    identity=None
    if model=='reciprocal-wall':
        import importlib.metadata
        from scripts.ring_stage_provenance import fingerprint
        from scripts.reciprocal_ring_wall import memory_identity,worker_hashes
        versions={key:importlib.metadata.version(key) for key in ('jax','jaxlib','fdtdx','scipy','numpy','tidy3d')}
        implementation,reusable=fingerprint(Path(sys.path[0]),runtime,versions)
        if not reusable:
            raise ValueError('Benchmark GPU and numerical installation could not be verified.')
        hashes.update(worker_hashes(Path(sys.path[0])))
        identity=memory_identity(implementation)
    return dict(physical_model=g,performance=info['performance'],actual_material_sha256=info['material_sha256'],
                role='reference' if reference else 'device',cells=int(np.prod(g['shape'])),
                allocator_peak_bytes_in_use=statistics.get('peak_bytes_in_use'),
                origin='Fresh native 20 µm/c stage, no checkpoint lookup.',
                numerical_source_hashes=hashes,memory_identity=identity)


def profile(source, vendor, base_mesh, candidate_mesh, model):
    from scripts.calibrate_loaded_ring3d import geometry
    used,free,total=memory()
    candidate=geometry(vendor,candidate_mesh,.65,1.8,model)
    cells=math.prod(candidate['shape'])
    environment=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    records={}
    for role in ('reference','device'):
        with tempfile.TemporaryDirectory() as tmp:
            receipt=Path(tmp)/'receipt.json'
            command=[sys.executable,str(Path(__file__).resolve()),'--source',str(source),'--vendor',vendor,
                     '--base-mesh',str(base_mesh),'--model',model,'--sample',role,'--output',str(receipt)]
            samples=[memory()[0]]
            with subprocess.Popen(command,env=environment) as proc:
                while proc.poll() is None:
                    samples.append(memory()[0])
                    time.sleep(.2)
                if proc.returncode:
                    raise subprocess.CalledProcessError(proc.returncode,command)
            result=json.loads(receipt.read_text())
        sampled_delta=max(samples)-samples[0]
        peak=max(sampled_delta,(result['allocator_peak_bytes_in_use'] or 0)/2**20)
        if peak<=0:
            raise RuntimeError('No positive baseline GPU memory usage measured.')
        # Arrays with volume storage scale cubically. Cross-section and modal weights
        # grow more slowly. Runtime/context overhead is retained before scaling.
        projected=peak*cells/result['cells']
        result.update(samples_observed=len(samples),baseline_idle_used_mib=samples[0],
                      sampled_peak_global_mib=max(samples),peak_delta_used_for_estimate_mib=peak,
                      cell_ratio=cells/result['cells'],estimated_candidate_mib=projected,
                      required_free_with_reserve_mib=1.4*projected+2048)
        records[role]=result
    required=max(row['required_free_with_reserve_mib'] for row in records.values())
    available=memory()[1]
    return dict(schema=1,vendor=vendor,ring_interface=model,baseline_mesh=base_mesh,candidate_mesh=candidate_mesh,
                candidate_shape=candidate['shape'],candidate_cells=cells,initial_idle_used_mib=used,
                initial_free_mib=free,total_mib=total,free_at_completion_mib=available,
                admission_required_mib=required,admitted_at_measurement=required<=available,roles=records,
                policy='Use the higher of observed board-usage increment and JAX peak buffer bytes for each actual smaller-mesh stage; scale by full-domain cell count, apply 40 percent overhead and 2048 MiB graphics reserve. Recheck free memory immediately before field acquisition under the same device lock. This is a hardware-specific estimate.',
                planning_program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Acquisition source tree, including numerical provenance files.')
    parser.add_argument('--vendor',choices=('lumerical','tidy3d'),required=True)
    parser.add_argument('--model',choices=('wall-tensor','reciprocal-wall'),default='wall-tensor')
    parser.add_argument('--base-mesh',type=int,choices=(16,20),default=16)
    parser.add_argument('--candidate-mesh',type=int,choices=(20,24))
    parser.add_argument('--sample',choices=('reference','device'))
    parser.add_argument('--admit',type=Path,help='Recheck an existing measured budget immediately before the GPU worker.')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.source.resolve()))
    if args.admit:
        from scripts.ring_stage_provenance import source_fingerprint
        budget=json.loads(args.admit.read_text())
        current=source_fingerprint(args.source)
        if args.model=='reciprocal-wall':
            from scripts.reciprocal_ring_wall import worker_hashes
            current.update(worker_hashes(args.source))
        if (budget.get('ring_interface','wall-tensor')!=args.model or budget['vendor']!=args.vendor or not budget['baseline_mesh']<=args.candidate_mesh<=budget['candidate_mesh']
                or any(v['numerical_source_hashes']!=current for v in budget['roles'].values())):
            raise ValueError('Measured budget and acquisition source or geometry disagree.')
        _,free,capacity=memory()
        if capacity!=budget['total_mib']:
            raise ValueError('The available board capacity differs from the observation.')
        from scripts.calibrate_loaded_ring3d import geometry
        candidate=geometry(args.vendor,args.candidate_mesh,.65,1.8,args.model)
        cells=math.prod(candidate['shape'])
        required=max(1.4*r['peak_delta_used_for_estimate_mib']*cells/r['cells']+2048 for r in budget['roles'].values())
        if args.candidate_mesh==budget['candidate_mesh'] and not math.isclose(required,budget['admission_required_mib']):
            raise ValueError('Budget does not reproduce the stored requirement.')
        result=dict(vendor=args.vendor,candidate_mesh=args.candidate_mesh,candidate_cells=cells,
                    baseline_measurement_mesh=budget['baseline_mesh'],admitted=free>=required,
                    available_at_launch_mib=free,required_with_reserve_mib=required,
                    budget_sha256=hashlib.sha256(args.admit.read_bytes()).hexdigest())
        print('admit' if result['admitted'] else 'withhold')
    elif args.sample:
        result=numerical_probe(args.vendor,args.base_mesh,args.sample=='reference',args.model)
    else:
        if args.candidate_mesh is None or args.candidate_mesh<=args.base_mesh:
            parser.error('Candidate mesh must be finer than the measured base mesh.')
        result=profile(args.source.resolve(),args.vendor,args.base_mesh,args.candidate_mesh,args.model)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
