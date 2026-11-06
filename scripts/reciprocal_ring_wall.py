"""A bounded symmetric inverse-permittivity operator linking staggered ring Ex and Ey edges."""
import numpy as np

# Ey neighbors of each Ex, in Yee grid indices. Each stored link is also used by the transpose.
OFFSETS=((0,0),(1,0),(0,-1),(1,-1))


def reciprocal_links(inverse, lower, upper, periodic=False):
    """Preserve diagonal entries and bound both M-lower*I and upper*I-M.

    Every x or y edge has at most four partners. The factor 1/4 makes the sum
    of pairwise bounds at each edge no larger than its own diagonal margin.
    """
    dx,dy=inverse[0],inverse[4]
    if not (0<lower<=upper) or np.min(inverse[[0,4,8]])<lower-2e-6 or np.max(inverse[[0,4,8]])>upper+2e-6:
        raise ValueError('Ring constitutive diagonals exceed the declared dielectric bounds.')
    links=np.empty((4,*dx.shape),np.float32)
    asymmetry=0.;clipped=0;active=0
    for i,(sx,sy) in enumerate(OFFSETS):
        offset=(-sx,-sy)
        partner=np.roll(dy,offset,axis=(0,1))
        partner_xy=np.roll(inverse[3],offset,axis=(0,1))
        mean=(inverse[1]+partner_xy)/2
        limit=np.minimum(np.sqrt(np.maximum(dx-lower,0)*np.maximum(partner-lower,0)),
                         np.sqrt(np.maximum(upper-dx,0)*np.maximum(upper-partner,0)))
        links[i]=np.clip(mean,-limit,limit)/4
        if not periodic:
            if sx:
                links[i,-sx:]=0
            if sy:
                links[i,:,:-sy]=0
        asymmetry=max(asymmetry,float(np.max(abs(inverse[1]-partner_xy)))/4)
        active+=int(np.count_nonzero(mean))
        clipped+=int(np.count_nonzero(abs(mean)>limit+1e-8))
    return links,dict(inverse_lower=float(lower),inverse_upper=float(upper),active_candidate_links=active,
                      links_limited=clipped,legacy_max_transpose_discrepancy=asymmetry,
                      construction='Each stored xy edge pair is shared by the forward and transpose actions.')


def apply_inverse(curl,inverse,links):
    """Apply the bounded symmetric operator without a per-cell matrix solve."""
    import jax.numpy as jnp
    x=inverse[0]*curl[0]
    y=inverse[4]*curl[1]
    for i,(sx,sy) in enumerate(OFFSETS):
        x=x+links[i]*jnp.roll(curl[1],(-sx,-sy),axis=(0,1))
        y=y+jnp.roll(links[i]*curl[0],(sx,sy),axis=(0,1))
    return jnp.stack((x,y,inverse[8]*curl[2]))


def worker_hashes(root):
    import hashlib
    files=('scripts/reciprocal_ring_wall.py','scripts/grating_cuda_identity.py')
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}


def memory_identity(provenance):
    """Portable GPU/software identity; never publishes the card's private identity."""
    import os
    from scripts.ring_stage_provenance import json_digest
    hardware=provenance.get('hardware')
    payload=provenance.get('installed_numerics')
    if not hardware or not payload:
        raise ValueError('A verified native GPU and installed numerical payload are required for memory admission.')
    return dict(gpu_model=hardware['model'],driver=hardware['driver_build'],cuda_driver_api=hardware['cuda_driver_api'],
                versions=provenance['versions'],python_abi=provenance['python_abi'],
                numerical_payload_sha256=json_digest(payload),jax_configuration=provenance['jax_configuration'],
                allocator_flags={key:os.environ.get(key) for key in ('XLA_FLAGS','XLA_PYTHON_CLIENT_PREALLOCATE',
                    'XLA_PYTHON_CLIENT_ALLOCATOR','XLA_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_MEM_FRACTION')})


def fresh_admission(root,g,target,provenance):
    """Check only a new high-resolution field evolution, after the cache lookup."""
    import hashlib
    import json
    import math
    import os
    from pathlib import Path
    from scripts.grating_cuda_identity import available_mib
    if g['resolution']<20:
        return None
    if target.platform!='gpu' or int(target.local_hardware_id)!=0:
        raise ValueError('The measured ring budget supports the worker’s first explicitly selected CUDA GPU.')
    default=f"docs/verification/v014-ring-memory-{g['vendor']}-r24-reciprocal.json"
    path=Path(os.environ.get('PHOTONICS_RING_MEMORY_PLAN') or root/default)
    if not path.is_file():
        raise ValueError('Fresh high-resolution reciprocal runs require a measured budget for this GPU and model.')
    contents=path.read_bytes()
    budget=json.loads(contents)
    if (budget['vendor']!=g['vendor'] or budget.get('ring_interface')!='reciprocal-wall' or
            not budget['baseline_mesh']<=g['resolution']<=budget['candidate_mesh'] or
            g['pml']!=round(.65*g['resolution']) or g['half_ap']!=round(.9*g['resolution'])):
        raise ValueError('Requested mesh, aperture, PML or geometry is outside this measured budget.')
    identity=memory_identity(provenance)
    if identity['gpu_model']!=target.device_kind or set(budget['roles'])!={'reference','device'}:
        raise ValueError('Selected GPU or measured reference/device roles differ.')
    cells=math.prod(g['shape'])
    estimates=[]
    for role,data in budget['roles'].items():
        model=data['physical_model']
        observed=max(data['sampled_peak_global_mib']-data['baseline_idle_used_mib'],
                     (data['allocator_peak_bytes_in_use'] or 0)/2**20)
        if (not math.isclose(observed,data['peak_delta_used_for_estimate_mib']) or
                data['performance']['platform']!='gpu' or data['performance']['steps']<=0 or
                data['performance']['grid_shape']!=model['shape'] or data['performance']['device_kind']!=target.device_kind or
                data['role']!=role or data['numerical_source_hashes']!=provenance['source_files']
                or data.get('memory_identity')!=identity or data['cells']!=math.prod(model['shape'])
                or data['cells']<=0 or not 0<data['peak_delta_used_for_estimate_mib']<=budget['total_mib']
                or model['resolution']!=budget['baseline_mesh'] or model['ring_interface']!='reciprocal-wall'
                or any(model[k]!=g[k] for k in ('vendor','width_um','height_um','radius_um','edge_gap_um','top_index','silicon_index','substrate_index'))):
            raise ValueError('Measured GPU, software, source, allocation or physical ring differs from this run.')
        estimates.append(1.4*data['peak_delta_used_for_estimate_mib']*cells/data['cells']+2048)
    required=max(estimates)
    available=available_mib()
    if available<required:
        raise MemoryError(f'Fresh ring stage requires {required:.0f} MiB free including reserve; measured {available} MiB.')
    return dict(budget_sha256=hashlib.sha256(contents).hexdigest(),fixture=default,
                selected_device=str(target),device_kind=target.device_kind,
                cells=cells,available_at_launch_mib=available,required_with_reserve_mib=required,
                external_budget_override=bool(os.environ.get('PHOTONICS_RING_MEMORY_PLAN')))
