"""Dispersive cell coefficients, carrier source auditing and frequency-resolved port bases for the ring."""
import hashlib

import numpy as np

from backend.adapters.fdtdx_ports3d_adapter import port_basis
from scripts import ring_palik_dispersion as material


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build(g,reference,dt,static_permittivity):
    measured=list(material.material_samples().values())
    si,oxide=measured
    values=[static_permittivity(g,reference,si[i],oxide[i],np.float64) for i in material.TRAIN]
    eps_inf,c3=material.fit_effective_cells(values,dt)
    c1,c2,_=material.recurrence(dt)
    # Compare the actual float32 device coefficients against the optical table,
    # including two wavelengths withheld from fitting and partially filled wall voxels.
    eps_inf=eps_inf.astype(np.float32)
    c3=c3.astype(np.float32)
    c1,c2=float(np.float32(c1)),float(np.float32(c2))
    del values
    residual=[]
    for i,w in enumerate(material.WAVELENGTHS):
        target=static_permittivity(g,reference,si[i],oxide[i],np.float64)
        fit=material.epsilon_at(c1,c2,c3,eps_inf,w,dt)
        residual.append(float(np.max(abs(fit-target))))
    if max(residual)>.01 or np.any(c3<0) or np.min(eps_inf)<1:
        raise ValueError('Outside the passive-cell permittivity error or positivity bound.')
    carrier=material.epsilon_at(c1,c2,c3,eps_inf,1.55,dt).astype(np.float32)
    return {'inf':eps_inf,'c3':c3,'c1':c1,'c2':c2,'dt':dt},carrier,{
        'profile':'palik-ade','source_sha256':material.MATERIAL_SOURCE_SHA256,'wavelength_um':material.WAVELENGTHS.tolist(),
        'applied_eps_inf_sha256':digest(eps_inf),'applied_c3_sha256':digest(c3),'c1':c1,'c2':c2,
        'maximum_cell_epsilon_errors':residual,'maximum_allowed':.01,'held_out_indices':material.HOLDOUT,
        'carrier_permittivity_sha256':digest(carrier),'qualified_band_um':[1.5,1.6]}


def attach(arrays,coeffs,target):
    import jax
    import jax.numpy as jnp
    put=lambda v:jax.device_put(v,target)
    p=jnp.zeros((1,*arrays.fields.E.shape),dtype=arrays.fields.E.dtype)
    arrays=arrays.aset('inv_permittivities',put(1/coeffs['inf']))
    arrays=arrays.aset('dispersive_P_curr',put(p)).aset('dispersive_P_prev',put(jnp.zeros_like(p)))
    for name in ('c1','c2'):
        scalar=np.array(coeffs[name],dtype=np.float32).reshape(1,1,1,1,1)
        arrays=arrays.aset('dispersive_'+name,put(scalar))
    arrays=arrays.aset('dispersive_c3',put(coeffs['c3'][None]))
    return arrays.aset('dispersive_inv_c2',put(np.array(1/coeffs['c2'],dtype=np.float32).reshape(1,1,1,1,1)))


def port_profiles(coeffs,port_slice,dx,stage_dt,cladding,core):
    import jax.numpy as jnp
    basis={}
    edges=[]
    for w in material.WAVELENGTHS:
        ep=material.epsilon_at(coeffs['c1'],coeffs['c2'],coeffs['c3'][port_slice],coeffs['inf'][port_slice],w,stage_dt)
        data,detail=port_basis(jnp.asarray(1/ep),[1/w],[material.LIGHT*1e6/w],dx,stage_dt,cladding,core)
        edges.append(detail['edge_electric_l2_fraction'][0])
        for key,val in data.items():
            basis.setdefault(key,[]).append(val[0])
    basis={k:np.asarray(v) for k,v in basis.items()}
    return basis,{'neff_real':basis['mode_neff'].real.tolist(),'edge_electric_l2_fraction':edges}


def audit_waveform(native):
    """Hash incident E/H drives on both tangential components for every switched-on step."""
    import jax
    import jax.numpy as jnp
    if native._temporal_H_filter is not None:
        raise ValueError('Dispersive ring requires the prescribed, common carrier-mode source pulse.')
    active=np.flatnonzero(np.asarray(native._is_on_at_time_step_arr))
    local=np.asarray(native._time_step_to_on_idx)[active]
    dt=native._config.time_step_duration
    def pulse(step,offset,field):
        return (native.temporal_profile.get_amplitude(time=(step+offset)*dt,period=native.wave_character.get_period(),
                    phase_shift=native.wave_character.phase_shift)*native.static_amplitude_factor*field)
    axes=(native.horizontal_axis,native.vertical_axis)
    details={}
    for key,offset,data,advance in [('E',native._time_offset_E,native._E,.5),('H',native._time_offset_H,native._H,0.)]:
        sha=hashlib.sha256()
        for axis in axes:
            def one(idx):
                return pulse(idx+advance,offset[axis],data[axis])
            sample=jax.jit(jax.vmap(one))
            for i in range(0,len(local),64):
                vals=np.asarray(sample(jnp.asarray(local[i:i+64],dtype=jnp.float32)))
                if not np.isfinite(vals).all():
                    raise ValueError('Invalid driven source samples.')
                sha.update(np.ascontiguousarray(vals).tobytes())
        details['driven_'+key+'_sha256']=sha.hexdigest()
        details['source_offset_'+key+'_sha256']=digest(np.asarray(offset))
    details['active_steps_sha256']=digest(active.astype('<i8'))
    details['active_steps']=len(active)
    return details
