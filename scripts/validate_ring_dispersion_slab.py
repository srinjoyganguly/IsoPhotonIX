"""Measure passive Palik-fit silicon and oxide films against the normal-incidence Fresnel solution."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from scripts import ring_palik_dispersion as material


def fresnel(wavelength, epsilon, thickness):
    n = np.sqrt(epsilon)
    phase = 2*np.pi*thickness*n/wavelength
    r = (1-n)/(1+n)
    return ((1-r*r)*np.exp(1j*(phase-2*np.pi*thickness/wavelength))/
            (1-r*r*np.exp(2j*phase)))


def evolve(runtime, resolution, medium):
    from fdtdx.fdtd.forward import forward
    jax,jnp,fdtdx,gpu = runtime
    dx = 1/resolution
    shape = (6*resolution,4,4)
    sim = fdtdx.SimulationConfig(time=320e-15,resolution=dx*1e-6,backend='gpu',dtype=jnp.float32,courant_factor=.99)
    key = jax.random.PRNGKey(0)
    volume = fdtdx.SimulationVolume(name='domain',partial_grid_shape=shape,material=fdtdx.Material(permittivity=1.))
    boundaries = fdtdx.BoundaryConfig(
        boundary_type_miny='periodic',boundary_type_maxy='periodic',boundary_type_minz='periodic',boundary_type_maxz='periodic',
        thickness_grid_minx=resolution//2,thickness_grid_maxx=resolution//2)
    objects, constraints = fdtdx.boundary_objects_from_config(boundaries,volume)
    carrier = fdtdx.WaveCharacter(wavelength=1.55e-6)
    source = fdtdx.UniformPlaneSource(name='input',partial_grid_shape=(1,4,4),direction='+',fixed_E_polarization_vector=(0.,1.,0.),
        wave_character=carrier,normalize_by_energy=False,temporal_profile=fdtdx.GaussianPulseProfile(
            spectral_width=fdtdx.WaveCharacter(frequency=20e12),center_wave=carrier),
        switch=fdtdx.OnOffSwitch(end_time=100e-15))
    constraints=list(constraints)+[source.set_grid_coordinates(axes=(0,1,2),sides=('-','-','-'),coordinates=(resolution,0,0))]
    objects, arrays, params, sim, _ = fdtdx.place_objects(object_list=[volume,*objects.values(),source],config=sim,
                                                           constraints=constraints,key=key)
    arrays,objects,_=fdtdx.apply_params(arrays,objects,params,key)
    dt=float(sim.time_step_duration)
    if medium!='air':
        si,ox,c1,c2,k_si,k_ox=material.pole_config(dt)
        high,c3=(si,k_si) if medium=='silicon' else (ox,k_ox)
        slab=slice(2*resolution,2*resolution+resolution//2)
        inf=np.ones((1,*shape),np.float32)
        strength=np.zeros_like(inf)
        inf[:,slab]=high
        strength[:,slab]=c3
        state_shape=(1,3,*shape)
        arrays=arrays.aset('inv_permittivities',jax.device_put(1/inf,gpu))
        arrays=arrays.aset('dispersive_c1',jnp.full((1,1,1,1,1),c1,jnp.float32))
        arrays=arrays.aset('dispersive_c2',jnp.full((1,1,1,1,1),c2,jnp.float32))
        arrays=arrays.aset('dispersive_inv_c2',jnp.full((1,1,1,1,1),1/c2,jnp.float32))
        arrays=arrays.aset('dispersive_c3',jax.device_put(strength[None,:],gpu))
        arrays=arrays.aset('dispersive_P_curr',jnp.zeros(state_shape,jnp.float32))
        arrays=arrays.aset('dispersive_P_prev',jnp.zeros(state_shape,jnp.float32))
    def step(state,_):
        state=forward(state=state,config=sim,objects=objects,key=key,record_detectors=False,
                      record_boundaries=False,simulate_boundaries=True)
        e=state[1].fields.E
        return state,jnp.stack([jnp.mean(e[1,x]) for x in (3*resolution//2,4*resolution)])
    advance=jax.jit(lambda state:jax.lax.scan(step,state,None,length=sim.time_steps_total))
    final,signals=advance((jnp.array(0,jnp.int32),arrays))
    if gpu.platform!='gpu' or final[1].fields.E.device!=gpu:
        raise ValueError('The slab must run on the selected NVIDIA GPU.')
    result=np.asarray(signals)
    if not np.isfinite(result).all() or np.max(abs(result))<=1e-6:
        raise ValueError('Nonfinite or vanishing electromagnetic field.')
    return dt,result,str(gpu),gpu.device_kind


def spectrum(signal, dt):
    time=(1+np.arange(len(signal)))*dt
    omega=2*np.pi*material.LIGHT/(material.WAVELENGTHS*1e-6)
    return np.exp(1j*omega[:,None]*time) @ signal


def compare(reference, device, dt, target):
    def ratios(frac):
        stop=round(frac*len(reference))
        air=spectrum(reference[:stop],dt)
        film=spectrum(device[:stop],dt)
        if np.any(np.abs(air)<0.01*np.abs(air).max()):
            raise ValueError('Incident pulse does not support all sampled wavelengths.')
        return (film[:,0]-air[:,0])/air[:,0],film[:,1]/air[:,1]
    r,t=ratios(1.)
    earlier_r,earlier_t=ratios(.80)
    exact=fresnel(material.WAVELENGTHS,target,.5)
    power=np.abs(t)**2
    reflected=np.abs(r)**2
    return {'transmitted_power':power.tolist(),'reflected_power':reflected.tolist(),
            'amplitude_real':t.real.tolist(),'amplitude_imag':t.imag.tolist(),
            'target_power':(np.abs(exact)**2).tolist(),
            'max_transmission_power_error':float(np.max(abs(power-abs(exact)**2))),
            'max_transmission_phase_error_rad':float(np.max(abs(np.angle(t*exact.conj())))),
            'max_energy_balance_error':float(np.max(abs(power+reflected-1))),
            'max_prefix_power_change':float(max(np.max(abs(power-abs(earlier_t)**2)),np.max(abs(reflected-abs(earlier_r)**2))))}


def run(resolution,output):
    from backend.adapters.fdtdx_ports_adapter import _ports_runtime
    runtime=_ports_runtime('gpu')
    output.mkdir(parents=True,exist_ok=False)
    dt,ref,device,name=evolve(runtime,resolution,'air')
    signals={'air':ref}
    records={}
    for medium,table in zip(('silicon','oxide'),material.material_samples().values()):
        material_dt,signal,device_id,_=evolve(runtime,resolution,medium)
        if dt!=material_dt or device!=device_id:
            raise ValueError('Changed timestep or physical target.')
        signals[medium]=signal
        high,c3=material.fit_effective_cells(table[material.TRAIN],dt)
        c1,c2,_=material.recurrence(dt)
        fitted=material.epsilon_at(c1,c2,c3,high,material.WAVELENGTHS,dt)
        records[medium]=dict(**compare(ref,signal,dt,table),fit_epsilon=fitted.tolist(),
            max_heldout_index_error=float(max(abs(np.sqrt(fitted[material.HOLDOUT])-np.sqrt(table[material.HOLDOUT])))))
    archive=output/'arrays.npz'
    np.savez_compressed(archive,**signals)
    report={'resolution':resolution,'thickness_um':.5,'wavelength_um':material.WAVELENGTHS.tolist(),
        'actual_device':device,'device_name':name,'dt_seconds':dt,'measurements':records,
        'source_sha256':material.MATERIAL_SOURCE_SHA256,'fit_source_sha256':hashlib.sha256(Path(material.__file__).read_bytes()).hexdigest(),
        'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'versions':{name:importlib.metadata.version(name) for name in ('fdtdx','jax','jaxlib','numpy')},
        'arrays_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'passed':all(s['max_transmission_power_error']<=.015 and s['max_transmission_phase_error_rad']<=.10
            and s['max_energy_balance_error']<=.010 and s['max_prefix_power_change']<=.001
            and s['max_heldout_index_error']<=.001 for s in records.values())}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution',type=int,choices=(16,48,96),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.resolution,args.output)
