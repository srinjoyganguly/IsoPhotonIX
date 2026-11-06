"""Measure a tilted Gaussian beam coupled into a finite-width 3D grating on NVIDIA FDTDX."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'backend').is_dir():
    sys.path.insert(0, str(ROOT))
from backend.adapters.fdtdx_ports_adapter import _ports_runtime
from backend.adapters.fdtdx_ports3d_adapter import modal_amplitudes, port_basis
from scripts.grating_angular_spectrum import air_angular_flux
from scripts import lumerical_grating3d as ansys

TIDY_URL = 'https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/'
ANSYS_URL = 'https://optics.ansys.com/hc/en-us/articles/360042305334-Grating-coupler'
WL = np.array([1.5, 1.525, 1.55, 1.575, 1.6])
C = .299792458  # µm/fs


def settings(name, resolution):
    if name == 'lumerical' and resolution in (12,16):
        return ansys.model(resolution)
    matched = name == 'tidy3d'
    if name not in ('tidy3d', 'compact', 'compact-narrow') or resolution not in (12, 16):
        raise ValueError('Choose a supported grating and mesh.')
    return dict(geometry=name, resolution=resolution, period=.62, fill=.8, teeth=20 if matched else 8,
                width=10.8 if matched else (2. if name == 'compact-narrow' else 3.),
                guide_width=.5, taper=50. if matched else 3., height=.26, etch=.16,
                oxide_top=.68, oxide_bottom=2., n_si=3.47, n_oxide=1.44,
                beam_x=3.6 if matched else 1.8, waist=5.4 if matched else 1.5,
                angle_deg=14.5, source_z=1.18, buffer=.93, pml=.667,
                max_time_fs=2000. if matched else 750.,
                residual_slab_laterally_extended=matched)


def grid(m):
    r = m['resolution']
    if m['geometry']=='lumerical':
        low,high=np.asarray(m['physical_bounds'])
    else:
        length = m['period'] * m['teeth']
        # The Tidy3D simulation extent is the nonabsorbing region; PML is outside it.
        domain_width = max(m['width'], 3.)
        low = np.array([-m['taper']-m['buffer'], -domain_width/2-m['buffer'], -m['oxide_bottom']-m['buffer']])
        high = np.array([length+m['buffer'], domain_width/2+m['buffer'], m['height']+m['oxide_top']+m['buffer']])
    cells = np.ceil((high-low)*r).astype(int)
    cells[1] += cells[1] % 2
    pml = round(m['pml']*r)
    corner = (low+high-cells/r)/2 - pml/r
    shape = tuple(int(v) for v in cells+2*pml)
    axes = tuple(corner[i]+(np.arange(n)+.5)/r for i,n in enumerate(shape))
    return shape, pml, axes


def material(m, reference):
    shape, _, (xs,ys,zs) = grid(m)
    if m['geometry']=='lumerical':
        return ansys.epsilon(m,(xs,ys,zs),reference)
    eps = np.ones(shape, np.float32)
    if reference:
        return eps
    eps[:,:,zs < m['oxide_top']] = m['n_oxide']**2
    eps[:,:,zs < -m['oxide_bottom']] = m['n_si']**2
    # Volume fractions account for the shallow etch even on the coarsest grid.
    dx = 1/m['resolution']
    def coverage(axis, lower, upper):
        return np.clip((np.minimum(axis+dx/2, upper)-np.maximum(axis-dx/2, lower))/dx, 0., 1.)
    ridge_y = coverage(ys, -m['width']/2, m['width']/2)
    taper_width = m['guide_width']+(m['width']-m['guide_width'])*np.clip(1+xs/m['taper'],0.,1.)
    taper_xy = coverage(ys[None,:], -taper_width[:,None]/2, taper_width[:,None]/2)
    taper_xy *= coverage(xs, -1e4, 0.)[:,None]
    tooth_weights = np.zeros_like(xs)
    for tooth in range(m['teeth']):
        x0=tooth*m['period']
        tooth_weights += coverage(xs, x0, x0+m['fill']*m['period'])
    etched_y = np.ones_like(ys) if m['residual_slab_laterally_extended'] else ridge_y
    base_xy = coverage(xs, 0., 1e4)[:,None]*etched_y
    teeth_xy = tooth_weights[:,None]*ridge_y
    delta = m['n_si']**2-m['n_oxide']**2
    for k,z in enumerate(zs):
        full = coverage(z, 0., m['height'])
        etched = coverage(z, 0., m['height']-m['etch'])
        if full:
            eps[:,:,k] += delta*(full*taper_xy + etched*base_xy + (full-etched)*teeth_xy)
    return eps


def aligned(eps):
    def shift(a, axis):
        first = [slice(None)]*3
        first[axis]=slice(0,1)
        prior = [slice(None)]*3
        prior[axis]=slice(0,-1)
        return .5*(a+np.concatenate((a[tuple(first)],a[tuple(prior)]),axis))
    return np.stack([shift(shift(eps,1),2), shift(shift(eps,0),2), shift(shift(eps,0),1)])


def normal_flux(phasors, dx):
    # Colocated +z power, FDTDX phasors [frequency, component, x, y].
    p = phasors
    return .5*dx**2*np.real(p[:,0]*p[:,4].conj()-p[:,1]*p[:,3].conj()).sum(axis=(-1,-2), dtype=np.float64)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def native_stage(m, reference):
    jax,jnp,fdtdx,target = _ports_runtime('gpu')
    from fdtdx.core.jax.pytrees import autoinit, frozen_field
    from fdtdx.objects.sources.linear_polarization import LinearlyPolarizedPlaneSource
    from fdtdx.fdtd.forward import forward

    fiber = m['geometry']=='lumerical'
    lp = ansys.lp01(m['fiber_radius'],m['fiber_core_n'],m['fiber_clad_n'],1.55) if fiber else None

    @autoinit
    class WaistSource(LinearlyPolarizedPlaneSource):
        waist_cells: float = frozen_field()
        center_x: float = frozen_field()
        center_y: float = frozen_field()

        def _get_center(self, key):
            return jnp.asarray((self.center_x,self.center_y),dtype=jnp.float32)

        def _get_amplitude_raw(self, center):
            # The source is injected along z. Keep x and y distinct for a rectangular illuminated aperture.
            if lp is not None:
                x = np.arange(self.grid_shape[0])[:,None]-self.center_x
                y = np.arange(self.grid_shape[1])[None,:]-self.center_y
                field = ansys.fiber_radial_field(np.hypot(x,y)/(m['fiber_radius']*m['resolution']),lp[0],lp[1])
                return jnp.asarray(field[:,:,None],dtype=jnp.float32)
            x = jnp.arange(self.grid_shape[0])[:,None]-center[0]
            y = jnp.arange(self.grid_shape[1])[None,:]-center[1]
            return jnp.exp(-(x*x+y*y)/self.waist_cells**2)[:,:,None]

    shape,pml,axes = grid(m)
    nx,ny,nz = shape
    dx = 1/m['resolution']
    at = lambda axis,value: int(np.argmin(abs(axes[axis]-value)))
    frequencies = 1/WL
    center = 299792458e6/1.55
    bandwidth = .5*299792458e6*(1/WL[0]-1/WL[-1])
    pulse_end = 12/(2*np.pi*bandwidth)*1e15
    sim = fdtdx.SimulationConfig(time=m['max_time_fs']*1e-15, resolution=dx*1e-6,
                                 backend='gpu', dtype=jnp.float32, courant_factor=.99)
    if sim.backend!='gpu' or target.platform!='gpu':
        raise RuntimeError('Requested NVIDIA FDTDX execution is unavailable.')
    volume = fdtdx.SimulationVolume(name='domain', partial_grid_shape=shape, material=fdtdx.Material(permittivity=1.))
    boundaries = fdtdx.BoundaryConfig(**{f'thickness_grid_{s}{a}':pml for s in ('min','max') for a in 'xyz'},
                                      **{f'boundary_type_{s}{a}':'pml' for s in ('min','max') for a in 'xyz'})
    objects, constraints = fdtdx.boundary_objects_from_config(boundaries, volume)
    items = [volume,*objects.values()]
    constraints=list(constraints)
    def add(item,start):
        items.append(item)
        constraints.append(item.set_grid_coordinates(axes=(0,1,2),sides=('-','-','-'),coordinates=start))
    source_half = m['source_half_span'] if fiber else 2*m['waist']
    xmin=max(pml,at(0,m['beam_x']-source_half))
    xmax=min(nx-pml,at(0,m['beam_x']+source_half)+1)
    source_shape = (xmax-xmin,ny-2*pml,1)
    source=WaistSource(name='beam',partial_grid_shape=source_shape, direction='-',
              fixed_E_polarization_vector=(0.,1.,0.), azimuth_angle=m['angle_deg'],
              center_x=(m['beam_x']-axes[0][xmin])/dx,center_y=-axes[1][pml]/dx,
              waist_cells=(m['fiber_radius'] if fiber else m['waist'])/dx, wave_character=fdtdx.WaveCharacter(frequency=center),
              temporal_profile=fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(frequency=bandwidth),
                                                          center_wave=fdtdx.WaveCharacter(frequency=center)),
              switch=fdtdx.OnOffSwitch(end_time=pulse_end*1e-15))
    add(source,(xmin,pml,at(2,m['source_z'])))
    waves=tuple(fdtdx.WaveCharacter(frequency=float(f*299792458e6)) for f in frequencies)
    def detector(name, size, start, wave=waves):
        add(fdtdx.PhasorDetector(name=name, partial_grid_shape=size, wave_characters=wave,
                                 scaling_mode='pulse',exact_interpolation=True), start)
    # Keep the top-plane collection region local to the beam, including its transverse edges.
    aperture_half = m['upper_half_span'] if fiber else 2.2*m['waist']
    ax0=max(pml,at(0,m['beam_x']-aperture_half))
    ax1=min(nx-pml,at(0,m['beam_x']+aperture_half)+1)
    ay=slice(pml,ny-pml)
    az=at(2,m.get('sample_z',.93))
    detector('upper',(ax1-ax0,ny-2*pml,1),(ax0,pml,az))
    py=int(round((2.5 if fiber else 2)/dx)); pz=int(round((2.25 if fiber else 6*m['height'])/dx))
    px=at(0,m['output_x'] if fiber else -m['taper']-m['buffer']/2)
    sy=(ny-py)//2; sz=at(2,(.05 if fiber else m['height']/2)-pz*dx/2+dx/2)
    if not reference:
        detector('output',(1,py,pz),(px,sy,sz))
        detector('center',(nx-2*pml,1,nz-2*pml),(pml,ny//2,pml),(waves[2],))
    energy = fdtdx.EnergyDetector(name='energy',partial_grid_shape=tuple(n-2*pml for n in shape),
                reduce_volume=True,exact_interpolation=True,plot=False,
                switch=fdtdx.OnOffSwitch(interval=max(1,round(4e-15/sim.time_step_duration))))
    add(energy,(pml,pml,pml))
    key = jax.random.PRNGKey(0)
    obj,arrays,params,sim,_=fdtdx.place_objects(object_list=items,config=sim,constraints=constraints,key=key)
    eps = material(m,reference)
    eps_hash=digest(eps)
    staggered=aligned(eps)
    arrays=arrays.aset('inv_permittivities',jax.device_put(1/staggered,target))
    arrays,obj,_=fdtdx.apply_params(arrays,obj,params,key)
    if arrays.fields.E.device!=target or not np.allclose(np.asarray(arrays.inv_permittivities),1/staggered,rtol=2e-6):
        raise RuntimeError('Grid placement or material assignment differs from requested NVIDIA study.')
    del staggered
    source=obj['beam']
    electric,magnetic=np.asarray(source._E),np.asarray(source._H)
    vector=np.cross(electric.squeeze().transpose(1,2,0),magnetic.squeeze().transpose(1,2,0)).sum(axis=(0,1))
    direction=vector/np.linalg.norm(vector)
    expected=np.array([-np.sin(np.deg2rad(m['angle_deg'])),0.,-np.cos(np.deg2rad(m['angle_deg']))])
    if not np.allclose(direction,expected,atol=2e-4):
        raise RuntimeError(f'Injected beam points in unexpected direction: {direction}, expected {expected}')
    basis = None
    if not reference:
        basis,meta = port_basis(arrays.inv_permittivities[:,px:px+1,sy:sy+py,sz:sz+pz], frequencies,
                                frequencies*299792458e6, dx,sim.time_step_duration,m['n_oxide'],m['n_si'])
    def evolve(state,end):
        return jax.lax.while_loop(lambda s:s[0]<end,
            lambda s:forward(state=s,config=sim,objects=obj,key=key,record_detectors=True,
                             record_boundaries=False,simulate_boundaries=True),state)
    dt=sim.time_step_duration*1e15
    state=(jnp.asarray(0,dtype=jnp.int32),arrays)
    compile_start=time.monotonic()
    compiled=jax.jit(evolve,donate_argnums=(0,)).lower(state,jnp.asarray(1,dtype=jnp.int32)).compile()
    compile_seconds=time.monotonic()-compile_start
    schedule = np.flatnonzero(np.asarray(obj['energy']._is_on_at_time_step_arr))
    spectra=[]; history=[]; before=None; consecutive=0
    start=time.monotonic()
    for end in range(math.ceil(60/dt),sim.time_steps_total+math.ceil(60/dt),math.ceil(60/dt)):
        end=min(end,sim.time_steps_total)
        state=compiled(state,jnp.asarray(end,dtype=jnp.int32)); step, final=state
        jax.block_until_ready(final)
        upper=np.asarray(final.detector_states['upper']['phasor'])[0,:,:,:,:,0]
        if reference:
            measure=-normal_flux(upper,dx)
        else:
            port=np.asarray(final.detector_states['output']['phasor'])[0]
            measure=modal_amplitudes(port,basis['mode_e'],basis['mode_h'],dx**2)[0][:,1]
        times = schedule[schedule<end]*dt
        energy_trace = np.asarray(final.detector_states['energy']['energy'])[:len(times),0]
        if not np.isfinite(measure).all() or not np.isfinite(energy_trace).all():
            raise RuntimeError('Non-finite native fields.')
        recent = energy_trace[times>=times[-1]-60.]
        ratio = float(np.max(recent)/max(1e-30,np.max(energy_trace)))
        current_energy = float(energy_trace[-1])
        change=None if before is None else float(np.max(abs(measure-before))/max(1e-30,np.max(abs(measure))))
        passed=(end*dt>pulse_end and change is not None and change<.002 and ratio<1e-4)
        consecutive=consecutive+1 if passed else 0
        spectra.append(measure.copy()); history.append(dict(time_fs=end*dt,energy=current_energy,energy_ratio=ratio,relative_spectral_change=change))
        print(f"{'reference' if reference else 'device'} t={end*dt:.0f} fs, relative energy={ratio:.3g}, DFT-change={change}",flush=True)
        if consecutive>=2 or end==sim.time_steps_total: break
        before=measure.copy()
    if final.fields.E.device!=target:
        raise RuntimeError('FDTD fields left the requested device.')
    data={'upper':upper,'upper_x_um':axes[0][ax0:ax1],'upper_y_um':axes[1][ay],
          'wavelength_um':WL,'history_observable':np.array(spectra),'source_E':electric,'source_H':magnetic,
          'energy':energy_trace,'energy_time_fs':times,
          'material_yz':eps[px],'material_xy':eps[:,:,at(2,.18)],'material_xz':eps[:,ny//2,:],
          'grid_x_um':axes[0],'grid_y_um':axes[1],'grid_z_um':axes[2]}
    if not reference:
        data.update(output=port,mode_e=basis['mode_e'],mode_h=basis['mode_h'],mode_neff=basis['mode_neff'],
                    center_fields=np.asarray(final.detector_states['center']['phasor'])[0,0,:,:,0,:],
                    output_amplitudes=modal_amplitudes(port,basis['mode_e'],basis['mode_h'],dx**2)[0])
    detail=dict(stage='reference' if reference else 'device',actual_device=str(final.fields.E.device),device_name=target.device_kind,
                shape=list(shape),grid_resolution=m['resolution'],pml_cells=pml,time_steps=int(step),settled=consecutive>=2,
                history=history,compile_seconds=compile_seconds,stepping_seconds=time.monotonic()-start,source_direction=direction.tolist(),
                source_eh_sha256=digest(np.stack([electric,magnetic])),scalar_permittivity_sha256=eps_hash,
                sample_plane_z_um=float(axes[2][az]),output_plane_x_um=float(axes[0][px]),
                sampled_vertical_um=[float(axes[2][sz]),float(axes[2][sz+pz-1])],
                port_basis=None if reference else meta,
                fiber_mode=None if not fiber else dict(approximation='Weak-guidance LP01 transverse circular-fiber envelope on the tilted source plane',
                                                        u=lp[0],w=lp[1],neff=lp[2],source_z_um=float(axes[2][at(2,m['source_z'])])),
                native_worker_sha256=file_hash(__file__),
                versions={k:importlib.metadata.version(k) for k in ('fdtdx','jax','jaxlib','tidy3d','numpy')})
    return detail,data


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def stage_archive(m, label, cache):
    import os
    import subprocess
    sources=[Path(__file__),ROOT/'backend/adapters/fdtdx_ports_adapter.py',ROOT/'backend/adapters/fdtdx_ports3d_adapter.py',ROOT/'backend/adapters/fdtdx3d_adapter.py',
             ROOT/'backend/adapters/fdtdx_adapter.py',ROOT/'scripts/grating_angular_spectrum.py',ROOT/'scripts/lumerical_grating3d.py']
    identity={'settings':m,'stage':label,'sources':{p.name:file_hash(p) for p in sources},
              'versions':{name:importlib.metadata.version(name) for name in ('fdtdx','jax','jaxlib','numpy','tidy3d')}}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    dest=cache/(label+'-'+signature[:20]); cache.mkdir(exist_ok=True,parents=True)
    with (cache/(label+'-'+signature[:20]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not dest.exists():
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                staged=Path(tmp)/'stage'; staged.mkdir()
                cmd=[sys.executable,str(Path(__file__).resolve()),'--device','gpu','--geometry',m['geometry'],
                     '--resolution',str(m['resolution']),'--stage',label,'--output',str(staged)]
                # Workers are separate processes so native device memory is released after each stage.
                gpu_lock=os.environ.get('PHOTONICS_GPU_LOCK')
                if gpu_lock:
                    with Path(gpu_lock).open('a') as hardware:
                        fcntl.flock(hardware,fcntl.LOCK_EX); subprocess.run(cmd,check=True)
                else:
                    subprocess.run(cmd,check=True)
                receipt=json.loads((staged/'receipt.json').read_text())
                receipt.update(identity=identity,signature=signature,arrays_sha256=file_hash(staged/'arrays.npz'))
                (staged/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
                staged.rename(dest)
        receipt=json.loads((dest/'receipt.json').read_text())
        if receipt.get('native_worker_sha256')!=identity['sources']['calibrate_finite_grating.py']:
            raise ValueError('The worker changed during native acquisition.')
        if receipt['identity']!=identity or receipt['signature']!=signature or file_hash(dest/'arrays.npz')!=receipt['arrays_sha256']:
            raise ValueError(f'Native stage failed provenance checks: {label}')
        return dest,receipt


def run(m, output, cache):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'report.json').exists() or (output/'arrays.npz').exists():
        raise ValueError('Choose a fresh result directory.')
    ref,rr=stage_archive(m,'reference',cache)
    dev,dr=stage_archive(m,'device',cache)
    if rr['source_eh_sha256']!=dr['source_eh_sha256'] or rr['device_name']!=dr['device_name'] or rr['actual_device']!=dr['actual_device']:
        raise ValueError('Reference and device acquisitions differ in source or hardware.')
    with np.load(ref/'arrays.npz',allow_pickle=False) as r,np.load(dev/'arrays.npz',allow_pickle=False) as d:
        stored={**{f'ref_{k}':r[k] for k in r.files},**{f'dev_{k}':d[k] for k in d.files}}
        incident=-normal_flux(r['upper'],1/m['resolution'])
        output_mode=d['output_amplitudes'][:,1]
        efficiency=abs(output_mode)**2/incident
        valid=(incident>=.01*np.max(incident))&(efficiency>=0)&(efficiency<=1.02)&np.isfinite(efficiency)
        if not rr['settled'] or not dr['settled']: valid[:]=False
        # Remove the independently acquired incident field before projecting the reflected angular spectrum.
        stored.update(incident_power=incident,output_mode=output_mode,coupling=efficiency,valid=valid)
        if m['geometry']=='lumerical':
            reflected=ansys.reflected_fiber_amplitude(d['upper']-r['upper'],r['upper'],incident,1/m['resolution'])
            stored['reflected_fiber_mode']=reflected
            radiation={'fiber_return_power':(abs(reflected)**2).tolist(),
                       'scope':'Projection of the scattered upward field onto the time reverse of the measured incoming tilted fiber field on the same finite plane.'}
        else:
            angles=air_angular_flux(d['upper']-r['upper'],1/m['resolution'],WL)
            angular_fraction=angles['polar_bin_power']/incident[:,None]
            stored.update(reflected_angular_fraction=angular_fraction,**{'angular_'+key:value for key,value in angles.items()})
            radiation={'polar_edges_deg':angles['polar_edges_deg'].tolist(),'fraction_of_incident':angular_fraction.tolist(),
                       'scope':'Upgoing propagating power below 80°, estimated from the finite rectangular air-plane aperture after complex incident-field subtraction. Each bin integrates azimuth; uncollected lateral light and grazing waves lie outside the aperture estimate.'}
        horiz=d['grid_x_um'][dr['pml_cells']:-dr['pml_cells']].tolist()
        vert=d['grid_z_um'][dr['pml_cells']:-dr['pml_cells']].tolist()
    np.savez_compressed(output/'arrays.npz',**stored)
    external=json.loads((ROOT/'docs/verification/v012-grating-commercial-references.json').read_text())
    tidy=external['tidy3d']; lum=external['lumerical']
    comparison={'vendor':'Tidy3D','url':TIDY_URL,'geometry_matched':m['geometry']=='tidy3d',
                'wavelength_um':WL.tolist(),'published_power':tidy['coupling'],
                'published_db':tidy['coupling_db'],'figure_sha256':tidy['figure_sha256'],
                'scope':'Same dimensions, finite teeth, 50 µm taper, residual film extending laterally and same port observable; uniform local grid, rectangular aperture and a paraxial TFSF surface beam.' if m['geometry']=='tidy3d' else 'Workbench compact geometry has a different width, taper, tooth count, beam waist and lateral residual film.'}
    if comparison['geometry_matched']:
        comparison['local_minus_published_percentage_points']=(100*(efficiency-np.array(tidy['coupling']))).tolist()
        comparison['mean_absolute_difference_percentage_points']=float(np.mean(abs(100*(efficiency-np.array(tidy['coupling'])))))
    lcomparison={'vendor':'Lumerical','url':ANSYS_URL,'geometry_matched':m['geometry']=='lumerical',
                 'wavelength_um':WL.tolist(),'published_s21_power':lum['samples']['S21']['power'],
                 'vendor_file_sha256':lum['s_file_sha256'],'scope':lum['scope']}
    if m['geometry']=='lumerical':
        comparison['scope']='Different geometry and incoming mode: Tidy3D uses straight teeth and an air-side Gaussian.'
        lcomparison['scope']='The filed curved sectors, taper and tilted physical fiber are reconstructed. Local uniform mesh, sampled fiber profile, weak-guidance LP01 injection, and frozen central Palik indices differ from the commercial solver.'
        delta=100*(efficiency-np.asarray(lcomparison['published_s21_power']))
        lcomparison['local_minus_published_percentage_points']=delta.tolist()
        lcomparison['mean_absolute_difference_percentage_points']=float(np.mean(abs(delta)))
    report={'schema_version':1,'case_id':'finite-grating-3d','vendor':{'tidy3d':'Tidy3D','lumerical':'Lumerical'}.get(m['geometry'],'Workbench'),
            'actual_device':dr['actual_device'],'device_name':dr['device_name'],'passed':bool(np.all(valid)),
            'settings':m,'stages':{'reference':rr,'device':dr},'arrays_sha256':file_hash(output/'arrays.npz'),
            'result':{'resolution':m['resolution'],'wavelength_um':WL.tolist(),'beam_to_guide':efficiency.tolist(),
                      'center_beam_to_guide':float(efficiency[2]),'valid_beam_bins':valid.tolist(),
                      'incident_power':incident.tolist(),'mode_real':output_mode.real.tolist(),'mode_imag':output_mode.imag.tolist()},
            'field_preview':{'array':'dev_center_fields','index':[1],'axis_order':'xy','horizontal_axis':'x','vertical_axis':'z',
                             'horizontal_um':horiz,'vertical_um':vert,'component':'Ey','wavelength_um':1.55,
                             'description':'Real part of the Ey phasor on the recorded x–z plane nearest y = 0.'},
            'radiation':radiation,'comparisons':[comparison,lcomparison]}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Measured coupling:',efficiency.tolist(),'qualified:',report['passed'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['gpu'],required=True)
    parser.add_argument('--resolution',type=int,choices=(12,16),default=12)
    parser.add_argument('--geometry',choices=('compact','compact-narrow','tidy3d','lumerical'),default='compact')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--stage',choices=('reference','device'),help=argparse.SUPPRESS)
    args=parser.parse_args(); m=settings(args.geometry,args.resolution)
    if args.stage:
        report,arrays=native_stage(m,args.stage=='reference')
        np.savez_compressed(args.output/'arrays.npz',**arrays)
        (args.output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    else:
        if args.checkpoint:
            run(m,args.output,args.checkpoint)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                run(m,args.output,Path(tmp))


if __name__=='__main__':
    main()
