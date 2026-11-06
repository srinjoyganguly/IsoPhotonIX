"""Measure a tilted Gaussian beam coupled into a finite-width 3D grating on NVIDIA FDTDX."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'backend').is_dir():
    sys.path.insert(0, str(ROOT))
from backend.adapters.fdtdx_ports_adapter import _ports_runtime
from backend.adapters.fdtdx_ports3d_adapter import modal_amplitudes, port_basis, canonical_source_profile
from scripts.grating_angular_spectrum import air_angular_flux
from scripts import lumerical_grating3d as ansys
from scripts import grating_cuda_identity as cuda_identity
from scripts.gpu_timing import StageTiming
from scripts import grating_mirror

TIDY_URL = 'https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/'
ANSYS_URL = 'https://optics.ansys.com/hc/en-us/articles/360042305334-Grating-coupler'
WL = np.array([1.5, 1.525, 1.55, 1.575, 1.6])
C = .299792458  # µm/fs


def monitoring_wavelengths(model):
    if model.get('grating_spectrum', 'standard') == 'dense':
        return np.arange(1500, 1601, 5, dtype=np.float64)/1000
    return WL


def settings(name, resolution, grating_material_model='volume-average', grating_spectrum='standard', grating_domain='full'):
    if grating_domain not in ('full','mirror-y') or (grating_domain == 'mirror-y' and name not in ('lumerical','tidy3d')):
        raise ValueError('The y mirror requires a symmetric benchmark grating.')
    mirrors = grating_domain == 'mirror-y'
    if mirrors and (grating_spectrum != 'standard' or (name == 'lumerical' and grating_material_model != 'direct-yee')):
        raise ValueError('Mirror symmetry requires the standard spectrum and direct Yee integration for the Lumerical grating.')
    if grating_spectrum not in ('standard', 'dense'):
        raise ValueError('Choose a standard or dense grating spectrum.')
    if grating_spectrum=='dense' and (name!='lumerical' or resolution!=12):
        raise ValueError('The dense spectrum has a memory profile for the 12-cell Lumerical geometry only.')
    if grating_material_model not in ('volume-average','direct-yee'):
        raise ValueError('Unknown grating material model.')
    if grating_material_model=='direct-yee' and name!='lumerical':
        raise ValueError('Direct Yee-cell integration is supported for the Lumerical curved grating.')
    if name == 'lumerical' and resolution in ((12,14,16,18,20) if mirrors else (12,14,16)):
        model=ansys.model(resolution)
        if grating_spectrum=='dense':
            model['grating_spectrum']='dense'
        if grating_material_model!='volume-average':
            model['grating_material_model']=grating_material_model
        if mirrors:
            model['grating_domain']='mirror-y'
        if resolution >= 16 and grating_material_model == 'direct-yee':
            model['grating_allocator'] = 'platform'
        return model
    matched = name == 'tidy3d'
    allowed=(12,14,16) if matched else (12,16)
    if name not in ('tidy3d', 'compact', 'compact-narrow', 'compact-radiation') or resolution not in allowed:
        raise ValueError('Choose a supported grating and mesh.')
    model = dict(geometry=name, resolution=resolution, period=.62, fill=.8, teeth=20 if matched else 8,
                width=10.8 if matched else (2. if name == 'compact-narrow' else 3.),
                guide_width=.5, taper=50. if matched else 3., height=.26, etch=.16,
                oxide_top=.68, oxide_bottom=2., n_si=3.47, n_oxide=1.44,
                beam_x=3.6 if matched else 1.8, waist=5.4 if matched else 1.5,
                angle_deg=14.5, source_z=1.18, buffer=.93, pml=.667,
                max_time_fs=2000. if matched else 750.,
                residual_slab_laterally_extended=matched)
    if mirrors:
        model['grating_domain']='mirror-y'
    if matched and resolution == 16:
        model['grating_allocator'] = 'platform'
    return model


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


def sampled_gaussian_fields(xs, ys, center_x_um, waist_um, angle_deg):
    """Fixed E/H amplitudes for the native tilted air-side Gaussian injection sheet.

    Project the same sampled transverse Gaussian onto the horizontal plane using
    linear interpolation, then normalize in deterministic float64 arithmetic.
    The FDTDX TFSF source supplies the staggered component time delays.
    """
    theta=math.radians(angle_deg)
    xx=np.asarray(xs,dtype=np.float64)
    yy=np.asarray(ys,dtype=np.float64)
    projected_x=center_x_um+(xx-center_x_um)*math.cos(theta)
    x_profile=np.interp(projected_x,xx,np.exp(-((xx-center_x_um)/waist_um)**2))
    y_profile=np.interp(-yy,yy,np.exp(-(yy/waist_um)**2))
    amplitude=x_profile[:,None]*y_profile[None,:]
    amplitude=amplitude/np.sqrt(np.sum(amplitude**2,dtype=np.float64))
    electric=np.zeros((3,len(xx),len(yy),1),dtype=np.float32)
    magnetic=np.zeros_like(electric)
    electric[1,:,:,0]=amplitude
    magnetic[0,:,:,0]=math.cos(theta)*amplitude
    magnetic[2,:,:,0]=-math.sin(theta)*amplitude
    return electric,magnetic


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def waist_source_type(m, lp):
    import jax.numpy as jnp
    from fdtdx.core.jax.pytrees import autoinit, frozen_field
    from fdtdx.objects.sources.linear_polarization import LinearlyPolarizedPlaneSource

    @autoinit
    class WaistSource(LinearlyPolarizedPlaneSource):
        waist_cells: float = frozen_field()
        center_x: float = frozen_field()
        center_y: float = frozen_field()

        def _get_center(self, key):
            return jnp.asarray((self.center_x,self.center_y),dtype=jnp.float32)

        def apply(self, key, inv_permittivities, inv_permeabilities, dispersive_c1=None,
                  dispersive_c2=None, dispersive_c3=None):
            if lp is not None:
                # A single fiber eigenmode has one propagation constant across its entire
                # transverse profile, including the evanescent tail. The launch sheet's
                # phase delays and H/E ratio use that neff. Forward() still evolves the
                # actual tilted core, cladding and physical three-component Yee grid.
                inv_permittivities=inv_permittivities.at[(slice(None),*self.grid_slice)].set(1/lp[2]**2)
            return LinearlyPolarizedPlaneSource.apply(self,key,inv_permittivities,inv_permeabilities,
                                                       dispersive_c1,dispersive_c2,dispersive_c3)

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

    return WaistSource


def memory_profile_name(m):
    qualifier = '-mirror' if m.get('grating_domain') == 'mirror-y' else ''
    return f"v014-grating-memory-{m['geometry']}{qualifier}-r{m['resolution']}.json"


def memory_admission(m, available_mib):
    record=json.loads((ROOT/'docs/verification/v012-grating-memory-budget.json').read_text())
    allocator = m.get('grating_allocator')
    samples = [v for v in record['refinements'] if v['geometry'] == m['geometry']]
    if allocator == 'platform' and m['resolution'] != 12:
        name = memory_profile_name(m)
        path = ROOT/'docs/verification'/name
        if not path.exists():
            raise ValueError(f"This grating mesh requires its measured GPU admission profile {name}.")
        profile = json.loads(path.read_text())
        required = profile['required_free_with_reserve_mib']
        matched = (profile['geometry'] == m['geometry'] and profile['resolution'] == m['resolution']
                   and profile['allocator'] == allocator and profile['domain'] == m.get('grating_domain','full')
                   and profile['material_model'] == m.get('grating_material_model', 'volume-average')
                   and profile['grating_spectrum'] == m.get('grating_spectrum', 'standard')
                   and profile['device'] == record['reference_runs'][m['geometry']]['shape_only']['gpu']
                   and set(profile['roles']) == {'reference', 'device'}
                   and required == max(v['estimate']['required_free_with_reserve_mib'] for v in profile['roles'].values())
                   and math.isfinite(required) and required > 0)
        if not matched:
            raise ValueError('The on-demand grating memory profile differs from the requested GPU model.')
    elif m['resolution'] == 12:
        required = 1024. + 1.2*samples[0]['baseline_native_peak_mib']
    else:
        required = next(v['required_free_with_reserve_mib'] for v in samples if v['resolution'] == m['resolution'])
    if allocator == 'platform' and os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR') != allocator:
        raise ValueError('This grating mesh requires the on-demand allocator before creating its CUDA client.')
    if m.get('grating_spectrum')=='dense':
        profile=json.loads((ROOT/'docs/verification/v013-grating-dense-memory.json').read_text())
        if profile['resolution']!=m['resolution'] or profile['wavelength_um']!=monitoring_wavelengths(m).tolist():
            raise ValueError('Dense GPU memory profile differs from the requested native monitors.')
        required=max(required,profile['required_free_with_reserve_mib'])
    if available_mib<required:
        raise ValueError(f"{m['geometry']} at {m['resolution']} cells/µm requires {math.ceil(required)} MiB free "
                         f"under the measured GPU budget, but {available_mib} MiB is available. Choose a smaller mesh or free GPU memory.")
    return {'available_mib':available_mib,'required_free_with_reserve_mib':required,'profile_device':record['reference_runs'][m['geometry']]['shape_only']['gpu'], 'allocator':allocator or 'default'}


def native_stage(m, reference, *, memory_plan=False, observe_one_step=False):
    worker_started=time.monotonic()
    if 'grating_allocator' in m:
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = m['grating_allocator']
    jax,jnp,fdtdx,gpu = _ports_runtime('gpu')
    # The optional planner builds fields only on host RAM, then passes abstract
    # GPU shapes to XLA. Actual electromagnetic acquisition always uses the GPU.
    target=jax.devices('cpu')[0] if memory_plan else gpu
    timer = StageTiming(target)
    if memory_plan:
        jax.config.update('jax_default_device',target)
    from fdtdx.fdtd.forward import forward

    hardware=cuda_identity.fingerprint(bytes.fromhex(os.environ['PHOTONICS_GRATING_DEVICE_SALT']))
    if gpu.id!=hardware['ordinal'] or gpu.device_kind!=hardware['name']:
        raise RuntimeError('The native JAX target differs from the CUDA device selected for the checkpoint.')
    admission=None
    if not memory_plan and not observe_one_step and m['geometry'] in ('lumerical','tidy3d'):
        admission=memory_admission(m,cuda_identity.available_mib())
        if 'grating_allocator' in m and gpu.device_kind != admission['profile_device']:
            raise RuntimeError('The on-demand grating memory profile is specific to its measured GPU model.')
    fiber = m['geometry']=='lumerical'
    guided = m['geometry']=='compact-radiation'
    lp = ansys.lp01(m['fiber_radius'],m['fiber_core_n'],m['fiber_clad_n'],1.55) if fiber else None

    WaistSource = waist_source_type(m,lp)

    shape,pml,axes = grid(m)
    nx,ny,nz = shape
    full_shape=shape
    mirror=m.get('grating_domain')=='mirror-y'
    if mirror and m['geometry'] not in ('lumerical','tidy3d'):
        raise ValueError('The reduced domain requires a symmetric TE-illuminated grating.')
    mid=ny//2
    shift=mid if mirror else 0
    native_y0=0 if mirror else pml
    if mirror:
        shape=(nx, mid, nz)
    dx = 1/m['resolution']
    at = lambda axis,value: int(np.argmin(abs(axes[axis]-value)))
    wavelengths = monitoring_wavelengths(m)
    center_bin = wavelengths.tolist().index(1.55)
    frequencies = 1/wavelengths
    center = 299792458e6/1.55
    bandwidth = .5*299792458e6*(1/WL[0]-1/WL[-1])
    pulse_end = 12/(2*np.pi*bandwidth)*1e15
    sim = fdtdx.SimulationConfig(time=m['max_time_fs']*1e-15, resolution=dx*1e-6,
                                 backend='cpu' if memory_plan else 'gpu', dtype=jnp.float32, courant_factor=.99)
    if not memory_plan and (sim.backend!='gpu' or target.platform!='gpu'):
        raise RuntimeError('Requested NVIDIA FDTDX execution is unavailable.')
    volume = fdtdx.SimulationVolume(name='domain', partial_grid_shape=shape, material=fdtdx.Material(permittivity=1.))
    bkw={f'thickness_grid_{s}{a}':pml for s in ('min','max') for a in 'xyz'}
    bkw.update({f'boundary_type_{s}{a}':'pml' for s in ('min','max') for a in 'xyz'})
    if mirror:
        bkw.update(thickness_grid_miny=1,boundary_type_miny='pec')
    boundaries=fdtdx.BoundaryConfig(**bkw)
    objects, constraints = fdtdx.boundary_objects_from_config(boundaries, volume)
    items = [volume,*objects.values()]
    constraints=list(constraints)
    def add(item,start):
        items.append(item)
        constraints.append(item.set_grid_coordinates(axes=(0,1,2),sides=('-','-','-'),coordinates=start))
    source_half = m['source_half_span'] if fiber else 2*m['waist']
    xmin=max(pml,at(0,m['beam_x']-source_half))
    xmax=min(nx-pml,at(0,m['beam_x']+source_half)+1)
    source_shape = (xmax-xmin,ny-pml-native_y0-shift,1)
    source=WaistSource(name='beam',partial_grid_shape=source_shape, direction='-',
              fixed_E_polarization_vector=(0.,1.,0.), azimuth_angle=m['angle_deg'],
              normalize_by_energy=not fiber and m['geometry']!='tidy3d',
              center_x=(m['beam_x']-axes[0][xmin])/dx,center_y=-axes[1][pml]/dx,
              waist_cells=(m['fiber_radius'] if fiber else m['waist'])/dx, wave_character=fdtdx.WaveCharacter(frequency=center),
              temporal_profile=fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(frequency=bandwidth),
                                                          center_wave=fdtdx.WaveCharacter(frequency=center)),
              switch=fdtdx.OnOffSwitch(end_time=pulse_end*1e-15))
    if not guided:
        add(source,(xmin,native_y0,at(2,m['source_z'])))
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
    upper_n=grating_mirror.positive_count(pml,ny-2*pml,mid) if mirror else ny-2*pml
    detector('upper',(ax1-ax0,upper_n,1),(ax0,native_y0,az))
    py=int(round((2.5 if fiber else 2)/dx)); pz=int(round((2.25 if fiber else 6*m['height'])/dx))
    px=at(0,m['output_x'] if fiber else -m['taper']-m['buffer']/2)
    sy=(ny-py)//2; sz=at(2,(.05 if fiber else m['height']/2)-pz*dx/2+dx/2)
    if guided:
        source=fdtdx.ModePlaneSource(name='guide',partial_grid_shape=(1,py,pz),mode_index=0,filter_pol='te',
                    direction='+', wave_character=fdtdx.WaveCharacter(frequency=center),
                    temporal_profile=fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(frequency=bandwidth),
                                                                center_wave=fdtdx.WaveCharacter(frequency=center)),
                    switch=fdtdx.OnOffSwitch(end_time=pulse_end*1e-15))
        add(source,(at(0,-m['taper']-.8),sy,sz))
    if guided or not reference:
        port_n=grating_mirror.positive_count(sy,py,mid) if mirror else py
        detector('output',(1,port_n,pz),(px,0 if mirror else sy,sz))
    if not reference:
        detector('center',(nx-2*pml,1,nz-2*pml),(pml,mid-shift,pml),(waves[center_bin],))
    energy = fdtdx.EnergyDetector(name='energy',partial_grid_shape=(nx-2*pml,ny-shift-native_y0-pml,nz-2*pml),
                reduce_volume=True,exact_interpolation=True,plot=False,
                switch=fdtdx.OnOffSwitch(interval=max(1,round(4e-15/sim.time_step_duration))))
    add(energy,(pml,native_y0,pml))
    key = jax.random.PRNGKey(0)
    obj,arrays,params,sim,_=fdtdx.place_objects(object_list=items,config=sim,constraints=constraints,key=key)
    eps = material(m,reference and not guided)
    if reference and guided:
        eps = np.broadcast_to(eps[px:px+1],shape).copy()
    eps_hash=digest(eps)
    staggered=aligned(eps)
    if not reference and m.get('grating_material_model')=='direct-yee':
        ansys.silicon_on_yee_grid(m,axes,staggered)
    component_hash=tuple(digest(component) for component in staggered)
    port_section=1/staggered[:,px:px+1,sy:sy+py,sz:sz+pz] if (mirror and not reference) else None
    staggered=staggered[:,:,shift:,:]
    arrays=arrays.aset('inv_permittivities',jax.device_put(1/staggered,target))
    arrays,obj,_=fdtdx.apply_params(arrays,obj,params,key)
    if arrays.fields.E.device!=target or not np.allclose(np.asarray(arrays.inv_permittivities),1/staggered,rtol=2e-6):
        raise RuntimeError('Grid placement or material assignment differs from requested NVIDIA study.')
    del staggered
    if guided:
        from fdtdx.core.physics.modes import compute_mode
        source=obj['guide']
        e,h,n=compute_mode(frequency=center,inv_permittivities=source._inv_permittivity,
                          inv_permeabilities=source._inv_permeability,resolution=dx*1e-6,
                          direction='+',mode_index=0,filter_pol='te',dtype=jnp.float32)
        if not m['n_oxide']<np.real(n)<m['n_si']:
            raise ValueError('The guided source mode is outside the physical index interval.')
        electric,magnetic,_=canonical_source_profile(e,h)
        source=source.aset('_E',jax.device_put(electric,target)).aset('_H',jax.device_put(magnetic,target))
        obj=obj.aset('object_list',[source if item.name=='guide' else item for item in obj.object_list])
    else:
        source=obj['beam']
        if m['geometry']=='tidy3d':
            electric,magnetic=sampled_gaussian_fields(axes[0][xmin:xmax],axes[1][pml:ny-pml],
                                                     m['beam_x'],m['waist'],m['angle_deg'])
            source=source.aset('_E',jax.device_put(electric,target)).aset('_H',jax.device_put(magnetic,target))
            obj=obj.aset('object_list',[source if item.name=='beam' else item for item in obj.object_list])
        else:
            electric,magnetic=np.asarray(source._E),np.asarray(source._H)
    delays_e=np.asarray(source._time_offset_E);delays_h=np.asarray(source._time_offset_H)
    if mirror:
        # Evaluate the identical full-width source on its original sheet, preserving the
        # projected LP01 profile, phase origin and FDTDX component-specific time lags.
        source_dims=(xmax-xmin,ny-2*pml,1)
        sheet=source.aset('_grid_slice_tuple',tuple((0,n) for n in source_dims))
        index=lp[2] if fiber else 1.
        sheet=sheet.apply(key,jnp.full((3,*source_dims),1/index**2,dtype=jnp.float32),1.0)
        if fiber:
            electric,magnetic=np.asarray(sheet._E),np.asarray(sheet._H)
        else:
            electric,magnetic=sampled_gaussian_fields(axes[0][xmin:xmax],axes[1][pml:ny-pml],
                                                     m['beam_x'],m['waist'],m['angle_deg'])
        delays_e,delays_h=np.asarray(sheet._time_offset_E),np.asarray(sheet._time_offset_H)
        for attribute,field in zip(('_E','_H','_time_offset_E','_time_offset_H'),(electric,magnetic,delays_e,delays_h)):
            source=source.aset(attribute,jax.device_put(field[:,:,mid-pml:,:],target))
        obj=obj.aset('object_list',[source if item.name=='beam' else item for item in obj.object_list])
    vector=np.cross(electric.squeeze().transpose(1,2,0).astype(np.float64),
                    magnetic.squeeze().transpose(1,2,0).astype(np.float64)).sum(axis=(0,1))
    direction=vector/np.linalg.norm(vector)
    expected=np.array([1.,0.,0.]) if guided else np.array([-np.sin(np.deg2rad(m['angle_deg'])),0.,-np.cos(np.deg2rad(m['angle_deg']))])
    if not np.allclose(direction,expected,atol=2e-4):
        raise RuntimeError(f'Injected beam points in unexpected direction: {direction}, expected {expected}')
    impedance_error=None
    if fiber:
        eabs=np.linalg.norm(electric,axis=0)
        habs=np.linalg.norm(magnetic,axis=0)
        impedance_error=float(np.max(abs(habs-lp[2]*eabs))/np.max(habs))
        if impedance_error>2e-4:
            raise RuntimeError('Tilted LP01 sheet does not carry its selected modal impedance.')
    basis = None
    if guided or not reference:
        section=jax.device_put(port_section,target) if mirror else arrays.inv_permittivities[:,px:px+1,sy:sy+py,sz:sz+pz]
        basis,meta = port_basis(section, frequencies,
                                frequencies*299792458e6, dx,sim.time_step_duration,m['n_oxide'],m['n_si'])
    def evolve(state,end):
        return jax.lax.while_loop(lambda s:s[0]<end,
            lambda s:forward(state=s,config=sim,objects=obj,key=key,record_detectors=True,
                             record_boundaries=False,simulate_boundaries=True),state)
    dt=sim.time_step_duration*1e15
    state=(jnp.asarray(0,dtype=jnp.int32),arrays)
    compile_start=time.monotonic()
    if memory_plan:
        from jax.sharding import SingleDeviceSharding
        sharding=SingleDeviceSharding(gpu)
        def abstract(a):
            array=jnp.asarray(a)
            return jax.ShapeDtypeStruct(array.shape,array.dtype,sharding=sharding,weak_type=array.weak_type)
        spec=jax.tree.map(abstract,state)
        end_spec=jax.ShapeDtypeStruct((),jnp.int32,sharding=sharding)
        jax.config.update('jax_default_device',gpu)
        program=jax.jit(evolve,donate_argnums=(0,),in_shardings=(sharding,sharding),out_shardings=sharding)
        compiled=program.lower(spec,end_spec).compile()
        devices=compiled.runtime_executable().local_devices()
        if not devices or any(device.platform!='gpu' for device in devices):
            raise RuntimeError('Memory inspection requires an actual GPU-compiled timestep.')
        stats=compiled.memory_analysis()
        names=('argument_size_in_bytes','output_size_in_bytes','alias_size_in_bytes','temp_size_in_bytes','generated_code_size_in_bytes')
        memory={name:int(getattr(stats,name)) for name in names}
        live=memory['argument_size_in_bytes']+memory['output_size_in_bytes']-memory['alias_size_in_bytes']+memory['temp_size_in_bytes']
        return {'geometry':m['geometry'],'resolution':m['resolution'],'shape':list(shape),'cells':math.prod(shape),
                'gpu':gpu.device_kind,'cpu_field_allocation_verified':arrays.fields.E.device.platform=='cpu',
                'executed_fdtd_steps':0,'compiled_cuda_bytes':memory,'compiled_live_bytes':live,
                'compilation_seconds':time.monotonic()-compile_start}
    with timer.compile():
        compiled=jax.jit(evolve,donate_argnums=(0,)).lower(state,jnp.asarray(1,dtype=jnp.int32)).compile()
    compile_seconds=time.monotonic()-compile_start
    if observe_one_step:
        if m['resolution'] not in (12, 14) or (m['resolution'] == 14 and m['geometry'] not in ('lumerical', 'tidy3d')):
            raise ValueError('One-step memory observations require a mesh with a published full native run.')
        state=compiled(state,jnp.asarray(1,dtype=jnp.int32))
        jax.block_until_ready(state)
        if int(state[0])!=1 or state[1].fields.E.device!=gpu:
            raise ValueError('Resource probe did not execute exactly one native GPU step.')
        stats=compiled.memory_analysis(); allocator=gpu.memory_stats()
        import subprocess
        import shutil
        smi_path=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
        smi=subprocess.check_output([smi_path,'--query-gpu=memory.free,memory.used,memory.total',
                                     '--format=csv,noheader,nounits'],text=True).strip().splitlines()
        if len(smi)!=1: raise ValueError('The memory observation needs a single unambiguous NVIDIA device.')
        free,used,total=[int(v) for v in smi[0].split(',')]
        return {'geometry':m['geometry'],'resolution':m['resolution'],'shape':list(shape),'cells':math.prod(shape),
                'executed_fdtd_steps':1,'gpu':gpu.device_kind,'global_free_mib':free,'global_used_mib':used,
                'global_total_mib':total,'allocator_bytes':dict(allocator or {}),
                'actual_compiled_live_bytes':stats.argument_size_in_bytes+stats.output_size_in_bytes-
                    stats.alias_size_in_bytes+stats.temp_size_in_bytes}
    schedule = np.flatnonzero(np.asarray(obj['energy']._is_on_at_time_step_arr))
    spectra=[]; history=[]; before=None; consecutive=0
    start=time.monotonic()
    timer.begin_steps()
    for end in range(math.ceil(60/dt),sim.time_steps_total+math.ceil(60/dt),math.ceil(60/dt)):
        end=min(end,sim.time_steps_total)
        with timer.step():
            state=compiled(state,jnp.asarray(end,dtype=jnp.int32))
            jax.block_until_ready(state)
        step, final=state
        upper=np.asarray(final.detector_states['upper']['phasor'])[0,:,:,:,:,0]
        if mirror:
            upper=grating_mirror.full_plane(upper,3,1,pml,ny-2*pml,mid)
        if reference and not guided:
            measure=-normal_flux(upper,dx)
        else:
            port=np.asarray(final.detector_states['output']['phasor'])[0]
            if mirror:
                port=grating_mirror.full_plane(port,3,1,sy,py,mid)
            amplitudes=modal_amplitudes(port,basis['mode_e'],basis['mode_h'],dx**2)[0]
            measure=amplitudes[:,0] if guided else amplitudes[:,1]
            if guided and not reference:
                # Track the radiated aperture independently of the straight incoming wave.
                measure=normal_flux(upper,dx)
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
    timer.end_steps()
    if final.fields.E.device!=target:
        raise RuntimeError('FDTD fields left the requested device.')
    data={'upper':upper,'upper_x_um':axes[0][ax0:ax1],'upper_y_um':axes[1][ay],
          'wavelength_um':wavelengths,'history_observable':np.array(spectra),'source_E':electric,'source_H':magnetic,
          'source_time_offset_E':delays_e,'source_time_offset_H':delays_h,
          'energy':energy_trace,'energy_time_fs':times,
          'material_yz':eps[px],'material_xy':eps[:,:,at(2,.18)],'material_xz':eps[:,ny//2,:],
          'grid_x_um':axes[0],'grid_y_um':axes[1],'grid_z_um':axes[2]}
    if guided or not reference:
        data.update(output=port,mode_e=basis['mode_e'],mode_h=basis['mode_h'],mode_neff=basis['mode_neff'],
                    output_amplitudes=modal_amplitudes(port,basis['mode_e'],basis['mode_h'],dx**2)[0])
    if not reference:
        center_plane=np.asarray(final.detector_states['center']['phasor'])[0,0]
        if mirror:
            center_plane=grating_mirror.full_plane(center_plane,2,0,mid,1,mid)
        data['center_fields']=center_plane[:,:,0,:]
    performance = timer.report(shape=shape, steps=int(step), dt_seconds=sim.time_step_duration,
                               spacing_um=dx, pml_cells=pml)
    performance['setup_before_compile_seconds']=compile_start-worker_started
    if mirror:
        performance['represented_full_shape']=list(full_shape)
        performance['boundary_faces']={'min_y':'PEC mirror, one clamped Yee plane',
                                       'remaining_five_faces':f'PML, {pml} cells per face'}
    detail=dict(energy_volume='native allocated non-PML region', full_shape=list(full_shape), grating_domain=m.get('grating_domain','full'), performance=performance, stage='reference' if reference else 'device',actual_device=str(final.fields.E.device),device_name=target.device_kind,
                shape=list(shape),grid_resolution=m['resolution'],pml_cells=pml,time_steps=int(step),settled=consecutive>=2,
                history=history,compile_seconds=compile_seconds,stepping_seconds=time.monotonic()-start,source_direction=direction.tolist(),
                source_eh_sha256=digest(np.stack([electric,magnetic])),
                source_time_offsets_sha256=digest(np.stack([delays_e,delays_h])),scalar_permittivity_sha256=eps_hash,
                electric_component_permittivity_sha256=component_hash,
                sample_plane_z_um=float(axes[2][az]),output_plane_x_um=float(axes[0][px]),
                sampled_vertical_um=[float(axes[2][sz]),float(axes[2][sz+pz-1])],
                port_basis=meta if guided or not reference else None,
                fiber_mode=None if not fiber else dict(approximation='Weak-guidance LP01 transverse circular-fiber envelope on the tilted source plane',
                                                        u=lp[0],w=lp[1],neff=lp[2],source_impedance_relative_error=impedance_error,source_z_um=float(axes[2][at(2,m['source_z'])])),
                native_worker_sha256=file_hash(__file__),cuda_identity=hardware,memory_admission=admission,
                versions={k:importlib.metadata.version(k) for k in ('fdtdx','jax','jaxlib','tidy3d','numpy','scipy')})
    return detail,data


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def stage_signature(identity):
    return hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()


def stage_archive(m, label, cache):
    gpu_lock=Path(os.environ.get('PHOTONICS_GPU_LOCK', '/tmp/photonics-v012-gpu.lock'))
    print(f"Waiting for NVIDIA device: {m['geometry']} {label}",flush=True)
    wait_start=time.monotonic()
    with gpu_lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        wait_seconds=time.monotonic()-wait_start
        return _locked_stage_archive(m,label,cache,wait_seconds)


def _locked_stage_archive(m, label, cache, wait_seconds):
    import subprocess
    salt=cuda_identity.cache_salt(cache)
    hardware=cuda_identity.fingerprint(salt)
    sources=[Path(__file__),ROOT/'backend/adapters/fdtdx_ports_adapter.py',ROOT/'backend/adapters/fdtdx_ports3d_adapter.py',ROOT/'backend/adapters/fdtdx3d_adapter.py',
             ROOT/'backend/adapters/fdtdx_adapter.py',ROOT/'scripts/grating_angular_spectrum.py',ROOT/'scripts/lumerical_grating3d.py',ROOT/'scripts/grating_emission.py',ROOT/'scripts/grating_cuda_identity.py',ROOT/'scripts/gpu_timing.py']
    if m['geometry'] in ('lumerical', 'tidy3d'):
        sources.append(ROOT/'docs/verification/v012-grating-memory-budget.json')
        if m.get('grating_allocator') == 'platform' and m['resolution'] != 12:
            profile = ROOT/'docs/verification'/memory_profile_name(m)
            if not profile.exists():
                raise ValueError(f"The selected mesh and domain require the measured memory profile {profile.name}.")
            sources.append(profile)
        if m.get('grating_spectrum') == 'dense':
            sources.append(ROOT/'docs/verification/v013-grating-dense-memory.json')
    sources.append(ROOT/"scripts/grating_mirror.py")
    identity={'settings':m,'stage':label,'cuda_identity':hardware,'sources':{p.name:file_hash(p) for p in sources},
              'versions':{name:importlib.metadata.version(name) for name in ('fdtdx','jax','jaxlib','numpy','tidy3d','scipy')}}
    signature=stage_signature(identity)
    dest=cache/(label+'-'+signature[:20]); cache.mkdir(exist_ok=True,parents=True)
    with (cache/(label+'-'+signature[:20]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        fresh = not dest.exists()
        if fresh:
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                staged=Path(tmp)/'stage'; staged.mkdir()
                cmd=[sys.executable,str(Path(__file__).resolve()),'--device','gpu','--geometry',m['geometry'],
                     '--resolution',str(m['resolution']),'--stage',label,'--output',str(staged)]
                if 'grating_material_model' in m:
                    cmd.extend(('--grating-material-model',m['grating_material_model']))
                if m.get('grating_allocator') == 'platform':
                    cmd.append('--on-demand-memory')
                if m.get('grating_domain')=='mirror-y': cmd.extend(('--grating-domain','mirror-y'))
                if 'grating_spectrum' in m:
                    cmd.extend(('--grating-spectrum',m['grating_spectrum']))
                # Workers are separate processes so native device memory is released after each stage.
                env={**os.environ,'PHOTONICS_GRATING_DEVICE_SALT':salt.hex()}
                print(f"Running {m['geometry']} {label} on NVIDIA",flush=True)
                subprocess.run(cmd,check=True,env=env)
                receipt=json.loads((staged/'receipt.json').read_text())
                cuda_identity.validate_worker(hardware,receipt.get('cuda_identity'))
                receipt['metadata_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,allow_nan=False).encode()).hexdigest()
                receipt.update(identity=identity,signature=signature,arrays_sha256=file_hash(staged/'arrays.npz'))
                (staged/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
                staged.rename(dest)
        receipt=json.loads((dest/'receipt.json').read_text())
        cuda_identity.validate_worker(hardware,receipt.get('cuda_identity'))
        native={k:v for k,v in receipt.items() if k not in ('metadata_sha256','identity','signature','arrays_sha256')}
        measured_metadata=hashlib.sha256(json.dumps(native,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if measured_metadata!=receipt.get('metadata_sha256'):
            raise ValueError('Native FDTDX stage metadata changed.')
        if receipt.get('native_worker_sha256')!=identity['sources']['calibrate_finite_grating.py']:
            raise ValueError('The worker changed during native acquisition.')
        if receipt['identity']!=identity or receipt['signature']!=signature or file_hash(dest/'arrays.npz')!=receipt['arrays_sha256']:
            raise ValueError(f'Native stage failed provenance checks: {label}')
        return dest, dict(receipt, acquisition={'status':'fresh' if fresh else 'cache-hit',
                                                'gpu_lock_wait_seconds':wait_seconds})


def run(m, output, cache):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'report.json').exists() or (output/'arrays.npz').exists():
        raise ValueError('Choose a fresh result directory.')
    wavelengths=monitoring_wavelengths(m)
    center_bin=wavelengths.tolist().index(1.55)
    ref,rr=stage_archive(m,'reference',cache)
    dev,dr=stage_archive(m,'device',cache)
    for key in ('source_eh_sha256','source_time_offsets_sha256','device_name','actual_device','cuda_identity'):
        if rr[key]!=dr[key]:
            raise ValueError(f'Reference and device acquisitions differ in {key}.')
    if m['geometry']=='compact-radiation':
        from scripts.grating_emission import publish
        beam,br=stage_archive(settings('compact',m['resolution']),'reference',cache)
        if br['actual_device']!=dr['actual_device'] or br['device_name']!=dr['device_name']:
            raise ValueError('The receiving Gaussian basis must be measured on the selected hardware.')
        return publish(m,output,ref,rr,dev,dr,beam,br)
    with np.load(ref/'arrays.npz',allow_pickle=False) as r,np.load(dev/'arrays.npz',allow_pickle=False) as d:
        stored={**{f'ref_{k}':r[k] for k in r.files},**{f'dev_{k}':d[k] for k in d.files}}
        np.testing.assert_array_equal(r['wavelength_um'],wavelengths)
        np.testing.assert_array_equal(d['wavelength_um'],wavelengths)
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
    published_lumerical=lum['samples']['S21']['power']
    if m.get('grating_spectrum')=='dense':
        source=json.loads((ROOT/'docs/verification/v013-grating-dense-reference.json').read_text())
        if source['s_file_sha256']!=lum['s_file_sha256'] or source['wavelength_um']!=wavelengths.tolist():
            raise ValueError('The dense comparison does not match the verified commercial reference and monitor grid.')
        published_lumerical=source['published_s21_power']
    lcomparison={'vendor':'Lumerical','url':ANSYS_URL,'geometry_matched':m['geometry']=='lumerical',
                 'wavelength_um':wavelengths.tolist(),'published_s21_power':published_lumerical,
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
            'result':{'resolution':m['resolution'],'wavelength_um':wavelengths.tolist(),'beam_to_guide':efficiency.tolist(),
                      'center_beam_to_guide':float(efficiency[center_bin]),'valid_beam_bins':valid.tolist(),
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
    parser.add_argument('--resolution',type=int,choices=(12,14,16,18,20),default=12)
    parser.add_argument('--geometry',choices=('compact','compact-narrow','tidy3d','lumerical','compact-radiation'),default='compact')
    parser.add_argument('--grating-material-model',choices=('volume-average','direct-yee'),default='volume-average',
                        help='Use volume averaging or integrate the Lumerical silicon directly on the Yee cells.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grating-spectrum',choices=('standard','dense'),default='standard',
                        help='Measure the published five samples, or 21 native wavelengths on the 12-cell Lumerical grid.')
    parser.add_argument('--grating-domain',choices=('full','mirror-y'),default='full')
    parser.add_argument('--on-demand-memory',action='store_true',
                        help='Use the measured on-demand CUDA allocator for the 12-cell control or a qualified 16-cell grating.')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--stage',choices=('reference','device'),help=argparse.SUPPRESS)
    args=parser.parse_args(); m=settings(args.geometry,args.resolution,args.grating_material_model,args.grating_spectrum,args.grating_domain)
    if args.on_demand_memory:
        if m['geometry'] not in ('lumerical', 'tidy3d') or (m['resolution'] != 12 and 'grating_allocator' not in m):
            parser.error('On-demand grating memory is qualified for the matched 12-cell controls and selected 16-cell models.')
        m['grating_allocator']='platform'
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
