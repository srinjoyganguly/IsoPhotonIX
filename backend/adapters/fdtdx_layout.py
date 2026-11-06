"""FDTDX study on the common passive voxel grid and explicitly selected device."""
from pathlib import Path
import importlib.metadata
import time
import numpy as np

from ..layout_grid import rasterize
from .fdtdx_adapter import _runtime
from .fdtdx_taper import _save_fields
from scripts.gpu_timing import StageTiming


def run(config, output_dir):
    if (config.get('schema_version'),config.get('study_type'),config.get('solver')) != (6,'layout-3d','fdtdx'):
        raise ValueError('Select an FDTDX target for a finite-layout field study.')
    started = time.perf_counter()
    spec, axes, epsilon = rasterize(config)
    jax,jnp,fdtdx,target = _runtime(config['device'])
    timer=StageTiming(target)
    from fdtdx.core.jax.pytrees import autoinit, field
    from fdtdx.objects.static_material.static import StaticMultiMaterialObject
    from fdtdx.materials import compute_ordered_names

    @autoinit
    class SceneMaterial(StaticMultiMaterialObject):
        mapping: jax.Array = field()

        def get_voxel_mask_for_shape(self):
            return jnp.ones(self.mapping.shape, dtype=bool)

        def get_material_mapping(self):
            return self.mapping

    shape = tuple(spec['grid'])
    dx = 1/config['resolution']
    values, inverse = np.unique(epsilon,return_inverse=True)
    materials = {str(i):fdtdx.Material(permittivity=float(value)) for i,value in enumerate(values)}
    order = compute_ordered_names(materials)
    remap = np.array([order.index(str(i)) for i in range(len(values))],dtype=np.int32)
    mapping = jax.device_put(remap[inverse.reshape(shape)],target)
    scene = SceneMaterial(name='layout',partial_grid_shape=shape,materials=materials,mapping=mapping)
    sim = fdtdx.SimulationConfig(time=config['duration_fs']*1e-15,resolution=dx*1e-6,
        backend=config['device'],dtype=jnp.float32,courant_factor=.99)
    if sim.backend != config['device']:
        raise RuntimeError('FDTDX changed the requested target.')
    volume = fdtdx.SimulationVolume(name='domain',partial_grid_shape=shape,
        material=fdtdx.Material(permittivity=config['cladding_index']**2))
    pml=round(config['pml_um']*config['resolution'])
    bc=fdtdx.BoundaryConfig(**{f'thickness_grid_{s}{a}':pml for s in ('min','max') for a in 'xyz'},
                            **{f'boundary_type_{s}{a}':'pml' for s in ('min','max') for a in 'xyz'})
    bounds,constraints=fdtdx.boundary_objects_from_config(bc,volume)
    source=fdtdx.PointDipoleSource(name='current',partial_grid_shape=(1,1,1),polarization=1,
        temporal_profile=fdtdx.SingleFrequencyProfile(num_startup_periods=4),
        wave_character=fdtdx.WaveCharacter(wavelength=config['wavelength_um']*1e-6))
    constraints = [*constraints,scene.set_grid_coordinates(axes=(0,1,2),sides=('-','-','-'),coordinates=(0,0,0)),
                   source.set_grid_coordinates(axes=(0,1,2),sides=('-','-','-'),coordinates=tuple(spec['source_grid_index']))]
    key=jax.random.PRNGKey(0)
    objects,arrays,params,sim,_=fdtdx.place_objects(object_list=[volume,scene,source,*bounds.values()],
        config=sim,constraints=constraints,key=key)
    arrays,objects,_=fdtdx.apply_params(arrays,objects,params,key)
    actual=np.asarray(1/arrays.inv_permittivities[0])
    if arrays.fields.E.device != target or actual.shape!=shape or not np.allclose(actual,epsilon,rtol=1e-6,atol=1e-6):
        raise RuntimeError('The placed device or material grid differs from the validated layout.')
    initialized=time.perf_counter()
    with timer.compile():
        compiled=jax.jit(lambda a: fdtdx.run_fdtd(arrays=a,objects=objects,config=sim,key=key,show_progress=False)).lower(arrays).compile()
    timer.begin_steps()
    try:
        with timer.step():
            step,final=compiled(arrays)
            jax.block_until_ready((step,final))
    finally:
        timer.end_steps()
    if final.fields.E.device != target:
        raise RuntimeError('FDTDX result is on a different device.')
    field_shape,peaks=_save_fields(Path(output_dir),{'centres':axes,'dx':dx},final)
    return {'solver':'fdtdx','device':config['device'],'actual_device':str(target),'device_name':target.device_kind,
        'versions':{'fdtdx':importlib.metadata.version('fdtdx'),'jax':jax.__version__,'jaxlib':importlib.metadata.version('jaxlib'),'numpy':np.__version__},
        'dimensions':'3D','field_component':'Ey','field_components':['Ex','Ey','Ez'],'field_shape':field_shape,'field_peaks':peaks,
        'field_units':'arbitrary point-current source units','array_order':'component,z,y,x','grid_shape':list(shape),'precision':'float32',
        'simulated_time_fs':int(step)*sim.time_step_duration*1e15,'time_steps':int(step),'geometry_translation':spec,
        'source':{'type':'Ey soft electric point dipole, ramped sinusoid','position_um':spec['source_um'],
                  'requested_um':spec['requested_source_um'],'wavelength_um':config['wavelength_um'],'startup_periods':4},
        'boundaries':{'faces':'six PML','pml_cells':pml,'pml_um':pml*dx},'field_sampling':'Spatially averaged Yee samples, positive outermost layer omitted in display, raw fields also exported.',
        'material_sampling':'Scalar permittivity at cell centres. Checked every initialized voxel before evolving.',
        'material_grid_check':{'cells':int(actual.size),'maximum_absolute_error':float(np.max(abs(actual-epsilon)))},
        'performance':timer.report(shape=shape,steps=int(step),dt_seconds=sim.time_step_duration,spacing_um=dx,pml_cells=pml),
        'timings':{'initialization_s':initialized-started,'total_s':time.perf_counter()-started},
        'assumptions':['Finite as-drawn solids in homogeneous cladding. Fields include the near-source field.']}
