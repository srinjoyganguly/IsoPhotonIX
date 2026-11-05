"""Meep Maxwell evolution of the common staircase permittivity field."""
import math
import platform
import time
from pathlib import Path

import numpy as np
from ..layout_grid import rasterize
from .meep3d_adapter import C_UM_PER_FS
from ..volume import validate_volume


def run(config, output_dir):
    if (config.get('schema_version'),config.get('study_type'),config.get('solver'),config.get('device')) != (6,'layout-3d','meep','cpu'):
        raise ValueError('Select Meep CPU for a finite-layout field study.')
    import meep as mp
    started = time.perf_counter()
    spec, axes, epsilon = rasterize(config)
    counts = spec['grid']
    d = [spec['domain'][a] for a in 'xyz']
    resolution = config['resolution']
    # The callback supplies the saved piecewise-constant voxel model to the
    # native Yee update. get_array(Dielectric) reports Meep's sampled material.
    def eps_at(point):
        i = math.floor(point.x*resolution + counts[0]/2 + 1e-10)
        j = math.floor(point.y*resolution + counts[1]/2 + 1e-10)
        k = math.floor(point.z*resolution + counts[2]/2 + 1e-10)
        if 0 <= i < counts[0] and 0 <= j < counts[1] and 0 <= k < counts[2]:
            return float(epsilon[i,j,k])
        return config['cladding_index']**2

    wavelength = config['wavelength_um']
    def current(t):
        return -math.cos(2*math.pi*t/wavelength) * min(1., max(0.,t/(4*wavelength)))
    point = mp.Vector3(*(spec['source_um'][a] for a in 'xyz'))
    mp.verbosity(0)
    sim = mp.Simulation(cell_size=mp.Vector3(*d), dimensions=3, resolution=resolution,
        epsilon_func=eps_at, eps_averaging=False, boundary_layers=[mp.PML(config['pml_um'])],
        sources=[mp.Source(mp.CustomSource(current,start_time=0,center_frequency=1/wavelength),
                           component=mp.Ey,center=point)], Courant=.5)
    try:
        sim.init_sim()
        initialized = time.perf_counter()
        sim.run(until=config['duration_fs']*C_UM_PER_FS)
        vol = mp.Volume(center=mp.Vector3(),size=mp.Vector3(*d))
        fields = np.stack([np.asarray(sim.get_array(component=c,vol=vol),dtype=np.float32).transpose(2,1,0)
                           for c in (mp.Ex,mp.Ey,mp.Ez)])
        actual = np.asarray(sim.get_array(component=mp.Dielectric,vol=vol),dtype=np.float32).transpose(2,1,0)
        x,y,z,_=sim.get_array_metadata(vol=vol)
        expected = epsilon.transpose(2,1,0)
        if actual.shape != expected.shape:
            raise RuntimeError('Meep output coordinates do not cover the compiled physical grid.')
        for native, design in zip((x,y,z),axes):
            if np.max(abs(native-design)) > .51/resolution:
                raise RuntimeError('Meep native samples are displaced by more than half a grid cell.')
        bulk = np.ones(expected.shape, dtype=bool)
        for iz in (-1,0,1):
            for iy in (-1,0,1):
                for ix in (-1,0,1):
                    bulk &= expected == np.roll(expected,(iz,iy,ix),(0,1,2))
        bulk_error = float(np.max(abs(actual[bulk]-expected[bulk])))
        if bulk_error > 1e-5:
            raise RuntimeError(f'Meep dielectric differs inside homogeneous grid regions by {bulk_error}.')
        arrays=validate_volume({'fields':fields,'field':fields[1],'epsilon':actual,'x_um':x,'y_um':y,'z_um':z})
        # Preserve the requested physical raster separately from the sampled
        # dielectric at interpolated output locations (particularly interfaces).
        output = Path(output_dir)
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'arrays.npz',**arrays, design_epsilon=epsilon.transpose(2,1,0),
                            **{f'design_{a}_um':v for a,v in zip('xyz',axes)})
        actual_time,steps = float(sim.meep_time()/C_UM_PER_FS),int(sim.timestep())
    finally:
        sim.reset_meep()
    return {'solver':'meep','device':'cpu','actual_device':'CPU','device_name':'CPU','dimensions':'3D',
        'versions':{'meep':mp.__version__,'numpy':np.__version__,'python':platform.python_version()},
        'field_component':'Ey','field_components':['Ex','Ey','Ez'],'field_shape':list(fields[1].shape),
        'grid_shape':counts,'field_units':'arbitrary point-current source units','array_order':'component,z,y,x',
        'precision':'Meep native evolution, float32 exported fields','actual_duration_fs':actual_time,'time_steps':steps,
        'field_sampling':'Native Meep final-time get_array electric fields on common output coordinates.',
        'material_sampling':'Cell-centre staircase, no subpixel smoothing. epsilon is the native sampled Meep dielectric; design_epsilon records the input grid.',
        'geometry_translation':spec,
        'material_grid_check':{'bulk_voxels':int(bulk.sum()),'maximum_bulk_error':bulk_error,
            'rule':'Compare only uniform 3x3x3 neighborhoods. Interface values include Meep interpolation.'},
        'source':{'type':'Ey point current, ramped sinusoid','requested_um':spec['requested_source_um'],
            'position_um':spec['source_um'],'wavelength_um':wavelength,'startup_periods':4},
        'boundaries':{'faces':'six PML','pml_um':config['pml_um']},
        'timings':{'initialization_s':initialized-started,'total_s':time.perf_counter()-started},
        'assumptions':['Finite as-drawn solids in homogeneous cladding. Fields include the near-source field.']}
