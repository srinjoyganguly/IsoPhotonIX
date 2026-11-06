"""Reproduce the Lumerical ring design's 100 nm parallel-guide supermode splitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.ring_palik_dispersion import material_samples, MATERIAL_SOURCE_SHA256

URL = 'https://optics.ansys.com/hc/en-us/articles/360042800293-Ring-resonator-getting-started-Design-and-initial-simulation'
WAVE, WIDTH, HEIGHT = 1.55, .4, .18
REFERENCE = .109


def boundaries(cuts, mesh):
    return np.concatenate([*(np.linspace(a,b,max(1, round((b-a)*mesh))+1)[:-1]
                             for a,b in zip(cuts,cuts[1:])), [cuts[-1]]])


def tidy(mesh, gap, span, si, oxide):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    td.config.simulation.use_local_subpixel = False
    y = boundaries([-span/2,-gap/2-WIDTH,-gap/2,0,gap/2,gap/2+WIDTH,span/2], mesh)
    z = boundaries([-span/2,-HEIGHT/2,HEIGHT/2,span/2], mesh)
    grid = td.GridSpec(grid_x=td.UniformGrid(dl=1/mesh), grid_y=td.CustomGridBoundaries(coords=y),
                       grid_z=td.CustomGridBoundaries(coords=z))
    sub = td.Structure(geometry=td.Box.from_bounds((-td.inf,-td.inf,-span),(td.inf,td.inf,-HEIGHT/2)),
                       medium=td.Medium(permittivity=oxide))
    rails = [td.Structure(geometry=td.Box(size=(td.inf,WIDTH,HEIGHT),center=(0,c,0)),medium=td.Medium(permittivity=si))
             for c in (-(WIDTH+gap)/2,(WIDTH+gap)/2)]
    sim = td.Simulation(size=(.2,span,span),structures=[sub,*rails],grid_spec=grid,run_time=1e-12,
                        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()))
    data = ModeSolver(simulation=sim,plane=td.Box(size=(0,span,span)),freqs=[td.C_0/WAVE],
                      mode_spec=td.ModeSpec(num_modes=4,target_neff=2.1,precision='double')).data
    n=np.asarray(data.n_eff).ravel()
    te=np.asarray(data.pol_fraction.te).ravel()
    indices=np.flatnonzero((te>.65)&(n>np.sqrt(oxide)))
    if len(indices)!=2:
        raise ValueError(f'Expected two guided quasi-TE supermodes, received {n}, TE fractions {te}.')
    rows=[]
    for i in indices:
        field=data.Ey.isel(f=0,mode_index=int(i)).squeeze(drop=True).transpose('y','z')
        # Interpolate at mirror locations to honor Yee field staggering.
        amp=[complex(field.interp(y=c,z=0).item()) for c in (-(WIDTH+gap)/2,(WIDTH+gap)/2)]
        coherence=float(np.real(amp[0]*np.conj(amp[1])) / abs(amp[0]*amp[1]))
        rows.append((float(n[i]),float(te[i]),coherence))
    return paired(rows)


def mpb_modes(mesh, gap, span, si, oxide):
    import meep as mp
    from meep import mpb
    mp.verbosity(0)
    substrate_height=(span-HEIGHT)/2
    objects=[mp.Block(size=mp.Vector3(mp.inf,mp.inf,substrate_height),
                      center=mp.Vector3(z=-HEIGHT/2-substrate_height/2),material=mp.Medium(epsilon=oxide))]
    for c in (-(WIDTH+gap)/2,(WIDTH+gap)/2):
        objects.append(mp.Block(size=mp.Vector3(mp.inf,WIDTH,HEIGHT),center=mp.Vector3(y=c),material=mp.Medium(epsilon=si)))
    solver=mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0,span,span)),resolution=mesh,
                          geometry=objects,num_bands=2,deterministic=True,tolerance=1e-8,mesh_size=5)
    fields={}
    def sample(solver,band):
        e=np.asarray(solver.get_efield(band,bloch_phase=False)).reshape((round(span*mesh),round(span*mesh),3))
        energy=np.sum(abs(e)**2,axis=(0,1))
        points=[solver.get_field_point(mp.Vector3(y=c)).y for c in (-(WIDTH+gap)/2,(WIDTH+gap)/2)]
        fields[band]=(float(energy[1]/(energy[1]+energy[2])),
                      float(np.real(points[0]*np.conj(points[1]))/abs(points[0]*points[1])))
    f=1/WAVE
    k=solver.find_k(mp.NO_PARITY,f,1,2,mp.Vector3(1,0,0),1e-8,2*f,np.sqrt(oxide)*f*1.0001,np.sqrt(si)*f,sample)
    return paired([(float(k[i]/f),*fields[i+1]) for i in range(2)])


def fdtdx_modes(mesh, gap, span, si, oxide, wall):
    import jax
    import jax.numpy as jnp
    from fdtdx.core.physics.modes import compute_mode
    from scripts.calibrate_loaded_ring3d import disk_fractions, lateral_fraction, permittivity
    from scripts.ring_wall_tensor import wall_inverse
    size=round(mesh*span)
    cross=dict(shape=(1,size,size),resolution=mesh,radius_um=3.1,width_um=WIDTH,height_um=HEIGHT,
               silicon_index=np.sqrt(si),substrate_index=np.sqrt(oxide),top_index=1.,
               bus_y_um=(gap+WIDTH)/2,ring_interface='exact-area')
    # Use the same stepped coefficients as the native ring, retaining only straight rails.
    def sampled(model, reference):
        return permittivity(model,reference,si,oxide)
    inv=(wall_inverse(cross,True,disk_fractions,lateral_fraction,sampled)[[0,4,8]]
         if wall else 1/sampled(cross,True))
    rows=[]
    for mode in (0,1):
        e,_,n=compute_mode(frequency=299792458e6/WAVE,inv_permittivities=jnp.asarray(inv),inv_permeabilities=1.,
                           resolution=1e-6/mesh,direction='+',mode_index=mode,filter_pol='te',dtype=jnp.float32)
        if e.device.platform!='cpu':
            raise ValueError('Transverse control must identify its actual local CPU.')
        energy=np.sum(abs(np.asarray(e))**2,axis=(1,2,3))
        points=[complex(e[1,0,round(size/2+c*mesh-.5),size//2]) for c in (-(WIDTH+gap)/2,(WIDTH+gap)/2)]
        rows.append((float(np.real(n)),float(energy[1]/(energy[1]+energy[2])),
                     float(np.real(points[0]*np.conj(points[1]))/abs(points[0]*points[1]))))
    if min(row[0] for row in rows)<=np.sqrt(oxide):
        raise ValueError('The second mode is below the substrate light line.')
    return paired(rows)


def paired(rows):
    rows=sorted(rows, reverse=True)
    if min(row[1] for row in rows)<.65 or rows[0][2]<.98 or rows[1][2]>-.98:
        raise ValueError(f'Quasi-TE mode parity failed: {rows}')
    n=rows[0][0]-rows[1][0]
    return dict(even_neff=rows[0][0],odd_neff=rows[1][0],even_odd_te_fraction=[row[1] for row in rows],
                rail_center_phase_cosines=[row[2] for row in rows],delta_n=n,
                length_um_for_amplitude_0_31=WAVE*np.arcsin(.31)/(np.pi*n))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=('tidy3d','mpb','fdtdx-area','fdtdx-wall'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.engine=='tidy3d':
        import tidy3d as engine
        solve=tidy
    elif args.engine.startswith('fdtdx'):
        from importlib.metadata import version
        from functools import partial
        solve=partial(fdtdx_modes,wall=args.engine=='fdtdx-wall')
        version_string=version('fdtdx')
    else:
        import meep as engine
        solve=mpb_modes
    si,oxide=(float(row[2]) for row in material_samples().values())
    result=dict(source=URL,material_audit_sha256=MATERIAL_SOURCE_SHA256,
                reference_delta_n=REFERENCE,wavelength_um=WAVE, width_um=WIDTH,height_um=HEIGHT,
                silicon_epsilon=si,oxide_epsilon=oxide,engine=args.engine,version=version_string if args.engine.startswith('fdtdx') else engine.__version__,
                observable='Even minus odd quasi-TE index for two parallel straight rails; ring tangent section.',
                rows=[])
    settings=[(32,.1,2.5),(64,.1,2.5),(96,.1,2.5),(128,.1,2.5),(64,.1,3.),(64,.09,2.5),(64,.11,2.5)]
    if args.engine.startswith('fdtdx'):
        settings=[(16,.1,2.5),(20,.1,2.5),(24,.1,2.5),*settings[:-3]]
    for mesh,gap,span in settings:
        row=dict(mesh_cells_per_um=mesh,gap_um=gap,span_um=span,**solve(mesh,gap,span,si,oxide))
        if gap==.1:
            row['difference_from_published_delta_n']=row['delta_n']-REFERENCE
        result['rows'].append(row)
        print(args.engine,mesh,gap,span,row['delta_n'],flush=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
