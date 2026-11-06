"""Measure propagation in an isolated curved guide with the same cross-section as each 3D ring."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.check_ring_gap_modes import boundaries

BASE = Path(__file__).resolve().parents[1]/'docs/verification'
WAVES = np.array([1.548, 1.549, 1.55, 1.551, 1.552])
MODE_DOC = 'https://docs.flexcompute.com/projects/tidy3d/en/v2.11.2/api/_autosummary/tidy3d.ModeSpec.html'


def guided_mode(data, model, span):
    """Track the fundamental quasi-TE branch and reject box/PML solutions."""
    field = data.Ey.squeeze('x').transpose('f','mode_index','y','z')
    fields = np.asarray(field)
    ys, zs = np.asarray(field.y), np.asarray(field.z)
    area = np.gradient(ys)[:, None] * np.gradient(zs)[None, :]
    central = ((abs(ys)[:, None] < model['width_um']/2+.15) &
               (abs(zs)[None, :] < model['height_um']/2+.15))
    intensity = abs(fields)**2 * area
    total = intensity.sum(axis=(2, 3))
    fraction = (intensity * central).sum(axis=(2, 3)) / total
    te = np.asarray(data.pol_fraction.te)
    n = np.asarray(data.n_complex)
    usable = (te>.8) & (fraction>.50) & (n.real>model['substrate_index'])
    if not np.all(np.any(usable, axis=1)):
        raise ValueError('No guided quasi-TE mode localized around the physical rail.')
    index = np.argmax(np.where(usable,n.real,-np.inf), axis=1)
    take = np.arange(len(WAVES)), index
    ey = fields[take]
    edge_power, total_power = np.zeros(len(WAVES)), np.zeros(len(WAVES))
    for component in (data.Ex,data.Ey,data.Ez):
        transverse=component.squeeze('x').transpose('f','mode_index','y','z')
        squared=abs(np.asarray(transverse)[take])**2
        edge=(abs(np.asarray(transverse.y))[:,None]>=.47*span) | (abs(np.asarray(transverse.z))[None,:]>=.47*span)
        edge_power+=(squared*edge).sum(axis=(1,2))
        total_power+=squared.sum(axis=(1,2))
    edge_fraction=edge_power/total_power
    if np.max(edge_fraction)>.01:
        raise ValueError('The selected mode carries significant electric intensity near the transverse boundary.')
    adjacent = np.abs((np.conj(ey[:-1])*ey[1:]*area).sum(axis=(1, 2)))**2 / (total[take][:-1]*total[take][1:])
    if adjacent.min()<.99:
        raise ValueError('Adjacent samples do not track the same field.')
    branch = n[take]
    def group_index(first,last):
        return branch[2].real - WAVES[2]*(branch[last].real-branch[first].real)/(WAVES[last]-WAVES[first])
    group = group_index(1,3)
    centroid = (intensity[take]*ys[None, :, None]).sum(axis=(1, 2)) / total[take]
    return dict(n_eff=branch.real.tolist(),imaginary_index=branch.imag.tolist(),quasi_te_fraction=te[take].tolist(),
                mode_index=index.tolist(),outer_electric_intensity_fraction=edge_fraction.tolist(),
                central_ey_intensity_fraction=fraction[take].tolist(),adjacent_squared_ey_overlap=adjacent.tolist(),
                center_wavelength_ey_centroid_um=float(centroid[2]),group_index=float(group),
                group_index_with_2nm_half_step=float(group_index(0,4)),
                isolated_round_trip_fsr_nm=float(1000*WAVES[2]**2/(2*np.pi*model['radius_um']*group)))


def measure(model, mesh, span, sign):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver

    td.config.simulation.use_local_subpixel=False
    width, height = model['width_um'], model['height_um']
    y = boundaries([-span/2,-width/2,width/2,span/2],mesh)
    z = boundaries([-span/2,-height/2,height/2,span/2],mesh)
    grid = td.GridSpec(grid_x=td.UniformGrid(dl=1/mesh),grid_y=td.CustomGridBoundaries(coords=y),
                       grid_z=td.CustomGridBoundaries(coords=z))
    def medium(key):
        return td.Medium(permittivity=model[key]**2)
    structures = [td.Structure(geometry=td.Box.from_bounds((-td.inf,-td.inf,-span),(td.inf,td.inf,-height/2)),
                               medium=medium('substrate_index')),
                  td.Structure(geometry=td.Box(size=(td.inf,width,height)),medium=medium('silicon_index'))]
    sim = td.Simulation(size=(.2,span,span),structures=structures,medium=medium('top_index'),grid_spec=grid,
                        run_time=1e-12,boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()))
    radius = sign * model['radius_um'] if sign else None
    bend = dict(bend_axis=1,bend_radius=radius) if radius else {}
    pml = round(.25*mesh)
    target=2.05 if model['top_index']==1 else 2.45
    modes = td.ModeSpec(num_modes=2,target_neff=target,num_pml=(pml,pml),precision='double',**bend)
    data = ModeSolver(simulation=sim,plane=td.Box(size=(0,span,span)),freqs=(td.C_0/WAVES).tolist(),
                      mode_spec=modes).data
    np.testing.assert_allclose(np.asarray(data.n_complex.f), td.C_0/WAVES, rtol=1e-13)
    return dict(cells_per_um=mesh,span_um=span,signed_bend_radius_um=radius,pml_cells=pml,
                **guided_mode(data,model,span))


def main():
    import tidy3d as td
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,choices=range(1,5),default=1)
    parser.add_argument('--max-mesh',type=int,choices=(64,96),default=64)
    args=parser.parse_args()
    td.config.simulation.use_local_subpixel=False
    result=dict(tidy3d=td.__version__,source=MODE_DOC,solver='Local CPU 2D vector bent-guide eigensolver',
                wavelengths_um=WAVES.tolist(),local_subpixel=False,
                scope='Isolated constant-index bend. Round-trip spacing is the group-index estimate for the centerline circumference. '
                      'The calculation omits both buses and their coupling phase. No broadband power, loaded poles or loss are inferred.',
                cases={})
    for vendor in ('lumerical','tidy3d'):
        receipt=BASE/f'v014-ring3d-{vendor}-r20-t{3200 if vendor=="lumerical" else 4800}-wall-tensor.json'
        native=json.loads(receipt.read_bytes())
        model=native['physical_model']
        case=dict(source_receipt=receipt.name,source_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                  model={k:model[k] for k in ('width_um','height_um','radius_um','top_index','substrate_index','silicon_index')},
                  loaded_workbench_fsr_nm=native['measured_drop_fsr_nm'],
                  published_loaded_fsr_nm=native['published_drop_fsr_nm'],samples=[])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            settings=[(mesh,span,sign) for mesh,span in [(24,2.5),(32,2.5),(48,2.5),(64,2.5),(96,2.5),(64,3.)] if mesh<=args.max_mesh
                      for sign in (0,1,-1)]
            jobs=[pool.submit(measure,model,*setting) for setting in settings]
            for (mesh,span,sign),job in zip(settings,jobs):
                sample=job.result()
                case['samples'].append(sample)
                print(vendor,mesh,span,sign,sample['n_eff'][2],sample['isolated_round_trip_fsr_nm'],flush=True)
        result['cases'][vendor]=case
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
