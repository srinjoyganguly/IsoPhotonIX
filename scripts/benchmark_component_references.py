"""Reproduce published coupler and strip observables with local MPB on CPU.

Run with the Meep/MPB Python. --output must be new. The report contains every
mesh and domain check. Sampled mode fields can be saved for inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.component_library import build_component

LUMERICAL = 'https://optics.ansys.com/hc/en-us/articles/360042304694-Evanescent-waveguide-couplers'
PROJECT_URL = 'https://optics.ansys.com/hc/en-us/article_attachments/360056406114'
TIDY3D = 'https://www.flexcompute.com/tidy3d/examples/notebooks/WaveguidePluginDemonstration/'
WAVELENGTH = 1.55
FREQUENCY_STEP = .005

# Scalar samples, not redistributed model coefficients. The optional extraction
# command in the guide evaluates the named materials in the upstream SDK.
PERMITTIVITY = {
    .995: (12.071312067444534, 2.124948290829251, .0006245144056152194),
    1.: (12.075663669334666, 2.1249983698346497, .0006277085542436616),
    1.005: (12.080040531703059, 2.1250487043548034, .0006309035508102008),
}
REFERENCES = {
    'lumerical-coupler': {
        'even_neff': ('Even supermode index', '1', 2.378022),
        'odd_neff': ('Odd supermode index', '1', 2.317864),
        'delta_neff': ('Supermode splitting', '1', .0601584),
        'transfer_length_um': ('Full-transfer length', 'µm', 12.8827),
    },
    'tidy3d-strip-plugin': {
        'te_neff': ('Quasi-TE effective index', '1', 2.44624476),
        'tm_neff': ('Quasi-TM effective index', '1', 1.7895686),
        'te_group': ('Quasi-TE group index', '1', 4.17353874),
        'tm_group': ('Quasi-TM group index', '1', 3.73989668),
        'te_area_um2': ('Quasi-TE mode area', 'µm²', .19145485),
        'tm_area_um2': ('Quasi-TM mode area', 'µm²', .35608204),
    },
}


def mode(case, label, frequency_ratio, resolution, span, output):
    import meep as mp
    from meep import mpb

    mp.verbosity(0)
    frequency = frequency_ratio / WAVELENGTH
    if case == 'lumerical-coupler':
        core_n, cladding_n, upper_n = 3.5, 1.5, 1.
        family, params = 'directional_coupler', {'width_um': .5, 'height_um': .2, 'gap_um': .05}
        parity = mp.ODD_Y if label == 'even' else mp.EVEN_Y
        expected_component = 1
        guess = 2.4
    else:
        eps_core, eps_clad, _ = PERMITTIVITY[frequency_ratio]
        core_n, cladding_n = np.sqrt(eps_core), np.sqrt(eps_clad)
        upper_n = cladding_n
        family, params = 'strip', {'width_um': .5, 'height_um': .22}
        parity = mp.ODD_Y + mp.EVEN_Z if label == 'te' else mp.EVEN_Y + mp.ODD_Z
        expected_component = 1 if label == 'te' else 2
        guess = 2.45 if label == 'te' else 1.79
    geometry = build_component(family, params)['geometry']
    objects = geometry['objects']
    ly, lz = span
    blocks = []
    if case == 'lumerical-coupler':
        bottom = -params['height_um']/2
        substrate_height = lz/2 + bottom
        blocks.append(mp.Block(center=mp.Vector3(0, 0, bottom - substrate_height/2),
            size=mp.Vector3(mp.inf, mp.inf, substrate_height), material=mp.Medium(index=cladding_n)))
    for obj in objects:
        pos, size = obj['position'], obj['size']
        if obj['kind'] != 'box' or obj['rotation_deg'] != 0:
            raise ValueError('The benchmark requires a rectangular longitudinally invariant section.')
        blocks.append(mp.Block(center=mp.Vector3(0,pos['y'],pos['z']),
            size=mp.Vector3(mp.inf,size['y'],size['z']), material=mp.Medium(index=core_n)))
    ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0,ly,lz)),
        geometry=blocks,default_material=mp.Medium(index=upper_n),resolution=resolution,
        num_bands=1,deterministic=True,tolerance=1e-10,mesh_size=3)
    field = {}
    def capture(solver, band):
        mpb.fix_efield_phase(solver, band)
        e = np.asarray(solver.get_efield(band,bloch_phase=False)).copy()
        eps = np.asarray(solver.get_epsilon()).copy()
        field.update(E=e.reshape(*eps.shape,3),epsilon=eps,frequency=float(solver.freqs[band-1]))
    k = ms.find_k(parity,frequency,1,1,mp.Vector3(1,0,0),1e-9,guess*frequency,
                  cladding_n*frequency*1.000001, core_n*frequency,capture)[0]
    neff = float(k/frequency)
    electric = field['E']
    component_norms = np.sum(abs(electric)**2,axis=(0,1))
    fraction = float(component_norms[expected_component]/component_norms.sum())
    residual = abs(field['frequency']-frequency)/frequency
    if not (np.isfinite(electric).all() and np.isfinite(field['epsilon']).all() and
            cladding_n < neff < core_n and fraction > .5 and residual < 1e-7):
        raise RuntimeError(f'Failed mode identification or fixed-frequency root for {case}, {label}.')
    # Check the signed dominant field, not just mode ordering by n_eff. MPB
    # symmetry labels act on polar vectors, while this overlap tests samples.
    dominant = electric[:, :, expected_component]
    reflected = dominant[(-np.arange(dominant.shape[0])) % dominant.shape[0]]
    mirror_overlap = float(np.vdot(dominant, reflected).real / np.vdot(dominant, dominant).real)
    expected_sign = -1 if label == 'odd' else 1
    if expected_sign * mirror_overlap < .97:
        raise RuntimeError(f'Unexpected lateral parity of {label} mode: {mirror_overlap}.')
    yz = [length*(np.arange(n)/n-.5) for length,n in zip(span,field['epsilon'].shape)]
    probes = {'core': ((-.275 if case == 'lumerical-coupler' else 0., 0.), core_n**2),
              'upper': ((0., .45), upper_n**2), 'lower': ((0., -.45), cladding_n**2)}
    if case == 'lumerical-coupler':
        probes['centre_gap'] = ((0., 0.), upper_n**2)
    actual_probes = {}
    for name, (point, expected) in probes.items():
        iy, iz = (int(np.argmin(np.abs(axis-coordinate))) for axis,coordinate in zip(yz,point))
        observed = float(field['epsilon'][iy, iz])
        if not np.isclose(observed, expected, rtol=1e-6, atol=1e-8):
            raise RuntimeError(f'{name} material sample disagrees with its intended medium.')
        actual_probes[name] = observed
    intensity = (abs(electric)**2).sum(axis=2)
    cell_area = ly*lz / intensity.size
    area = float(cell_area * intensity.sum()**2 / (intensity**2).sum())
    filename = f'{case}-{label}-r{resolution}-span{ly:g}x{lz:g}-f{frequency_ratio:g}.npz'
    if frequency_ratio == 1:
        np.savez_compressed(output/filename, electric=electric, epsilon=field['epsilon'],
            y_um=ly*(np.arange(intensity.shape[0])/intensity.shape[0]-.5),
            z_um=lz*(np.arange(intensity.shape[1])/intensity.shape[1]-.5))
    return {'neff':neff,'mode_area_um2':area,'expected_electric_component':['Ex','Ey','Ez'][expected_component],
        'component_l2_fraction':fraction,'fixed_frequency_relative_residual':residual,
        'dominant_field_lateral_mirror_overlap':mirror_overlap,'sampled_epsilon':actual_probes,
        'grid_shape_yz':list(intensity.shape),'frequency_ratio':frequency_ratio,
        'field_file':filename if frequency_ratio==1 else None,
        'field_sha256':hashlib.sha256((output/filename).read_bytes()).hexdigest() if frequency_ratio==1 else None}


def isolated_mode(case, label, ratio, resolution, span, output):
    """Release native MPB allocations after each high-resolution eigensolve."""
    target = output / f'{case}-{label}-r{resolution}-span{span[0]:g}x{span[1]:g}-f{ratio:g}.json'
    command = [sys.executable, str(Path(__file__).resolve()), '--output', str(output),
               '--sample', case, label, str(ratio), str(resolution), str(span[0]), str(span[1])]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RuntimeError(f'{target.stem} failed. {result.stderr[-1500:]}')
    return json.loads(target.read_text())


def calculate(case, resolution, span, output):
    quantities, modes = {}, {}
    if case == 'lumerical-coupler':
        for label in ('even','odd'):
            modes[label] = isolated_mode(case,label,1.,resolution,span,output)
            quantities[label+'_neff'] = modes[label]['neff']
        splitting = quantities['even_neff'] - quantities['odd_neff']
        if splitting <= 0:
            raise RuntimeError('Even and odd supermodes have unexpected ordering.')
        quantities.update(delta_neff=splitting,transfer_length_um=WAVELENGTH/(2*splitting))
    else:
        for label in ('te','tm'):
            samples = [isolated_mode(case,label,ratio,resolution,span,output) for ratio in PERMITTIVITY]
            modes[label] = samples
            n0 = samples[1]['neff']
            quantities[label+'_neff'] = n0
            quantities[label+'_group'] = n0 + (samples[2]['neff'] - samples[0]['neff'])/(2*FREQUENCY_STEP)
            quantities[label+'_area_um2'] = samples[1]['mode_area_um2']
    return {'resolution_cells_per_um':resolution,'transverse_span_um':list(span),'quantities':quantities,'modes':modes}


def describe(case):
    if case == 'lumerical-coupler':
        return {'id':case, 'title':'Uniform SOI directional coupler', 'vendor':'Ansys Lumerical MODE',
            'source_url':LUMERICAL,'source_kind':'Published FDE supermode indices and transfer length calculated from their splitting.',
            'project_url':PROJECT_URL,'project_sha256':'9985f1670a106394f38b0a9a53e59a24bd0b3d56629168744c9b9d0e4e551b33',
            'physics':{'wavelength_um':WAVELENGTH,'catalog_family':'directional_coupler','core_index':3.5,'substrate_index':1.5,'upper_index':1.,
                       'guide_width_um':.5,'height_um':.2,'edge_gap_um':.05},
            'limitations':['Uniform cross-section, real isotropic media. The finite coupler entrance, bends, radiation and output ports are excluded.',
                'The source FDE uses a local 2 nm gap mesh. MPB uses Fourier modes in a periodic transverse supercell. The enlarged cell also changes the material sampling.',
                'The source reports splitting with more precision than its rounded two individual indices. Each quantity uses its own published reference.',
                'Only the FDE 50 nm-gap case is compared. The source also discusses distinct adjusted-gap varFDTD runs and an EME propagation run.']}
    return {'id':case,'title':'Dispersive strip, quasi-TE and quasi-TM','vendor':'Tidy3D waveguide plugin',
        'source_url':TIDY3D,'source_kind':'Six scalar outputs printed by the waveguide plugin example.',
        'physics':{'wavelength_um':WAVELENGTH,'catalog_family':'strip','guide_width_um':.5,'height_um':.22,
                   'media':['cSi/Li1993_293K','SiO2/Horiba'],'sampled_with':'Tidy3D 2.11.2 material evaluation only',
                   'eps_samples':{str(r):{'core_real':v[0],'clad_real':v[1],'clad_imaginary':v[2]} for r,v in PERMITTIVITY.items()},
                   'group_frequency_step_fraction':FREQUENCY_STEP},
        'limitations':['MPB uses Re(ε) at each frequency. Horiba silica has a small nonzero imaginary part, retained in the input record; attenuation is outside this real-valued comparison.',
             'Effective area uses the full electric intensity integral on the MPB uniform grid. Discrete field sampling and finite solver windows differ.',
             'Published values belong to the notebook’s mode solver and its default grid. The reference does not specify the backend version or export corresponding field arrays.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--case',choices=['all',*REFERENCES],default='all')
    parser.add_argument('--resolutions',nargs='+',type=int,default=[40,80,160,256])
    parser.add_argument('--report',type=Path)
    parser.add_argument('--sample', nargs=6, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sample:
        case, label, ratio, resolution, ly, lz = args.sample
        ratio, resolution, ly, lz = float(ratio), int(resolution), float(ly), float(lz)
        if (case not in REFERENCES or label not in (('even','odd') if case == 'lumerical-coupler' else ('te','tm'))
                or ratio not in PERMITTIVITY or not 24 <= resolution <= 256 or not 2 <= min(ly,lz) <= max(ly,lz) <= 4):
            parser.error('Unknown or out-of-range internal mode sample.')
        record = mode(case,label,ratio,resolution,(ly,lz),args.output)
        target = args.output / f'{case}-{label}-r{resolution}-span{ly:g}x{lz:g}-f{ratio:g}.json'
        target.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        return
    resolutions=args.resolutions
    if len(resolutions)<2 or sorted(set(resolutions))!=resolutions or not all(24<=r<=256 for r in resolutions):
        parser.error('Use at least two increasing mesh densities between 24 and 256 cells/µm.')
    from meep import mpb
    import meep as mp
    args.output.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'generated_at':datetime.now(timezone.utc).isoformat(),'solver':'MPB, fixed-frequency Maxwell eigenproblem',
        'versions':{'mpb':'.'.join(map(str,mpb.__version__)),'meep':mp.__version__,'python':platform.python_version(),'numpy':np.__version__},
        'cases':[]}
    for case in (REFERENCES if args.case=='all' else [args.case]):
        entry=describe(case)
        rows=[]
        for r in resolutions:
            rows.append(calculate(case,r,(3.,3.),args.output))
            print(case,r,rows[-1]['quantities'],flush=True)
        domain=calculate(case,resolutions[-1],(4.,4.),args.output)
        last=rows[-1]['quantities']
        metrics=[]
        for key,(label,unit,ref) in REFERENCES[case].items():
            computed=last[key]
            metrics.append({'key':key,'label':label,'unit':unit,'reference':ref,'computed':computed,
                'absolute_error':abs(computed-ref),'relative_error':abs(computed-ref)/abs(ref),
                'last_mesh_change':abs(computed-rows[-2]['quantities'][key]),
                'larger_domain_change':abs(computed-domain['quantities'][key])})
        entry.update(metrics=metrics,refinement=rows,domain_check=domain)
        report['cases'].append(entry)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.report:
        args.report.write_bytes((args.output/'report.json').read_bytes())

if __name__=='__main__':
    main()
