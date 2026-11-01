"""Local MPB qualification cases from public examples, separate from FDTD jobs.

Meep/MPB is imported only inside solve_case. Run it in a fresh worker process,
never mixed with a Meep Simulation or the web application's process.
"""
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np


CASES = {
    'comsol-slab': {
        'title': 'COMSOL dielectric slab: fundamental TE mode',
        'source_url': 'https://doc.comsol.com/6.3/doc/com.comsol.help.models.woptics.dielectric_slab_waveguide/dielectric_slab_waveguide.html',
        'reference_kind': 'analytic_reference', 'reference_neff': None,
        'core_index': 1.5, 'cladding_index': 1., 'thickness_um': 1.,
        'width_um': None, 'domain_y_um': 7., 'domain_z_um': 0., 'guess_neff': 1.4,
        'notes': ['Reference is the analytic TE0 dispersion equation described by COMSOL, not a numerical COMSOL export.',
                  'Invariant slab mode only; no finite-height 3D driven-field or transmission comparison.'],
    },
    'tidy3d-strip': {
        'title': 'Tidy3D straight-strip mode benchmark',
        'source_url': 'https://www.flexcompute.com/tidy3d/examples/notebooks/ModeSolverFemwellBenchmark/',
        'reference_kind': 'published_solver_value', 'reference_neff': 2.448339,
        'core_index': 3.48, 'cladding_index': 1.44, 'thickness_um': .22,
        'width_um': .5, 'domain_y_um': 4.3, 'domain_z_um': 3.02, 'guess_neff': 2.45,
        'notes': ['Reference is the published Tidy3D simple_strip min_steps_per_wvl=100 scalar, printed to six decimals.',
                  'Reference uses an automatic grid and polarized subpixel averaging; MPB uses a Fourier basis and material averaging.',
                  'Reference propagation y and transverse x/z are mapped to workbench propagation x and transverse y/z.',
                  'No downloaded commercial field array or local commercial runtime; this is an effective-index comparison only.'],
    },
    'lumerical-slab': {
        'title': 'Lumerical MODE silicon slab: 220 nm TE mode',
        'source_url': 'https://optics.ansys.com/hc/en-us/articles/1500007378182-Inverse-design-of-an-efficient-narrow-band-grating-coupler',
        'reference_kind': 'published_solver_value', 'reference_neff': 2.84825,
        'core_index': 3.48, 'cladding_index': 1.44, 'thickness_um': .22,
        'width_um': None, 'domain_y_um': 4., 'domain_z_um': 0., 'guess_neff': 2.85,
        'notes': ['Reference is the published intermediate 1D MODE table at 220 nm, printed to five decimals; its mesh and solver version are unspecified.',
                  'The official associated grating_coupler_2D_apodized.py specifies the 1.55 µm wavelength; only its slab-mode step is reproduced.',
                  'This does not reproduce or validate the complete grating, optimization or coupling efficiency.',
                  'Published MODE value differs from the analytic slab root; both discrepancies are retained without fitting the geometry.'],
        'wavelength_source_url': 'https://optics.ansys.com/hc/en-us/article_attachments/1500010548101',
    },
}
CASES['lumerical-slab-script'] = {
    **CASES['lumerical-slab'],
    'title': 'Lumerical slab: script-index variant n = 3.47668',
    'core_index': 3.47668,
    'notes': ['The official article states core index 3.48, but its associated Python script sets eps_in=3.47668**2. This separately named variant preserves that source inconsistency.',
              'Same published 220 nm MODE table value is shown for context; the article does not establish which exact index generated that table.',
              'No material fitting: 3.47668 is copied from the official script, and the prose-index case is retained independently.',
              'Only the invariant slab TE0 mode is computed, not the full grating or its coupling efficiency.'],
}


def analytic_te0(core_index, cladding_index, thickness_um, wavelength_um=1.55):
    """Fundamental symmetric lossless slab root: u*tan(u)=sqrt(V²-u²)."""
    if not all(math.isfinite(v) for v in (core_index, cladding_index, thickness_um, wavelength_um)) or not (core_index > cladding_index > 0 and thickness_um > 0 and wavelength_um > 0):
        raise ValueError('The analytic slab requires positive dimensions and core index above cladding.')
    a = thickness_um / 2
    k0 = 2 * math.pi / wavelength_um
    v = k0 * a * math.sqrt(core_index**2 - cladding_index**2)
    lo, hi = 0., min(v, math.pi/2)
    for _ in range(100):
        u = (lo + hi) / 2
        residual = u * math.tan(u) - math.sqrt(max(0., v*v-u*u))
        if residual > 0:
            hi = u
        else:
            lo = u
    return math.sqrt(core_index**2 - (((lo+hi)/2)/(a*k0))**2)


def solve_case(case_id, resolution, output_dir, domain_scale=1.):
    """Compute and save one actual MPB eigenmode; return JSON-safe metadata."""
    if case_id not in CASES:
        raise ValueError(f'Unknown benchmark {case_id!r}.')
    if type(resolution) is not int or not 16 <= resolution <= 256:
        raise ValueError('Resolution must be an integer from 16 through 256 cells/µm.')
    if not math.isfinite(domain_scale) or not 1 <= domain_scale <= 2:
        raise ValueError('Domain scale must be between 1 and 2.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    import meep as mp
    from meep import mpb

    mp.verbosity(0)
    case = CASES[case_id]
    wavelength, frequency = 1.55, 1/1.55
    slab = case['width_um'] is None
    ly, lz = case['domain_y_um']*domain_scale, case['domain_z_um']*domain_scale
    core_size = mp.Vector3(mp.inf, case['thickness_um'] if slab else case['width_um'],
                          mp.inf if slab else case['thickness_um'])
    solver = mpb.ModeSolver(
        geometry_lattice=mp.Lattice(size=mp.Vector3(0, ly, lz)),
        geometry=[mp.Block(center=mp.Vector3(), size=core_size, material=mp.Medium(index=case['core_index']))],
        default_material=mp.Medium(index=case['cladding_index']),
        resolution=resolution, num_bands=1, tolerance=1e-11,
        mesh_size=3, deterministic=True,
    )
    saved = {}
    start = time.perf_counter()

    def capture(ms, band):
        mpb.fix_efield_phase(ms, band)
        electric = np.asarray(ms.get_efield(band, bloch_phase=False)).copy()
        ny, nz = electric.shape[:2]
        electric_yz = electric.reshape(ny, nz, 3)
        coordinate_error = 0.
        for iy, iz in ((0, 0), (ny//2, nz//2), (ny//3, nz//3)):
            point = mp.Vector3(0, iy*ly/ny-ly/2, 0 if slab else iz*lz/nz-lz/2)
            value = ms.get_field_point(point)
            coordinate_error = max(coordinate_error, float(np.max(abs(np.array([value.x, value.y, value.z])-electric_yz[iy, iz]))))
        if coordinate_error > 1e-10:
            raise RuntimeError('MPB point sampling does not agree with the exported coordinate convention.')
        magnetic = np.asarray(ms.get_hfield(band, bloch_phase=False)).copy()
        epsilon = np.asarray(ms.get_epsilon()).copy()
        shape = epsilon.shape
        ny, nz = (shape[0], 1) if slab else shape
        saved.update(fields=np.transpose(electric.reshape(ny, nz, 3), (2, 1, 0)),
                     magnetic_fields=np.transpose(magnetic.reshape(ny, nz, 3), (2, 1, 0)),
                     epsilon=epsilon.reshape(ny, nz).T,
                     y_um=np.arange(ny)*ly/ny-ly/2,
                     z_um=np.array([0.]) if slab else np.arange(nz)*lz/nz-lz/2)
        saved['computed_frequency'] = float(ms.freqs[0])
        saved['coordinate_check_max_abs_error'] = coordinate_error

    # ODD_Z selects Ez-polarized TE slab modes when z is invariant. The strip
    # uses its lowest-frequency full-vector mode, checked for dominant Ey below.
    parity = mp.ODD_Z if slab else mp.NO_PARITY
    k = solver.find_k(parity, frequency, 1, 1, mp.Vector3(1, 0, 0), 1e-9,
                      case['guess_neff']*frequency,
                      case['cladding_index']*frequency*1.000001,
                      case['core_index']*frequency, capture)[0]
    elapsed = time.perf_counter() - start
    neff = float(k / frequency)
    if not case['cladding_index'] < neff < case['core_index']:
        raise RuntimeError('MPB did not find a guided mode between the material light lines.')
    if not all(np.isfinite(saved[key]).all() for key in ('fields', 'magnetic_fields', 'epsilon')):
        raise RuntimeError('MPB produced non-finite mode samples.')
    electric = saved['fields']
    component_sums = np.sum(np.abs(electric)**2, axis=(1, 2))
    if not component_sums.sum() > 0:
        raise RuntimeError('MPB produced a zero electric mode.')
    fractions = component_sums / component_sums.sum()
    expected_component = 2 if slab else 1
    if fractions[expected_component] < .5:
        raise RuntimeError('The computed mode does not have the expected dominant electric component.')
    frequency_residual = abs(saved.pop('computed_frequency')-frequency)/frequency
    coordinate_error = saved.pop('coordinate_check_max_abs_error')
    if frequency_residual > 1e-7:
        raise RuntimeError(f'Fixed-frequency root residual is too large: {frequency_residual}.')
    analytic = analytic_te0(case['core_index'], case['cladding_index'], case['thickness_um']) if slab else None
    reference = case['reference_neff'] if case['reference_neff'] is not None else analytic
    # One scalar is deliberately compared without fitting geometry, phase or amplitude.
    report = {
        'id': case_id, 'title': case['title'], 'source_url': case['source_url'],
        'reference_kind': case['reference_kind'], 'reference_neff': reference,
        'computed_neff': neff, 'absolute_error': abs(neff-reference),
        'relative_error': abs(neff-reference)/abs(reference),
        'resolution_cells_per_um': resolution, 'domain_scale': domain_scale,
        'grid_shape': list(saved['epsilon'].shape), 'grid_order': '[z,y]; invariant z has one sample',
        'elapsed_s': elapsed, 'status': 'computed-not-certified',
        'versions': {'meep': mp.__version__, 'mpb': '.'.join(map(str, mpb.__version__)),
                     'numpy': np.__version__, 'python': platform.python_version()},
        'physics': {'wavelength_um': wavelength, 'core_index': case['core_index'],
                    'cladding_index': case['cladding_index'], 'thickness_um': case['thickness_um'],
                    'width_um': case['width_um'], 'domain_y_um': ly, 'domain_z_um': lz,
                    'propagation_axis': 'x', 'model': 'invariant dielectric slab' if slab else 'invariant straight rectangular strip',
                    'materials': 'real isotropic nondispersive lossless',
                    'boundaries': 'periodic transverse supercell; no PML; modal tails checked by enlarged domain'},
        'mode_info': {'band': 1, 'dominant_component': ['Ex', 'Ey', 'Ez'][expected_component],
                      'component_l2_fractions': dict(zip(('Ex', 'Ey', 'Ez'), map(float, fractions))),
                      'component_fraction_definition': 'sum(abs(E_component)**2)/sum(abs(E)**2); not modal power or energy fractions',
                      'fixed_frequency_relative_residual': frequency_residual,
                      'coordinate_point_check_max_abs_error': coordinate_error,
                      'eigensolver_tolerance': 1e-11, 'find_k_tolerance': 1e-9,
                      'field_normalization': 'MPB eigenmode energy normalization; no incident source, power calibration or driven final time',
                      'field_array_order': '[component,z,y], complex Bloch envelopes; fields=E and magnetic_fields=H',
                      'sampling': 'Uniform MPB periodic grid from -L/2 to L/2 exclusive; optimized FFT grid sizes recorded explicitly'},
        'analytic_slab_neff': analytic,
        'analytic_relative_error': None if analytic is None else abs(neff-analytic)/analytic,
        'notes': list(case['notes']),
    }
    if 'wavelength_source_url' in case:
        report['wavelength_source_url'] = case['wavelength_source_url']
    np.savez_compressed(output/'mode.npz', **saved)
    report['array_sha256'] = hashlib.sha256((output/'mode.npz').read_bytes()).hexdigest()
    (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
