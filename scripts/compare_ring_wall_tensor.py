"""Quantify the change from exact-area scalar to rotated sidewall material at fixed physical inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
from scripts.calibrate_loaded_ring3d import directional_spectrum, PORTS
from scripts.compare_ring_interface_models import BASE, check_record, digest
from scripts.compare_loaded_ring_controls import compare as compare_planes


def pair(directory, vendor, mesh):
    duration = 3200 if vendor=='lumerical' else 4800
    stem = f'ring3d-{vendor}-r{mesh}-t{duration}'
    scalar_path = directory / f'v013-{stem}-exact-area.json'
    wall_path = directory / f'v014-{stem}-wall-tensor.json'
    scalar = check_record(scalar_path, vendor, mesh, duration, 'exact-area')
    wall = check_record(wall_path, vendor, mesh, duration, 'wall-tensor')
    for a, b in [(scalar['versions'], wall['versions']), (scalar['physical_model'] | {'ring_interface':'wall-tensor'}, wall['physical_model'])]:
        if a != b:
            raise ValueError('Matched interface control changed its geometry or solver versions.')
    comparison = compare_planes(scalar_path, wall_path)
    source = wall['vendor_comparison']['source']
    stats = {}
    for port, each in wall['vendor_comparison']['channels'].items():
        old = scalar['vendor_comparison']['channels'][port]['paired_spectrum']
        new = each['paired_spectrum']
        w,i,j = np.intersect1d(old['wavelength_um'],new['wavelength_um'],return_indices=True)
        previous = np.array(old['local_power'])[i]
        changed = np.array(new['local_power'])[j]
        published = np.array(new['published_power'])[j]
        np.testing.assert_array_equal(np.array(old['published_power'])[i],published)
        indices = np.flatnonzero(np.isclose(w,1.55,rtol=0,atol=1e-12))
        center = int(indices[0]) if len(indices)==1 else None
        stats[port] = dict(paired_samples=len(w),
            old_exact_area_mae=float(abs(published-previous).mean()),
            wall_tensor_mae=float(abs(published-changed).mean()),
            at_1550=None if center is None else dict(scalar=float(previous[center]),tensor=float(changed[center]),
                                                     published=float(published[center])))
    with np.load(directory/wall['arrays_file'],allow_pickle=False) as data:
        mask = data['valid_bins'] & (data['wavelength_um']>=1.5) & (data['wavelength_um']<=1.6)
        if not mask.any():
            raise ValueError('No qualified wavelengths in the comparison band.')
        s = data['s'][mask]
        residual = abs(s - data['s_early'][mask])
        reference={k.removeprefix('reference_'):data[k] for k in data.files if k.startswith('reference_')}
        straight=directional_spectrum(reference,1.)
        background=straight[:,0,1]/straight[:,0,0]
        center=int(np.argmin(abs(data['wavelength_um']-1.55)))
        ret=data['s'][center,0]
        delta=data['s'][center,0]-data['s_early'][center,0]
        weak_center=dict(wavelength_um=float(data['wavelength_um'][center]),qualified=bool(data['valid_bins'][center]),
                         straight_background_abs_s=float(abs(background[center])),
                         device_before_subtraction_abs_s=float(abs(ret+background[center])),
                         background_subtracted_abs_s=float(abs(ret)),late_window_abs_s_change=float(abs(delta)),
                         change_relative_to_final_amplitude=float(abs(delta)/abs(ret)) if abs(ret)>0 else None)
        weak={}
        for index, port in enumerate(PORTS):
            power=np.abs(s[:,index])**2
            weak[port]=dict(power_min=float(power.min()),power_max=float(power.max()),
                p80_window_delta_abs_s_max=float(residual[:,index].max()),
                defined_phase_bins_above_1pct_power=int((power>=.01).sum()),qualified_bins=len(power))
    nh=wall['native_stage_provenance']
    return dict(vendor=vendor,mesh_cells_per_um=mesh,edge_gap_nm=wall['physical_model']['edge_gap_um']*1000,
                cells_across_gap=wall['physical_model']['edge_gap_um']*mesh,
                scalar_report=scalar_path.name,scalar_arrays_sha256=scalar['arrays_sha256'],
                tensor_report=wall_path.name,tensor_report_sha256=digest(wall_path),
                tensor_arrays_sha256=wall['arrays_sha256'],source=source,versions=wall['versions'],
                launch_mode_neff_at_1550=[r['stages']['reference']['source_mode_neff'] for r in (scalar,wall)],
                central_drop_pair_um=[r['measured_spacing_pair_um'] for r in (scalar,wall)],
                fsr_nm=[r['measured_drop_fsr_nm'] for r in (scalar,wall)],published_fsr_nm=wall['published_drop_fsr_nm'],
                all_wavelengths_passed_native_controls=wall['passed'],power_agreement=stats,
                peak_comparisons=wall['vendor_comparison']['published_drop_resonances'],channels_at_workbench_planes=weak,
                reflection_background_check_at_1550=weak_center,
                matched_scalar_tensor_planes=comparison,
                stage_calculation_signatures={key:v['checkpoint_signature'] for key,v in nh.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=BASE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',action='append',choices=('lumerical16','lumerical20','tidy3d16'))
    args=parser.parse_args()
    cases=args.case or ['lumerical16','lumerical20','tidy3d16']
    records=[pair(args.directory,key[:-2],int(key[-2:])) for key in cases]
    output={'scope':'Same mesh, finite geometry, physical gap, wavelength axis, fixed material indices, matched reference and port apertures. Analytic fractions in both; x/y wall normal constitutive response differs. No phase offset or resonance shift fitted.', 'pairs':records}
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
