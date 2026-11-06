"""Verify a completed native 3D grating run and optionally publish its numerical evidence."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.calibrate_finite_grating import monitoring_wavelengths, file_hash, normal_flux, digest, sampled_gaussian_fields, return_window_status
from scripts.grating_angular_spectrum import air_angular_flux, incident_beam_check
from scripts.lumerical_grating3d import reflected_fiber_amplitude
from scripts.grating_mirror import PARITY
from backend.adapters.fdtdx_ports3d_adapter import modal_amplitudes


def verify_mirror_archive(data, stage, prefix):
    full = [len(data[prefix + '_grid_' + axis + '_um']) for axis in 'xyz']
    native = [full[0], full[1]//2, full[2]]
    if full[1] % 2 or full != stage['full_shape'] or native != stage['shape']:
        raise ValueError('Grating represented and allocated mirror grids differ.')
    if stage['performance']['cells'] != int(np.prod(native)):
        raise ValueError('Grating timing counts a different native grid.')
    planes=[(data[prefix+'_upper'], stage['pml_cells'])]
    if prefix+'_output' in data:
        port=data[prefix+'_output']
        planes.append((port,(full[1]-port.shape[3])//2))
    for plane, first in planes:
        center=full[1]//2-first
        pairs=min(center,plane.shape[3]-center-1)
        left=np.take(plane,center-np.arange(1,pairs+1),axis=3)
        right=np.take(plane,center+np.arange(1,pairs+1),axis=3)
        signs=PARITY.reshape((1,6,*([1]*(right.ndim-2))))
        np.testing.assert_array_equal(left,right*signs)


def verify(run):
    report=json.loads((run/'report.json').read_text())
    archive=run/'arrays.npz'
    if file_hash(archive)!=report['arrays_sha256']:
        raise ValueError('Grating numerical archive digest mismatch.')
    m=report['settings']; dx=1/m['resolution']
    wavelengths=monitoring_wavelengths(m)
    d=report['stages']['device']; r=report['stages']['reference']
    if not d['actual_device'].startswith('cuda:') or d['actual_device']!=r['actual_device']:
        raise ValueError('Reference and device must identify the same CUDA device.')
    with np.load(archive,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['ref_wavelength_um'],wavelengths)
        np.testing.assert_array_equal(data['dev_wavelength_um'],wavelengths)
        incident_direction=None
        # Verify matched sources, solver archive reproducibility and material maps.
        stages=[(r,'ref'),(d,'dev')]
        if 'beam_reference' in report['stages']:
            stages.append((report['stages']['beam_reference'],'beam'))
        for stage,prefix in stages:
            if m.get('grating_domain')=='mirror-y':
                verify_mirror_archive(data,stage,prefix)
            profile=np.stack([data[prefix+'_source_E'],data[prefix+'_source_H']])
            if stage.get('source_sampling')=='host-sampled-air-gaussian':
                xs=data[prefix+'_grid_x_um']; ys=data[prefix+'_grid_y_um']; pml=stage['pml_cells']
                lo=max(pml,int(np.argmin(abs(xs-(m['beam_x']-2*m['waist'])))))
                hi=min(len(xs)-pml,int(np.argmin(abs(xs-(m['beam_x']+2*m['waist']))))+1)
                expected=sampled_gaussian_fields(xs[lo:hi],ys[pml:-pml],m['beam_x'],m['waist'],m['angle_deg'])
                np.testing.assert_array_equal(profile,np.stack(expected))
            if digest(profile)!=stage['source_eh_sha256']:
                raise ValueError('Source fields changed.')
            if 'source_time_offsets_sha256' in stage:
                offsets=np.stack([data[prefix+'_source_time_offset_E'],data[prefix+'_source_time_offset_H']])
                if digest(offsets)!=stage['source_time_offsets_sha256']:
                    raise ValueError('Source phase delays changed.')
            energies=data[prefix+'_energy'];times=data[prefix+'_energy_time_fs']
            if not len(energies) or len(times)!=len(energies) or np.any(np.diff(times)<=0):
                raise ValueError('Invalid time trace.')
            final_window=energies[times>=times[-1]-60.]
            np.testing.assert_allclose(final_window.max()/energies.max(),stage['history'][-1]['energy_ratio'])
            if stage['settled']:
                for row in stage['history'][-2:]:
                    if row['energy_ratio']>=1e-4 or row['relative_spectral_change']>=.002:
                        raise ValueError('Settling metadata contradicts its measurements.')
        if r.get('source_time_offsets_sha256')!=d.get('source_time_offsets_sha256'):
            raise ValueError('Reference source phase delays differ from device.')
        if r['source_eh_sha256']!=d['source_eh_sha256']:raise ValueError('Reference source differs from device.')
        if 'dev_reciprocal_return_history' in data:
            series=data['dev_reciprocal_return_history']
            if series.shape!=(len(d['history']),len(wavelengths)):
                raise ValueError('Return history is missing measured time windows.')
            expected={'signature':r['signature'],'arrays_sha256':r['arrays_sha256']}
            if d.get('return_reference_arrays_sha256')!=r['arrays_sha256'] or d['identity'].get('incident_reference')!=expected:
                raise ValueError('Weak-return history has a different incident reference.')
            incident=-normal_flux(data['ref_upper'],dx)
            final_return=reflected_fiber_amplitude(data['dev_upper']-data['ref_upper'],data['ref_upper'],incident,dx)
            np.testing.assert_allclose(series[-1],final_return,rtol=1e-7,atol=1e-10)
            ref_changes=np.max([h['relative_incident_plane_l2_change'] for h in r['history'][-2:]],axis=0)
            if return_window_status(series,ref_changes)!=report['radiation']['reciprocal_return_settling']:
                raise ValueError('Weak-return qualification contradicts the recorded time windows.')
        if m['geometry']=='compact-radiation':
            from scripts.grating_emission import observables
            archives=[{key[len(prefix):]:data[key] for key in data.files if key.startswith(prefix)}
                      for prefix in ('ref_','dev_','beam_')]
            derived=observables(*archives,m['resolution'])
            for key,value in derived.items(): np.testing.assert_allclose(data[key],value,rtol=1e-7,atol=1e-12)
            efficiency=abs(derived['gaussian_channel'])**2
            np.testing.assert_allclose(efficiency,report['result']['guide_to_beam'])
            np.testing.assert_allclose(derived['upgoing_fraction'],report['result']['guide_radiation_up'])
            np.testing.assert_array_equal(data['valid'],report['result']['valid_energy_bins'])
            if report['passed']!=bool(np.all(data['valid'])):raise ValueError('Incorrect angular validity mask.')
        else:
            incident=-normal_flux(data['ref_upper'],dx)
            modes=modal_amplitudes(data['dev_output'],data['dev_mode_e'],data['dev_mode_h'],dx**2)[0]
            np.testing.assert_allclose(modes[:,1],data['output_mode'],rtol=2e-7)
            np.testing.assert_allclose(incident,data['incident_power'],rtol=1e-7)
            efficiency=abs(modes[:,1])**2/incident
            np.testing.assert_allclose(efficiency,report['result']['beam_to_guide'],rtol=1e-7)
            valid=(incident>=.01*max(incident))&(efficiency>=0.)&(efficiency<=1.02)&d['settled']&r['settled']
            np.testing.assert_array_equal(valid,data['valid'])
            np.testing.assert_array_equal(valid,report['result']['valid_beam_bins'])
            if report['passed']!=bool(valid.all()):raise ValueError('Incorrect qualification flag.')
            scatter=data['dev_upper']-data['ref_upper']
            if m['geometry']=='lumerical':
                refl=reflected_fiber_amplitude(scatter,data['ref_upper'],incident,dx)
                np.testing.assert_allclose(refl,data['reflected_fiber_mode'],atol=1e-12)
                np.testing.assert_allclose(abs(refl)**2,report['radiation']['fiber_return_power'])
            else:
                incident_direction=incident_beam_check(data['ref_upper'],dx,wavelengths,incident)
                ang=air_angular_flux(scatter,dx,wavelengths)
                frac=ang['polar_bin_power']/incident[:,None]
                np.testing.assert_allclose(frac,data['reflected_angular_fraction'],atol=1e-12)
                np.testing.assert_allclose(frac,report['radiation']['fraction_of_incident'])
        preview=report['field_preview']; plane=data[preview['array']][tuple(preview['index'])]
        if plane.shape!=(len(preview['horizontal_um']),len(preview['vertical_um'])) or not np.any(plane):
            raise ValueError('Missing measured x–z field preview.')
        for key in data.files:
            if not np.isfinite(data[key]).all():
                raise ValueError(f'Nonfinite values in {key}.')
    report['verification']={'source_eh_checked':True,'incident_and_mode_power_reconstructed':True,
        'angular_or_fiber_return_reconstructed':True,'energy_time_and_last_two_windows_checked':True,
        'field_preview_shape':list(plane.shape),'numerical_archive_entries':len(data.files),
        'captured_program_sha256':d.get('native_worker_sha256',d['identity']['sources']['calibrate_finite_grating.py'])}
    if incident_direction is not None:
        report['verification']['incident_beam_direction']=incident_direction
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    report=verify(args.run)
    if args.receipt:
        companion=args.receipt.with_name(args.receipt.stem+'-arrays.npz')
        with tempfile.NamedTemporaryFile(dir=companion.parent,delete=False) as f:
            temp=Path(f.name)
            with (args.run/'arrays.npz').open('rb') as source:shutil.copyfileobj(source,f)
        temp.chmod(0o644)
        temp.replace(companion)
        report['arrays_file']=companion.name
        with tempfile.NamedTemporaryFile(dir=companion.parent,mode='w',delete=False) as f:
            temp=Path(f.name);f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temp.chmod(0o644)
        temp.replace(args.receipt)
    print(report['settings']['geometry'],report['result'].get('center_beam_to_guide',report['result'].get('center_guide_to_beam')),report['passed'])
