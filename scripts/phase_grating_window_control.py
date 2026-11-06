"""Register fresh GPU grating time histories and compare with the earlier same-mesh fields."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.phase_component_comparison import analyze, compare, file_sha


def check(current_path, previous_path, previous_registry, vendor_file=None):
    current = analyze('grating',[current_path],vendor_file,require_return_windows=True)
    current_sample = current['native_samples'][0]
    windows = current_sample['registered_return_time_history']
    if windows is None:
        raise ValueError('This comparison requires a measured return time history.')
    old = json.loads(previous_path.read_text())
    archival_scan = json.loads(previous_registry.read_text())
    source = json.loads(current_path.read_text())
    if source['settings']!=old['settings'] or source['arrays_sha256']==old['arrays_sha256']:
        raise ValueError('Expected a fresh acquisition of the same physical, numerical and aperture settings.')
    samples=[s for s in archival_scan['native_samples'] if s['native_receipt']==previous_path.name]
    if len(samples)!=1 or samples[0]['arrays_sha256']!=old['arrays_sha256']:
        raise ValueError('Prior registered observation does not match the requested numerical archive.')
    if file_sha(previous_path.parent/old['arrays_file'])!=old['arrays_sha256']:
        raise ValueError('Prior numeric archive has changed.')
    prior=samples[0]
    if prior['physical_model']!=current_sample['physical_model'] or archival_scan['vendor']!=current['vendor']:
        raise ValueError('The two registered observations have different physical geometries.')
    wave=np.asarray(current_sample['wavelength_um'])
    np.testing.assert_array_equal(wave,prior['wavelength_um'])
    # Plane-location metadata is checked independently from measured basis/index
    # sensitivities, which can change slightly when a run is repeated.
    for name in ('sample_z_um','guide_x_um','target_fiber_z_um','target_beam_z_um','target_guide_x_um','fiber_lead_um','guide_lead_um'):
        if prior['planes'].get(name)!=current_sample['planes'].get(name):
            raise ValueError(f'The matching grating runs differ in {name}.')
    fields=[np.asarray(s['registered_complex_s_real'])+1j*np.asarray(s['registered_complex_s_imag'])
            for s in (prior,current_sample)]
    mask=np.asarray(prior['valid_bins']) & np.asarray(current_sample['valid_bins'])
    phase_mask=np.asarray(prior['phase_ok']) & np.asarray(current_sample['phase_ok'])
    current['same_mesh_final_registration']={
        'earlier_native_report':previous_path.name,'earlier_native_arrays_sha256':old['arrays_sha256'],
        'earlier_registered_report':previous_registry.name,'earlier_registered_report_sha256':file_sha(previous_registry),
        'common_valid_bins':mask.tolist(),'channels':compare('grating',fields[0],fields[1],mask,phase_mask,wavelengths=wave),
        'scope':'Compare independent final source-subtracted coefficients at the same registered planes. Only the fresh capture has a reciprocal-return time history. Phase-repeatability statistics require its temporal gate and the 0.02 registration amplitude floor.'}
    return current


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('current','previous','previous-registry','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--vendor-file',type=Path)
    args=p.parse_args()
    proof=check(args.current,args.previous,args.previous_registry,args.vendor_file)
    args.output.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
