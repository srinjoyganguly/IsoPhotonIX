"""Cross-check the new GUI records against the source numerical archives and replay receipts."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_spatial_calibration_index import BASE, spatial_index


def read(path):
    return json.loads(path.read_text())


def audit(require_qualified=False):
    index = spatial_index()
    if read(BASE/'v012-calibrated-index.json') != index:
        raise ValueError('Regenerate the spatial index from the native results.')
    cases = {}
    vendor_meshes = {case: {} for case in ('ring-3d', 'grating-3d')}
    for case, entry in index.items():
        records = []
        for view in entry['recorded']:
            src = read(BASE/view['receipt'])
            raw = (BASE/src['arrays_file'] if 'arrays_file' in src else BASE/src['source_arrays']['file'])
            with np.load(raw,allow_pickle=False) as data:
                if case == 'ring-3d':
                    expected = src['series'][src['vendor']]
                    s = data['s']
                    for i, key in enumerate(('reflection','through','drop','add_side')):
                        np.testing.assert_allclose(abs(s[:,i])**2,expected[key],atol=1e-11,rtol=1e-11)
                    np.testing.assert_array_equal(data['valid_bins'],expected['valid_bins'])
                elif case == 'grating-3d':
                    row = src['result']
                    key = 'beam_to_guide' if 'beam_to_guide' in row else 'guide_to_beam'
                    # Acquisition workers preserve independently normalized modal power in the NPZ.
                    if key == 'beam_to_guide':
                        np.testing.assert_allclose(abs(data['output_mode'])**2/data['incident_power'],row[key],rtol=1e-10)
                    else:
                        np.testing.assert_allclose(abs(data['gaussian_channel'])**2,row[key],rtol=1e-10)
                    mask = row.get('valid_beam_bins',row.get('valid_energy_bins'))
                    np.testing.assert_array_equal(data['valid'], mask)
                    for comparison in src['comparisons']:
                        if comparison.get('geometry_matched') and key == 'beam_to_guide':
                            published = comparison.get('published_power', comparison.get('published_s21_power'))
                            np.testing.assert_array_equal(row['wavelength_um'], comparison['wavelength_um'])
                            delta = 100 * (np.asarray(row[key]) - np.asarray(published))
                            np.testing.assert_allclose(delta, comparison['local_minus_published_percentage_points'])
                            np.testing.assert_allclose(np.abs(delta).mean(), comparison['mean_absolute_difference_percentage_points'])
                else:
                    s = np.asarray(src['s_real']) + 1j*np.asarray(src['s_imag'])
                    np.testing.assert_allclose(abs(s)**2,src['power'],atol=1e-12)
                    mesh = src['parameters']['resolution']
                    native = data[f'mesh{mesh}_s'][0] if case=='multimode-tm' else data['s']
                    np.testing.assert_allclose(native,s,atol=1e-12,rtol=1e-12)
            if case in vendor_meshes and src['passed']:
                if case == 'ring-3d':
                    vendor, resolution = src['vendor'], src['physical_model']['resolution']
                else:
                    vendor, resolution = src['settings']['geometry'], src['settings']['resolution']
                if vendor in ('lumerical', 'tidy3d'):
                    vendor_meshes[case].setdefault(vendor, set()).add(resolution)
            records.append({'receipt':view['receipt'],'passed':src['passed'],
                            'actual_device':src['actual_device'],'has_native_field':bool(view.get('field'))})
        if require_qualified and not any(r['passed'] for r in records):
            raise ValueError(f'No numerically qualified public record for {case}.')
        cases[case]=records
    if require_qualified:
        for case, vendors in vendor_meshes.items():
            for vendor in ('lumerical', 'tidy3d'):
                if len(vendors.get(vendor, ())) < 2:
                    raise ValueError(f'{case} needs two qualified meshes for the {vendor} geometry.')
    meshes = {case: {vendor: sorted(values) for vendor, values in vendors.items()}
              for case, vendors in vendor_meshes.items()}
    return {'passed':True,'cases':cases,'qualified_vendor_meshes':meshes,
            'scope':'All included records are re-read. Diagnostic ring spectra retain their original pass flag and wavelength mask.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-qualified',action='store_true',help='require a passing native result in each new preset')
    args=parser.parse_args()
    result=audit(args.require_qualified)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({case:len(v) for case,v in result['cases'].items()})
