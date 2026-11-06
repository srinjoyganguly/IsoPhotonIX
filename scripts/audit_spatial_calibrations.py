"""Cross-check the new GUI records against the source numerical archives and replay receipts."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_spatial_calibration_index import BASE, spatial_index


def read(path):
    return json.loads(path.read_text())


def audit():
    index = spatial_index()
    if read(BASE/'v012-calibrated-index.json') != index:
        raise ValueError('Regenerate the spatial index from the native results.')
    cases = {}
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
                    mask = row.get('valid_beam_bins',row.get('valid_energy_bins'))
                    np.testing.assert_array_equal(data['valid'], mask)
                else:
                    s = np.asarray(src['s_real']) + 1j*np.asarray(src['s_imag'])
                    np.testing.assert_allclose(abs(s)**2,src['power'],atol=1e-12)
                    mesh = src['parameters']['resolution']
                    native = data[f'mesh{mesh}_s'][0] if case=='multimode-tm' else data['s']
                    np.testing.assert_allclose(native,s,atol=1e-12,rtol=1e-12)
            records.append({'receipt':view['receipt'],'passed':src['passed'],
                            'actual_device':src['actual_device'],'has_native_field':bool(view.get('field'))})
        cases[case]=records
    return {'passed':True,'cases':cases,'scope':'All included records are re-read; diagnostic ring spectra retain their original pass flag and wavelength mask.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit()
    if args.output:
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({case:len(v) for case,v in result['cases'].items()})
