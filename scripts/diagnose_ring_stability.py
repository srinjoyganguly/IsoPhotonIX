"""Measure late probe growth and broadband qualification for a completed ring acquisition."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scripts.calibrate_loaded_ring3d import field_stability
from scripts.check_ring_wall_windows import load, check_window
from scripts.compare_ring_interface_models import digest


def rejection_for(path):
    record=json.loads(path.read_text())
    current=record.get('stability',{})
    if current.get('usable_spectrum') is False:
        return dict(current,source_receipt=path.name,source_report_sha256=digest(path),source_arrays_sha256=record['arrays_sha256'])
    receipt=path.parent/'v014-ring-wall-stability.json'
    if not receipt.is_file():
        return None
    check=json.loads(receipt.read_text())
    if check['source_receipt']!=path.name or check['usable_spectrum']:
        return None
    if check['source_report_sha256']!=digest(path) or check['source_arrays_sha256']!=json.loads(path.read_text())['arrays_sha256']:
        raise ValueError('Global stability decision and measured archive do not agree.')
    return check


def diagnose(path):
    r,ref,dev,wl,s,original=load(path)
    window,mask,_=check_window(ref,dev,wl,r['duration_um_c'])
    if 'stability' in r:
        np.testing.assert_array_equal(original,mask & r['stability']['usable_spectrum'])
    else:
        np.testing.assert_array_equal(mask,original)
    screen=field_stability(dev['ring_Hz'],(abs(s)**2).sum(1),r['duration_um_c'])
    failing=Counter(g for bin in window['failed_bins'] for g in bin['gates_failed'])
    return dict(source_receipt=path.name,source_report_sha256=digest(path),source_arrays_sha256=r['arrays_sha256'],
                usable_spectrum=screen['usable_spectrum'],reason=screen['reason'],
                original_mask_qualified=int(mask.sum()),original_qualified_wavelengths_um=wl[mask].tolist(),
                native_gate_failure_counts=dict(failing),first_to_fourth_quarter_probe_peak_abs_Hz=screen['first_to_fourth_quarter_probe_peak_abs_Hz'],
                fourth_to_first_half_peak_per_probe=screen['fourth_to_first_half_peak_per_probe'],
                max_collected_power=float((abs(s)**2).sum(1).max()),
                source_shutoff_um_c=2*4/.14,window_um_c=r['duration_um_c'],
                limits='A global rejection overrides even an isolated nominally valid bin. Absence of this severe growth signal does not establish vendor agreement or convergence.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.write_text(json.dumps(diagnose(args.receipt),indent=2,allow_nan=False)+'\n')
