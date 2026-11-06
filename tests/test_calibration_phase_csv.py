import csv
import io
import json
from pathlib import Path
import unittest
from backend.calibration_phase import registered_csv

class PhaseCSV(unittest.TestCase):
    def test_exported_mask_phase_and_port_match_registered_samples(self):
        path=Path(__file__).resolve().parents[1]/'docs/verification/v014-ring-phase-wall.json'
        study=json.loads(path.read_text())
        rows=list(csv.DictReader(io.StringIO(registered_csv(study))))
        self.assertTrue(any(r['signed_phase_minus_vendor_rad'] for r in rows))
        self.assertTrue(any(r['native_spectrum_valid']=='False' for r in rows))
        self.assertTrue(all(not r['phase_rad'] and not r['signed_phase_minus_vendor_rad']
                            for r in rows if r['phase_reportable']=='False'))
        for row in rows:
            self.assertAlmostEqual(float(row['power_fraction']),float(row['real'])**2+float(row['imag'])**2)
        s=study['native_samples'][0]
        self.assertEqual(rows[1]['port'],'through')
        self.assertEqual(float(rows[1]['real']),s['registered_complex_s_real'][0][1])
        fixed_rows=[r for r in rows if r['basis']=='single central mode' and r['native_arrays_sha256']==s['arrays_sha256']]
        self.assertEqual([r['native_spectrum_valid']=='True' for r in fixed_rows[::4]],s['single_central_basis_band']['native_valid_bins'])
