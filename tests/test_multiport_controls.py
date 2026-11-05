"""Synthetic fixtures verify comparison arithmetic, not electromagnetic performance."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from backend.calibrated_geometry import multiport_model
from scripts.compare_multiport_controls import compare


def write_case(directory, inputs, resolution, matrix):
    directory.mkdir()
    model=multiport_model('splitter')
    ports=[p['name'] for p in model['ports']]
    np.savez(directory/'arrays.npz',s=matrix,frequency_per_um=model['frequency_per_um'])
    digest=hashlib.sha256((directory/'arrays.npz').read_bytes()).hexdigest()
    encoded=np.stack([matrix.real,matrix.imag],axis=-1).tolist()
    report={'solver':'Meep','version':'fixture','actual_device':'CPU','case':'splitter','ports':ports,
            'input_ports':inputs,'s':encoded,'valid_bins':[True]*3,'arrays_sha256':digest}
    (directory/'report.json').write_text(json.dumps(report))
    (directory/'config.json').write_text(json.dumps({'model':model,'resolution':resolution,'max_after_source_time':80.}))


class MultiportControls(unittest.TestCase):
    def test_named_input_subset_and_small_or_wrapped_phase(self):
        with TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'base',Path(tmp)/'fine'
            inputs=['left_center','right_lower','right_upper']
            baseline=np.zeros((3,3,3),dtype=complex)
            baseline[:,0,2]=np.exp(1j*(np.pi-.02))
            control=baseline[:,:,[2]].copy()
            control[:,0,0]=np.exp(1j*(-np.pi+.01))
            write_case(a,inputs,16,baseline)
            write_case(b,[inputs[2]],20,control)
            result=compare(a,b)
            self.assertEqual(result['control'],'mesh')
            self.assertEqual(result['entries'][0]['incident'],'right_upper')
            self.assertAlmostEqual(result['entries'][0]['max_absolute_power_change'],0.)
            self.assertAlmostEqual(result['entries'][0]['max_principal_phase_change_rad'],.03)
            self.assertIsNone(result['entries'][1]['max_principal_phase_change_rad'])
            newer=json.loads((b/'report.json').read_text());newer['phase_convention_version']=2
            (b/'report.json').write_text(json.dumps(newer))
            with self.assertRaisesRegex(ValueError,'Phase conventions'): compare(a,b)
            newer.pop('phase_convention_version')
            (b/'report.json').write_text(json.dumps(newer))
            config=json.loads((b/'config.json').read_text());config['model']['width'] *= 1.1
            (b/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'Physical'):
                compare(a,b)

    def test_changed_archive_is_refused(self):
        with TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'base',Path(tmp)/'fine'
            s=np.zeros((3,3,1),complex)
            write_case(a,['left_center'],16,s);write_case(b,['left_center'],20,s)
            with (b/'arrays.npz').open('ab') as f: f.write(b'Unexpected change')
            with self.assertRaisesRegex(ValueError,'digest'): compare(a,b)
