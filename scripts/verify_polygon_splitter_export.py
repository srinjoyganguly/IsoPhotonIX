"""Compare the GUI matrix to an independent native column and freshly replay its notebook."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from scripts.verify_calibration_replays import replay
from scripts.verify_vendor_grating_notebooks import check_executed_notebook


def verify(archive, column_path, dest):
    with zipfile.ZipFile(archive) as z:
        config=json.loads(z.read('config.json'))
        if (config['case'],config['device'],config['solver'],config['material_model'])!=('splitter','gpu','fdtdx','polygon-area'):
            raise ValueError('Select the measured polygon-area NVIDIA splitter export.')
        before=json.loads(z.read('reference/report.json'))
        digest=json.loads(z.read('reference-sha256.json'))['gpu-performance.csv']
        if hashlib.sha256(z.read('reference/gpu-performance.csv')).hexdigest()!=digest:
            raise ValueError('Timing CSV changed after the measured run.')
    column=json.loads(column_path.read_text())
    if not column['passed'] or not before['sampled_setup_checks_passed']:
        raise ValueError('Require the qualified matrix and independent column.')
    i=before['port_order'].index(column['launch'])
    a=np.asarray(before['s'])[:,:,i,:]
    b=np.asarray(column['s'])
    np.testing.assert_allclose(a,b,rtol=2e-5,atol=1e-7)
    stages=[s['performance'] for s in before['stages'].values()]
    if len(stages)!=6 or not all(p['synchronized_step_seconds']>0 and p['cuda_event_elapsed_seconds']>0 for p in stages):
        raise ValueError('Missing native reference or device GPU timing.')
    # A new output directory and an empty stage history force independent acquisitions.
    os.environ.pop('PHOTONICS_CHECKPOINT',None)
    dest.mkdir(exist_ok=False)
    reproduction=replay('notebook',archive,dest)
    check_executed_notebook(dest/'notebook')
    after=json.loads((dest/'notebook/notebook-result/report.json').read_text())
    new=[s['performance'] for s in after['stages'].values()]
    if (not reproduction['passed'] or not after['sampled_setup_checks_passed'] or after['actual_device']!='cuda:0'
            or {p['acquisition_id'] for p in new}&{p['acquisition_id'] for p in stages}):
        raise ValueError('The independent CUDA replay did not reproduce the selected matrix.')
    return {'passed':True,'case':'splitter','actual_device':before['actual_device'],'material_model':config['material_model'],
            'resolution':config['resolution'],'launches':before['port_order'],'frequencies':before['frequency_per_um'],
            'gui_matrix_complex_entries':int(np.asarray(before['s']).size//2),'independent_column':column_path.name,
            'maximum_column_complex_difference':float(np.max(np.hypot(a[...,0]-b[...,0],a[...,1]-b[...,1]))),
            'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'exported_timing_csv_sha256':digest,
            'reproduction':reproduction,'gpu_stages_gui':{k:v['performance'] for k,v in before['stages'].items()},
            'gpu_stages_fresh_notebook':{k:v['performance'] for k,v in after['stages'].items()}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--column',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(verify(args.archive,args.column,args.work),indent=2,allow_nan=False)+'\n')
