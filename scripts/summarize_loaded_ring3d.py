"""Verify native modal arrays and publish a portable loaded-ring measurement and vendor comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.calibrate_loaded_ring3d import directional_spectrum,PORTS
from scripts.compare_loaded_ring3d import compare
from scripts.measure_loaded_ring_losses import measure
from scripts.ring_stage_provenance import public_fingerprint


def validated_report(folder):
    report=json.loads((folder/'report.json').read_text())
    path=folder/'arrays.npz'
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum!=report['arrays_sha256'] or report['actual_device']!='cuda:0' or not report['measurement_complete']:
        raise ValueError('The source data or native GPU record did not verify.')
    with np.load(path,allow_pickle=False) as a:
        stages={name:{key.removeprefix(name+'_'):a[key] for key in a.files if key.startswith(name+'_')}
                for name in ('reference','device')}
        r=directional_spectrum(stages['reference'],1.)
        for fraction,stored in ((1.,'s'),(.8,'s_early')):
            dev=directional_spectrum(stages['device'],fraction)
            expected=np.stack([dev[:,0,1]-r[:,0,1],dev[:,1,0],dev[:,2,1],dev[:,3,0]],axis=1)/r[:,0,0,None]
            np.testing.assert_allclose(expected,a[stored],rtol=1e-10,atol=1e-11)
        series=report['series'][report['vendor']]
        for i,port in enumerate(PORTS):
            np.testing.assert_allclose(abs(a['s'][:,i])**2,series[port],rtol=1e-11,atol=1e-12)
        np.testing.assert_array_equal(a['valid_bins'],series['valid_bins'])
        snap=report['field_preview']
        field=a[snap['array']][tuple(snap['index'])]
        if field.shape!=(len(snap['horizontal_um']),len(snap['vertical_um'])) or not np.isfinite(field).all() or np.ptp(field)<=0:
            raise ValueError('Field preview failed native array verification.')
        np.testing.assert_allclose(snap['time_um_c'],a['device_snapshot_times_um_c'][snap['index'][0]])
    return report


def stage_provenance(checkpoint, folder, report):
    from scripts.calibration_checkpoint import MeasurementCheckpoints
    tags={}
    with np.load(folder/'arrays.npz',allow_pickle=False) as all_arrays:
        for label in ('reference','device'):
            candidates=[]
            for receipt_path in checkpoint.glob(label+'-*/receipt.json'):
                receipt=json.loads(receipt_path.read_text())
                spec=receipt['specification']
                if (spec['settings']['geometry']!=report['physical_model'] or receipt['report']!=report['stages'][label]):
                    continue
                archive=receipt_path.parent/'arrays.npz'
                assert MeasurementCheckpoints.json_hash(spec)==receipt['signature']
                assert MeasurementCheckpoints.json_hash(receipt['report'])==receipt['report_sha256']
                assert MeasurementCheckpoints.file_hash(archive)==receipt['arrays_sha256']
                with np.load(archive,allow_pickle=False) as raw:
                    for key in raw.files:
                        np.testing.assert_array_equal(raw[key],all_arrays[label+'_'+key])
                candidates.append({'checkpoint_signature':receipt['signature'],'native_arrays_sha256':receipt['arrays_sha256'],
                                   'provenance':public_fingerprint(spec['provenance'])})
            if len(candidates)!=1:
                raise ValueError(f'Expected one verified {label} acquisition for {folder}.')
            tags[label]=candidates[0]
    return tags


def publish(folder, source, output, checkpoint):
    report=validated_report(folder)
    vendor=report['vendor']
    base=f"v012-ring3d-{vendor}-r{report['physical_model']['resolution']}-t{int(report['duration_um_c'])}"
    g=report['physical_model']
    if g.get('ring_interface') in ('wall-tensor','reciprocal-wall'):
        base=base.replace('v012-', 'v014-', 1) + '-' + g['ring_interface']
    elif g.get('ring_interface','legacy')!='legacy':
        base=base.replace('v012-', 'v013-', 1) + '-' + g['ring_interface']
    if g.get('ring_material')=='palik-ade':
        base+='-palik-ade'
    if g['pml']!=round(.65*g['resolution']):
        base+=f"-pml{g['pml']}c"
    if g['half_ap']!=round(.9*g['resolution']):
        base+=f"-aperture{2*g['half_ap']}c"
    dest=output/(base+'-arrays.npz')

    report['arrays_file']=dest.name
    report['loaded_loss']=measure(folder)
    report['vendor_comparison']=compare(report,source)
    report['validation_scope']='One physical port launch and its matched unpatterned two-bus reference. Checking the complete output spectrum uses all 241 illuminated bins; partial spectra preserve each wavelength mask.'
    report['native_stage_provenance']=stage_provenance(checkpoint,folder,report)
    report['source_report_sha256']=hashlib.sha256((folder/'report.json').read_bytes()).hexdigest()
    if g.get('ring_interface')=='reciprocal-wall':
        from datetime import datetime,timezone
        report['native_report_file_written_utc']=datetime.fromtimestamp((folder/'report.json').stat().st_mtime,timezone.utc).isoformat()
    # Publish verified numerical data first; each visible file is complete after rename.
    temporary=dest.with_suffix('.npz.tmp')
    shutil.copyfile(folder/'arrays.npz',temporary)
    temporary.replace(dest)
    json_dest=output/(base+'.json')
    temporary=json_dest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(json_dest)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    record=publish(args.run,args.source,args.output,args.checkpoint)
    print(record['vendor'],record['valid_bins'],record['measured_drop_fsr_nm'],record['loaded_loss']['qualified_peaks'])
