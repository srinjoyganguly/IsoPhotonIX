"""Make a small verified display record plus a complete source-bound export of one dense GPU run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from backend.calibrations import Calibration, create_bundle, result_zip
from scripts.verify_finite_grating import verify

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'v0.12.0a2'
DOWNLOAD = 'https://github.com/srinjoyganguly/photonics-workbench/releases/download/'+RELEASE+'/'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_result_export(target, run, report, frozen):
    model=report['settings'].get('grating_material_model','volume-average')
    config=Calibration(revision=2,case='grating-3d',solver='fdtdx',device='gpu',resolution=12,
                       geometry_3d='lumerical',grating_spectrum='dense',
                       grating_material_model=model if model!='volume-average' else None)
    with tempfile.TemporaryDirectory() as temp:
        directory=Path(temp)
        bundle=create_bundle(config,directory/'replay')
        manifest=json.loads((bundle/'manifest.json').read_text())
        for name in manifest['files']:
            if not name.startswith('source/'):
                continue
            source=frozen/name.removeprefix('source/')
            if not source.is_file():
                raise ValueError(f'Frozen release input absent: {name}')
            shutil.copyfile(source,bundle/name)
            manifest['files'][name]=sha(source)
        (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        by_filename={Path(name).name:digest for name,digest in manifest['files'].items() if name.startswith('source/')}
        for stage in report['stages'].values():
            for name,digest in stage['identity']['sources'].items():
                if by_filename.get(name)!=digest:
                    raise ValueError(f'Exported worker differs from recorded native acquisition: {name}')
        if '--grating-spectrum' not in manifest['arguments'] or 'dense' not in manifest['arguments']:
            raise ValueError('The captured configuration does not request dense detectors.')
        (directory/'calculation').mkdir()
        for name in ('arrays.npz','report.json'):
            shutil.copyfile(run/name,directory/'calculation'/name)
        if not target.exists():
            pending=target.with_suffix('.pending')
            result_zip(directory,pending)
            pending.replace(target)
    with zipfile.ZipFile(target) as archived:
        if archived.testzip() is not None:
            raise ValueError('Invalid dense result ZIP.')
        if json.loads(archived.read('manifest.json'))!=manifest:
            raise ValueError('Dense export already exists with a different captured program.')
        required = json.loads(archived.read('reference-sha256.json'))
        for name,digest in required.items():
            with archived.open('reference/'+name) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=digest or sha(run/name)!=digest:
                    raise ValueError(f'Measured output differs in exported {name}.')
        for name,digest in manifest['files'].items():
            if hashlib.sha256(archived.read(name)).hexdigest()!=digest:
                raise ValueError(f'Exported program differs at {name}.')


def publish(run,frozen,stage,evidence):
    report=verify(run)
    m=report['settings']; model=m.get('grating_material_model','volume-average')
    if not report['passed'] or m.get('grating_spectrum')!='dense' or sum(report['result']['valid_beam_bins'])!=21:
        raise ValueError('Only a fully qualified native dense spectrum may be published.')
    stem=f'v013-grating3d-lumerical-dense-{model}-r12'
    stage.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    asset=f'grating-lumerical-dense-{model}-3d-gpu-r12-{RELEASE}'
    raw=stage/(asset+'.npz')
    if not raw.exists():
        shutil.copyfile(run/'arrays.npz',raw)
    if sha(raw)!=report['arrays_sha256']:
        raise ValueError('The full unmodified numerical archive changed.')
    export=stage/(asset+'.zip')
    write_result_export(export,run,report,frozen)

    # The full upper-plane vector phasors dominate storage. Keep the native
    # archive intact for radiation/flux reconstruction; the browser needs a slice.
    companion=evidence/(stem+'-arrays.npz')
    with np.load(raw,allow_pickle=False) as full:
        content={name:full[name] for name in full.files if name not in ('ref_upper','dev_upper')}
        wave=full['ref_wavelength_um']
        center=wave.tolist().index(1.55)
        for key in ('ref_upper','dev_upper'):
            content[key+'_1550nm']=full[key][center]
        np.savez_compressed(companion,**content)
        with np.load(companion,allow_pickle=False) as check:
            for key in check.files:
                original=full[key.removesuffix('_1550nm')][center] if key.endswith('_1550nm') else full[key]
                if check[key].tobytes()!=original.tobytes():
                    raise ValueError(f'Display companion altered native samples in {key}.')
    report['native_arrays_sha256']=sha(raw)
    report['native_report_sha256']=sha(run/'report.json')
    report['arrays_file']=companion.name;report['arrays_sha256']=sha(companion)
    report['arrays_scope']='Display companion contains native samples; full upper-plane phasors are included in the original NPZ.'
    report['native_artifacts']={key:{'asset':item.name,'url':DOWNLOAD+item.name,'sha256':sha(item),'bytes':item.stat().st_size}
                                for key,item in [('raw_arrays',raw),('python_notebook_and_results',export)]}
    output=evidence/(stem+'.json')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(output,raw.stat().st_size,export.stat().st_size)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True,help='Frozen source and evidence inputs used by the native worker')
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,default=ROOT/'docs/verification')
    args=parser.parse_args()
    publish(args.run,args.frozen,args.stage,args.evidence)
