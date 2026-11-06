"""Finite calibrated component presets and their reproducible local worker bundles."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from pydantic import ConfigDict, BaseModel, model_validator
from typing import Literal

from .runtime import ROOT, interpreter

CASES = {
    'coupler': ('Directional coupler', 'Four separated fundamental-mode ports in a finite-height silicon circuit.'),
    'splitter': ('Y splitter', 'Three guided ports, with an independent launch from the stem and each branch.'),
    'ring': ('Add/drop ring', 'Calibrated bus and drop spectra in a planar effective-index ring.'),
    'grating': ('Grating radiation', 'Guided input, upward radiation and Gaussian coupling in a planar grating.'),
    'multimode': ('Multimode slab joint', 'Two TE modes on each side of a planar offset joint, checked against independent mode matching.'),
    'multimode-3d': ('Finite-height multimode joint', 'Two quasi-TE modes on each side of a finite-height offset silicon strip.'),
}


class Calibration(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    study_type: Literal['calibrated-component'] = 'calibrated-component'
    revision: Literal[1,2] = 1
    case: Literal['coupler','splitter','ring','grating','multimode','multimode-3d']
    solver: Literal['meep','fdtdx']
    device: Literal['cpu','gpu']
    resolution: int | None = None
    offset_um: float | None = None
    material_model: Literal['legacy-x','yee-aligned'] | None = None
    ring_scope: Literal['full','lumerical','tidy3d'] | None = None
    grating_scope: Literal['full','tidy3d-compact','lumerical-article','lumerical-project','lumerical-output','tidy3d-radiation','lumerical-radiation'] | None = None

    @model_validator(mode='after')
    def available_route(self):
        if self.revision==2:
            if self.solver=='fdtdx' and self.material_model is None:
                self.material_model='yee-aligned'
            if self.case=='ring' and self.ring_scope is None:
                self.ring_scope='lumerical'
            if self.case=='grating' and self.grating_scope is None:
                self.grating_scope='tidy3d-compact'
        if self.solver=='meep' and self.device!='cpu':
            raise ValueError('Meep requires CPU execution.')
        if self.solver=='fdtdx' and self.case not in ('coupler','splitter'):
            raise ValueError('This calibrated preset supports Meep CPU.')
        if self.material_model is not None and self.solver != 'fdtdx':
            raise ValueError('Only FDTDX offers a selectable material grid.')
        if self.ring_scope is not None and self.case != 'ring':
            raise ValueError('Choose a ring preset for a loaded spectrum.')
        if self.grating_scope is not None and self.case != 'grating':
            raise ValueError('Choose a grating preset for beam comparison.')
        meshes = {'coupler':(16,20,24),'splitter':(16,20,24),'multimode':(32,48,64),'multimode-3d':(16,20,24)}
        if self.case=='grating' and self.grating_scope not in (None,'full'):
            meshes['grating']=(28,40,60)
        if self.solver=='fdtdx':
            meshes[self.case]=(12,16,20,24)
        if self.resolution is not None and self.resolution not in meshes.get(self.case,()):
            raise ValueError('Choose a supported mesh for this preset.')
        if self.offset_um is not None and (not self.case.startswith('multimode') or not 0 <= self.offset_um <= .25):
            raise ValueError('The offset is available only for multimode joints, between 0 and 0.25 µm.')
        return self


def specification(config):
    multiport=config.case in ('coupler','splitter')
    if multiport:
        name='calibrate_fdtdx_multiport' if config.solver=='fdtdx' else 'calibrate_multiport'
        args=['--case',config.case]
        helpers=['backend/__init__.py','backend/calibrated_geometry.py']
        if config.solver=='fdtdx':
            args+=['--device',config.device,'--material-model',config.material_model or 'legacy-x']
            helpers+=['backend/adapters/'+n+'.py' for n in (
                'fdtdx_adapter','fdtdx3d_adapter','fdtdx_ports_adapter','fdtdx_ports3d_adapter')]
    else:
        name='calibrate_multimode' if config.case.startswith('multimode') else 'calibrate_'+config.case
        args=['--finite'] if config.case=='multimode-3d' else []
        helpers=['scripts/calibration_checkpoint.py'] if config.case in ('ring','grating') else []
    if config.ring_scope in ('lumerical','tidy3d'):
        args+=['--loaded-case',config.ring_scope,'--profile','fine']
    beam_arguments={'tidy3d-compact':['--beam-only','tidy3d','--vendor-box'],
                    'lumerical-article':['--beam-only','lumerical'],
                    'lumerical-project':['--beam-only','lumerical','--project-source'],
                    'lumerical-output':['--beam-only','lumerical','--project-output'],
                    'tidy3d-radiation':['--guide-only','tidy3d'],
                    'lumerical-radiation':['--guide-only','lumerical']}
    args.extend(beam_arguments.get(config.grating_scope, []))
    if config.resolution is not None:
        args+=['--resolution',str(config.resolution)]
    if config.offset_um is not None:
        args+=['--offset',str(config.offset_um)]
    return name,args,helpers


def create_bundle(config, destination):
    config=Calibration.model_validate(config)
    destination=Path(destination)
    destination.mkdir(exist_ok=False)
    name,args,helpers=specification(config)
    (destination/'config.json').write_text(config.model_dump_json(indent=2)+'\n')
    shutil.copyfile(ROOT/'backend/calibration_replay.py',destination/'run.py')
    shutil.copyfile(ROOT/'backend/solver_environment.py',destination/'workbench_solver_environment.py')
    for filename in ['scripts/'+name+'.py',*helpers]:
        target=destination/'source'/filename
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/filename,target)
    for filename in ('CREDITS.md','LICENSE'):
        shutil.copyfile(ROOT/filename,destination/filename)
    requirements=['environment-meep.yml','environment-meep-linux-64.lock'] if config.solver=='meep' else [
        'requirements-fdtdx.txt', 'environment-fdtdx-cpu-linux-64.lock' if config.device=='cpu' else 'environment-fdtdx-cuda12-linux-64.lock']
    if config.solver=='fdtdx' and config.device=='gpu':
        requirements.append('requirements-fdtdx-cuda.txt')
    for file in requirements:
        shutil.copyfile(ROOT/file,destination/file)
    (destination/'README.md').write_text(
        '# '+CASES[config.case][0]+'\n\n'+CASES[config.case][1]+'\n\n'
        'Extract the entire ZIP. Use Python 3.11 with NumPy for the controller and a Jupyter kernel to open calibration.ipynb. '
        'Install the selected native worker with the workbench setup command or use MEEP_PYTHON / FDTDX_PYTHON for an existing environment. '
        'With a nondefault workbench store, set PHOTONICS_DATA_DIR to that directory. Keep the original export for replay.\n\n'
        'Run `python run.py --output fresh-result` or execute the notebook in this directory. '
        'The program launches its solver afresh. For long studies, run.py accepts --checkpoint PATH to reuse matching completed stages after an interruption. '
        'In the notebook, pass checkpoint=Path("stages") to run() when you want that behavior. '
        'A completed run export contains reference/report.json, numerical NPZ files and their hashes. '
        'A fresh replay compares numerical arrays with the saved samples. Small floating-point differences are listed in reproduction.json. '
        'Physical lengths are in micrometres unless identified otherwise in the report.\n')
    cells=[
        {'cell_type':'markdown','metadata':{},'id':'guide','source':[
            '# '+CASES[config.case][0]+'\n',
            'Run with local Python and NumPy. The controller selects the saved solver and device. ',
            'The registered worker or MEEP_PYTHON / FDTDX_PYTHON must refer to that environment. ',
            'Choose a fresh output directory for each execution. Results are in report.json and arrays.npz.']},
        {'cell_type':'code','execution_count':None,'outputs':[],'metadata':{},'id':'execute','source':[
            'from pathlib import Path\n','from run import run\n',
            "report = run(Path.cwd(), Path('notebook-result'))\n",'report']}
    ]
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    (destination/'calibration.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    files={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(destination.rglob('*')) if p.is_file()}
    manifest={'format':1,'native_checkpoints':True,'worker':'scripts.'+name,'arguments':args,'files':files}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return destination


def design_archive(config):
    """Capture the runnable inputs before a solver or local device is available."""
    with tempfile.TemporaryDirectory() as folder:
        bundle = create_bundle(config, Path(folder)/'program')
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(bundle.rglob('*')):
                if file.is_file():
                    archive.write(file, file.relative_to(bundle))
        return buffer.getvalue()


def run_config(config, output, checkpoint=None):
    output=Path(output)
    config=Calibration.model_validate(config).model_dump()
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    status = run_job(config, output, checkpoint)
    if status == 0:
        result_zip(output, output/'python-notebook-results.zip')
    return status


def run_job(config, directory, checkpoint=None):
    from .calibration_replay import verify_source
    bundle=directory/'replay'
    if bundle.exists():
        manifest=verify_source(bundle)
        if not manifest.get('native_checkpoints') or json.loads((bundle/'config.json').read_text())!=config:
            raise ValueError('The saved worker or configuration cannot resume this job.')
    else:
        bundle=create_bundle(config,bundle)
    if checkpoint is not None:
        Path(checkpoint).mkdir(parents=True,exist_ok=True)
    env = {**os.environ,config['solver'].upper()+'_PYTHON':interpreter(config['solver'],config['device'])}
    command=[sys.executable,str(bundle/'run.py'),'--output',str(directory.resolve()/'calculation')]
    if checkpoint is not None:
        command+=['--checkpoint',str(Path(checkpoint).resolve())]
    return subprocess.call(command,env=env)


def result_zip(directory, target):
    from .calibration_replay import verify_source
    verify_source(directory/'replay')
    reference_hashes={}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
        for folder,prefix in [(directory/'replay',''),(directory/'calculation','reference/')]:
            for path in sorted(folder.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                    name=str(path.relative_to(folder))
                    archive.write(path,prefix+name)
                    if prefix:
                        with path.open('rb') as stream:
                            reference_hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
        archive.writestr('reference-sha256.json',json.dumps(reference_hashes,indent=2))


if __name__=='__main__':
    job=Path(sys.argv[1]).resolve()
    raise SystemExit(run_job(json.loads((job/'config.json').read_text()),job,job/'checkpoints'))
