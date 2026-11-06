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

from pydantic import ConfigDict, BaseModel, Field, model_validator
from typing import Literal

from .runtime import ROOT, interpreter

CASES = {
    'coupler': ('Directional coupler', 'Four separated fundamental-mode ports in a finite-height silicon circuit.'),
    'splitter': ('Y splitter', 'Three guided ports, with an independent launch from the stem and each branch.'),
    'ring': ('Add/drop ring', 'Calibrated bus and drop spectra in a planar effective-index ring.'),
    'grating': ('Grating radiation', 'Guided input, upward radiation and Gaussian coupling in a planar grating.'),
    'multimode': ('Multimode slab joint', 'Two TE modes on each side of a planar offset joint, checked against independent mode matching.'),
    'ring-3d': ('3D bus-loaded ring', 'A full-vector, finite-height ring with two bus guides and a separate incident reference.'),
    'grating-3d': ('3D finite-width grating', 'A finite-width grating and taper with Gaussian illumination and a calibrated guided output.'),
    'layout-ports-3d': ('3D layout ports', 'Full-vector scattering between user-positioned ports of a finite-height polygon layout on NVIDIA.'),
    'layout-ports-2d': ('Planar layout ports', 'Excite user-positioned ports on a lossless polygon layout and measure its complex fundamental-mode matrix.'),
    'multimode-tm': ('TM multimode slab joint', 'Two independently launched transverse-magnetic modes on each side of a planar joint.'),
    'multimode-3d': ('Finite-height multimode joint', 'Two quasi-TE modes on each side of a finite-height offset silicon strip.'),
}


GPU_COMPONENTS = ('ring-3d', 'grating-3d')


LAYOUT_CASES = ('layout-ports-2d','layout-ports-3d')


def grating_meshes(geometry, domain, material):
    if material=='direct-yee' and geometry!='lumerical':
        return []
    if geometry not in ('lumerical','tidy3d'):
        return [] if domain=='mirror-y' else [12,16]
    if domain=='mirror-y' and geometry=='lumerical' and material!='direct-yee':
        return []
    defaults=[] if domain=='mirror-y' else [12,14]
    catalogue=ROOT/'docs/verification/v014-grating-meshes.json'
    if catalogue.is_file():
        routes=json.loads(catalogue.read_text())['qualified']
        for row in routes:
            if (row['geometry'],row['domain'],row['material'])==(geometry,domain,material):
                defaults.append(row['resolution'])
    return sorted(set(defaults))


def calibration_routes(case):
    if case in GPU_COMPONENTS or case=='layout-ports-3d':
        return ['fdtdx:gpu']
    if case in ('coupler', 'splitter'):
        return ['meep:cpu', 'fdtdx:cpu', 'fdtdx:gpu']
    return ['meep:cpu']


class Calibration(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    study_type: Literal['calibrated-component'] = 'calibrated-component'
    revision: Literal[1,2] = 2
    case: Literal['coupler','splitter','ring','grating','multimode','multimode-3d','ring-3d','grating-3d','multimode-tm','layout-ports-2d','layout-ports-3d']
    solver: Literal['meep','fdtdx']
    device: Literal['cpu','gpu']
    resolution: int | None = None
    offset_um: float | None = None
    material_model: Literal['legacy-x','yee-aligned','polygon-area'] | None = None
    ring_scope: Literal['full','lumerical','tidy3d'] | None = None
    grating_scope: Literal['full','tidy3d-compact','lumerical-article','lumerical-project','lumerical-output','tidy3d-radiation','lumerical-radiation'] | None = None

    # Omit new optional keys from historical specifications and captured job settings.
    geometry_3d: Literal['compact','compact-narrow','compact-radiation','lumerical','tidy3d'] | None = Field(default=None, exclude_if=lambda v: v is None)
    port_distance_um: float | None = Field(default=None, exclude_if=lambda v: v is None)
    planar_layout: dict | None = Field(default=None, exclude_if=lambda v: v is None)
    duration_um_c: float | None = Field(default=None, exclude_if=lambda v: v is None)
    ring_interface: Literal['legacy','exact-area','wall-tensor'] | None = Field(default=None, exclude_if=lambda v: v is None)
    ring_material: Literal['constant','palik-ade'] | None = Field(default=None, exclude_if=lambda v: v is None)
    grating_material_model: Literal['volume-average','direct-yee'] | None = Field(default=None, exclude_if=lambda v: v is None)
    grating_domain: Literal['full','mirror-y'] | None = Field(default=None, exclude_if=lambda v: v is None)
    grating_spectrum: Literal['standard','dense'] | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode='before')
    @classmethod
    def load_saved_revision(cls, settings):
        # Earlier exported configurations include the study_type discriminator
        # but predate explicit revisions. Preserve their numerical defaults.
        if isinstance(settings,dict) and settings.get('study_type')=='calibrated-component' and 'revision' not in settings:
            return {**settings,'revision':1}
        return settings

    @model_validator(mode='after')
    def available_route(self):
        if self.planar_layout is not None and self.case not in LAYOUT_CASES:
            raise ValueError('A planar layout belongs to the planar layout ports preset.')
        if self.case in LAYOUT_CASES:
            from .planar_port_layout import default_planar_port_layout, validate_planar_port_layout
            document = default_planar_port_layout() if self.planar_layout is None else self.planar_layout
            self.planar_layout = validate_planar_port_layout(document)
        if self.case in (*GPU_COMPONENTS, 'multimode-tm', *LAYOUT_CASES) and self.revision != 2:
            raise ValueError('New calibrated components require revision 2.')
        if self.geometry_3d is not None and self.case not in GPU_COMPONENTS:
            raise ValueError('Choose a full 3D ring or grating for the reference geometry.')
        if self.port_distance_um is not None and self.case != 'multimode-tm':
            raise ValueError('Movable port planes are supported on the planar TM joint.')
        if self.duration_um_c is not None and self.case != 'ring-3d':
            raise ValueError('Choose the 3D ring to set an observation duration.')
        if (self.ring_interface is not None or self.ring_material is not None) and self.case != 'ring-3d':
            raise ValueError('Ring interface and material settings require a full 3D ring.')
        if any(v is not None for v in (self.grating_material_model,self.grating_spectrum,self.grating_domain)) and self.case != 'grating-3d':
            raise ValueError('Grating material or spectrum settings require a full 3D grating.')
        if self.case == 'layout-ports-3d':
            if f'{self.solver}:{self.device}' not in calibration_routes(self.case):
                raise ValueError('Select the NVIDIA FDTDX target for finite-height layout ports.')
            if any(value is not None for value in (self.offset_um, self.ring_scope, self.grating_scope, self.material_model)):
                raise ValueError('The finite-height layout uses its captured constant-index stack.')
            if self.resolution is None:
                self.resolution = 16
            if self.resolution not in (12,16):
                raise ValueError('Choose 12 or 16 cells/µm for the finite-height layout.')
            return self
        if self.case in GPU_COMPONENTS:
            if f'{self.solver}:{self.device}' not in calibration_routes(self.case):
                raise ValueError('This 3D preset requires explicit FDTDX NVIDIA execution.')
            if any(value is not None for value in (self.offset_um, self.ring_scope, self.grating_scope, self.material_model)):
                raise ValueError('The 3D component worker uses its directly specified geometry and material sampling.')
            ring = self.case == 'ring-3d'
            geometries = ('lumerical','tidy3d') if ring else ('compact','compact-narrow','compact-radiation','tidy3d','lumerical')
            self.geometry_3d = self.geometry_3d or geometries[0]
            if self.grating_material_model=='direct-yee' and self.geometry_3d!='lumerical':
                raise ValueError('Direct Yee-cell material sampling requires the Lumerical curved grating.')
            if self.geometry_3d not in geometries:
                raise ValueError(f'Available {self.case} geometries: {", ".join(geometries)}.')
            meshes = (12,16,20,24) if ring else grating_meshes(self.geometry_3d,self.grating_domain or 'full',
                                                              self.grating_material_model or 'volume-average')
            if self.grating_domain=='mirror-y' and self.grating_spectrum=='dense':
                raise ValueError('Mirror symmetry is qualified only for the standard grating spectrum.')
            if self.resolution is None:
                self.resolution = 16 if ring else 12
            if self.ring_interface=='wall-tensor' and self.resolution<16:
                raise ValueError('The ring wall tensor requires at least 16 cells/µm.')
            if self.ring_material=='palik-ade' and (self.geometry_3d!='lumerical' or self.ring_interface!='exact-area' or self.resolution!=16):
                raise ValueError('The measured dispersive ring uses the Lumerical geometry, exact area and 16 cells/µm.')
            if self.grating_spectrum=='dense' and (self.geometry_3d!='lumerical' or self.resolution!=12):
                raise ValueError('The 21-sample spectrum requires the Lumerical grating at 12 cells/µm.')
            if self.resolution not in meshes:
                raise ValueError(f'Available {self.case} meshes: {meshes} cells/µm.')
            if ring and self.duration_um_c is None:
                self.duration_um_c = 3200 if self.geometry_3d=='lumerical' else 4800
            if self.duration_um_c is not None and not 100 <= self.duration_um_c <= 6000:
                raise ValueError('Choose a ring observation duration between 100 and 6,000 µm/c.')
            return self
        if self.revision==2:
            if self.solver=='fdtdx' and self.material_model is None:
                self.material_model='yee-aligned'
            if self.case=='ring' and self.ring_scope is None:
                self.ring_scope='lumerical'
            if self.case=='grating' and self.grating_scope is None:
                self.grating_scope='tidy3d-compact'
            if self.resolution is None:
                default_mesh={'coupler':16,'splitter':16,'multimode':48,'multimode-3d':20,'multimode-tm':64,'layout-ports-2d':32}
                if self.case=='grating' and self.grating_scope!='full':
                    default_mesh['grating']=40
                self.resolution=default_mesh.get(self.case)
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
        meshes = {'coupler':(16,20,24),'splitter':(16,20,24),'multimode':(32,48,64),'multimode-3d':(16,20,24),'multimode-tm':(48,64,80),'layout-ports-2d':(24,32,40)}
        if self.case=='grating' and self.grating_scope not in (None,'full'):
            meshes['grating']=(28,40,60)
        if self.solver=='fdtdx':
            meshes[self.case]=(12,16,20,24)
        if self.resolution is not None and self.resolution not in meshes.get(self.case,()):
            raise ValueError('Choose a supported mesh for this preset.')
        if self.offset_um is not None and (not self.case.startswith('multimode') or not 0 <= self.offset_um <= .25):
            raise ValueError('The offset is available only for multimode joints, between 0 and 0.25 µm.')
        if self.case in LAYOUT_CASES:
            for port in self.planar_layout['ports']:
                axis = 0 if port['outward'].endswith('x') else 1
                cells = port['center_um'][axis] * self.resolution
                if abs(cells - round(cells)) > 1e-8:
                    raise ValueError(f"Place the {port['name']} port plane on the selected mesh.")
        if self.case == 'multimode-tm':
            if self.port_distance_um is None:
                self.port_distance_um = 1.25
            if self.offset_um is None:
                self.offset_um = .125
            distance = self.port_distance_um
            if not .75 <= distance <= 1.75:
                raise ValueError('Choose port planes 0.75–1.75 µm from the TM joint.')
            cells = distance * self.resolution
            if abs(cells - round(cells)) > 1e-8:
                raise ValueError('Choose a TM port plane on the selected mesh.')
        return self


def specification(config):
    if config.case in GPU_COMPONENTS:
        ring = config.case == 'ring-3d'
        worker = 'calibrate_loaded_ring3d' if ring else 'calibrate_finite_grating'
        flag = '--vendor' if ring else '--geometry'
        args = ['--device', 'gpu', flag, config.geometry_3d, '--resolution', str(config.resolution)]
        if config.duration_um_c is not None:
            args += ['--duration-um-c', str(config.duration_um_c)]
        if config.ring_interface is not None:
            args += ['--ring-interface',config.ring_interface]
        if config.ring_material is not None:
            args += ['--ring-material',config.ring_material]
        if config.grating_material_model is not None:
            args += ['--grating-material-model',config.grating_material_model]
        if config.grating_domain is not None:
            args += ['--grating-domain',config.grating_domain]
        if config.grating_spectrum is not None:
            args += ['--grating-spectrum',config.grating_spectrum]
        helpers = ['backend/__init__.py', 'scripts/gpu_timing.py', *['backend/adapters/' + n + '.py' for n in (
            'fdtdx_adapter', 'fdtdx3d_adapter', 'fdtdx_ports_adapter', 'fdtdx_ports3d_adapter')]]
        if ring:
            helpers += ['scripts/calibrate_ring.py','scripts/measure_loaded_ring_losses.py','scripts/calibration_checkpoint.py',
                        'scripts/ring_stage_provenance.py','scripts/ring_wall_tensor.py']
            if config.ring_material=='palik-ade':
                helpers += ['scripts/ring_ade_stage.py','scripts/ring_palik_dispersion.py','docs/verification/v013-ring-project-audit.json']
        else:
            helpers += ['scripts/grating_mirror.py','scripts/grating_cuda_identity.py','scripts/grating_emission.py','scripts/lumerical_grating3d.py','scripts/grating_angular_spectrum.py', 'docs/verification/v012-grating-commercial-references.json','docs/verification/v012-grating-memory-budget.json']
            if config.resolution >= 16 and config.geometry_3d in ('lumerical','tidy3d'):
                mirror='-mirror' if config.grating_domain=='mirror-y' else ''
                helpers += [f'docs/verification/v014-grating-memory-{config.geometry_3d}{mirror}-r{config.resolution}.json']
            if config.grating_spectrum=='dense':
                helpers += ['docs/verification/v013-grating-dense-reference.json','docs/verification/v013-grating-dense-memory.json']
        return worker, args, helpers
    if config.case == 'layout-ports-3d':
        helpers = ['backend/__init__.py','backend/planar_port_layout.py','backend/geometry_draft.py','backend/calibrated_geometry.py',
                   'scripts/gpu_timing.py','scripts/phase_polygon_material.py','scripts/grating_cuda_identity.py','scripts/calibration_checkpoint.py','scripts/calibrate_fdtdx_multiport.py']
        helpers += ['backend/adapters/'+name+'.py' for name in (
            'fdtdx_adapter','fdtdx3d_adapter','fdtdx_ports_adapter','fdtdx_ports3d_adapter')]
        return 'calibrate_layout_ports3d', ['--design','layout.json','--device','gpu','--resolution',str(config.resolution)],helpers
    if config.case == 'layout-ports-2d':
        helpers = ['backend/__init__.py','backend/planar_port_layout.py','backend/geometry_draft.py','scripts/calibration_checkpoint.py']
        return 'calibrate_layout_ports', ['--design','layout.json','--resolution',str(config.resolution)], helpers
    if config.case == 'multimode-tm':
        args = ['--resolution', str(config.resolution)]
        if config.offset_um is not None:
            args += ['--offset', str(config.offset_um)]
        if config.port_distance_um is not None:
            args += ['--port-distance', str(config.port_distance_um)]
        return 'calibrate_planar_tm', args, ['scripts/calibration_checkpoint.py']
    multiport=config.case in ('coupler','splitter')
    if multiport:
        name='calibrate_fdtdx_multiport' if config.solver=='fdtdx' else 'calibrate_multiport'
        args=['--case',config.case]
        helpers=['backend/__init__.py','backend/calibrated_geometry.py','scripts/calibration_checkpoint.py']
        if config.solver=='fdtdx':
            args+=['--device',config.device,'--material-model',config.material_model or 'legacy-x']
            helpers+=['scripts/gpu_timing.py','scripts/phase_polygon_material.py']
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
    if config.planar_layout is not None:
        (destination/'source/layout.json').write_text(json.dumps(config.planar_layout,indent=2,allow_nan=False)+'\n')
    if config.solver=='fdtdx':
        notices=destination/'docs/licenses'
        notices.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'docs/licenses/FDTDX-MIT.txt',notices/'FDTDX-MIT.txt')
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
        'Set PHOTONICS_CHECKPOINT to the path of a compatible local checkpoint to reuse measured stages in an unmodified notebook. '
        'In the notebook, pass checkpoint=Path("stages") to run() when you want that behavior. '
        'A completed run export contains reference/report.json, numerical NPZ files and their hashes. '
        'A fresh replay compares numerical arrays with the saved samples. Small floating-point differences are listed in reproduction.json. '
        'Physical lengths are in micrometres unless identified otherwise in the report.\n')
    cells=[
        {'cell_type':'markdown','metadata':{},'id':'guide','source':[
            '# '+CASES[config.case][0]+'\n',
            'Run with local Python and NumPy. The controller selects the saved solver and device. ',
            'The registered worker or MEEP_PYTHON / FDTDX_PYTHON must refer to that environment. ',
            'Each execution writes a new result directory. The notebook can be saved or edited; the captured solver and settings are checked before execution. Results are in report.json and arrays.npz.']},
        {'cell_type':'code','execution_count':None,'outputs':[],'metadata':{},'id':'execute','source':[
            'from pathlib import Path\n','from uuid import uuid4\n','from run import run\n',
            "output = Path('notebook-result')\n",
            "if output.exists():\n","    output = Path('notebook-result-' + uuid4().hex)\n",
            "report = run(Path.cwd(), output)\n","print(output.resolve())\n",'report']}
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
    from .calibration_replay import diagnostic_ring, verify_source
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
        report_path=directory/'calculation/report.json'
        if report_path.is_file() and 'gpu-performance.csv' not in reference_hashes:
            from .calibration_performance import stage_rows, timing_csv
            report=json.loads(report_path.read_text())
            if diagnostic_ring(report,report_path.parent):
                archive.writestr('diagnostic.json',json.dumps({'numerical_gate_passed':False,'qualified_wavelengths':report['valid_bins'],
                    'total_wavelengths':report['total_bins'],'scope':'Read the wavelength validity mask when using these measured fields.'},indent=2))
            rows=stage_rows(report,'report.json')
            if rows:
                timing=timing_csv(rows).encode()
                archive.writestr('reference/gpu-performance.csv',timing)
                reference_hashes['gpu-performance.csv']=hashlib.sha256(timing).hexdigest()
        archive.writestr('reference-sha256.json',json.dumps(reference_hashes,indent=2))


if __name__=='__main__':
    job=Path(sys.argv[1]).resolve()
    raise SystemExit(run_job(json.loads((job/'config.json').read_text()),job,job/'checkpoints'))
