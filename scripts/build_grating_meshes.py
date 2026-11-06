"""Expose refined grating configurations only after field acquisition and symmetry validation."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]/'docs/verification'


def collect(base):
    candidates={}
    for path in sorted(base.glob('v014-grating3d-*.json')):
        run=json.loads(path.read_text())
        if run.get('case_id')!='finite-grating-3d' or run.get('actual_device')!='cuda:0' or not run['passed']:
            continue
        model=run['settings']
        geometry,mesh=model['geometry'],model['resolution']
        domain=model.get('grating_domain','full')
        if geometry not in ('lumerical','tidy3d') or (domain=='full' and mesh<16):
            continue
        raw=base/run['arrays_file']
        if raw.parent!=base:
            raise ValueError('Grating field data must be in the published verification directory.')
        with raw.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=run['arrays_sha256']:
                raise ValueError(f'Native field data changed for {path.name}.')
        material=model.get('grating_material_model','volume-average')
        key=(geometry,domain,material,mesh)
        item={'geometry':geometry,'domain':domain,'material':material,'resolution':mesh,'run':path.name}
        if domain=='mirror-y':
            reference=base/f'v014-grating-mirror-{geometry}-r12.json'
            if not reference.is_file():
                continue
            proof=json.loads(reference.read_text())
            if not (proof['qualified'] and proof['identical_full_source'] and proof['geometry']==geometry):
                continue
            item['reference']=reference.name
        if mesh>=16:
            mirror='-mirror' if domain=='mirror-y' else ''
            profile=base/f'v014-grating-memory-{geometry}{mirror}-r{mesh}.json'
            if not profile.is_file():
                raise ValueError(f'No measured memory profile for {path.name}.')
            item['memory_profile']=profile.name
        tracks_return='reciprocal_return_settling' in run.get('radiation',{})
        candidates.setdefault(key,[]).append((tracks_return,path.name,item))
    qualified=[]
    for key in sorted(candidates):
        # Prefer measured return histories while retaining the independent run
        # that first established this mesh. Filenames alone do not encode coverage.
        records=sorted(candidates[key],key=lambda entry:(entry[0],entry[1]),reverse=True)
        selected=records[0][2]
        if len(records)>1:
            selected['other_acquisitions']=[entry[1] for entry in records[1:]]
        qualified.append(selected)
    return {'qualified':qualified}


if __name__=='__main__':
    destination=BASE/'v014-grating-meshes.json'
    destination.write_text(json.dumps(collect(BASE),indent=2,allow_nan=False)+'\n')
