"""Execute one real finite 3D field solve for each advertised geometry family."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.component_library import get_catalog, build_component
from backend.layout_study import LayoutStudy
from backend.runtime import interpreter,worker_env

# Explicit changes keep thin etched layers and long devices inside the bounded
# grid budget. Retain the original default recipes for unmodified cases.
OVERRIDES = {
 'rib': {'slab_height_um':.15,'height_um':.32,'length_um':4.},
 'grating_coupler': {'periods':6,'taper_um':1.,'aperture_um':2.,'slab_height_um':.14,'height_um':.28},
 'fan_grating': {'periods':4,'focus_um':2.,'fan_deg':30.,'slab_height_um':.14,'height_um':.28},
 'euler_bend': {'min_radius_um':1.2},
}
RESOLUTION = {'racetrack':14,'serpentine':14,'mmi_1x4':14}


def case(recipe, target, solver, device):
    family=recipe['id']
    params=OVERRIDES.get(family,{})
    design=build_component(family,params)
    # Place the source slightly inside the first geometric port.
    port=design['component']['ports'][0]
    angle=np.radians(port['angle_deg'])
    source={'x':port['x']-.20*np.cos(angle),'y':port['y']-.20*np.sin(angle),'z':port['z']}
    config=LayoutStudy(geometry=design['geometry'],source_um=source,solver=solver,device=device,
        resolution=RESOLUTION.get(family,16),duration_fs=24,pml_um=.8)
    folder=target/family
    folder.mkdir()
    (folder/'config.json').write_text(config.model_dump_json(indent=2))
    with (folder/'worker.log').open('w') as log:
        subprocess.run([interpreter(solver, device),'-m','backend.worker',folder/'config.json',folder],cwd=ROOT,
            env=worker_env(solver,device),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    meta=json.loads((folder/'metadata.json').read_text())
    with np.load(folder/'arrays.npz') as a:
        fields=a['fields']
        z,y,x=np.meshgrid(a['z_um'],a['y_um'],a['x_um'],indexing='ij',sparse=True)
        actual_source=meta['source']['position_um']
        separated=(x-actual_source['x'])**2+(y-actual_source['y'])**2+(z-actual_source['z'])**2 >= .6**2
        away=float(np.max(abs(fields[1][separated])))
        assert away > 0 and np.isfinite(fields).all() and np.ptp(a['epsilon'])>0
        record={'family':family,'overrides':params,'resolution':config.resolution,'duration_fs':config.duration_fs,
                'pml_um':config.pml_um,'wavelength_um':config.wavelength_um,'cladding_index':config.cladding_index,
                'source_um':source,'cells':config.layout()['cells'],'field_peak_ey_outside_0_6_um':away,
                'field_shape_zyx':list(a['field'].shape),'field_sha256':hashlib.sha256(fields.tobytes()).hexdigest(),
                'actual_device':meta['actual_device'],'compiled_material_sha256':meta['geometry_translation']['epsilon_xyz_sha256'],
                'native_material_check':meta['material_grid_check'],'implementation_sha256':meta['implementation_snapshot']['sha256']}
    print(family,record['cells'],away,flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--solver',choices=['meep','fdtdx'],default='meep')
    parser.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--families',nargs='+',help='Catalog family IDs; the default checks every family.')
    args=parser.parse_args()
    folder=args.output.resolve()
    folder.mkdir(parents=True,exist_ok=False)
    families=get_catalog()['components']
    if args.families:
        unknown=set(args.families)-{family['id'] for family in families}
        if unknown: parser.error('Unknown families: '+', '.join(sorted(unknown)))
        families=[f for f in families if f['id'] in args.families]
    with ThreadPoolExecutor(max_workers=2 if args.solver=='meep' else 1) as pool:
        cases=list(pool.map(lambda f: case(f,folder,args.solver,args.device),families))
    report={'checked_at':datetime.now(timezone.utc).isoformat(),'passed':True,'solver':args.solver,'device':args.device,
            'scope':'Native finite point-current field/material tests of one stated configuration per catalog family. No measured port efficiencies or calibrated device performance.', 'cases':cases}
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
