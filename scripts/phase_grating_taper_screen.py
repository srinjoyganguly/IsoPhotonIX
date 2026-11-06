"""Compare measured mesh-dependent guide phase to an untuned uniform-slab propagation screen."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.grating_phase_matching import sampled_te


def screen(path):
    data=json.loads(path.read_text())
    results=[]
    for first,second in zip(data['native_samples'],data['native_samples'][1:]):
        m=first['physical_model']
        if m!=second['physical_model'] or first['wavelength_um']!=second['wavelength_um']:
            raise ValueError('The physical model and monitor wavelengths must match.')
        w=np.array(first['wavelength_um'])
        meshes=[first['resolution'],second['resolution']]
        if data['vendor']=='lumerical':
            from scripts.lumerical_grating3d import model
            length=model(meshes[0])['target_length']
        else:
            length=m['taper']
        indices=[]
        for mesh in meshes:
            material=m.get('grating_material_model') or 'volume-average'
            indices.append(np.array([sampled_te(m,m['height'],wave,mesh,material) for wave in w]))
        phase=(indices[1]-indices[0])*2*np.pi*length/w
        signals=[np.array(r['registered_complex_s_real'])[:,1]+1j*np.array(r['registered_complex_s_imag'])[:,1] for r in (first,second)]
        valid=np.array(first['valid_bins'])&np.array(second['valid_bins'])&np.array(first['phase_ok'])[:,1]&np.array(second['phase_ok'])[:,1]
        measured=np.angle(signals[1]*signals[0].conj())
        difference=np.angle(np.exp(1j*(measured-phase)))
        results.append({'mesh_pair':meshes,'taper_length_um':length,'wavelength_um':w.tolist(),'valid':valid.tolist(),
                        'slab_neff_at_first_mesh':indices[0].tolist(),'slab_neff_at_second_mesh':indices[1].tolist(),
                        'wide_uniform_slab_phase_change_rad':phase.tolist(),'measured_principal_guide_phase_change_rad':
                            [float(v) if q else None for v,q in zip(measured,valid)],
                        'principal_residual_rad':[float(v) if q else None for v,q in zip(difference,valid)]})
    return {'vendor_geometry':data['vendor'],'scope':'A uniform, symmetric oxide-clad unetched slab of the actual silicon thickness, using the actual z mesh and the Yee propagation symbols, over the physical taper length. It omits narrowing, distributed grating radiation, substrate and fiber phase. No fitted optical length, phase shift, or modification of S. Residuals are modulo 2π; this cannot identify an absolute vendor phase or unwrapped delay.',
            'mesh_pairs':results}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();args.output.write_text(json.dumps(screen(args.receipt),indent=2,allow_nan=False)+'\n')
