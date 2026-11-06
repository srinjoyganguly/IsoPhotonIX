"""Reproject measured native E/H with modes of the discrete propagation stencil."""
import argparse
import json
from pathlib import Path

import numpy as np

from backend.adapters.fdtdx_ports3d_adapter import collocate_mode, modal_amplitudes, port_basis
from scripts.measure_fdtdx_mesh_column import project_column
from scripts.phase_lead_dispersion import C_UM_S, section
from scripts.qualify_port_phase import native_column


def projection_bases(eps,f,dx,dt,model):
    def solve(freq):
        return port_basis(1/eps,freq,freq*C_UM_S,dx,dt,model['cladding_index'],model['core_index'])[0]
    current=solve(f)
    ft=f*np.sinc(f*C_UM_S*dt)
    discrete=solve(ft)
    kdx=2*np.arcsin(np.pi*ft*discrete['mode_neff'].real*dx)
    pairs=[collocate_mode(e,h,k,2*np.pi*ff*C_UM_S*dt)
           for e,h,k,ff in zip(discrete['raw_mode_e'],discrete['raw_mode_h'],kdx,f)]
    return ((current['mode_e'],current['mode_h']),
            (np.stack([e for e,h in pairs]),np.stack([h for e,h in pairs])))


def ribbon(folder):
    plan,report,column,_,hashes=native_column(folder)
    m,r=plan['model'],plan['resolution']
    if plan['material_model']!='polygon-area':
        raise ValueError('Require fractional extruded polygons.')
    launch=report['launch'];names=[p['name'] for p in m['ports']]
    idx=names.index(launch)
    detail=json.loads((folder/f'{launch}-reference.json').read_text())
    dt=detail['performance']['dt_seconds'];dx=1/r
    f=np.asarray(m['frequency_per_um'])
    all_bases={}
    bases={}
    for j,name in enumerate(names):
        eps=section(m,r,j)
        key=eps.tobytes()
        if key not in all_bases:
            all_bases[key]=projection_bases(eps,f,dx,dt,m)
        bases[name]=all_bases[key]
    bases['straight_through']=bases[launch]
    records=[[],[]]
    incident_side=0 if m['ports'][idx]['inward_sign']==1 else 1
    for role,monitors in [('reference',[launch,'straight_through']),('device',names)]:
        with np.load(folder/f'{launch}-{role}.npz') as native:
            source={k:native[k] for k in ('frequency_per_um','source_e','source_h')}
            for variant in range(2):
                extracted=np.stack([modal_amplitudes(native['phasors_'+name],*bases[name][variant],dx**2)[0] for name in monitors])
                if variant==0:
                    np.testing.assert_allclose(extracted,native['amplitudes'],rtol=1e-5,atol=1e-7)
                records[variant].append(dict(source,amplitudes=extracted))
    native,corrected=[project_column(m,idx,*pair)[0] for pair in records]
    np.testing.assert_allclose(native,column,rtol=3e-5,atol=1e-6)
    result={'case':m['case'],'resolution':r,'launch':launch,'native_archive_sha256':hashes,
            'native_source_and_evolution':'Identical measured FDTDX CUDA phasors; only the diagnostic modal projection changes.',
            'reference_reflected_amplitude':{},'ports':[]}
    for variant,name in enumerate(('original','discrete')):
        ref=records[variant][0]['amplitudes'][0]
        result['reference_reflected_amplitude'][name]=abs(ref[:,1-incident_side]/ref[:,incident_side]).tolist()
    for j,name in enumerate(names):
        a,b=native[:,j],corrected[:,j]
        result['ports'].append({'port':name,'original_power':(abs(a)**2).tolist(),'discrete_power':(abs(b)**2).tolist(),
                                'discrete_complex_real':b.real.tolist(),'discrete_complex_imag':b.imag.tolist(),
                                'complex_change':abs(a-b).tolist(),'phase_change_rad':[
                                    float(v) if min(abs(aa),abs(bb))>=.02 else None
                                    for aa,bb,v in zip(a,b,np.angle(b*a.conj()))]})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    result={'method':'First reconstruct the original port amplitudes from the stored full E/H planes. Evaluate transverse eigenprofiles at the temporal Yee symbol and collocate each with the discrete longitudinal propagation angle and original physical temporal angle. Normalize to the independently measured straight run. No phase alignment or fitted correction is used.',
            'scope':'Diagnostic reprojection of the same 3D GPU measurements; it does not change the native solver or production S matrix.',
            'samples':[ribbon(directory) for directory in args.directories]}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
