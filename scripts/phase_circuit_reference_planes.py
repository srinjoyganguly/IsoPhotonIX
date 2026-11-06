"""Transport the measured column only through verified uniform external leads."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.phase_lead_dispersion import section
from scripts.qualify_port_phase import native_column


def check_uniform_lead(model,outer_x,edge_x):
    lo,hi=sorted([outer_x,edge_x])
    if hi==lo:
        raise ValueError('The reference plane must move along a nonzero uniform lead.')
    for polygon in model['polygons']:
        for (x0,y0),(x1,y1) in zip(polygon,polygon[1:]+polygon[:1]):
            overlap=min(hi,max(x0,x1))-max(lo,min(x0,x1))
            if overlap>1e-12 and abs(y1-y0)>1e-12:
                raise ValueError('A port translation would cross a varying circuit section.')


def registered(folder,lead_rows):
    plan,report,s,through,hashes=native_column(folder)
    model,r=plan['model'],plan['resolution']
    lead=next(row for row in lead_rows if row['case']==model['case'] and row['resolution']==r)
    port=model['ports'][0]['name']
    if lead['native_reference_sha256']!=hashes[f'{port}-reference.npz']:
        raise ValueError('The propagation constant must come from this exact native straight guide.')
    f=np.array(model['frequency_per_um'])
    k_est=2*np.pi*f*np.array(lead['yee_longitudinal_neff'])
    length=lead['straight_plane_distance_um']
    residual=np.angle(through*np.exp(-1j*k_est*length))
    if max(abs(residual))>.05:
        raise ValueError('The independent discrete lead model does not resolve the phase branch.')
    beta=k_est+residual/length
    # These two physical coordinates bound the sampled varying polygons.
    bounds={'coupler':4.,'splitter':2.}
    end=bounds[model['case']]
    targets=[];removed=[]
    for j,p in enumerate(report['sampled_ports']):
        edge=np.sign(p['actual_x'])*end
        check_uniform_lead(model,p['actual_x'],edge)
        # The straight control supplies the propagation constant for all ports
        # only when their physical sampled strip cross sections agree.
        np.testing.assert_array_equal(section(model,r,j),section(model,r,0))
        targets.append(float(edge))
        removed.append(abs(edge-p['actual_x']))
    corrected=s*np.exp(-1j*beta[:,None]*(removed[0]+np.array(removed))[None,:])
    np.testing.assert_allclose(abs(corrected),abs(s),rtol=1e-14,atol=1e-15)
    return {'case':model['case'],'mesh':r,'port_order':report['port_order'],'actual_port_x_um':[p['actual_x'] for p in report['sampled_ports']],
            'virtual_port_x_um':targets,'uniform_length_removed_per_port_um':removed,'measured_beta_rad_um':beta.tolist(),
            'native_column_sha256':hashes['arrays.npz'],'frequency_per_um':f.tolist(),
            'registered_s_real':corrected.real.tolist(),'registered_s_imag':corrected.imag.tolist()},corrected


def analyze(paths,lead_file):
    rows=json.loads(lead_file.read_text())['samples'];all_data=[registered(path,rows) for path in paths]
    result={'method':'Remove only the uniform external lengths by multiplying by exp(-iβ(L_in+L_out)). The straight propagated field supplies β, its independently solved Yee dispersion chooses the number of turns. No phase offset is fitted. The published native physical-plane S is unchanged.',
            'scope':'Fundamental-mode virtual reference planes at the start/end of the varying polygon. The entire curved circuit remains between them.',
            'studies':[record for record,s in all_data],'mesh_pairs':[]}
    for (a,x),(b,y) in zip(all_data,all_data[1:]):
        if a['case']!=b['case'] or a['mesh']>=b['mesh'] or a['virtual_port_x_um']!=b['virtual_port_x_um']:
            raise ValueError('Use increasing meshes of one circuit with matching virtual planes.')
        ports=[]
        for i,name in enumerate(b['port_order']):
            valid=(abs(x[:,i])>=.02)&(abs(y[:,i])>=.02)
            change=np.angle(y[:,i]*x[:,i].conj())
            ports.append({'port':name,'complex_amplitude_delta_max':float(np.max(abs(y[:,i]-x[:,i]))),
                          'physical_phase_delta_rad':[float(v) if ok else None for v,ok in zip(change,valid)]})
        result['mesh_pairs'].append({'meshes':[a['mesh'],b['mesh']],'ports':ports})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directories',nargs='+',type=Path)
    p.add_argument('--lead-receipt',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();args.output.write_text(json.dumps(analyze(args.directories,args.lead_receipt),indent=2,allow_nan=False)+'\n')
