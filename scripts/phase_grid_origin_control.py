"""Compare fresh native columns at two fixed vertical mesh registrations."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_fdtdx_multiport import load_stage
from scripts.phase_publish_columns import stage_performance
from scripts.qualify_port_phase import native_column


def prediction(original):
    """Predict the effect on straight propagation before observing the shifted 3D fields."""
    from scripts.phase_lead_dispersion import C_UM_S,mode_index,section
    plan,report,_,_,hashes=native_column(original)
    model,r=plan['model'],plan['resolution']
    if model.get('z_grid_origin_um',0)!=0:
        raise ValueError('Use the original zero-registration reference.')
    source=report['launch']
    ref=json.loads((original/f'{source}-reference.json').read_text())
    dt=ref['performance']['dt_seconds']
    f=np.array(model['frequency_per_um'])
    effective=f*np.sinc(f*C_UM_S*dt)
    length=max(p['actual_x'] for p in report['sampled_ports'])-min(p['actual_x'] for p in report['sampled_ports'])
    betas=[]
    for shift in (0.,.5):
        eps=section(model,r,origin_shift_cells=(0.,shift))
        n=np.array([mode_index(eps,fe,1/r) for fe in effective])
        betas.append(2*r*np.arcsin(np.pi*effective*n/r))
    return {'prepared_utc':datetime.now(timezone.utc).isoformat(),'case':model['case'],'mesh':r,
            'frequency_per_um':f.tolist(),'old_reference_sha256':hashes[f'{source}-reference.npz'],'old_column_sha256':hashes['arrays.npz'],
            'straight_guide_length_um':length,'predicted_half_minus_zero_straight_phase_rad':((betas[1]-betas[0])*length).tolist(),
            'method':'Independent matched transverse eigensolutions at the temporal Yee frequencies, translated by zero/half a z cell; fuse with the longitudinal Yee wavenumber. No shifted 3D measurement is an input.'}


def summarize(baseline,shifted,prior_reference,forecast):
    cases=[]
    for path in (baseline,shifted):
        plan,report,column,straight,hashes=native_column(path)
        model=plan['model'].copy()
        z0=model.pop('z_grid_origin_um')
        r=plan['resolution'];p=report['sampled_ports'][0]
        perf,origin=stage_performance(path,p['name'],report)
        if any(value!='fresh' for value in origin.values()):
            raise ValueError('The two grid-origin controls must both be freshly acquired.')
        nz=round(model['cell'][2]*r);domain=z0+np.array([-nz/2,nz/2])/r
        inner=domain+np.array([1,-1])*round(model['pml_um']*r)/r
        source_ap=z0+(np.array([p['start'][2],p['start'][2]+p['shape'][2]])-nz/2)/r
        fingerprints={}
        for role in ('reference','device'):
            stem=p['name']+'-'+role
            native=load_stage(path/(stem+'.json'),path/(stem+'.npz'))
            fingerprints[role]={k:native[k] for k in ('yee_component_grid_sha256','source_profile_sha256','field_device')}
        item={'stage_material_and_source':fingerprints,'requested_gpu_allocator':plan['requested_gpu_allocator'],'z_grid_origin_um':z0,'z_grid_origin_cells':z0*r,'domain_z_bounds_um':domain.tolist(),
              'pml_inner_z_faces_um':inner.tolist(),'source_and_port_aperture_z_bounds_um':source_ap.tolist(),
              'physical_si_z_faces_um':[-model['height']/2,model['height']/2],'port_x_y_um':[
                  [x['actual_x'],x['actual_y']] for x in report['sampled_ports']],
              's_real':column.real.tolist(),'s_imag':column.imag.tolist(),
              'straight_transfer_real':straight.real.tolist(),'straight_transfer_imag':straight.imag.tolist(),
              'stage_performance':perf,'stage_acquisition':origin,'stage_sha256':report['stage_sha256'],
              'native_arrays_sha256':hashes['arrays.npz'],'memory_preflight':report['memory_preflight']}
        cases.append((plan,model,item,column))
    a,b=cases
    for key in ('actual_device','device_name','resolution','material_model','numerical_sha256','versions','requested_gpu_allocator'):
        if a[0][key]!=b[0][key]:
            raise ValueError(f'The paired native plans differ in {key}.')
    if (a[1]!=b[1] or a[2]['z_grid_origin_cells']!=0. or b[2]['z_grid_origin_cells']!=.5
            or a[2]['port_x_y_um']!=b[2]['port_x_y_um']):
        raise ValueError('Only the vertical grid registration may change.')
    original=np.load(prior_reference,allow_pickle=False)
    with original:
        replay_delta=abs(a[3]-original['s_column'])
        if replay_delta.max()>.00001:
            raise ValueError('Fresh zero-offset result does not reproduce the original native column.')
    result={'method':'Zero versus positive half-cell z lattice origin at unchanged resolution. The physical silicon edges and x/y port planes stay fixed. Both z-domain faces, inner PML faces, and finite launch/receiver aperture edges move by the declared amount. This experiment retains their combined sensitivity.',
            'case':a[1]['case'],'mesh':a[0]['resolution'],'numerical_sha256':a[0]['numerical_sha256'],
            'prior_reference_sha256':hashlib.sha256(prior_reference.read_bytes()).hexdigest(),'fresh_zero_vs_prior_complex_delta_max':float(replay_delta.max()),
            'acquisitions':[a[2],b[2]],'ports':[]}
    forecast_data=json.loads(forecast.read_text())
    if (forecast_data['case']!=a[1]['case'] or forecast_data['mesh']!=a[0]['resolution']
            or forecast_data['old_column_sha256']!=result['prior_reference_sha256']):
        raise ValueError('The independent forecast does not match the native circuit/mesh.')
    straight=[np.array(v['straight_transfer_real'])+1j*np.array(v['straight_transfer_imag']) for v in result['acquisitions']]
    predicted=np.array(forecast_data['predicted_half_minus_zero_straight_phase_rad'])
    measured=np.angle(straight[1]*straight[0].conj())
    result['straight_phase_prediction']={'forecast_sha256':hashlib.sha256(forecast.read_bytes()).hexdigest(),
        'predicted_half_minus_zero_rad':predicted.tolist(),'measured_half_minus_zero_rad':measured.tolist(),
        'signed_residual_rad':np.angle(np.exp(1j*(measured-predicted))).tolist()}
    for j,port in enumerate(a[1]['ports']):
        x,y=a[3][:,j],b[3][:,j]
        strong=(abs(x)>=.02)&(abs(y)>=.02)
        result['ports'].append({'name':port['name'],'max_absolute_complex_change':float(max(abs(x-y))),
            'max_absolute_power_change':float(max(abs(abs(x)**2-abs(y)**2))),
            'phase_change_rad':[float(v) if good else None for v,good in zip(np.angle(y*x.conj()),strong)]})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_subparsers(dest='action',required=True)
    forecast=modes.add_parser('predict');forecast.add_argument('original',type=Path);forecast.add_argument('--output',required=True,type=Path)
    compare=modes.add_parser('compare');compare.add_argument('baseline',type=Path);compare.add_argument('shifted',type=Path)
    compare.add_argument('--prior-reference',required=True,type=Path);compare.add_argument('--prediction',required=True,type=Path)
    compare.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    result=prediction(args.original) if args.action=='predict' else summarize(args.baseline,args.shifted,args.prior_reference,args.prediction)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
