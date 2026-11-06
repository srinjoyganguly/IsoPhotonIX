"""Audit a rejected 3D Gaussian source pair against a deliberately prescribed sampled sheet."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.calibrate_finite_grating import normal_flux,grid,sampled_gaussian_fields,digest,file_hash
from scripts.grating_angular_spectrum import incident_beam_check


def field_difference(a,b):
    x=a.astype(np.float64); y=b.astype(np.float64)
    scale=float(np.vdot(x,y)/np.vdot(x,x))
    return {'sample_shape':list(x.shape),'max_absolute':float(np.max(abs(y-x))),
            'relative_l2':float(np.linalg.norm(y-x)/np.linalg.norm(x)),
            'fitted_scale':scale,'relative_shape_residual':float(np.linalg.norm(y-scale*x)/np.linalg.norm(x)),
            'polarity_changes':int(np.sum(np.sign(x)!=np.sign(y)))}


def angle(e,h):
    direction=np.cross(e[:,:,:,0].transpose(1,2,0).astype(float),h[:,:,:,0].transpose(1,2,0).astype(float))
    vector=direction.sum(axis=(0,1))
    return float(np.rad2deg(np.arctan2(vector[0],-vector[2])))


def coarse_reference_control():
    base=ROOT/'docs/verification'
    report=json.loads((base/'v012-grating3d-tidy3d-r12.json').read_text())
    archive=base/report['arrays_file'];m=report['settings']
    if file_hash(archive)!=report['arrays_sha256']:raise ValueError('Coarse numerical archive changed.')
    shape,pml,axes=grid(m)
    left=max(pml,int(np.argmin(abs(axes[0]-m['beam_x']+2*m['waist']))))
    right=min(shape[0]-pml,int(np.argmin(abs(axes[0]-m['beam_x']-2*m['waist'])))+1)
    e,h=sampled_gaussian_fields(axes[0][left:right],axes[1][pml:-pml],m['beam_x'],m['waist'],m['angle_deg'])
    with np.load(archive,allow_pickle=False) as data:
        return {'receipt':'v012-grating3d-tidy3d-r12.json','arrays_sha256':report['arrays_sha256'],
                'source_E':field_difference(data['ref_source_E'],e),
                'source_H':field_difference(data['ref_source_H'],h),
                'prescribed_x_angle_deg':angle(e,h),
                'measured_original_x_angle_deg':report['verification']['incident_beam_direction']['mean_x_angle_deg']}


def audit(reference,device,corrected=None):
    rr=json.loads((reference/'receipt.json').read_text());dr=json.loads((device/'receipt.json').read_text())
    m=rr['identity']['settings']
    if m!=dr['identity']['settings'] or m['geometry']!='tidy3d':
        raise ValueError('Expected the same Tidy3D geometry and mesh.')
    for r,base in [(rr,reference),(dr,device)]:
        if r['arrays_sha256']!=file_hash(base/'arrays.npz'):raise ValueError('Changed native archive.')
    shape,pml,axes=grid(m)
    left=max(pml,int(np.argmin(abs(axes[0]-m['beam_x']+2*m['waist']))))
    right=min(shape[0]-pml,int(np.argmin(abs(axes[0]-m['beam_x']-2*m['waist'])))+1)
    e,h=sampled_gaussian_fields(axes[0][left:right],axes[1][pml:-pml],m['beam_x'],m['waist'],m['angle_deg'])
    with np.load(reference/'arrays.npz',allow_pickle=False) as a,np.load(device/'arrays.npz',allow_pickle=False) as b:
        before={k:field_difference(a[k],b[k]) for k in ('source_E','source_H')}
        fixed={stage:{k:field_difference(item[k],value) for k,value in [('source_E',e),('source_H',h)]}
               for stage,item in [('reference',a),('device',b)]}
        same_time=all(np.array_equal(a[k],b[k]) for k in ('source_time_offset_E','source_time_offset_H'))
        direct=-normal_flux(a['upper'],1/m['resolution'])
        physical=incident_beam_check(a['upper'],1/m['resolution'],a['wavelength_um'],direct)
        result={'status':'rejected_native_pair','resolution':m['resolution'],'settings':m,
                'reason':'The strict source check detected independent float32 normalization differences. This pair did not qualify. The replacement explicitly constructs the unit-energy sampled sheet in float64 and then casts E/H to the FDTD dtype.',
                'raw_sha256':{'reference':rr['arrays_sha256'],'device':dr['arrays_sha256']},
                'source_eh_sha256':{'reference':rr['source_eh_sha256'],'device':dr['source_eh_sha256'],'prescribed':digest(np.stack((e,h)))},
                'original_source_differences':before,'prescribed_vs_original':fixed,
                'matched_time_offsets':same_time,'time_offsets_sha256':rr['source_time_offsets_sha256'],
                'matched_coordinates':all(np.array_equal(a[k],b[k]) for k in ('grid_x_um','grid_y_um','grid_z_um','upper_x_um','upper_y_um')),
                'matched_actual_gpu':rr['cuda_identity']==dr['cuda_identity'],'actual_device':rr['actual_device'],'device_name':rr['device_name'],
                'source_sheet_x_angle_deg':{'reference':angle(a['source_E'],a['source_H']),'device':angle(b['source_E'],b['source_H']),'prescribed':angle(e,h)},
                'original_reference_propagation':physical,'independent_coarse_reference':coarse_reference_control()}
    if corrected:
        new=json.loads((corrected/'receipt.json').read_text())
        if new['arrays_sha256']!=file_hash(corrected/'arrays.npz'):raise ValueError('Corrected source archive changed.')
        with np.load(corrected/'arrays.npz',allow_pickle=False) as sample:
            np.testing.assert_array_equal(e,sample['source_E']);np.testing.assert_array_equal(h,sample['source_H'])
            net=-normal_flux(sample['upper'],1/m['resolution'])
            actual=incident_beam_check(sample['upper'],1/m['resolution'],sample['wavelength_um'],net)
        result.update(corrected_reference_sha256=new['arrays_sha256'],corrected_reference_propagation=actual,
                      original_and_corrected_time_offsets_match=new['source_time_offsets_sha256']==rr['source_time_offsets_sha256'],
                      corrected_minus_original_x_angle_deg=(np.array(actual['mean_x_angle_deg'])-physical['mean_x_angle_deg']).tolist())
    return result


def source_profile_archive(reference,device,output):
    """Preserve the actual rejected source samples without presenting device fields as qualified."""
    data={}
    for prefix,path in [('reference',reference),('device',device)]:
        with np.load(path/'arrays.npz',allow_pickle=False) as native:
            for key in ('source_E','source_H','source_time_offset_E','source_time_offset_H'):
                if key in native:data[prefix+'_'+key]=native[key]
            if prefix=='reference':
                data['wavelength_um']=native['wavelength_um']
                data['upper_reference_1550']=native['upper'][2]
                data['upper_x_um']=native['upper_x_um'];data['upper_y_um']=native['upper_y_um']
    np.savez_compressed(output,**data)
    return {'arrays_file':output.name,'arrays_sha256':file_hash(output)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--device',type=Path,required=True)
    p.add_argument('--corrected-reference',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--save-original-profiles',action='store_true')
    args=p.parse_args()
    report=audit(args.reference,args.device,args.corrected_reference)
    if args.save_original_profiles:
        dst=args.output.with_name(args.output.stem+'-arrays.npz')
        report.update(source_profile_archive(args.reference,args.device,dst))
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
