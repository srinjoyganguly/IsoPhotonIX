"""Measure native straight-guide near/far phasors and one-cell translations of both planes.

The short spacing resolves the sign of longitudinal propagation without fitting
a phase to either device spectrum. Lengths are in µm, phase in radians.
Run under the chosen native solver's Python environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT/'backend').is_dir():
    sys.path.insert(0,str(ROOT))
from backend.calibrated_geometry import multiport_model


def evaluate(frequency, neff, near, far, near_shifted, far_shifted, points, length, dx):
    through=far/near
    shorter=far_shifted/near_shifted
    point_through=points['far']/points['near']
    point_shorter=points['far_shifted']/points['near_shifted']
    beta_mode=2*np.pi*frequency*neff
    beta_measured=np.angle(through/shorter)/(2*dx)
    phase=lambda values: np.angle(values).tolist()
    return {
        'reference_through_power':(abs(through)**2).tolist(),
        'through_phase_rad':phase(through),'shifted_through_phase_rad':phase(shorter),
        'eigenmode_beta_per_um':beta_mode.tolist(),'positive_eigenmode_beta_L_rad':(beta_mode*length).tolist(),
        'through_minus_positive_beta_L_rad':phase(through*np.exp(-1j*beta_mode*length)),
        'through_minus_negative_beta_L_rad':phase(through*np.exp(1j*beta_mode*length)),
        'short_phase_dx_near_rad':phase(near_shifted/near),'short_phase_dx_far_rad':phase(far/far_shifted),
        'modal_short_interval_beta_per_um':beta_measured.tolist(),
        'point_Ey_short_interval_beta_per_um':(np.angle(point_through/point_shorter)/(2*dx)).tolist(),
        'point_Ey_minus_modal_through_rad':phase(point_through/through),
        'measured_unwrapped_through_phase_rad':(np.angle(through)+2*np.pi*np.rint(
            (beta_measured*length-np.angle(through))/(2*np.pi))).tolist(),
        'through_minus_short_interval_beta_L_rad':phase(through*np.exp(-1j*beta_measured*length)),
    }


def meep_reference(output,resolution):
    import meep as mp
    from scripts.calibrate_multiport import build_simulation,project,run_until_settled
    mp.verbosity(0)
    model=multiport_model('splitter')
    sim,standard=build_simulation(mp,model,0,resolution,True)
    source_port=model['ports'][0]
    x=source_port['x']
    dx=1/resolution
    coords={'near':x,'near_shifted':x+dx,'far_shifted':-x-dx,'far':-x}
    frequencies=np.array(model['frequency_per_um'])
    monitors,point_monitors={},{}
    for name,location in coords.items():
        monitors[name]=standard[0] if name=='near' else sim.add_mode_monitor(
            frequencies.mean(),np.ptp(frequencies),len(frequencies),
            mp.ModeRegion(center=mp.Vector3(location,0,0),size=mp.Vector3(0,*model['aperture'])))
        point_monitors[name]=sim.add_dft_fields([mp.Ey],frequencies.mean(),np.ptp(frequencies),len(frequencies),
                                                center=mp.Vector3(location,0,0),size=mp.Vector3())
    settled=run_until_settled(mp,sim,monitors,model,80.)
    if not settled['settled']:
        raise RuntimeError('The straight Meep reference did not settle.')
    modes,points,locations={}, {}, {}
    for name,location in coords.items():
        mode,neff,_,_=project(mp,sim,monitors[name],dict(source_port,x=location),model,1,resolution)
        modes[name]=mode
        points[name]=np.array([np.asarray(sim.get_dft_array(point_monitors[name],mp.Ey,i)).item() for i in range(3)])
        coords_at_point=sim.get_array_metadata(dft_cell=point_monitors[name])[:3]
        locations[name]=[np.asarray(v).tolist() for v in coords_at_point]
    result=evaluate(frequencies,neff,**modes,points=points,length=-2*x,dx=dx)
    np.savez_compressed(output/'measured.npz',frequency_per_um=frequencies,**modes,
                        **{'Ey_'+k:v for k,v in points.items()})
    sim.reset_meep()
    return {**result,'solver':'Meep CPU','frequency_per_um':frequencies.tolist(),'resolution':resolution,
            'coordinates_um':coords,'point_grid_coordinates_um':locations,'settling':settled}


def fdtdx_reference(output,resolution,device,material_model):
    from scripts.calibrate_fdtdx_multiport import _ports_runtime,stage
    runtime=_ports_runtime(device)
    model=multiport_model('splitter')
    port=model['ports'][0];dx=1/resolution;x0=port['x']
    coordinates={'near':x0,'near_shifted':x0+dx,'far_shifted':-x0-dx,'far':-x0}
    modes,points,metadata={}, {}, {}
    for shift,suffix in ((0.,''),(dx,'_shifted')):
        port['x']=x0+shift
        archive=output/('straight'+suffix+'.npz')
        _,record=stage(runtime,model,resolution,80.,0,True,archive,{},material_model)
        metadata['straight'+suffix]=record
        if not record['settled']:
            raise RuntimeError('The straight FDTDX reference did not settle.')
        with np.load(archive,allow_pickle=False) as data:
            for idx,(position,name) in enumerate((('near','left_center'),('far','straight_through'))):
                modes[position+suffix]=data['amplitudes'][idx,:,0].copy()
                phasors=data['phasors_'+name]
                points[position+suffix]=phasors[:,1,0,phasors.shape[-2]//2,phasors.shape[-1]//2].copy()
    if metadata['straight']['source_profile_sha256']!=metadata['straight_shifted']['source_profile_sha256']:
        raise RuntimeError('The source mode differed between the two reference studies.')
    frequencies=np.array(model['frequency_per_um'])
    neff=np.array(metadata['straight']['mode_neff']['left_center'])
    result=evaluate(frequencies,neff,**modes,points=points,length=-2*x0,dx=dx)
    return {**result,'solver':'FDTDX','actual_device':str(runtime[-1]),'material_model':material_model,
            'coordinates_um':coordinates,'frequency_per_um':frequencies.tolist(),'resolution':resolution,'stages':metadata,
            'sample_convention':'Phasors are natively interpolated to Ez Yee positions; center Ey samples have z=+dx/2 and x at the port grid plane.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver',choices=('meep','fdtdx'),required=True)
    parser.add_argument('--device',choices=('cpu','gpu'),required=True)
    parser.add_argument('--resolution',type=int,choices=(12,16,20,24),default=16)
    parser.add_argument('--material-model',choices=('legacy-x','yee-aligned'),default='legacy-x')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.solver=='meep' and args.device!='cpu':
        parser.error('The native Meep phase control runs on CPU.')
    args.output.mkdir(parents=True,exist_ok=False)
    result=meep_reference(args.output,args.resolution) if args.solver=='meep' else fdtdx_reference(
        args.output,args.resolution,args.device,args.material_model)
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
