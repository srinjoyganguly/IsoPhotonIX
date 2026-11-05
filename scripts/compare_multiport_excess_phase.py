"""Measure and compare the excess device phase relative to a straight through reference.

These auxiliary phases retain the physical-plane scattering matrix unchanged.
Use one independently measured straight transfer per target with the same input
waveguide, x planes, materials, aperture and pulse as the device measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from calibrate_multiport import build_simulation, project, run_until_settled
from backend.calibrated_geometry import multiport_model


def pairs(v):
    return np.stack([v.real,v.imag],axis=-1).tolist()


def unpack(v):
    array=np.asarray(v)
    return array[...,0]+1j*array[...,1]


def straight(case,resolution,out):
    import meep as mp
    mp.verbosity(0)
    model=multiport_model(case)
    port=model['ports'][0]
    sim,source_monitor=build_simulation(mp,model,0,resolution,True)
    near=port['x'];far=-near
    frequencies=np.array(model['frequency_per_um'])
    plane=mp.Vector3(0,*model['aperture'])
    monitors={'near':source_monitor[0],'far':sim.add_flux(float(frequencies.mean()),float(np.ptp(frequencies)),3,
              mp.FluxRegion(center=mp.Vector3(far,port['y']),size=plane,direction=mp.X),decimation_factor=1)}
    points={key:sim.add_dft_fields([mp.Ey],float(frequencies.mean()),float(np.ptp(frequencies)),3,
                   center=mp.Vector3(x,port['y']),size=mp.Vector3(),decimation_factor=1)
            for key,x in [('near',near),('far',far)]}
    try:
        settled=run_until_settled(mp,sim,monitors,model,80.)
        if not settled['settled']:
            raise RuntimeError('Straight reference did not meet its native DFT and energy windows.')
        amps={}; eys={}; indices={}
        for tag,x in [('near',near),('far',far)]:
            amps[tag],indices[tag],_,_=project(mp,sim,monitors[tag],dict(port,x=x),model,1,resolution)
            eys[tag]=np.array([np.asarray(sim.get_dft_array(points[tag],mp.Ey,i)).item() for i in range(3)])
        ratio=amps['far']/amps['near']
        direct=eys['far']/eys['near']
        residual=np.angle(ratio*direct.conj())
        if np.min(abs(amps['near']))<=1e-8 or max(abs(residual))>.025:
            raise RuntimeError('Straight modal phase is inconsistent with direct centre Ey.')
        result={'case':case,'solver':'Meep','version':mp.__version__,'device':'cpu','resolution':resolution,
            'incident_port':port['name'],'x_planes_um':[near,far],'source_x_um':port['source_x'],'guide_y_um':port['y'],
            'frequency_per_um':frequencies.tolist(),'complex_far_over_near':pairs(ratio),
            'through_minus_point_phase_rad':residual.tolist(),'transmission_power':(abs(ratio)**2).tolist(),
            'amplitudes':{k:pairs(v) for k,v in amps.items()},'point_Ey':{k:pairs(v) for k,v in eys.items()},
            'mode_neff':{k:v.tolist() for k,v in indices.items()},'settled':settled}
        with out.open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    finally:
        sim.reset_meep()


def excess_value(s, reference):
    if np.min(abs(reference))<1e-6:
        raise ValueError('Reference transmission is too small to define an excess phase.')
    return np.angle(s*reference.conj())


def make_comparison(case, meep_dir, fdtdx_dir, meep_reference):
    m=json.loads((meep_dir/'report.json').read_text())
    d=json.loads((fdtdx_dir/'report.json').read_text())
    straight=json.loads(meep_reference.read_text())
    if 'complex_far_over_near' not in straight:
        # The independently verified four-plane Meep dispersion audit contains
        # centre-guide complex amplitudes at both required physical planes.
        if (straight.get('solver')!='Meep' or not straight['setup_settled']['settled'] or
            straight['point_grid_coordinates_um']['near'][1]!=[0.0] or
            straight['point_grid_coordinates_um']['far'][1]!=[0.0]):
            raise ValueError('Invalid centred four-plane reference.')
        straight={'case':'splitter','guide_y_um':0.,'version':straight['versions']['meep'],
           'resolution':straight['resolution'],'frequency_per_um':straight['frequency_per_um'],
           'x_planes_um':[straight['monitor_x_um'][k] for k in ('near','far')],
           'complex_far_over_near':pairs(unpack(straight['amplitudes']['far'])/unpack(straight['amplitudes']['near']))}
    model=multiport_model(case)
    if m['case']!=case or d['case']!=case or straight['case']!=case or m['ports']!=d['port_order']:
        raise ValueError('Case, straight incident lane and measured port names must agree.')
    if d['resolution']!=m['resolution'] or d['physical_model']!=m['model']:
        raise ValueError('Measured geometry, mesh, material settings or monitors differ.')
    if not all(m['valid_bins']) or not np.all(d['valid_column']):
        raise ValueError('Only qualified bins from settled studies can be compared.')
    if m.get('phase_convention_version')!=2 or m['resolution']!=straight['resolution'] or m['version']!=straight['version']:
        raise ValueError('Meep mesh/version/phase basis does not match its straight reference.')
    if straight['guide_y_um']!=model['ports'][0]['y'] or straight['x_planes_um']!=[-4.75,4.75]:
        raise ValueError('The straight guide must follow the incident lane and physical planes.')
    fd_archive=fdtdx_dir/'arrays.npz'
    if not fd_archive.exists():
        fd_archive=fdtdx_dir/'sparameters.npz'  # earlier native FDTDX receipt
    with np.load(meep_dir/'arrays.npz',allow_pickle=False) as am,np.load(fd_archive,allow_pickle=False) as af:
        if not np.array_equal(am['frequency_per_um'],af['frequency_per_um']) or not np.array_equal(am['frequency_per_um'],straight['frequency_per_um']):
            raise ValueError('Measured frequency bins must agree exactly.')
        t_meep=unpack(straight['complex_far_over_near']);t_fd=af['reference_through'][:,0]
        if t_fd.shape!=t_meep.shape: raise ValueError('Missing matched FDTDX straight transfer.')
        entries=[]
        for i,p in enumerate(model['ports']):
            if p['inward_sign']!= -1: continue
            sm=am['s'][:,i,0]; sf=af['s'][:,i,0]
            if min(np.min(abs(sm)),np.min(abs(sf)))<.01:
                raise ValueError('A transmitted amplitude is too weak for this diagnostic.')
            qm=excess_value(sm,t_meep); qf=excess_value(sf,t_fd)
            entries.append({'incident':model['ports'][0]['name'],'outgoing':p['name'],'meep_S':pairs(sm),'fdtdx_S':pairs(sf),
                'meep_power':(abs(sm)**2).tolist(),'fdtdx_power':(abs(sf)**2).tolist(),
                'raw_fdtdx_minus_meep_phase_rad':np.angle(sf*sm.conj()).tolist(),
                'meep_device_excess_rad':qm.tolist(),'fdtdx_device_excess_rad':qf.tolist(),
                'fdtdx_minus_meep_excess_phase_rad':np.angle(np.exp(1j*(qf-qm))).tolist()})
        return {'case':case,'resolution':m['resolution'],'frequency_per_um':am['frequency_per_um'].tolist(),
          'x_planes_um':straight['x_planes_um'],'straight_guide_y_um':straight['guide_y_um'],
          'meep_reference_far_over_near':pairs(t_meep),'fdtdx_reference_far_over_near':pairs(t_fd),
          'straight_fdtdx_minus_meep_phase_rad':np.angle(t_fd*t_meep.conj()).tolist(),
          'meep_device_archive_sha256':hashlib.sha256((meep_dir/'arrays.npz').read_bytes()).hexdigest(),
          'fdtdx_device_archive_sha256':hashlib.sha256(fd_archive.read_bytes()).hexdigest(),
          'meep_straight_receipt_sha256':hashlib.sha256(meep_reference.read_bytes()).hexdigest(),
          'description':'Excess = principal arg(S_device / T_straight) at the named physical x planes. Straight reference uses an uninterrupted guide in the incident lane. Off-lane device outputs therefore compare to a straight path of the same longitudinal length. This diagnostic changes no physical S coefficient and measures neither loss nor an independently optimized phase correction.',
          'actual_execution':['Meep CPU',d['actual_device']],
          'fdtdx_material_model':d.get('material_model','legacy-x'),'entries':entries}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    st=sub.add_parser('straight');st.add_argument('--resolution',type=int,choices=(16,20,24),default=16)
    cm=sub.add_parser('compare')
    cm.add_argument('--meep',type=Path,required=True);cm.add_argument('--fdtdx',type=Path,required=True)
    cm.add_argument('--straight-reference',type=Path,required=True)
    for item in (st,cm):
        item.add_argument('--case',choices=('coupler','splitter'),required=True)
        item.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Choose a fresh output file.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.action=='straight':
        straight(a.case,a.resolution,a.output)
    else:
        result=make_comparison(a.case,a.meep,a.fdtdx,a.straight_reference)
        with a.output.open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
