"""Read measured ring geometry, lead modes and material data from the pinned public Ansys files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np

from scripts.compare_loaded_ring3d import ANSYS_SHA,lumerical_spectrum
from scripts.reference_finite_grating import group_properties,material_samples,WAVELENGTHS

FSP_SHA = '6132a406d16606f4a20122eb362e7a5895a3a9ad91492e035789b8d1698634b4'
LDF_SHA = 'b366f4234fadd989534fe8fad48b6ecc80a34c9269aa0eaeb99aa103d74f3649'
BASE = 'https://optics.ansys.com/hc/en-us/article_attachments/'


def verified(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest:
        raise ValueError(f'Unexpected revision of {path.name}.')
    return data


def scalar(data, name, kind, start=0):
    token=name.encode()+b'\0'
    pos=data.index(token,start)+len(token)
    if struct.unpack_from('<II',data,pos)!=(kind,1):
        raise ValueError(f'Unexpected scalar format for {name}.')
    return struct.unpack_from('<d' if kind==1 else '<i',data,pos+8)[0]


def array(data, name, start=0):
    token=name.encode()+b'\0'
    pos=data.index(token,start)+len(token)
    code, flags, count, nd, n, unit, real, unit2 = struct.unpack_from('<8I',data,pos)
    if (code,flags,nd,n,unit,unit2)!=(3,0,2,count,1,1) or real not in (0,1) or not count:
        raise ValueError(f'Unexpected array format for {name}.')
    size=count*(1 if real else 2)
    values=np.frombuffer(data,dtype='<f8',count=size,offset=pos+32)
    if not np.isfinite(values).all():
        raise ValueError(f'Invalid values for {name}.')
    return values[:count] if real else values[:count]+1j*values[count:]


def ldf_array(data,name):
    token=name.encode()+b'\0'
    pos=data.index(token)+len(token)
    if struct.unpack_from('<6I',data,pos)!=(500,2,500,1,1,1):
        raise ValueError(f'Unexpected LDF dimensions for {name}.')
    return np.frombuffer(data,dtype='<f8',count=500,offset=pos+24)


def project_receipt(project, flux_file, scattering_file):
    b=verified(project,FSP_SHA)
    flux=verified(flux_file,LDF_SHA)
    sw,_,scattering,_=lumerical_spectrum(scattering_file)
    if not flux.startswith(b'Lumerical data file 2'):
        raise ValueError('Wrong total-flux attachment type.')
    fw=ldf_array(flux,'lambda_3DFDTD')*1e6
    power=ldf_array(flux,'Tdrop_3DFDTD')
    if not np.all(np.diff(fw)<0):
        raise ValueError('Drop-file wavelengths must descend.')
    fw,power=fw[::-1],power[::-1]
    error=abs(power-np.interp(fw,sw,scattering))
    props=group_properties(b,'Lc',8)
    if set(props)!={'Lc','gap','radius','base angle','base width','base height','material','x span'}:
        raise ValueError('Unexpected ring object parameters.')
    geometry={key+'_um':props[key]*1e6 for key in ('radius','Lc','gap','base width','base height','x span')}
    geometry['base_angle_deg']=props['base angle']
    # The project names the material by UUID in its object properties. Bind the object
    # to the same stored material database entry whose optical table is read below.
    si_position=b.index(b'Si (Silicon) - Palik\0')
    if props['material'].encode() not in b[si_position-2048:si_position]:
        raise ValueError('The ring object does not use the stored Palik silicon material.')
    m=float(re.search(rb'(?m)^m=([.\d]+);',b)[1])
    t=np.linspace(0,1,10001)
    # Bernstein form of one quadrant, rotating the equivalent FSP cubics into (1,0) -> (0,1).
    x=(1-t)**3+3*(1-t)**2*t+3*m*(1-t)*t**2
    y=3*m*(1-t)**2*t+3*(1-t)*t**2+t**3
    radius=geometry['radius_um']
    geometry.update(bezier_control_fraction=m,maximum_radial_difference_nm=float(abs(np.hypot(x,y)-1).max()*radius*1000),
                    perimeter_difference_nm=float(1000*radius*(4*np.hypot(np.diff(x),np.diff(y)).sum()-2*np.pi)))
    simulation_position=b.index(b'GUIboundary\0')
    grids={}
    for axis,location in [('x',0.),('y',3.6),('z',.09)]:
        grid=array(b,axis+'Grid',simulation_position).real*1e6
        steps=np.diff(grid)
        if not np.all(steps>0):
            raise ValueError('Non-monotonic native coordinates.')
        nearest=np.searchsorted(grid,location,side='right')-1
        grids[axis]={'coordinates':len(grid),'minimum_step_nm':float(1000*steps.min()),
                     'step_at_guide_center_nm':float(1000*steps[nearest])}
    ports=[]
    for port in range(1,5):
        monitor=b.index(f'::port {port} Mode Expansion Monitor'.encode())
        lead=b.rfind(b'freqNumberOfModes\0',0,monitor)
        freq=scalar(b,'modeFrequency',1,lead)
        neff=array(b,'neff',monitor)
        if len(neff)!=1 or abs(neff.imag[0])>1e-12:
            raise ValueError('Expected one bound real-index port mode.')
        port_start=list(re.finditer(b'extraState::globalCenterX\0',b))[port-1].start()
        ports.append({'port':port,'center_um':[1e6*scalar(b,'extraState::globalCenter'+axis,1,port_start) for axis in 'XYZ'],
                     'aperture_yz_um':[1e6*scalar(b,'GUISpan'+axis,1,port_start) for axis in 'YZ'],
                     'mode_wavelength_um':299792458e6/freq,'neff':float(neff[0].real),
                     'reported_mode_TE_fraction':float(array(b,'mdTEFrac',lead)[0])})
    mats={m:material_samples(b,m) for m in ('Si (Silicon) - Palik','SiO2 (Glass) - Palik')}
    return {'geometry':geometry,'port_modes':ports,'grid':grids,'material_wavelengths_um':WAVELENGTHS.tolist(),
            'material_tables_epsilon':mats,'source_port_numbers':[i+1 for i,m in enumerate(re.finditer(b'isSourcePort\0',b)) if scalar(b,'isSourcePort',2,m.start())],
            'full_symmetry_flag':bool(scalar(b,'fullSymmetry',2,simulation_position)),
            'simulation_time_fs':1e15*scalar(b,'MaxSimTime',1,simulation_position),
            'saved_pulse_settings':{'global_length_fs':1e15*scalar(b,'pulseLength',1,simulation_position),
                                    'local_port_length_fs':1e15*scalar(b,'local::pulseLength',1,simulation_position),
                                    'scope':'Saved configuration fields. The launched time signal and its phase are not recovered from these scalar fields.'},
            'total_flux_and_modal_drop':{'number_of_points':len(power),'range_um':[float(fw[0]),float(fw[-1])],
                'total_flux_fraction_at_1550':float(np.interp(1.55,fw,power)),
                'fundamental_S21_power_at_1550':float(np.interp(1.55,sw,scattering)),
                'mean_absolute_difference':float(error.mean()),'maximum_absolute_difference':float(error.max()),
                'definition':'The LDF stores total recorded drop flux. The TXT stores modal S magnitudes; square S21 for fundamental power.'},
            'sources':{'project':{'url':BASE+'50334806316307','sha256':FSP_SHA},
                       'total_flux':{'url':BASE+'50334805250579','sha256':LDF_SHA},
                       's_parameters':{'url':BASE+'50334805245587','sha256':ANSYS_SHA}},
            'provenance_limit':'The project contains the port modes, grid and S-parameter export script. Neither separate measurement attachment identifies the exact project revision or mesh of its originating run.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--total-flux',type=Path,required=True)
    parser.add_argument('--s-parameters',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    opts=parser.parse_args()
    result=project_receipt(opts.project,opts.total_flux,opts.s_parameters)
    opts.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
