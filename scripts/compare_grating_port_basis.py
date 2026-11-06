"""Compare a local grating's native fields with the two cached modes in the published Ansys project."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import zipfile

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from backend.adapters.fdtdx_ports3d_adapter import modal_amplitudes
from scripts.reference_finite_grating import LUM

BASE=Path(__file__).resolve().parents[1]/'docs/verification'


def tag_offsets(data, name):
    label=name.encode()+b'\0'
    prefix=struct.pack('<I',len(label))+label
    return [match.end() for match in re.finditer(re.escape(prefix),data)]


def read_matrix(data, name, port, expected_count=2):
    offsets=tag_offsets(data,name)
    if len(offsets)!=expected_count:
        raise ValueError(f'Expected {expected_count} records for {name}.')
    start=offsets[port]
    kind,label_size=struct.unpack_from('<II',data,start)
    if kind!=3 or label_size>16:
        raise ValueError(f'Unexpected matrix format for {name}.')
    at=start+8+label_size
    count,ndim=struct.unpack_from('<II',data,at)
    if not 1<=ndim<=4:
        raise ValueError(f'Invalid rank of {name}.')
    dims=struct.unpack_from('<'+'I'*ndim,data,at+8)
    is_real,unit=struct.unpack_from('<II',data,at+8+4*ndim)
    if count!=np.prod(dims) or unit!=1 or is_real not in (0,1):
        raise ValueError(f'Invalid shape of {name}.')
    values_at=at+16+4*ndim
    values=np.frombuffer(data,'<f8',count=count,offset=values_at)
    if not is_real:
        values=values+1j*np.frombuffer(data,'<f8',count=count,offset=values_at+8*count)
    return values.reshape(dims,order='F').squeeze()


def project_basis(path):
    data=path.read_bytes()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            data=archive.read('grating_coupler_3D.fsp')
    reference=json.loads((BASE/'v012-grating-commercial-references.json').read_text())['lumerical']
    if hashlib.sha256(data).hexdigest()!=reference['project_sha256']:
        raise ValueError('Supply the original public 3D example captured by the reference manifest.')
    markers=tag_offsets(data,'neff')
    frequency=[]
    for port,stop in enumerate(markers,1):
        tag=tag_offsets(data[:stop],'modeFrequency')[-1]
        if struct.unpack_from('<II',data,tag)!=(1,1) or f'::port {port} Mode Expansion Monitor'.encode() not in data[tag:stop]:
            raise ValueError('Unexpected mode frequency or port order.')
        frequency.append(struct.unpack_from('<d',data,tag+8)[0])
    indices=[complex(read_matrix(data,'neff',p)) for p in (0,1)]
    if frequency[0]!=frequency[1] or any(abs(n.imag)>1e-8 for n in indices):
        raise ValueError('Expected lossless reference port modes at the same frequency.')
    y,z=(read_matrix(data,axis+'Matrix',1)*1e6 for axis in ('y','z'))
    if np.any(np.diff(y)<=0) or np.any(np.diff(z)<=0):
        raise ValueError('Port grid must be strictly increasing.')
    fields=np.array([[read_matrix(data,'freqEHn'+axis+str(field),1) for axis in 'xyz'] for field in (0,1)])
    if fields.shape!=(2,3,len(y),len(z)):
        raise ValueError('Filed E/H modes do not match the strip port coordinates.')
    # Lumerical stores H in A/m; the local update uses vacuum-impedance-scaled H.
    fields[1]*=376.730313668
    if np.real(fields[0,1]*fields[1,2].conj()-fields[0,2]*fields[1,1].conj()).sum()<=0:
        raise ValueError('The filed basis does not carry power along positive x.')
    valid_flags=[]
    for marker in markers:
        at=tag_offsets(data[:marker],'modeCurrentlyValid')[-1]
        valid_flags.append(bool(struct.unpack_from('<I',data,at+8)[0]))
    if not valid_flags[1]:
        raise ValueError('The published project flags the output basis as unverified.')
    summary={'project_sha256':reference['project_sha256'],'frequency_hz':frequency[0],
             'wavelength_um':299792458e6/frequency[0],'fiber_neff':indices[0].real,'guide_neff':indices[1].real,
             'stored_mode_valid_flags':{'fiber':valid_flags[0],'guide':valid_flags[1]},
             'guide_plane_x_um':float(read_matrix(data,'xMatrix',1)*1e6),'guide_samples_yz':[len(y),len(z)],
             'core_interval_step_nm':{'y':1000*float(np.median(np.diff(y)[abs(y[:-1])<.25])),
                                      'z':1000*float(np.median(np.diff(z)[(z[:-1]>=0)&(z[:-1]<.22)]))}}
    return summary,(y,z),fields


def basis_check(receipt, vendor, axes, fields):
    row=json.loads(receipt.read_text())
    if not row['passed'] or row['actual_device']!='cuda:0' or row['settings']['geometry']!='lumerical':
        raise ValueError('Input must be a qualified native Lumerical grating.')
    arrays=receipt.parent/row['arrays_file']
    if hashlib.sha256(arrays.read_bytes()).hexdigest()!=row['arrays_sha256']:
        raise ValueError('Changed native array archive.')
    mesh=row['settings']['resolution']
    dx=1/mesh
    with np.load(arrays,allow_pickle=False) as native:
        wavelengths=native['dev_wavelength_um']
        band=int(np.flatnonzero(wavelengths==1.55)[0])
        local=native['dev_mode_e'][band,1]
        local_h=native['dev_mode_h'][band]
        phasors=native['dev_output'][band:band+1]
        py,pz=local.shape
        y,z=native['dev_grid_y_um'],native['dev_grid_z_um']
        sy=(len(y)-py)//2
        sz=int(np.argmin(abs(z-(.05-pz*dx/2+dx/2))))
        # FDTDX 0.6.2 exact_interpolation centers E/H on E_z at (i,j,k+1/2).
        y,z=y[sy:sy+py]-.5*dx,z[sz:sz+pz]
        matched=np.interp(vendor['wavelength_um'],wavelengths,native['dev_mode_neff'].real)
        interpolator=RegularGridInterpolator(axes,fields.transpose(2,3,0,1),bounds_error=False,fill_value=0.)
        results=[]
        for y_shift in (-.5,0.,.5):
            for z_shift in (-.5,0.,.5):
                yy,zz=np.meshgrid(y+dx*y_shift,z+dx*z_shift,indexing='ij')
                modal=interpolator((yy,zz)).transpose(2,3,0,1)
                e,h=modal
                shape_overlap=abs(np.vdot(local,e[1]))**2/(np.vdot(local,local).real*np.vdot(e[1],e[1]).real)
                a=modal_amplitudes(phasors,e[None],h[None],dx**2)[0][0,1]
                work=abs(a)**2/native['incident_power'][band]
                results.append({'y_offset_cells':y_shift,'z_offset_cells':z_shift,
                                'ey_shape_overlap':float(shape_overlap),'reprojected_power_fraction':float(work)})
        native_amps=modal_amplitudes(phasors,native['dev_mode_e'][band:band+1],local_h[None],dx**2)[0][0,1]
        own=float(abs(native_amps)**2/native['incident_power'][band])
        np.testing.assert_allclose(own,row['result']['center_beam_to_guide'],rtol=2e-7)
    return {'receipt':receipt.name,'received_arrays_sha256':row['arrays_sha256'],'resolution':mesh,
            'material_model':row['settings'].get('grating_material_model','volume-average'),
            'guide_neff_interpolated_to_filed_frequency':float(matched),'guide_neff_minus_published':float(matched-vendor['guide_neff']),
            'local_detector_coordinate':'FDTDX E_z co-location (i,j,k+1/2)',
            'local_detector_x_um':row['stages']['device']['output_plane_x_um']-.5*dx,
            'native_power_fraction_at_1550':own,'registered_grid_diagnostic':results[4],
            'half_cell_axis_uncertainty_power_fraction':[min(x['reprojected_power_fraction'] for x in results),
                                                        max(x['reprojected_power_fraction'] for x in results)]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--receipts',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    v,axes,fields=project_basis(args.project)
    summary={'source':LUM,'published_ports':v,'comparisons':[basis_check(p,v,axes,fields) for p in args.receipts],
             'scope':'A diagnostic reprojects the unchanged 1550 nm local output phasors on the filed lossless strip E/H basis from the nearby stored frequency. It retains the native incident-power normalization and uses bilinear interpolation and vacuum-impedance-scaled H. The offset range covers ±half a native cell on both transverse axes; it is a registration sensitivity, not a vendor rerun or an error bar. No commercial field arrays are distributed.'}
    args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
