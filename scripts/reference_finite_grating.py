"""Read wavelength samples from the two vendors' published 3D grating examples."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zipfile

import numpy as np

WAVELENGTHS = np.array([1.50, 1.525, 1.55, 1.575, 1.60])
TIDY = 'https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/'
BOOK = 'https://raw.githubusercontent.com/flexcompute/tidy3d-notebooks/18fd47f8bd87ad89582b507c20bf4997f79d55c0/GratingCoupler.ipynb'
LUM = 'https://optics.ansys.com/hc/en-us/articles/360042305334-Grating-coupler'
LUM_EXAMPLE = 'https://api.lumerical.com/app-gallery/fetch-download-web?id=c118296f-grating-coupler'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digitize(notebook):
    from PIL import Image
    data=notebook.read_bytes(); book=json.loads(data)
    matches=[cell for cell in book['cells'] if cell['cell_type']=='code' and 'ce_3d = np.abs(' in ''.join(cell['source'])]
    if len(matches)!=1: raise ValueError('Cannot locate the 3D plot in the supplied notebook.')
    image=next(base64.b64decode(o['data']['image/png']) for o in matches[0]['outputs'] if 'image/png' in o.get('data',{}))
    pixels=np.asarray(Image.open(io.BytesIO(image)).convert('RGB'))
    if sha(image)!='2eb81a1f20080105be90bbca17fd2ecfe5295327f1c148e40da86a4c88f35a90':
        raise ValueError('Changed plot: inspect and recalibrate the pixel coordinates.')
    red=(pixels[:,:,0]>210)&(pixels[:,:,1]<100)&(pixels[:,:,2]<100)
    # Major labeled x tick centers and plot axes calibrated on the original chart.
    x0,x1,y0,y1=91.,542.,15.,385.
    columns=np.arange(int(x0),int(x1)+1)
    pixel_rows=np.array([np.median(np.flatnonzero(red[:,x])) for x in columns])
    if not np.isfinite(pixel_rows).all(): raise ValueError('Incomplete color trace.')
    trace_db=-12*(pixel_rows-y0)/(y1-y0)
    trace_um=1.50+(columns-x0)*.10/(x1-x0)
    sample_db=np.interp(WAVELENGTHS,trace_um,trace_db)
    best=int(np.argmax(trace_db))
    return dict(vendor='Tidy3D',source=TIDY,notebook=BOOK,notebook_sha256=sha(data),figure_sha256=sha(image),
                observable='Gaussian incident beam to backward output TE0 after the 50 µm taper',
                calibration={'wavelength_um':[1.5,1.6],'x_px':[x0,x1],'decibels':[0,-12],'y_px':[y0,y1],
                             'conservative_readout_uncertainty_db':.08},
                wavelength_um=WAVELENGTHS.tolist(),coupling_db=sample_db.tolist(),coupling=(10**(sample_db/10)).tolist(),
                digitized_peak_db=float(trace_db[best]),digitized_peak_wavelength_um=float(trace_um[best]))


def group_properties(data, first, count):
    # The filed 3D model's user-properties block stores length-prefixed names,
    # property types and length-prefixed printable values. Validate each field.
    i=data.rfind(first.encode()+b'\0')-5
    result={}
    for _ in range(count):
        if i<0 or data[i]!=2: raise ValueError('Unknown project property format.')
        n=struct.unpack_from('<I',data,i+1)[0]
        name=data[i+5:i+5+n-1].decode()
        p=i+5+n+5
        if data[p]!=2: raise ValueError('Unknown project property type.')
        length=struct.unpack_from('<I',data,p+1)[0]
        value=data[p+5:p+5+length-1].decode()
        try: result[name]=float(value)
        except ValueError: result[name]=value
        i=p+5+length
    return result


def material_samples(project, label):
    name=project.index(label.encode())
    freq_start=project.rfind(b'frequency\0',0,name)+len(b'frequency\0')
    eps_start=project.index(b'permittivity\0',name)+len(b'permittivity\0')
    count=struct.unpack_from('<I',project,freq_start+8)[0]
    if count!=struct.unpack_from('<I',project,eps_start+8)[0]:
        raise ValueError('Filed permittivity and frequency arrays differ.')
    freq=np.frombuffer(project,dtype='<f8',count=count,offset=freq_start+32)
    real=np.frombuffer(project,dtype='<f8',count=count,offset=eps_start+32)
    imag=np.frombuffer(project,dtype='<f8',count=count,offset=eps_start+32+count*8)
    if np.any(np.diff(freq)<0) or freq[-1]<=freq[0]: raise ValueError('Material frequencies must ascend.')
    f=299792458e6/WAVELENGTHS
    return {'real':np.interp(f,freq,real).tolist(),'imaginary':np.interp(f,freq,imag).tolist()}


def project_bounds(project, name):
    marker=project.rindex(name.encode()+b'\0')
    vals=np.frombuffer(project,dtype='<f8',count=6,offset=marker-68)*1e6
    return {'x':vals[[0,2]].tolist(),'y':vals[[1,3]].tolist(),'z':vals[4:].tolist()}


def port_centers(project):
    result={}
    for axis in 'XYZ':
        token=('extraState::globalCenter'+axis).encode()+b'\0'
        values=[struct.unpack_from('<d',project,m.end()+8)[0]*1e6 for m in re.finditer(token,project)]
        if len(values)!=2: raise ValueError('Expected exactly two filed physical port centers.')
        result[axis.lower()]=values
    return result


def lumerical(archive, wavelengths=WAVELENGTHS):
    data=archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        s_data=z.read('gc_strip_te_c_S_params.dat')
        project=z.read('grating_coupler_3D.fsp')
    rows={}
    for m in re.finditer(r'\("port (\d)".*?"port (\d)".*?\)\s*\(50, 3\)\s*([\d.e+\-\s]*)',s_data.decode()):
        val=np.fromstring(m[3],sep=' ').reshape(-1,3)
        if val.shape!=(50,3) or np.any(np.diff(val[:,0])<=0):
            raise ValueError('Invalid frequency ordering in the published S-parameter file.')
        # File stores scattering magnitude and phase. Interpolate measured power on wavelength.
        wv=299792458e6/val[:,0]
        rows[f'S{m[1]}{m[2]}']=dict(power=np.interp(wavelengths,wv[::-1],val[::-1,1]**2).tolist(),
                                    peak_power=float(np.max(val[:,1]**2)),
                                    peak_wavelength_um=float(wv[np.argmax(val[:,1]**2)]))
    if set(rows)!=set(('S11','S12','S21','S22')): raise ValueError('Missing published matrix block.')
    return dict(vendor='Lumerical',source=LUM,download=LUM_EXAMPLE,archive_sha256=sha(data),s_file_sha256=sha(s_data),
                project_sha256=sha(project),wavelength_um=wavelengths.tolist(),samples=rows,
                fiber_group=group_properties(project,'core diameter',6),
                grating_group=group_properties(project,'target length',13),
                port_centers_um=port_centers(project),
                material_epsilon={name:material_samples(project,name) for name in ('Si (Silicon) - Palik','SiO2 (Glass) - Palik')},
                layers_xyz_bounds_um={name:project_bounds(project,name) for name in ('SiO2 TOX','SiO2 BOX','Si substrate')},
                scope='Filed curved 38-period SOI sector, unlike the Tidy3D uniform linear grating. The S21 magnitude in this file is squared to obtain the transfer power between the tilted fiber port and the integrated guide.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',type=Path,required=True)
    parser.add_argument('--ansys-archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dense-output',type=Path,help='Also write the derived 21-point, hash-checked Lumerical S21 reference.')
    args=parser.parse_args()
    receipt=dict(tidy3d=digitize(args.notebook),lumerical=lumerical(args.ansys_archive))
    args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    if args.dense_output:
        from scripts.calibrate_finite_grating import monitoring_wavelengths
        root=Path(__file__).resolve().parents[1]
        nominal=json.loads((root/'docs/verification/v012-grating-commercial-references.json').read_text())['lumerical']
        wavelengths=monitoring_wavelengths({'grating_spectrum':'dense'})
        lum=lumerical(args.ansys_archive,wavelengths)
        if any(lum[k]!=nominal[k] for k in ('archive_sha256','s_file_sha256','project_sha256')):
            raise ValueError('The input differs from the pinned commercial example.')
        if lum['samples']['S21']['power'][::5]!=nominal['samples']['S21']['power']:
            raise ValueError('Dense commercial powers do not reproduce the five archived anchor values.')
        output={k:lum[k] for k in ('source','download','archive_sha256','s_file_sha256','wavelength_um')}
        output.update(input_file='gc_strip_te_c_S_params.dat',input_samples=50,
                      published_s21_power=lum['samples']['S21']['power'],
                      peak_wavelength_um=lum['samples']['S21']['peak_wavelength_um'],
                      peak_power=lum['samples']['S21']['peak_power'],
                      method='Linear interpolation in wavelength of squared S21 magnitude from the 50-row ascending frequency table.')
        args.dense_output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
