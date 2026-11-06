"""Compare only measured, like-for-like finite grating sensitivity controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    report=json.loads(path.read_text())
    if not report['passed'] or not report['verification']['incident_and_mode_power_reconstructed']:
        raise ValueError(f'Unqualified measurement: {path.name}')
    with (path.parent/report['arrays_file']).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=report['arrays_sha256']:
            raise ValueError(f'Changed numerical source: {path.name}')
    return report


def summarize(directory):
    def load(geometry, mesh):
        file=directory/f'v012-grating3d-{geometry}-r{mesh}.json'
        return file.name,read(file)
    def difference(a,b):
        return (100*(np.asarray(a)-np.asarray(b))).tolist()
    nominal_name,n=load('compact',12)
    narrow_name,w=load('compact-narrow',12)
    fine_name,m=load('compact',16)
    rad_name,g=load('compact-radiation',12)
    if n['stages']['reference']['source_eh_sha256']!=w['stages']['reference']['source_eh_sha256']:
        raise ValueError('Width control has a different source field.')
    if n['stages']['reference']['shape']!=w['stages']['reference']['shape']:
        raise ValueError('Width control has a different numerical aperture.')
    wave=n['result']['wavelength_um']
    for item in [w,m,g]:
        if item['result']['wavelength_um']!=wave:raise ValueError('Wavelength samples differ.')
    nominal=n['result']['beam_to_guide']
    width=difference(w['result']['beam_to_guide'],nominal)
    mesh=difference(m['result']['beam_to_guide'],nominal)
    reciprocal=difference(g['result']['guide_to_beam'],nominal)
    return {'wavelength_um':wave,'reference_geometry':nominal_name,
            'width':{'comparison':narrow_name,'smaller_minus_nominal_percentage_points':width,
                     'source_eh_sha256':n['stages']['reference']['source_eh_sha256'],
                     'domain_shape':n['stages']['reference']['shape']},
            'mesh':{'comparison':fine_name,'fine_minus_nominal_percentage_points':mesh,
                    'max_absolute_difference_percentage_points':max(abs(z) for z in mesh)},
            'reciprocity':{'comparison':rad_name,'emission_minus_reception_percentage_points':reciprocal,
                           'max_absolute_difference_percentage_points':max(abs(z) for z in reciprocal),
                           'nominal_reception':nominal[2],'nominal_emission':g['result']['guide_to_beam'][2]},
            'scope':'These quantify changes on one compact device. Numerical settling and mesh convergence are separate.'}

def vendor_mesh_changes(directory):
    result={}
    for geometry in ('tidy3d','lumerical'):
        a=read(directory/f'v012-grating3d-{geometry}-r12.json')
        b=read(directory/f'v012-grating3d-{geometry}-r14.json')
        if {k:v for k,v in a['settings'].items() if k!='resolution'}!={k:v for k,v in b['settings'].items() if k!='resolution'}:
            raise ValueError(f'{geometry} geometry/source differs beyond the resolution setting.')
        if a['device_name']!=b['device_name'] or a['actual_device']!=b['actual_device']:
            raise ValueError(f'{geometry} targets differ.')
        x=np.array(a['result']['beam_to_guide'])
        y=np.array(b['result']['beam_to_guide'])
        va=next(row for row in a['comparisons'] if row['geometry_matched'])
        vb=next(row for row in b['comparisons'] if row['geometry_matched'])
        result[geometry]={'wavelength_um':a['result']['wavelength_um'],'coarse_resolution':12,'fine_resolution':14,
                         'coarse_coupling':x.tolist(),'fine_coupling':y.tolist(),
                         'fine_minus_coarse_percentage_points':(100*(y-x)).tolist(),
                         'maximum_power_fraction_change':float(np.max(abs(y-x))),
                         'coarse_vendor_mae_percentage_points':va['mean_absolute_difference_percentage_points'],
                         'fine_vendor_mae_percentage_points':vb['mean_absolute_difference_percentage_points'],
                         'fine_vendor_differences_percentage_points':vb['local_minus_published_percentage_points'],
                         'upper_collection_plane_z_um':[a['stages']['device']['sample_plane_z_um'],b['stages']['device']['sample_plane_z_um']],
                         'nominal_source_z_um':a['settings']['source_z'],
                         'guide_plane_x_um':[a['stages']['device']['output_plane_x_um'],b['stages']['device']['output_plane_x_um']],
                         'pml_width_um':[a['stages']['device']['pml_cells']/12,b['stages']['device']['pml_cells']/14]}
    return {'mesh_pairs':result,'scope':'Two independently acquired uniform grids establish the observed sensitivity at five wavelengths; absolute continuum convergence requires further evidence.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--vendor-mesh',action='store_true')
    args=parser.parse_args()
    result=vendor_mesh_changes(args.directory) if args.vendor_mesh else summarize(args.directory)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
