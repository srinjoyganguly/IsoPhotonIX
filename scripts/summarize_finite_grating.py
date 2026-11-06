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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.directory),indent=2,allow_nan=False)+'\n')
