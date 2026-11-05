"""Coherent normal incidence on one homogeneous, isotropic, nonmagnetic film."""
import cmath
import csv
import io
import json
import math
from pathlib import Path
import zipfile

from .geometry_draft import evaluate_material, validate_material


def _in_range(value, low, high, label):
    if type(value) not in (float, int) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number between {low} and {high}.')
    return value


def _sample(wavelength, n, k, thickness, incident, substrate):
    index = complex(n, k)
    h = 2 * math.pi * thickness / wavelength
    phase = h * index
    propagation = cmath.exp(1j * phase)
    # Scale the field continuity equations by exp(iδ). These terms only
    # decay in a passive medium, even when transmission underflows to zero.
    scaled_cos = (1 + propagation**2) / 2
    if abs(phase) < 1e-4:
        sinc = 1 - phase**2 / 6 + phase**4 / 120
        scaled_sin = -1j * propagation * h * sinc
    else:
        scaled_sin = (1 - propagation**2) / (2 * index)
    denominator = (incident + substrate) * scaled_cos + (index**2 + incident * substrate) * scaled_sin
    r = ((incident - substrate) * scaled_cos + (incident * substrate - index**2) * scaled_sin) / denominator
    t = 2 * incident * propagation / denominator
    reflectance, transmittance = abs(r)**2, (substrate / incident) * abs(t)**2
    absorptance = 1 - reflectance - transmittance
    if absorptance < -1e-10 or (k == 0 and abs(absorptance) > 1e-10):
        raise ArithmeticError('The passive film failed its flux balance check.')
    return {'wavelength_um':wavelength,'n':n,'k':k,'r':[r.real,r.imag],'t':[t.real,t.imag],
            'R':reflectance,'T':transmittance,'A':0.0 if k == 0 else max(0.0, absorptance)}


def film_spectrum(config):
    fields = {'material','thickness_um','incident_index','substrate_index','wavelengths_um'}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError('Supply one film material, thickness, incident index, substrate index and wavelengths.')
    material = validate_material(config['material'])
    thickness = _in_range(config['thickness_um'], 0, 1000, 'Film thickness (µm)')
    incident = _in_range(config['incident_index'], 0.001, 20, 'Lossless incident index')
    substrate = _in_range(config['substrate_index'], 0.001, 20, 'Lossless substrate index')
    waves = config['wavelengths_um']
    if not isinstance(waves, list) or not 1 <= len(waves) <= 201:
        raise ValueError('Supply 1–201 strictly increasing vacuum wavelengths.')
    waves = [_in_range(w, 0.001, 1000, 'Vacuum wavelength (µm)') for w in waves]
    if any(b <= a for a,b in zip(waves, waves[1:])):
        raise ValueError('Vacuum wavelengths must increase strictly.')
    samples = []
    for wave in waves:
        optical = evaluate_material(material, wave)
        samples.append(_sample(wave, optical['n'], optical['k'], thickness, incident, substrate))
    return {'schema_version':1,'study':'normal-incidence-film','convention':'exp(-iωt), forward exp(+i 2π(n+ik)z/λ)',
            'reference_planes':'r at the entrance face, t from entrance to exit face, unit incident electric field',
            'samples':samples}


def film_export(config):
    """Reproduce the exact input through the same calculation outside the checkout."""
    result = film_spectrum(config)
    root = Path(__file__).resolve().parent
    payload = io.BytesIO()
    replay = """import json
from pathlib import Path
from backend.thin_film import film_spectrum

root = Path(__file__).resolve().parent
config = json.loads((root / 'input.json').read_text(encoding='utf-8'))
result = film_spectrum(config)
(root / 'replayed.json').write_text(json.dumps(result, indent=2) + '\\n', encoding='utf-8')
print(f"Computed {len(result['samples'])} wavelengths.")
"""
    notebook_code = replay.replace("root = Path(__file__).resolve().parent", "root = Path.cwd()") + "\nresult['samples']"
    notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'}},
                'cells':[{'id':'film-spectrum','cell_type':'code','metadata':{},'outputs':[],'execution_count':None,
                          'source':notebook_code.splitlines(keepends=True)}]}
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(['wavelength_um','n','k','R','T','A','r_real','r_imag','t_real','t_imag'])
    for row in result['samples']:
        writer.writerow([row[key] for key in ('wavelength_um','n','k','R','T','A')] + row['r'] + row['t'])
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('backend/__init__.py', '')
        for name in ('geometry_draft.py','thin_film.py'):
            archive.write(root / name, 'backend/' + name)
        for name,data in [('input.json',config),('result.json',result),('film.ipynb',notebook)]:
            archive.writestr(name, json.dumps(data, indent=2, allow_nan=False) + '\n')
        archive.writestr('film.py', replay)
        archive.writestr('spectrum.csv', table.getvalue())
        archive.writestr('README.txt', 'Extract all files together. Run python film.py or open film.ipynb with the extracted directory as the working directory. Python 3.11 and its standard library suffice; Jupyter is needed only for the notebook. The input describes an infinite planar film with lossless half spaces. Complex electric field amplitudes use [real, imaginary].\n')
    return payload.getvalue()
