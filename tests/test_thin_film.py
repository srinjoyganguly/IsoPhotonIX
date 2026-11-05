"""Independent film limits, passive power and portable GUI calculations."""
import cmath
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from fastapi.testclient import TestClient
from backend.app import app
from backend.thin_film import film_spectrum


def spec(n=2, k=0, thickness=0.4, incident=1, substrate=1.45):
    return {'material':{'model':'constant-nk','id':'film','name':'Film','source':'control','notes':'','reference_wavelength_um':1.55,'n':n,'k':k},
            'thickness_um':thickness,'incident_index':incident,'substrate_index':substrate,'wavelengths_um':[1.55]}


class FilmTests(unittest.TestCase):
    def row(self, **kwargs):
        return film_spectrum(spec(**kwargs))['samples'][0]

    def test_zero_thickness_is_the_single_interface_even_with_loss(self):
        row=self.row(k=5,thickness=0)
        r=(1-1.45)/(1+1.45); t=2/(1+1.45)
        self.assertAlmostEqual(complex(*row['r']),r)
        self.assertAlmostEqual(complex(*row['t']),t)
        self.assertAlmostEqual(row['T'],1.45*t*t)
        self.assertAlmostEqual(row['A'],0)

    def test_matched_medium_phase_and_quarter_wave_anti_reflection(self):
        row=self.row(n=1.5,incident=1.5,substrate=1.5)
        self.assertAlmostEqual(complex(*row['r']),0)
        self.assertAlmostEqual(complex(*row['t']),cmath.exp(2j*math.pi*1.5*.4/1.55))
        self.assertAlmostEqual(row['T'],1)
        index=math.sqrt(1.45)
        row=self.row(n=index,thickness=1.55/(4*index))
        self.assertLess(row['R'],1e-25)
        self.assertAlmostEqual(row['T'],1)
        self.assertAlmostEqual(complex(*row['t']),1j/index)
        self.assertEqual(row['A'],0)

    def test_passive_opaque_limit_near_zero_index_and_reciprocity(self):
        row=self.row(n=3,k=20,thickness=1000)
        index=3+20j
        self.assertAlmostEqual(complex(*row['r']),(1-index)/(1+index))
        self.assertEqual(row['T'],0)
        self.assertAlmostEqual(row['A'],1-row['R'])
        tiny=self.row(n=1e-300)
        self.assertAlmostEqual(tiny['R']+tiny['T'],1)
        for n,k in [(0.005,.02),(1.5,0),(2,.12),(15,3)]:
            row=self.row(n=n,k=k);reverse=self.row(n=n,k=k,incident=1.45,substrate=1)
            self.assertAlmostEqual(row['T'],reverse['T'])
            self.assertTrue(0 <= row['A'] <= 1)
            self.assertAlmostEqual(row['R']+row['T']+row['A'],1)

    def test_tabulated_interpolation_and_invalid_study_rejection(self):
        cfg=spec()
        cfg['material'].update(model='tabulated-nk',wavelength_um=[1.4,1.6],n=[1.5,2.5],k=[0,.1])
        cfg['wavelengths_um']=[1.4,1.5,1.6]
        rows=film_spectrum(cfg)['samples']
        self.assertAlmostEqual(rows[1]['n'],2)
        self.assertAlmostEqual(rows[1]['k'],.05)
        client=TestClient(app)
        for patch in ({'wavelengths_um':[1.39]},{'wavelengths_um':[1.5,1.4]}, {'wavelengths_um':[1.4,1.4]},
                      {'incident_index':True},{'substrate_index':0},{'wavelengths_um':[]},{'wavelengths_um':[1.5]*202},
                      {'thickness_um':-1},{'material':dict(cfg['material'],k=[0,-.2])}):
            self.assertEqual(client.post('/api/materials/film',json={**cfg,**patch}).status_code,422)

    def test_export_runs_standalone_python_and_actual_notebook_cells(self):
        cfg=spec(k=.08)
        client=TestClient(app)
        expected=client.post('/api/materials/film',json=cfg).json()
        response=client.post('/api/materials/film/export',json=cfg)
        self.assertEqual(response.status_code,200)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(directory)
            self.assertEqual(json.loads((directory/'result.json').read_text()),expected)
            self.assertEqual(json.loads((directory/'input.json').read_text()),cfg)
            subprocess.run([sys.executable,'film.py'],check=True,cwd=directory,capture_output=True)
            self.assertEqual(json.loads((directory/'replayed.json').read_text()),expected)
            (directory/'replayed.json').unlink()
            cell=''.join(json.loads((directory/'film.ipynb').read_text())['cells'][0]['source'])
            subprocess.run([sys.executable,'-c',cell],check=True,cwd=directory,capture_output=True)
            self.assertEqual(json.loads((directory/'replayed.json').read_text()),expected)
