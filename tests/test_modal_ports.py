"""Synthetic complex amplitudes test contracts, not electromagnetic accuracy."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from fastapi.testclient import TestClient

from backend.ports3d import example_ports3d, ModalPortStudy, STAGES
from backend.sparameters import validate_sparameters, sparameters_response, sparameters_csv
from backend.study3d import parse_study


def fixture():
    config = example_ports3d().model_dump()
    f = np.linspace(1/config['wavelength_max_um'],1/config['wavelength_min_um'],5)[::-1]
    s = np.zeros((5,2,2),complex)
    s[:,0,0] = .1+.2j
    s[:,1,1] = .1-.2j
    s[:,1,0] = s[:,0,1] = .8*np.exp(1j*np.linspace(2.8,3.5,5))
    incident = np.tile([1.+.1j,-.3+.7j],(5,1))
    a = dict(wavelength_um=1/f,frequency_per_um=f,s=s,incident_amplitude=incident,
             outgoing_amplitude=s*incident[:,None,:],reference_through=np.ones((5,2),complex),
             valid_bin=np.ones(5,bool),mode_valid=np.ones(5,bool),mode_neff=np.full(5,1.7+0j))
    m = {'config':config,'solver':'meep','actual_device':'CPU','versions':{'meep':'fixture'},
         'sparameters_valid':True,'field_capture_stage':'device_forward',
         'ports':{'port_1_x_um':-1.75,'port_2_x_um':1.75,'phase_convention':'exp(-i omega t)',
                  'mode_gauge':'Positive real Ey anchor','normalization':'Unit modal power'},
         'settlement':{name:{'settled':True,'energy_ratio':1e-8,'spectral_relative_change':1e-6,
                            'decay_by':1e-7,'spectrum_tolerance':1e-4} for name in STAGES}}
    return a,m


class ModalStudyTests(unittest.TestCase):
    def test_examples_and_refinement_are_bounded_and_exact(self):
        for kind in ('straight','gap'):
            c = example_ports3d(kind)
            self.assertEqual(parse_study(c.model_dump()).model_dump(),c.model_dump())
            self.assertEqual(c.estimate()['stage_count'],4)
            self.assertEqual(c.layout()['port_1_x_um'],-1.75)
        c = example_ports3d().model_dump(); c['resolution']=20
        self.assertGreater(ModalPortStudy.model_validate(c).estimate()['cells'],1_000_000)

    def test_unsupported_design_physics_target_and_budget_fail(self):
        for change in ({'device_kind':'straight'}, {'gap_um':.01}, {'device':'gpu'},
                       {'wavelength_min_um':1.65}, {'resolution':20,'max_after_sources_fs':1200.},
                       {'frequency_points':2}, {'extra_option':True}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                ModalPortStudy.model_validate({**example_ports3d().model_dump(),**change})
        c = example_ports3d().model_dump();c['geometry']['objects'][0]['kind']='ring'
        with self.assertRaises(ValueError):ModalPortStudy.model_validate(c)


class ComplexMeasurementTests(unittest.TestCase):
    def test_complex_normalization_matrix_order_and_phase_are_preserved(self):
        a,m = fixture(); v=validate_sparameters(a,m); r=sparameters_response(v,m)
        np.testing.assert_allclose(r['parameters']['S11']['real'],.1)
        np.testing.assert_allclose(r['parameters']['S21']['power'],.64)
        np.testing.assert_allclose(r['parameters']['S21']['phase_unwrapped_rad'],np.linspace(2.8,3.5,5))
        rows=list(csv.DictReader(io.StringIO(sparameters_csv(v))))
        self.assertEqual(float(rows[0]['S11_imag']),.2)
        self.assertEqual(rows[0]['valid_bin'],'true')
        self.assertEqual(r['reciprocity_abs'],[0.]*5)

    def test_undefined_phase_and_invalid_bins_are_not_invented(self):
        a,m=fixture();a['s'][:,0,0]=0;a['outgoing_amplitude'][:,0,0]=0
        r=sparameters_response(validate_sparameters(a,m),m)
        self.assertEqual(r['parameters']['S11']['phase_rad'],[None]*5)
        self.assertEqual(r['parameters']['S11']['magnitude'],[0.]*5)
        a,m=fixture();a['valid_bin'][:]=False;m['sparameters_valid']=False
        m['settlement']['device_reverse'].update(settled=False,energy_ratio=None,spectral_relative_change=None)
        r=sparameters_response(validate_sparameters(a,m),m)
        self.assertEqual(r['parameters']['S21']['real'],[None]*5)
        self.assertIn('false',sparameters_csv(a))

    def test_gain_and_negative_complex_components_are_never_clipped(self):
        a,m=fixture();a['s'][:,1,0]=-1.1+.1j
        a['outgoing_amplitude']=a['s']*a['incident_amplitude'][:,None,:]
        r=sparameters_response(validate_sparameters(a,m),m)
        self.assertTrue(all(v>1 for v in r['parameters']['S21']['power']))
        self.assertTrue(all(v<0 for v in r['unaccounted_modal_fraction'][0]))

    def test_tampered_phase_modes_planes_frequency_and_settlement_fail(self):
        changes = [lambda a,m:a['s'].__setitem__((0,1,0),a['s'][0,1,0]*1j),
                   lambda a,m:a.__setitem__('s',a['s'].real),
                   lambda a,m:a['mode_neff'].__setitem__(0,.8+0j),
                   lambda a,m:m['ports'].update(port_1_x_um=-1.),
                   lambda a,m:m['ports'].pop('mode_gauge'),
                   lambda a,m:m['settlement']['reference_reverse'].update(energy_ratio=1.),
                   lambda a,m:m['settlement']['reference_forward'].update(spectrum_tolerance=.1),
                   lambda a,m:m['settlement'].pop('device_reverse'),
                   lambda a,m:m['config'].update(wavelength_min_um=1.4),
                   lambda a,m:a['mode_valid'].__setitem__(0,False)]
        for mutate in changes:
            a,m=fixture();mutate(a,m)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):validate_sparameters(a,m)

    def test_http_matrix_volume_export_and_corrupt_data(self):
        from backend.app import app
        from backend.passport import file_record
        a,m=fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);out=directory/'modal-test';out.mkdir()
            (out/'config.json').write_text(json.dumps(m['config']))
            (out/'metadata.json').write_text(json.dumps(m))
            np.savez(out/'sparameters.npz',**a)
            (out/'sparameters.csv').write_text(sparameters_csv(a))
            shape=(4,5,6);fields=np.zeros((3,*shape),np.float32)
            np.savez(out/'arrays.npz',field=fields[1],fields=fields,epsilon=np.ones(shape),
                     x_um=np.arange(6.),y_um=np.arange(5.),z_um=np.arange(4.))
            for stage in STAGES:np.savez(out/f"{stage.replace('_','-')}-monitors.npz",complex_samples=np.array([1+2j]))
            job={'id':'modal-test','status':'completed','config':m['config'],'result':m,'logs':''}
            class Manager:
                def get(self,_):return job
            manager=Manager();manager.directory=directory
            with patch('backend.app.manager',manager),TestClient(app,raise_server_exceptions=False) as client:
                self.assertEqual(client.get('/api/jobs/modal-test/sparameters').status_code,200)
                self.assertEqual(client.get('/api/jobs/modal-test/field').status_code,409)
                self.assertEqual(client.get('/api/jobs/modal-test/volume').status_code,200)
                response=client.get('/api/jobs/modal-test/export');self.assertEqual(response.status_code,200)
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    manifest=json.loads(z.read('manifest.json'))['files']
                    self.assertEqual(len(manifest),9)
                    self.assertEqual(z.read('sparameters.csv'),(out/'sparameters.csv').read_bytes())
                    for name,record in manifest.items():self.assertEqual(record['sha256'],file_record(out/name)['sha256'])
                (out/'sparameters.csv').write_text('tampered')
                self.assertEqual(client.get('/api/jobs/modal-test/export').status_code,409)
                (out/'sparameters.csv').write_text(sparameters_csv(a))
                changed=copy.deepcopy(m['config']);changed['gap_um']=.5
                (out/'config.json').write_text(json.dumps(changed))
                for endpoint in ('sparameters','passport','export'):
                    self.assertEqual(client.get(f'/api/jobs/modal-test/{endpoint}').status_code,409)
                # Restore for context shutdown, which only needs this no-op.
                manager.shutdown=lambda:None


if __name__ == '__main__':unittest.main()
