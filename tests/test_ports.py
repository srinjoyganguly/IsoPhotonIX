"""Port configuration, integrity and HTTP contracts; arrays are synthetic fixtures."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from fastapi.testclient import TestClient
from backend.app import app
from backend.ports import PortStudy,example_ports
from backend.study3d import parse_study
from backend.spectra import validate_spectrum,spectrum_response,spectrum_csv


def fixture():
    frequency=np.linspace(1/1.65,1/1.45,3)[::-1]
    return dict(wavelength_um=1/frequency,frequency_per_um=frequency,incident_flux=np.array([2.,2.,2.]),
                reflected_flux=np.array([.04,-.4,-.6]),transmitted_flux=np.array([2.08,1.4,1.2]),reference_transmitted_flux=np.array([2.,2.,2.]),
                reflectance=np.array([-.02,.2,.3]),transmittance=np.array([1.04,.7,.6]),uncollected_fraction=np.array([-.02,.1,.1]),valid_bin=np.ones(3,dtype=bool))


class PortStudyTests(unittest.TestCase):
    def test_examples_roundtrip_without_silent_geometry_changes(self):
        for kind in ('straight','gap'):
            config=example_ports(kind)
            self.assertEqual(parse_study(config.model_dump()),config)
            self.assertEqual(config.layout()['source_x_um'],-4.5)
            self.assertGreater(config.estimate()['maximum_steps_per_stage'],20000)
        invalid=example_ports('straight').model_dump(); invalid['gap_um']=.25
        with self.assertRaises(ValueError): PortStudy.model_validate(invalid)

    def test_unsupported_physics_budget_and_target_are_rejected(self):
        for change in ({'solver':'meep','device':'gpu'},{'device_kind':'ring'},{'gap_um':.01},
                       {'wavelength_min_um':1.7},{'wavelength_max_um':2.8},{'core_index':1.},
                       {'resolution':32.5},{'decay_by':float('nan')},{'spectrum_tolerance':1e-12},
                       {'length_um':16.,'height_um':8.,'resolution':48},{'unknown':1}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                PortStudy.model_validate(example_ports().model_dump()|change)


class PortSpectrumTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.output=self.root/'run'; self.output.mkdir()
        self.config=example_ports().model_dump();self.config['frequency_points']=3
        self.data=fixture()
        self.metadata={'config':self.config,'solver':'meep','actual_device':'CPU','field_component':'Ez',
                       'settlement':{stage:{'settled':True,'energy_ratio':1e-10,'spectral_relative_change':1e-6,
                                            'thresholds':{'energy_decay':self.config['decay_by'],
                                                          'spectral_relative_change':self.config['spectrum_tolerance']}}
                                     for stage in ('reference','device')},'spectra_valid':True}
        self.job={'id':'run','config':self.config,'result':self.metadata,'status':'completed','logs':'synthetic fixture only'}
        self.save()

    def tearDown(self): self.temp.cleanup()

    def save(self):
        np.savez_compressed(self.output/'spectra.npz',**self.data)
        np.savez_compressed(self.output/'arrays.npz',field=np.zeros((4,5)),epsilon=np.ones((4,5)),x_um=np.arange(5.),y_um=np.arange(4.))
        for name,value in [('config.json',self.config),('metadata.json',self.metadata)]: (self.output/name).write_text(json.dumps(value))
        (self.output/'spectrum.csv').write_text(spectrum_csv(self.data))
        np.savez_compressed(self.output/'reference-monitors.npz',native=np.array([1+2j]))

    def test_negative_above_one_and_unaccounted_values_remain_unclipped(self):
        values=validate_spectrum(self.data,self.metadata)
        result=spectrum_response(values,self.metadata)
        self.assertEqual(result['reflectance'][0],-.02)
        self.assertEqual(result['transmittance'][0],1.04)
        self.assertEqual(result['uncollected_fraction'][0],-.02)

    def test_unsettled_bins_are_null_in_api_and_flagged_in_csv_without_fake_zeros(self):
        self.data['valid_bin'][:]=False; self.metadata['spectra_valid']=False
        self.metadata['settlement']['device']['settled']=False
        values=validate_spectrum(self.data,self.metadata)
        result=spectrum_response(values,self.metadata)
        self.assertEqual(result['transmittance'],[None]*3)
        rows=list(csv.DictReader(io.StringIO(spectrum_csv(values))))
        self.assertEqual(rows[0]['valid_bin'],'false')
        self.assertAlmostEqual(float(rows[0]['transmittance']),1.04)

    def test_misnormalization_flags_and_frequency_mismatch_fail(self):
        for change in ('sign','ratio','frequency','settlement','count'):
            data=fixture();meta=json.loads(json.dumps(self.metadata))
            if change=='sign': data['reflected_flux']*=-1
            if change=='ratio': data['transmittance'][0]=1
            if change=='frequency': data['wavelength_um']+=.1
            if change=='settlement':meta['settlement']['device']['settled']=False
            if change=='count':meta['config']['frequency_points']=21
            with self.subTest(change=change),self.assertRaises(ValueError):validate_spectrum(data,meta)

    def test_claimed_settling_requires_finite_diagnostics_meeting_requested_thresholds(self):
        for stage in ('reference','device'):
            for metric in ('energy_ratio','spectral_relative_change'):
                for value in (None,True,-1.,float('nan'),float('inf'),1.):
                    metadata=json.loads(json.dumps(self.metadata))
                    metadata['settlement'][stage][metric]=value
                    with self.subTest(stage=stage,metric=metric,value=value),self.assertRaisesRegex(ValueError,'claims settled'):
                        validate_spectrum(fixture(),metadata)
                metadata=json.loads(json.dumps(self.metadata))
                metadata['settlement'][stage].pop(metric)
                with self.subTest(stage=stage,missing=metric),self.assertRaisesRegex(ValueError,'claims settled'):
                    validate_spectrum(fixture(),metadata)
        for update in ('nested','direct'):
            metadata=json.loads(json.dumps(self.metadata))
            stage=metadata['settlement']['device']
            if update=='nested':stage['thresholds']['spectral_relative_change']=1e-2
            else:stage['decay_by']=1e-4
            with self.subTest(update=update),self.assertRaisesRegex(ValueError,'differs from its requested'):
                validate_spectrum(fixture(),metadata)

    def test_unsettled_early_stop_may_have_unavailable_diagnostics(self):
        data=fixture();data['valid_bin'][:]=False
        metadata=json.loads(json.dumps(self.metadata));metadata['spectra_valid']=False
        metadata['settlement']['device'].update(settled=False,energy_ratio=None,spectral_relative_change=None)
        values=validate_spectrum(data,metadata)
        self.assertEqual(spectrum_response(values,metadata)['transmittance'],[None]*3)

    def test_http_export_hashes_spectra_csv_and_native_monitor_streams(self):
        with patch('backend.app.manager') as manager:
            manager.directory=self.root;manager.get.return_value=self.job
            client=TestClient(app)
            response=client.get('/api/jobs/run/spectrum');self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['reflectance'][0],-.02)
            export=client.get('/api/jobs/run/export');self.assertEqual(export.status_code,200)
            with zipfile.ZipFile(io.BytesIO(export.content)) as bundle:
                manifest=json.loads(bundle.read('manifest.json'))
                for name in ('spectra.npz','spectrum.csv','reference-monitors.npz'):
                    self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(),manifest['files'][name]['sha256'])
                self.assertIn(b'not absorption',bundle.read('README.txt'))
                self.assertIn(b'Measured nonzero incident reference',bundle.read('passport.html'))
            self.assertEqual(client.get('/api/jobs/run/spectrum.csv').content,(self.output/'spectrum.csv').read_bytes())
            self.job['status']='running'
            self.assertEqual(client.get('/api/jobs/run/spectrum').status_code,409)
            self.assertEqual(client.get('/api/jobs/run/export').status_code,409)

    def test_tampered_csv_cannot_enter_export_and_unavailable_target_cannot_start(self):
        with patch('backend.app.manager') as manager:
            manager.directory=self.root;manager.get.return_value=self.job
            (self.output/'spectrum.csv').write_text('fabricated')
            self.assertEqual(TestClient(app).get('/api/jobs/run/export').status_code,409)
        with patch('backend.app.port_capabilities',return_value={'targets':[]}),patch('backend.app.manager.submit') as submit:
            response=TestClient(app).post('/api/ports/jobs',json=self.config)
            self.assertEqual(response.status_code,409);submit.assert_not_called()

    def test_tampered_saved_snapshot_is_rejected_by_spectrum_passport_and_export(self):
        with patch('backend.app.manager') as manager:
            manager.directory=self.root;manager.get.return_value=self.job
            client=TestClient(app)
            for changed in ('config_file','metadata_snapshot','both_saved_files','study_type_downgrade'):
                self.save()
                if changed in ('config_file','both_saved_files'):
                    config=json.loads((self.output/'config.json').read_text())
                    config['gap_um']=.5
                    (self.output/'config.json').write_text(json.dumps(config))
                if changed in ('metadata_snapshot','both_saved_files'):
                    metadata=json.loads((self.output/'metadata.json').read_text())
                    metadata['config']['gap_um']=.5
                    (self.output/'metadata.json').write_text(json.dumps(metadata))
                if changed=='study_type_downgrade':
                    config=json.loads((self.output/'config.json').read_text())
                    config.pop('study_type')
                    (self.output/'config.json').write_text(json.dumps(config))
                for endpoint in ('spectrum','passport','export'):
                    with self.subTest(changed=changed,endpoint=endpoint):
                        self.assertEqual(client.get(f'/api/jobs/run/{endpoint}').status_code,409)
