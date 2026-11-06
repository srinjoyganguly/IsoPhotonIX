"""Scope validation, transport and source capture of full-vector calibrated studies."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from backend.calibration_fields import field_plane
from backend.calibrations import Calibration, design_archive


class SpatialCalibrations(unittest.TestCase):
    def test_3d_targets_and_geometry_are_explicit(self):
        for case in ('ring-3d','grating-3d','layout-ports-3d'):
            for solver, device in [('meep','cpu'),('fdtdx','cpu')]:
                with self.subTest(case=case, solver=solver), self.assertRaises(ValueError):
                    Calibration(case=case,solver=solver,device=device)
        for extra in [{'geometry_3d':'compact'}, {'material_model':'legacy-x'}, {'duration_um_c':50}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                Calibration(case='ring-3d',solver='fdtdx',device='gpu',**extra)
        for extra in [{'geometry_3d':'compact'}, {'duration_um_c':100}, {'port_distance_um':1.25}]:
            with self.assertRaises(ValueError):
                Calibration(case='multimode',solver='meep',device='cpu',**extra)

    def test_old_captured_studies_keep_their_shape(self):
        previous={'study_type':'calibrated-component','revision':1,'case':'coupler','solver':'fdtdx','device':'gpu',
                  'resolution':None,'offset_um':None,'material_model':None,'ring_scope':None,'grating_scope':None}
        self.assertEqual(previous, Calibration.model_validate(previous).model_dump())

    def test_3d_export_contains_the_indirect_numerical_dependencies(self):
        expected={'ring-3d':['scripts/calibrate_ring.py','scripts/calibration_checkpoint.py'],
                  'grating-3d':['scripts/grating_cuda_identity.py','scripts/grating_emission.py','scripts/grating_angular_spectrum.py','docs/verification/v012-grating-commercial-references.json']}
        for case, dependencies in expected.items():
            config=Calibration(case=case,solver='fdtdx',device='gpu')
            with self.subTest(case=case), zipfile.ZipFile(io.BytesIO(design_archive(config))) as archive:
                files = json.loads(archive.read('manifest.json'))['files']
                self.assertTrue(all('source/'+name in files for name in dependencies))
                self.assertTrue(any('cuda12' in name for name in files))

    def test_transverse_axes_and_native_sample_coordinates(self):
        data = np.arange(5*2*3).reshape(5,2,3) + 2j
        descriptor={'array':'field','index':[1],'axis_order':'xy','horizontal_um':[1,2],
                    'vertical_um':[4,5,6],'horizontal_axis':'x','vertical_axis':'z','component':'Ey',
                    'description':'Real part at y=0','wavelength_um':1.55}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'arrays.npz'
            np.savez(path,field=data)
            preview=field_plane({'field_preview':descriptor},path)
            np.testing.assert_array_equal(preview['values'],data[1].real.T)
            self.assertEqual(preview['horizontal_um'],[1,2])
            self.assertEqual(preview['source_shape'],[3,2])
            with self.assertRaises(ValueError):
                field_plane({'field_preview':dict(descriptor,vertical_um=[4,4,6])},path)
            self.assertIsNone(field_plane({},path))

    def test_planar_port_document_is_captured_with_the_program(self):
        config=Calibration(case='layout-ports-2d',solver='meep',device='cpu')
        config.planar_layout['name']='Reproducible port example'
        with zipfile.ZipFile(io.BytesIO(design_archive(config))) as archive:
            design=json.loads(archive.read('source/layout.json'))
            self.assertEqual(design, json.loads(archive.read('config.json'))['planar_layout'])
            self.assertIn('source/layout.json',json.loads(archive.read('manifest.json'))['files'])
        gpu=Calibration(case='layout-ports-3d',solver='fdtdx',device='gpu',planar_layout=config.planar_layout)
        with zipfile.ZipFile(io.BytesIO(design_archive(gpu))) as archive:
            self.assertEqual(json.loads(archive.read('source/layout.json')),gpu.planar_layout)
            self.assertIn('source/scripts/calibrate_layout_ports3d.py',archive.namelist())
        with self.assertRaises(ValueError):
            Calibration(case='multimode-tm',solver='meep',device='cpu',port_distance_um=1.0123)
