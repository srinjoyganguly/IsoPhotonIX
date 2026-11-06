"""Stored circuit intent, rigid placement, and offline simulation/geometry document contracts."""
import copy
import io
import json
import unittest
import zipfile

from fastapi.testclient import TestClient
import numpy as np

from backend.app import app
from backend.geometry_draft import validate_geometry
from backend.layout_grid import rasterize
from backend.layout_study import LayoutStudy
from backend.library_layout import build_design, place_component


class CircuitConnections(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(app)
        fixed=build_design('strip',{'length_um':2.})
        moving=place_component(fixed['geometry'],'strip',{'length_um':1.5},
                               {'x':.1,'y':1.,'z':.1,'rotation_deg':0.},fit_domain=True)
        self.geometry=moving['geometry']
        self.fixed,self.moving=fixed['component']['id'],moving['component']['id']
        self.selection=dict(moving_component=self.moving,moving_port='in',fixed_component=self.fixed,fixed_port='out')

    def post(self,operation,geometry,**parameters):
        response=self.client.post('/api/library/'+operation,json={'geometry':geometry,**parameters})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def test_planned_aligned_disconnected_and_port_regeneration(self):
        initial=copy.deepcopy(self.geometry)
        result=self.post('connect',self.geometry,**self.selection)
        edge=result['connection']
        plan=result['geometry']
        self.assertEqual({k:v for k,v in plan.items() if k!='connections'},initial)
        self.assertEqual(plan,validate_geometry(json.loads(json.dumps(plan))))
        planned=self.post('connections',plan)['connections'][0]
        self.assertEqual(planned['state'],'needs_route')
        self.assertGreater(planned['distance_um'],1)
        snapped=self.post('connect',plan,**self.selection,align=True,fit_domain=True)['geometry']
        self.assertEqual(snapped['connections'],[edge])
        self.assertEqual(self.post('connections',snapped)['connections'][0]['state'],'aligned')
        self.assertEqual(snapped['components'][0],initial['components'][0])
        self.assertEqual(self.post('connect',snapped,moving_component=self.fixed,moving_port='out',
                                   fixed_component=self.moving,fixed_port='in')['geometry'],snapped)
        moved=place_component(snapped,'strip',placement={'y':.5,'rotation_deg':15},
                              replace_component_id=self.moving,fit_domain=True)['geometry']
        status=self.post('connections',moved)['connections'][0]
        self.assertEqual(status['state'],'needs_route')
        self.assertEqual(status['angle_error_deg'],15.)
        self.assertEqual(moved['connections'],[edge])
        removed=self.post('disconnect',snapped,connection_id=edge['id'])['geometry']
        self.assertEqual(removed,{**snapped,'connections':[]})
        self.assertEqual(plan['objects'],initial['objects'])

    def test_invalid_endpoints_width_and_occupancy_are_atomic(self):
        narrow=place_component(self.geometry,'strip',{'width_um':.32},replace_component_id=self.moving)['geometry']
        plan=self.post('connect',narrow,**self.selection)['geometry']
        self.assertFalse(self.post('connections',plan)['connections'][0]['widths_match'])
        refused=self.client.post('/api/library/connect',json={'geometry':plan,**self.selection,'align':True})
        self.assertEqual(refused.status_code,422)
        self.assertNotIn('connections',narrow)
        for invalid in ({**self.selection,'fixed_component':self.moving},
                        {**self.selection,'fixed_port':'missing'},
                        {**self.selection,'fixed_port':'in'}):
            reply=self.client.post('/api/library/connect',json={'geometry':plan,**invalid})
            self.assertEqual(reply.status_code,422)
        edge=plan['connections'][0]
        for bad in ({**edge,'id':'bad id'}, {**edge,'a':{**edge['a'],'port':'deleted'}},
                    {**edge,'b':edge['a']},{**edge,'a':{**edge['a'],'inferred':True}}):
            with self.assertRaises(ValueError):
                validate_geometry({**plan,'connections':[bad]})
        with self.assertRaises(ValueError):
            validate_geometry({**plan,'connections':[edge,{**edge,'id':'second'}]})
        with self.assertRaisesRegex(ValueError,'missing port'):
            place_component(plan,'directional_coupler',replace_component_id=self.moving,fit_domain=True)
        self.assertEqual(plan['connections'],[edge])

    def test_link_metadata_does_not_create_material_and_exports_preserve_it(self):
        plan=self.post('connect',self.geometry,**self.selection)['geometry']
        def study(geometry):
            return LayoutStudy(geometry=geometry,source_um={'x':-.8,'y':0.,'z':0.},
                               solver='fdtdx',device='gpu',resolution=16,duration_fs=15)
        original,linked=study(self.geometry).model_dump(),study(plan).model_dump()
        np.testing.assert_array_equal(rasterize(original)[2],rasterize(linked)[2])
        for endpoint,payload,filename in [('geometry',plan,'geometry.json'),('study',linked,'config.json')]:
            response=self.client.post('/api/reproduce/'+endpoint,json=payload)
            self.assertEqual(response.status_code,200,response.text[:120] if response.status_code!=200 else '')
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                exported=json.loads(archive.read(filename))
                self.assertEqual(exported if endpoint=='geometry' else exported['geometry'],plan)
                self.assertIn('design_source/library_layout.py',archive.namelist())
                notebook=json.loads(archive.read('geometry.ipynb' if endpoint=='geometry' else 'study.ipynb'))
                self.assertIn('inspect-connections',[cell['id'] for cell in notebook['cells']])


if __name__=='__main__':
    unittest.main()
