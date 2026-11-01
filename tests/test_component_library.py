"""Geometry/catalog evidence only: these tests do not simulate light."""
from copy import deepcopy
import json
import math
import unittest

from backend.component_library import build_component, get_catalog
from backend.geometry_draft import validate_geometry


def _inside_polygon(x,y,points):
    inside=False
    for a,b in zip(points,points[1:]+points[:1]):
        if (a['y']>y)!=(b['y']>y) and x<(b['x']-a['x'])*(y-a['y'])/(b['y']-a['y'])+a['x']:
            inside=not inside
    return inside


def _contains(item,x,y,z):
    dx=x-item['position']['x'];dy=y-item['position']['y'];a=math.radians(item['rotation_deg'])
    x,y=math.cos(a)*dx+math.sin(a)*dy,-math.sin(a)*dx+math.cos(a)*dy
    if abs(z-item['position']['z'])>item['size']['z']/2: return False
    if item['kind']=='box':return abs(x)<item['size']['x']/2 and abs(y)<item['size']['y']/2
    if item['kind']=='ring':return item['inner_radius']<math.hypot(x,y)<item['radius']
    if item['kind']=='cylinder':return math.hypot(x,y)<item['radius']
    return _inside_polygon(x,y,item['vertices']) and not any(_inside_polygon(x,y,hole) for hole in item['holes'])


class ComponentLibraryTests(unittest.TestCase):
    def test_catalog_defaults_are_independent_and_all_twenty_families_round_trip(self):
        catalog=get_catalog()
        self.assertEqual(len(catalog['components']),20)
        self.assertEqual(len({c['id'] for c in catalog['components']}),20)
        self.assertEqual(len(catalog['materials']),6)
        for recipe in catalog['components']:
            with self.subTest(recipe=recipe['id']):
                built=build_component(recipe['id']);doc=built['geometry'];instance=built['component']
                self.assertEqual(recipe['simulation_support'],'design-only')
                self.assertEqual(doc['schema_version'],2)
                self.assertEqual(validate_geometry(json.loads(json.dumps(doc))),doc)
                self.assertEqual(instance,doc['components'][0])
                self.assertEqual(instance['parameters'],recipe['defaults'])
                self.assertEqual(set(instance['object_ids']),{o['id'] for o in doc['objects']})
                self.assertTrue(all(o['component_id']==instance['id'] for o in doc['objects']))
                self.assertEqual(built,build_component(recipe['id']))
        catalog['components'][0]['defaults']['width_um']=999
        catalog['materials'][0]['n']=999
        self.assertEqual(get_catalog()['components'][0]['defaults']['width_um'],.5)
        self.assertEqual(build_component('strip')['geometry']['materials'][0]['n'],2.)

    def test_real_port_apertures_touch_material_on_inward_side_only(self):
        # A slot's centre is intentionally empty; sample its whole declared
        # aperture. This checks actual solid membership, not recipe formulas.
        for recipe in get_catalog()['components']:
            doc=build_component(recipe['id'],placement={'x':2.3,'y':-1.7,'z':.4,'rotation_deg':37.})['geometry']
            for port in doc['components'][0]['ports']:
                with self.subTest(recipe=recipe['id'],port=port['name']):
                    a=math.radians(port['angle_deg']);normal=(math.cos(a),math.sin(a));tangent=(-normal[1],normal[0])
                    sides={}
                    for side in (-1,1):
                        sides[side]=any(_contains(obj,port['x']+side*.005*normal[0]+f*port['width_um']*tangent[0],
                            port['y']+side*.005*normal[1]+f*port['width_um']*tangent[1],port['z'])
                            for f in (-.4,-.2,0.,.2,.4) for obj in doc['objects'])
                    self.assertTrue(sides[-1],'Port must lead inward into actual material.')
                    self.assertFalse(sides[1],'Port outward direction must leave the component.')

    def test_translation_rotation_and_material_assignment_preserve_parametric_geometry(self):
        for recipe in get_catalog()['components']:
            with self.subTest(recipe=recipe['id']):
                base=build_component(recipe['id'])
                moved=build_component(recipe['id'],placement={'x':7.,'y':-4.,'z':.6,'rotation_deg':90.},material_id='silicon-reference')
                self.assertEqual(moved['component']['parameters'],base['component']['parameters'])
                for old,new in zip(base['geometry']['objects'],moved['geometry']['objects']):
                    self.assertAlmostEqual(new['position']['x'],7-old['position']['y'])
                    self.assertAlmostEqual(new['position']['y'],-4+old['position']['x'])
                    self.assertAlmostEqual(new['position']['z'],.6+old['position']['z'])
                    self.assertEqual(new['size'],old['size'])
                    self.assertEqual(new.get('vertices'),old.get('vertices'))
                    self.assertAlmostEqual(new['rotation_deg'],(old['rotation_deg']+90)%360)
                    self.assertEqual(new['material_id'],'silicon-reference');self.assertEqual(new['refractive_index'],3.48)
                for old,new in zip(base['component']['ports'],moved['component']['ports']):
                    self.assertAlmostEqual(new['x'],7-old['y']);self.assertAlmostEqual(new['y'],-4+old['x'])
                    self.assertAlmostEqual(new['z'],.6+old['z'])
                    self.assertAlmostEqual(new['angle_deg'],(old['angle_deg']+90)%360)

    def test_holes_slots_and_coupler_gaps_remain_empty(self):
        doc=build_component('nanobeam_cavity')['geometry'];solid=doc['objects'][0]
        self.assertEqual(len(solid['holes']),16)
        for hole in solid['holes']:
            x=sum(v['x'] for v in hole)/len(hole);y=sum(v['y'] for v in hole)/len(hole)
            self.assertFalse(_contains(solid,x,y,0.))
        track=build_component('racetrack')['geometry']['objects'][0]
        self.assertEqual(len(track['holes']),1);self.assertFalse(_contains(track,0.,0.,0.))
        for family in ('slot','directional_coupler'):
            geometry=build_component(family)['geometry']
            self.assertFalse(any(_contains(o,0.,0.,0.) for o in geometry['objects']))

    def test_discrete_counts_and_correlated_physical_constraints_are_checked(self):
        self.assertIs(type(build_component('bragg',{'periods':7.})['component']['parameters']['periods']),int)
        for family,parameters in [('bragg',{'periods':7.2}),('bragg',{'periods':True}),('rib',{'slab_height_um':.3}),
                ('ring_bus',{'radius_um':.3,'width_um':1.}),('racetrack',{'length_um':2.}),
                ('mmi_1x2',{'body_width_um':.5}),('mmi_2x2',{'pitch_um':.2}),
                ('s_bend',{'length_um':2.,'offset_um':12.,'width_um':3.}),
                ('nanobeam_cavity',{'hole_radius_um':.7}),('grating_coupler',{'slab_height_um':.5}),
                ('mzi',{'length_um':4.,'transition_um':4.}),('serpentine',{'pitch_um':.2,'width_um':.5})]:
            with self.subTest(family=family,parameters=parameters):
                with self.assertRaises(ValueError):build_component(family,parameters)

    def test_untrusted_values_are_rejected_without_mutating_inputs(self):
        parameters={'width_um':.65};placement={'x':3.,'rotation_deg':-25.};original=deepcopy((parameters,placement))
        result=build_component('strip',parameters,placement)
        self.assertEqual((parameters,placement),original)
        self.assertEqual(result['component']['placement']['rotation_deg'],335.)
        for kwargs in [{'component_id':'unknown'}, {'component_id':'strip','parameters':{'unknown':1}},
                {'component_id':'strip','parameters':{'width_um':float('nan')}},
                {'component_id':'strip','parameters':{'width_um':'0.5'}},
                {'component_id':'strip','parameters':{'width_um':False}},
                {'component_id':'strip','placement':{'rotation_deg':float('inf')}},
                {'component_id':'strip','placement':{'camera':1}},
                {'component_id':'strip','material_id':'secret-file'},
                {'component_id':'strip','placement':{'x':5000.}}]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):build_component(**kwargs)


if __name__=='__main__':
    unittest.main()
