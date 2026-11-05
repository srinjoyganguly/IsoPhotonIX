"""Independent geometry probes and study admission, including hidden solids and holes."""
import copy
import unittest
import numpy as np
from backend.layout_study import LayoutStudy, example_layout
from backend.layout_grid import rasterize
from backend.study3d import parse_study


class FiniteLayoutTests(unittest.TestCase):
    def test_rotated_polygon_hole_reveals_underlying_material(self):
        config=example_layout().model_dump()
        g=config['geometry']
        lower=copy.deepcopy(g['objects'][0])
        lower.pop('component_id'); lower.pop('material_id')
        lower.update(id='under',position=dict(x=0.,y=0.,z=0.),size=dict(x=3.,y=3.,z=.5),refractive_index=1.5)
        top={**lower,'id':'pattern','kind':'polygon','visible':False,'refractive_index':2.,
             'rotation_deg':90.,'position':dict(x=.5,y=0.,z=0.),'size':dict(x=2.,y=1.,z=.3),
             'vertices':[dict(x=x,y=y) for x,y in [(-1,-.5),(1,-.5),(1,.5),(-1,.5)]],
             'holes':[[dict(x=x,y=y) for x,y in [(-.2,-.2),(-.2,.2),(.2,.2),(.2,-.2)]]]}
        g.update(objects=[lower,top],materials=[],components=[],domain=dict(x=5.,y=5.,z=3.))
        s=LayoutStudy.model_validate(config)
        spec,axes,eps=rasterize(s.model_dump())
        def epsilon(x,y,z):
            return eps[tuple(np.argmin(abs(v-p)) for v,p in zip(axes,(x,y,z)))]
        self.assertEqual(epsilon(.5,0.,0.),2.25)   # through hole → lower material
        self.assertEqual(epsilon(.5,.7,0.),4.)   # rotated long axis → y
        self.assertEqual(epsilon(1.25,0.,0.),2.25)
        self.assertEqual(epsilon(.5,.7,.45),1.)
        self.assertEqual(spec['source_um'],s.layout()['source_um'])
        g['objects'].reverse()
        self.assertEqual(rasterize(config)[2].max(),2.25)  # opaque lower slab last

    def test_regenerating_member_keeps_material_precedence(self):
        from backend.library_layout import build_design, place_component
        config=example_layout().model_dump()
        first=build_design('strip',{'length_um':3.,'width_um':1.,'height_um':.4})['geometry']
        together=place_component(first,'strip',{'length_um':3.,'width_um':.5,'height_um':.4},
                                 material_id='silicon-reference',fit_domain=True)['geometry']
        config.update(geometry=together,resolution=16,wavelength_um=2.5)
        before=rasterize(config)[2]
        selected=together['components'][0]
        regenerated=place_component(together,'strip',replace_component_id=selected['id'])['geometry']
        after=rasterize({**config,'geometry':regenerated})[2]
        self.assertGreater(before.max(),10.)
        np.testing.assert_array_equal(before,after)

    def test_no_silent_substitution_or_device_fallback(self):
        base=example_layout().model_dump()
        for edit in (lambda c:c['geometry']['materials'][0].update(k=.1),
                     lambda c:c.update(source_um=dict(x=500.,y=0.,z=0.)),
                     lambda c:c.update(device='gpu'),
                     lambda c:c['geometry']['domain'].update(x=100.,y=100.),
                     lambda c:c['geometry']['objects'][0]['size'].update(z=.01)):
            c=copy.deepcopy(base);edit(c)
            with self.assertRaises(ValueError):parse_study(c)
        self.assertEqual(parse_study(base).model_dump(),base)


class MeasuredFieldExamples(unittest.TestCase):
    def test_every_example_still_compiles_to_its_measured_grid(self):
        from backend.component_library import get_catalog
        from backend.layout_examples import qualified_layout_example
        for item in get_catalog()['components']:
            with self.subTest(family=item['id']):
                self.assertIn('layout-3d', item['eligible_studies'])
                example=qualified_layout_example(item['id'])
                self.assertEqual(example.geometry['components'][0]['family'],item['id'])
                self.assertEqual(example.solver,'meep')
