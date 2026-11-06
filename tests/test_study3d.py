import copy
import unittest
from pydantic import ValidationError
from backend.study3d import Waveguide3DConfig, example3d, parse_study
from backend.model import WaveguideConfig


class Study3DTests(unittest.TestCase):
    def config(self):
        return example3d().model_dump()

    def test_default_is_finite_bounded_and_losslessly_roundtrips(self):
        config = self.config()
        self.assertEqual(Waveguide3DConfig.model_validate(config).model_dump(), config)
        self.assertEqual(config['geometry']['schema'], 'isophotonix-geometry')
        self.assertEqual(example3d().estimate()['grid'], [128, 64, 64])
        self.assertEqual(example3d().estimate()['cells'], 524288)
        self.assertEqual(parse_study(WaveguideConfig().model_dump()).schema_version, 1)
        self.assertEqual(parse_study(config).schema_version, 2)

    def test_drafts_are_not_projected_into_another_geometry(self):
        changes = [lambda c: c['geometry'].update(mode='slab-preview'),
                   lambda c: c['geometry']['objects'][1].update(kind='ring'),
                   lambda c: c['geometry']['objects'][1].update(id='bus'),
                   lambda c: c['geometry']['objects'][1].update(rotation_deg=30),
                   lambda c: c['geometry']['objects'][1]['position'].update(y=.1),
                   lambda c: c['geometry']['objects'][1]['size'].update(x=7),
                   lambda c: c['geometry']['objects'].append(copy.deepcopy(c['geometry']['objects'][1]))]
        for change in changes:
            config = self.config(); change(config)
            with self.subTest(config=config), self.assertRaises(ValidationError):
                Waveguide3DConfig.model_validate(config)

    def test_substrate_touching_surface_must_be_explicit_and_visibility_is_not_physics(self):
        config = self.config(); config['geometry']['objects'][1]['size']['z'] = .625
        with self.assertRaisesRegex(ValidationError, 'Fit substrate'):
            Waveguide3DConfig.model_validate(config)
        config['geometry']['objects'][0]['size']['z'] = (4 - .625) / 2
        config['geometry']['objects'][0]['position']['z'] = -2 + (4 - .625) / 4
        config['geometry']['objects'][0]['visible'] = False
        parsed = Waveguide3DConfig.model_validate(config)
        self.assertIsNotNone(parsed.substrate)
        self.assertFalse(parsed.substrate.visible)
        config['geometry']['objects'] = [config['geometry']['objects'][1]]
        self.assertIsNone(Waveguide3DConfig.model_validate(config).substrate)

    def test_physics_resources_unknown_fields_and_target_fail_early(self):
        changes = [lambda c: c.update(resolution=32), lambda c: c.update(resolution=10),
                   lambda c: c.update(device='gpu'), lambda c: c.update(extra=1),
                   lambda c: c.update(wavelength_um=float('nan')),
                   lambda c: c.update(duration_fs='60'),
                   lambda c: c.update(cladding_index=2.1),
                   lambda c: c['geometry']['objects'][1]['size'].update(z=.01),
                   lambda c: c['geometry']['domain'].update(z=2),
                   lambda c: c['geometry']['objects'][0].update(refractive_index=2.5)]
        for change in changes:
            config = self.config(); change(config)
            with self.subTest(config=config), self.assertRaises(ValidationError):
                Waveguide3DConfig.model_validate(config)
        config = self.config(); config.update(solver='fdtdx', device='gpu')
        self.assertEqual(Waveguide3DConfig.model_validate(config).device, 'gpu')

    def test_unknown_version_never_downgrades_to_2d(self):
        config = self.config(); config['schema_version'] = 3
        with self.assertRaises(ValidationError):
            parse_study(config)
