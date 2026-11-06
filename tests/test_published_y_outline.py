import json
from pathlib import Path
import unittest

import numpy as np

from scripts.measure_published_y_outline import NOTEBOOK_SHA, area, vertical_span


class PublishedYOutline(unittest.TestCase):
    def test_archived_paths_match_both_mirrored_arms_and_original_illustration(self):
        file=Path(__file__).resolve().parents[1]/'docs/verification/v011-published-y-photonforge-outline.json'
        record=json.loads(file.read_text())
        self.assertEqual(record['notebook_sha256'],NOTEBOOK_SHA)
        self.assertEqual(record['photonforge_version'],'2.0.0')
        lower,upper=record['arms_lower_upper']
        self.assertAlmostEqual(abs(area(lower)),abs(area(upper)),places=10)
        for x in np.linspace(3.2,9.8,21):
            low_span=vertical_span(lower,x)
            high_span=vertical_span(upper,x)
            np.testing.assert_allclose(low_span,-np.array(high_span)[::-1],atol=1e-8,rtol=0)
        audit=record['render_check']
        for cut in audit['samples']:
            centre=np.mean(vertical_span(upper,cut['x_um']))
            self.assertAlmostEqual(centre,cut['regenerated_span_midpoint_um'])
            self.assertLess(abs(centre-cut['absolute_rendered_centre_um']),audit['pixel_pitch_um'])
        # This cut distinguishes the executable outline from the written sine trajectory.
        quarter=[v for v in audit['samples'] if v['x_um']==4.5][0]
        self.assertGreater(abs(quarter['regenerated_span_midpoint_um']-quarter['text_sine_centre_um']),.04)
