import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import calibrate_finite_grating as grating
from scripts.grating_memory_budget import MIB, estimate


class GratingRuntimeBudget(unittest.TestCase):
    def test_fresh_nearby_mesh_requires_both_stage_estimates(self):
        with TemporaryDirectory() as td, patch.object(grating, "ROOT", Path(td)), \
             patch.dict("os.environ", XLA_PYTHON_CLIENT_ALLOCATOR="platform"):
            path = Path(td)/"docs/verification"
            path.mkdir(parents=True)
            (path/"v012-grating-memory-budget.json").write_text(json.dumps({
                "refinements": [], "reference_runs": {"lumerical": {"shape_only": {"gpu": "test GPU"}}}}))
            m=grating.settings("lumerical",16,"direct-yee")
            profile={"geometry":"lumerical","resolution":16,"allocator":"platform","domain":"full",
                     "grating_spectrum":"standard","material_model":"direct-yee","device":"test GPU",
                     "required_free_with_reserve_mib":7000,
                     "roles":{"reference":{"estimate":{"required_free_with_reserve_mib":6000}},
                              "device":{"estimate":{"required_free_with_reserve_mib":7000}}}}
            measured=path/"v014-grating-memory-lumerical-r16.json"
            measured.write_text(json.dumps(profile))
            self.assertEqual(grating.memory_admission(m,7000)["required_free_with_reserve_mib"],7000)
            with self.assertRaisesRegex(ValueError,"requires"):
                grating.memory_admission(m,6999)
            del profile["roles"]["device"]
            measured.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError,"profile"):
                grating.memory_admission(m,9900)

    def test_memory_baseline_and_candidate_must_refer_to_the_same_native_stage(self):
        base={"resolution":14,"geometry":"lumerical","gpu":"test","cells":100,"compiled_live_bytes":100*MIB,
              "cpu_field_allocation_verified":True,"executed_fdtd_steps":0,"stage":"reference"}
        actual={**base,"executed_fdtd_steps":1,"actual_compiled_live_bytes":100*MIB,"global_used_mib":900,
                "allocator_bytes":{}}
        candidate={**base,"resolution":16,"cells":150,"compiled_live_bytes":150*MIB,"stage":"device"}
        with self.assertRaisesRegex(ValueError,"stage"):
            estimate(base,actual,candidate,600,2000)
