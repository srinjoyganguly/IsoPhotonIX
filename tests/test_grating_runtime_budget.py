import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
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
        with self.assertRaisesRegex(ValueError,"grating_domain"):
            estimate(dict(base,grating_domain="full"),actual,
                     dict(candidate,stage="reference",grating_domain="mirror-y"),600,2000)


class GratingReturnWindows(unittest.TestCase):
    def test_phase_requires_resolved_signal_and_a_settled_reference(self):
        series=np.array([[.04,.001,.04],[.04,.001,.04],[.04,.001,.04]],complex)
        result=grating.return_window_status(series,[0.,0.,.1])
        self.assertEqual(result["phase_qualified"],[True,False,False])
        self.assertEqual(result["amplitude_qualified"],[True,True,False])
        series[-1,0]*=np.exp(.08j)
        self.assertFalse(grating.return_window_status(series,[0.,0.,0.])["phase_qualified"][0])

    def test_return_basis_fails_closed_for_a_different_checkpoint(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/"receipt.json").write_text(json.dumps({"stage":"reference","identity":{"settings":{}},
                                                        "arrays_sha256":"prior"}))
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                grating.return_reference(path,"changed",{},None,"source","delays")

    def test_matched_basis_normalizes_a_known_complex_return(self):
        m=grating.settings("compact-narrow",12)
        inc=np.zeros((5,6,3,4),dtype=np.complex64)
        inc[:,1]=1; inc[:,3]=1
        tag={"name":"test GPU","per_cache_device_sha256":"fixture"}
        with TemporaryDirectory() as tmp:
            path=Path(tmp)
            np.savez(path/"arrays.npz",upper=inc,wavelength_um=grating.monitoring_wavelengths(m))
            sha=grating.file_hash(path/"arrays.npz")
            rec={"stage":"reference","identity":{"settings":m},"arrays_sha256":sha,"cuda_identity":tag,
                 "source_eh_sha256":"launch","source_time_offsets_sha256":"offsets","settled":True}
            (path/"receipt.json").write_text(json.dumps(rec))
            basis,power=grating.return_reference(path,sha,m,tag,"launch","offsets")
            coefficient=.11+.03j
            scattered=inc*coefficient
            scattered[:,3:]*=-1
            recovered=grating.ansys.reflected_fiber_amplitude(scattered,basis,power,1/12)
            np.testing.assert_allclose(recovered,coefficient,rtol=2e-7)
            with self.assertRaisesRegex(ValueError,"identical source"):
                grating.return_reference(path,sha,m,tag,"different","offsets")
