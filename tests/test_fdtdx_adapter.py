"""Fast guards that do not import or initialize a solver runtime."""
import sys
import unittest
from unittest.mock import patch

from backend.adapters.fdtdx_adapter import probe


class FdtdxDeviceContractTests(unittest.TestCase):
    def test_target_must_be_explicit(self):
        for value in ("auto", "cuda", "", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "explicit"):
                probe(value)

    def test_reject_already_initialized_process(self):
        with patch.dict(sys.modules, {"jax": object()}):
            with self.assertRaisesRegex(RuntimeError, "fresh worker"):
                probe("gpu")
