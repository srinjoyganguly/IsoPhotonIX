# How Codex helped

The human supplied the product idea, target users, required engines, explicit-device policy, and hackathon deadline. Codex inspected the existing workspace, implemented the new React/Python application and solver adapters, and ran the checks described below. Human acceptance and the event submission are separate from automated tests.

## Concrete example: a GPU request must never become a CPU run

Task: connect a validated slab-waveguide model to real FDTDX execution while preserving an explicit CPU or NVIDIA GPU choice.

Codex inspected FDTDX's installed configuration implementation and found an automatic CPU fallback path. The new adapter sets `JAX_PLATFORMS` **before importing JAX**, checks the FDTDX backend, synchronizes a JAX probe, and verifies the actual final field-array device. The API launches a fresh worker for every run. It rejects an unavailable target instead of retrying on CPU.

Evidence in this repository:

- `backend/adapters/fdtdx_adapter.py`: pre-import platform selection and actual-array placement checks.
- `backend/runtime.py` and `backend/jobs.py`: per-target process environment and isolated execution.
- `tests/test_fdtdx_adapter.py`: ambiguous targets and already-initialized runtimes are rejected.
- `tests/test_api.py`: an unavailable GPU returns an error and never submits a CPU job.
- `scripts/smoke_api.py`: starts a real job through HTTP, checks finite nonzero fields, and verifies the exported NumPy array exactly matches the field response.
- `docs/FDTDX_VERIFICATION.md`: actual CPU/CUDA measurements and numerical consistency evidence from the event build.

The actual API GPU run completed with a 100 × 200 field array, maximum absolute Ez 1.1409116983, and a 1,678,207-byte result ZIP. The archive held the immutable configuration, actual device metadata, numerical arrays, and sanitized run log. Those values describe one measured run, not a general performance promise.

The [initial implementation commit](https://github.com/srinjoyganguly/photonics-workbench/commit/e4d02d9) records the application and verified FDTDX path at 14:54 London time during the event. Subsequent commits record integration and verification work.

## A review correction that prevented a misleading claim

An independent Codex agent noticed that the recommendation could call FDTDX an “available alternative” even when its probes failed. The implementation now derives that sentence from the readiness results. `tests/test_capabilities.py` verifies the missing-backend case and the no-ready-target case.

## Post-event example: real simulations exposed directional errors

While adding finite-height complex ports, Codex ran both incidence directions
through a symmetric gap. The first implementation failed reflection symmetry.
Inspection of native modal data identified Meep's backward-mode normalization;
separate signed-k projections corrected it without modifying measured outputs.
FDTDX checks separately exposed a backward source gauge problem and tangential
permittivity sampled at the wrong x positions. The adapter fixes travel with
the exported source, and fresh four-direction simulations checked them.

[Meep](docs/verification/meep-ports3d.json) and
[FDTDX](docs/verification/fdtdx-ports3d.json) retain the failed prototype findings,
corrected matrices and sensitivity checks. Meep mesh refinement still changes
modal power by 0.02066, so neither the interface nor documentation claims
converged commercial accuracy. Standalone replay also reports any numerical
differences instead of silently substituting saved GUI results.

Run focused checks with `python -m unittest discover -s tests -v`. Run real solver checks separately using the setup instructions; passing unit tests alone does not establish solver or GPU availability.

The application has no runtime chatbot, OpenAI API dependency, or API key. Official [Codex documentation](https://developers.openai.com/codex) and the [scoped-work Cookbook example](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex) informed the outcome/constraint/verification workflow; no example code was copied. Raw chat logs and private handoff material are excluded.
