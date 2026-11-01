# FDTDX verification

The adapter was written during the 1 November 2026 event. No code from the
pre-existing benchmark projects was copied. It follows the upstream
[2D simulation tutorial](https://fdtdx.readthedocs.io/en/latest/notebooks/advanced/01_2d_simulation.html)
and was checked against installed FDTDX 0.6.2 source. The documentation currently
describes 0.6.3; the tested runtime is **0.6.2**.

The physical model is a nondispersive dielectric slab with Ez polarization,
1.55 µm wavelength, 0.6 µm core width, indices 2.0 and 1.0, a 10 × 5 µm domain,
20 cells/µm and 1 µm PML at x/y boundaries. A three-cell periodic z extrusion
implements the invariant dimension. A finite uniform plane source at x = −3.5 µm
is directed along +x and ramps over four optical periods. It is not an eigenmode
source and differs from Meep's electric-current line injection.

Real CPU and NVIDIA solves completed before the verified clock check at
14:39:39 London time on 1 November 2026. Both
advanced 839 steps to 79.9808158 fs and returned finite, nonzero 100 × 200 Ez
arrays, permittivity arrays with values 1 and 4, and 25 recorded field frames.

| Measurement | CPU | NVIDIA RTX 3080 Ti Laptop (16 GB) |
| --- | ---: | ---: |
| Setup including imports (s) | 13.525 | 15.681 |
| Main FDTD compilation (s) | 0.938 | 0.929 |
| Synchronized main FDTD execution (s) | 1.642 | 0.853 |
| Adapter total including export (s) | 16.441 | 17.856 |
| Maximum absolute final Ez | 1.14091158 | 1.14091170 |

Versions: Python 3.11, FDTDX 0.6.2, JAX/JAXlib 0.10.2, NumPy 2.4.6. The GPU
execution was verified as `cuda:0`, including synchronized JIT arithmetic and
the actual final field array device. Driver 617.42 was reported by NVIDIA SMI.
These are single-run measurements, not a performance benchmark or speed claim.
Setup includes additional JAX compilation; the separately reported compilation
time refers only to the main FDTD loop.

CPU/GPU final-field maximum absolute difference was 2.8610e−6, with relative L2
difference 1.4238e−6. This is a device consistency check, not physical convergence
or cross-solver validation. Grid/PML convergence, modal purity, transmission, and
experimental agreement have not been established.

A nondefault core width of 0.8 µm also completed on CPU at the recorded UTC
timestamp 2026-11-01T14:40:48.172375+00:00. It produced finite nonzero Ez
(maximum magnitude 1.2034022), the expected material values, and 25 frames.
Adapter total was 15.218 seconds. The actual width edit changes the discretized
core from 12 to 16 cells.

The same 0.8 µm edit was run on the NVIDIA GPU after tightening the adapter to
enforce the exact selected JAX device, not just its platform. Verification was
recorded at 2026-11-01T14:48:03.804905+00:00; maximum field magnitude was
1.2034016 and adapter total was 23.262 seconds. The final output was on `cuda:0`.

The sandbox blocks GPU access. A separate probe in that sandbox failed instead
of choosing CPU. Normal WSL execution with GPU access completed successfully.
The adapter sets `JAX_PLATFORMS` before JAX import, checks FDTDX's backend (which
upstream may otherwise change to CPU), and checks actual field-array placement.
Requests made after JAX has already been imported are rejected; use isolated
worker processes. No automatic GPU-to-CPU retry exists.

Reproduce a run from the repository root with the selected solver environment:

```bash
python scripts/smoke_fdtdx.py --device cpu --config examples/slab-waveguide.json --output .local/fdtdx-cpu
python scripts/smoke_fdtdx.py --device gpu --config examples/slab-waveguide.json --output .local/fdtdx-gpu
```

Use `--core-width-um 0.8` for the nondefault geometry check. The script executes
the real solver and validates the saved arrays. It records the exact config and
returned metadata beside the arrays. Do not run the GPU command from a context
that denies GPU access.

The new adapter is original application code using FDTDX's public API.
[FDTDX](https://github.com/ymahlau/fdtdx) is MIT licensed;
[JAX](https://github.com/jax-ml/jax) is Apache-2.0 licensed. NVIDIA's optional CUDA
runtime packages have their own licenses; the application does not relicense
them. Read the upstream notices when redistributing binary environments.
