# Meep CPU adapter

This adapter models a lossless, nondispersive dielectric slab in two dimensions.
The slab extends along x and is invariant along z. A continuous sinusoidal,
uniform electric-current line excites Ez across the core width. This source is
not an eigenmode source. Its amplitude ramps linearly during four optical
periods, matching the scalar FDTDX profile. The finite-time example does not
assert steady state or convergence.

Meep uses length units of 1 µm and c = 1. The adapter converts requested time
from femtoseconds using c = 0.299792458 µm/fs. PML surrounds the x-y domain;
the core continues through the x PML. The full field map includes PML cells.
Meep uses its default subpixel dielectric averaging. The sampled material map
can therefore contain intermediate values at interfaces, unlike the stepped
material grid in this FDTDX example.

`arrays.npz` contains actual sampled `field` (Ez), `epsilon`, `frames`,
`times_fs`, `x_um`, and `y_um`. Array order is `[y, x]`, and frame order is
`[time, y, x]`. Meep performs the calculation in double precision; field and
permittivity exports use float32. Field amplitude depends on the arbitrary
source amplitude and is not a normalized power or transmission measurement.

Meep's line source differs from the directed electric/magnetic plane injection
in the FDTDX adapter. The app preserves the same geometry and requested
wavelength, but it does not claim that absolute amplitudes are calibrated
across solvers.

Install with the official conda-forge package into a separate environment:

```bash
conda env create -p .local/meep-conda -f environment-meep.yml
.local/meep-conda/bin/python -m unittest discover -s tests -p test_meep_smoke.py -v
```

The smoke test executes two real waveguides, checks that a finite nonzero field
has propagated to the far half of the domain, verifies coordinate and frame
consistency, and checks that changing core width changes both material and
field arrays. A separate test rejects an explicit GPU request before solver
import. Missing Meep causes the numerical test to be skipped; a skip does not
establish a working solver route.

## Recorded verification

The actual shared worker completed the default example on CPU using Meep
1.35.0 (conda-forge `nompi_py311h17b0aba_100`), Python 3.11.18, and NumPy 2.4.6.
It advanced 960 time steps to 80.05538285 fs and saved 25 frames of 100 × 200
samples. The final maximum absolute Ez was 0.32329729199 in arbitrary source
units. The material map contained 1, 1.25, 3.33333325, and 4, including the
subpixel interface samples. [Recorded metadata](verification/meep-cpu.json)
retains the exact input, versions, source, device, timings, and adapter hash.

The first adapter run measured 1.338 seconds for imports/initialization, 0.344
seconds for time evolution with sampling, and 0.111 seconds for export (1.793
seconds total). This is a single-run observation, not a benchmark.

Both dedicated tests passed. The numerical test ran the default and a wider
0.9 µm core, checked finite nonzero propagation to the far half of the domain,
and verified that material and field arrays changed. The wider core's final
maximum absolute Ez was 0.331968. The separate target guard rejected GPU.

The exact installed Linux package set is retained in
[`environment-meep-linux-64.lock`](../environment-meep-linux-64.lock). Native
Windows, MPI, and GPU execution are not supported by this adapter.

The documented YAML environment was also created in a second fresh prefix
using the cached official packages in offline mode. Its real shared-worker
run reproduced all exported arrays exactly. That check validates prefix
creation and native-library relocation on the tested WSL machine.

Official references used during implementation:

- [Meep installation](https://meep.readthedocs.io/en/latest/Installation/)
- [Waveguide tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/)
- [Array export and coordinate metadata](https://meep.readthedocs.io/en/latest/Python_User_Interface/#array-slices)

The adapter was newly written during the event using the documented Simulation,
Block, CustomSource, Source, PML, get_array, and get_array_metadata APIs. No upstream tutorial
file was copied into the project. Meep and its numerical dependencies retain
their upstream licenses; an application license does not relicense them.
