# Install FDTDX for CPU or NVIDIA CUDA 12

The workbench can provision a separate FDTDX environment from its installed
wheel on **Linux x86-64, including WSL2, with CPython 3.11**. Choose the
profile for your execution target. CPU and NVIDIA environments have separate
settings, so installing either preserves the other. For GPU, the NVIDIA driver
must already be visible to Linux or WSL2. JAX specifies Linux driver 525 or
newer for CUDA 12. Its pip CUDA profile supplies the CUDA libraries.
[JAX installation](https://docs.jax.dev/en/latest/installation.html) ·
[FDTDX quickstart](https://fdtdx.readthedocs.io/en/latest/01_quickstart.html).

```bash
photonics-workbench setup fdtdx-cpu
photonics-workbench setup fdtdx-cuda12
photonics-workbench doctor --json
```

Each command installs into a new local virtual environment, checks every
locked distribution version, runs `pip check`, and simulates the bundled slab.
Setup verifies the native device, finite propagated electric field and actual
material range. The CPU profile checks CPU. The CUDA12 profile checks the GPU
first and also checks CPU execution in its CUDA-enabled environment. Only
after all checks succeed does setup publish the interpreter. A repeat setup
rechecks the environment. If a check or installation fails, the prior
interpreter stays registered. Set `FDTDX_PYTHON` to use an existing environment.

The lock files pin FDTDX 0.6.2, JAX/JAXlib 0.10.2, NumPy 2.4.6 and the full
binary wheel closure for this platform. The CPU lock has 77 wheels. The CUDA12
lock has 92, including NVIDIA libraries. Both install with hash verification.
Allow several gigabytes for the CUDA environment. Use `--data-dir PATH` to
choose a store and use the same location when serving the GUI. If the GPU
driver or selected GPU is unavailable, the CUDA12 setup fails.

## Offline installation

Prepare a wheelhouse on a machine matching the specified Python and platform.
With the corresponding lock from this release, download its wheel artifacts:

```bash
python3.11 -m pip download --require-hashes --only-binary=:all: \
  -r environment-fdtdx-cuda12-linux-64.lock --dest ./fdtdx-wheels
photonics-workbench setup fdtdx-cuda12 --offline --wheelhouse ./fdtdx-wheels
```

For CPU use `environment-fdtdx-cpu-linux-64.lock` and `setup fdtdx-cpu`.
Offline setup reads only that local wheelhouse. A missing or altered artifact
fails the install. Wheel versions can be inspected directly in the
[CPU lock](../environment-fdtdx-cpu-linux-64.lock) and
[CUDA12 lock](../environment-fdtdx-cuda12-linux-64.lock).
The wheelhouse method uses the same pinned artifacts as online setup.

## Measured installations

Two fresh managed virtual environments passed installation from a local
wheelhouse, package checks and repeated setup on WSL2. The CPU profile used
77 locked artifacts. The CUDA12 profile used 92 and ran both on the RTX 3080 Ti
Laptop GPU (driver 617.42) and on CPU. Each readiness solve advanced 839
steps and sampled a 100 × 200 electric-field plane. The propagated peak was
1.14091158 on CPU and 1.14091170 on CUDA in source-relative units. The
simulated permittivity spanned 1 to 4 on both. A separate check hid the GPU
and confirmed explicit CUDA execution failed. The 92 wheel hashes were checked
against their published PyPI release records. [Installation evidence](verification/v010-managed-fdtdx.json).
