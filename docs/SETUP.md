# Local setup and verification

Use Linux x86-64 or WSL2. The current development machine is WSL2 on Linux
6.6.87.2 with an RTX 3080 Ti Laptop GPU (16 GB), NVIDIA driver 617.42, and Python
3.11.15 for the API/FDTDX environment (3.11.18 for Meep). CPU execution does not need an NVIDIA GPU. Native Windows solver
execution is not supported by this milestone.

The API and numerical solvers use separate interpreters. The backend launches
workers in those interpreters, so Meep's native dependencies and JAX's CUDA
runtime do not need to share one environment. No environment files or secrets
need to be uploaded anywhere.

For the v0.9 installed preview, use the local wheel, then
`photonics-workbench setup meep` with Conda to install Meep and MPB.
`photonics-workbench web` serves the bundled GUI and `photonics-workbench doctor`
checks the solver targets. The [packaging guide](PACKAGING.md) gives the
installation commands and tested platforms.

For source development, complete sections 1 and 4 to edit the 27-family
component library locally. Install the appropriate solver environment to run
the [finite-height taper](TAPER_FIELDS.md) and other qualified studies, including [finite-layout fields](LAYOUT_FIELDS.md) and [taper ports](TAPER_PORTS.md).
The remaining library families support design editing and geometric export.

**Verification note:** The API was installed and tested in fresh isolated
virtual environments using official PyPI wheels with verified SHA256 hashes.
Meep was installed into a fresh Conda prefix from official cached packages and
reproduced the numerical output. FDTDX CPU/GPU verification reused an existing
environment with recorded versions; a full fresh FDTDX/CUDA installation and
installation on a different machine have not been established.

## 1. API environment

From the repository root, using Python 3.11:

```bash
git clone https://github.com/srinjoyganguly/photonics-workbench.git
cd photonics-workbench
python3.11 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir --timeout 120 -r requirements-api.txt
```

The pinned core packages are FastAPI 0.120.4, Uvicorn 0.38.0, Pydantic 2.13.4,
Starlette 0.49.3, and NumPy 2.4.6. The requirements file also pins the installed
test/HTTP dependencies. If pip reports no versions for a known package, check
the configured package index and network access; do not install a similarly
named substitute package. The ordinary index-based pip command above timed out
on the event network. The tested fallback fetches the exact interpreter-compatible
wheels from official PyPI release metadata, verifies their hashes, and installs
locally:

```bash
.venv/bin/python scripts/fetch_api_wheels.py
.venv/bin/python -m pip install --no-index --find-links .local/api-wheelhouse --no-deps -r requirements-api.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
```

The v0.10 application suite collected 196 tests. The application interpreter
passed 190 and skipped six checks that require native Meep. All 11 tests
in the separate Meep suite passed, including those six. The frontend passed 28 unit
tests and the production TypeScript build. [Verification summary](verification/v010-checks.json).
Actual solver checks are recorded separately in the relevant study guides.
The API wheelhouse is a local cache and is not committed.

## 2. Meep CPU environment

Install Conda or Miniforge first, then use the official conda-forge PyMeep
package. The unrelated package named `meep` on PyPI is not the solver.

```bash
conda env create -p .local/meep-conda -f environment-meep.yml
.local/meep-conda/bin/python -c 'import meep; print(meep.__version__)'
.local/meep-conda/bin/python -m unittest discover -s tests -p test_meep_smoke.py -v
.local/meep-conda/bin/python -m unittest tests.test_meep_ports_adapter -v
```

The tested versions are PyMeep 1.35.0, Python 3.11.18, and NumPy 2.4.6.
Port studies also require the included MPB support (`from meep import mpb`),
tested with MPB 1.12.0. Their own readiness probe checks it before submission.
The environment location above is detected automatically. To use another
installation, set `MEEP_PYTHON` to its Python executable before starting the API.
An exact Linux x86-64 package export is retained in
[`environment-meep-linux-64.lock`](../environment-meep-linux-64.lock); it records
all packages and artifact checksums from the tested environment.

The YAML setup was verified in a second, fresh environment prefix using the
already-downloaded official conda package cache (`--offline`). The shared
worker ran successfully there and reproduced every exported field, material,
frame, coordinate, and time array exactly. This establishes fresh-prefix
installation on the tested WSL machine; it is not a test on another machine.

## 3. FDTDX CPU or NVIDIA environment

For a CPU-only environment:

```bash
python3.11 -m venv .local/fdtdx-env
.local/fdtdx-env/bin/python -m pip install --no-cache-dir -r requirements-fdtdx.txt
```

For NVIDIA CUDA 12 support, install the CUDA requirement set into the FDTDX
environment instead:

```bash
python3.11 -m venv .local/fdtdx-env
.local/fdtdx-env/bin/python -m pip install --no-cache-dir -r requirements-fdtdx-cuda.txt
```

Use one of those installation alternatives. A CUDA-enabled environment can
also run the explicit CPU route. The tested numerical versions are FDTDX 0.6.2,
JAX/JAXlib 0.10.2, CUDA12 plugin/PJRT 0.10.2, and NumPy 2.4.6. The GPU route
requires a compatible host NVIDIA driver; follow
[JAX's official installation instructions](https://docs.jax.dev/en/latest/installation.html).
For WSL, verify NVIDIA visibility inside WSL before probing JAX.

```bash
nvidia-smi
JAX_PLATFORMS=cuda .local/fdtdx-env/bin/python -m backend.probe fdtdx gpu
JAX_PLATFORMS=cpu .local/fdtdx-env/bin/python -m backend.probe fdtdx cpu
```

The probe runs synchronized JAX arithmetic and verifies placement. It does not
substitute for a real FDTD run. GPU access can be denied by an execution
sandbox even when normal WSL processes can use the GPU. Run in a context with
GPU access; do not treat an unavailable GPU as authorization to switch devices.

The default FDTDX interpreter is `.local/fdtdx-env/bin/python`. Set
`FDTDX_PYTHON` to the Python executable of another suitable environment if
needed. Both interpreter override variables are local shell settings, not
credentials. A checkout-local `.local/runtime.json` may also hold interpreter
paths; that file is ignored by Git and must not be published.

## 4. Build and serve the interface

If you downloaded the release bundle, `frontend/dist` is already included;
skip the build commands and run `bash scripts/start.sh` after setting up the
API and solver environments. A source checkout needs the build below.

Use Node 22.12 or newer and pnpm 11.25.0. A pnpm lockfile records the frontend
dependencies. If Corepack is not bundled with your Node installation, install
Corepack according to the [official pnpm instructions](https://pnpm.io/installation).

```bash
corepack prepare pnpm@11.25.0 --activate
cd frontend
corepack pnpm install --frozen-lockfile --fetch-timeout=600000
corepack pnpm build
cd ..
bash scripts/start.sh
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The same process serves the
built interface and local API. Use `PORT=8001 bash scripts/start.sh` for another
port. Keep the service local; it is not a multi-user hosted compute service.

The TypeScript check and Vite production build passed with Node 24.22.1 and
pnpm 11.25.0 on the Windows side of the development machine. The resulting
static build is served by the WSL Python API. A native Linux Node build remains
a separate setup check. The longer fetch timeout accommodates slow package
downloads; it does not alter application runtime behavior.

## 5. Tests and genuine solver checks

Focused application tests cover physical validation, unsupported execution
targets, failure reporting, cancellation, and API contracts:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The application environment may skip the real Meep test because that solver
lives in a different interpreter. Run its dedicated test explicitly:

```bash
.local/meep-conda/bin/python -m unittest discover -s tests -p test_meep_smoke.py -v
```

Focused v0.7 design checks need no numerical solver:

```bash
.venv/bin/python -m unittest tests.test_component_library tests.test_geometry_draft tests.test_library_layout -v
cd frontend
corepack pnpm test:geometry
cd ..
```

These cover generated geometry, polygon topology, materials, component
membership, placement, API rejection and standalone design source bundles.
They do not measure electromagnetic performance. The separate
`frontend/browser-library-smoke.cjs` exercises the built local interface with
the optional Playwright setup described below.

FDTDX numerical checks, using a fresh worker process for each device:

```bash
.local/fdtdx-env/bin/python scripts/smoke_fdtdx.py --device cpu --output .local/verify-fdtdx-cpu
.local/fdtdx-env/bin/python scripts/smoke_fdtdx.py --device gpu --output .local/verify-fdtdx-gpu
.local/fdtdx-env/bin/python scripts/smoke_fdtdx.py --device cpu --core-width-um 0.8 --output .local/verify-fdtdx-width
```

Only run the GPU command when a CUDA environment and device are available. A
failure is reported as a failure. The checks validate actual finite nonzero
fields, material arrays, coordinates, recorded frames, and device placement.
See [FDTDX evidence](FDTDX_VERIFICATION.md) and [Meep details](meep.md).

With the local app running, the HTTP smoke test submits a genuine solver job,
checks the returned field, and verifies that the exported NumPy data and
configuration match exactly:

```bash
.venv/bin/python scripts/smoke_api.py --solver fdtdx --device cpu
.venv/bin/python scripts/smoke_api.py --solver fdtdx --device gpu
.venv/bin/python scripts/smoke_api.py --solver meep --device cpu
```

All three HTTP checks have passed on the development machine. Browser checks
also passed for Meep CPU, FDTDX CPU, and FDTDX GPU, with an edited 0.65 µm core
width, invalid-physics feedback, actual field/material views, ZIP export,
explicit target selection, JSON import, and the saved run retaining its
original configuration after later edits. Checks at 1440, 1280, and 390 px
found no horizontal overflow or browser errors.

For the browser check, open the example, change core width, select a ready
target, run, inspect Ez and material views, export a ZIP, and import its
configuration. Check the completed metadata against the selected backend and
device. Editing the current design must not change the saved run metadata.

The optional automated browser check is in `frontend/browser-smoke.cjs` and
uses a separate Playwright 1.62.1 installation. Follow
[the browser-testing instructions](browser-tests.md), including `NODE_PATH`
and browser setup. Playwright is not part of the application's build/runtime
dependency set. The event checks used installed Google Chrome on Windows.
Only select installed solver targets; these checks execute real computations
and do not mock field data.

The [finite-height modal study](3D_PORTS.md) uses four reference/device solves
and a twenty-minute worker cap. `frontend/browser-modal-smoke.cjs` verifies its
complete Meep CPU and FDTDX CPU/NVIDIA journeys, with real complex S arrays,
field slices and downloads. `frontend/browser-code-export.cjs` separately
checks geometry and every executable editor's code export. New workers archive
their first-party numerical source before execution; do not edit that numerical
source during a solve, because a changed implementation invalidates its
provenance and causes a visible failure.

## Reproduce an export

For a bundle independent of this repository, choose **Export run Python +
notebook**. It includes the immutable configuration, captured numerical source,
saved arrays, dependency instructions and automatic replay comparisons. Older
runs without a captured source archive are explicitly labeled export-time
compatibility checks. Follow [Python/Jupyter instructions](REPRODUCIBLE_EXPORTS.md)
for the script or notebook. A general geometry draft exports geometry/rendering
code, complete polygon holes, material records and component parameters/ports.
Its bundled validator and generators support standalone parametric edits;
no solver is required to recreate the design. Only supported executable
studies export real solver execution.

Extract a result ZIP to a local directory. Its `config.json` can be imported
through the UI or passed to the CLI:

```bash
.venv/bin/python scripts/run_simulation.py exported-run/config.json --output runs/reproduction-001
```

The output directory must not already exist. The CLI validates the model and
launches the exact requested solver/device; it never silently substitutes a
different target. Results are NumPy `.npz` data with `[y, x]` field/material
arrays, coordinate vectors in micrometres, and any saved time frames in
femtoseconds. Read `metadata.json` for the input snapshot, actual device,
versions, source assumptions, and measured timings.
