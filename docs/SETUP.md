# Local setup and verification

Use Linux x86-64 or WSL2. The current development machine is WSL2 on Linux
6.6.87.2 with an RTX 3080 Ti Laptop GPU (16 GB), NVIDIA driver 617.42, and Python
3.11. CPU execution does not need an NVIDIA GPU. Native Windows solver
execution is not supported by this milestone.

The API and numerical solvers use separate interpreters. The backend launches
workers in those interpreters, so Meep's native dependencies and JAX's CUDA
runtime do not need to share one environment. No environment files or secrets
need to be uploaded anywhere.

**Verification note:** Initial FDTDX CPU/GPU runs used an existing environment.
The API environment was assembled from available installed packages plus
isolated dependencies while package-index access was being diagnosed. The
standard installation commands below are being checked; installation from a
completely clean machine has not yet been established. Do not interpret the
version pins alone as proof of clean installation.

## 1. API environment

From the repository root, using Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements-api.txt
```

The pinned core packages are FastAPI 0.120.4, Uvicorn 0.38.0, Pydantic 2.13.4,
Starlette 0.49.3, and NumPy 2.4.6. The requirements file also pins the installed
test/HTTP dependencies. If pip reports no versions for a known package, check
the configured package index and network access; do not install a similarly
named substitute package.

## 2. Meep CPU environment

Install Conda or Miniforge first, then use the official conda-forge PyMeep
package. The unrelated package named `meep` on PyPI is not the solver.

```bash
conda env create -p .local/meep-conda -f environment-meep.yml
.local/meep-conda/bin/python -c 'import meep; print(meep.__version__)'
.local/meep-conda/bin/python -m unittest tests.test_meep_smoke -v
```

The selected versions are PyMeep 1.35.0, Python 3.11.18, and NumPy 2.4.6.
The environment location above is detected automatically. To use another
installation, set `MEEP_PYTHON` to its Python executable before starting the API.

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

Use Node 22.12 or newer and pnpm 11.25.0. A pnpm lockfile records the frontend
dependencies. If Corepack is not bundled with your Node installation, install
Corepack according to the [official pnpm instructions](https://pnpm.io/installation).

```bash
corepack prepare pnpm@11.25.0 --activate
cd frontend
corepack pnpm install --frozen-lockfile
corepack pnpm build
cd ..
bash scripts/start.sh
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The same process serves the
built interface and local API. Use `PORT=8001 bash scripts/start.sh` for another
port. Keep the service local; it is not a multi-user hosted compute service.

## 5. Tests and genuine solver checks

Focused application tests cover physical validation, unsupported execution
targets, failure reporting, cancellation, and API contracts:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The application environment may skip the real Meep test because that solver
lives in a different interpreter. Run its dedicated test explicitly:

```bash
.local/meep-conda/bin/python -m unittest tests.test_meep_smoke -v
```

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

For the browser check, open the example, change core width, select a ready
target, run, inspect Ez and material views, export a ZIP, and import its
configuration. Check the completed metadata against the selected backend and
device. Editing the current design must not change the saved run metadata.

## Reproduce an export

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
