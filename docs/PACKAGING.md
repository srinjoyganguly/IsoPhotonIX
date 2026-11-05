# Install Photonics Workbench locally

The **0.11.0a1 wheel** contains the local service, the compiled React interface,
27 component families, solver adapters, examples and dependency notices. The
wheel installs the pinned application dependencies through pip. The tested
platform is Python 3.11 on Linux x86-64 under WSL2. Native solvers run in
separate environments. Tagged wheels are distributed through [GitHub Releases](https://github.com/srinjoyganguly/photonics-workbench/releases).

The [calibration checklist](CALIBRATED_COMPONENTS.md) tracks the v0.11 results. The [v0.10 verification summary](verification/v010-checks.json) records the preceding preview. The [v0.9 wheel check](verification/v09-wheel-install.json) records a fresh
offline installation, an installed CLI run of the Lumerical comparison, and
real simulations of the photonic-crystal line-defect layout on CPU and NVIDIA.
The installed NVIDIA field bytes and material digest matched the corresponding
recorded family case.

## Install and launch

Use the wheel from this release, or build one as described below. In a new
Python 3.11 environment, run the following commands with the actual wheel path.

```bash
python3.11 -m venv .venv-workbench
source .venv-workbench/bin/activate
python -m pip install /path/to/photonics_workbench-0.11.0a1-py3-none-any.whl
photonics-workbench doctor
photonics-workbench web
```

Choose the native solvers you want for this machine:

```bash
photonics-workbench setup meep
photonics-workbench setup fdtdx-cpu
photonics-workbench setup fdtdx-cuda12
photonics-workbench setup beamz-cpu
```

Meep setup includes MPB. The CUDA12 setup checks the actual NVIDIA target
and also verifies CPU execution. A separate CPU profile is optional. See
[managed FDTDX installation](FDTDX_MANAGED_SETUP.md) for offline wheelhouses
and the runtime selection order. The [BeamZ reference guide](BEAMZ_REFERENCE.md) describes its scoped CPU/CUDA benchmarks and the tested managed CPU profile.

Open `http://127.0.0.1:8000` in a browser. `web --port 8010` selects another
loopback port. Node and the source checkout are build-time tools. Installed
runs, registry and caches live under `$XDG_DATA_HOME/photonics-workbench`
(default `~/.local/share/photonics-workbench`). Use `--data-dir /path/to/state`
on any command or set `PHOTONICS_DATA_DIR` to select a different store. A
second installed server can use a separate store and port.

## Run or export a calibrated preset

Open **Calibrated components** in the web or desktop interface. Select an execution target. **Export Python + notebook** saves the preset before simulation. After a successful local run, the result export includes its measured arrays and a reference manifest. Both Python and Jupyter execute the captured numerical source in the chosen worker.

The installed command uses the same code:

```bash
photonics-workbench calibrate multimode --solver meep --device cpu --resolution 48 --offset-um 0.125 --output joint-results
photonics-workbench calibrate splitter --solver fdtdx --device gpu --material-model yee-aligned --export-only --output splitter-program
photonics-workbench calibrate grating --solver meep --device cpu --grating-scope tidy3d-compact --resolution 40 --output beam-results
```

The first command creates a report, raw numerical samples and `python-notebook-results.zip`. The second creates `python-notebook.zip` for execution on a CUDA machine. The third measures the published grating geometry with the compact beam aperture. Use `calibrate ring --solver meep --device cpu --ring-scope lumerical --output ring-results` for one loaded-ring spectrum. In the extracted directory, run `python run.py --output new-result` or open `calibration.ipynb`. A replay of a result export checks its numerical arrays and recorded observables. For long calculations, pass `--checkpoint PATH` to the installed calibrate command or exported run.py. Choose a fresh result directory on retry. The worker checks the saved stages and the numerical source before reuse. Solvers, sources, materials, mesh and wavelength samples are specified in each export. [Per-preset verification](CALIBRATED_COMPONENTS.md).

## Set up the numerical runtimes

For the supported managed Meep/MPB profile, install Conda and run:

```bash
photonics-workbench setup meep --conda /path/to/conda
photonics-workbench doctor --json
```

`setup meep --offline` uses the Conda cache and the bundled Linux x86-64 lock.
The installer checks package identities, runs a Meep field calculation and an
MPB mode check, then registers the working interpreter. It can reuse a
previously qualified managed prefix. To use existing environments instead,
set `MEEP_PYTHON` and/or `FDTDX_PYTHON` to their Python executables. Follow the
[FDTDX CPU and NVIDIA environment instructions](SETUP.md) to prepare FDTDX.
The doctor command probes each execution target and reports its availability.

To reproduce a saved study through the installed CLI:

```bash
photonics-workbench run config.json --output reproduced-run
```

The output directory must be new. Saved-run Python/notebook exports also
contain their numerical source and can be replayed independently. [Standalone
exports](REPRODUCIBLE_EXPORTS.md).

The [v0.8 baseline verification](verification/v08-checks.json) records the
tested artifact's SHA-256. A fresh offline installation provisioned all 90
locked Conda packages and checked the expected MPB homogeneous-mode frequency
of 0.125. Repeating setup reused the same interpreter. The installed CLI
reproduced the GUI taper arrays with managed Meep and with the existing
NVIDIA FDTDX environment. [Managed profile record](verification/v08-managed-install.json).
The installed browser also completed the real taper journey, including
field and material inspection and export. [Browser record](verification/v08-wheel-browser.json).

## Build the wheel from source

First build `frontend/dist` with the pinned package manager as described in
[SETUP.md](SETUP.md). From the repository root, run:

```bash
python3.11 -m pip wheel --no-deps . --wheel-dir dist
```

For an offline build, use an environment with the declared setuptools build
requirement and add `--no-build-isolation --no-index`. The build selects the
script and stylesheet referenced by the compiled index, along with the listed
application modules, examples, measured public benchmark, environment locks
and redistributable notices. The installed service keeps writable state separate
from these resources. [PyPA packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/),
[setuptools package data](https://setuptools.pypa.io/en/stable/userguide/datafiles.html).

`scripts/verify_wheel_install.py` creates a clean environment outside the
checkout, installs via local wheels, checks dependencies, runs an actual Meep
taper through the installed service, restarts it and compares fresh CLI and
standalone Python results with the saved arrays. [Verification record](verification/v08-wheel-install.json).

The installed GUI also completed a real edited taper through Meep CPU.
The browser compared all nine component/plane field views against their
numerical samples and downloaded the results and Python/notebook bundle.
[Installed GUI verification](verification/v08-wheel-browser.json).

The current preview includes managed FDTDX and BeamZ CPU profiles and a dedicated desktop browser window. The following notes retain dependency research and platform goals from the original roadmap.

## Dependency facts and current evidence

| Component | Official distribution / requirement | What is established here |
| --- | --- | --- |
| Meep | Conda-forge `pymeep`; latest documentation also describes a PyPI `pymeep` wheel route | Conda PyMeep 1.35.0 passed fresh-prefix and real numerical checks; PyPI `pymeep` availability is unresolved, as detailed below |
| MPB Python interface | Uses the Meep Python environment through `from meep import mpb` | The managed profile verifies a homogeneous mode in MPB 1.12.0. The public slab/strip benchmarks have separate numerical evidence |
| FDTDX | PyPI `fdtdx`, with a compatible JAX installation | FDTDX 0.6.2 / JAX 0.10.2 passed real slab simulations after fresh CPU and CUDA12 installations |
| BeamZ | PyPI `beamz`, upstream `beamzorg/beamz` | BeamZ 0.6.2 ran independent CPU and NVIDIA propagation benchmarks and a CPU slab eigenmode comparison |
| NVIDIA execution | Compatible host driver plus JAX CUDA dependencies for the JAX route | RTX 3080 Ti Laptop GPU execution verified inside WSL; this does not establish support on other hosts |

### Meep and MPB

[Meep's official installation page](https://meep.readthedocs.io/en/latest/Installation/)
recommends Conda and now also describes `pymeep` binary wheels for Linux and
Apple silicon macOS. However, direct reads of the official PyPI project and
`https://pypi.org/pypi/pymeep/json` both returned **HTTP 404** during this check.
The documented wheel route must therefore remain an evaluation item until
real artifacts and installation can be verified. Retain the proven Conda route
as the initial managed installer. Never substitute `pip install meep`: that
[PyPI project](https://pypi.org/project/meep/) is an unrelated task runner.

Use a pinned, platform-specific Conda environment with conda-forge channel
priority and a recorded artifact lock. A managed micromamba/Conda bootstrap is
an implementation choice still to qualify; downloading it requires a pinned
upstream artifact and checksum. Detect an existing compatible installation
before downloading another. A fallback installer must be explicitly identified
in the setup report; it must not change the requested execution device.

MPB's [Python tutorial](https://mpb.readthedocs.io/en/latest/Python_Tutorial/)
imports `meep.mpb`. Probe that module and run an actual eigenproblem separately
from Meep's FDTD smoke test. Do not assume that an arbitrary distribution named
`mpb` on PyPI supplies this interface.

### FDTDX, BeamZ and JAX

The [FDTDX project](https://github.com/ymahlau/fdtdx) documents pip installation.
Direct official PyPI metadata returned FDTDX **0.6.3**, requiring Python
`>=3.11,<3.15`, on the check date. This does not supersede our verified **0.6.2**
baseline: upgrade only after the adapter regression and numerical tests pass.

The [BeamZ installation guide](https://www.beamz.tech/docs/getting-started/)
identifies `pip install beamz`. Direct official PyPI metadata returned release
**0.6.0**, Python `>=3.11,<3.15`, and wheels for Linux x86-64/aarch64, macOS
x86-64/arm64, and Windows x86-64. Those wheel tags establish artifact availability,
not working CPU/GPU execution. Use release metadata and a lock, not a moving
repository branch, to select its version. Its adapter must identify the exact
backend used and reject unavailable targets.

[JAX's installation guide](https://docs.jax.dev/en/latest/installation.html)
distinguishes CPU, Linux NVIDIA, and experimental WSL2 GPU support. Native
Windows NVIDIA JAX execution is not a supported route. NVIDIA drivers remain
a host prerequisite; pip-installed CUDA components do not install the host
driver. Current upstream guidance includes CUDA 12 and CUDA 13 routes. Keep
our verified CUDA 12 profile until a separate CUDA 13 profile has passed device
and real-solver checks. A wheel being installable or a GPU appearing in a device
list is insufficient: synchronize a small device operation, then run a solver.

## Environment and adapter architecture

Use one lightweight application environment and separately versioned worker
environments. Initial managed profiles should be:

| Profile | Contents | Isolation reason |
| --- | --- | --- |
| `meep-mpb-cpu` | Qualified PyMeep and its compatible native dependencies | Native numerical libraries and MPI/BLAS compatibility |
| `fdtdx-cpu` or `fdtdx-cuda12` | Qualified FDTDX/JAX dependency lock | Explicit CPU or CUDA installation and reproducible JAX versions |
| `beamz-cpu` or a qualified BeamZ GPU profile | Qualified BeamZ backend dependencies | Its dependency constraints and backend lifecycle evolve independently |
| App environment | API, schema, CLI, local file services; optional desktop extra | Starting and repairing the GUI should not import a broken solver |

Keep user data and projects outside the package installation directory. Use
platform-appropriate per-user data, cache and configuration paths. Store each
runtime under a lock-derived identifier and activate it only after installation,
`pip check` or equivalent, import probing, and a small real calculation succeed.
An interrupted install must not replace a working runtime.

A runtime registry should record the interpreter, platform/architecture,
dependency lock digest, installed versions, installer source, probe results,
supported studies and targets, and last verification time. Local absolute paths
stay local; exported provenance uses useful version information without leaking
personal directory names.

Each adapter implements a narrow process protocol: `probe`, `capabilities`,
`validate`, `estimate`, `run`, `cancel`, and result loading. A study and output
directory are serialized data. The application launches a fixed worker module
with explicit arguments and a controlled environment, not a user-composed shell
command. Structured events carry progress, warnings, errors, and completion.
All output is tied to an immutable study snapshot. Unsupported capabilities
must fail before a run, with the reason visible in both GUI and CLI.

Upgrade profiles side by side, then retain the previous verified profile until
the replacement is usable. Repair should leave projects and results intact.
Uninstall must distinguish application code, managed solver environments,
download caches and user projects. Removing the app should never silently
remove research data.

## Local web and desktop architecture

Run `photonics-workbench desktop` for the dedicated local Chromium application window on the qualified WSL2 host. The port and profile remain stable between launches. See [Desktop window](DESKTOP_WINDOW.md) for browser selection, local storage and shutdown behavior. The following design notes describe possible future native shells.

Use the same built React interface and API for both modes. The CLI starts a
loopback-only service, obtains an available port, waits for health, and opens
the browser or desktop window. On shutdown, stop only processes owned by that
application instance and preserve completed runs. Define behavior for active
jobs and a second application instance.

Prefer a **pywebview prototype** for the pip-oriented desktop milestone, subject
to testing. Its [architecture guide](https://pywebview.flowrl.com/guide/architecture.html)
supports a window pointed at a local web service. Its
[installation guide](https://pywebview.flowrl.com/guide/installation.html)
documents platform-specific web-engine requirements, including WebView2 on
Windows and Qt/GTK choices on Linux. A working browser WebGL viewport does not
guarantee the embedded engine works; test Three.js, rendering, picking, files,
downloads, keyboard shortcuts and high-DPI scaling inside the actual window.

| Desktop option | Fit for this project | Decision |
| --- | --- | --- |
| pywebview | Python launcher with an embedded web engine; close to the pip application model | Prototype first; qualify Linux/WSLg and a native Windows shell separately |
| Tauri | Native installer and window lifecycle, with a Python service sidecar | Revisit if a signed standalone installer becomes the primary distribution; adds Rust/platform build requirements |
| Electron | Bundled Chromium and a familiar web application process model | Alternative if consistent WebGL behavior outweighs the larger runtime and extra distribution stack |

These are architecture choices, not current product features. Official
[Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) and
[Electron process documentation](https://www.electronjs.org/docs/latest/tutorial/process-model)
describe their additional runtime/build model.

For Windows, distinguish a native desktop window from native solver execution.
A Windows shell talking to a WSL2 service is a separate integration with startup,
path translation and process-lifecycle tests. Until qualified, the supported
Windows experience is the local browser with the service and solvers inside
WSL2. Do not advertise native Windows Meep or NVIDIA JAX merely because the
desktop window opens on Windows.

Before exposing package installation or richer file operations through the
local API, add session authentication and origin/host checks. A random per-launch
token and constrained API routes are appropriate even on loopback. pywebview's
[security guide](https://pywebview.flowrl.com/guide/security.html) explicitly
addresses local REST API CSRF protection. Solver installation is not an endpoint
that arbitrary web pages should be able to trigger.

## Distribution and notices

The project's own code is MIT licensed. Preserve separate notices for
[Meep GPL-2.0-or-later](https://meep.readthedocs.io/en/latest/License_and_Copyright/),
[MPB GPL-2.0-or-later](https://mpb.readthedocs.io/en/latest/License_and_Copyright/),
[FDTDX MIT](https://github.com/ymahlau/fdtdx/blob/main/LICENSE),
[BeamZ Apache-2.0](https://github.com/beamzorg/beamz/blob/main/LICENSE), and
[JAX Apache-2.0](https://github.com/jax-ml/jax/blob/main/LICENSE).
CUDA and desktop dependencies have their own terms and notices. A per-release
dependency inventory must describe the exact artifacts distributed. Process
isolation is an engineering choice, not a blanket conclusion about licensing.
If native solver binaries are redistributed, include their required notices and
source-availability obligations for that distribution.

Use built frontend assets as wheel package data with hashes and frontend notices.
Build wheels in a clean pipeline, inspect their contents, install into a fresh
environment, and verify startup without a repository or Node installation.
Private handoffs, contact details, raw chats, credentials, local paths and
reference screenshots remain excluded.

## Acceptance gates for a supported package

1. Core wheel installs in a fresh supported Python environment and launches the
   UI without a source checkout or Node.
2. Managed setup installs each requested profile from pinned official artifacts;
   interruption, retry, unavailable network and disk exhaustion are handled.
3. Every offered solver passes a real numerical smoke test. GPU selection is
   checked on actual arrays and never retried silently on CPU.
4. The installed app completes the GUI journey and exports a reproducible result.
5. Offline restart works after dependency installation; all simulation traffic
   remains on the machine.
6. The selected desktop web engine passes the same interactions and can close
   cleanly without abandoned worker processes.
7. Upgrades preserve projects and earlier results; repair leaves a working
   runtime available; uninstall preserves user data by default.
8. CI and release notes distinguish tested platform/profile combinations from
   experimental or unsupported ones. Linux/WSL2 x86-64 comes first; macOS and
   other platforms require their own installation and numerical evidence.
