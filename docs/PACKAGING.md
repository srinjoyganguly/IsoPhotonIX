# Local application and solver packaging design

Design proposal checked against official sources on **1 November 2026**.
This is a proposed installation architecture, not a claim that a published
`photonics-workbench` wheel, desktop application, or solver installer exists.
The current tested setup remains [SETUP.md](SETUP.md).

## Intended user experience

The application wheel will contain the Python service, command-line entry
point, schema definitions, solver adapters, examples, notices, and a prebuilt
React interface. End users should not need Node, pnpm, a source checkout, or
a compiler to launch the application.

Proposed commands, **not available yet**:

```text
python -m pip install photonics-workbench
photonics-workbench setup --solvers meep,mpb,fdtdx,beamz --device cpu
photonics-workbench doctor
photonics-workbench web

# Optional, on a qualified NVIDIA platform:
photonics-workbench setup --solvers fdtdx,beamz --device cuda

# Optional desktop extra and launcher:
python -m pip install "photonics-workbench[desktop]"
photonics-workbench desktop
```

The setup command or first-run setup screen automates the selected dependencies
in managed environments. It reports download size, target device, platform
requirements and installed versions. It must offer existing-runtime discovery
and an explicit noninteractive mode for reproducible setup. Missing drivers or
unsupported platforms are reported with an actionable explanation.

Do not implement network downloads, driver changes, shell profile changes, or
Conda environment creation as arbitrary `pip` build/install hooks. Installing
the core package should have normal Python package behavior. Automated solver
setup is a visible application action, and can be retried, cancelled or repaired
independently. An expert may continue using the already-supported interpreter
overrides instead.

## Dependency facts and current evidence

| Component | Official distribution / requirement | What is established here |
| --- | --- | --- |
| Meep | Conda-forge `pymeep`; latest documentation also describes a PyPI `pymeep` wheel route | Conda PyMeep 1.35.0 passed fresh-prefix and real numerical checks; PyPI `pymeep` availability is unresolved, as detailed below |
| MPB Python interface | Uses the Meep Python environment through `from meep import mpb` | Local MPB 1.12.0 public slab/strip benchmark harness verified through isolated workers; general interactive mode studies and clean installation remain unqualified |
| FDTDX | PyPI `fdtdx`, with a compatible JAX installation | Existing 0.6.2 / JAX 0.10.2 environment passed real CPU/GPU runs; clean FDTDX installation is still unverified |
| BeamZ | PyPI `beamz`, upstream `beamzorg/beamz` | Identity and official release metadata verified; no local installation or run yet |
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
