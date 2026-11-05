# Credits and dependency notices

The human project owner supplied the photonics workflow, target users, solver
selection, local-compute constraint, and hackathon scope. Codex implemented the
application, investigated environment issues, and ran checks. See
[CODEX_USAGE.md](CODEX_USAGE.md) for a specific evidenced contribution and
[PROVENANCE.md](docs/PROVENANCE.md) for the event-time build record.

The API, interface, validation model, worker orchestration, and two adapters
were newly written for this event. Earlier local research projects supplied an
already-installed FDTDX/JAX environment for initial testing; none of their
source, datasets, reports, or images was copied. No application starter
template was copied. The interface uses original CSS, SVG icons, and canvas
drawings with system fonts. Commercial photonics tools informed the broad
workflow idea; their branding, screenshots, and assets are not included.

The demonstration is an original recording of this application using Google
Chrome and Playwright 1.62.1, encoded with FFmpeg 7.0.2. No browser, recorder, or
encoder binaries are redistributed. The recording includes actual solver
execution and output; its end card describes the tested Codex contribution.

The supplementary journey animation uses an original Blender 5.2 scene,
procedurally created geometry, built-in typography, and textures derived from
our actual saved FDTDX field arrays. No stock models, music, commercial-tool
screenshots, or external image assets are used. Blender is a GPL-licensed
creation tool; its executable is not redistributed. The scene and rendered
artwork are original project assets. See [Blender's licensing explanation](https://www.blender.org/about/license/)
and [animation reproduction notes](docs/BLENDER.md).

## Numerical engines

| Component | Version used or selected | Role | Upstream license/source |
| --- | --- | --- | --- |
| Meep / PyMeep | 1.35.0, conda-forge serial build | Native 2D and finite-height 3D CPU FDTD | [GPL-2.0-or-later](https://github.com/NanoComp/meep/blob/master/src/meep.hpp), [license text](https://github.com/NanoComp/meep/blob/master/LICENSE) |
| MIT Photonic Bands (MPB) | 1.12.0 through `meep.mpb` | Post-event local eigenmode benchmark calculations | [GPL-2.0-or-later](https://github.com/NanoComp/mpb), [retained license](docs/licenses/MPB-GPL-2.0.txt) |
| FDTDX | 0.6.2 | CPU/CUDA FDTD; invariant slab and finite-height 3D studies | [MIT](https://github.com/ymahlau/fdtdx/blob/main/LICENSE) |
| JAX / JAXlib | 0.10.2 | Array execution and compilation | [Apache-2.0](https://github.com/jax-ml/jax/blob/main/LICENSE) |
| JAX CUDA12 plugin / PJRT | 0.10.2 | CUDA execution integration | [JAX source and license](https://github.com/jax-ml/jax) |
| NumPy | 2.4.6 | Numerical interchange and validation | [BSD-3-Clause and bundled notices](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |

Meep's source header explicitly permits GPL version 2 or later. The MIT license
for this application's own code does not replace the licenses of Meep, its
dependencies, or any other external component. Solver environments and vendor
binaries are not committed to the source repository.
The [GPL text supplied with PyMeep 1.35.0](docs/licenses/MEEP-GPL-2.0.txt)
is retained verbatim; its conda package metadata declares GPL-2.0-or-later.
The MPB 1.12.0 package declares the same license and its supplied GPL text is
also retained. Public benchmark parameters and reference values are attributed
in [PUBLIC_BENCHMARKS.md](docs/PUBLIC_BENCHMARKS.md); their numerical comparison
code and result visualizations are new project work. Commercial software,
proprietary example projects and vendor screenshots are not redistributed.

The tested CUDA route also loads NVIDIA libraries, including CUDA runtime
12.9.79, CUDA NVCC 12.9.86, cuDNN 9.24.0.43, cuBLAS 12.9.2.10, and NCCL 2.30.7.
Their installed metadata identifies NVIDIA proprietary licenses. These runtime
dependencies are accepted by this project's scope; avoiding paid commercial
solver and GPU-enablement licenses does not make every driver/runtime component
open source. See [NVIDIA CUDA licensing](https://docs.nvidia.com/cuda/eula/index.html)
and the notices distributed with each package.

FDTDX's installed dependency set includes Equinox 0.13.8 (Apache-2.0),
PyTreeClass 0.11.0 (Apache-2.0), gdstk 1.0.0 (BSL-1.0), and Tidy3D 2.11.2
(LGPLv2+, as declared in its installed metadata). The guided-port adapter uses
FDTDX's ModePlaneSource and its Tidy3D CPU mode-preparation dependency locally;
the selected FDTDX CPU/CUDA runtime performs the time-domain evolution. No
Tidy3D cloud service is called. This use does not implement those libraries'
broader features. Dependency licenses remain with their respective copyright
holders.

## Interface and service

| Component | Pinned version | License |
| --- | --- | --- |
| React / React DOM | 19.2.0 | [MIT](https://github.com/facebook/react/blob/main/LICENSE) |
| Three.js / OrbitControls (post-event 3D workspace) | 0.187.0 | [MIT](https://github.com/mrdoob/three.js/blob/dev/LICENSE) |
| TypeScript | 5.9.3 | [Apache-2.0](https://github.com/microsoft/TypeScript/blob/main/LICENSE.txt) |
| Vite | 7.2.2 | [MIT](https://github.com/vitejs/vite/blob/main/LICENSE) |
| FastAPI | 0.120.4 | [MIT](https://github.com/fastapi/fastapi/blob/master/LICENSE) |
| Pydantic | 2.13.4 | [MIT](https://github.com/pydantic/pydantic/blob/main/LICENSE) |
| Uvicorn | 0.38.0 | [BSD-3-Clause](https://github.com/encode/uvicorn/blob/main/LICENSE.md) |
| Starlette | 0.49.3 | [BSD-3-Clause](https://github.com/encode/starlette/blob/main/LICENSE.md) |
| HTTPX | 0.28.1 | [BSD-3-Clause](https://github.com/encode/httpx/blob/master/LICENSE.md) |
| Playwright (browser tests) | 1.62.1 | [Apache-2.0](https://github.com/microsoft/playwright/blob/main/LICENSE) |

Package metadata, lockfiles, and upstream repositories identify additional
transitive dependencies and their notices. Retain their notices when
redistributing built bundles or environments.
The frontend's runtime MIT notices are retained in
[frontend/THIRD_PARTY_NOTICES.txt](frontend/THIRD_PARTY_NOTICES.txt).

The post-event geometry workspace uses new React editor code and parametric
templates, rendered through Three.js. Its camera controls use the upstream
OrbitControls addon. It does not contain code, assets, screenshots, or geometry
copied from COMSOL or Tidy3D. Type definitions use `@types/three` 0.187.0 in the
development toolchain; these and their dependency metadata remain in the lockfile.

The post-event v0.7 library adds 20 original parametric geometry recipes,
polygon/material validation, component placement and standalone design source
bundles. Public [Tidy3D component examples](https://www.flexcompute.com/tidy3d/examples/notebooks/PICComponents/)
and the [Ansys FDTD reference](https://optics.ansys.com/hc/en-us/articles/360033154434-FDTD-product-reference-manual)
informed family coverage and terminology. No vendor generator code, project
files, PDK, material database or performance dataset was copied. The recipes are original editable geometry. The v0.8 taper uses first-party
adapters around Meep and FDTDX for its bounded 3D field study.

The nominal silicon/silica constant indices are attributed to the public
[Tidy3D/Femwell benchmark](https://www.flexcompute.com/tidy3d/examples/notebooks/ModeSolverFemwellBenchmark/).
Other catalog values are explicitly illustrative approximations, with source
notes stored in each record. The III–V preset is a passive placeholder, not
semiconductor gain data. Users remain responsible for the provenance and
redistribution rights of material tables they add. The
[component/material/laser roadmap](docs/COMPONENT_ROADMAP.md) cites the official
references used to plan subsequent physics and validation.

## Documentation used

Standalone Python/notebook exports optionally use Matplotlib 3.10.8
(Matplotlib's PSF-based license), nbformat 5.10.4, nbclient 0.10.2, ipykernel
7.1.0 and JupyterLab 4.4.10 (BSD-family licenses). These packages supply plotting,
notebook serialization/execution and the notebook interface. The bundles
contain first-party source and dependency instructions, not third-party
libraries or binaries; installed packages retain their own license notices.

The finite-height modal adapter's local source-mode gauge handling and bounded
x-interface Yee permittivity placement are first-party implementation work,
tested against real runs. They do not modify installed Meep/FDTDX code or claim
to implement a general commercial conformal-meshing system.

The adapters use the current documented APIs, checked against installed
versions. They do not copy an upstream tutorial file wholesale.

- [Meep installation](https://meep.readthedocs.io/en/latest/Installation/),
  [waveguide tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/),
  and [Python interface](https://meep.readthedocs.io/en/latest/Python_User_Interface/).
- [FDTDX 2D tutorial](https://fdtdx.readthedocs.io/en/latest/notebooks/advanced/01_2d_simulation.html)
  and installed 0.6.2 source. Online latest documentation describes 0.6.3;
  runtime compatibility was checked rather than assumed.
- [Meep mode decomposition](https://meep.readthedocs.io/en/latest/Mode_Decomposition/),
  [interface averaging](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/),
  and installed FDTDX 0.6.2 mode-source/phasor implementations informed the
  finite-height modal adapters and their sampling conventions.
- [JAX platform configuration](https://docs.jax.dev/en/latest/config_options.html#platforms)
  and [installation](https://docs.jax.dev/en/latest/installation.html).
- [Official Codex documentation](https://developers.openai.com/codex)
  and [scoped Codex work example](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex).
  These informed the implementation workflow. There is no OpenAI runtime API
  dependency and no API key requirement.
