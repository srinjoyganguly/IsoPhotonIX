# Python and Jupyter exports

The current design bundle includes the same named-port alignment and parametric
regeneration routines as the editor. Numerical Python/notebook exports find
the managed Meep or device-specific FDTDX worker from local workbench settings.
An explicit interpreter in the script or `MEEP_PYTHON` / `FDTDX_PYTHON` takes
precedence. Use `PHOTONICS_DATA_DIR` if the workers live in a custom data store.

The workbench exports both editable Python and a Jupyter notebook for every
supported executable study, including the original 2D fields, finite-height 3D
fields, 2D port measurements, finite-height 3D modal studies, [taper fields](TAPER_FIELDS.md),
[finite-layout fields](LAYOUT_FIELDS.md) and [taper S-parameters](TAPER_PORTS.md). A separate
geometry export preserves editor boxes, cylinders, rings and polygon prisms
with holes, including rotations, material records, component recipes, geometric
ports and hidden-state flags. Use the taper **study** export for a numerical
replay and the **geometry** export for standalone layout editing.

The export is a ZIP containing source code, configuration and setup files. It
does not call the workbench API when executed. A completed-run export also
includes the original numerical arrays, so a replay can be checked against the
actual GUI result. Solver binaries and environments remain separate.

## What is included

| Artifact | Purpose |
| --- | --- |
| `study.py` / `study.ipynb` | Visible, editable CONFIG and real local execution |
| `config.json` | Exact original executable study |
| `source/backend/` | Numerical worker, adapters, validators and array helpers; provenance stated in manifest |
| `workbench_runtime.py` | Local subprocess launch, result inspection and numerical comparison |
| `reference/` | Completed-run configuration, provenance, numerical arrays and available spectral CSV |
| `bundle-manifest.json` | SHA256 fingerprints of source and reference files; base revision and observed versions |
| Environment and requirements files | Pinned core solver requirements, recorded Python-package versions and notebook setup |
| `LICENSE`, `CREDITS.md` | Application license and dependency attribution |

Geometry-only exports replace the study entry points with `geometry.py`,
`geometry.ipynb` and `geometry.json`. Their scripts reproduce the exact validated
design and draw a 3D view. Primitive ring/cylinder radii remain exact in JSON;
polygon contours, including sampled curved paths, are preserved as explicit
coordinates. Their geometry approximation is documented in the
[component guide](COMPONENT_LIBRARY.md). A geometry draft is not converted
into an unsupported electromagnetic simulation.

Design bundles contain `geometry_contract.py` and a `design_source/` snapshot
of the component generators, layout operations and geometry validator. Source
hashes are checked before use. This preserves the generating implementation
without requiring the application checkout, server or solvers. Material
records retain their reference wavelength, source notes and tabulated n/k data;
interpolation stays within the table range and does not create a causal
time-domain material fit.

The notebook includes an optional `regenerate_component` example for editing
one instance through the bundled generator while preserving its neighbors.
Parameter edits create a new design. Leaving that example commented preserves
the exact exported geometry; neither path produces electromagnetic results.

Local runtime paths, environment contents, credentials, worker logs, private
handoffs and conversations are excluded. Notebooks are exported without stale
outputs or execution counts. Third-party solver source and binaries are not
copied into the bundle.

New runs record a deterministic `implementation.zip` before importing the
selected solver. The worker rejects a run if numerical source changes while
it is running. Completed-run exports verify that archive's recorded SHA256,
use those captured source bytes, and label their provenance `run_time`. A replay
must produce the same implementation archive checksum. The archive is retained
under `reference/` so the original provenance remains inspectable.

Older runs without this archive use an explicitly labeled
`export_time_compatibility` snapshot. Agreement demonstrates that the exported
implementation reproduces the historical arrays; it cannot establish that
those exact code bytes produced the historical run. Unrun editor exports use
`export_time`. The Git revision is the checkout revision at export time;
source-file hashes identify actual bytes, including uncommitted development.

## Run outside the repository

For a **design-only** export, install its notebook requirements, then run:

```bash
python -m pip install -r requirements-notebook.txt
python geometry.py
# Or: jupyter lab geometry.ipynb
```

The script writes `geometry.recreated.json` and `geometry.png`. No solver
installation is needed for this design/rendering route. Use a fresh extraction
folder when retaining several versions of the design.

For an **executable study** export:

Extract the bundle anywhere on the local machine. Use Python 3.11 and install
the controller requirements. Install Meep in its official Conda environment or
the selected FDTDX CPU/CUDA requirements as described in the bundle README.

```bash
python -m pip install -r requirements-controller.txt
python study.py --solver-python /path/to/selected/solver/python
```

The example path is a placeholder, not an embedded machine-specific location.
If the current Python already contains the solver, omit `--solver-python`.
`MEEP_PYTHON` or `FDTDX_PYTHON` can also select that interpreter. These options
never change the solver/device in CONFIG. An unavailable NVIDIA target fails;
there is no retry on CPU. A solver subprocess uses the noninteractive Matplotlib
backend so notebook-only plotting dependencies do not leak into a separate
Meep environment.

For Jupyter:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab study.ipynb
```

Select a kernel with those controller/notebook dependencies, set
`SOLVER_PYTHON` in the notebook if needed, and Run All. The notebook's CONFIG is
independently editable and initially matches the script and original JSON.
Every execution writes a new output directory; existing results are not
overwritten. A wall-clock cap is separate from the physical simulation time.

The notebook displays actual final fields, sampled permittivity and available
port spectra. Complex S-parameter phase uses the recorded physical port planes
and mode convention. Principal phase is shown in radians; phase below
`|S| = 1e-8` is masked because near-zero amplitude has no useful phase.

## What “same results” means

An unchanged configuration triggers an automatic comparison with its saved GUI
reference. Every original numerical NPZ array is checked; shape and dtype must
match. Coordinate/time axes, masks and integer arrays require exact equality.
Real/complex values use the fixed, declared `rtol=1e-6`, `atol=1e-12`. Each array
also reports exact numerical equality, maximum absolute difference and
relative L2 difference. Undefined values retain their masks. Actual execution
must match the requested CPU/GPU route, and recorded physical snapshot times
and solver/runtime versions must agree. No fitted scaling or phase alignment
is applied. Captured-source runs also require the original implementation
checksum to match; legacy compatibility results disclose that this check is
unavailable.

Meep's native iterative mode solver can slightly change an unused incoming
residual coefficient even when all fields, outgoing coefficients and S values
are exact. One such replay produced a 3.40 × 10⁻⁸ coefficient difference and
failed the original pointwise test. That failure is retained in the evidence.
The explicit supplementary rule applies only to device-run
`port1_alpha[:,0]`, `port2_alpha[:,1]` and their raw-alpha counterparts. These
channels do not enter S, reference normalization or outgoing amplitudes.
Their maximum error and L2 residual may be bounded using the same `rtol` and
`atol` against the original complete two-direction coefficient array's peak
and L2 norm. This uses no fit or phase adjustment. The original pointwise
failure remains visible. All fields, S, raw E/H and outgoing coefficients keep
their strict pointwise rule.

The resulting `reproduction-check.json` distinguishes exact equality from
tolerance-based equality. It reports `physical_results_passed`,
`all_arrays_pointwise_passed` and `diagnostic_variations` separately. The overall
`passed` flag requires every physical check and the explicitly declared
diagnostic criterion; it does not relabel a pointwise failure as a pass.
Failure of those criteria raises an error. Wall-clock performance, textual labels and
compressed NPZ bytes are not numerical equality criteria. Different hardware
or solver versions may change results; observed versions and execution devices
remain recorded. Universal bitwise equality on arbitrary machines is not
promised.

Editing CONFIG produces a new experiment and explicitly disables the original
GUI equality claim. Editing bundled solver source or reference bytes invalidates
the original reproduction bundle; retain an original copy before developing a
different implementation. SHA256 identifies files, not scientific accuracy.
All model limitations, convergence requirements and unsettled-bin flags still
apply to reproduced output.

## Verification

The v0.9 finite-layout field study passed six Python/notebook replays
from actual GUI downloads across Meep CPU and FDTDX CPU/NVIDIA, with identical
numerical array bytes. The 27-family geometry check passed 56 actual script
and notebook executions. [Layout replay record](verification/v09-layout-exports.json)
· [All design replays](verification/v09-library-exports.json).
Taper modal measurements retain each end's complex effective indices and
native reference/device monitor arrays. The [taper measurement guide](TAPER_PORTS.md)
documents the physical planes, source conventions and sensitivity limits.

The v0.8 taper checks cover six actual GUI-download replays. Python and Jupyter
each reproduced every saved numerical array byte-for-byte on Meep CPU, FDTDX
CPU and FDTDX NVIDIA. Each notebook also displayed the computed field.
[Taper execution records](verification/taper-exports.json)

The v0.7 design-export checks execute Python and notebooks outside the checkout
for all 20 component defaults and a custom-material design: **42 replays** with
exact geometry preservation. Software tests also exercise bundled parametric
regeneration and reject changed source hashes. The library's geometric ports
and material records survive these checks; they are not numerical field or
S-parameter measurements.

The [all-family execution record](verification/component-library-exports.json)
includes archive and geometry hashes. The actual browser-downloaded design
bundle also passed both executions with exact JSON reproduction
([GUI export record](verification/component-library-gui-export.json)). Re-run
the all-family checks with the notebook dependencies installed:

```bash
python scripts/verify_design_exports.py --output /tmp/photonics-design-replay
```

Choose a new output directory outside the checkout. Each script and notebook
runs in an isolated directory; no solver is invoked by this design check.

Structural tests validate complete source snapshots, safe configuration
embedding, exact geometry preservation, source/reference hashes, target
rejection, changed-configuration handling, complex numerical comparison and
isolation of the notebook plotting environment. Synthetic test arrays qualify
those software contracts only.

Actual exported scripts and notebooks are executed outside the checkout using
the real Meep and FDTDX runtimes. The notebook harness uses the official
[NBClient execution API](https://nbclient.readthedocs.io/en/latest/client.html)
with notebook validation, a fresh local kernel and visible failures. The
isolated QA environment uses nbformat 5.10.4, NBClient 0.10.2, ipykernel 7.1.0,
Matplotlib 3.10.8 and JupyterLab 4.4.10. Numerical replay evidence is recorded in
[`verification/reproducible-exports.json`](verification/reproducible-exports.json).

The v0.6 numerical replay matrix contains all three execution routes—Meep CPU, FDTDX CPU
and FDTDX NVIDIA GPU—for each of the four executable schemas. Every one of
these 12 saved studies was replayed as both standalone Python and an executed
notebook outside the checkout: 24 real solver replays. A mixed geometry draft
with a box, rotated ring and hidden cylinder also passed both forms with exact
JSON preservation.

The nine earlier studies are export-time compatibility checks. The three new
3D modal GUI studies use source captured at run time, and all six of their
replays preserve its checksum. Every final replay passes the declared strict
pointwise comparisons. FDTDX CPU's modal arrays are exact in both forms. The
GPU notebook has small float32 energy-reduction differences (relative L2 below
6.4 × 10⁻⁸), while its complex S and raw fields are exact. Both Meep modal
replays have a maximum complex-S difference of 1.55 × 10⁻¹⁰, within the original
pointwise tolerances. These differences remain recorded; universal bitwise
equality is not claimed.

The earlier failed Meep diagnostic attempt remains separately recorded with
its original strict report checksum. A labeled post-hoc comparison confirms
that the same saved arrays satisfy the documented residual bound while still
reporting `all_arrays_pointwise_passed: false`. That reanalysis is not counted
as another solver execution or as an originally successful pointwise test.

Fresh controller/notebook installation was tested on the existing WSL machine.
Numerical replay uses the already qualified solver environments; installing
native Meep/CUDA on every platform has not been established.
