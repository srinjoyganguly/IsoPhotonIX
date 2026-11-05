# Product roadmap

Plan prepared on 1 November 2026 after the hackathon submission. The submitted
v0.2 release remains the numerical baseline. Items below describe planned work
unless an implementation and its verification are explicitly linked. Release
numbers are milestones, not delivery-date promises.

The product direction is a local photonics environment with a shared browser
and desktop interface, a Python API, and independently managed solver runtimes.
The ambition is broad professional photonics capability. “Same as commercial
licenses” is not a bounded acceptance criterion: geometry, meshing, materials,
sources, monitors, solver types, accuracy, optimization, and application domains
each need their own supported scope and evidence. We will publish that scope
feature by feature instead of claiming parity with COMSOL or Tidy3D.

## What we already have

The [v0.2 evidence](../README.md#verification-status) covers one 2D dielectric
slab journey using Meep CPU and FDTDX CPU/NVIDIA GPU, validated input, explicit
target selection, real recorded fields, cancellation, immutable run metadata,
and export with a run passport and file hashes. That route's three-cell
periodic extrusion is a model invariant along z; it is not a finite-height 3D
device. The Blender film is a presentation, not a geometry editor or a solved
3D field.

The post-event v0.4 preview adds a distinct schema-version-2 study for a centred
finite-height straight guide and optional substrate. It connects the actual
editor geometry to Meep CPU and FDTDX CPU/NVIDIA GPU, saves full Ex/Ey/Ez
volumes, and exposes coordinate-aware field/material slices and exports.
[3D usage and evidence](3D_SIMULATION.md) distinguish exact geometry checks,
CPU/GPU consistency, and limited resolution/PML sensitivity from scientific
convergence. This is a bounded first study, not a general CAD-to-solver pipeline.
A saved-run comparison reports repeatability for identical studies from one
solver, including CPU/GPU pairs; it does not compare calibrated observables
across solvers. [Validation status and remaining gates](VALIDATION.md) explicitly
record the limited public mode benchmarks separately from unfinished driven-3D
analytical qualification and commercial agreement.

The v0.5 preview began milestone 6 with a separate bounded 2D guided-port
study: a straight guide or cladding gap, mode-profile pulse, incident reference,
coherent reflection subtraction and finite-aperture R/T spectra. It records
settling, actual execution, raw monitors, CSV and file hashes. See
[port measurement scope and evidence](PORT_MEASUREMENTS.md). The subsequent
v0.6 modal study below extends this measurement work. Packaging and the BeamZ
adapter have not been implemented.

The v0.6 preview adds [finite-height 3D modal ports](3D_PORTS.md): four
directional reference/device solves, a complex fundamental-channel 2×2 S-matrix,
explicit physical reference planes and measured numerical sensitivities. It
also adds [standalone Python/Jupyter source bundles](REPRODUCIBLE_EXPORTS.md)
for all executable study formats and geometry-only code for arbitrary drafts.
This is source-based reproducibility, not the planned pip-installed desktop
application. General multimode ports, arbitrary device geometries, automated
convergence/sweeps, BeamZ, and clean-machine application packaging remain open.

The v0.7 preview expands the **design** model with 20 searchable parametric
families, component placement/regeneration, polygon extrusions with holes and
reusable constant/tabulated n/k material records. Standalone design bundles
carry the original generators, validator, exact contours, materials and
geometric ports. [Implemented catalog and limits](COMPONENT_LIBRARY.md) ·
[Export contract](REPRODUCIBLE_EXPORTS.md). This was the starting point for the v0.8 taper field study.
The [component/material/laser roadmap](COMPONENT_ROADMAP.md) defines separate
gates for general passive studies, quantitative measurements, physical material
models, layout/process support and active devices.

Keep these useful properties through every milestone: local computation,
visible assumptions, reproducible files, actual device verification, and no
silent fallback or simulated results presented as computed data.

## Delivery sequence

| Milestone | Useful outcome | Release gate |
| --- | --- | --- |
| 1 — Interactive 3D design workspace | Inspect and build simple photonic geometry with orbit, pan, zoom, selection, parameter editing, reusable templates, and portable drafts | Browser interaction and serialization checks pass; 2D solver journey still passes; drafts cannot be mistaken for executable 3D studies |
| 2 — Shared project and study model | One versioned project describes geometry, materials, layers, domain, sources, monitors, boundaries, units, and named studies | Python and UI agree on units and transforms; migrations preserve v1 slab configurations; unsupported study/solver combinations are rejected before running |
| 3 — First actual 3D simulation | A finite-height straight dielectric guide with a specified substrate/cladding, driven source, and saved field slices runs locally | Real Meep CPU and FDTDX CPU/GPU runs; geometry/material checks, convergence evidence, placement checks, cancellation and export; no inference of 3D results from extruded 2D fields |
| 4 — Installable application and desktop preview | Install the application wheel, configure managed solver environments, and launch the same UI in a browser or desktop window | Clean-machine install, offline restart, repair and uninstall checks on each claimed platform; real solver smoke test after setup; desktop/WebGL lifecycle tests |
| 5 — Mode and band studies; BeamZ adapter | MPB supplies mode/band calculations; BeamZ becomes a separately verified FDTD choice | Analytical or published reference cases, correct normalization and units, versioned capabilities, independently verified CPU/GPU placement where offered |
| 6 — Engineering photonics workflow | Calibrated ports, broadband spectra, power flux, S-parameters, convergence studies, sweeps, and useful device libraries | Reference-normalized measurements, explicit mode conventions, passivity/reciprocity checks where applicable, reusable uncertainty/convergence reports |
| 7 — Layout and optimization | GDS/layer-stack import, geometry operations, reproducible optimization, fabrication constraints, and larger jobs | Round-trip/layout tests, differentiation checks, constraint enforcement, resource-budget tests, recoverable checkpoints |
| 8 — Additional physics and solver families | Add only the supported material/physics models and numerical methods that users need | Each addition has an appropriate solver adapter, benchmark suite, documented coupling assumptions, and platform qualification |

Packaging research and project-schema design can proceed during the 3D work.
Public packaging support still depends on actual installation tests. A desktop
shell should reuse the UI and local API; it should not create a second modeling
application to maintain.

## Milestone 1: interactive workspace

Status: implemented as the post-event v0.3 preview on
`feat/interactive-3d-workspace`. See the [workspace guide](3D_WORKSPACE.md),
[editor verification](verification/3d-editor.json), and
[real 2D solver regression](verification/3d-solver-regression.json).

The first increment introduced a separate **3D design** workspace alongside the
tested **2D simulation** workspace. Its original implemented scope was:

- Orbit, pan, zoom, fit-to-scene, and isometric/XY/XZ/YZ camera views.
- A visible coordinate system, simulation-domain outline, and grid controls.
- Object selection in the viewport and scene list; hide/show objects.
- Boxes, cylinders, and rings with editable dimensions, positions, z rotation,
  and a scalar refractive index; add, duplicate, and delete operations.
- Straight-strip, directional-coupler, and ring templates as editable geometry.
- Versioned, validated JSON import/export, local draft recovery, and undo/redo.
- An explicitly labeled extrusion preview of the current slab configuration.

The v0.7 increment adds the [20-family library](COMPONENT_LIBRARY.md), editable
polygon contours and holes, referenced material records and reusable component
instances. Their geometry validation and standalone replays qualify design
editing and preservation; they do not satisfy solver qualification gates.

Three.js is the rendering layer; it does not solve Maxwell's equations.
[OrbitControls](https://threejs.org/docs/pages/OrbitControls.html) provides the
camera interaction. The editable 3D draft must not be submitted to a 2D solver
as though it were supported. A “simulation ready” state requires the study
schema and adapters in subsequent milestones.

Acceptance checks should exercise pointer interaction, selection, edits,
undo/redo, malformed imports, refresh/recovery, exported data, viewport resizing,
and failure to create a WebGL context. A useful screenshot is evidence of the
appearance, not evidence that geometry manipulation or solver integration works.

## Milestone 2: establish the shared model

Current increment: the strict finite-waveguide study embeds its bounded legacy
geometry, material indices, domain, numerical settings and explicit target,
while retaining slab compatibility. The separate version-2 **design geometry**
schema adds polygons/holes, materials and component instances; this does not
make that design schema executable by the legacy study adapters. Sources and
boundaries follow each study's documented fixed model. A generalized
project/layer/monitor schema remains future work.

Introduce a versioned `Project` containing stable identifiers and explicit units.
Separate editable design state, a validated study specification, and an immutable
run snapshot. Camera positions and object visibility are presentation state;
they must not change the material geometry sent to a solver. For overlapping
objects, define material precedence rather than letting rendering order decide.

The design vocabulary now includes boxes, cylinders, rings, polygon extrusions
with holes, in-plane rotations and bounded material records. Component recipes
and explicit contours are preserved together. Layer stacks, general Boolean
operations, 3D rotations and solver-specific material fits remain open.
Solvers may rasterize differently; record that fact and export the actual
material grid used in the calculation.

Each study declares its dimensionality, materials, sources, monitors, boundary
conditions, frequency/time settings, and numerical settings. A solver capability
manifest answers whether that exact study is supported. The GUI should explain
a rejection, not substitute a different source, boundary, material, or dimension.
Imported files are data, never executable Python or shell commands.

## Milestone 3: prove a 3D numerical journey

Current increment: the first full 3D straight-guide journey is implemented.
See [Meep numerical checks](verification/meep-3d.json),
[FDTDX CPU/GPU checks](verification/fdtdx-3d.json), and the
[3D guide](3D_SIMULATION.md). Outputs are final-time volume slices; an automatic
convergence framework, analytical qualification and broader studies remain open.
The later [bounded 3D modal study](3D_PORTS.md) supplies its own guided-port
normalization and phase contract. Implementing these journeys does not complete all of this
milestone's numerical release gates.

Begin with one finite-height straight guide before rings, bends, or arbitrary
CAD. Establish the coordinate convention and material stack, then run a modest
domain that fits the development GPU and ordinary CPU memory. Inspect the
solver-produced permittivity volume and measured source location before trusting
the electric field.

Add XY/XZ/YZ field slices with an exact component, coordinate, time/frequency,
and scale. A volume renderer can follow; it must use saved numerical samples
and expose any interpolation or thresholding. For each run, record the mesh,
boundary placement, source, normalization, precision, execution device, solver
version, and the full configuration hash.

Required evidence includes:

1. A homogeneous-medium propagation or other analytical reference case.
2. A dielectric guide with a resolution sweep and boundary/domain sensitivity.
3. Finite, nonzero fields and a physically consistent material grid and source.
4. FDTDX CPU/GPU comparison under documented tolerances and matched precision.
5. A cross-solver comparison only after aligning geometry, source, monitors,
   sampling and normalization; agreement in a colored image is insufficient.
6. Resource estimation that distinguishes the field arrays, PML state, recorded
   output, and compilation overhead; oversized jobs fail before exhausting RAM.

Meep and FDTDX expose 3D facilities upstream; their existence does not establish
our adapters' correctness. Use their current primary documentation during each
implementation: [Meep Python API](https://meep.readthedocs.io/en/latest/Python_User_Interface/)
and [FDTDX documentation](https://fdtdx.readthedocs.io/en/latest/).

## Solver roles and support accounting

| Solver | Intended role | Workbench baseline | Next evidence required |
| --- | --- | --- | --- |
| Meep | Time-domain electromagnetics; CPU route | Real 2D slab and finite-height 3D guide on CPU; 3D grid/PML sensitivity recorded | Analytical benchmarks, calibrated sources/monitors, broader geometry and installation qualification |
| FDTDX / JAX | Time-domain electromagnetics on CPU and NVIDIA GPU; later differentiable studies | Real invariant-z slab and finite-height 3D guide on both targets; explicit device checks and bounded planning memory estimate | Analytical/convergence benchmarks; measured peak memory; clean installation; gradient tests before optimization |
| MPB | Eigenmodes and photonic band structures | Local public slab/strip mode benchmark harness with fixed-frequency residuals, mesh/domain checks and inspectable evidence | General interactive mode studies, band reference, mode tracking, degeneracy and calibrated FDTD coupling |
| BeamZ | Additional FDTD backend for supported photonic studies | No adapter or numerical verification yet | Locked release, isolated installation, minimal real CPU case, GPU placement, field/monitor and export tests |
| Other engines | Additional numerical methods or coupled physics when justified | None | A specific unmet user study and its validation plan before integration |

MPB is an eigenproblem solver, not a drop-in replacement for an FDTD time trace;
its official [Python tutorial](https://mpb.readthedocs.io/en/latest/Python_Tutorial/)
uses `from meep import mpb`. The requested BeamZ project is
[beamzorg/beamz](https://github.com/beamzorg/beamz), linked from its
[official installation guide](https://www.beamz.tech/docs/getting-started/).
Upstream features remain unavailable in this workbench until an adapter exposes
and verifies them.

## Professional feature backlog

The [linear taper](TAPER_FIELDS.md) now runs on Meep CPU and FDTDX CPU/NVIDIA with
a verified physical material grid, explicit excitation and standalone solver
exports. The [taper port study](TAPER_PORTS.md) measures complex scattering into different end guides. The [finite-layout study](LAYOUT_FIELDS.md) handles bounded passive drawings. Broader multiport, multimode and resonance measurements require independent qualification.
The remaining backlog is:

- **Geometry:** qualify selected library geometries for simulation; add routing
  connectivity, parametric dependencies, alignment/snapping, groups, Boolean operations,
  fabrication layers, GDS import/export, and qualified mesh/CAD import.
- **Materials and boundaries:** turn preserved wavelength-dependent design data
  into validated solver models; loss, causal dispersion fits with validity
  ranges, anisotropy where supported,
  symmetry and periodic/Bloch studies, and boundary diagnostics.
- **Sources and measurements:** eigenmode and Gaussian excitation, calibrated
  ports, flux/DFT monitors, overlap integrals, spectra, S-parameters, resonance
  analysis and near-to-far fields where supported.
- **Study management:** parameter sweeps, local queues, convergence reports,
  calibrated cross-solver comparison views, optimization, checkpoints, Python
  APIs and richer notebook workflows. Standalone study/design exports are
  already implemented. The present same-study comparison is restricted to
  repeatability on identical saved grids and numerical conventions.
- **Product quality:** accessible controls, searchable examples, command history,
  autosave/recovery, large-data loading, documented limits, diagnostics,
  packaging, upgrades, and reproducible bug reports without personal paths.
- **Later physics:** thermal, carrier, mechanical or other coupling only after
  choosing the responsible engine and verifying the transfer of units, meshes,
  boundaries and conserved quantities. Installing the four requested solvers
  does not itself provide general multiphysics or a general FEM solver.

On-chip laser work starts with validated passive cavity modes, resonance and
loss. It then needs an independently checked gain model, semiconductor carrier
and rate-equation parameters, and an appropriate travelling-wave/circuit or
coupled-field formulation before making threshold or light–current claims.
Electrical injection and thermal effects require additional engines and
validated coupling. A DBR or nanobeam template and a nominal III–V index are
only geometry foundations; the [staged laser plan](COMPONENT_ROADMAP.md)
defines the evidence required at each step.

The first supported installation target remains Linux/WSL2 x86-64. Broader
platform claims follow clean installation and numerical checks, not merely
upstream platform labels. The [packaging design](PACKAGING.md) describes how a
pip-installable application can manage that complexity without mixing every
solver into the user's Python environment.
