# Photonic component, material and laser roadmap

The original roadmap was researched on **1 November 2026**. [Current delivery status](NEXT_CHUNKS.md) and the [vendor coverage review](VENDOR_COMPONENT_COVERAGE.md) reflect later work. This document
defines the next product increments and their evidence requirements. It is a
plan, not a claim that every family or material below is executable. The
[release guide](../README.md) and each study's verification record identify
what is actually implemented.

## What the vendor libraries teach us

Tidy3D's examples span routing, coupling, splitting, filtering, polarization,
periodic structures, detectors and active devices. Its common-PIC construction
tutorial converts parameterized paths into polygon extrusions. This suggests
a reusable geometry model beneath the catalog, rather than a growing list of
fixed pictures. [Tidy3D examples](https://www.flexcompute.com/tidy3d/examples/)
and [common PIC construction](https://www.flexcompute.com/tidy3d/examples/notebooks/PICComponents/).

Ansys separates primitives and complex structures from layer construction,
sources, ports, mesh controls, materials and analysis. A component library
therefore needs explicit study and material contracts as well as shape
generators. [Ansys FDTD reference](https://optics.ansys.com/hc/en-us/articles/360033154434-FDTD-product-reference-manual).

Public vendor examples inform terminology, coverage and benchmark selection.
They do not establish redistribution rights for project files, screenshots,
material databases, foundry PDKs or compact models. Implement our own recipes
and diagrams; retain source/license notices for anything deliberately reused.
No commercial cloud execution, account or runtime dependency is required by
this plan.

## First catalog families

The following is the initial target catalog, with variants added only when
their geometry and parameter invariants are checked. A family can be editable
before it has a qualified electromagnetic study.

| Family | Useful variants and editable parameters |
| --- | --- |
| Waveguides | Strip, slot and rib; width, gap, slab/etch thickness, layer and material |
| Tapers | Linear and smooth width transitions; endpoint widths, length, profile |
| Bends | Circular and Euler; angle, radius, width, curvature controls |
| S-bends | Cosine or spline paths; offset, length and width |
| Directional couplers | Parallel and curved coupling regions; separation and coupling length |
| Y branches | Split angle, branch offset, transition length and width |
| MMI splitters | 1×2 and 2×2; multimode body and access tapers |
| Crossings | Orthogonal and tapered; core width and transition dimensions |
| Ring resonators | All-pass and add/drop; radius, bus separation and coupling region |
| Racetrack resonators | Arc radius, straight section and bus geometry |
| Mach–Zehnder layouts | Split/merge stages, arm separation and path-length difference |
| Spirals | Turns, spacing, bend constraints and total path length |
| Bragg gratings | Period, duty cycle, corrugation, chirp/apodization and phase defect |
| Grating couplers | Linear and focusing; pitch, duty cycle, aperture and etch depth |
| Photonic-crystal cavities | Lattice/hole parameters, missing/shifted cells and nanobeam defects |
| Passive DBR cavities | Two reflectors, spacer length and layer/width modulation |

The electrical or thermal behavior of an MZI, and the lasing behavior of a
DBR/ring cavity, require additional physics. Their passive geometry must not
be labelled a working modulator or laser. A template is not an optimized or
fabrication-qualified component merely because its name matches an example.

## Increment sequence and release gates

| Increment | Deliverable | Evidence before calling it supported |
| --- | --- | --- |
| A — Design library and shape vocabulary | Searchable parametric recipes; polygon extrusions and editable paths; reusable materials; component instances with transforms | Parameter bounds, polygon validity, placement/domain checks, deterministic generation, undo/redo and JSON/Python/notebook round-trips |
| B — General passive device studies | Named ports and explicit sources on selected generated components; actual rasterized material view | Real Meep and FDTDX CPU/GPU runs for each newly supported geometry/source combination; no discarded solids or device fallback |
| C — Quantitative passive engineering | Multimode/multiport S matrices, sweeps, automatic convergence, resonances and uncertainty reports | Mesh/PML/domain/source sensitivities; applicable reciprocity/passivity checks; analytic or redistributable independent references |
| D — Materials and process technology | Dispersive fits, anisotropy, layers, etch operations and process-specific data | Fit residuals and validity bands; passive/causal/stable model checks; material-grid probes; backend-specific physical benchmarks |
| E — Layout and product delivery | Connectivity, routing constraints, hierarchy, GDS/layer mapping, DRC, managed installs and desktop shell | Geometry/layout round-trips; meaningful process rules; clean installation and offline restart on every claimed platform |
| F — Active devices and lasers | Validated gain/carrier/circuit models coupled to optical measurements | The staged laser gates below, followed by electrical/thermal qualification for named device platforms |

The first part of increment B is available for the centered [linear taper](TAPER_FIELDS.md).
A dedicated [taper port study](TAPER_PORTS.md) now measures its two different end widths. [Finite-layout fields](LAYOUT_FIELDS.md) cover bounded drawn scenes. Larger resonant layouts need a resource
estimate and an adequate physical settling time. The existing tested studies
must remain available throughout this expansion.

Every component instance should preserve its generator identifier/version,
parameters, units, material/layer references, local coordinate system and
transform. Named ports need position, outward direction, width, layer and
polarization/mode intent; a schematic port is not automatically a solved mode.
Record tessellation tolerance and actual solver sampling separately from the
parametric design. Define overlap/material precedence, including etched voids.
Object visibility remains a presentation setting, not a way to remove material
from a simulation.

For curves and arbitrary polygons, validate finite coordinates, positive
thickness, winding, self-intersection, degeneracy and domain containment.
Boolean operations, holes, sidewall angles and 3D rotations should enter as
explicitly supported operations with tests, rather than being approximated
silently. Connectivity and fabrication rules need their own validation; two
rendered shapes appearing to touch is insufficient.

## Material customization

A material entry needs a stable identifier, model type, parameter units,
reference source, model/version hash, wavelength validity, temperature and any
composition or carrier-density assumptions. Keep raw user data beside its fit,
and reject out-of-band use unless an explicit, recorded extrapolation policy
is selected. Material names alone are not numerical models.

Start with custom lossless constant-index materials, clearly labelled as such.
Any nominal silicon, silica, nitride, lithium-niobate or III–V preset needs a
named source and valid conditions; an illustrative constant must not masquerade
as a measured broadband dataset. Subsequent model types should include:

- Sampled wavelength/frequency versus complex index or permittivity, preserving
  units and the time/imaginary-sign convention.
- Passive Lorentz/Drude or other supported dispersive fits, with displayed
  residuals, valid bands and stability checks.
- Anisotropic tensors and crystal orientation, independently qualified for
  each backend and source/monitor workflow.
- Temperature/carrier-dependent perturbations, nonlinear response and active
  gain as separate models with their own units and validation.

Tidy3D's material library identifies variants and literature sources. Ansys
imports measured complex-index/permittivity data and exposes fit range, error,
passivity and stability controls. Those are useful requirements for our own
material browser and fitting workflow, rather than justification to copy a
vendor database. [Tidy3D material library](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/material_library.html),
[Ansys sampled materials](https://optics.ansys.com/hc/en-us/articles/360034915093-Creating-new-sampled-data-materials-in-FDTD)
and [Ansys material fitting](https://optics.ansys.com/hc/en-us/articles/360034915033-Using-the-Material-Explorer-to-view-and-adjust-optical-material-models).

The current workbench adapters qualify constant real refractive indices.
Upstream API availability is a separate fact: Meep documents dispersive and
nonlinear models, and the installed FDTDX 0.6.2 source contains a dispersion
model hook. FDTDX's older quickstart says dispersion is absent, while its API
and installed implementation include it. Consequently, inspect the pinned
implementation and run dedicated tests before expanding advertised support;
do not infer current adapter capability from either documentation page alone.
[Meep materials](https://meep.readthedocs.io/en/latest/Materials/),
[FDTDX Material API](https://fdtdx.readthedocs.io/en/stable/api/fdtdx.Material.html)
and [FDTDX quickstart](https://fdtdx.readthedocs.io/en/latest/notebooks/quickstart/02_basic_materials.html).

## On-chip lasers: a staged physical workflow

A passive cavity calculation can measure optical resonances and confinement.
It cannot by itself predict electrical threshold, light–current curves,
saturation, carrier dynamics, noise or self-heating. Constant negative loss is
also insufficient to establish those properties.

Ansys's laser examples combine optical mode information, material-gain data
and a one-dimensional travelling-wave model. Its MQW workflow uses optical
mode, quantum-well gain and laser calculations for a named semiconductor
structure. This supports a coupled workflow rather than attempting to solve
every laser time and length scale in one brute-force 3D FDTD run.
[Ansys laser platform](https://optics.ansys.com/hc/en-us/articles/39727136204051-INTERCONNECT-as-a-Laser-Design-Platform)
and [MQW edge-emitting laser workflow](https://optics.ansys.com/hc/en-us/articles/360041725214-Multi-Quantum-Well-MQW-Edge-Emitting-Laser).

The proposed local implementation proceeds through these independently
testable stages:

1. **Cold cavity.** Passive Fabry–Perot/DBR/DFB/ring layouts; measured
   resonances, linewidth/Q, reflector response, mode confinement and loss.
   Check analytic elementary cavities and converged optical references.
2. **Gain demonstration.** A small Meep Maxwell–Bloch/multilevel example with
   explicit pump, populations, relaxation and saturation. Reproduce a published
   threshold benchmark. This is a gain-model demonstration, not yet a realistic
   electrically pumped semiconductor laser. Meep supplies an official
   [multilevel-atom/SALT comparison](https://meep.readthedocs.io/en/latest/Python_Tutorials/Multilevel_Atomic_Susceptibility/).
3. **Semiconductor reduced model.** Explicit carrier/photon rate equations,
   active volume, confinement, injection efficiency, recombination, internal
   and mirror loss, gain compression and spontaneous-emission assumptions.
   Verify below-threshold decay, threshold/slope against stated analytic
   limits, time-step sensitivity and reproducible initial/noise conditions.
4. **Travelling-wave devices.** Counterpropagating complex envelopes and
   distributed carriers, calibrated gain/dispersion, grating coupling and
   passive optical feedback. Add FP, DFB, DBR, external-ring/Vernier and SOA
   cases one at a time. Qualify spectra, mode competition and modulation only
   when appropriate reference data and temporal convergence are available.
5. **Electrical/thermal coupling.** Doping, contacts, transport, injection,
   heat flow and temperature-dependent gain/index for selected III–V or hybrid
   stacks. Require conservative/unit-consistent transfers, device-specific
   parameters and experimentally supported comparisons before quantitative
   engineering claims.

This is a proposed architecture, not a promise that the current FDTDX GPU
adapter supplies gain or semiconductor carrier physics. A future capability
table must identify the exact engine, model and verified execution target for
each stage. Exported designs, material data, numerical source, initial
conditions and results must remain fully local and reproducible throughout.
