# Parametric photonic component library

The library adds **27 editable component families** to the local 3D design
editor. Each instance retains its recipe, dimensions, placement, material,
member solids and named geometric ports. These are original geometry
generators, not imported vendor component files or measured performance data.

The [finite-height taper](TAPER_FIELDS.md) has a dedicated study with straight
access leads. The [finite-layout field study](LAYOUT_FIELDS.md) validates the
actual saved solids, materials, grid and local source independently of recipe
family. An eligible layout yields instantaneous electric fields and sampled
permittivity. Each catalog port describes geometric connection intent. Power
splitting, loss, resonance and modal S measurements need calibrated sources
and monitors for the particular device.

Every catalog family has a tested finite-layout example on all three execution
targets. [Per-family field qualification](LAYOUT_FIELDS.md#verification) ·
[Deletion, undo and named-port assembly](EDITOR_RECOVERY.md).

## Using an instance

Open a recipe in the component library, choose its dimensions and material,
and inspect the generated solids in the 3D editor. Place additional instances
with an x/y/z translation and an in-plane rotation. Component edits regenerate
the member geometry from its parameters. Named ports retain their outward
direction after a rotation; they describe connection intent, not computed
eigenmodes.

All lengths are in **micrometres**. Every recipe has a finite positive core
height. The [exported catalog](verification/v09-component-catalog.json) records the exact names,
units, defaults, minima and maxima of every physical parameter. The backend checks
these bounds when it builds a component.
Related dimensions also have constraints: for example, a slab must be thinner
than its ridge, a ring bus must extend beyond the ring, and a hole must leave
material between itself and the next hole. A parameter's individual maximum
is not a promise that every combination of maximum values is valid.

Opening a component sizes an origin-centred design domain to contain its
transformed solids with at least 1 µm of geometric margin on every side.
This margin is for design display; it is not a qualified absorbing boundary
or simulation padding recommendation.

## Available families

| Family / recipe ID | Generated geometry and principal controls | Geometric ports |
| --- | --- | --- |
| Strip / `strip` | Rectangular core; length, width and height | 2 |
| Slot / `slot` | Two rails; rail width and edge-to-edge slot gap | 2 across the rail pair |
| Rib / `rib` | Raised ridge and residual slab; slab width and thickness | 2 at the ridge ends |
| Linear taper / `taper` | Trapezoidal extrusion; input/output width and length | 2 |
| Circular bend / `bend` | Constant-width circular arc; centreline radius and turn angle | 2 |
| Cosine S-bend / `s_bend` | Smooth endpoint tangents; length and lateral offset | 2 |
| Directional coupler / `directional_coupler` | Parallel finite guides; gap and interaction length | 4 |
| Y branch / `y_branch` | Input trunk and symmetric curved branches; length and output spacing | 3 |
| 1×2 MMI / `mmi_1x2` | Multimode rectangle with tapered access guides; body length/width, pitch and access length | 3 |
| 2×2 MMI / `mmi_2x2` | Two input and two output access tapers around a multimode body | 4 |
| Tapered crossing / `crossing` | Orthogonal widened guide polygons; central and access widths | 4 |
| Ring and bus / `ring_bus` | Annular guide and straight bus; radius, gap and bus length | 2 on the bus |
| Add/drop ring / `ring_add_drop` | Annular guide between two buses | 4 on the buses |
| Racetrack / `racetrack` | Closed capsule-shaped guide with a hole and adjacent bus; radius and straight length | 2 on the bus |
| Mach–Zehnder / `mzi` | Symmetric split/merge paths with straight arms; length, separation and transition length | 2 |
| Bragg grating / `bragg` | Continuous guide plus periodic width corrugations; period, count, duty cycle and corrugation | 2 |
| Linear grating coupler / `grating_coupler` | Access taper, residual slab and raised grating teeth; period, count, duty cycle, aperture and slab thickness | 1 guided input |
| Nanobeam cavity / `nanobeam_cavity` | Two arrays of through holes around a defect; hole radius, period, mirror count and defect length | 2 |
| Passive DBR cavity / `dbr_cavity` | Two width-corrugated reflector sections around a plain spacer | 2 |
| Serpentine / `serpentine` | Alternating straight rows joined by semicircular turns; row count, pitch and straight length | 2 |
| Euler bend / `euler_bend` | Two symmetric clothoids; minimum curvature radius, turn angle, width and height | 2 |
| Strip-to-slot transition / `strip_slot_converter` | Single notched polygon; strip width, rail width, slot width, notch onset and length | 1 strip, 1 across both rails |
| 1×4 MMI / `mmi_1x4` | Rectangular multimode region, input taper and four output tapers; body dimensions, output pitch and access length | 5 |
| Apodized fan grating / `fan_grating` | Circular fan taper, residual slab and concentric elevated teeth; radius, opening angle, radial pitch, tooth count and endpoint duty cycles | 1 guided input |
| Tapered two-guide coupler / `tapered_coupler` | Two truncated tapers exchange widths while keeping a fixed edge gap; narrow/wide widths, interaction length and gap | 4 on straight access leads |
| Photonic-crystal line defect / `phc_line_defect` | Triangular array of through holes on either side of an omitted row; lattice constant, radius, columns, row pairs and strip width | 2 on strip leads |
| Pulley-coupled ring / `pulley_ring` | Annular core with a wrapped bus and tangent leads; core/bus widths, gap, radius, wrap angle and lead length | 2 on the bus |

Port counts describe named endpoints. A grating's radiated field is not
represented as a guided output port. The MZI is balanced passive geometry,
without a heater or phase calibration. The two cavity templates contain no
gain, pumping, carrier dynamics or lasing-threshold prediction. None of the
defaults claims an optimized splitting ratio, transmission, resonance, delay,
coupling efficiency, fabrication tolerance or foundry qualification.

## Geometry representation and accuracy

Version-2 geometry stores boxes, rings and extruded XY polygons, including
polygon holes. Component placement supports translation and rotation around
the z axis. General 3D rotations, Boolean CAD operations, sidewall angles,
process stacks and arbitrary material precedence are separate roadmap work.

Curved polygon boundaries are explicitly tessellated: circular bends use 48
angular intervals, S-bends use 64 centreline intervals, Y branches use 48,
racetracks use 32 intervals per semicircle, MZI arms use 96 centreline
intervals, serpentine turns use 12 intervals, and nanobeam holes use 20
vertices. The saved polygon is the design geometry; the library does not
claim a user-selected chord-error tolerance or solver convergence. A mesh
study must distinguish this geometry approximation from electromagnetic
discretization error.

The Euler bend has triangular curvature over arclength. Its minimum radius
occurs at the midpoint and curvature vanishes at both ends. Its centerline
length is twice the minimum radius times the turn angle (radians). The
generator samples 96 equal arclength intervals, integrates analytic tangent
angles by Simpson quadrature and offsets the centerline by half the guide width.
The pulley bus uses 48 angular intervals with tangent extensions. Gap
validation accounts for the chord sag on the inner bus edge.

The input fan and adjoining annular teeth use matching 36-segment angular
samples at their shared radial boundaries. Radial tooth fill varies linearly between the first and last duty
cycles; the residual slab occupies the lower part of the etch. The circular
fan is a design primitive. The published focusing workflow derives elliptical phase fronts and
uses concentric-circle approximations with a varying radial pitch. The line-defect slab uses at most 32 through holes
with 20 vertices per hole. Adjacent triangular-lattice rows are offset by half
a period, with row spacing √3/2 times the period. These approximations are
part of the exported geometry.

The Y junction, MZI junctions, crossing and width gratings deliberately overlap
same-material solids. Slab/ridge combinations meet at a shared horizontal
surface. Polygon holes remove material from their own extrusion; they are
not global etch operations that remove other overlapping objects.

## Material presets and custom data

The six catalog presets are lossless constant-index **approximations at
1.55 µm**, with provenance and limitations stored alongside their values:

| Preset ID | Index | Meaning |
| --- | --- | --- |
| `generic-core` | 2 | Original illustrative dielectric |
| `air` | 1 | Vacuum-like approximation; no atmospheric dispersion |
| `silica-reference` | 1.44 | Constant cladding value from the linked public telecom benchmark |
| `silicon-nitride-reference` | 2 | Nominal illustrative placeholder; no film/process data |
| `silicon-reference` | 3.48 | Constant core value from the linked public telecom benchmark |
| `iii-v-reference` | 3.2 | Passive geometric placeholder; no alloy, gain or quantum-well model |

The silicon/silica references use the constants stated in the official
[Tidy3D/Femwell mode-solver benchmark](https://www.flexcompute.com/tidy3d/examples/notebooks/ModeSolverFemwellBenchmark/).
No vendor material database or foundry dataset is bundled. A preset name does
not imply broadband, temperature-dependent or fabrication-specific validity.

The version-2 material model also preserves user-entered constant n/k or
tabulated wavelength/n/k data, including source notes. Evaluation interpolates
within the supplied table; it does not fit a causal dispersive time-domain
model. Preserving an absorbing or dispersive design material does not mean
either simulation backend has qualified support for it. Use material data
with appropriate redistribution rights and retain its source and valid range.

## API, reproducibility and checks

`GET /api/library` returns the recipe descriptors and material presets.
`POST /api/library/build` generates a standalone document;
`POST /api/library/place` inserts or replaces an instance in a layout.

Replacing an instance preserves omitted dimensions, placement axes and material
assignment. Updating dimensions regenerates that instance's solids with fresh
solid IDs while retaining its component ID. Manual per-solid edits explicitly
detach the recipe; undo restores it. Domain growth is an explicit placement
option, and invalid edits leave the previous layout unchanged.

The backend implementation can also be called directly from the repository:

```python
from backend.component_library import build_component

design = build_component(
    "ring_add_drop",
    parameters={"radius_um": 3.0, "length_um": 12.0, "gap_um": 0.25},
    placement={"x": 0.0, "y": 0.0, "z": 0.0, "rotation_deg": 30.0},
    material_id="silicon-reference",
)
geometry = design["geometry"]
assert geometry["schema_version"] == 2
```

Recipe generation is deterministic. Saved instances include
`generator_version`, resolved parameters, material reference and explicit
solid membership. Layout placement assigns distinct instance/object IDs so
multiple copies can coexist. JSON and design Python/notebook exports preserve
the explicit geometry, materials and instances. The taper field study has a
separate numerical Python/notebook export that reruns the selected device.

Run focused generator checks with:

```bash
.venv/bin/python -m unittest tests.test_component_library -v
```

The tests validate all 27 defaults through the shared geometry validator,
JSON round trips, rigid transforms, material assignment, independently sampled
inward/outward port apertures, empty slots/cavity holes, and rejection of
nonfinite or physically inconsistent parameters. These are geometry tests;
the physical qualification records for each executable study have separate scope.

The historical compatibility check requires a checkout containing the v0.7
commit. Reproduce it with `python scripts/audit_component_compatibility.py
--output /tmp/library-audit.json`. All 20 v0.7 designs were compared against
the original generator source at
commit `175a198`. For each family, the full geometry matched in three cases:
defaults, edited dimensions, and edited dimensions with translation, rotation
and a silicon material. All 27 defaults and one multi-component layout also
reproduced exactly in standalone Python and executed notebooks.
[60 archived-geometry comparisons](verification/v09-legacy-library-audit.json) ·
[56 script/notebook executions](verification/v09-library-exports.json) ·
[Vendor coverage and comparison requirements](VENDOR_COMPONENT_COVERAGE.md).
Each of the 27 families also passed a finite-layout Meep field calculation
with the specific settings and source listed in its [numerical record](verification/v09-family-fields.json).
All seven additional families also have real [FDTDX CPU](verification/v09-new-family-fields-fdtdx-cpu.json)
and [FDTDX NVIDIA](verification/v09-new-family-fields-fdtdx-gpu.json) field records.

The family taxonomy was informed by the public
[Tidy3D component tutorial](https://www.flexcompute.com/tidy3d/examples/notebooks/PICComponents/)
and [Ansys FDTD reference](https://optics.ansys.com/hc/en-us/articles/360033154434-FDTD-product-reference-manual).
All geometry recipes and tests in this release are original workbench code.
