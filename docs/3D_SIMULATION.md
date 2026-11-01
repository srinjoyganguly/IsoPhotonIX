# Run and inspect a finite-height 3D waveguide

This milestone connects a supported geometry from the 3D editor to real Meep
CPU and FDTDX CPU/NVIDIA GPU calculations. It evolves Maxwell fields in all
three spatial dimensions and saves Ex, Ey, Ez and the actual material volume.
There is no invariant or periodic z direction in this study. The earlier 2D
slab workflow remains separate.

## Complete the local journey

1. Start the application using [SETUP.md](SETUP.md) and open **3D design**.
2. In the **Finite-height waveguide study** panel, choose **Open runnable 3D
   waveguide**. This replaces the draft with the supported example; Undo retains
   the editor's earlier design.
3. Select the core in the scene and edit its width or finite height. Numeric
   edits commit on Enter or blur. After changing height or the domain, choose
   **Fit substrate to core** and review validation. Domain X and core length
   must agree.
4. Review wavelength, background index, resolution, duration and PML thickness.
   The study displays a bounded grid estimate and a planning memory allowance;
   this allowance is not a measured peak-memory guarantee.
5. Explicitly select an available **Meep CPU**, **FDTDX CPU** or **FDTDX NVIDIA
   GPU** target, then **Run 3D simulation**. GPU initialization or execution
   failure never starts a CPU retry. Only one simulation runs at a time; an
   active run can be cancelled.
6. Inspect the computed final snapshot using **Plane**, **Component**, and
   **Slice position**. XY cuts vary z, XZ cuts vary y, and YZ cuts vary x.
   Each slice reports the actual saved coordinate and simulated time. The
   field color scale is fixed across slices of the selected component.
   **Material grid** shows the corresponding relative permittivity samples.
7. Use **Export 3D results .zip** for numerical arrays, the exact study,
   execution metadata, logs, checksums and an offline run passport. **Save run
   configuration** retains the immutable settings of that run. Later geometry
   edits do not relabel earlier results. **Recent 3D runs** reopens saved jobs.
8. To compare repeated runs, open **Same-study comparison**, select a saved
   reference from the same solver, then **Compare saved runs**. The service
   rejects incompatible studies; export accepted metrics with **Export
   comparison JSON**. This checks repeatability, not commercial accuracy.

The 3D viewport remains a geometry view. Field slices below it come from the
saved solver volume; the application does not paint an invented field onto the
preview. This increment records final-time 3D fields, not a 3D time animation.

## Supported physical model

The runnable example has a centred rectangular dielectric core on a substrate:

| Quantity | Default |
| --- | --- |
| Domain X × Y × Z | 8 × 4 × 4 µm |
| Core X × Y × Z | 8 × 0.75 × 0.5 µm |
| Core / substrate / background index | 2.0 / 1.44 / 1.0 |
| Substrate | Full domain X/Y; z from −2 to −0.25 µm |
| Free-space wavelength | 1.55 µm |
| Resolution | 16 cells/µm; Δ = 0.0625 µm |
| Computational grid | 128 × 64 × 64 cells |
| Requested duration | 60 fs; actual final time is solver-step dependent |
| Absorbing boundary | 0.75 µm PML on all six faces |
| Excitation | Ey-polarized finite sheet/plane at x = −2.75 µm |

All materials are real, isotropic, lossless and nondispersive. The core spans
both x PML regions so the straight guide continues to the domain faces. Its
finite transverse dimensions are essential: this is not the earlier thin
periodic extrusion. The source spans the core's y/z cross-section and ramps a
continuous sinusoid over four periods. Meep injects an electric-current sheet;
FDTDX injects a directed +x E/H plane. Neither is a solved waveguide eigenmode,
and their amplitudes are not cross-solver calibrated.

Only a centred, unrotated box with ID `core` and an optional flush box with ID
`substrate` are supported by this study. The substrate fills domain X/Y from
the bottom z face to the core bottom. Rings, couplers, cylinders, extra solids,
rotated cores, and arbitrary scene imports remain editable but cannot run as
this study. Visibility is presentation state: hiding a solid does not remove
its material from the simulation.

Validation enforces material-index ordering, at least four cells across each
core dimension, ten cells per wavelength in the highest-index material, eight
cells per PML, and clearance from transverse PML. The current application caps
the grid at 1,200,000 cells and the estimated grid × step count at 1,200,000,000.
The isolated worker retains the five-minute execution limit. These bounds make
the local milestone manageable; they do not establish convergence.

## Sampling and discretization

Meep uses subpixel material averaging. Its `get_array` output interpolates
staggered field components to cell centres; `Dielectric` reports an effective
scalar permittivity at interfaces. The canonical export is 64 × 64 × 128 in
`[z,y,x]` order.

FDTDX rounds boxes and PML to its voxel grid. The adapter records requested and
actual box bounds and checks the initialized material array against that
translation before running. One shared integer interface keeps substrate and
core touching after rounding. FDTDX's raw fields occupy staggered Yee locations:
Ex is offset `(0.5,0,0)`, Ey `(0,0.5,0)` and Ez `(0,0,0.5)` cells from each voxel
origin. To display all components on one coordinate grid, the adapter averages
four neighbouring samples onto cell centres. It omits the positive outermost
cell on x/y/z so no edge values are extrapolated. Consequently the canonical
display export is 63 × 63 × 127, while the simulated grid remains 128 × 64 × 64.
The full raw fields and material grid are also retained.

These are explicit spatial sampling operations on actual numerical output,
not temporal interpolation. The exported metadata describes them. Scalar
permittivity grids near interfaces and source conventions differ between
solvers, so matching colors or peak amplitudes is not an accuracy comparison.
The saved domain includes absorbing PML cells.

## Configuration and command line

An executable study is a version 2 JSON document with
`study_type: "waveguide-3d"`. It contains the exact versioned geometry plus
wavelength, background index, grid resolution, duration, PML, solver and device.
**Export study JSON** saves this document. **Export geometry** saves only a
geometry draft and cannot replace an executable study configuration. Version 1
slab configurations still use the original 2D adapter.

From the repository root, after exporting a study or retrieving the default
from the running local service:

```bash
curl -fsS http://127.0.0.1:8000/api/example3d -o waveguide-3d-study.json
.venv/bin/python scripts/run_simulation.py waveguide-3d-study.json --output runs/my-first-3d-run
```

The default API example explicitly selects Meep CPU. To reproduce a FDTDX GPU
run, use the configuration exported from that selected target, or deliberately
set `solver` to `fdtdx` and `device` to `gpu` in a copy of the study. The output
directory must be new. Existing `MEEP_PYTHON` and `FDTDX_PYTHON` interpreter
overrides still apply. This milestone reuses the documented solver environments;
it is not the future pip-managed installer.

## Local API and exported arrays

| Endpoint | Purpose |
| --- | --- |
| `GET /api/example3d` | Canonical version 2 study |
| `POST /api/validate3d` | Validate a study; return grid estimates and assumptions |
| `POST /api/jobs3d` | Submit that study with an explicit available target |
| `GET /api/jobs/{id}` | Status, immutable configuration and process log |
| `POST /api/jobs/{id}/cancel` | Stop the active worker; partial arrays are not completed results |
| `GET /api/jobs/{id}/volume` | Saved coordinate axes, components, peaks and metadata |
| `GET /api/jobs/{id}/slice?plane=xz&index=32&component=Ey` | Exact saved field/material slice with coordinates |
| `GET /api/jobs/{id}/passport` | Inspectable provenance and file checks |
| `GET /api/jobs/{id}/export` | ZIP with numerical arrays and reproduction material |
| `POST /api/compare3d` | Compare compatible completed runs using `reference_run_id` and `candidate_run_id` |

Slice planes are `xy`, `xz` or `yz`; components are `Ex`, `Ey` or `Ez`. An
omitted index selects the nearest saved coordinate to zero. An invalid index or
component is rejected. The old `/field` endpoint intentionally rejects 3D runs
instead of flattening their data into a 2D result.

`arrays.npz` contains `fields[component,z,y,x]` in Ex/Ey/Ez order,
`field[z,y,x]` equal to the Ey component, `epsilon[z,y,x]`, and increasing
`x_um`, `y_um`, `z_um` coordinates. FDTDX additionally preserves
`raw_fields[component,z,y,x]`, `raw_epsilon[z,y,x]`, and `raw_x_um`, `raw_y_um`,
`raw_z_um` voxel-origin coordinates. Add the component offset multiplied by
`grid_spacing_um` from metadata to locate a raw Yee sample.

For example, after extracting a result ZIP:

```python
import numpy as np

with np.load("arrays.npz", allow_pickle=False) as result:
    ey = result["fields"][1]
    zi = np.argmin(abs(result["z_um"]))
    xy_field = ey[zi, :, :]
    xy_material = result["epsilon"][zi, :, :]
    print("Actual slice z (µm):", result["z_um"][zi])
```

## Compare saved runs

The comparison requires the same solver, physical geometry, materials,
wavelength, resolution, requested duration and PML settings. Recorded solver
and runtime versions, source, boundaries and sampling conventions must match.
Saved coordinate axes and permittivity arrays must be identical; actual final
times must agree within 10⁻⁶ fs. Only execution device and presentation settings
may differ. A FDTDX CPU/GPU pair can therefore qualify, while a Meep/FDTDX pair,
different grids, or an edited core cannot. Missing or incompatible evidence is
an error, not a comparison result.

Metrics use the saved common-centre Ex/Ey/Ez arrays over the entire exported
volume, including source and PML regions. Each component reports maximum
absolute difference and relative L2,
`norm(candidate - reference) / norm(reference)`; the combined value uses all
three components. A zero reference norm produces an undefined relative value
(`null` in JSON). The tool does no resampling, interpolation between runs,
phase fitting or amplitude normalization, and assigns no accuracy pass/fail
threshold. FDTDX raw Yee arrays remain in each run export but are not the arrays
used by this comparison panel.

A small difference demonstrates repeatability under these conditions; it can
still reproduce a shared modeling error. See [VALIDATION.md](VALIDATION.md) for
the measured sensitivity, current evidence and requirements for an independent
accuracy or commercial-solver comparison.

## Verification and limits

[Meep evidence](verification/meep-3d.json) records actual 3D CPU runs, geometry
and downstream-field checks, finite-height response, and grid/PML sensitivity
experiments. [FDTDX evidence](verification/fdtdx-3d.json) records actual CPU and
RTX 3080 Ti Laptop GPU runs inside WSL, exact material-grid checks, raw-to-display
sampling checks, an edited-height response, and explicit failure when GPU
visibility is removed. CPU/GPU relative L2 differences were approximately
1.4 × 10⁻⁶ for raw E and 1.2 × 10⁻⁶ for displayed E on the canonical case.
This is numerical implementation consistency, not a convergence certificate.
The [validation report](VALIDATION.md) separates those checks from unfinished
analytical and commercial benchmarks.

Grid, PML and domain sensitivity remain necessary for quantitative engineering.
This release does not claim normalized transmission, S-parameters, mode purity,
converged device performance, or equality between Meep and FDTDX source
amplitudes. The tests reused existing qualified environments and do not establish
fresh installation on another machine. Planned mode studies, broader geometry,
desktop packaging and managed installation remain in [ROADMAP.md](ROADMAP.md).
