# Numerical validation and commercial comparisons

**Status: experimental real 3D simulation; not commercially validated.**
Meep CPU and FDTDX CPU/NVIDIA GPU produce and export actual electromagnetic
fields. The present evidence establishes that the implementation runs, represents
the supported geometry, and responds to physical changes. It does not establish
agreement with Lumerical, Tidy3D or COMSOL, nor an engineering accuracy guarantee.

Exact equality is not a useful acceptance criterion across different numerical
methods. A credible claim names the physical model, measured quantity, reference,
convergence evidence and error tolerance. Ansys's own FDTD guidance describes
finite numerical error and separate contributions from mesh, boundaries, source,
material fitting and monitor sampling. [Ansys convergence guidance](https://optics.ansys.com/hc/en-us/articles/360034915833-Convergence-testing-process-for-FDTD-simulations).

## Evidence currently available

| Check | Current evidence | What it establishes |
| --- | --- | --- |
| Full 3D geometry and fields | [Meep](verification/meep-3d.json), [FDTDX](verification/fdtdx-3d.json) | Finite Ex/Ey/Ez volumes, material probes, downstream signal, variation along z, response to changed core height |
| Explicit NVIDIA execution | [FDTDX](verification/fdtdx-3d.json) | Actual CUDA device execution; unavailable GPU does not trigger a CPU fallback |
| Same FDTDX case on CPU/GPU | Displayed E relative L2 difference approximately 1.16 × 10⁻⁶; raw E approximately 1.40 × 10⁻⁶ | Consistency between two execution devices for one implementation and case |
| Meep spatial refinement | Three resolutions and one thicker-PML experiment below | Measured sensitivity of one final-time field diagnostic |
| Analytical accuracy | Not completed for this 3D workflow | No passed analytical benchmark claimed |
| Independent commercial comparison | Not completed | No passed Lumerical/Tidy3D/COMSOL comparison claimed |
| Normalized flux, modal power or S-parameters | Not implemented for this 3D workflow | No transmission, loss or modal-purity claim |

CPU/GPU agreement can reproduce the same modeling or discretization error on
both devices. A nonzero plausible field, a smooth image, passing software tests,
or a short runtime also does not establish external numerical correctness.

## Independent public mode benchmarks

The workbench now reproduces four explicit input contracts from three public
vendor examples using local MPB 1.12.0. These are cross-sectional eigenmode
calculations, separate from the driven FDTD studies above. At 256 cells/µm:

| Case / reference | Local effective index | Relative discrepancy |
| --- | ---: | ---: |
| COMSOL slab / analytic TE0 equation | 1.4029088674 | 0.0001931% |
| Tidy3D silicon strip / published solver scalar | 2.4486493810 | 0.0126772% |
| Lumerical slab / article's core index 3.48 | 2.8512137026 | 0.104053% |
| Lumerical slab / associated script's core index 3.47668 | 2.8479286190 | 0.0112835% |

All four resolutions and enlarged-domain runs are retained in the
[numerical report](verification/public-mode-benchmarks.json). The closer Tidy3D
value at the coarser 128-cell grid was not selected over the finer result.
Lumerical's prose and script specify different material indices; neither input
was fitted to the printed reference. Its mode mesh/version are unspecified,
so neither variant establishes an exactly matched commercial solve.
The [benchmark guide](PUBLIC_BENCHMARKS.md) records source provenance, limitations
and reproduction commands. No scalar result certifies a whole solver or device.

An additional [six-run propagation check](verification/public-slab-propagation.json)
uses the unchanged 2D product adapters for COMSOL's slab model. At 45 cells/µm,
projecting actual late-time phasors onto a fixed analytic transverse TE0 profile
gives effective-index discrepancies of 0.06324% for Meep CPU and 0.06016% for
FDTDX NVIDIA GPU. The template uses the analytic transverse reference; the
longitudinal phase is independently measured from the saved fields. Larger
centerline discrepancies, all grid points and window sensitivities remain in
the report. This is a bounded propagation diagnostic, not calibrated modal
power or a qualification of the finite-height 3D adapters.

The current **Same-study comparison** panel reports differences between saved
runs from one solver only after matching physical inputs, recorded versions,
source, sampling, coordinates, material arrays and final time. It permits a
different execution device but performs no resampling, phase fitting or
amplitude normalization and applies no accuracy pass threshold. These are
repeatability metrics on saved field samples, not the independent comparison
protocol below. [Usage and metric definitions](3D_SIMULATION.md#compare-saved-runs).

The reproducible model is documented in [3D_SIMULATION.md](3D_SIMULATION.md):
8 × 4 × 4 µm domain; centred 8 × 0.75 × 0.5 µm core of index 2; index-1.44
substrate below z = −0.25 µm; index-1 background; 1.55 µm free-space wavelength;
0.75 µm PML on every face. The qualification runs used Meep 1.35.0 and FDTDX
0.6.2/JAX 0.10.2. Full versions, geometry translations and checksums are in the
linked evidence.

## What the Meep refinement actually shows

The recorded diagnostic compares final Ey on the centreline y = z = 0. Samples
are interpolated to 129 common x coordinates from −1.5 to 2.5 µm. All compared
runs reached 60.04153713566737 fs. Relative L2 is
`norm(E_A - E_B) / norm(E_B)`, with the second case as denominator.

| Comparison | Relative L2 discrepancy |
| --- | ---: |
| 16 versus 20 cells/µm | 8.21% |
| 20 versus 24 cells/µm | 1.64% |
| 16 versus 24 cells/µm | 9.64% |
| PML 0.75 versus 1.00 µm, at 16 cells/µm | 0.00852% |

The thicker-PML experiment enlarged every domain dimension by 0.5 µm. It kept
the PML inner faces, source position and material interface fixed, so it tests
PML thickness without moving the interior boundary. It does not separately test
clearance between the guide and PML. The small PML discrepancy applies only to
this diagnostic, configuration and final time; it is not a reflection bound.

The default grid is visibly sensitive to refinement. These three grids do not
prove an asymptotic convergence order or bound the error relative to the exact
solution. The 24-cells/µm direct-adapter reference has 1,769,472 cells, exceeding
the application's 1,200,000-cell interactive limit; it is qualification evidence,
not an available GUI setting. Longer-duration and independent domain-clearance
studies remain necessary. Meep recommends checking resolution, runtime and PML
thickness when assessing convergence. [Meep FAQ](https://meep.readthedocs.io/en/latest/FAQ/).

## Why the present solver images cannot be compared as absolute fields

Meep injects an electric-current sheet. FDTDX injects a directed +x electric and
magnetic plane. Both span the core cross-section and use a four-period startup,
but neither is a solved waveguide eigenmode. Their source strengths have not
been calibrated to a common incident power. Different forward/backward and
radiating content can remain even after normalizing a peak value.

The solvers also use different material treatment: Meep subpixel smoothing
versus FDTDX's voxelized boxes. Exports describe actual translated bounds and
field sampling. Meep evolves double-precision fields and exports float32;
FDTDX uses float32. Their final times differ slightly because of their time
steps. The displayed arrays contain instantaneous real electric fields, not
complex steady-state amplitudes or time-averaged power. Meep distinguishes
steady-state response from transient fields and documents DFT or frequency-domain
methods for obtaining it. [Meep steady-state guidance](https://meep.readthedocs.io/en/latest/FAQ/#how-do-i-compute-the-steady-state-fields).

A color-map match or peak-amplitude ratio would therefore be misleading. This
release does not derive power from `abs(Ey)**2`; power measurements require the
appropriate E/H fields, normalization and integration.

## Protocol for a defensible comparison

The following is planned work, not a list of passed tests. It applies to an
independent open-source reference as well as a commercial export.

1. **Freeze a benchmark contract.** Version one geometry with axis conventions,
   physical dimensions, material permittivities and dispersion, wavelength,
   source definition, incident-mode identity, measurement planes, boundaries
   and units. Save the exact input for every solver. Initially use lossless,
   nondispersive, straight dielectric guides so material databases do not add
   an uncontrolled difference.
2. **Add measured quantities.** Implement complex frequency-domain E/H monitors
   and incident-power normalization before comparing power. Use either the same
   specified current distribution, or matched eigenmode ports normalized to the
   same incident power. Record the normalization/reference run and its source
   and monitor positions. Require a measured steady-state or decay criterion;
   the current 60 fs duration is not such a criterion.
3. **Establish independent elementary checks.** Verify homogeneous propagation
   and a planar dielectric interface against analytic results, then a slab mode
   against its dispersion equation. These require dedicated benchmark studies;
   passing them alone does not qualify the finite-height 3D adapter.
4. **Converge each solver independently.** Sweep grid spacing, PML thickness,
   physical clearance and runtime separately, retaining actual rounded geometry
   and sampled material information. Recheck their interaction at the fine grid.
   For frequency-domain FEM also refine element order/mesh and solver tolerance.
   Preserve all points, including non-monotonic results; do not select a single
   grid merely because it agrees with the reference.
5. **Compare the same observable.** Start with propagation constant/effective
   index and normalized forward/backward power. Compare complex vector fields
   only on a declared common interior region, excluding source and PML cells.
   Interpolate once to common physical coordinates and retain original arrays.
   Declare phasor sign and a single global phase reference. Any shape-only
   amplitude normalization must be labeled and cannot support an absolute-field
   or power claim. Do not adjust values point by point to improve agreement.
6. **Record discrepancy and limits.** For nonzero fields, report weighted
   relative L2 and maximum absolute discrepancy; for power fractions, report
   absolute differences, particularly near zero. Track energy balance including
   radiation and absorption where applicable. Publish all solver versions,
   devices, settings, actual coordinates, errors, evidence files and checksums.

Candidate gates for the first matched, lossless straight-guide benchmark are
below. They are project targets to freeze before execution, not universal
accuracy standards or results already achieved. A new observable or physical
regime needs its own error budget.

| Quantity | Proposed agreement gate | Required qualification |
| --- | --- | --- |
| Effective index | Relative difference ≤ 10⁻³ | Same tracked mode and phase convention; each solver's final refinement change ≤ one quarter of this tolerance |
| Normalized power fraction | Absolute difference ≤ 0.01 | Matched excitation and reference power; mesh, boundary and runtime sensitivities each ≤ 0.0025 |
| Complex vector field | Weighted relative L2 ≤ 0.05 | Common region/coordinates; declared phase and amplitude normalization; each sensitivity ≤ 0.0125 |

These sensitivity checks are necessary evidence, not rigorous error bounds.
Failure should produce a failed benchmark report and investigation, not a
relaxed threshold after seeing the answer. A reported success must name the
specific benchmark and tested range; it must not become a blanket claim of
commercial parity.

## Public reference search

References below were inspected on 1 November 2026. No matching public driven
3D dataset for the workbench's default guide was established. No commercial or
cloud solve was launched.

| Official reference | Inspected conditions/data | Use and limitation |
| --- | --- | --- |
| [Tidy3D versus Femwell mode benchmark](https://www.flexcompute.com/tidy3d/examples/notebooks/ModeSolverFemwellBenchmark/) | Straight strip: 0.50 × 0.22 µm, core index 3.48, uniform cladding 1.44, wavelength 1.55 µm; cross-section domain 4.3 × 3.02 µm. The public HTML includes numerical effective-index output: 2.438241 at `min_steps_per_wvl=10` and 2.448339 at 100, with intermediate values. | The cross-sectional mode is now reproduced by the local MPB benchmark. It is not a driven 3D E/H dataset, has different materials/geometry, and supplies no matching source-power normalization for our final-time volumes. |
| [COMSOL dielectric slab verification](https://doc.comsol.com/6.3/doc/com.comsol.help.models.woptics.dielectric_slab_waveguide/dielectric_slab_waveguide.html) | 1 µm slab, core/cladding indices 1.5/1.0, wavelength 1.55 µm; analytic equations and numerical-port model instructions. | The invariant TE0 slab mode is now checked by the local MPB benchmark. It does not validate a finite-width, finite-height guide on a substrate. |
| [Ansys FDTD convergence example](https://optics.ansys.com/hc/en-us/articles/360034915833-Convergence-testing-process-for-FDTD-simulations) | A 2D gold-nanowire scattering example with scripts, project and [numeric Mie-theory attachment](https://optics.ansys.com/hc/en-us/article_attachments/360049569393). The text attachment was inspected and contains numeric rows. | Concrete analytic-reference workflow, but unrelated geometry and dispersive metal physics; not evidence of our guide's accuracy. The attachment is theory data, not a matching commercial 3D field export. |

Tidy3D's page links a
[downloadable notebook](https://docs.flexcompute.com/projects/tidy3d/en/latest/_sources/notebooks/ModeSolverFemwellBenchmark.ipynb)
and references `misc/femwell.csv`. The notebook endpoint returned HTTP 403 in
this environment, so its downloaded contents and the separate CSV were not
verified. The inspected numeric values above came from the public HTML. Do not
represent that as a downloaded or locally reproduced benchmark. Its local
subpixel mode calculation also requires matching `tidy3d`/`tidy3d-extras` versions.

For an external finite-3D comparison, retain a redistributable reference export
with numerical arrays, full settings, normalization and a convergence report.
Local analysis of that export can remain entirely offline. A proprietary
project file or screenshot without these details is insufficient. This document
does not add a runtime commercial dependency or authorize uploading user models.
