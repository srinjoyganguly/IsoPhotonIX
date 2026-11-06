# Finite-height coupler and splitter ports

The workbench exposes the measured polygon-area sampling control on the NVIDIA route. The existing electric-field-aligned and original material grids remain available on the calibrated FDTDX CPU route.

The test devices have 500 × 250 nm silicon rails in oxide. Each source launches from a physical lead. The independently recorded straight reference uses the same source and receiver planes. The published matrices retain the electric-field port convention and the physical propagation phase.

## Material and propagation controls

The opt-in FDTDX material model `polygon-area` integrates the signed area under each original polygon edge over each staggered Yee dual cell, including the concave Y junction. It combines this area with the exact 250 nm vertical extent. The diagonal lateral material uses arithmetic averaging tangent to a sidewall, harmonic averaging normal to it, and the local coverage-gradient direction. Ex and Ey then combine horizontal layers arithmetically and Ez combines them harmonically. This is a diagonal interface approximation. A curved oblique wall can also have an off-diagonal permittivity. The legacy and `yee-aligned` models remain available for reproducing their published runs. [Implementation and formulas](../scripts/phase_polygon_material.py)

At 1550 nm and 16 cells/µm the transverse mode gives n_eff = 2.538205, while the independent time-stepped straight reference accumulates phase consistent with n_eff = 2.580392. Using the transverse mode directly misses 1.625 rad over 9.5 µm. The Cartesian Yee update's sine factors provide an independent test. For physical frequency f in cycles/µm, we solve the transverse section at f_eff = f sinc(f c Δt). The propagation angle per cell is 2 asin(π f_eff n_eff(f_eff) Δx). Across the three r16 wavelengths this predicts the measured straight phase to within 0.00030 rad. At r20 the corresponding central indices are 2.577520 transverse and 2.605398 propagated. The finite-stencil and transverse grid errors both affect physical phase. The independent 80-cell/µm MPB guide obtains 2.548773, versus 2.537127 at 32 cells/µm. At r16 the lower native transverse index partially offsets the positive longitudinal discretization shift. At r20 the native transverse index itself exceeds the 80-cell control. This explains why increasing the mesh from 16 to 20 can move a particular phase farther from an independent answer. The test uses the archived complex straight transfer and an independent local 2D eigenmode solve. [Propagation measurements](verification/v014-polygon-lead-dispersion.json) · [Schneider, numerical dispersion in the Yee grid](https://eecs.wsu.edu/~schneidj/ufdtd/chap7.pdf) · [Independent MPB transverse modes](verification/v011-multiport-mpb.json)


We repeated the transverse solve for six predeclared grid origins: two lateral positions (0 and 0.5 cell) and three vertical positions (0, 0.25 and 0.5 cell). Physical dimensions and indices stay fixed. The r16 index spans 2.53129–2.59174, and r20 spans 2.53814–2.57752. A half-cell vertical shift raises the r16 index by 0.05353 but lowers the r20 index by 0.03478. Extending every aperture edge by another 250 nm changes the translated result by less than 0.0000003 on these meshes. These are independent 2D checks, with each offset retained; the production device has its original coordinates. Interface treatment and refining the physical walls are needed as well as increasing nominal resolution. [All six grid registrations](verification/v014-polygon-lead-dispersion.json)

As a separate 2D study, local Tidy3D 2.11.2 used a grid registered to the physical core interfaces, double precision, and the open local mode solver with local subpixel averaging disabled. At 64/96/128/160 cells/µm it obtained 2.615591, 2.594225, 2.583478 and 2.577012. Independent MPB 1.12.0 obtained 2.549464, 2.550151 and 2.550468 at 96/128/160. The two raw 160-cell values differ by 1.04%. A linear-in-spacing estimate from the 96/128 Tidy3D pair predicts the held-out 160 result as 2.577030 (difference 0.000018). This trends toward the MPB sequence; neither extrapolated value is substituted for a measured 3D S coefficient. [Tidy3D transverse controls](verification/v014-polygon-tidy-lead.json) · [MPB transverse controls](verification/v014-polygon-mpb-lead.json)

We also reprojected the original GPU E/H phasors with modal profiles evaluated at f_eff, using the discrete propagation angle and the physical time staggering for collocation. First we reproduced the original projected amplitudes from the same E/H arrays. At r16, the apparent straight-guide backwave decreases from a maximum of about 0.00108 in amplitude to 0.000028. The normalized first-column coefficients of the two devices change by at most 0.000026 after straight-reference subtraction. The smaller residual improves measurement of the straight return. Physical phase still accumulates along the simulated leads. The published native matrices and their mesh gates continue to use the directly acquired convention. [Reprojection on the same fields](verification/v014-polygon-discrete-projection.json)

## Interpreting the checks

Forward qualification requires |Δphase| ≤ 0.10 rad over all three wavelengths, measured at the named physical planes. For returns we compare |ΔS| ≤ 0.01 and absolute power change ≤ 0.001. We omit phase where either |S| < 0.02. A separate forward comparison divides out the independently measured straight transfer on each mesh to quantify device-excess phase. It answers a different question from physical-plane phase. Commercial examples with different layouts and reference planes are unsuitable for direct circuit S comparisons. For these two geometries we provide independent same-layout Meep controls at 16 and 24 cells/µm, and reserve Lumerical/Tidy3D comparisons for the actual matching reference devices. [Commercial example scope](CALIBRATED_VENDOR_CHECKS.md)

At 16 cells/µm, each column used a 224 × 112 × 64 grid and 3,444 time steps per stage. The straight/device synchronized stepping times were 12.26/12.25 s for the coupler and 12.29/12.01 s for the splitter. At r20, the coupler used 3,136,000 cells, 4,305 steps and 27.34/30.89 s. Each timer encloses the native JAX stepping call and device completion. Setup, compilation, queue waiting, and result analysis lie outside it. Optional kernel activity collection failed in earlier attempts. These fresh runs explicitly disabled it and keep kernel-active timing empty. [Coupler 16](verification/v014-coupler-polygon-cuda16.json) · [Splitter 16](verification/v014-splitter-polygon-cuda16.json) · [Coupler 20](verification/v014-coupler-polygon-cuda20.json) · [Failed attempts](verification/v014-multiport-profiling-attempts.json)

For the coupler's 16→20 pair, maximum physical changes are 1.009 rad at the other end of the incident rail and 1.053 rad on the opposite rail. The respective straight-relative changes are 0.051 and 0.107 rad, while power changes reach 4.68 and 4.51 percentage points. The incident-port return changes by 0.02225 in complex amplitude, and the adjacent input-side return changes by 0.00635. [All wavelengths and phase gates](verification/v014-coupler-polygon-phase.json)



The independent full three-port GUI splitter run at r16 reproduces the separately acquired stem column exactly across all nine complex values. Its maximum |Sᵢⱼ−Sⱼᵢ| over all wavelengths and port pairs is 0.000803; the output-arm pair differs by 0.000000691. The largest measured singular value is 0.91577. These establish reciprocity and sampled passivity for the full r16 splitter. Mesh convergence still needs the independent refinements below. [Full three-port acquisition](verification/v014-splitter-polygon-matrix-gui16.json)


The 20→24 coupler pair has a maximum physical phase change of 1.695 rad. Both forward ports stay within 0.075 rad in the straight-relative check, while the incident-port return changes by 0.01240. Holding the positive z wall at the same fractional lattice position instead (16→24) still gives up to 0.666 rad on the through port. The JSON includes this supplementary equal-registration series and every adjacent mesh pair. Both comparisons retain the actual port-plane phase.

To reproduce one column from the repository root, use the registered FDTDX CUDA interpreter for the first command and its CPU environment for the independent transverse analysis. An existing run directory can be resumed with `--resume` only for the identical native plan. The report and NPZ contain one independent launch, validated physical planes, fields in the individual stage archives and the normalized column.

```bash
flock /tmp/photonics-v012-gpu.lock python -m scripts.measure_fdtdx_mesh_column \
  --device gpu --case coupler --resolution 24 --material-model polygon-area \
  --output runs/polygon-coupler-24
JAX_PLATFORMS=cpu python -m scripts.phase_lead_dispersion \
  --output lead.json runs/polygon-coupler-24
```


A separate comparison moves the ports to the actual ends of their uniform external leads. We verify that the original polygons have no varying edge along each removed section, all projected strip cross sections agree, and the measured straight propagation constant has the branch predicted by the independent Yee model. The coupler moves from |x|=4.75 to 4.0 µm; the splitter moves to 2.0 µm. This keeps modal power invariant and changes only reference phase. The 20→24 maximum forward phase change still reaches 1.423 rad for the coupler and 0.726 rad for the splitter. Their input-return complex changes are 0.01297 and 0.04024. Dividing out a 9.5 µm straight control is valuable for identifying dispersion, but only the actual uniform exterior lengths support this physical port relocation. [Coupler device-edge planes](verification/v014-coupler-uniform-lead-planes.json) · [Splitter device-edge planes](verification/v014-splitter-uniform-lead-planes.json)

The 28-cell/µm plan contains 8,605,184 cells (392 × 196 × 112). The existing reserve requires 12,424.375 MiB free at admission. The coupler had 12,292 MiB available and the splitter 12,324 MiB, so neither produced a new field or timing at that mesh. Separate global-memory samples during the r24 production runs provide observed totals for this laptop, including graphics and other processes. [Admission attempts](verification/v014-multiport-r28-admission.json) · [Observed board memory](verification/v014-multiport-memory-observations.json)

A higher-resolution route on this GPU could use mirror symmetry only after a full/half complex-column comparison at r24. A centre-launched symmetric splitter permits a single vector-parity sector. A lower-rail input to the coupler is asymmetric and requires both parity sectors and their coherent sum. Reproducing energy, weak return and physical phase on the existing full-domain fields would qualify that reduction before attempting a finer mesh.

At r24, reversing the forward launch gives maximum complex reciprocity residuals of 0.0000134 for the coupler’s through rail and 0.000579 for the splitter stem/arm pair. The separately excited adjacent-input weak coupler pair agrees to 0.000000117 in complex amplitude and 0.000000000267 in power; its phase stays below the reporting floor at all three wavelengths. These compare independent field acquisitions, including separate straight references for the reverse launches. [Coupler reciprocity pairs](verification/v014-coupler-polygon-reciprocity.json) · [Splitter reciprocity pair](verification/v014-splitter-polygon-reciprocity.json)

The six original mesh columns and three reverse columns contain 18 distinct fresh CUDA acquisitions. Together they executed 357,887,846,400 cell-steps in 746.51 synchronized solver seconds (12 min 26.51 s). The sum excludes setup, queue wait, CPU mode solves, rejected meshes and the separate GUI execution.

At the identical 24-cell/µm physical layouts and port planes, the independent Meep control differs from the NVIDIA coupler’s forward powers by at most 0.566 percentage points and its forward phases by at most 0.189 rad. For the splitter these maxima are 1.556 percentage points and 0.0795 rad. The largest input-side complex discrepancies are 0.00325 (coupler) and 0.0315 (splitter). Each comparison uses independently acquired field monitors at the same three wavelengths. Neither finite mesh supplies a continuum truth value. [Coupler same-layout comparison](verification/v014-coupler-polygon-cuda24.json) · [Splitter same-layout comparison](verification/v014-splitter-polygon-cuda24.json)

## Mesh and execution summary

Each number below is a maximum across all three sampled wavelengths and the indicated output group. Forward means either right-hand rail. Return means the launch side of the circuit. Power changes are absolute percentage points.

| Component | Mesh pair (cells/µm) | Forward power | Physical phase | Straight-relative phase | Return power | Complex return amplitude |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Coupler | 16 → 20 | 4.677 | 1.053 rad | 0.107 rad | 0.0195 | 0.02225 |
| Coupler | 20 → 24 | 2.329 | 1.695 rad | 0.074 rad | 0.0174 | 0.01240 |
| Splitter | 16 → 20 | 0.463 | 1.024 rad | 0.057 rad | 0.1398 | 0.05388 |
| Splitter | 20 → 24 | 0.210 | 1.722 rad | 0.009 rad | 0.0458 | 0.10211 |

| Component | Cells/µm | Actual cells | Steps, straight / device | Synchronized seconds, straight / device | Acquisition |
| --- | ---: | ---: | --- | --- | --- |
| Coupler | 16 | 1,605,632 | 3,444 / 3,444 | 12.26 / 12.25 | [Receipt](verification/v014-coupler-polygon-cuda16.json) |
| Coupler | 20 | 3,136,000 | 4,305 / 4,305 | 27.34 / 30.89 | [Receipt](verification/v014-coupler-polygon-cuda20.json) |
| Coupler | 24 | 5,419,008 | 5,166 / 5,166 | 54.82 / 59.41 | [Receipt](verification/v014-coupler-polygon-cuda24.json) |
| Splitter | 16 | 1,605,632 | 3,444 / 3,444 | 12.29 / 12.01 | [Receipt](verification/v014-splitter-polygon-cuda16.json) |
| Splitter | 20 | 3,136,000 | 4,305 / 4,305 | 26.43 / 28.99 | [Receipt](verification/v014-splitter-polygon-cuda20.json) |
| Splitter | 24 | 5,419,008 | 5,166 / 5,166 | 58.50 / 60.63 | [Receipt](verification/v014-splitter-polygon-cuda24.json) |
