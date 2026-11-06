# Finite-height coupler and splitter ports

The test devices have 500 × 250 nm silicon rails in oxide. Each source launches from a physical lead. The independently recorded straight reference uses the same source and receiver planes. The published matrices retain the electric-field port convention and the physical propagation phase.

## Material and propagation controls

The opt-in FDTDX material model `polygon-area` integrates the signed area under each original polygon edge over each staggered Yee dual cell, including the concave Y junction. It combines this area with the exact 250 nm vertical extent. The diagonal lateral material uses arithmetic averaging tangent to a sidewall, harmonic averaging normal to it, and the local coverage-gradient direction. Ex and Ey then combine horizontal layers arithmetically and Ez combines them harmonically. This is a diagonal interface approximation. A curved oblique wall can also have an off-diagonal permittivity. The legacy and `yee-aligned` models remain available for reproducing their published runs. [Implementation and formulas](../scripts/phase_polygon_material.py)

At 1550 nm and 16 cells/µm the transverse mode gives n_eff = 2.538205, while the independent time-stepped straight reference accumulates phase consistent with n_eff = 2.580392. Using the transverse mode directly misses 1.625 rad over 9.5 µm. The Cartesian Yee update's sine factors provide an independent test. For physical frequency f in cycles/µm, we solve the transverse section at f_eff = f sinc(f c Δt). The propagation angle per cell is 2 asin(π f_eff n_eff(f_eff) Δx). Across the three r16 wavelengths this predicts the measured straight phase to within 0.00030 rad. At r20 the corresponding central indices are 2.577520 transverse and 2.605398 propagated. The finite-stencil and transverse grid errors both affect physical phase. The independent 80-cell/µm MPB guide obtains 2.548773, versus 2.537127 at 32 cells/µm. At r16 the lower native transverse index partially offsets the positive longitudinal discretization shift. At r20 the native transverse index itself exceeds the 80-cell control. This explains why increasing the mesh from 16 to 20 can move a particular phase farther from an independent answer. The test uses the archived complex straight transfer and an independent local 2D eigenmode solve. [Propagation measurements](verification/v014-polygon-lead-dispersion.json) · [Schneider, numerical dispersion in the Yee grid](https://eecs.wsu.edu/~schneidj/ufdtd/chap7.pdf) · [Independent MPB transverse modes](verification/v011-multiport-mpb.json)

We also reprojected the original GPU E/H phasors with modal profiles evaluated at f_eff, using the discrete propagation angle and the physical time staggering for collocation. First we reproduced the original projected amplitudes from the same E/H arrays. At r16, the apparent straight-guide backwave decreases from a maximum of about 0.00108 in amplitude to 0.000028. The normalized first-column coefficients of the two devices change by at most 0.000026 after straight-reference subtraction. The smaller residual improves measurement of the straight return. Physical phase still accumulates along the simulated leads. The published native matrices and their mesh gates continue to use the directly acquired convention. [Reprojection on the same fields](verification/v014-polygon-discrete-projection.json)

## Interpreting the checks

Forward qualification requires |Δphase| ≤ 0.10 rad over all three wavelengths, measured at the named physical planes. For returns we compare |ΔS| ≤ 0.01 and absolute power change ≤ 0.001. We omit phase where either |S| < 0.02. A separate forward comparison divides out the independently measured straight transfer on each mesh to quantify device-excess phase. It answers a different question from physical-plane phase. Commercial examples with different layouts and reference planes are unsuitable for direct circuit S comparisons. For these two geometries we provide independent same-layout Meep controls at 16 and 24 cells/µm, and reserve Lumerical/Tidy3D comparisons for the actual matching reference devices. [Commercial example scope](CALIBRATED_VENDOR_CHECKS.md)

At 16 cells/µm, each column used a 224 × 112 × 64 grid and 3,444 time steps per stage. The straight/device synchronized stepping times were 12.26/12.25 s for the coupler and 12.29/12.01 s for the splitter. At r20, the coupler used 3,136,000 cells, 4,305 steps and 27.34/30.89 s. Each timer encloses the native JAX stepping call and device completion. Setup, compilation, queue waiting, and result analysis lie outside it. Optional kernel activity collection failed in earlier attempts. These fresh runs explicitly disabled it and keep kernel-active timing empty. [Coupler 16](verification/v014-coupler-polygon-cuda16.json) · [Splitter 16](verification/v014-splitter-polygon-cuda16.json) · [Coupler 20](verification/v014-coupler-polygon-cuda20.json) · [Failed attempts](verification/v014-multiport-profiling-attempts.json)

For the coupler's 16→20 pair, maximum physical changes are 1.009 rad at the other end of the incident rail and 1.053 rad on the opposite rail. The respective straight-relative changes are 0.051 and 0.107 rad, while power changes reach 4.68 and 4.51 percentage points. The incident-port return changes by 0.02225 in complex amplitude, and the adjacent input-side return changes by 0.00635. [All wavelengths and phase gates](verification/v014-coupler-polygon-phase.json)


## Mesh and execution summary

Each number below is a maximum across all three sampled wavelengths and the indicated output group. Forward means either right-hand rail. Return means the launch side of the circuit. Power changes are absolute percentage points.

| Component | Mesh pair (cells/µm) | Forward power | Physical phase | Straight-relative phase | Return power | Complex return amplitude |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Coupler | 16 → 20 | 4.677 | 1.053 rad | 0.107 rad | 0.0195 | 0.02225 |

| Component | Cells/µm | Actual cells | Steps, straight / device | Synchronized seconds, straight / device | Acquisition |
| --- | ---: | ---: | --- | --- | --- |
| Coupler | 16 | 1,605,632 | 3,444 / 3,444 | 12.26 / 12.25 | [Receipt](verification/v014-coupler-polygon-cuda16.json) |
| Coupler | 20 | 3,136,000 | 4,305 / 4,305 | 27.34 / 30.89 | [Receipt](verification/v014-coupler-polygon-cuda20.json) |
| Splitter | 16 | 1,605,632 | 3,444 / 3,444 | 12.29 / 12.01 | [Receipt](verification/v014-splitter-polygon-cuda16.json) |
