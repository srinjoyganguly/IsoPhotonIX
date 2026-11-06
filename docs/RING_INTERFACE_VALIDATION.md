# How much of the ring error comes from the geometry grid?

The `exact-area` ring option computes the area of each straight guide and circular annulus inside each lateral Yee cell. A square's intersection with each disk is integrated analytically. Subtracting the inner disk gives the ring. The original `legacy` setting counts occupied points on a 6 × 6 subgrid. Both use the same specified geometry, component-centered sampling, vertical material rule, indices, source, wavelength grid, aperture and observation time. The published-geometry comparisons use independent straight-reference and device runs on NVIDIA. [Worker](../scripts/calibrate_loaded_ring3d.py) · [Area and quadrature checks](../tests/test_ring_interface.py).

## What is in the Lumerical comparison

The public project specifies a 3.1 µm radius, a 400 nm-wide and 180 nm-high silicon strip, vertical sidewalls, and a 100 nm gap between the ring and each bus. It generates the ring from four cubic Bézier centerline arcs. Compared with our circular centerline, the largest radial displacement is 0.61 nm and the perimeter changes by 0.74 nm over a roughly 19,478 nm round trip. This bounds the geometric approximation to the *centerline*. Its effect on the coupled spectrum would require a matched device run. The archived native grid uses approximately 43 nm steps near the guide center, with different spacing farther from the core. The project uses a wavelength-dependent Palik silicon/oxide table. Its tabulated imaginary permittivities are zero at the five sampled wavelengths from 1500 to 1600 nm. [Project measurements and attachment hashes](verification/v013-ring-project-audit.json) · [Official Ansys project](https://optics.ansys.com/hc/en-us/articles/360042800213-Ring-resonator-getting-started-Final-parameter-extraction).

The filed fundamental mode has effective index 1.93252 and reported TE fraction 0.9471 at 1548.387 nm. The four ports span 3 µm by 3 µm at x = ±4.2 µm, y = ±3.6 µm, z = 0.09 µm. Port 1 excites the upper bus. Our device mirrors the launch across y, centers z on the silicon midplane and uses narrower, mesh-snapped apertures. Both normalize to incident guided power. The Lumerical article describes using one central mode for expansion, while the workbench interpolates five separately solved mode profiles. The archived project settings and the separate exported reference spectrum have no common run identifier. [Project extraction](../scripts/audit_ring_vendor_project.py) · [Measured aperture control](verification/v012-ring3d-lumerical-aperture-comparison.json).

Two Ansys attachments measure different quantities at the drop port. At 1550 nm the numerical S file gives 67.44% in the fundamental mode. The separate total-flux file gives 75.46%. Both show a resonance at about 1550.09 nm. Our modal spectra are compared with the fundamental S value. The two published attachments differ by an average of 0.56 percentage points across their common band and by up to 8.47 points at individual wavelengths. [Reference data and reproduction links](verification/v013-ring-project-audit.json).

## Separate phase and material checks

At 16, 20 and 24 cells/µm the original lateral samples represent one straight strip as 406.25, 400.00 and 395.83 nm wide. Exact integration represents the prescribed 400 nm width at each resolution. Width here means integrated silicon area per unit length. The materials at individual Yee nodes and the resulting mode can still differ with mesh. [Mode profiles and width integrals](verification/v013-ring-port-wall-mixing.json).

For the same nominal Lumerical strip at the filed mode wavelength, two higher-resolution transverse eigensolvers give these effective indices. Each Palik value uses the sampled permittivity at that wavelength. The workbench's fixed material has silicon and oxide indices of 3.48 and 1.44.

| Local transverse solver | Mesh (cells/µm) | Fixed indices | Sampled Palik indices |
| --- | ---: | ---: | ---: |
| Tidy3D | 128 | 1.91263 | 1.90966 |
| MPB | 128 | 1.91475 | 1.91176 |

These independently configured leads have residual model/grid uncertainty. Tidy3D's local calculation uses a uniform mesh without interface smoothing and also supplies FDTDX's internal transverse mode solver. MPB uses a separate plane-wave eigensolver with dielectric averaging. At 96 cells/µm, using the measured wavelength dependence changes the straight-lead group index from 4.5224 to 4.6970 in Tidy3D and 4.5445 to 4.7188 in MPB. Both predict a larger group delay. In the simple straight-path estimate, this reduces the predicted spacing by about 3.7%. A loaded ring also accumulates bend and coupler phase. Its poles require a full loaded-device calculation. [Local Tidy3D measurements](verification/v013-ring-ports-tidy3d.json) · [Independent MPB measurements](verification/v013-ring-ports-mpb.json) · [Reproduction program](../scripts/compare_ring_port_indices.py).

An additional straight-guide control treats the normal electric field at each vertical wall with harmonic dielectric averaging. At 16 cells/µm this moves the lead mode index from 1.97334 for exact-area arithmetic sampling to 1.92090, closer to the filed 1.93252. At 24 cells/µm it moves from 1.92949 to 1.90580, farther from that filed value. At 20 cells/µm the sampled walls align and both give 1.91753. Curved walls require normal/tangential tensors, including off-diagonal contributions, and membrane/substrate corners need distinct treatment. A curved-device measurement is needed to evaluate the resulting power error. [Straight-wall experiment](verification/v013-ring-port-wall-mixing.json) · [Subpixel dielectric theory](https://meep.readthedocs.io/en/master/Subpixel_Smoothing/).

## Measured ring comparisons

All three comparisons use full NVIDIA reference/device pulses and acquired 241 qualified wavelengths on both variants. The first two test the Lumerical geometry, and the last applies the same sampling change to the independently specified Tidy3D device. Each vendor-band difference is evaluated on 201 common samples. Each entry shows the original quadrature → exact lateral area.

| Case | Drop at 1550 nm | Published drop at 1550 nm | Through / drop mean absolute difference in power fraction | Central resonance spacing |
| --- | --- | --- | --- | --- |
| Lumerical, 16 cells/µm | 2.563% → 3.160% | 67.442% | 0.1553 / 0.1241 → 0.1622 / 0.1336 | 26.0 → 26.0 nm |
| Lumerical, 20 cells/µm | 1.131% → 1.122% | 67.442% | 0.1730 / 0.1357 → 0.1723 / 0.1365 | 27.0 → 27.0 nm |
| Tidy3D, 16 cells/µm | 71.976% → 69.390% | 2.159% | 0.1560 / 0.1298 → 0.1637 / 0.1379 | 17.5 → 17.5 nm |


The source-mode index at 1550 nm falls from 1.99050 to 1.97062 on the 16-cell mesh. The two central resonances still fall on the same 0.5 nm sample wavelengths. At 20 cells/µm the bus width was already represented exactly, although the circular-interface fractions still change. Correcting geometric area improves the represented width. It leaves a large absolute power error on both meshes, including sustained differences across the vendor band. On the Tidy3D geometry, the 437.5 nm integrated bus width becomes the specified 430 nm. Both central peak wavelengths keep their 0.5 nm sampled positions; the absolute drop-power error over the band still increases. [Tidy3D unshifted fields and spectrum](verification/v013-ring3d-tidy3d-r16-t4800-exact-area.json) · [All three controlled comparisons and array hashes](verification/v013-ring-interface-comparison.json) · [16-cell fields and spectra](verification/v013-ring3d-lumerical-r16-t3200-exact-area.json) · [20-cell fields and spectra](verification/v013-ring3d-lumerical-r20-t3200-exact-area.json).

## Passive material dispersion at the same mesh

A separate full 16-cell Lumerical ring measurement keeps the exact-area geometry and introduces a passive dispersion model. One undamped auxiliary differential equation (ADE) pole outside the passband fits the public Palik permittivities at 1500, 1550 and 1600 nm. The checks hold out 1525 and 1575 nm. The maximum absolute mixed-cell permittivity error at either held-out wavelength is 0.00449. The source uses a fixed fundamental mode at the fitted carrier permittivity. Each acquisition checks all 1,600 driven time steps and the staggered E/H offsets. The output bases use the realized discrete dielectric function at their five separate frequencies. [Fitted material model](../scripts/ring_palik_dispersion.py) · [Native dispersive ring](verification/v013-ring3d-lumerical-r16-t3200-exact-area-palik-ade.json).

| Observable | Constant indices, exact area | Passive dispersion, exact area | Published Lumerical |
| --- | ---: | ---: | ---: |
| Fundamental drop at 1550 nm | 3.160% | 4.919% | 67.442% |
| Fundamental through at 1550 nm | 96.336% | 94.321% | 9.392% |
| Central resonance spacing | 26.0 nm | 24.5 nm | 25.455 nm |
| Mean absolute through / drop error against published power | 0.1622 / 0.1336 | 0.1469 / 0.1212 | Reference |

All 201 wavelengths within the material fit passed the declared numerical checks. The 40 samples outside 1500–1600 nm are marked unqualified. The two central drop peaks move from 1530.5 / 1556.5 nm to 1530.0 / 1554.5 nm. This improves the two broadband mean errors but increases the absolute central spacing error from 0.55 to 0.95 nm. The normalized straight through-power spans 0.9981–1.0015; maximum collected port power is 1.0013. Dispersive material contributes measurably to detuning and group delay. Even with that change the 1550 nm drop is still 62.52 percentage points below the published value. [Controlled spectra and exported raw hashes](verification/v013-ring-ade-comparison.json).

**The 16-cell grid has a separate, substantial propagation error.** An independently launched half-micrometre silicon film on this same grid differs from analytic transmission by up to **19.96 percentage points** and transmitted phase by **0.247 rad**. At 96 cells/µm those errors fall to 0.64 points and 0.00794 rad. The refined film verifies the passive update; the coarser ring is a material sensitivity measurement. An accuracy claim for the device needs both another material-enabled ring mesh and tests of curved-interface polarization, finite coupler phase, backscatter and linewidth against the same ports and modes. [Real NVIDIA film controls and their predeclared criteria](verification/v013-palik-slab.json).

The normal exporter bundled the measured dispersive ring program and Jupyter notebook, preserving the original GPU fields. The readback checked thirteen captured acquisition files, the material input and both reference artifacts. The release scope for this archive is verification against the measured run. [Dispersive result export and checksums](verification/v013-ring-ade-example-export.json).

## Run the controlled device comparison

```bash
"$FDTDX_PYTHON" -m scripts.calibrate_loaded_ring3d --vendor lumerical --device gpu \
  --resolution 16 --duration-um-c 3200 --ring-interface exact-area --output exact-ring
```

Use `--ring-material palik-ade` with the Lumerical exact-area model to repeat the 1500–1600 nm material study. The other calibrated ring configurations use their original constant indices. Each run writes its actual fields, modal coefficients, qualification mask, acquisition duration and solver identity. Use an empty output directory. The Tidy3D held-out device has a 5 µm radius, a 430 × 220 nm strip, oxide above and below and a 4800 µm/c observation window. Its published notebook uses circular geometry, vertical symmetry, absorbers, an automatic mesh and dispersive lossless silicon/oxide models. Its numerical powers are digitized from the pinned plot and have greater uncertainty on steep lines. [Tidy3D add/drop example](https://www.flexcompute.com/tidy3d/examples/notebooks/RingResonator/) · [Pinned notebook](https://github.com/flexcompute/tidy3d-notebooks/blob/18fd47f8bd87ad89582b507c20bf4997f79d55c0/RingResonator.ipynb).

To reproduce the vendor audit, download the three attachments linked in its receipt outside the repository. The extraction program verifies every file digest. Run the mode programs with the corresponding local numerical interpreter.

```bash
python -m scripts.audit_ring_vendor_project --project /path/to/ring_resonator.fsp \
  --total-flux /path/to/fdtd_results.ldf --s-parameters /path/to/FDTDtoINTERCONNECT.txt \
  --output ring-project-audit.json
"$MEEP_PYTHON" -m scripts.compare_ring_port_indices --engine mpb --output mpb-port-modes.json
"$FDTDX_PYTHON" -m scripts.compare_ring_port_indices --engine tidy3d --output tidy3d-port-modes.json
```
