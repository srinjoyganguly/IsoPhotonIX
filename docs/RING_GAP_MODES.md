# The 100 nm gap in the Lumerical ring

Ansys publishes an independent check on the geometry that sets the ring coupling. At 1550 nm, two straight 400 nm × 180 nm silicon rails separated by 100 nm have an even–odd quasi-TE mode-index difference of **0.109**. A larger difference means guided power transfers over a shorter length. Ansys uses the straight pair for this measurement, then uses a curved ring with zero straight coupling length for the final device. [Lumerical design tutorial, step 2](https://optics.ansys.com/hc/en-us/articles/360042800293-Ring-resonator-getting-started-Design-and-initial-simulation).

We solved this transverse problem locally on CPU. All runs used the same physical edges, air over oxide and the two permittivities at 1550 nm from the audited Lumerical project. Each pair passed the polarization check and had opposite electric-field parity across the gap. The FDTDX columns use its Tidy3D transverse eigensolver with the same area or wall coefficients used in the native workbench ring. MPB is an independent eigensolver. The last column uses Tidy3D's local geometry and boundary-registered grid with its optional subpixel package disabled. [Material samples](verification/v013-ring-project-audit.json) · [Executable study](../scripts/check_ring_gap_modes.py).

| Cells/µm | Workbench area | Workbench wall | MPB | Tidy3D local geometry |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.21100 | 0.09563 | — | — |
| 20 | 0.11803 | 0.11803 | — | — |
| 24 | 0.13134 | 0.10610 | — | — |
| 32 | 0.14648 | 0.10952 | 0.11688 | 0.05862 |
| 64 | 0.11690 | 0.10965 | 0.11127 | 0.07774 |
| 96 | 0.12134 | 0.11060 | 0.11080 | 0.08837 |
| 128 | 0.11533 | 0.10992 | 0.11085 | 0.09293 |

At 128 cells/µm the workbench wall model gives 0.10992, within 0.00092 of the three-decimal reference. MPB gives 0.11085. Increasing its mesh from 96 to 128 changes the result by 0.000051. Tidy3D's local geometry calculation gives 0.09293 at 128 and continues to move with refinement. For this cross-section the configured boundary treatment matters even when the electromagnetic eigensolver is shared. [Area samples](verification/v014-ring-gap-area.json) · [Wall samples](verification/v014-ring-gap-wall.json) · [MPB samples](verification/v014-ring-gap-mpb.json) · [Tidy3D samples](verification/v014-ring-gap-tidy.json).

At the affordable full-device meshes, the wall control moves from 0.09563 at 16 to 0.11803 at 20 and 0.10610 at 24. A 100 nm gap spans just 1.6, 2.0 and 2.4 cells respectively. At 20 the two wall treatments agree because both rail edges coincide with the relevant cell faces. As a separate sensitivity test, MPB at 64 cells/µm gives 0.11966, 0.11127 and 0.10474 for gaps of 90, 100 and 110 nm. Widening the transverse window at the same mesh changes the 100 nm result by 0.00022. The 90 and 110 nm controls quantify sensitivity. The full-ring benchmark retains its specified 100 nm gap.

This check isolates the coupling cross-section. The loaded spectrum also depends on bend phase, both coupling regions, dispersion, loss, reference planes and observation time. The full-device reports have their own acceptance gates. [Calibration checklist](CALIBRATED_COMPONENTS.md).

## Isolated bent-guide phase and resonance spacing

A separate local Tidy3D vector-mode calculation follows the fundamental quasi-TE field around the isolated cross-section of each loaded-ring benchmark. It uses the same fixed indices, width, height and centerline radius as the 3D GPU study. Five frequencies around 1550 nm give a finite-difference group index. Both curvature signs are measured because reflecting this transverse coordinate system should give the same physical bend. Neighboring wavelength fields have squared Ey overlap above 0.999998. The grids have 24, 32, 48 and 64 cells/µm; a wider transverse domain at 64 checks boundary sensitivity. Local subpixel averaging was disabled. These controls were computed locally on CPU. [Tidy3D bend-axis and radius convention](https://docs.flexcompute.com/projects/tidy3d/en/v2.11.2/api/_autosummary/tidy3d.ModeSpec.html) · [All selected modes and convergence controls](verification/v014-ring-bent-modes.json) · [Reproduction program](../scripts/check_ring_bend_modes.py).

| Cross-section, 64 cells/µm | Isolated spacing estimate, both curvature signs | Loaded GPU spacing at 20 cells/µm | Published loaded spacing |
| --- | ---: | ---: | ---: |
| Lumerical, 3.1 µm radius | 27.14–27.23 nm | 26.50 nm | 25.455 nm |
| Tidy3D, 5.0 µm radius | 18.21–18.24 nm | 17.50 nm | 17.74 nm |

The isolated spacing estimate uses λ²/(2πR n_g). For the positive-curvature Lumerical guide, refining 48→64 changes this estimate by 0.074 nm, yet shifts its inferred centerline propagation phase by 3.55 radians in one round trip. The Tidy3D cross-section gives a 3.46-radian shift under the same refinement. This shows quantitatively how spacing can look stable while the absolute phase that sets a peak wavelength is still moving. The two buses, coupling phase and full-device time evolution are absent from this transverse estimate. Its spacing cannot be used to correct the measured loaded powers. At 64, widening the plane from 2.5 to 3 µm changes the estimated spacing by less than 0.001 nm for both curvature signs. A separate exploratory 96-cell series did not finish both geometries and was excluded.
