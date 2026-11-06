# Where the curved-grating power difference comes from

The Lumerical comparison uses a tilted polished fiber, a curved silicon grating and a strip-mode output. The Tidy3D example illuminates a different, straight grating and a 50 µm taper with an air-side Gaussian. Each comparison below uses its own source and observable. The baseline uses the original 12- and 14-cells/µm records. [Published Ansys project and S-parameter file](https://optics.ansys.com/hc/en-us/articles/360042305334-Grating-coupler) · [Published Tidy3D notebook](https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/) · [Baseline spectra](FINITE_GRATING_3D.md).

## Full-device measurements

On the RTX 3080 Ti Laptop GPU, independently normalized reference and device runs at both meshes passed the source, settling and power checks at all five wavelengths. The original and direct samplers used identical source electric and magnetic arrays, staggered phase delays, scalar geometry, refractive indices, pulse, aperture and physical domain. Only the silicon coefficients used to update the three electric field components changed.

| Wavelength (nm) | Ansys S21 power | Baseline 12 | Direct 12 | Baseline 14 | Direct 14 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1500 | 15.06% | 31.65% | 4.60% | 33.68% | 9.42% |
| 1525 | 29.45% | 31.90% | 12.41% | 34.60% | 20.85% |
| 1550 | 40.03% | 22.21% | 25.15% | 24.91% | 33.61% |
| 1575 | 37.02% | 11.21% | 34.36% | 12.99% | 38.00% |
| 1600 | 22.37% | 3.88% | 30.59% | 4.59% | 29.36% |

The 14-cell choice improved 1550 nm coupling from **24.91% to 33.61%**, against **40.03%** published. The five-sample mean absolute error fell from **16.14 to 5.73 percentage points**. At 12 cells/µm it fell from 16.23 to 10.65 points. Both meshes passed the prespecified improvement test at 1550 nm and across the five-point band. Direct sampling raised the recorded guide index from 2.391198 to 2.444075 on the 14-cell mesh at the exact filed mode frequency, bringing it closer to 2.452793. At 1550 nm the whole outgoing aperture now carries 36.88%; the external guide-mode reprojection gives 33.88%. Together these show that changing the discretized device changed the transfer through the aperture.

Absolute power still depends on mesh. Direct 12→14 changes the 1550 nm result by **8.46 percentage points**. The strongest of these five direct samples is at **1575 nm** on both meshes; the Ansys file peaks at **1557.62 nm**. The unchanged Tidy3D uniform-grating control gives **35.67% vs 38.02%** at 1550 nm and a five-sample mean error of **2.95 points**. The two physical designs have separate comparison curves. [Controlled paired results and raw reconstruction](verification/v013-grating-power-mismatch.json) · [12-cell GPU run](verification/v013-grating3d-lumerical-direct-r12.json) · [14-cell GPU run](verification/v013-grating3d-lumerical-direct-r14.json) · [Tidy3D geometry](verification/v012-grating3d-tidy3d-r14.json).

## Geometry, port basis and power normalization

The Ansys project sets the pitch to 671.3 nm, unetched duty cycle to 39.92%, ridge and film heights to 220 and 120 nm, radius to 25 µm and aperture to 15 µm. Its construction script begins each radial period with the etched interval. The local reconstruction follows those values and the finite simulation domain. On the 14-cell grid the 100 nm etch spans 1.4 cells and the 120 nm film spans 1.68. Ansys's saved guide mode has transverse samples approximately 42.92 nm apart in y and 43.10 nm apart near the silicon in z. The sampled spacings come from the filed **output port**. The device uses a nonuniform mesh. [Extracted geometry, material and source data](verification/v012-grating-commercial-references.json).

The project also stores the output strip's vector mode at 1548.387 nm with effective index 2.452793. The baseline Workbench guide gives 2.370135 and 2.391198 on the 12- and 14-cell meshes at that wavelength. The effective-index discrepancy is evidence that the discretized guide supports a different propagation constant. Its contribution to the power discrepancy requires a separate device measurement.

We reprojected the measured 1550 nm Workbench output fields onto the **actual cached Ansys output E/H mode**, interpolated onto the local detector coordinates. The guided power at 14 cells/µm moved from 24.9083% with the local mode to 25.1838% with the external basis. Even moving that basis by half a native cell in both transverse directions gave only 23.63–25.18%. The normalized transverse Ey shape overlap was 0.9930. The remaining difference from the published 40.0266% occurs upstream of this readout or in the measurement conditions. This check uses the vendor's single nearby mode frequency and the locally measured incident reference. The distributed project marks its saved output-strip mode as valid and its cached source-port mode as invalid. We used only the output mode for the reprojection. The independently checked LP01 approximation and strict incident E/H and delay equality constrain the source separately. [Measured receiver reprojection](verification/v013-grating-port-basis.json) · [Fiber-mode cross-check](verification/v012-grating-fiber-mode-reference.json).

Direct integration of the six native fields on the baseline 14-cell strip aperture gives 26.8784% net outgoing power at 1550 nm, including radiation and any other modes crossing that aperture. The workbench quasi-TE channel collects 24.9083%. The upstream finite-aperture scattered field carries 8.41% net upward power, of which 0.360% projects back into the incoming fiber mode. The two apertures collect part of the outgoing radiation. Radiation through the remaining faces needs its own monitors.

The scalar LP01 profile puts about 99.986% of its infinite-plane power inside the recorded incident aperture. This bounds clipping in the idealized launch calculation. The denominator comes from the six native incident fields. Ansys's own S12 and S21 powers differ by 0.93 percentage points at 1550 nm and up to 2.63 points over these five samples. We compare the correct fiber-to-guide direction, S21.

A separate **local Tidy3D** mode solve configured the same 500 × 220 nm strip, cover, buried oxide and substrate at the filed frequency. The guide indices were 2.49215, 2.50983, 2.47512, 2.45713 and 2.45337 at 32, 48, 72, 96 and 128 cells/µm. The finest index differs from the filed 2.45279 by +0.00058. The 96-cell calculation changed by −0.00253 on increasing the mode aperture. Local subpixel averaging was disabled, and the mesh trend is nonmonotonic. FDTDX also uses Tidy3D for transverse eigensolves, so this comparison changes the physical grid and setup while sharing the eigensolver family. [Strip mode meshes, localization and profile comparisons](verification/v013-grating-strip-tidy-modes.json) · [Local mode program](../scripts/validate_grating_guide_mode.py).

## Test of vertical interface sampling

The baseline first assigns fractional silicon coverage on cell-centered volumes and then averages those already-averaged values onto each electric Yee component. This applies a second smoothing at a sharp interface. The optional `direct-yee` calculation integrates the same silicon objects over control volumes centered directly on Ex, Ey and Ez. Ex and Ey use the tangential dielectric average at horizontal interfaces. Ez uses the harmonic average through the thickness of each silicon column, followed by an area average over ridge, residual film and oxide. The source, reference geometry, material indices, PML, physical guide aperture and native GPU solver are preserved. The lateral curved interfaces retain a diagonal approximation; the full interface-normal tensor is a separate improvement. [Ansys conformal mesh methods](https://optics.ansys.com/hc/en-us/articles/360034382594-Understanding-Mesh-Refinement-and-Conformal-Mesh-in-FDTD) · [Tidy3D subpixel discretization](https://docs.flexcompute.com/projects/tidy3d/en/v2.11.0/api/discretization/index.html).

Before the new device runs, a separate TE slab calculation predicted a substantial **redshift** from the direct cells. At 14 cells/µm its first-order phase-matching estimate changes from 1521.90 to 1573.98 nm. Both estimates approach the same continuous-slab value, 1543.47 nm, on a much finer mesh. Substituting the wavelength-dependent silicon and oxide indices from the project into the continuous slab changes the estimate by just +0.28 nm. The slab screens the direction and scale of numerical dispersion. Measuring coupling efficiency requires the full curved device, taper, finite substrate and fiber. The real 3D study uses the fixed 1550 nm Palik values across the band. [Independent slab calculation](../scripts/grating_phase_matching.py).

## What would establish stronger agreement

1. Resolve the peak and bandwidth on the same physical wavelengths, without shifting one spectrum onto the other. The original five samples are 25 nm apart. Resolving a peak center to a few nanometers requires closely spaced samples. Repeat the spectral check at a finer physical mesh before using a peak or loss for design.
2. Resolve the etched silicon more finely while holding the taper, fiber and physical planes fixed. A smaller global step improves both geometric sampling and wave propagation, but the full 16-cell geometry exceeded the measured memory admission on this GPU. A local nonuniform grid or a symmetry-reduced calculation could improve the cost. Before using a half domain, verify its TE mirror parity against the existing full domain at the same mesh with raw E/H fields, absolute power and phase. [Measured GPU budget](verification/v012-grating-memory-budget.json) · [Ansys nonuniform grid](https://optics.ansys.com/hc/en-us/articles/360034382634-Understanding-the-non-uniform-mesh-in-FDTD).
3. Test interface-normal tensor averaging at curved lateral walls and tooth ends. Separate the contribution of constitutive sampling from that of the finite Yee propagation step using a straight slab and guide, then a single radial period and finally the whole device. Broadband convergence requires stability at the same wavelengths over successive meshes.
4. Retain the Palik table's causal dispersion in the full device when extending to broader bands. Validate fitted real and imaginary permittivities against held-out wavelengths and slab propagation before assigning a spectral error to dispersion. Refine the fiber-to-chip source alignment and vector impedance if the remaining band error persists with a stable material grid. Measure outward power on a closed set of faces to distinguish uncaptured radiation from numeric loss.

## Reproduce the control

From a locally configured workbench, use a separate output directory for each mesh and sampling choice. The `volume-average` choice retains the released material calculation. `direct-yee` selects direct silicon integration for the curved Lumerical geometry. Both launch fresh incident and device stages if no matching checkpoint exists.

```bash
photonics-workbench calibrate grating-3d --solver fdtdx --device gpu --geometry-3d lumerical --resolution 14 --grating-material-model direct-yee --output grating-direct-14
```

The 14-cell source-bound export includes the measured arrays, frozen numerical worker, Python launcher and notebook. It was checked against the original native acquisition and read back from its ZIP. The separate notebook replay receipts cover the two v0.12.0a1 vendor examples. [Corrected export manifest](verification/v013-grating-direct-export.json) · [Original notebook executions](verification/v013-grating-notebook-replays.json).

The project-file check can be regenerated locally using the original download from the Ansys example. It reads the hash-verified published project, projects local detector fields onto its stored output mode and writes derived scalar measurements. The raw vendor fields stay outside the repository.

```bash
python -m scripts.compare_grating_port_basis --project ansys.zip --receipts docs/verification/v012-grating3d-lumerical-r12.json docs/verification/v012-grating3d-lumerical-r14.json --output grating-port-check.json
```
