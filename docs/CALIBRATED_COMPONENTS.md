# Calibrated components checklist

This checklist tracks measured performance separately from field propagation.
All 27 library presets already have field evidence on Meep CPU and FDTDX CPU
and NVIDIA. [Field runs](LAYOUT_FIELDS.md#verification).

| Chunk | Local numerical evidence | Lumerical and Tidy3D comparison |
| --- | --- | --- |
| Straight and tapered two-port guides | Complete for the documented fundamental-mode studies. Complex S, both launch directions, mesh sensitivity and replay are available. [Taper checks](TAPER_PORTS.md). | Published mode/coupling comparisons exist for specific observables. A public taper matrix with identical geometry and reference planes is still needed. |
| Four-port directional coupler | Complete four-launch Meep CPU and FDTDX NVIDIA matrices at the baseline mesh. Center cross-port powers from the lower-left input are 0.07320 for Meep, 0.07670 for corrected CUDA and 0.05436 for the original CUDA grid. [Corrected CUDA matrix](verification/v011-fdtdx-yee-coupler-gpu16.json). The complete corrected 12-cell FDTDX CPU/CUDA pair agrees within 1.25 × 10⁻⁷ in power fraction; 16-cell CPU and further sensitivity controls are in progress. [Same-grid CPU/CUDA check](verification/v011-fdtdx-yee-coupler-cpu-cuda12.json). [Meep](verification/v011-meep-multiport-coupler-electric.json) · [CUDA](verification/v011-fdtdx-coupler-gpu.json). | Published SOI supermode indices and transfer length compared across Lumerical, local Tidy3D and MPB. The bent multiport device has different access sections. |
| Three-port splitter | Complete three-launch Meep CPU and FDTDX NVIDIA matrices at the baseline mesh. Stem-to-arm powers at 1.55 µm are 0.40414 / 0.38880 for Meep and 0.36595 / 0.36595 for the corrected Yee-grid CUDA model. The earlier FDTDX grid gave 0.38074 / 0.33390. A full corrected 12-cell CPU/CUDA comparison differs by at most 6.23 × 10⁻⁸ in guided power fraction. [Same-grid comparison](verification/v011-fdtdx-yee-splitter-cpu-cuda12.json). [Corrected GPU matrix](verification/v011-fdtdx-yee-splitter-gpu16.json). [Meep](verification/v011-meep-multiport-splitter-electric.json) · [CUDA](verification/v011-fdtdx-splitter-gpu.json). | Local cosine Y measured 1.008 dB excess loss at 1.55 µm; Tidy3D reports <0.2 dB average for a different optimized junction. Lumerical reports near-equal splitting for its optimized Y. These compare designs, not identical solvers. |
| Ring and add/drop spectra | Complete normalized four-channel planar spectra for both vendor geometries, with mesh, boundary, aperture and time controls, and finite-height isolated-ring poles. All 561 fine-grid wavelengths satisfy the declared power and finite-time limits. [Complete measured study](verification/v011-ring.json). | Planar loaded FSR is 40.25 vs 25.455 nm (Lumerical) and 23.50 vs 17.740 nm (Tidy3D). At 48 cells/µm the finite-height bare-ring spacings are 26.919 and 18.143 nm. Absolute pole convergence is reported separately. A full finite-height loaded ring remains a further physics study. |
| Grating radiation | Compact Tidy3D-domain Meep calculation measured 58.23% Gaussian-to-guide coupling at 1.55 µm versus 56.23% in the example. The compact Python and notebook replays agreed. At 1310 nm the Lumerical article reconstruction measured 12.28% at 28 cells/µm and 49.42% at 40 cells/µm versus the published 55.7%. [28-cell](verification/v011-grating-ansys-article-r28.json) · [40-cell](verification/v011-grating-ansys-article-r40.json). Using the filed source and stack gives 49.38% at the same mesh. [Project-source control](verification/v011-grating-ansys-project-r40.json). The separate direct-flux comparison is in progress. [Measured control](verification/v011-grating-compact-tidy3d.json). | Compare the same collection aperture, direction and normalization. |
| Multimode coupling | Planar four-channel joint measured at three meshes. Largest power error against an independent open-boundary solution is 0.00124 at 64 cells/µm. A complete two-mode finite-height four-launch matrix is also available at 16 cells/µm. Higher-resolution and sensitivity controls are running. [3D measurements](verification/v011-multimode-finite.json). [Measurements](verification/v011-multimode-planar.json). | Compare named mode indices and modal powers with both vendors; distinguish matched numbers from method-level references. |

An item is qualified only for the dimensionality, geometry, target and observables
identified in its measurement record. The live Meep port convention is backed by a [two-sided Fresnel reflection test](verification/v011-meep-fresnel-phase.json) and an [independent electric-field audit of the actual splitter](verification/v011-meep-multiport-phase-audit.json). Historic records retain their original phase basis.

## Execution targets

| Preset | Meep CPU | FDTDX CPU | FDTDX NVIDIA |
| --- | --- | --- | --- |
| Directional coupler | Full four-port matrix | Full at 12 cells/µm; 16-cell grid in progress | Full at 12 and 16 cells/µm |
| Y splitter | Full three-port matrix | Full at 12 cells/µm; 16-cell grid in progress | Full at 12 and 16 cells/µm |
| Planar and finite-height multimode joints | Full four-channel matrices | Outside the validated preset routes | Outside the validated preset routes |
| Ring spectra and finite-height isolated poles | Both named loaded planar spectra and finite-height bare-ring measurements; convergence limits are reported per geometry | Outside the validated preset routes | Outside the validated preset routes |
| Grating coupling and radiation | Named Gaussian and guide launches; the total output aperture has a separate study | Outside the validated preset routes | Outside the validated preset routes |

## Delivery checklist

- [x] Local measured Meep and CUDA multiport matrices, verified electric-phase convention, and immutable historical records.
- [x] Measured planar and finite-height two-mode matrices, with independent planar power and phase comparisons.
- [x] Replayed Gaussian grating calculation as Python and an executed Jupyter notebook.
- [ ] Complete remaining per-target runs and numerical sensitivity checks. Record the differences from both vendors.
- [x] Replayed a CUDA splitter from its exported Jupyter notebook. The fresh calculation reproduced all seven numerical NPZ payloads byte-for-byte. [CUDA replay](verification/v011-fdtdx-yee-notebook-replay.json).
- [x] Replayed one GUI-exported four-channel Meep matrix as Python and an executed Jupyter notebook. Largest raw-array differences were 5.76e−8 and 2.12e−8. [Replay evidence](verification/v011-calibration-replays.json).
- [x] The [v0.11.0a1 pip wheel and source distribution](https://github.com/srinjoyganguly/photonics-workbench/releases/tag/v0.11.0a1) passed a clean offline install and an anonymous checksum check. [Installed numerical run](verification/v011-a1-wheel-install.json) · [Release verification](verification/v011-a1-release.json).
- [ ] Promote the corrected defaults after the remaining target and physics controls complete.


[Matched Lumerical and local Tidy3D port-mode measurements](CALIBRATED_VENDOR_CHECKS.md).

[See the actual local grating run](verification/v011-calibrated-grating.png) · [Browser checks and exported inputs](verification/v011-browser-grating.json).
