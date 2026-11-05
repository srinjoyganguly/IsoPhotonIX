# Calibrated components checklist

This checklist tracks measured performance separately from field propagation.
All 27 library presets already have field evidence on Meep CPU and FDTDX CPU
and NVIDIA. [Field runs](LAYOUT_FIELDS.md#verification).

| Chunk | Local numerical evidence | Lumerical and Tidy3D comparison |
| --- | --- | --- |
| Straight and tapered two-port guides | Complete for the documented fundamental-mode studies. Complex S, both launch directions, mesh sensitivity and replay are available. [Taper checks](TAPER_PORTS.md). | Published mode/coupling comparisons exist for specific observables. A public taper matrix with identical geometry and reference planes is still needed. |
| Four-port directional coupler | Complete four-launch Meep CPU and FDTDX NVIDIA matrices at the baseline mesh. Center cross-port powers from lower-left input are 0.07320 and 0.05436 respectively. FDTDX CPU and numerical sensitivities in progress. [Meep](verification/v011-meep-multiport-coupler-electric.json) · [CUDA](verification/v011-fdtdx-coupler-gpu.json). | Published SOI supermode indices and transfer length compared across Lumerical, local Tidy3D and MPB. The bent multiport device has different access sections. |
| Three-port splitter | Complete three-launch Meep CPU and FDTDX NVIDIA matrices at the baseline mesh. Stem-to-arm powers at 1.55 µm are 0.40414 / 0.38880 and 0.38074 / 0.33390 respectively. [Meep](verification/v011-meep-multiport-splitter-electric.json) · [CUDA](verification/v011-fdtdx-splitter-gpu.json). | Local cosine Y measured 1.008 dB excess loss at 1.55 µm; Tidy3D reports <0.2 dB average for a different optimized junction. Lumerical reports near-equal splitting for its optimized Y. These compare designs, not identical solvers. |
| Ring and add/drop spectra | In progress. Normalize bus/drop spectra, extract resonance and test spectral resolution, run time and mesh. | Compare equivalent FSR and loaded linewidth; record differences in coupling, material and dimension. |
| Grating radiation | Compact Tidy3D-domain Meep calculation measured 58.23% Gaussian-to-guide coupling at 1.55 µm versus 56.23% in the example. The exported compact program was independently executed as Python and a notebook. Expanded-domain and Lumerical comparisons are in progress. [Measured control](verification/v011-grating-compact-tidy3d.json). | Compare the same collection aperture, direction and normalization. |
| Multimode coupling | Planar four-channel joint measured at three meshes. Largest power error against an independent open-boundary solution is 0.00124 at 64 cells/µm. A complete two-mode finite-height four-launch matrix is also available at 16 cells/µm. Higher-resolution and sensitivity controls are running. [3D measurements](verification/v011-multimode-finite.json). [Measurements](verification/v011-multimode-planar.json). | Compare named mode indices and modal powers with both vendors; distinguish matched numbers from method-level references. |

An item is qualified only for the dimensionality, geometry, target and observables
identified in its measurement record. The live Meep port convention is backed by a [two-sided Fresnel reflection test](verification/v011-meep-fresnel-phase.json) and an [independent electric-field audit of the actual splitter](verification/v011-meep-multiport-phase-audit.json). Historic records retain their original phase basis.

## Delivery checklist

- [x] Local measured Meep and CUDA multiport matrices, verified electric-phase convention, and immutable historical records.
- [x] Measured planar and finite-height two-mode matrices, with independent planar power and phase comparisons.
- [x] Replayed Gaussian grating calculation as Python and an executed Jupyter notebook.
- [ ] Complete remaining per-target runs and numerical sensitivity checks. Record the differences from both vendors.
- [ ] Verify the installed calibration interface, replay its final downloads and publish the release wheel.


[Matched Lumerical and local Tidy3D port-mode measurements](CALIBRATED_VENDOR_CHECKS.md).
