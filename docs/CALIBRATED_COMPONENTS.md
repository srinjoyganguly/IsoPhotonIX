# Calibrated components checklist

This checklist tracks measured performance separately from field propagation.
All 27 library presets already have field evidence on Meep CPU and FDTDX CPU
and NVIDIA. [Field runs](LAYOUT_FIELDS.md#verification).

| Chunk | Local numerical evidence | Lumerical and Tidy3D comparison |
| --- | --- | --- |
| Straight and tapered two-port guides | Complete for the documented fundamental-mode studies. Complex S, both launch directions, mesh sensitivity and replay are available. [Taper checks](TAPER_PORTS.md). | Published mode/coupling comparisons exist for specific observables. A public taper matrix with identical geometry and reference planes is still needed. |
| Four-port directional coupler | Complete four-launch Meep CPU and FDTDX NVIDIA matrices at the baseline mesh. Center cross-port powers from lower-left input are 0.07320 and 0.05436 respectively. FDTDX CPU and numerical sensitivities in progress. [Meep](verification/v011-meep-multiport-coupler.json) · [CUDA](verification/v011-fdtdx-coupler-gpu.json). | Published SOI supermode indices and transfer length compared across Lumerical, local Tidy3D and MPB. The bent multiport device has different access sections. |
| Three-port splitter | Complete three-launch Meep CPU and FDTDX NVIDIA matrices at the baseline mesh. Stem-to-arm powers at 1.55 µm are 0.40414 / 0.38880 and 0.38074 / 0.33390 respectively. [Meep](verification/v011-meep-multiport-splitter.json) · [CUDA](verification/v011-fdtdx-splitter-gpu.json). | Local cosine Y measured 1.008 dB excess loss at 1.55 µm; Tidy3D reports <0.2 dB average for a different optimized junction. Lumerical reports near-equal splitting for its optimized Y. These compare designs, not identical solvers. |
| Ring and add/drop spectra | In progress. Normalize bus/drop spectra, extract resonance and test spectral resolution, run time and mesh. | Compare equivalent FSR and loaded linewidth; record differences in coupling, material and dimension. |
| Grating radiation | Compact Tidy3D-domain Meep calculation measured 58.23% Gaussian-to-guide coupling at 1.55 µm versus 56.23% in the example. Expanded-domain and Lumerical comparisons in progress. [Measured control](verification/v011-grating-compact-tidy3d.json). | Compare the same collection aperture, direction and normalization. |
| Multimode coupling | In progress. Resolve two guided modes at each port, with straight-guide controls, directional launches and an independent modal reference. | Compare named mode indices and modal powers with both vendors; distinguish matched numbers from method-level references. |

An item is qualified only for the dimensionality, geometry, target and observables
identified in its measurement record. Versioned Python/notebook exports and
the local package are part of the delivery checks.

[Matched Lumerical and local Tidy3D port-mode measurements](CALIBRATED_VENDOR_CHECKS.md).
