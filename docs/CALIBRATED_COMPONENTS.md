# Calibrated components checklist

This checklist tracks measured performance separately from field propagation.
All 27 library presets already have field evidence on Meep CPU and FDTDX CPU
and NVIDIA. [Field runs](LAYOUT_FIELDS.md#verification).

| Chunk | Local numerical evidence | Lumerical and Tidy3D comparison |
| --- | --- | --- |
| Straight and tapered two-port guides | Complete for the documented fundamental-mode studies. Complex S, both launch directions, mesh sensitivity and replay are available. [Taper checks](TAPER_PORTS.md). | Published mode/coupling comparisons exist for specific observables. A public taper matrix with identical geometry and reference planes is still needed. |
| Four-port directional coupler | Complete four-launch FDTDX NVIDIA matrix at the baseline mesh. Straight-reference through power is measured for each column. Other routes and sensitivities in progress. [CUDA receipt](verification/v011-fdtdx-coupler-gpu.json). | Published SOI supermode indices and transfer length compared across Lumerical, local Tidy3D and MPB. The bent multiport device has different access sections. |
| Three-port splitter | Complete three-launch Meep CPU matrix at the baseline mesh. Stem-to-arm powers at 1.55 µm are 0.40414 and 0.38880. Other routes and sensitivities in progress. [Meep receipt](verification/v011-meep-multiport-splitter.json). | Local cosine Y measured 1.008 dB excess loss at 1.55 µm; Tidy3D reports <0.2 dB average for a different optimized junction. Lumerical reports near-equal splitting for its optimized Y. These compare designs, not identical solvers. |
| Ring and add/drop spectra | In progress. Normalize bus/drop spectra, extract resonance and test spectral resolution, run time and mesh. | Compare equivalent FSR and loaded linewidth; record differences in coupling, material and dimension. |
| Grating radiation | In progress. Measure radiated power relative to incident power, distinguish upward radiation from fiber-mode overlap. | Compare the same collection aperture, direction and normalization. |
| Multimode coupling | In progress. Resolve two guided modes at each port, with straight-guide controls, directional launches and an independent modal reference. | Compare named mode indices and modal powers with both vendors; distinguish matched numbers from method-level references. |

An item is qualified only for the dimensionality, geometry, target and observables
identified in its measurement record. Versioned Python/notebook exports and
the local package are part of the delivery checks.

[Matched Lumerical and local Tidy3D port-mode measurements](CALIBRATED_VENDOR_CHECKS.md).
