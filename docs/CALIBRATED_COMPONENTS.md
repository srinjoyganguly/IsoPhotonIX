# Calibrated components checklist

This checklist tracks measured performance separately from field propagation.
All 27 library presets already have field evidence on Meep CPU and FDTDX CPU
and NVIDIA. [Field runs](LAYOUT_FIELDS.md#verification).

| Chunk | Local numerical evidence | Lumerical and Tidy3D comparison |
| --- | --- | --- |
| Straight and tapered two-port guides | Complete for the documented fundamental-mode studies. Complex S, both launch directions, mesh sensitivity and replay are available. [Taper checks](TAPER_PORTS.md). | Published mode/coupling comparisons exist for specific observables. A public taper matrix with identical geometry and reference planes is still needed. |
| Four-port directional coupler | In progress. Launch all four separated access ports, normalize to reference guides and measure bar/cross power and phase. | Compare coupling against even/odd splitting and identify which published device conditions match. |
| Three-port splitter | In progress. Launch from the stem and each output arm, with reciprocity, power-budget and symmetry checks. | Match source polarization, geometry and the definition of excess loss before comparing amplitudes. |
| Ring and add/drop spectra | In progress. Normalize bus/drop spectra, extract resonance and test spectral resolution, run time and mesh. | Compare equivalent FSR and loaded linewidth; record differences in coupling, material and dimension. |
| Grating radiation | In progress. Measure radiated power relative to incident power, distinguish upward radiation from fiber-mode overlap. | Compare the same collection aperture, direction and normalization. |
| Multimode coupling | In progress. Resolve two guided modes at each port, with straight-guide controls, directional launches and an independent modal reference. | Compare named mode indices and modal powers with both vendors; distinguish matched numbers from method-level references. |

An item is qualified only for the dimensionality, geometry, target and observables
identified in its measurement record. Versioned Python/notebook exports and
the local package are part of the delivery checks.
