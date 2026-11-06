# Vendor comparison measurements

The calibration checklist distinguishes a matched observable from a method
comparison. A transmission coefficient is comparable only when both studies
use the same structure, materials, launch mode, reference planes, normalization
and collection aperture.

## Matched Lumerical and Tidy3D cross sections

The published Lumerical SOI evanescent coupler has two 0.5 µm wide,
0.2 µm high rails, a 50 nm gap, an index-1.5 substrate and air above.
At 1.55 µm its published even/odd indices are 2.378022 and 2.317864 and its
full-transfer length is 12.8827 µm. The installed Tidy3D CPU mode solver,
using the same rail geometry and material indices at 12.5 nm spacing, measured
2.389748 and 2.329884, giving 12.94620 µm. The length differs by 0.493%.
An independent prior MPB refinement reached 12.96196 µm.
[Lumerical example](https://optics.ansys.com/hc/en-us/articles/360042304694-Evanescent-waveguide-couplers) ·
[Tidy3D local mode-solver workflow](https://www.flexcompute.com/tidy3d/examples/notebooks/ModalSourcesMonitors/) ·
[Mesh series and measured data](verification/v011-vendor-port-modes.json) ·
[MPB refinement](COMPONENT_REFERENCE_CHECKS.md).

The same optional script checks the published 220 nm planar silicon TE mode,
the two guided modes of our offset slab and strip, and the even/odd modes of
our coupled section. The local Tidy3D calculations use uniform grids with
local subpixel handling disabled; the mesh dependence is recorded. It runs
locally with the Tidy3D Python package and accepts `--output NEW_JSON_FILE`.
[Published Lumerical slab-mode step](https://optics.ansys.com/hc/en-us/articles/1500007378182-Inverse-design-of-an-efficient-narrow-band-grating-coupler) ·
[Comparison program](../scripts/calibrated_port_references.py).

A coupling length inferred from two straight-guide supermodes differs in meaning
from a complete port matrix of a device with bends and access guides. The
device records retain their own uncertainties and comparison conditions.

At the finer **6.25 nm** spacing, local Tidy3D measured 2.382815 and 2.323811 for the Lumerical section, giving **13.13462 µm**. Its transfer length differs from the published Lumerical value by **1.956%**, larger than the 12.5 nm result. The 50 nm gap puts some interface locations on different Yee-grid sampling offsets as the mesh changes. The unsmoothed sequence gives an empirical sensitivity range; a single favorable mesh is insufficient to establish agreement at 0.5%. For the workbench 125 nm gap, local Tidy3D changed from 25.4064 to 23.5646 µm while the Meep/MPB straight-section estimate is 21.8532 µm. Reproduce the extended local mode series with `--refine-couplers`. [All mode samples](verification/v011-vendor-port-modes.json) · [Independent workbench cross sections](verification/v011-multiport-mpb.json).

## Matched spectra and mode conversion

- The loaded Lumerical add/drop reconstruction supplies four measured spectra at 561 wavelengths. Its planar free spectral range is 40.25 nm compared with the published 25.4549 nm. The comparison includes absolute wavelength and a finite-time linewidth qualification. [Measured ring spectrum](verification/v011-ring-loaded-lumerical-fine.json) · [Lumerical final parameter extraction](https://optics.ansys.com/hc/en-us/articles/360042800213-Ring-resonator-getting-started-Final-parameter-extraction) · [Tidy3D add/drop ring](https://www.flexcompute.com/tidy3d/examples/notebooks/RingResonator/).
- The grating calculations include separate empty-domain beam references. The Tidy3D compact setup is 2.00 percentage points above its 56.23% reference. The Lumerical article reconstruction at 40 cells/µm is 6.28 points below 55.7% at 1310 nm. The article also supplies a total-output-aperture measurement for a separate comparison from our near-output guided mode. [Grating measurements](RING_GRATING_CALIBRATION.md) · [Lumerical broadband grating](https://optics.ansys.com/hc/en-us/articles/360042304774-Broadband-grating-coupler-2D) · [Tidy3D grating](https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/).
- For the two-mode planar guide, local Tidy3D measured even/odd indices 3.17002 and 2.14855. Meep's 64-cell port modes returned 3.16518 and 2.12984. The largest absolute difference from independent mode matching is 0.001243 in power fraction across the full 4 × 4, three-wavelength matrix. [Multimode measurements](CALIBRATED_MULTIMODE.md).

## Tidy3D optimized Y-junction reconstruction

Meep simulated a finite-height 220 nm silicon device using the thirteen published junction widths and the sinusoidal S-bend equation in the example text. The source was calibrated against a separate straight guide. Both outputs were measured independently. The Tidy3D plot doubles its upper-arm power, so that same positive excess-loss observable is compared here.

| Wavelength (µm) | Workbench, 20 cells/µm (dB) | Workbench, 24 cells/µm (dB) | Tidy3D plot (dB) |
| ---: | ---: | ---: | ---: |
| 1.500 | 0.2452 | 0.2728 | 0.2365 |
| 1.525 | 0.1704 | 0.1841 | 0.1595 |
| 1.550 | 0.1497 | 0.1580 | 0.1318 |
| 1.575 | 0.1534 | 0.1688 | 0.1372 |
| 1.600 | 0.1654 | 0.1855 | 0.1520 |

The five-point mean absolute differences are 0.0135 and 0.0305 dB. Graph-readout uncertainty is approximately ±0.004 dB. Summing both independently measured arms at 1.55 µm gives 0.1498 and 0.1585 dB. The executable notebook constructs its bends with PhotonForge's default Path.s_bend. Its plotted contours differ from the site's sine equation by about 0.05 µm at sampled positions. A separate regeneration of the PhotonForge footprint matched eleven figure cuts within one pixel. Workbench freezes the real central-wavelength permittivities across the band. The two grids and path constructions have independent uncertainties. [Measured complex outputs and spectrum](verification/v011-published-splitter.json) · [Regenerated footprint](verification/v011-published-y-photonforge-outline.json) · [Published Tidy3D example](https://www.flexcompute.com/tidy3d/examples/notebooks/YJunction/) · [PhotonForge API](https://docs.flexcompute.com/projects/photonforge/en/v1.0.1/_autosummary/photonforge.Path.html) · [Related Lumerical Y branch](https://optics.ansys.com/hc/en-us/articles/360042800353-Y-branch).

The independently regenerated PhotonForge notebook footprint was also simulated. Its 20-cell result gives **0.16288 dB at 1.55 µm**, versus 0.13176 dB read from the Tidy3D curve. The five-point mean absolute difference is 0.03540 dB. The regenerated outline agrees more closely with the vendor's geometry drawing, while the written sine-path model has the smaller 20-cell spectral difference. Both observations are retained. [Normalized fields and numerical receipt](verification/v011-published-y-photonforge-r20.json).

## Lumerical output-aperture control

The filed detector is 40 µm along the output guide and spans 1 µm. At 1310 nm the 40-cell reconstruction measures 46.09% of incident Gaussian power through that aperture, compared with 55.7% published. The fundamental-mode projection at the same plane is 40.68%. The seven-point mean absolute direct-flux discrepancy is 15.01 percentage points. The separately measured *near-port* 60-cell TE0 efficiency of 55.62% has a different collection definition. The numerical reports preserve the raw per-frequency flux and each wavelength mask. [Direct-aperture spectrum](verification/v011-grating-ansys-output-r40.json) · [Near-port 60-cell spectrum](verification/v011-grating-ansys-project-r60.json) · [Lumerical broadband grating](https://optics.ansys.com/hc/en-us/articles/360042304774-Broadband-grating-coupler-2D).
