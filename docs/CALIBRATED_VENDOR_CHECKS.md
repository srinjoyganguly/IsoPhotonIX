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
