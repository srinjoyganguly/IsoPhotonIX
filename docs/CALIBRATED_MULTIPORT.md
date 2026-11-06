# Finite-height directional coupler and Y splitter

Choose the directional coupler to measure a four-port matrix or the symmetric Y
splitter for a three-port matrix. Every column of a full result has a separate
native input excitation. Each input has its own straight-guide incident
reference. Off-diagonal entries retain both amplitude and phase at the physical
planes. All simulations in this guide run locally.

## Physical presets

The coupler uses two 0.50 × 0.25 µm silicon cores (constant index 3.48) in
index-1.45 cladding. Their parallel central section is 2 µm long with a 0.125 µm
edge gap. Two 3 µm cosine approaches on each rail connect to straight access
guides with 2.5 µm centre spacing. The Y splitter starts from one identical
strip. Over a 4 µm cosine fan-out it becomes two identical separated guides.
The exact finite-height contours and all source/port positions are saved in
`config.json`. They define these two studies. Changing the circuit geometry
would require new mode, isolation and numerical checks.

| Study | Port order, from left to right |
| --- | --- |
| Coupler | left lower, left upper, right lower, right upper |
| Splitter | left centre, right lower, right upper |

Both end planes are at x = ±4.75 µm. The two possible source planes are at
x = ±5.5 µm. The ports record the isolated fundamental quasi-TE mode over the
specified 2 × 2.5 µm transverse aperture. Material extends continuously into
the absorbing boundaries at the ends. The default wavelengths sample three
equally spaced frequencies centred at 1/1.55 cycles/µm. Waveguide ports in
the general layout editor describe geometry; a measured matrix belongs to its
exact, named physical preset.

## Meep measurement and outputs

In its Meep/MPB environment, run from the checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/calibrate_multiport.py \
    --case coupler --resolution 16 --jobs 2 --output coupler-results
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/calibrate_multiport.py \
    --case splitter --resolution 16 --jobs 2 --output splitter-results
```

The output directory must be new. On the shared WSL test host the full 16-cell/µm Meep splitter took about 26 minutes with three simultaneous columns; the four-column coupler took about 41 minutes. Host load and mesh determine actual runtime. `--launch-port left_lower` (coupler) and
`--launch-port left_center` (splitter) give individually excited columns for
controlled studies. Inspect `input_ports` before interpreting a partial matrix.
The `--resolution`, `--pml-um`, `--aperture-um` and `--minimum-ringdown`
controls permit independent sensitivity tests. Thickening the PML changes the
outer domain while the device, access lengths and input/output planes keep the
same physical coordinates.

For a long run, keep the result directory. Each input's straight-reference E/H
Fourier data, measured incident amplitude, mode indices and settling history
are checkpointed before evolving its device. Each finished device column is
also checkpointed. To continue an interrupted run, repeat its original
command with `--resume` and the same directory. The solver verifies the
original simulation settings, implementation and native archive hashes. For
explicitly separate jobs, first run `--stage reference`, then repeat the same
command with `--resume --stage device`. A reference-only job produces no S
matrix. An interrupted individual stage is rerun from its beginning.

MPB solves both signed propagation directions on the local cross section. The
centre Ey phase fixes each mode basis. The local guide must be bound, have the
correct signed group velocity, and have Ey as its dominant electric component.
At the illuminated plane Meep subtracts the incident reference E/H streams
before extracting the reflected amplitude. Each emerging power-normalized mode
coefficient is divided by the measured incoming coefficient at that frequency.
Consequently `S[f,out,in]` has power `|S|²`. The through phase includes the
entire distance between the recorded planes.

Each stage checks sampled field energy and changes in the complex monitor
streams after the source turns off. The output retains native E/H DFTs and
incident/outgoing complex coefficients in `*-monitors.npz`, stage histories
and mode diagnostics beside them, and the measured matrix in `arrays.npz`.
Inspect valid bins, per-column fundamental power sums, the largest singular
value (passivity diagnostic for *coherent* input combinations), and complex
reciprocity residuals. A deficit in fundamental power includes radiation,
unmeasured modes and discretization error; it does not identify absorption.

[Meep modal decomposition](https://meep.readthedocs.io/en/stable/Mode_Decomposition/)
· [Native normalization and incident-field subtraction](https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/).

## Commercial and independent references

The [Lumerical evanescent-coupler example](https://optics.ansys.com/hc/en-us/articles/360042304694-Evanescent-waveguide-couplers)
provides a numerical even/odd index splitting for two 500 × 200 nm guides with
a 50 nm gap, a substrate and an air upper cladding. The workbench's calibrated
coupler has a different thickness, gap, cladding and curved approaches. Use the
[matched cross-section reproduction](COMPONENT_REFERENCE_CHECKS.md) for the
published index/transfer-length comparison. Compare the *local* uniform
supermodes of the present circuit to its full FDTD power transfer separately.
Integrating the even/odd index splitting along x assumes adiabatic evolution
and neglects reflection and radiation. This gives an independent predictive
check with its own quadrature and transverse mesh sensitivity.

The [Tidy3D directional coupler](https://www.flexcompute.com/tidy3d/examples/notebooks/DirectionalCoupler/)
uses a GDS-derived profile with 30° sidewalls, 20 nm dilation, 220 nm height and
a substrate. It measures the forward TE-mode powers at both outputs and
searches for a 50/50 crossing. The workbench uses vertical sidewalls and
constant homogeneous cladding. Device transmission or phase agreement with
that GDS case requires its distinct three-dimensional geometry. A three-bin
spectrum cannot establish a 50/50 crossing outside those bins.

[Ansys Y branch](https://optics.ansys.com/hc/en-us/articles/360042800353-Y-branch)
reports a near 50/50 division for a symmetric input. The [Tidy3D Y junction](https://www.flexcompute.com/tidy3d/examples/notebooks/YJunction/)
uses a 13-segment optimized input and dispersive material records, reporting
less than 0.2 dB average loss over 1.5–1.6 µm. Their normalized arm balance is a
relevant symmetry check. Absolute loss and scattering phase require matched
geometry, material, mode normalization and plane positions. In particular, an
equal arm ratio can coexist with substantial radiation in an unoptimized Y.

The published Tidy3D notebook describes a sinusoidal bend in its prose, then
creates the actual bend with PhotonForge's `Path.s_bend`. Two distinct
reconstructions test those choices. For the software-generated outline, the
workbench replayed the original path and polygon-union calls locally with
PhotonForge Community 2.0.0. The resulting 52-vertex arms matched the original
notebook geometry illustration within 0.00988 µm over eleven transverse cuts.
One pixel spans 0.01452 µm in that illustration. The notebook does not record
the generating PhotonForge version. The checked-in vertices let Meep run
without PhotonForge. [Generated polygons and illustration measurements](verification/v011-published-y-photonforge-outline.json) · [PhotonForge Community](https://docs.flexcompute.com/projects/photonforge/en/latest/getting_started.html).

Run this separate model from the checkout in a Meep environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/measure_published_y_outline.py run \
    --vertices docs/verification/v011-published-y-photonforge-outline.json \
    --resolution 20 --output published-y-photonforge
```

The model measures a straight incident reference with the same source and
mirror symmetries as its device. It records the upper-arm guided power and
compares `−10 log10(2 P_upper)` with the published curve. The lower port in
this run is symmetry-related. The sinusoidal reconstruction measures both
arms without a transverse mirror and retains their independent powers.

Compare one measured numerical control to its baseline, for example:

```bash
python scripts/compare_multiport_controls.py coupler-results coupler-finer --output mesh-change.json
```

This checks the case, physical polygons, port identities, excitation, actual
target, solver version, wavelengths and archived complex matrix before
reporting changes. It intersects qualified bins and reports phase differences
only when both amplitudes exceed 0.01 (power 0.0001). A common monitor-norm
stopping tolerance can resolve the main transmitted amplitude while a much
smaller reflected coefficient remains sensitive. A mesh change gives a
measured sensitivity for those inputs; it cannot supply an asymptotic
convergence order from two meshes.

## Electric-amplitude convention and measured matrices

For a signed-k eigenmode, Meep 1.35 returns the **signed** normal-flux
overlap multiplied by a positive normalization. Its native same-mode
coefficient therefore carries the sign of the group velocity. This study
multiplies that coefficient by `sign(vgrp)` as well as fixing the local Ey
phase. In the electric basis, multiplying the mode by amplitude +1 gives a
positive centre Ey at its own plane for either propagation direction. A check
using the *actual stored* straight-reference monitor streams found that the
historic negative-x amplitude was about π radians from the directly sampled
electric field, whereas the corrected amplitudes for both directions were
within 0.003 radians of the measured Ey phase. [Native Meep implementation](https://github.com/NanoComp/meep/blob/v1.35.0/src/mpb.cpp#L963-L982) ·
[Device-port audit](verification/v011-meep-multiport-phase-audit.json).

The following are the authoritative matrices. Every stage settled at all three
sample frequencies. The superseded overlap-basis receipts are retained with
their original digests in the verification directory. Changing basis reverses
entries whose two ports are on the same end of the circuit; opposite-end
transmission entries, powers and singular values keep their measured values.
The original E/H streams were byte-checked during the conversion.
[Four-port Meep matrix](verification/v011-meep-multiport-coupler-electric.json) ·
[Three-port Meep matrix](verification/v011-meep-multiport-splitter-electric.json).

At 1.55 µm, launching into the lower-left coupler port gave 0.873874 bar
power, 0.073204 cross power and a 0.077294 forward-normalized cross fraction.
The largest complex reciprocity residual across the three bins was
7.42 × 10⁻⁶; the largest singular value was 0.985968. Independent MPB
eigensolves estimated a 21.8532 µm central straight-section transfer length
and an adiabatic cross fraction of 0.039901. The latter excludes radiation and
reflection and does not reproduce this finite circuit's cross fraction.
On the separately matched Ansys SOI geometry at 80 cells/µm, the computed
full-transfer length was 12.9732 µm versus 12.8827 µm published (0.702%
difference). [MPB modes, mesh and quadrature controls](verification/v011-multiport-mpb.json).

At the same wavelength, the centre-left splitter launch delivered 0.4041365
and 0.3887955 to the lower and upper fundamental modes. The normalized split
was 50.967% / 49.033%, with 1.008 dB excess loss from the sum of the two
measured output powers. The largest complex reciprocity residual was 0.001257
and the largest singular value was 0.925115. The measured imbalance and loss
are retained. A symmetric geometric layout does not establish exact sampled
field symmetry or reproduce the optimized Tidy3D junction's published
insertion loss.

To update the phase basis of a historic workbench Meep multiport result while
keeping the original result:

```bash
python scripts/reproject_meep_multiport.py historic-run --output electric-run
python scripts/check_multiport_reference_phase.py electric-run --output reference-phase.json
```

The second command requires the original Meep version and a complete saved run.
Comparison tools refuse to mix the two phase conventions.

The same electric-phase convention applies to new finite-height straight, gap and tapered Meep two-port runs. Those exports include the signed group velocities, the electric-field phase anchors, the unmodified solver coefficients and the direction factors at every frequency. Earlier archived workbench versions reproduce their historic reflected amplitudes. Updating a historic reflection to the electric basis changes its sign. The measured transmitted amplitudes and powers are identical under this basis conversion. [Independent analytic step and direct field checks](verification/v011-meep-fresnel-phase.json).

### Device excess relative to an independently measured straight guide

The auxiliary quantity is `arg(S_out,in / T_straight)`, where `T_straight`
is the directly measured far/near complex fundamental-mode amplitude of an
uninterrupted guide. The guide matches the incident rail, material, aperture,
mesh and the physical x planes. For a routed off-lane output the straight
path defines a longitudinal baseline, not a path along the device bends.
The physical S matrix and its phase are retained separately. This comparison
uses neither a fitted phase shift nor a static eigenmode estimate of
propagation. [Measured straight/physical/excess phases](verification/v011-multiport-excess-phase.json).

At 1.55 µm in the splitter, FDTDX's legacy CUDA sampling and Meep differed by
2.2346 and 2.2390 radians in the two forward device phases (FDTDX minus Meep);
their measured straight reference phases differed by 2.2026 radians.
With Yee-aligned FDTDX materials the raw device differences were 0.07283
and 0.07794 radians and the straight difference was −0.03350 radians. The
corresponding excess-phase differences were 0.10633 and 0.11144 radians.
These measured excess phases are distinct from the physical power and phase.

For the coupler, the matched Meep straight guide was measured in the lower
input lane (y = −1.25 µm). Its modal through phase and directly sampled central
Ey through phase agreed within 0.00318 rad across the three bins. At 1.55 µm
the Yee-aligned CUDA and Meep physical phases differed by 0.16941 rad (bar)
and 0.13355 rad (cross). After accounting for the independently measured
straight-phase difference of −0.03350 rad, their excess-phase differences were
0.20291 and 0.16705 rad. CUDA delivered 0.79177 bar and 0.07670 cross power,
compared with Meep's 0.87387 and 0.07320. These powers and phases quantify the
remaining device-level disagreement at 16 cells/µm.
[Native coupler straight reference and sampled electric fields](verification/v011-meep-coupler-straight-phase.json).

### Measured sensitivity of the incident columns

Each row is an independently evolved straight reference and device. The
coupler input was the lower-left port. The splitter input was the centre trunk.
All sixteen stages met the same energy and DFT settling checks at all three
frequencies. Differences below are maxima over those frequencies and output
ports. Power is a fraction of incident fundamental-mode power. The phase
columns use wrapped phase differences where both modal amplitudes exceeded
0.01. Forward refers to the two measured right-side outputs.

| Device | Change from the nominal run | Max absolute power change | Max phase change, all ports (rad) | Max forward phase change (rad) |
| --- | --- | ---: | ---: | ---: |
| Coupler | Port aperture width 2 → 2.25 µm | 0.00000212 | 0.0000263 | 0.00000664 |
| Coupler | PML thickness 0.75 → 1 µm | 0.0000197 | 0.02016 | 0.0000217 |
| Coupler | Longer DFT ringdown | 0.0000636 | 0.00846 | 0.0000821 |
| Coupler | Mesh 16 → 20 cells/µm | 0.04914 | 2.18441 | 0.96763 |
| Splitter | Port aperture width 2 → 2.25 µm | 0.00000479 | 0.0000140 | 0.00000321 |
| Splitter | PML thickness 0.75 → 1 µm | 0.00000652 | 0.0000806 | 0.00000928 |
| Splitter | Longer DFT ringdown | 0.0000662 | 0.01333 | 0.0000487 |
| Splitter | Mesh 16 → 20 cells/µm | 0.00943 | 1.05128 | 0.98938 |

The longer run accumulated at least 72 µm/c after the end of the pulse, with
a cap of 96 µm/c. PML enlargement kept the physical planes and access lengths
fixed. Mesh refinement dominated both forward-power and physical-phase
sensitivity. The isolated lead's measured MPB index at 1.55 µm changed from
2.494685 to 2.547224; the separately resolved 80-cell/µm MPB section gave
2.548773. Changes in longitudinal propagation accumulate over the 9.5 µm
between physical planes.

At 20 cells/µm the 1.55 µm coupler delivered bar/cross powers of
0.909446/0.043183, or 4.5330% normalized cross power versus 7.7294% at 16.
The splitter delivered 0.406182/0.396547, with a normalized 50.6002% lower-arm
fraction and 0.95431 dB excess loss. Its nominal values were 50.967% and
1.00764 dB. Weak returns show the largest phase shifts. The full nominal
matrices remain the measured 16-cell results; these refinements bound the
observed sensitivity of the two tested columns. Two mesh points do not
establish a converged continuum scattering phase.
[Native control matrices, convergence checks and numerical digests](verification/v011-meep-multiport-sensitivities.json).
For example, the lower-port power remained 0.40414 for Meep versus 0.36595
for the corrected FDTDX run.
