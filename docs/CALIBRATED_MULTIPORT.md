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

The output directory must be new. `--launch-port left_lower` (coupler) and
`--launch-port left_center` (splitter) give individually excited columns for
controlled studies. Inspect `input_ports` before interpreting a partial matrix.
The `--resolution`, `--pml-um`, `--aperture-um` and `--minimum-ringdown`
controls permit independent sensitivity tests. Thickening the PML changes the
outer domain while the device, access lengths and input/output planes keep the
same physical coordinates.

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
