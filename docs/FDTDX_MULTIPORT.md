# FDTDX finite-height multiport calibration

The local study has a four-port directional coupler and a three-port Y splitter.
It uses the physical polygons, materials, reference planes and optical band in
`backend/calibrated_geometry.py`, also used by the Meep measurement. These
are bounded test circuits. Editing a catalog component does not transfer a
calibration to the edited geometry.

## Run the numerical worker

Select a Workbench FDTDX CPU or CUDA environment, or the matching interpreter
in a standalone study export. From the source checkout:

```bash
"$FDTDX_PYTHON" scripts/calibrate_fdtdx_multiport.py \
  --case coupler --device gpu --resolution 16 --output coupler-cuda
"$FDTDX_PYTHON" scripts/calibrate_fdtdx_multiport.py \
  --case splitter --device cpu --resolution 16 --output splitter-cpu
```

The run requires a fresh output directory. To continue an interrupted run,
repeat the command with `--resume`. The worker checks the geometry, solver
versions, numerical implementation, target and SHA-256 of each completed
stage before reuse. `--reference-cache DIRECTORY` can share a measured
straight-through reference between studies with identical material masks,
source and monitor grid coordinates, pulse, spectral and convergence settings,
versions and hardware. Each device input is still excited independently.
Changing mesh, PML or aperture changes the study identity. GPU requests verify
JAX's actual device on initialization and at the end of every simulation.
The local mode preparation uses the pinned CPU eigensolver.

## Choosing the Yee material grid

`--material-model legacy-x` reproduces the original adjacent-x tangential
averaging. `--material-model yee-aligned` translates the same cell-center
permittivities to the physical Ex, Ey and Ez locations. Ex averages its four
neighboring scalar-centre samples in y/z, Ey in x/z, and Ez in x/y. At the
outer boundary the nearest value is extended. Each field computation uses
its recorded component permittivity grid and explicit model name; the two
models have independent reference caches. These are staircase models for the
specified physical polygons, without a general conformal boundary fit.

A direct, all-3D native CUDA test of the splitter's central launch at 16
cells/µm measured arm powers of 0.380738 and 0.333903 at 1.55 µm with the
legacy placement. With Yee-aligned placement the independently sampled powers
were 0.365951 and 0.365945. The simulation used no y-mirror reduction.
[Measured material comparison](verification/v011-fdtdx-yee-material-control.json).
Use `scripts/inspect_multiport_materials.py --output PORT_REPORT.json` in the
FDTDX CPU environment to compare the local mode indices separately.

The complete Yee-aligned splitter on CUDA passed the sampled checks at
16 cells/µm. All three launch ports were simulated independently. The largest
mirror-related guided-power difference across the three frequencies was
0.00000867, and the largest reciprocity difference in complex S was 0.000797.
Against Meep's signed-electric-phase result at the same resolution, the
largest guided-power difference was 0.03846 and the complex relative L2
difference was 0.15235. [CUDA splitter matrix and matched Meep comparison](verification/v011-fdtdx-yee-splitter-gpu16.json).

All four corrected-grid coupler launches also passed at 16 cells/µm on CUDA.
At 1.55 µm, the first launch delivered 0.79177 guided power to the through
port and 0.07670 to the cross port. The corresponding Meep values were
0.87387 and 0.07320. Across the band, the largest difference in guided power
was 0.08211. [CUDA coupler matrix and matched Meep comparison](verification/v011-fdtdx-yee-coupler-gpu16.json).

## Straight-guide phase control

A straight guide places the near and far monitors 9.5 µm apart. Moving both
planes inward by one grid cell provides two short phase increments that
resolve the direction of propagation. The modal projection and a point sample
of the native electric field independently measure each increment. At
1.55 µm and 16 cells/µm, Meep measured +0.6540 radians per cell and FDTDX
with `legacy-x` measured +0.6685. Their measured 9.5 µm phases were
+99.4040 and +101.6066 radians, respectively. The independent short spacing
accounts for the 2.2026-radian difference within 0.001 radians at this
wavelength. It does not establish agreement on a device or correct its
phases. [Measured near/far phase and sampling positions](verification/v011-multiport-phase-audit.json).

Repeating that control with `yee-aligned` gave +0.65375 radians per cell
and +99.37049 radians over 9.5 µm. Against the same Meep reference the
measured difference falls to −0.03350 radians at 1.55 µm and stays within
0.03683 radians across the band. [Corrected-grid phase control](verification/v011-fdtdx-yee-straight-phase.json).

An independent normal-incidence reflection test uses a material step with
indices 1 and 2. Both launch directions have a known complex electric-field
reflection at a port 1 µm from the interface. At 64 cells/µm the absolute
complex error against Fresnel theory was 0.00074 from the left and 0.00339
from the right. The projected reflection agreed with direct native Ey
subtraction within 0.000003 across both directions and both meshes.
Run `scripts/check_fdtdx_port_phase.py --output fresnel.json` in the FDTDX
CPU environment. [Signed reflected fields and analytic control](verification/v011-fdtdx-fresnel-phase.json).

Run the straight-guide phase control in each solver environment.

```bash
"$MEEP_PYTHON" scripts/verify_multiport_phase.py --solver meep --device cpu --output meep-phase
"$FDTDX_PYTHON" scripts/verify_multiport_phase.py --solver fdtdx --device gpu \
  --material-model legacy-x --output fdtdx-phase
```

## Mesh and hardware controls

The coupler's nominal 0.125 µm gap samples as 0.1667, 0.125 and 0.100 µm
at 12, 16 and 20 cells/µm. Local even/odd mode indices then imply uniform
transfer lengths of 27.66, 19.63 and 15.56 µm. At 32 cells/µm the gap
samples as 0.125 µm and the estimated transfer length is 21.12 µm.
The high-resolution MPB calculation gives 21.85 µm for the same section.
The local Tidy3D series and its distinct material sampling are recorded
alongside these results. [Measured grid gaps and local supermodes](verification/v011-fdtdx-yee-coupler-grid.json).

The 12- and 16-cell CUDA splitter matrices differ by up to 0.02164 in
guided port power. The relative L2 difference in their complex physical-plane
matrices is 1.003 using the 16-cell matrix as the denominator. The two meshes
give similar center-input output powers with appreciably different phases.
[Splitter mesh comparison](verification/v011-fdtdx-yee-splitter-gpu12.json).

To verify target repeatability, pass completed studies at identical settings
to `scripts/compare_fdtdx_targets.py --cpu CPU_DIR --gpu CUDA_DIR --output checks.json`.
The comparison verifies each archived native field checksum, material grid,
source profile, time-step count, monitor-plane field discrepancy and complex
S matrix. Its acceptance limits apply to execution on different hardware at
one fixed mesh.

For a separate first-input mesh control,
`scripts/measure_fdtdx_mesh_column.py --case coupler --device gpu --resolution 24 --compare BASE16 --output fresh-column`
measures a straight reference and one device launch. Its output records
`s_column[frequency, outgoing_port]` and identifies the sole incident port.

## What the matrix measures

The SOI-like test circuit here is a silicon core in *homogeneous cladding*,
without a separate substrate or oxide/air interface. Lengths are in µm. The
0.5 × 0.25 straight leads have a center-to-center separation of 2.5 at opposing
same-side ports. Each port window contains one guide. The FDTD cells sample the
actual outline; longitudinal tangential permittivities use the adjacent-cell
mean. Finite-grid transverse sampling can introduce a difference between
mirror-related powers, even for a mirror-symmetric scalar footprint.

For each incident port, one straight reference and one device solve use the
same source direction and canonical quasi-TE mode. Each recorded `Ex,Ey,Ez,Hx,Hy,Hz`
phasor is projected onto the frequency-specific numerical mode. Positive Ey at
the center fixes the arbitrary modal phase gauge. Incoming and outgoing
directions follow the port's side of the device. The launched port's outgoing
reference component is subtracted, and each complex outgoing amplitude is
divided by the independently measured incoming reference amplitude. The two
guide directions, ports on the same side, and split arms are distinct. Directional
and mirror symmetry are measured after these computations.

`arrays.npz` contains complex `s[frequency, outgoing_port, incoming_port]`,
`incident`, `reference_through`, `valid_column`, optical frequency and wavelength.
Each per-stage NPZ contains native monitor phasors, projected directional
amplitudes, source profiles and electromagnetic energy samples. The JSON report
gives the port order and the full shape and convergence history.

The terminal conditions measure both the total non-PML energy and change in
complex amplitudes over consecutive checkpoints. Two consecutive checkpoints
must satisfy both the 1e-4 energy-tail ratio and 0.002 amplitude-change limit.
Straight-guide power, reciprocity and the largest singular value of the
complete S matrix appear separately. The guided-mode power sum omits radiation
and unmonitored modes; a missing guided fraction alone cannot be assigned to
a particular loss mechanism. Small reflections, phase and splitting ratios are
mesh-, aperture- and boundary-sensitive. The sampled checks use a straight
reference power difference ≤0.03, complex reciprocity difference ≤0.05 and
squared maximum singular value ≤1.05; they characterize this integration
test and do not establish a general solver error bound.

The official APIs document the source and monitor used here. The study uses
the installed FDTDX 0.6.2 implementations. [ModePlaneSource](https://fdtdx.readthedocs.io/en/latest/api/fdtdx.ModePlaneSource.html) ·
[PhasorDetector](https://fdtdx.readthedocs.io/en/latest/api/fdtdx.PhasorDetector.html).
