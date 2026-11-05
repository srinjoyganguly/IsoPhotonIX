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
