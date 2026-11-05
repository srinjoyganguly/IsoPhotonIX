# Public component reference checks

Two published examples supply ten numerical targets. The workbench uses its
rectangular strip and directional-coupler recipes to construct the transverse
core geometry. MPB solves the full-vector Maxwell eigenproblem at fixed
frequency. The measured quantities describe uniform waveguide sections.

## Ansys Lumerical MODE directional coupler

The [Ansys evanescent-waveguide example](https://optics.ansys.com/hc/en-us/articles/360042304694-Evanescent-waveguide-couplers)
specifies two 500 nm wide, 200 nm tall rails with a 50 nm gap at 1.55 µm.
The material stack comes from inspection of the [public example project](https://optics.ansys.com/hc/en-us/article_attachments/360056406114).
Its substrate record has index 1.5 and ends at z = 0. Its two core records
have index 3.5 and span z = 0 to 0.2 µm. Their inner edges are 25 nm
from the symmetry plane. The FDE background index is 1.0. The reference file
was inspected offline as data. The source URL and SHA-256 identify those
inputs. The project file and its vendor material database stay outside the
workbench distribution.

We rotate the transverse direction to workbench y and shift the core centre
to z = 0, keeping the substrate flush with its lower face. The catalog
instance supplies both rail rectangles. Propagation is invariant along x.
MPB resolves the two quasi-TE supermodes in different lateral symmetry
classes. The dominant Ey mirror overlap checks the measured field parity.
Material probes check both claddings, a core interior and the gap.

For two identical uniform guides with equal coherent excitation of the
two supermodes, the ideal full-transfer length is
`lambda/(2*(n_even-n_odd))`. This is the length for full power exchange.
A full beat period is twice this length. Individual indices, their small
difference, and the resulting transfer length are each compared against the
published FDE values. The article also compares other propagation solvers
and different gaps. Those results describe separate physical/numerical cases.

## Tidy3D dispersive strip

The [waveguide plugin example](https://www.flexcompute.com/tidy3d/examples/notebooks/WaveguidePluginDemonstration/)
gives the effective indices, group indices and effective areas for both
polarizations of a 500 × 220 nm silicon strip in silica. It names the
materials `cSi/Li1993_293K` and `SiO2/Horiba`. At the centre wavelength,
1.55 µm, their sampled permittivities are 12.075663669334666 and
2.1249983698346497 + 0.0006277085542436616i. The lossless MPB calculation
uses the real permittivity at each of the three frequencies. The omitted
imaginary term is recorded. No effective index or material value is fitted
to the vendor results.

The group index uses `n_eff + f * dn_eff/df`, with frequencies at 0.995,
1 and 1.005 times the centre frequency. Both material dispersion and waveguide
dispersion enter this derivative. The effective area uses the entire vector
electric intensity, `(integral |E|² dA)² / integral |E|⁴ dA`.
The raw output records the dominant electric component, signed lateral parity
overlap, modal root residual and independently sampled permittivity. Scientific
field files contain the complex E field, y/z grid coordinates and scalar
permittivity at the centre frequency.

## Reproduce

An installed v0.9 workbench with the Meep/MPB profile can reproduce the cases directly:

```bash
photonics-workbench compare-examples --case all --output component-comparisons
```

Choose a new directory. The command retains each mesh, a larger transverse
window and the reference deltas. The GUI displays the qualified bundled
record. For a source checkout, run the dedicated script in the tested Meep/MPB Python environment from the
source checkout. Select a new output directory.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .local/meep-conda/bin/python \
  scripts/benchmark_component_references.py \
  --output .local/component-reference-rerun
```

The [recorded report](verification/public-component-comparisons.json)
includes all meshes and a larger transverse window at the finest mesh.
The fixed-frequency roots must meet a relative residual of 1e-7 and the
expected transverse electric component must account for over half of the
field's squared norm. The comparison retains the discrepancy for every metric.
Raw centre-frequency mode files are generated locally with hashes. Mesh and
window changes measure numerical sensitivity. They are not bounds on the
combined modelling, sampling and reference error.

An optional local SDK calculation evaluates the original complex materials
and compares the public notebook with another discretization. Use the
installed Tidy3D Python environment to run the following command. It uses the
local CPU mode solver, with subpixel averaging explicitly disabled.

```bash
python scripts/inspect_tidy3d_strip.py --report /tmp/tidy3d-strip-replay.json
```

For example, the local SDK's quasi-TE index changes from 2.5328785 at its
default grid resolution of 15 to 2.4527168 at 60. The published reference is
2.44624476. The discrepancy includes the finite mesh and local solver
conventions. The notebook does not export the reference field array or
identify the exact generating backend version. At grid resolution 60, the local SDK quasi-TM area is 0.34343145 µm²,
0.273% from the independent MPB value but 3.553% from the public notebook
value. These two comparisons use different references. The supplementary
SDK record retains all six quantities at all three grids.
[SDK replay](verification/tidy3d-local-plugin-check.json) ·
[Tidy3D local solver options](https://docs.flexcompute.com/projects/tidy3d/en/stable/configuration/reference.html)

The comparisons cover uniform-section modes and ideal coupling length.
Driven field snapshots, launch efficiency, finite-device S parameters and
radiative losses each require their own matched reference and convergence study.

## Choosing comparable examples

The vendor libraries include examples whose visible numbers answer different
questions. Ansys's ring walkthrough begins with a desired 25.6 nm FSR and
a target bandwidth. Treating those design targets as measured ring spectra
would give a misleading comparison. Tidy3D's tunable MZI reports a 3 dB length
in a finite directional-coupler circuit. That includes its access geometry.
The two uniform-guide supermodes above instead predict full transfer in a
constant section. The Tidy3D DBR example compares plotted reflection with an
analytical marker for a normal-incidence multilayer. This differs physically
from the catalog's laterally corrugated, finite-height on-chip Bragg cavity.
Each would require its own source, boundaries, readout and numerical samples.
[Ansys ring](https://optics.ansys.com/hc/en-us/articles/360042800293-Ring-resonator-getting-started-Design-and-initial-simulation) ·
[Tidy3D tunable MZI](https://www.flexcompute.com/tidy3d/community/notebooks/TunableMZI/) ·
[Tidy3D DBR and cavity](https://www.flexcompute.com/tidy3d/examples/notebooks/DistributedBraggReflectorCavity/).

## Measured results

These are the final 256 cells/µm runs in the 3 × 3 µm window. The last two columns give the absolute changes from 160 cells/µm and from enlarging the window to 4 × 4 µm at the finest mesh. Each change uses the units of its row.

| Example / quantity | Published | Local | Difference | Mesh change | Window change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ansys Lumerical MODE Even supermode index | 2.378022 | 2.37782136 | 0.0084371% | 0.0006107 | 2.008e-08 |
| Ansys Lumerical MODE Odd supermode index | 2.317864 | 2.31803103 | 0.0072061% | 0.0005756 | 1.592e-05 |
| Ansys Lumerical MODE Supermode splitting | 0.0601584 | 0.0597903356 | 0.61183% | 3.511e-05 | 1.59e-05 |
| Ansys Lumerical MODE Full-transfer length (µm) | 12.8827 | 12.961961 | 0.61525% | 0.007616 | 0.003449 |
| Tidy3D waveguide plugin Quasi-TE effective index | 2.44624476 | 2.44739467 | 0.047007% | 1.575e-07 | 1.065e-08 |
| Tidy3D waveguide plugin Quasi-TM effective index | 1.7895686 | 1.78454812 | 0.28054% | 0.0001464 | 6.446e-06 |
| Tidy3D waveguide plugin Quasi-TE group index | 4.17353874 | 4.1780543 | 0.10819% | 0.0002117 | 4.719e-08 |
| Tidy3D waveguide plugin Quasi-TM group index | 3.73989668 | 3.71168097 | 0.75445% | 0.0008117 | 0.0002962 |
| Tidy3D waveguide plugin Quasi-TE mode area (µm²) | 0.19145485 | 0.190468774 | 0.51504% | 2.463e-05 | 1.54e-08 |
| Tidy3D waveguide plugin Quasi-TM mode area (µm²) | 0.35608204 | 0.342495548 | 3.8156% | 4.727e-05 | 4.38e-05 |

The small supermode splitting amplifies the two individual index discrepancies when calculating transfer length. The quasi-TM mode area has the largest discrepancy from the Tidy3D example. Its numerical definition depends on spatially sampled fields; it remains an unresolved reference difference. Selecting a coarser grid simply because one metric moves closer to the public value would hide that behavior.
