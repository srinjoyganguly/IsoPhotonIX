# Planar port calibration

These two Meep CPU studies measure complex scattering amplitudes at physical port planes. Columns are independently launched input modes and rows are collected output modes. Power fractions are squared amplitude magnitudes relative to each input's straight-guide reference. Both studies use 1.50, 1.55 and 1.60 µm wavelengths, fixed lossless silicon and oxide indices of 3.48 and 1.44, and an invariant out-of-plane dimension.

## Two TM modes on each side of an offset joint

The 0.375 µm slab has a default lateral displacement of 0.125 µm. Meep launches TM0 and TM1 from both sides and records the two resulting 4 × 4 matrices at two sets of physical planes. The default planes are 1.25 µm from the joint, with a second observation at 0.75 µm. The port gauge fixes the sign and phase of Ey a quarter of the core width above each guide centre. The independent interface solver matches Hz and its permittivity-weighted normal derivative, including outgoing radiation modes. [Mesh receipts](verification/v012-planar-tm.json) · [Complex matrices, raw projections and nominal Hz field](verification/v012-planar-tm-arrays.npz).

| Cells/µm | Max power fraction error vs independent | Max complex amplitude error vs independent | Max weak-reflection phase error | Max complex plane-translation error |
| --- | --- | --- | --- | --- |
| 48 | 0.006950 | 0.10684 | 0.13336 rad | 0.03510 |
| 64 | 0.004869 | 0.05927 | 0.07299 rad | 0.01967 |
| 80 | 0.003762 | 0.03688 | 0.04491 rad | 0.01256 |

Weak-reflection statistics use 18 coefficients with amplitude at least 0.01 and below 0.1 in both calculations. The largest 64→80 change is 0.001107 in power fraction and 0.02241 in complex amplitude. On the default 64-cell mesh at 1.55 µm, 74.80% of left TM0 power emerges as right TM0 and 15.09% as right TM1. The selected port distance may range from 0.75 to 1.75 µm on a native mesh plane, and the offset from 0 to 0.25 µm. Each changed design requires its own checks.

At 1.55 µm, a separate vendor comparison uses the published Ansys Lumerical asymmetric slab's four analytical TM indices. At 128 cells/µm the largest absolute index discrepancy is 0.00000117 for Meep/MPB and 0.0000543 for local Tidy3D. In the workbench joint's own straight section, the corresponding Meep/MPB modal indices are 2.91564 and 1.54158. Local Tidy3D gives 2.91649 and 1.54206. The comparisons cover lead modes. The independent interface solver supplies the full junction matrix. [Vendor comparison, model inputs and sources](verification/v012-planar-tm-vendor.json) · [Ansys example](https://optics.ansys.com/hc/en-us/articles/360042304134-Asymmetric-slab-dielectric-waveguide).

## Fundamental Ez ports on editable polygon layouts

The default square-corner elbow has west and north ports. At 1.55 µm and 32 cells/µm it reflects 14.425% and transmits 0.109% of west-input power into the collected fundamental modes. The remaining power can include radiation, other modes and finite-domain error. The measured geometry class includes X/Y straight guides, the elbow at 32 and 40 cells/µm, the elbow with moved planes, a three-port tee and a four-port crossing. Every input has a native reference and device run. [Seven designs and controls](verification/v012-layout-ports.json) · [Complex matrices, early/late projections and nominal Ez field](verification/v012-layout-ports-arrays.npz).

| Control | Measured result |
| --- | --- |
| Rotate a 3.5 µm straight guide by 90° | Both axes transmit at least 99.9737% and reflect at most 0.00964%. Straight transmission phase differs by up to 0.366 rad from the native lead-index prediction over that length. |
| Independently solve the identical elbow and port planes in the frequency domain | At 32 cells/µm the largest discrepancy is 0.002013 in power fraction and 0.00753 in complex amplitude. At 40 cells/µm the values are 0.00004697 and 0.00271. |
| Refine the native elbow mesh from 32 to 40 | The largest changes are 0.01035 in power fraction and 0.03422 in complex amplitude. |
| Move the west plane outward and the north plane inward by 0.25 µm | The measured matrix differs from the propagation-phase prediction by up to 0.02503 in complex amplitude and 0.00460 in power fraction. |
| Compare matched X/Y lead modes with local Tidy3D | The largest same-mesh index difference over three wavelengths is 0.04317 at 32 cells/µm and 0.00114 at 40. The Tidy3D solve used a uniform mesh with local subpixel averaging disabled. |

The Helmholtz solver supplies the independent same-device S matrix. The local Tidy3D study supplies the lead-mode comparison. Each comparison uses the same width, indices, polarisation and wavelength as its native counterpart. The field archives retain the real, final-time broadband field and physical coordinates. These small residual fields follow the pulse and serve as spatial samples. [Helmholtz solver](../scripts/reference_layout_ports.py) · [Tidy3D mode calculation](../scripts/compare_layout_port_modes.py).

Use one to eight simple silicon polygon outlines with 3–64 vertices per outline, each coordinate within ±2.5 µm. Place two to four ports, each with a unique name and one distinct outward direction from −x, +x, −y and +y. The normal coordinate must lie 1.25–2.25 µm outward from the origin, on a native mesh plane. The transverse coordinate may be up to ±1.5 µm. Supported widths are 0.25–0.50 µm. Each port needs a full-width silicon attachment 1 µm inward from its plane. Keep each polygon at least 0.25 µm inward from each plane and other leads clear of the 2 µm source and monitor corridor. The solver adds straight leads through the absorbing boundary. The layout route accepts 24, 32 and 40 cells/µm. [Model schema and default layout](../backend/planar_port_layout.py).

## Run and replay

With the workbench CLI and its Meep CPU worker installed, these commands reproduce the measured settings of the two verified exports. Each output directory must be new.

```sh
photonics-workbench calibrate multimode-tm --solver meep --device cpu --resolution 48 --offset-um 0.125 --port-distance-um 1.25 --output planar-tm-run
photonics-workbench calibrate layout-ports-2d --solver meep --device cpu --resolution 32 --output planar-elbow-run
```

Select 64 cells/µm for the default TM matrix displayed above. To measure a custom polygon, add `--layout design.json` to the layout command. The verified designs are in each layout receipt's `parameters.design` field. For example, from a source checkout, export the tested crossing as a CLI input.

```sh
python -c "import json; from pathlib import Path; d=json.loads(Path('docs/verification/v012-layout-ports.json').read_text()); Path('crossing.json').write_text(json.dumps(d['native_runs']['cross32']['parameters']['design']))"
photonics-workbench calibrate layout-ports-2d --solver meep --device cpu --resolution 32 --layout crossing.json --output planar-crossing-run
```

Each completed study creates `python-notebook-results.zip` with the frozen worker, exact settings, reference report and numerical arrays. To replay the elbow, extract the whole archive and execute either entry point in its extracted directory. The same sequence applies to the TM archive. Set `MEEP_PYTHON` to the Meep interpreter path if using an existing unmanaged installation.

```sh
python -m zipfile -e planar-elbow-run/python-notebook-results.zip elbow-replay
cd elbow-replay
python run.py --output python-result
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=21600 --output executed.ipynb calibration.ipynb
```

The Python and notebook entries each launch fresh native stages. Their output directories contain `reproduction.json` with array comparisons and tolerances. Maximum S amplitude differences from the original were 4.25 × 10⁻¹⁰ for TM and 7.67 × 10⁻¹¹ for the elbow. [TM replay results](verification/v012-planar-tm-replays.json) · [Elbow replay results](verification/v012-layout-ports-replays.json).
