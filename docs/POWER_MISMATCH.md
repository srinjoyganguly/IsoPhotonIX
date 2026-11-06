# Why the published and local powers differ

The comparisons use physical wavelengths and incident guided power. No gain or wavelength offset is fitted to the vendor spectrum. Each candidate material or geometry change is measured against the unmodified workbench calculation at the same mesh. The published Tidy3D and Lumerical examples describe different devices.

## Ring: follow the power through all four ports

The Lumerical text attachment contains the complex fundamental-mode amplitudes for every port. For excitation at port 1, we square the amplitude magnitudes and interpolate power onto the qualified workbench wavelengths. At 1550 nm the 24-cell/µm workbench ring and the published file give:

| Output | Workbench | Published Lumerical |
| --- | ---: | ---: |
| Through | 96.369% | 9.392% |
| Drop | 2.890% | 67.442% |
| Reflection | 0.00086% | 0.07616% |
| Add side | 0.00006% | 0.08245% |

The dominant difference at this wavelength is routing between through and drop. The four guided powers account for about 99.26% locally and 76.99% in the published file. The remainder can include radiation, absorption, other fields or numerical/port error. The port data alone do not separate those contributions. The Ansys example discusses resonant backscattering and recommends testing the coupling-region mesh, vertical domain and observation time. [Ansys ring and S-parameter attachment](https://optics.ansys.com/hc/en-us/articles/360042800213-Ring-resonator-getting-started-Final-parameter-extraction) · [Hash-checked power measurements](verification/v013-power-diagnostics.json).

**The two spectra put 1550 nm at different places on a resonance.** The nearest published drop maximum is at 1550.092 nm, only 0.065 published linewidths from the target. The qualified workbench fit centers at 1558.713 nm, 2.81 local linewidths away. At its own nearest *sampled* peak, the local ring drops 82.0%, versus 71.5% at the published peak. The local fitted width is 3.10 nm versus 1.42 nm in the reference. Both the center position and the linewidth differ. Comparing peaks helps diagnose the fixed-wavelength mismatch; it does not establish their angular order or improve the fixed-wavelength result. [24-cell spectrum, native fits and wavelength masks](verification/v012-ring3d-lumerical-r24-t3200.json).

The same two central peaks move by −4.5 nm from 20 to 24 cells/µm while their spacing stays at 27.0 nm. For this geometry the 100 nm gap occupies only 2.4 cells at the finest mesh. At 16 cells/µm, widening the absorbing boundary or monitor aperture separately changed measured power by at most 0.00045. These controls place the first correction effort on optical phase and boundary sampling. They do not establish fine-grid convergence or isolate each loss channel. [Mesh](verification/v012-ring3d-lumerical-20to24-comparison.json) · [Boundary](verification/v012-ring3d-lumerical-pml-comparison.json) · [Aperture](verification/v012-ring3d-lumerical-aperture-comparison.json).

The Tidy3D ring demonstrates why a single favorable wavelength is insufficient. At 1550 nm the 20-cell result drops 1.67% versus 2.16% from the digitized reference, yet the mean absolute drop-power difference across the common band is 17.51 percentage points. The 16-cell ring drops 71.98% at that same wavelength. [Tidy3D add/drop example](https://www.flexcompute.com/tidy3d/examples/notebooks/RingResonator/) · [Measured mesh sensitivity](verification/v012-ring3d-tidy3d-mesh-comparison.json).

The filed Ansys monitor provides total drop flux as well as the separate fundamental-mode S-parameter attachment. At 1550 nm the former gives **75.46%**, the latter **67.44%**. The comparison above uses the latter because the workbench projects onto that guided mode. The stored port mode was calculated near 1548.39 nm. Independent 128-cell/µm Tidy3D and MPB calculations of the filed straight cross-section using the extracted material data give effective indices of 1.90966 and 1.91176, versus 1.93252 in the saved Ansys project. This is a useful phase diagnostic for the lead; the round-trip phase also depends on the curved guide and couplers. The public project and the separate text and flux files have individual digests; their byte contents cannot establish identical solver runs. [Project and monitor audit](verification/v013-ring-project-audit.json) · [Tidy3D lead modes](verification/v013-ring-ports-tidy3d.json) · [MPB lead modes](verification/v013-ring-ports-mpb.json).

**Controlled boundary correction.** The baseline quadrature gives the nominal 400 nm straight guide a represented width of 406.25 nm at 16 cells/µm. Integrating the lateral silicon area exactly removes that error and changes the lead index at 1550 nm from 1.99050 to 1.97062. In the full 3D calculation the target drop power moves from 2.56% to 3.16%. The central peak locations stay on the same 0.5 nm samples, and the mean absolute drop-power error across the band increases from 12.41 to 13.36 percentage points. Thus the guide width bias is real, but correcting it alone does not reproduce the published device. [Native 16-cell control](verification/v013-ring3d-lumerical-r16-t3200-exact-area.json) · [Interface and wall-normal tests](verification/v013-ring-port-wall-mixing.json).

## Grating: distinguish a power-scale error from a spectral-shape error

At 1550 nm the 14-cell Lumerical reconstruction transfers 24.91% into the guide versus 40.03% published. At 1500 nm it transfers 33.68% versus 15.06% published. The sign reversal matters. To agree within five percentage points at all five samples, the same positive multiplier of the local power would have to be **at least 3.78** at 1600 nm and **at most 0.60** at 1500 nm. Thus a single constant source-power calibration cannot explain the difference. The channel returning to the incident fiber is also too small to account for it on its own. At 1550 nm its power is 0.36% locally and 0.42% published. [Lumerical grating and S-parameter workflow](https://optics.ansys.com/hc/en-us/articles/360042305334-Grating-coupler) · [Native grating measurement](verification/v012-grating3d-lumerical-r14.json).

At both local meshes the strongest of the five tested samples is 1525 nm. The vendor's denser S-parameter file peaks at 1557.62 nm. The five local samples do not resolve an exact local peak wavelength. A separate slab phase-matching estimate puts the 14-cell root near 1522 nm and the 96-cell root near 1543 nm. That model omits curved teeth, the taper, fiber overlap and transverse radiation. It supports testing the representation of the etched film, but cannot predict the full device efficiency. At 1550 nm the 100 nm etch is only 1.4 cells deep at 14 cells/µm. [Slab diagnostic and assumptions](verification/v012-grating-lumerical-phase-screen.json) · [Measured mesh comparison](verification/v012-grating3d-vendor-mesh.json).

The Tidy3D uniform grating is a separate Gaussian-to-guide experiment. The local 14-cell result is 35.67% versus 38.02% published at 1550 nm, with a mean absolute difference of 2.95 percentage points over five samples. A successful reproduction of either workbench notebook tests reproducibility of its own fields. Commercial agreement is evaluated separately at identical wavelengths and output ports. [Tidy3D uniform grating](https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/) · [Finite-width measurements](FINITE_GRATING_3D.md).

## What the independent notebook runs establish

Both original v0.12.0a1 14-cell archives were downloaded signed out and checked against the published SHA-256 list. Their original notebook cells ran on the RTX 3080 Ti in fresh working directories with new reference and device stages. All five guided powers and the displayed field were exact in both runs. For Lumerical 43 of 45 arrays were byte-identical; for Tidy3D 49 of 50 were byte-identical. Only integrated-energy traces varied, within the originally declared tolerances. The public executed notebooks redact the local output path. The reproduction receipt distinguishes the original asset digest, original cells, private execution digest and published file digest. [Fresh numerical checks](verification/v013-grating-notebook-replays.json) · [Lumerical notebook](verification/v013-grating-notebook-replays-lumerical.ipynb) · [Tidy3D notebook](verification/v013-grating-notebook-replays-tidy3d.ipynb).

## What would qualify a correction

The following are workbench investigation targets, not vendor accuracy specifications. Keep the incident-source, wavelength-mask, finite-time and native-device checks mandatory.

| Observable | Acceptance check |
| --- | --- |
| Ring center, width and absolute power | Track the same identified line across two meshes. Aim for a center displacement from the vendor below one resolved published linewidth, through/drop power within 5 percentage points at 1550 nm, and through/drop mean absolute differences below 5 points across the common band. Examine the linewidth and backward channels separately. |
| Finite-width grating transfer | Aim for absolute differences below 5 percentage points at 1550 nm and on average across the sampled band. Check all five values and peak location, with at most a 2-point mesh change at each sample before treating the spectrum as mesh-stable. |
| Port phase and weak reflections | Retain the existing per-observable gates and independent propagation and reference-plane controls. [Port phase tests](PORT_PHASE_SENSITIVITY.md) · [Finite-height controls](PLANAR_PORT_CALIBRATION.md#finite-height-control-results). |

In the interface, open **Calibrated components**, select a measured ring or grating, and inspect **Published spectrum comparison**. The powers in this guide can be regenerated from the native arrays and the public Lumerical text attachment:

```bash
python -m scripts.diagnose_component_power --lumerical-ring FDTDtoINTERCONNECT.txt --output power-diagnostics.json
```
