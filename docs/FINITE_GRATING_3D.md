# Finite-width 3D grating measurements

The FDTDX study launches a tilted beam or a step-index fiber mode into a finite silicon grating. A separate reference measures the incident downward power through a horizontal plane below the source. In the device run, the six recorded E/H components on the silicon guide cross section are projected onto its backward quasi-TE fundamental mode at each wavelength. Dividing that modal power by the measured incident power gives the coupling fraction. The grating teeth, taper, beam and devices have finite transverse extent.

## Geometries and external observables

| Preset | Local measurement | External reference |
| --- | --- | --- |
| Compact | A 3 µm-wide, eight-tooth grating and 3 µm taper illuminated with a 1.5 µm waist. The narrow control reduces the teeth and residual silicon film to 2 µm, keeping the computational domain, guide, beam and aperture fixed. | This is a workbench test of transverse geometry. |
| Tidy3D | Straight 10.8 µm-wide teeth, 620 nm period, 80% fill, 260 nm silicon with a 160 nm etch. The 50 µm linear taper connects to a 500 nm output guide. The residual 100 nm silicon film extends laterally past the finite teeth in the published 3D notebook. The wafer has a 2 µm buried oxide and a top oxide ending at z = 0.68 µm. An air-side, y-polarized 5.4 µm-waist beam hits at 14.5°. | The published **3D** Gaussian-to-TE0 plot gives approximately −4.20 dB (38.02%) at 1550 nm. The paper's 56.23% belongs to its earlier **2D** sweep. The local program uses a uniform mesh and a tilted paraxial surface source. [Tidy3D notebook](https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/) |
| Lumerical | A 25 µm-radius sector, 15 µm aperture and a nonlinear taper join a 500 nm guide. The 38 designed radial periods have 671.3 nm pitch, 39.92% duty cycle and a 100 nm etch into 220 nm silicon. The computation clips the complete geometry to the filed nominal FDTD domain, including its x maximum at 16 µm. The 9 µm step-index fiber core and 50 µm cladding are tilted by 10°. The top oxide and buried oxide are 0.70 and 1.00 µm. | The supplied native 50-bin S-parameter file gives `abs(S21)² = 0.40027` at 1550 nm after interpolation in wavelength. Local excitation uses the weak-guidance LP01 transverse mode. The impressed sheet uses its single propagation index for the tilt phase and H/E impedance. A source-only native FDTDX check preserved the stepped 3D material and measured a maximum relative impedance error of 2.1 × 10⁻⁷ at the 1550 nm LP01 index of 1.440997. The time-domain fields evolve on the actual tilted 3D core/cladding grid. Silicon and oxide use the project's real 1550 nm Palik permittivities throughout the comparison band. Lumerical uses its own vector fiber mode, material fit and nonuniform mesh. [Lumerical project](https://optics.ansys.com/hc/en-us/articles/360042305334-Grating-coupler) |

The reference extraction reads only the publicly downloaded input files. The receipt includes vendor file/plot digests, five calibrated plot samples, the numerical S-matrix values, layer limits and actual fiber/guide port centers. The original projects and images are omitted from the distribution. [Reference measurements](verification/v012-grating-commercial-references.json) · [Extraction program](../scripts/reference_finite_grating.py).

The full Tidy3D reference used 788 × 168 × 74 cells on the RTX 3080 Ti Laptop GPU. At 1550 nm the measured incident angular centroid was −14.50059°, and upward contamination in the nominally downward beam was 7.75 × 10⁻⁵ of incident power. [Independent source measurement](verification/v012-grating3d-tidy3d-source.json).

A separate local full-vector Tidy3D mode solve also checked the Lumerical fiber cross section. At 12 cells/µm its modal index differs from the scalar LP01 value by −1.17 × 10⁻⁵, and the squared Ey-profile overlap is 0.999981. Two independent modal resolutions are recorded. [Fiber-mode control](verification/v012-grating-fiber-mode-reference.json) · [Local mode program](../scripts/validate_grating_fiber_mode.py).

## Fields, convergence and directions

- The uniform grid samples the straight finite ridges using fractional horizontal/vertical cell coverage. The curved Ansys sectors use 4 × 4 lateral subsamples and fractional vertical coverage. Component permittivities are averaged to the electric Yee locations. These approximations require mesh checks, particularly for the roughly 100 nm etched features. The incident reference uses the actual truncated numerical aperture, which can differ from a full-space analytical beam normalization.
- The archived upper-plane and output-guide phasors each contain Ex, Ey, Ez, Hx, Hy and Hz. The application preview shows the real Ey component on an x–z plane near y = 0, at 1550 nm. Coordinate axes, material slices, sampled modes, native source fields and stage hashes accompany the numerical arrays.
- The final worker samples electromagnetic energy every 4 fs. It requires two consecutive 60 fs observation windows after the pulse with a tail-to-peak energy ratio below 10⁻⁴ and a maximum relative change of the measured spectrum below 0.002. Each accepted wavelength also requires incident power above 1% of the maximum and guided power between zero and 102% of incident power. Mesh change and agreement with external data require separate evaluation.
- In the free-space cases, complex subtraction removes the incident reference before separating upward TE/TM plane waves on the finite air aperture. The archive contains the two-dimensional angular spectrum, its spatial-frequency axes and eight polar bins between 0° and 80°. These are aperture-dependent estimates of reflected power. Lateral rays beyond the collection aperture and grazing bins are excluded. The polished-fiber case instead projects the scattered field onto the time reversal of the incoming fiber field on the same horizontal plane.
- A 12 cells/µm exploratory compact run measured 18.28% at 1550 nm using the preliminary endpoint energy check. Its five-sample power fractions were 12.64%, 16.01%, 18.28%, 21.89% and 20.58%. The vendor-sized devices and the final interval-max qualification are tracked separately.

## Reproduce locally

With a managed NVIDIA FDTDX worker and the installed workbench command:

```bash
photonics-workbench calibrate grating-3d --solver fdtdx --device gpu --geometry-3d compact --resolution 12 --output compact-run
photonics-workbench calibrate grating-3d --solver fdtdx --device gpu --geometry-3d tidy3d --resolution 12 --output tidy3d-run
photonics-workbench calibrate grating-3d --solver fdtdx --device gpu --geometry-3d lumerical --resolution 12 --output lumerical-run
```

Use separate new result directories. The program export includes the solver sources, native environment lock, settings, Python launcher and notebook. The direct numerical worker also supports a persistent `--checkpoint` directory. Reference and device acquisition use separate native CUDA processes, and only complete hash-verified stages are reused.

**Guided-input angular radiation.** Select `compact-radiation` to launch the fundamental quasi-TE mode down the strip. A straight strip measures the incident guided amplitude and the launch background. A third native empty-domain run supplies the receiving Gaussian basis on the same air aperture. The device reports TE-guide-to-Gaussian efficiency, reflected guided power, upper-aperture flux and polar-angle bins. Comparison with the same compact device driven by the Gaussian checks reciprocity across independent acquisitions.

New checkpoints bind each native acquisition to solver, SciPy, JAX, numerical source and actual CUDA device/driver. The device tag is salted per local cache. The native solver and coordinator compare the selected device before reusing a completed stage. [FDTDX phasor detector](https://fdtdx.readthedocs.io/en/latest/api/fdtdx.PhasorDetector.html) · [FDTDX guided source](https://fdtdx.readthedocs.io/en/stable/api/fdtdx.ModePlaneSource.html).

## Measured finite-width results

| Geometry | Workbench at 1550 nm | Comparison |
| --- | ---: | --- |
| Tidy3D uniform grating, 12 cells/µm | 35.2994% Gaussian to guide | Published 3D curve 38.0189%. The five-wavelength mean absolute difference is 2.5773 percentage points. |
| Lumerical curved grating, 12 cells/µm | 22.2103% fiber to guide | Published S21 file 40.0266%. The five-wavelength mean absolute difference is 16.2332 percentage points. |
| Compact, 12 cells/µm | 18.2758% Gaussian to guide | Narrow teeth at identical source and domain give 17.5538%. |
| Compact, 16 cells/µm | 22.5344% Gaussian to guide | The finer mesh changes the nominal coupling by 4.2586 points. |
| Compact with the guide excited, 12 cells/µm | 21.1016% guide to Gaussian | The upward finite-aperture estimate is 36.8530%. Reversing the excitation changes the nominal Gaussian transfer by 2.8258 points and by up to 3.2303 points over the five wavelengths. |

The Tidy3D matched-geometry comparison uses 1500, 1525, 1550, 1575 and 1600 nm. Workbench minus the digitized vendor powers are −2.3055, −5.5363, −2.7195, −0.6401 and −1.6852 percentage points respectively. Each reference/device pair passed the incident support and finite-window checks at all five samples. [Tidy3D geometry receipt](verification/v012-grating3d-tidy3d-r12.json) · [Native fields and modes](verification/v012-grating3d-tidy3d-r12-arrays.npz) · [Compact mesh, width and reciprocity](verification/v012-grating3d-controls.json) · [Guided angular spectrum](verification/v012-grating3d-compact-radiation-r12.json).

The Lumerical fiber’s original source pair failed the strict equality check. Its two solver-generated profiles had a fitted global amplitude ratio of 1.00000016 and sub-10⁻⁷ residual shape differences. Those data are retained as a rejected diagnostic. The revised source uses a fixed LP01 amplitude and saves both temporal offset arrays. Independent new reference/device acquisitions produced identical E/H source and time-offset hashes. All five bins passed the numerical checks. The local coupling fractions at 1500–1600 nm were 31.65%, 31.90%, 22.21%, 11.21% and 3.88%. The substantial spectral mismatch survives the source correction. [Qualified Lumerical device](verification/v012-grating3d-lumerical-r12.json) · [Native fields, source delays and modes](verification/v012-grating3d-lumerical-r12-arrays.npz). [Rejected input check](verification/v012-grating-lumerical-rejected-source-pair.json).
