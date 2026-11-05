# Commercial example review and workbench coverage

The 5 November 2026 review surveyed both official example indexes, then examined
public workflows for passive devices, photonic crystals, material fitting,
electro-optics, detectors and on-chip lasers. The comparison unit is a **specific
observable from a specific example**. A component family, a saved optical
design, a finite-field calculation and a calibrated device measurement have
different verification requirements. Commercial galleries also include circuit,
multiphysics and free-space applications. [Tidy3D examples](https://www.flexcompute.com/tidy3d/examples/) · [Ansys Application Gallery](https://optics.ansys.com/hc/en-us/categories/360002993513-Application-Gallery).

The library currently supplies **27 original parametric geometries**. The
[component guide](COMPONENT_LIBRARY.md) lists their physical controls, units,
port counts and curve approximations. The [finite-layout study](LAYOUT_FIELDS.md)
validates an actual design against its material, mesh and resource limits.
The [taper study](TAPER_FIELDS.md) has separate access-lead and excitation
conventions. Published modal comparisons are recorded in the [public
benchmark guide](PUBLIC_BENCHMARKS.md) and [component reference checks](COMPONENT_REFERENCE_CHECKS.md).

| Example family | Workbench design coverage | Observable required to qualify the corresponding vendor demo |
| --- | --- | --- |
| Straight guides and modes | Strip, rib and slot recipes; finite guide studies | Match cross section, wavelength, index data, polarization and boundaries, then compare effective index or power-normalized modal overlap. A guide-mode comparison does not qualify a driven field volume. [Tidy3D modal reference](https://www.flexcompute.com/tidy3d/examples/notebooks/ModeSolverFemwellBenchmark/) |
| Transitions and bends | Linear taper, circular bend, S-bend, Euler bend and strip-to-slot transition | Match launch/receive cross sections and compare guided output power and reflection after mesh and access-lead refinement. The Tidy3D Euler example includes a circular section and uses an effective-radius constraint. Our full clothoid uses minimum curvature radius. The Ansys tutorial uses a circular arc. [Tidy3D Euler bend](https://www.flexcompute.com/tidy3d/examples/notebooks/EulerWaveguideBend/) · [Tidy3D strip-to-slot converters](https://www.flexcompute.com/tidy3d/examples/notebooks/StripToSlotConverters/) · [Ansys bend geometry](https://optics.ansys.com/hc/en-us/articles/52523416098835-Getting-Started-with-Lumerical-FDTD-for-Photonic-Integrated-Circuits-Part-1) |
| Couplers, splitters and crossings | Straight directional and width-exchanging tapered couplers, Y-branch, 1×2, 2×2 and 1×4 MMIs, crossing | Match body/access dimensions, material stack, TE/TM modes and measurement planes. Compare per-port power, excess loss and spectral bandwidth. Ansys's MMI workflows use mode expansions and normalized S elements. [Tidy3D 1×4 MMI](https://www.flexcompute.com/tidy3d/examples/notebooks/MMI1x4/) · [Ansys MMI](https://optics.ansys.com/hc/en-us/articles/360042305194-Multi-Mode-Interference-MMI-Coupler) |
| Rings, interferometers and delays | Ring/bus, add/drop ring, racetrack, pulley ring, balanced MZI and serpentine | Measure resonance position, free spectral range, loaded Q, and through/drop spectra with resolved linewidths. The pulley recipe uses a bent bus with its own wrap angle. Commercial straight-bus and circuit examples use different coupling models. [Tidy3D add/drop ring](https://www.flexcompute.com/tidy3d/examples/notebooks/RingResonator/) · [Ansys ring parameter extraction](https://optics.ansys.com/hc/en-us/articles/360042800213-Ring-resonator-getting-started-Final-parameter-extraction) |
| Gratings and fibre interfaces | Width-corrugated Bragg grating, linear partial-etch coupler, partial-etch apodized fan grating | Compare fibre-mode overlap versus wavelength with specified beam waist, tilt, polarization, cladding thickness, etch and numerical aperture. Our fan has fixed radial pitch and smoothly varying tooth fill. The vendor notebook derives confocal ellipses, approximates them by concentric arcs and varies its pitch. [Tidy3D focusing grating](https://www.flexcompute.com/tidy3d/examples/notebooks/FocusedApodGC/) · [Ansys grating workflow](https://optics.ansys.com/hc/en-us/articles/360042305334-Grating-coupler) |
| Photonic crystals and cavities | Nanobeam holes, passive DBR, finite triangular-lattice line-defect slab with strip leads | Finite through power, cavity Q and a periodic Bloch band are different observables. The finite missing-row slab preserves through holes and finite boundaries. Reproducing a dispersion diagram requires periodic/Bloch excitation and a wavevector sweep. [Tidy3D crystal construction](https://www.flexcompute.com/tidy3d/examples/notebooks/PhotonicCrystalsComponents/) · [Ansys line-defect band structure](https://optics.ansys.com/hc/en-us/articles/360042039193-Bandstructure-of-photonic-crystal-waveguide-Line-defect-varFDTD) |
| Polarization and anisotropic couplers | The new tapered two-guide recipe defines width exchange and a constant gap. Dedicated polarization devices and anisotropic materials require further models. | Compare resolved TE/TM mode conversions, extinction, crystal orientation and stack. Geometry-only tapering does not establish adiabatic following or polarization rotation. [Tidy3D polarization splitter](https://www.flexcompute.com/tidy3d/examples/notebooks/PolarizationSplitterRotator/) · [Tidy3D lithium-niobate coupler](https://www.flexcompute.com/tidy3d/examples/notebooks/AdiabaticCouplerLN/) |
| Wavelength routers and circuit-scale networks | Existing MMI/ring/grating primitives provide pieces of filters. AWGs, star couplers and circuit compact-model composition remain open. | AWGs need two slab couplers, an array with prescribed optical path differences and multiport channel spectra. A single Bragg or MMI response is insufficient. [Ansys AWG workflow](https://optics.ansys.com/hc/en-us/articles/360042800633-Arrayed-waveguide-grating-AWG) |
| Active devices and on-chip lasers | Passive cavity layouts exist. Electrodes, doping, current injection, gain, carrier transport, noise and heating require separate solvers and data. | A Ge photodetector needs absorption plus carrier collection, a traveling-wave MZM needs electrical dynamics, and a quantum-well laser needs modal confinement, gain and carrier/photon feedback. [Ansys Ge photodetector](https://optics.ansys.com/hc/en-us/articles/360042828913-Germanium-on-Silicon-lateral-photodetector) · [Ansys traveling-wave MZM](https://optics.ansys.com/hc/en-us/articles/360042328774-Traveling-Wave-Mach-Zehnder-Modulator) · [Ansys MQW laser](https://optics.ansys.com/hc/en-us/articles/360041725214-Multi-Quantum-Well-MQW-Edge-Emitting-Laser) |
| Fabrication, materials and general optics | Polygons, holes, scalar or tabulated material design records and finite-height layouts. Hierarchical GDS/PDKs, dispersive fitting, fully anisotropic media, thermal/charge coupling, inverse design and many free-space applications remain open. | Preserve physical process layers, model fit bands, data provenance and a qualified physical study for each claimed application. [Tidy3D feature index](https://www.flexcompute.com/tidy3d/examples/) · [Ansys materials](https://optics.ansys.com/hc/en-us/articles/360034915033-Using-the-Material-Explorer-to-view-and-adjust-optical-material-models) |

## Using a published number

The Tidy3D 1×4 MMI demo reports about 0.4 dB excess loss at 1.55 µm for its
first design and below 0.2 dB after changing the design. These are two different
reference configurations. Our MMI geometry test checks that four distinct
access guides meet the body at their specified pitch. It produces no excess
loss measurement. A comparison to either vendor number requires the matching
design, stack, launch, normalization, monitor positions and power summation.
[Tidy3D MMI results](https://www.flexcompute.com/tidy3d/examples/notebooks/MMI1x4/).

The public Ansys MMI example sweeps the interaction length and gives a peak
near 32 µm for its particular geometry, taper and mode. This is a useful
candidate for future guided-port qualification. Matching a peak length
alone would leave the spectral transfer, modal normalization and excess loss
unchecked. [Ansys MMI sweep](https://optics.ansys.com/hc/en-us/articles/360042305194-Multi-Mode-Interference-MMI-Coupler).

For each quantitative comparison record the original reference values and
precision, units, geometry, dispersion, process stack, polarization, dimensional
reduction, source, reference planes, sampling convention, simulation stopping
condition, refinement changes, absolute/relative error and tolerance. A finite
dipole field, a converged guided S-parameter, and a fitted commercial circuit
response warrant independent records. Vendor examples informed coverage and
comparison targets. Their source code, datasets, project files and graphics
are excluded from the component generators.
