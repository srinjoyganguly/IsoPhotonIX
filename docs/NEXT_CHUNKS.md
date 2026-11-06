# What works and what remains

This is the product backlog through the current GPU mesh-refinement campaign. For each piece, the
release gate describes a result that a user could independently reproduce.
Previous evidence for the [27-family library](verification/v09-checks.json),
[finite-layout fields](LAYOUT_FIELDS.md) and [taper ports](TAPER_PORTS.md) has
a narrower scope than commercial coverage.

The current calibration runs and the next Circuit Layout milestone have separate acceptance criteria. Once the ring and grating measurements are evaluated and published, persistent named-port connections can proceed while failed power and phase gates remain open. [Measured checklist](CALIBRATED_COMPONENTS.md) · [Per-stage GPU costs](GPU_TIMING.md).

| Chunk | Already usable | Remaining implementation and evidence |
| --- | --- | --- |
| Layout editing and circuit construction | 27 parametric families, polygon holes, finite-height routed guides, undoable placement, port alignment, component deletion and tab-local undo/redo recovery | Add persistent connection constraints and explicit disconnect, auto-routing with minimum radius and clearance, hierarchy, Boolean operations, layer-aware GDS import/export and process-specific DRC. Round-trip ports, layers, IDs, polarities and material precedence through imported layouts. |
| Calibrated passive components | Full coupler and splitter matrices on Meep CPU and FDTDX CPU/CUDA, planar and finite-height multimode matrices, a TM0/TM1 joint with movable planes, free polygon layouts with cardinal ports, two vendor loaded-ring and finite-width grating reconstructions in full 3D on NVIDIA, resolved loaded-ring Q estimates and a compact grating angular spectrum. [Evidence and exact target matrix](CALIBRATED_COMPONENTS.md). | Establish convergence of absolute ring peak positions, grating transfer and mesh-sensitive weak signals. Extend custom ports to oblique orientations, higher modes and additional fabrication stacks with separate native measurements. |
| Numerical studies and validation | Explicit targets, independent port launches, signed complex amplitudes, native field samples, reproducible Python/notebook exports, calibrated-job restart, and named mesh, aperture, PML and duration studies. | Generalize the bounded controls to user-defined sweeps and queued studies. Extend memory estimates beyond the current vendor gratings, track modes through crossings and apply the measured weak-reflection and phase-change criteria to additional geometries. |
| Materials and fabrication stacks | Constant-index passive layout simulation and a [single-film spectrum](PLANAR_FILM.md) from constant or tabulated n/k, checked against independent Meep runs at several wavelengths | Fit documented complex-index data to passive causal material models, enforce validity bands and check plane-wave dispersion/absorption against analytic slabs on each supported solver. Extend to anisotropy, sidewalls, etch operations, conformal meshing and process-specific layers. |
| Passive resonators and lasers | A 1D passive Fabry–Pérot benchmark against COMSOL Table 1, and finite-height isolated-ring spacings, plus loaded-ring linewidth and decay Q estimates under an explicit two-equal-bus model. [Loaded-ring measurements](LOADED_RING_3D.md). | Qualify intrinsic and external loss beyond the equal-bus model and measure cavity mode volume. Then verify carrier/photon rate equations, threshold and light–current curves against a named active device before adding spatial gain and thermal coupling. |
| Open-source engines and distribution | A pip wheel, managed Meep/MPB, FDTDX CPU/CUDA12 and BeamZ CPU installations, and a dedicated desktop browser window, verified under WSL2. BeamZ has separate CPU/GPU propagation and CPU mode benchmarks. [Installation and worker support](PACKAGING.md) | General interactive MPB bands/mode tracking, BeamZ parity with specific finite-height GUI studies, cross-platform distribution, upgrades/migrations, unattended offline installation and platform-specific installer testing. Sign and publish only the platforms actually qualified. |
| External comparison coverage | Matched Lumerical and Tidy3D port modes, measured Y and grating spectra, loaded and isolated ring comparisons and the passive COMSOL cavity. The comparison guide separates scalar, spectral and method-level references. [Commercial measurements](CALIBRATED_VENDOR_CHECKS.md). | Seek public full complex matrices with the same materials, planes, modes and launch normalization. Further refine mesh-sensitive phase, finite-height loaded spectra and broadband collection. |

## Order of the next physics work

1. Extend the [calibrated component measurements](CALIBRATED_COMPONENTS.md) with additional reference spectra, paired mesh studies for weak signals and modes beyond the current presets.
2. Further refine the loaded ring’s absolute pole wavelengths and external versus intrinsic Q estimates on the same resonance across mesh, boundary and observation-time controls. Use the [1D cavity benchmark](PASSIVE_CAVITY.md) as a numerical
   control for the fitting method.
3. Extend the tested narrow-band Palik material control beyond its reference slab and 16-cell ring. General finite-height absorption and anisotropy require causality, loss, interface and independent measurement checks. [Existing slab and ring evidence](CALIBRATED_COMPONENTS.md).
4. A laser threshold model with a documented active material and injection
   conditions, checked against a public experimental or manufacturer dataset.

[Vendor component coverage](VENDOR_COMPONENT_COVERAGE.md) ·
[Detailed laser gates](COMPONENT_ROADMAP.md) ·
[Public scalar comparisons](COMPONENT_REFERENCE_CHECKS.md).
