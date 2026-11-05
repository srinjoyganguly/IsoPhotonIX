# What works and what remains

This is the product backlog following the v0.10 preview. For each piece, the
release gate describes a result that a user could independently reproduce.
Previous evidence for the [27-family library](verification/v09-checks.json),
[finite-layout fields](LAYOUT_FIELDS.md) and [taper ports](TAPER_PORTS.md) has
a narrower scope than commercial coverage.

| Chunk | Already usable | Remaining implementation and evidence |
| --- | --- | --- |
| Layout editing and circuit construction | 27 parametric families, polygon holes, finite-height routed guides, undoable placement, port alignment, component deletion and tab-local undo/redo recovery | Add persistent connection constraints and explicit disconnect, auto-routing with minimum radius and clearance, hierarchy, Boolean operations, layer-aware GDS import/export and process-specific DRC. Round-trip ports, layers, IDs, polarities and material precedence through imported layouts. |
| Calibrated passive components | Fields in all 27 sampled families on Meep CPU and FDTDX CPU/CUDA, and complex fundamental-mode two-port data for bounded straight/gap and linear-taper studies | First qualify a four-port directional coupler and a splitter using normalized launches in both directions. Then measure ring/add-drop spectra, grating radiation efficiency and multimode coupling. Sweep mesh, run time, PML and port aperture, check reciprocity and power accounting, and compare each measured quantity with a named reference. |
| Numerical studies and validation | Explicit Meep CPU and FDTDX CPU/CUDA jobs, immutable results, Python/notebook replays and paired taper mesh sensitivity | Generalize convergence and parameter sweeps beyond the taper; add time/domain/PML sensitivities and mode tracking. Support queued studies, resource estimates, checkpoint/restart and multi-observable regression references. Establish thresholds per observable, especially weak reflection and phase. |
| Materials and fabrication stacks | Constant-index passive layout simulation and a [single-film spectrum](PLANAR_FILM.md) from constant or tabulated n/k, checked against independent Meep runs at several wavelengths | Fit documented complex-index data to passive causal material models, enforce validity bands and check plane-wave dispersion/absorption against analytic slabs on each supported solver. Extend to anisotropy, sidewalls, etch operations, conformal meshing and process-specific layers. |
| Passive resonators and lasers | Measured decay frequencies and Q for a 1D passive Fabry–Pérot slab, compared with the analytical solution and COMSOL Table 1 | Next verify a finite 3D on-chip cavity with leakage channels, mode volume and Q sensitivity. Then validate carrier and photon rate equations against an independent threshold and light–current curve for a named material/device, followed by spatial gain, injection and thermal coupling. Passive DBR/ring geometry alone cannot supply those measurements. |
| Open-source engines and distribution | A pip wheel, managed Meep/MPB, FDTDX CPU/CUDA12 and BeamZ CPU installations, and a dedicated desktop browser window, verified under WSL2. BeamZ has separate CPU/GPU propagation and CPU mode benchmarks. [Installation and worker support](PACKAGING.md) | General interactive MPB bands/mode tracking, BeamZ parity with specific finite-height GUI studies, cross-platform distribution, upgrades/migrations, unattended offline installation and platform-specific installer testing. Sign and publish only the platforms actually qualified. |
| External comparison coverage | Matched local results for specific Lumerical and Tidy3D mode/coupling quantities, plus the passive COMSOL cavity case | Add inspectable reference S matrices or spectra for the coupler, splitter and ring, with matching excitation, planes, normalization and material models. Revisit Tidy3D quasi-TM area (3.82% difference), weak taper reflection and mesh-sensitive phase. Preserve every discrepancy and evaluate uncertainties separately. |

## Order of the next physics work

1. A four-port directional coupler with per-port reference solves and a
   check against independently computed even/odd supermode splitting. Export
   both the raw amplitudes and the normalized complex matrix from each target.
2. A finite-height passive resonator with a decay trace, known external versus
   intrinsic loss, and the same named pole under mesh, PML and time-window
   changes. Use the [1D cavity benchmark](PASSIVE_CAVITY.md) as a numerical
   control for the fitting method.
3. A dispersive material slab with fitted parameters, held-out data and measured
   complex transmission. Only then use that model in a calibrated ring or DBR.
4. A laser threshold model with a documented active material and injection
   conditions, checked against a public experimental or manufacturer dataset.

[Vendor component coverage](VENDOR_COMPONENT_COVERAGE.md) ·
[Detailed laser gates](COMPONENT_ROADMAP.md) ·
[Public scalar comparisons](COMPONENT_REFERENCE_CHECKS.md).
