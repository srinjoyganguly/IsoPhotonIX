# Build provenance

Implementation began on 1 November 2026 at approximately 14:35 Europe/London for the DevDay Exchange London Community Hackathon, following the 14:33:31 clock and workspace checks. This is the actual start of this application's implementation, not the event's 11:00 start.

The selected WSL workspace contained earlier research and benchmarking projects. They were inspected for available environments and left unchanged. No earlier project source, datasets, reports, or images were copied into this application. The new application lives in its own directory and repository.

A private planning handoff informed the scope. It is deliberately excluded from this repository, as are reference screenshots and raw conversations. The interface uses original HTML/CSS/canvas drawings. Solver adapters are new code informed by the cited upstream documentation; numerical algorithms are supplied by Meep and FDTDX, not implemented by this project.

An existing FDTDX 0.6.2 / JAX 0.10.2 environment was identified and used for initial execution tests. Dependency reuse is distinct from application code written during the event. Environment setup and exact tested versions are documented separately.

The first public release was prepared during the event with real runs on all
three execution routes and an original working-application recording. Before
the 16:00 feature freeze, the application gained playback of its existing
numerical frames and an offline run passport with file checksums. These add
inspection and portability; they do not introduce a new physics model.

The presentation's Blender scene is original work made during the event.
Its geometry is a 3D schematic and its field textures come from a recorded 2D
FDTDX GPU run. The scientific calculation and the presentation rendering are
separate, and that distinction is displayed in the film.

## Post-event development

The interactive 3D design workspace was started after the successful submission,
at approximately 18:59 Europe/London on 1 November 2026, from clean commit
`8fc60bd`. This work is separate from the submitted v0.2 release and its videos.
The first development branch is `feat/interactive-3d-workspace`.

Its geometry editor, validation, templates, and browser checks are new project
code. Three.js 0.187.0 supplies rendering and OrbitControls supplies camera
interaction; their MIT notice is retained. No commercial-tool assets, private
screenshots, or private submission material were reused. At that stage, solver APIs
and numerical scope remained the existing 2D slab model. The product and packaging
plans distinguish implemented features from later qualification work.

The next post-event increment began at approximately 19:15 Europe/London on
1 November 2026 from commit `8f5c0cf`, on branch `feat/finite-3d-solvers`.
It adds a bounded schema-version-2 finite-height waveguide study, new Meep and
FDTDX 3D adapters, explicit CPU/GPU routes, actual volume slices, and numerical
checks. Existing qualified environments were reused. This is new post-event
application code, not part of the hackathon submission. Neither solver source
nor earlier research-project code was copied into the adapters.

After that increment, the user requested comparison with public vendor examples.
The local benchmark harness and evidence panel were added from commit `b36a7f3`.
Public COMSOL, Tidy3D and Ansys example parameters and scalar references are
attributed in `PUBLIC_BENCHMARKS.md`; no commercial program or cloud solver is
used by the benchmarks. MPB 1.12.0 was already present in the qualified Meep
environment. The mode implementations, numerical reports and convergence plots
are new post-event work. Reference conflicts, refinement differences and the
separate qualification scope of mode calculations are retained explicitly.

The guided-port increment began at approximately 20:13 Europe/London on
1 November 2026 from clean commit `3d0ca84`, on branch
`feat/guided-port-measurements`. It adds an independent bounded 2D study,
reference-normalized power monitors, spectra inspection and audited exports.
The application adapters, interface and tests are new post-event code. Meep/MPB
and FDTDX supply the numerical engines; FDTDX's installed Tidy3D dependency
computes the local CPU source mode, with FDTD evolving on the explicitly chosen
CPU or NVIDIA device. No hosted solver, commercial project assets, private
handoff, screenshots or raw chats were used. Existing environments and all
earlier 2D/3D work were preserved.

The finite-height modal-port and reproducible-code increment began at 20:48
Europe/London on 1 November 2026 from clean commit `7ba7fa7`, on branch
`feat/3d-modal-ports-reproduction`. It introduces four-directional reference/device
studies, complex modal projection, phase-aware inspection, and standalone Python
and Jupyter bundles. New source and tests are first-party post-event work; the
existing qualified solver environments were reused. Exported bundles identify
their actual source bytes with SHA256 and do not include runtime paths, raw
worker logs, secrets or private planning files.

Real symmetric-device checks exposed and corrected a signed-mode projection
issue in the Meep adapter and staggered interface placement in the FDTDX
adapter. Both corrections live in exported application source; installed solver
packages were not patched. Failed prototype evidence is distinguished from the
corrected qualification runs. Original screenshots show this application and
its actual numerical results.

The component-library increment began at approximately 21:56 Europe/London
on 1 November 2026 from clean commit `9900bf8`, on branch
`feat/photonic-component-library`. Twenty parametric geometry families,
polygon contours/holes, material records, layout composition and standalone
design generators are new first-party code. Official Tidy3D and Ansys examples
informed the component coverage and staged roadmap; their code, project files,
material databases and assets were not copied. Existing application work and
qualified solver environments were preserved.

This increment qualifies design editing and reproduction. It does not turn
new library geometries, sampled materials or passive cavities into supported
electromagnetic or laser models. Original browser screenshots and recorded
Python/notebook executions document the actual design features; separate
real Meep/FDTDX checks exercise the previously supported finite-height guide.

On 5 November 2026 the taper and packaging increments started from clean commit
`175a198` on branch `feat/passive-taper-studies`. New first-party code checks the
complete editor taper, resolves its access guides and simulation cell, and
runs the native Meep and FDTDX adapters. A straight-guide control exposed an
artificial interface between separately represented pieces in Meep. Extruding
one continuous contour eliminated the difference against an independent native
block in both its material grid and vector fields. The [native records](TAPER_FIELDS.md)
retain the original discrepancy, correction and independent CPU/GPU checks.

The first application wheel packages the existing service and compiled interface
with an explicit file allowlist. The managed Meep/MPB profile uses the existing
official conda-forge artifact lock. Native numerical checks run before the
installed application switches to the newly provisioned interpreter.

The v0.9 work began on 5 November 2026 from clean commit `eb7b402`. It adds
seven first-party geometry generators, an XY drawing workspace, finite passive
layout simulation, and independently normalized two-ended taper measurements.
The [commercial example review](VENDOR_COMPONENT_COVERAGE.md) records the
public sources and gaps. The [new comparison cases](COMPONENT_REFERENCE_CHECKS.md)
preserve published targets, local calculations, mesh and domain changes, and
remaining differences. No commercial project files, private screenshots or
vendor material databases are distributed. Screenshots show the running app.

## v0.10 editing, delivery and numerical qualification

This increment began from local commit `25353f2` after v0.9. The editor reuses
its validated geometry documents and material ordering. The new controls
remove an entire instance atomically, recover local history, and place
components using the coordinates of their own named ports. The application
screenshot shows a workbench-generated strip and directional coupler.

New numerical evidence uses local Meep, FDTDX/JAX, MPB and BeamZ. The COMSOL
cavity and Ansys slab references use published observables. The simulations,
reports, code and notebook replays were generated independently in this
project. BeamZ 0.6.2 is obtained from upstream, with its release and licensing
details recorded in [BeamZ reference studies](BEAMZ_REFERENCE.md). The
[FDTDX setup](FDTDX_MANAGED_SETUP.md) and [desktop window](DESKTOP_WINDOW.md)
record the host-specific qualifications. No solver credentials or commercial
application installations are required for these paths.
