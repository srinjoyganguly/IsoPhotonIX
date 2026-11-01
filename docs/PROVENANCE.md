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
