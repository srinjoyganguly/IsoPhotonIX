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
