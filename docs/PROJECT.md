# One experiment you can inspect and keep

Photonics students, researchers, and engineers need to understand how geometry,
numerical choices, and execution hardware produce a field. This event build
connects those decisions in one small local waveguide experiment.

## What makes this useful

- **A short path to real physics.** The bundled example runs through Meep or
  FDTDX. Physical validation explains when an edit exceeds the supported model.
- **A device choice you can audit.** The user chooses a probed CPU or NVIDIA GPU
  target. Each result records the device actually used; GPU failure never starts
  a CPU retry.
- **A result that belongs to the researcher.** Configuration, numerical arrays,
  versions, source assumptions, and logs leave the application together. The
  offline HTML run passport includes checksums and a reproduction command that
  uses the original explicit solver/device selection.
- **An honest teaching surface.** Fields are actual stored samples. The interface
  replays recorded frames using their simulated times and one fixed color scale.
  It describes source conventions and numerical limitations instead of presenting
  unverified transmission or performance claims.

## Position relative to commercial tools

The differentiator is this focused combination of an approachable local
workflow, open numerical engines, explicit hardware control, and inspectable
research output without a paid commercial solver license. It is not a claim
that local execution, exports, or GPU support are unique inventions.

For concrete context, Ansys documents GPU execution and its licensing conditions
for Lumerical FDTD in its [official GPU guide](https://optics.ansys.com/hc/en-us/articles/17518942465811-Getting-started-with-running-FDTD-on-GPU).
Flexcompute documents Tidy3D's `web.run` as submitting simulations to its cloud
server in its [official workflow documentation](https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/faq/web.run.html).
This project instead executes the supported model on the user's own machine
through Meep or FDTDX/JAX. These are specific workflow differences, not a
feature, price, speed, or accuracy ranking. Sources checked 1 November 2026.

Commercial platforms cover much more: production workflows, advanced geometry,
materials, analysis, and support. This milestone is a lossless 2D slab teaching
model with no convergence certification, normalized transmission, or broad
commercial parity claim. NVIDIA CUDA components remain subject to NVIDIA's
licenses; “no paid solver license” does not mean every dependency has the same
license.

## Evidence against the judging criteria

| Criterion | Concrete evidence |
| --- | --- |
| Idea and usefulness | One accessible laptop experiment, with physical validation, explicit hardware choice, and a portable numerical record |
| Meaningful use of Codex | [The fallback debugging example](../CODEX_USAGE.md): identified FDTDX's automatic CPU fallback, enforced platform selection, and checked actual array placement |
| Execution | [Verification records](verification/), real Meep CPU, FDTDX CPU, and FDTDX NVIDIA GPU runs, plus complete browser journeys and [exact-frame/checksum checks](verification/run-passport.json) |
| Demo and clarity | [Public working demonstration](https://srinjoyganguly.github.io/photonics-workbench/), original application captures, visible scientific limitations, and source/setup links |

The application was built during the event. [Provenance](PROVENANCE.md) and
[credits](../CREDITS.md) distinguish new work from numerical libraries and other
reused dependencies. The public repository contains neither private planning
material nor raw chats.
