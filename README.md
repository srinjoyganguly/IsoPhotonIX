# Photonics Workbench

A local photonics workbench for students, researchers, and engineers: open a
dielectric waveguide, edit validated physical parameters, choose a real solver
and execution device, and inspect the electromagnetic fields it computes.

Built for the DevDay Exchange London Community Hackathon on 1 November 2026.
The application code was started during the event. Existing numerical libraries
provide the solvers; [build provenance](docs/PROVENANCE.md) and
[credits](CREDITS.md) distinguish that reuse from new work.

**[Watch the 84.4-second working demo](https://srinjoyganguly.github.io/photonics-workbench/)** — recorded from a real local NVIDIA run, with the solver wait shown in real time. The hosted page plays the video; simulations run locally after installation.

![Actual FDTDX GPU field and execution metadata in the workbench](docs/application.png)

[Public source repository](https://github.com/srinjoyganguly/photonics-workbench)

## The complete journey

1. Open the bundled slab-waveguide example.
2. Edit wavelength, core width, and refractive indices. Numerical settings expose
   domain size, grid resolution, simulation duration, and PML thickness.
3. Read the capability-based solver recommendation and validation messages.
4. Explicitly select **Meep CPU**, **FDTDX CPU**, or a verified **FDTDX NVIDIA GPU**.
   Unavailable targets remain unavailable; no GPU-to-CPU fallback exists.
5. Run the simulation in an isolated local worker. Inspect status and logs, or
   cancel it.
6. View the actual final Ez field and discretized permittivity. Hover to inspect
   values, inspect solver/device metadata, and export the results ZIP.

Configuration JSON can be saved and imported. A completed run retains its own
configuration, so later edits do not relabel an earlier result. The result ZIP
contains the configuration, numerical NumPy arrays, run metadata, and sanitized
worker log. Computation stays on the local machine; no OpenAI API key, runtime
chatbot, cloud solver, or paid commercial solver license is required.

## Run locally

Use Linux or WSL2, Python 3.11, Conda for Meep, and Node 22.12+ for building the
interface. Follow [the full setup instructions](docs/SETUP.md) to install the
separate API and solver environments, then:

```bash
cd frontend
corepack pnpm install --frozen-lockfile --fetch-timeout=600000
corepack pnpm build
cd ..
bash scripts/start.sh
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The start script binds only
to the loopback interface. Node is needed to build the interface, not to serve
the completed local application.

Reproduce a saved configuration without the browser:

```bash
.venv/bin/python scripts/run_simulation.py examples/slab-waveguide.json --output runs/reproduction-001
```

The output directory must be new. The solver and device recorded in the JSON
are used explicitly. Set `MEEP_PYTHON` or `FDTDX_PYTHON` when the corresponding
solver lives outside the default project-local environment.

## Verification status

| Route or check | Evidence and status |
| --- | --- |
| FDTDX 0.6.2 / JAX 0.10.2 on CPU | Real default and edited-width waveguides passed; finite nonzero fields and correct material arrays |
| FDTDX 0.6.2 / JAX 0.10.2 on NVIDIA | Real default waveguide passed on RTX 3080 Ti Laptop GPU, 16 GB; actual CUDA placement checked |
| GPU unavailable/fallback guard | GPU-blocked execution failed explicitly; no CPU retry |
| Meep 1.35.0 on CPU | Real shared-worker default and edited-width waveguides passed; finite nonzero propagated fields, material/coordinate checks, and explicit GPU rejection |
| Frontend build | TypeScript check and Vite production build passed using Node 24.22.1 / pnpm 11.25.0 |
| HTTP journey and export | Meep CPU, FDTDX CPU, and FDTDX GPU passed; exported arrays exactly matched the field API; config and target metadata matched |
| Browser journey on all three routes | Real edited-width runs, invalid-input feedback, field/material views, ZIP export, explicit selection, JSON import, and immutable run snapshots checked |
| Failure and cancellation | Unit/API checks passed; a cancelled worker cannot return a completed field or result export |
| Responsive interface | Browser checks passed at 1440, 1280, and 390 px with no overflow or browser errors |
| Meep fresh-environment setup | Documented YAML created a fresh prefix from cached official packages; real worker output matched the first environment exactly |
| API clean install | Fresh isolated virtual environments installed from SHA256-verified official PyPI wheels; dependency check passed; 15 application tests passed with the separately tested Meep case skipped |
| FDTDX clean install | Not yet established; numerical verification used an existing environment with recorded versions |

[FDTDX verification](docs/FDTDX_VERIFICATION.md) records the exact measured
times, devices, and numerical checks. The first default Meep run took about
1.8 seconds inside the adapter; the first default FDTDX runs took about
16.4 seconds on CPU and 17.9 seconds on GPU including imports, setup, main-loop
compilation, execution, and export. These are single-run observations on one
machine, not a GPU speedup claim. Small jobs can spend more time compiling than
solving. The [Meep adapter notes](docs/meep.md) and
[recorded metadata](docs/verification/meep-cpu.json) describe its model and tests.

Run the focused application tests and dedicated numerical checks as described
in [setup and testing](docs/SETUP.md). A skipped solver test is not evidence that
the route works.

[HTTP, browser, cancellation, and demo evidence](docs/verification/) records the
checks. [Optional browser-test instructions](docs/browser-tests.md) describe the
separate Playwright setup used for UI verification.

## Scientific scope and limitations

The example is a 2D, lossless, nondispersive dielectric slab with Ez
polarization, a finite-width source, and absorbing x/y boundaries. FDTDX uses a
three-cell periodic z extrusion; Meep uses native 2D. Both sources ramp over four
optical periods. Meep injects an electric-current line while FDTDX injects a
directed E/H plane, so absolute amplitudes are not cross-solver calibrated.

The viewer shows a finite-time field snapshot in arbitrary source units,
including the PML region. It does **not** report normalized transmission,
S-parameters, mode purity, convergence, or experimentally validated device
performance. Material interfaces and sampled coordinate grids differ between
solvers. A minimum grid rule prevents obviously coarse inputs but does not
replace a convergence study.

The application permits one job at a time, caps the x-y grid at 180,000 cells,
and stops jobs exceeding five minutes. NVIDIA support depends on a compatible
Windows/Linux host driver and a working JAX CUDA environment; GPU visibility
alone does not establish working execution. Native Windows solver execution,
other GPUs, remote compute, multi-GPU jobs, and MPI are not supported by this
milestone.

MPB, BeamZ, generalized CAD/GDS editing, optimization, multiphysics, and broad
commercial-tool feature parity are future work.

## Contribution and licensing

[CODEX_USAGE.md](CODEX_USAGE.md) records a concrete implementation/debugging
contribution and its verification evidence. The human supplied the problem,
users, solver choices, local-compute requirement, and delivery constraints.
Codex implemented and tested the application under those directions.

New application source is [MIT licensed](LICENSE). Meep is GPL-2.0-or-later;
FDTDX is MIT licensed; JAX is Apache-2.0 licensed; optional NVIDIA runtime
components have their own licenses. See [credits and dependency notices](CREDITS.md).
The repository does not redistribute solver environments, private handoffs,
reference screenshots, credentials, or raw chats.
