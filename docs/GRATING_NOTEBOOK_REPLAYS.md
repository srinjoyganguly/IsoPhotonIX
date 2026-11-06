# Replaying the two published 14-cell grating notebooks

Both programs are complete local FDTDX simulations of finite-width gratings on NVIDIA. Each notebook computes the incident field in a separate reference simulation, evolves the actual device, and projects the output onto the guided mode. The release embeds five commercial reference values and the original workbench arrays so a new run can be compared directly.

## Reproduce

Download an example below, unpack it into an empty directory and execute every cell of `calibration.ipynb`. The notebook kernel needs Python 3.11 and NumPy. Use a registered FDTDX CUDA12 worker or set `FDTDX_PYTHON` to its interpreter. The supported worker versions for these comparisons include FDTDX 0.6.2, JAX/JAXLIB 0.10.2 and Tidy3D 2.11.2. The validated hardware was an RTX 3080 Ti Laptop GPU with 16 GB. [Worker installation](PACKAGING.md) · [Lumerical curved-grating notebook and fields](https://github.com/srinjoyganguly/photonics-workbench/releases/download/v0.12.0a1/grating-lumerical-3d-gpu-r14-v0.12.0a1.zip) · [Tidy3D uniform-grating notebook and fields](https://github.com/srinjoyganguly/photonics-workbench/releases/download/v0.12.0a1/grating-tidy3d-3d-gpu-r14-v0.12.0a1.zip).

To repeat the complete audit from the repository, use a notebook environment with NumPy, nbformat, nbclient and ipykernel. The following command downloads the two verified example archives, runs both unchanged notebooks in separate directories and writes a report. Use a new work directory. The native replay workers use the explicitly selected GPU. Supplying `--archives DIRECTORY` selects saved copies with the same release filenames.

```bash
python -m scripts.verify_vendor_grating_notebooks \
  --work grating-notebook-runs --report grating-notebook-verification.json
```

The audit compares all native arrays, reconstructs the modal output and incident Poynting flux from recorded E and H, checks both source fields and their staggered time delays, and reads the displayed x–z field from its saved array. The fresh stage signatures are distinct from those in each original export. A small companion NPZ, when present, contains the complete fresh arrays whose bytes changed. The five commercial values compare efficiency at identical wavelengths, without shifting peaks or fitting an amplitude factor.

## Measured result

The unchanged released notebooks both completed separate incident and device runs on the RTX 3080 Ti Laptop GPU. All five wavelengths qualified in both cases. The Lumerical run measured the polished-fiber-to-guide transfer. The Tidy3D run measured Gaussian-to-guide transfer through a finite taper. The table compares each reconstruction with its own published geometry and collection channel. [Lumerical reference](https://optics.ansys.com/hc/en-us/articles/360042305334-Grating-coupler) · [Tidy3D reference](https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/).

| Wavelength | Lumerical notebook | Lumerical published | Tidy3D notebook | Tidy3D published |
| ---: | ---: | ---: | ---: | ---: |
| 1500 nm | 33.678% | 15.061% | 4.971% | 7.110% |
| 1525 nm | 34.601% | 29.455% | 20.615% | 25.905% |
| 1550 nm | 24.908% | 40.027% | 35.666% | 38.019% |
| 1575 nm | 12.989% | 37.022% | 30.336% | 27.448% |
| 1600 nm | 4.592% | 22.373% | 14.199% | 12.128% |

Across the five wavelengths the mean absolute difference is **16.139 percentage points** for Lumerical and **2.948 points** for Tidy3D. The new local results exactly reproduce the original coupling values. These reruns validate repeatability. The broadband difference from the commercial examples persists. [Physical model and mesh checks](FINITE_GRATING_3D.md).

- **Lumerical.** 43 of 45 numerical arrays matched byte-for-byte. The rerun differed only in ref_energy (maximum absolute difference 0.01562), dev_energy (maximum absolute difference 0.03125), within the saved tolerances. The selected native field and the five coupling values have zero maximum difference. [Executed notebook](verification/v013-grating-notebook-replays-lumerical.ipynb).
- **Tidy3D.** 49 of 50 numerical arrays matched byte-for-byte. The rerun differed only in dev_energy (maximum absolute difference 3.815e-06), within the saved tolerances. The selected native field and the five coupling values have zero maximum difference. [Executed notebook](verification/v013-grating-notebook-replays-tidy3d.ipynb).

The largest energy-trace difference is 8.81 × 10⁻⁸ of its original trace peak. All recorded electric and magnetic field arrays matched byte-for-byte.

The public notebook copies retain the execution order, cell sources, measured outputs and kernel messages. Their printed absolute result directory is shown as `notebook-result`. The receipt records separate SHA-256 values for each original release notebook, original private execution output and its public copy. The Tidy3D stderr includes CUDA allocator retries. Both stages continued on the selected GPU and passed the independent settling and field checks. [Notebook verification receipt](verification/v013-grating-notebook-replays.json) · [Changed numerical arrays](verification/v013-grating-notebook-replays-changed-arrays.npz).
