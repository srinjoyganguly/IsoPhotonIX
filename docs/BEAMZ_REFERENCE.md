# BeamZ passive reference studies

The workbench measures two quantities with BeamZ 0.6.2. A short electromagnetic
time-domain run measures the phase velocity and wave impedance of a dielectric
on an explicitly chosen CPU or NVIDIA GPU. A separate CPU eigenmode calculation
measures the fundamental TE index of a silicon slab from a published Lumerical
example. The BeamZ reference command runs locally. General editor geometry and
port runs still use the Meep and FDTDX adapters.

Open the published benchmarks in the workbench and expand **BeamZ propagation · CPU and NVIDIA** to inspect both measured phase-velocity series.

## Run and export

On Linux or WSL2 x86-64 with CPython 3.11, the managed CPU profile creates a
separate environment and verifies its field and mode calculations before
registering it. It uses the 27 exact wheels in the CPU lock.
[Wheel lock](../environment-beamz-cpu-linux-64.lock) · [Offline setup, reuse and rollback checks](verification/v010-beamz-setup.json).

```bash
isophotonix setup beamz-cpu
isophotonix benchmark-beamz --device cpu --output beamz-cpu-run
```

For a fresh offline installation, add `--offline --wheelhouse /path/to/wheels`
to the setup command. Every locked wheel must be in that directory.

For the separately tested JAX CUDA 12 route, install BeamZ and the matching JAX
CUDA runtime in an isolated environment and select its interpreter explicitly.
The driver must expose NVIDIA CUDA to WSL. The solver validates device placement
on the computed field arrays and stops on a mismatch.
[JAX CUDA installation](https://docs.jax.dev/en/latest/installation.html) ·
[BeamZ installation](https://www.beamz.tech/docs/getting-started/).

```bash
python3.11 -m venv .beamz-gpu
.beamz-gpu/bin/python -m pip install 'beamz==0.6.2' 'jax[cuda12]==0.10.2'
.beamz-gpu/bin/python -m pip check
BEAMZ_PYTHON="$PWD/.beamz-gpu/bin/python" isophotonix benchmark-beamz \
  --device gpu --output beamz-gpu-run
```

Each run produces `fields.npz`, `report.json`, a standalone
`run_beamz_reference.py`, and a self-contained `beamz-reference.ipynb`.
Run the exported Python in the same solver environment, specifying the same
`--device` and a fresh `--output` directory. A notebook kernel can use its own
BeamZ installation or a solver interpreter supplied in `BEAMZ_PYTHON`. Both
CPU and GPU replays reproduced every array and the full report byte for byte.
The CPU archive contains 38 arrays. The GPU archive contains 18 and covers
the Maxwell propagation calculation. [Replay checks](verification/v010-beamz-replays.json).

## Measured fields

A homogeneous material of index 1.5 carries a forward wave with vacuum
wavelength 1.55 µm. We initialize continuum electric and magnetic fields at
the staggered positions and half-step times of the Yee grid. BeamZ advances
eight actual Maxwell steps. We measure phase and impedance on the central
row over four material wavelengths, farther from every wall than disturbances
can travel during the measurement. The exported NPZ includes signed complete
final Ez and Hy arrays, their initial measurement rows and the staggered
sampling positions.

| Cells per material wavelength | Measured phase speed (m/s) | Difference from c/n |
| ---: | ---: | ---: |
| 12 | 197,874,956 | 0.9940% |
| 24 | 199,375,007 | 0.2435% |
| 48 | 199,743,192 | 0.0593% |

Both targets produced these speeds. On the finest mesh, impedance differed
from the analytic dielectric impedance by 0.00126%. The maximum relative L2
difference in the complete CPU and GPU fields over the three grids was
1.49 × 10⁻⁸. The GPU produced real float32 arrays on the RTX 3080 Ti Laptop
GPU through BeamZ’s JAX backend. The upstream native CUDA kernels have
separate qualification requirements. [CPU run](verification/v010-beamz-cpu.json) ·
[GPU run](verification/v010-beamz-gpu.json).

## Silicon slab mode

The isolated CPU mode solver also solved a 220 nm thick symmetric slab with
core index 3.48, cladding index 1.44 and wavelength 1.55 µm. We identify the
TE mode by its dominant out-of-plane electric field. Mesh spacings 20, 10, 5
and 2.5 nm preserve the physical layer thickness on the discrete grid. At
2.5 nm the measured effective index was **2.85142053**, the exact even-TE slab
root was **2.85135635** and the published Lumerical MODE value was
**2.84825**. The relative differences are **0.00225%** and **0.11131%**,
respectively. A fifth solve enlarged the aperture from 4 to 6 µm. This
reproduces the uniform slab step of the Ansys design. Coupling efficiency of
the full grating requires a separate calculation. [Ansys example](https://optics.ansys.com/hc/en-us/articles/1500007378182-Inverse-design-of-an-efficient-narrow-band-grating-coupler) ·
[Measured indices and aperture check](verification/v010-beamz-cpu.json).

## Provenance

The tested Linux wheel was published 4 November 2026. Its SHA-256 is
`092d9e0302b12cb37f71d14a514c465570d71d01282e47af2d1e5ed0ca049b9e`.
The official annotated v0.6.2 source tag resolves to
`b2db93f11b6ae94160b7980866106320d2f161f5`. BeamZ is licensed Apache-2.0.
The workbench uses its public solver interface and writes its own reference
study. The upstream validation examples informed the staggered-field method
and the mapping between BeamZ’s polarization label and slab TE.
[PyPI release](https://pypi.org/project/beamz/0.6.2/) ·
[Upstream license](https://github.com/beamzorg/beamz/blob/v0.6.2/LICENSE) ·
[Plane-wave validation](https://github.com/beamzorg/beamz/blob/v0.6.2/tests/validation/analytical/test_plane_wave_dispersion.py) ·
[Slab-mode validation](https://github.com/beamzorg/beamz/blob/v0.6.2/tests/validation/analytical/test_slab_waveguide_modes.py).
