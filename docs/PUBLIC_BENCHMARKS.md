# Public waveguide reference benchmarks

These cases reproduce the physical parameters of small waveguide examples
published by solver vendors and compare a specific numerical observable:
the fundamental mode's effective index, `n_eff = beta / k0`. All computation
runs locally using the open-source MPB mode solver supplied with Meep. No
commercial executable, account or cloud solve is used.

This is a separate mode-study benchmark. A mode is invariant along its
propagation axis; it is not the driven final-time volume from the
[finite-height 3D FDTD study](3D_SIMULATION.md). Agreement in effective index
does not establish matching excitation, transmitted power, arbitrary geometries,
or general accuracy across commercial solvers.

## Cases and reference provenance

| Case ID | Physical model at 1.55 µm | Reference observable and origin |
| --- | --- | --- |
| `comsol-slab` | Symmetric slab: thickness 1 µm, core index 1.5, cladding index 1 | Fundamental TE effective index from the analytic slab dispersion equation used by [COMSOL's dielectric slab verification](https://doc.comsol.com/6.3/doc/com.comsol.help.models.woptics.dielectric_slab_waveguide/dielectric_slab_waveguide.html). An analytic-reference check, not a new COMSOL run. |
| `tidy3d-strip` | Rectangular strip: 0.50 × 0.22 µm, core index 3.48, uniform cladding index 1.44 | Published fundamental quasi-TE mode index 2.448339 at `min_steps_per_wvl=100` in the [Tidy3D/Femwell mode benchmark](https://www.flexcompute.com/tidy3d/examples/notebooks/ModeSolverFemwellBenchmark/). This is a scalar value from the public output, not a downloaded field dataset or a newly run Tidy3D solve. |
| `lumerical-slab` | Symmetric slab: thickness 0.22 µm, core index 3.48, cladding index 1.44 | Published 1D MODE effective index 2.84825 from step 1 of [Ansys's grating-coupler example](https://optics.ansys.com/hc/en-us/articles/1500007378182-Inverse-design-of-an-efficient-narrow-band-grating-coupler). Only its uniform slab precursor is replicated; the grating and its coupling efficiency are outside this benchmark. |
| `lumerical-slab-script` | Same slab, with core index 3.47668 from the accompanying script | Separate interpretation of the source's material parameter; compared to the same printed 2.84825, with the ambiguity retained. |

For the Lumerical case, the page states the slab thickness, silicon/background
indices and MODE result. The accompanying
[optimization parameter script](https://optics.ansys.com/hc/en-us/article_attachments/1500010548101)
sets the centre wavelength to 1550 nm, and the example uses TE excitation.
That script also specifies `eps_in = 3.47668 ** 2`, whereas the page states
3.48. Both documented indices are retained in separate cases. The source does
not establish which index generated its printed MODE table; the script variant
is not an index fitted to improve agreement.
The page does not publish the MODE release, mode-solver mesh, transverse domain
or a convergence sweep for this table. Those missing details limit a discrepancy
claim; matching the printed digits is not a justified accuracy requirement.

The independent exact symmetric-slab TE calculation gives
`n_eff = 2.851356345104066` for that stated 220 nm model. Its difference from
the published MODE value is about **0.1091%**. At 110 nm the published value
2.26951 differs from the analytic result 2.2715310649710316 by about 0.0891%.
Using the independently sourced script index 3.47668 gives an analytic
220 nm result of 2.8480709419082277, about 0.00629% below the printed value.
The source's parameter inconsistency is therefore material to the comparison.
It does not by itself identify every contribution to the remaining discrepancy.
Both input interpretations and their results are retained.

## Reproduce and inspect the local results

Use the API environment to launch the benchmark. The driver resolves the
configured Meep interpreter through the same runtime settings as the application;
that solver environment must include MPB. From the repository root:

```bash
.venv/bin/python scripts/benchmark_waveguide_modes.py \
  --case all --output .local/public-benchmarks \
  --resolutions 32 64 128 256 \
  --report docs/verification/public-mode-benchmarks.json
```

`MEEP_PYTHON` can explicitly select a different Meep environment. Use a new
output directory when reproducing a saved run. A single case can be selected
with `--case lumerical-slab`. The benchmark records an additional calculation
with a larger transverse domain to check sensitivity to the finite computational
window. These resolution settings belong to the mode benchmark, not the
interactive 3D FDTD grid limits.

Read the [numerical report](verification/public-mode-benchmarks.json) for actual
versions, mesh and domain settings, computed indices, reference differences and
sensitivity. A rounded public scalar and a locally converged estimate have
different provenance. Any acceptance gate applies only to its named case and
observable; retain failures and unavailable reference details. A color-map
comparison is never substituted for a numerical measurement.

The recorded 256-cells/µm results are:

| Case | Local MPB effective index | Relative discrepancy from declared reference | Last grid change, 128 → 256 |
| --- | ---: | ---: | ---: |
| COMSOL analytic slab | 1.402908867 | 0.0001931% | 0.0005775% |
| Tidy3D strip scalar | 2.448649381 | 0.0126772% | 0.0089566% |
| Lumerical slab, page index | 2.851213703 | 0.1040535% | 0.0175654% |
| Lumerical slab, script index | 2.847928619 | 0.0112835% | 0.0175464% |

The report retains every coarse point and the larger-domain checks. The
Lumerical sequence is non-monotonic; choosing its closer 64-cell result would
conceal that behavior. Small reference differences alongside finite refinement
changes do not certify converged commercial agreement.

## Independent audit of Ansys's asymmetric slab references

Ansys also publishes an
[asymmetric slab example](https://optics.ansys.com/hc/en-us/articles/360042304134-Asymmetric-slab-dielectric-waveguide)
with a 5 µm core of index 1.44, air above and an index-1.33 substrate below,
at 1.55 µm. Its downloadable
[TE table](https://optics.ansys.com/hc/en-us/article_attachments/360056405434)
and [TM table](https://optics.ansys.com/hc/en-us/article_attachments/360056405454)
are explicitly **analytical values used to validate MODE**, not MODE output.
The [associated analytic script](https://optics.ansys.com/hc/en-us/article_attachments/360057294573)
provides the model parameters and dispersion relations.

Use x for propagation, y normal to the layers and z invariant. A convenient
origin puts the core at `−t/2 < y < t/2`; air is above and substrate below.
TE has electric field along z; TM has magnetic field along z. Ansys calls the
fundamental modes TE1/TM1; the phase-order index below calls them `m = 0`.

For `k0 = 2*pi/lambda`, define

```text
h = k0 * sqrt(n_core² - n_eff²)
q = k0 * sqrt(n_eff² - n_upper²)
p = k0 * sqrt(n_eff² - n_lower²)
```

The TE condition is
`h*t = atan(q/h) + atan(p/h) + m*pi`.
For TM, replace `q` and `p` in that condition by
`(n_core/n_upper)²*q` and `(n_core/n_lower)²*p`.
Bisection between `max(n_upper,n_lower)` and `n_core` avoids the poles in a
tangent-based form of the dispersion equation. It independently reproduces all
four listed TE and TM roots:

| Order in Ansys table | Published analytical TE | Published analytical TM |
| --- | ---: | ---: |
| 1, fundamental | 1.433544819278192 | 1.433098106928464 |
| 2 | 1.414165709728225 | 1.412444019217784 |
| 3 | 1.381986467578483 | 1.378468071822638 |
| 4 | 1.338922217753225 | 1.334814055288823 |

The largest absolute difference in this independent audit is
`4.440892098500626e-16`. That checks the interpretation of the published analytic
data; it is not a discretized Maxwell-solver test. The
[reference audit](verification/public-reference-analytic.json) records calculated
values, source URLs, download sizes and SHA256 hashes.

## Scope, credits and next evidence

The implementation and reports are original project work. Vendor pages and
examples retain their owners' copyrights; proprietary projects and downloaded
vendor scripts are not redistributed or executed. Only the model parameters and
scalar reference facts needed for these checks are recorded with attribution.

## Propagation measured with the existing FDTD product adapters

An additional [recorded benchmark](verification/public-slab-propagation.json)
runs the unchanged 2D Meep CPU and FDTDX NVIDIA GPU product adapters using the
COMSOL slab's physical model. It measures propagation from actual saved time
frames, separately from the MPB mode study. The schema-version-1 configurations
can also be imported into the **2D simulation** workspace.

The guide has thickness 1 µm, core/cladding indices 1.5/1.0, wavelength 1.55 µm,
an 8 × 9 µm domain and 1 µm x/y PML. The finite source is at x = −2.5 µm and
the requested duration is 180 fs. Meep injects a uniform current line; FDTDX
injects a directed E/H plane in its three-cell periodic-z extrusion. These
excitations differ from COMSOL's numerical-port source and are not solved
eigenmodes. No commercial runtime is used.

At each saved position, least squares fits `a*cos(omega*t) + b*sin(omega*t) + c`
to recorded frames with `t >= 100 fs`, using the known drive frequency and the
actual sample times. The complex phasor is `a - i*b` for the convention
`Re(F*exp(+i*omega*t))`. Its unwrapped phase is fit over x from −1 to 2 µm;
`beta` is minus the slope and `n_eff = beta/k0`. There is no temporal or spatial
resampling. Saved frames are too sparse to infer an unconstrained frequency
spectrum; prescribing the known frequency is essential.

Two observables are retained. **Centerline** uses the nearest saved y coordinate
to zero. **Template projection** integrates the complex Ez phasor against the
fixed analytic TE0 transverse profile over non-PML y from −3.5 to 3.5 µm,
using trapezoidal integration and a fixed template-squared normalization. That
template depends explicitly on the analytic reference index 1.402911575825797;
the x phase slope is still measured from the solver output. This scalar overlap
does not independently identify a mode or calculate E/H-normalized modal power.

| Target | Cells/µm | Centerline apparent index | Template-projected index | Projected discrepancy from analytic |
| --- | ---: | ---: | ---: | ---: |
| Meep CPU | 30 | 1.399099544 | 1.404614742 | 0.121402% |
| Meep CPU | 40 | 1.398336932 | 1.403867859 | 0.068164% |
| Meep CPU | 45 | 1.398272223 | 1.403798833 | 0.063244% |
| FDTDX NVIDIA GPU | 30 | 1.399973256 | 1.404814731 | 0.135657% |
| FDTDX NVIDIA GPU | 40 | 1.399164812 | 1.403980204 | 0.076172% |
| FDTDX NVIDIA GPU | 45 | 1.399003445 | 1.403755525 | 0.060157% |

The centerline discrepancy does not disappear with refinement and changes when
the spatial fitting window changes. Temporal settling alone therefore does not
establish mode purity. The fixed transverse projection reduces the finest-grid
phase-fit RMSE from approximately 0.0150/0.0121 rad to
8.38 × 10⁻⁶/4.48 × 10⁻⁶ rad for Meep/FDTDX. This is consistent with transverse
radiation or other spatial content biasing the centerline measurement. It does
not prove a complete physical decomposition or bound every remaining error.

The report preserves all six runs, device placement, versions, exact inputs,
sampling masks, fit conditioning, residuals, later-time and shifted-spatial-window
checks, and SHA256 hashes. Derived files retain the complex full-XY harmonic
phasors and projected data; original solver arrays are not modified. No
amplitude is fitted to a commercial result, and no accuracy pass threshold is
applied.

Reproduce using the configured solver runtimes:

```bash
.venv/bin/python scripts/benchmark_slab_propagation.py \
  --output .local/slab-propagation-rerun \
  --report .local/slab-propagation-rerun/report.json \
  --resolutions 30 40 45 --targets meep-cpu fdtdx-gpu
.venv/bin/python -m unittest discover -s tests -p test_slab_propagation.py -v
```

An unavailable requested GPU fails explicitly. Existing completed case outputs
can be reanalyzed with `--reuse` only when their exact configurations match.
The five unit tests use labeled synthetic waves to check the measurement math,
including that the analytic transverse template does not impose its propagation
index on another traveling wave. These fixtures are not physics validation.

This benchmark does not qualify the finite-height 3D workflow, measure
transmission, or establish commercial parity. Independent boundary/domain
refinement, calibrated modal excitation, and power normalization remain needed
for broader engineering claims. See [VALIDATION.md](VALIDATION.md).
