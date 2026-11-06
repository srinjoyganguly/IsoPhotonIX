# Finite-height ring with two bus guides

The 3D loaded-ring preset launches a guided pulse along the lower straight bus. The silicon ring and both buses have finite height. Four physical monitor apertures collect the returned input, through, drop and add-side amplitudes. A second 3D simulation removes the ring while retaining both buses, the source, substrate and monitor apertures. Its measured incident mode normalizes the device spectrum. The full vector Maxwell time evolution runs on the explicitly selected FDTDX NVIDIA target.

| Geometry | Centerline radius | Width × height | Edge gap | Background above / below |
| --- | ---: | ---: | ---: | --- |
| Lumerical add/drop | 3.10 µm | 0.40 × 0.18 µm | 0.10 µm | Air / oxide |
| Tidy3D add/drop | 5.00 µm | 0.43 × 0.22 µm | 0.10 µm | Oxide / oxide |

The published examples excite the upper bus; the local layout mirrors it about the center in y. The external spectra use measured power, which is invariant under that mirror in these symmetric bus layouts. Local analysis uses fixed silicon and oxide indices of 3.48 and 1.44 through the band. The vendor material libraries include dispersion. Local ports sit approximately 0.76 µm outside the ring radius on x, with dimensions snapped to the selected mesh. The reference source corrects incident amplitude and phase at the local plane. Absolute power comparisons do not fit a wavelength shift. [Lumerical 3D add/drop example and port coordinates](https://optics.ansys.com/hc/en-us/articles/360042800213-Ring-resonator-getting-started-Final-parameter-extraction) · [Tidy3D add/drop example](https://www.flexcompute.com/tidy3d/examples/notebooks/RingResonator/) · [Pinned Tidy3D notebook](https://github.com/flexcompute/tidy3d-notebooks/blob/18fd47f8bd87ad89582b507c20bf4997f79d55c0/RingResonator.ipynb).

## Numerical measurements

The GPU advances all three electric and magnetic components on the spatial Yee grid. Six PML faces truncate the open domain. Each electric component uses its own physical position, lateral area sampling of the ring and buses, exact vertical material fractions, and series or parallel averaging for normal or tangential vertical interfaces. The 100 nm gap spans 1.6 cells at 16 cells/µm and 2 cells at 20 cells/µm. Curve, gap and propagation errors need a mesh comparison.

FDTDX’s fundamental quasi-TE port profiles are solved independently at five wavelengths between 1.49 and 1.61 µm. Fields are spatially collocated and magnetic fields are synchronized to electric-field time before projection. An offline Fourier transform and interpolation of these consistently phased mode profiles produce the complex output amplitudes. Reference and device input profiles are compared. Saved raw arrays also include source profiles, mode profiles, the time-domain directional projections, three ring field probes, and instantaneous XY electric-field snapshots on the device midplane. [Native GPU worker](../scripts/calibrate_loaded_ring3d.py) · [FDTDX mode-source and detector guide](https://fdtdx.readthedocs.io/en/latest/notebooks/components/02_mode_source_detector.html).

For each wavelength, the reference through power must be within 0.04 of unity, the device and reference incident amplitudes must agree, and total measured outgoing power must stay below 1.05. Discarding the last 20% of the device signal must move each output by less than 0.05 in power fraction. The receipt preserves rejected bins. The vendor comparison separately evaluates through and drop power at the same wavelengths without spectral registration. Lumerical supplies a numerical text file. Tidy3D’s through/drop curves are read from its pinned figure, with approximately 0.22 nm per pixel and larger vertical readout uncertainty on steep lines. [Vendor comparison code](../scripts/compare_loaded_ring3d.py).

The device time traces come from a loaded ring. A supplementary analysis fits the same resonance in independent time windows at several ring probes, and in the measured through and drop spectra. The amplitude decay time and fitted linewidth constrain the loaded Q. An equal-bus, single-resonance model then estimates external coupling to each bus and apparent internal loss. This step requires resolved linewidth, stable field-decay fits, agreement between temporal and spectral Q, weak unmonitored-port power and small through/drop fit residual. When apparent internal loss is comparable to the fit uncertainty, the analysis gives a lower bound on internal Q. [Loss analysis](../scripts/measure_loaded_ring_losses.py).

## Reproduce a measurement

From a checkout, use the registered FDTDX CUDA interpreter in a fresh process.

```bash
"$FDTDX_PYTHON" scripts/calibrate_loaded_ring3d.py --vendor lumerical --device gpu \
  --resolution 16 --duration-um-c 3200 --checkpoint ring-stages --output ring-result
```

To recompute a public comparison, download the [Lumerical S-parameter file](https://optics.ansys.com/hc/en-us/article_attachments/50334805245587) or the [pinned Tidy3D notebook](https://raw.githubusercontent.com/flexcompute/tidy3d-notebooks/18fd47f8bd87ad89582b507c20bf4997f79d55c0/RingResonator.ipynb). Pass that file as `--source` to `scripts/compare_loaded_ring3d.py` with `--run ring-result --output comparison.json`. The comparison checks its SHA-256 and reports the mean and RMS difference in power fraction. The source ring run also retains the raw E/H projections needed to evaluate the 0.1 nm loss-fit samples independently of the 0.5 nm overview plot.

The preset exports Python and Jupyter through the local workbench. The standalone worker accepts both vendor names, verifies the actual CUDA field device, and saves a measured result with its qualification mask. A new ring checkpoint binds the captured Python sources, installed numerical package contents, backend flags, settings, CUDA driver build and physical card. Moving a checkpoint to another same-model card triggers a fresh solve. If the physical GPU or numerical installation cannot be identified, the selected CUDA simulation can still run with an isolated temporary cache. The public run record omits the stable card fingerprint. Older exported programs retain their captured source and original cache behavior. In the recorded 16-cell default, a fresh run took 342 seconds and a verified cache reuse took 10.7 seconds; both reproduced the original numerical archive byte-for-byte. [Checkpoint proof](verification/v012-ring3d-stage-cache.json). Opening a recorded result or inspecting a field snapshot only reads the saved data. [JAX device identity](https://docs.jax.dev/en/latest/_autosummary/jax.Device.html) · [CUDA driver device management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html).

## Measured results

The spectra are sampled at 241 wavelengths from 1.49 to 1.61 µm. The commercial comparison uses common qualified samples between 1.50 and 1.60 µm. Errors below are absolute differences in power fraction across that band. The earliest records retain wavelengths excluded by the time and port checks.

| Geometry | Cells/µm | Time (µm/c) | Qualified bins | Local / published drop FSR (nm) | Through / drop mean absolute error | Native data |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Lumerical | 16 | 1600 | 233/241 | 26.00 / 25.45 | 0.152 / 0.120 | [Result](verification/v012-ring3d-lumerical-r16-t1600.json) |
| Lumerical | 16 | 3200 | 241/241 | 26.00 / 25.45 | 0.155 / 0.124 | [Result](verification/v012-ring3d-lumerical-r16-t3200.json) |
| Lumerical | 20 | 3200 | 241/241 | 27.00 / 25.45 | 0.173 / 0.136 | [Result](verification/v012-ring3d-lumerical-r20-t3200.json) |
| Lumerical | 24 | 3200 | 241/241 | 27.00 / 25.45 | 0.198 / 0.160 | [Result](verification/v012-ring3d-lumerical-r24-t3200.json) |
| Tidy3D | 16 | 2700 | 236/241 | 17.50 / 17.74 | 0.152 / 0.126 | [Result](verification/v012-ring3d-tidy3d-r16-t2700.json) |
| Tidy3D | 16 | 4800 | 241/241 | 17.50 / 17.74 | 0.156 / 0.130 | [Result](verification/v012-ring3d-tidy3d-r16-t4800.json) |
| Tidy3D | 20 | 4800 | 241/241 | 18.00 / 17.74 | 0.199 / 0.175 | [Result](verification/v012-ring3d-tidy3d-r20-t4800.json) |

**Ring spacing and absolute wavelength.** The two central Lumerical-geometry peaks move from 1530.5/1556.5 nm at 16 cells/µm to 1536.0/1563.0 nm at 20 cells/µm. At the published 1550 nm drop peak, the 20-cell local model gives 0.0113 in power versus 0.6744 in the filed data. The largest 16→20 fixed-wavelength power differences are 0.970 in through and 0.861 in drop. The greatest through phase difference at common physical planes is 3.08 rad where both powers exceed 0.01. Straight-reference group delay gives separate spacings of 26.257 and 26.889 nm. [Lumerical mesh and phase comparison](verification/v012-ring3d-lumerical-mesh-comparison.json).

For the Tidy3D geometry, the 20-cell central pair occurs at 1540.5 and 1558.5 nm. The largest 16→20 fixed-wavelength through/drop power changes are 0.983/0.950, whereas the largest early/full temporal change on the 20-cell grid is 0.00496. Straight-reference group delays predict 17.559 and 17.939 nm. A close FSR can coexist with large errors in power at a particular wavelength. [Tidy3D mesh and phase comparison](verification/v012-ring3d-tidy3d-mesh-comparison.json) · [Lumerical duration control](verification/v012-ring3d-lumerical-duration-comparison.json) · [Tidy3D duration control](verification/v012-ring3d-tidy3d-duration-comparison.json).

**Loaded loss.** The longer 16-cell Lumerical study qualifies two spectral linewidths. The 1556.5 nm peak has loaded Q 723 from the spectrum and 753 from independent ring-field decay. The equal-bus model gives external Q 1542 per bus and apparent internal Q about 11,678. The nearby filed Lumerical peak is at 1550.09 nm with half-height Q 1091. Its 1577.24 nm line gives Q 752 versus 548 at the nearest local peak; the attachment's shorter-wavelength split resonance fails the single-line width rule. Nearest peaks within half a published FSR are paired by wavelength; this does not establish identical angular order. [Lumerical spectrum, decay fits and vendor lines](verification/v012-ring3d-lumerical-r16-t3200.json).

In the 16-cell Tidy3D-geometry spectrum five lines qualify. At 1550.5 nm, Q is 1259 from the spectrum and 1318 from field decay, and external Q is 2560 per bus. The empirical internal-Q lower estimate is approximately 48,100; loss is too weak for a two-sided estimate. The nearest digitized Tidy3D peak is at 1546.78 nm with half-height Q about 1441. Six 20-cell lines qualify, including 1558.5 nm with spectral Q 953 and field-decay Q 983. Internal loss includes radiation and numerical leakage. [16-cell decay fits](verification/v012-ring3d-tidy3d-r16-t4800.json) · [20-cell decay fits](verification/v012-ring3d-tidy3d-r20-t4800.json).

**Open boundaries and ports.** Both independent 16-cell Lumerical controls retain all 241 wavelengths, 26.0 nm FSR and the same two qualified linewidths. The stage archives were checked against their individual acquisition receipts.

| Change at 3200 µm/c | Largest change in any output power fraction | FSR change | Evidence |
| --- | ---: | ---: | --- |
| PML thickness, 10→14 cells | 0.00003894 | 0 nm | [Matched spectra](verification/v012-ring3d-lumerical-pml-comparison.json) · [Native run](verification/v012-ring3d-lumerical-r16-t3200-pml14c.json) |
| Port aperture, 1.75→2.25 µm | 0.00044924 | 0 nm | [Matched spectra](verification/v012-ring3d-lumerical-aperture-comparison.json) · [Native run](verification/v012-ring3d-lumerical-r16-t3200-aperture36c.json) |

The 20→24 cells/µm control changes through/drop power fractions at a fixed wavelength by up to 0.970/0.838 on 241 jointly qualified bins. The report preserves the higher-resolution validity mask and source/field provenance. [20→24 mesh control](verification/v012-ring3d-lumerical-20to24-comparison.json) · [GPU memory preflight](verification/v012-ring3d-lumerical-r24-memory.json).

A fresh Python run and an independently executed Jupyter notebook repeated the qualified 16-cell Lumerical study in separate directories. Both compared their native arrays, complex coefficients, measured fields and qualified loss estimates against the saved result. [Export replay evidence](verification/v012-ring3d-python-notebook-replay.json).
