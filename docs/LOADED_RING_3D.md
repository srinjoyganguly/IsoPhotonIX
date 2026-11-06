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

The preset exports Python and Jupyter through the local workbench. The standalone worker accepts both vendor names, verifies the actual CUDA field device, and saves a measured result with its qualification mask. Completed reference and device checkpoints are checked against source, solver identity, settings and hardware. Opening a recorded result or inspecting a field snapshot only reads the saved data.

## Measured results

The 16-cell Lumerical run at 1600 µm/c reached 233 of 241 qualified bins. Its two central measured peaks occurred at 1530.5 and 1556.5 nm, giving 26.0 nm spacing versus the published 25.4549 nm. Eight wavelengths near the shortest-band resonance require a longer acquisition. That first run withheld all linewidth-based Q estimates. [Native 16-cell partial spectrum, matched vendor samples and decay fits](verification/v012-ring3d-lumerical-r16-t1600.json).
