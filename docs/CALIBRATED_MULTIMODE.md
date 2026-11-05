# Multimode offset joints

Each side of the joint carries two guided transverse modes. Four independent launches each have their own straight incident reference. The complex scattering matrix records the reflected and transmitted amplitudes at physical planes 1.25 µm from the discontinuity. The electric field is real-positive a quarter-width above each guide centre. Reflecting a guide about its centre changes its odd-mode field sign.

## Planar silicon slab

The silicon strip is 0.375 µm wide and shifts laterally by 0.125 µm. The core and cladding indices are 3.48 and 1.44. An independent frequency-domain calculation matches the electric field and its normal derivative at the step. A transverse absorbing layer carries outward radiation. Expanding its box from 8 to 10 µm changes the reference powers by at most 0.00000040. An earlier hard-wall box trapped outgoing radiation and produced box-dependent results.

| Meep cells per µm | Largest power-fraction error vs independent solve | Largest complex-amplitude error |
| ---: | ---: | ---: |
| 32 | 0.004315 | 0.17045 |
| 48 | 0.001828 | 0.07289 |
| 64 | 0.001243 | 0.03924 |

At 1.55 µm the finest Meep run puts 27.62% of the left fundamental input into the transmitted first higher-order mode. The independent calculation gives 27.69%. The public working preset uses 48 cells/µm. The 32-cell result exceeds the 0.15 complex-amplitude acceptance limit, even though its power error is below half a percentage point. The data files include both the incoming and outgoing native projections from every launch and the measured matrices on each mesh. [Measurement and raw-array hashes](verification/v011-multimode-planar.json) · [Modal arrays](verification/v011-multimode-planar.npz) · [Native simulation](../scripts/calibrate_multimode.py) · [Audit and mesh comparison](../scripts/summarize_multimode.py).

## Signed mode convention

Meep returns signed flux overlaps when each direction is solved with its own wavevector on a +x-oriented monitor. The workbench converts these to electric amplitudes using the measured group-velocity sign as well as the phase of the electric mode. Both normal-incidence launches were checked against a separate direct field measurement and analytic Fresnel reflection. At 96 cells/µm the two maximum absolute complex-amplitude differences were 0.000000070 against the native field probe and 0.00154 against theory. The coupler and splitter phase audits use their own saved native monitor fields. Previous signed-overlap receipts and self-contained exports retain the original convention. [Fresnel samples](verification/v011-meep-fresnel-phase.json) · [Native Meep implementation](https://github.com/NanoComp/meep/blob/v1.35.0/src/mpb.cpp#L963-L982) · [Device field audit](verification/v011-meep-multiport-phase-audit.json).

## Commercial comparisons

The installed local Tidy3D mode solver was run on the same straight cross sections. Its solutions compare propagation constants and mode parity. For the offset step, the independent calculation above supplies the full complex-matrix comparison. The published Lumerical and Tidy3D eigenmode-expansion examples use different devices and provide a method comparison. No full commercial matrix for this exact joint was available in the examined examples. [Matched port measurements](CALIBRATED_VENDOR_CHECKS.md) · [Lumerical EME](https://optics.ansys.com/hc/en-us/articles/360034396614-MODE-EigenMode-Expansion-EME-solver-introduction) · [Tidy3D EME](https://www.flexcompute.com/tidy3d/examples/notebooks/EMESolver/).

## Finite-height silicon joint

The strip cross section is 0.875 × 0.25 µm in oxide, with the same 0.125 µm offset. A full-vector Meep run at 16 cells/µm launched both quasi-TE modes from both sides. At 1.55 µm the left fundamental input delivered 11.14% into the first higher-order transmitted mode. The maximum reciprocity residual across the measured power fractions was 0.000786. The collected guided output never exceeded 0.9854 of the incident power. Mesh and monitor controls are tracked independently. [Measured 3D matrices and native modal arrays](verification/v011-multimode-finite.json).
