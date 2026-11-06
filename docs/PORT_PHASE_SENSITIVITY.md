# Physical-plane scattering sensitivity

FDTDX on NVIDIA independently excited one physical input of the finite-height coupler and Y splitter. The comparison uses the same three wavelengths, ports, oxide-clad silicon layout, electric-field-aligned material sampling and signed field convention at 16, 24 and 28 cells/µm. Each column has a separate measured straight-guide reference. The checker reconstructs the normalized coefficients from the actual monitor amplitudes and verifies them against the saved native matrix. Both 28-cell reference and device stages passed their temporal settling checks.

For forward ports, the gate is a maximum physical phase change of 0.10 rad over the three wavelengths. For return ports, the maximum allowed changes are 0.001 in power fraction and 0.01 in complex amplitude. Phase is reported only where |S| ≥ 0.02 on both meshes. The straight-relative phase divides each forward transfer by the measured straight transfer on that mesh. It quantifies excess device phase and has its own reference.

| Component | Mesh (cells/µm) | Output | Max power change | Max complex change | Max physical phase change (rad) | Max straight-relative phase change (rad) | Gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| coupler | 16→24 | left_lower | 0.01028 | 0.12901 | 2.6094 | — | Outside limit |
| coupler | 16→24 | left_upper | 0.00099 | 0.03481 | — | — | Outside limit |
| coupler | 16→24 | right_lower | 0.05280 | 0.78473 | 0.8864 | 0.1150 | Outside limit |
| coupler | 16→24 | right_upper | 0.03416 | 0.21404 | 0.8275 | 0.1236 | Outside limit |
| coupler | 24→28 | left_lower | 0.00070 | 0.01746 | — | — | Outside limit |
| coupler | 24→28 | left_upper | 0.00014 | 0.00735 | — | — | Pass |
| coupler | 24→28 | right_lower | 0.05648 | 0.16127 | 0.1711 | 0.0157 | Outside limit |
| coupler | 24→28 | right_upper | 0.01558 | 0.04326 | 0.1502 | 0.0121 | Outside limit |
| splitter | 16→24 | left_center | 0.01190 | 0.07565 | 0.6071 | — | Outside limit |
| splitter | 16→24 | right_lower | 0.01683 | 0.45898 | 0.7721 | 0.0034 | Outside limit |
| splitter | 16→24 | right_upper | 0.01683 | 0.45898 | 0.7721 | 0.0034 | Outside limit |
| splitter | 24→28 | left_center | 0.00470 | 0.07951 | 0.8644 | — | Outside limit |
| splitter | 24→28 | right_lower | 0.01507 | 0.12469 | 0.1998 | 0.0488 | Outside limit |
| splitter | 24→28 | right_upper | 0.01507 | 0.12469 | 0.1998 | 0.0488 | Outside limit |

Only the coupler’s opposite-input-side weak return meets both return gates at 24→28. Neither device’s forward phase qualifies on that pair of meshes. The comparison evaluates one full incident column per device. It preserves disagreements and records every signed complex amplitude without fitting a phase offset. The external coupler and Y-junction references measure other geometries or observables; independently verified matching port indices do not establish convergence of a bent circuit’s accumulated phase. [Coupler results](verification/v012-coupler-phase.json) · [Splitter results](verification/v012-splitter-phase.json) · [Previous paired Meep/CUDA controls](CALIBRATED_MULTIPORT.md) · [Lumerical and Tidy3D comparisons](CALIBRATED_VENDOR_CHECKS.md).

The two 28-cell studies each use 8,605,184 Yee cells. Under the measured-column resource policy, increasing this same full domain to 32 cells/µm would reserve about 17,600 MiB of free GPU memory, including workspace and headroom. The tested laptop has 16,384 MiB total. An additional mesh requires a different validated memory strategy or a larger device. [28-cell coupler and memory check](verification/v012-coupler-cuda28.json) · [28-cell splitter and memory check](verification/v012-splitter-cuda28.json).
