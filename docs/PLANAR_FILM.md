# Passive planar film spectrum

Select a solid and open **Inspector → Material model → Planar film spectrum**.
The calculation uses the selected solid's Z thickness and its assigned optical
material, including imported wavelength-dependent n,k samples. A solid with
a scalar index uses k=0. Set the incident and substrate indices and the
wavelength band, then select **Calculate film spectrum**. Move the wavelength
slider to read reflected, transmitted and absorbed power and the two complex
electric-field amplitudes. This study treats the film as infinite in XY.
Sidewalls, guided modes and the rest of the 3D design are outside its geometry.

The two exterior media are semi-infinite, lossless and nonmagnetic. The film
is isotropic, nonmagnetic and passive. At normal incidence both transverse
polarizations share these coefficients. With time convention exp(−iωt), the
forward wave in the film has the factor exp(+i 2π(n+ik)z/λ). Positive k
attenuates it. The incident electric amplitude is 1 at the entrance.
Reflection amplitude r is referenced to that face and transmission amplitude
t is referenced to the exit. Reported powers are R=|r|²,
T=(n_substrate/n_incident)|t|² and A=1−R−T. The absorbing-film calculation
uses decaying propagation factors, including when a thick film's transmitted
field underflows. The lossless result must conserve power within floating
point tolerance. [Byrnes, *Multilayer optical calculations*](https://arxiv.org/abs/1603.02720).

A constant n,k record carries its stated assumption over the requested band.
For a table the existing material validator interpolates n and k separately
inside the supplied interval. Requests outside the table are rejected. A
sweep accepts 1–201 strictly increasing vacuum wavelengths between 0.001 and
1,000 µm. For one wavelength set the two endpoints equal and select one sample.
The adjacent exterior indices must be between 0.001 and 20.

**Export film Python + notebook** downloads the exact spectrum settings, full
precision JSON, CSV, Python and a notebook. Extract the ZIP and run
`python film.py` or open `film.ipynb` in that folder. The analytic calculation
uses the Python standard library. [Browser and export verification](verification/v010-thin-film-exports.json).

## Independent Meep checks

The test film is 0.4 µm thick with air in front and a substrate of index
1.45. Every case uses an empty homogeneous reference and a real device run at
80, 160 and 320 cells/µm. Meep subtracts the incident Fourier fields for the
reflection monitor. The report retains both flux monitors, both complex Ex
phasors, and the phases used to move r and t to the physical interfaces.
For each lossy target wavelength, the separate Meep run sets
ε∞=Re[(n+ik)²] and σ_D=2πf Im[(n+ik)²]/ε∞. This matches the material at
that bin. Each run measures power from its fields. Absorption in both columns
is inferred from the power balance.
[Meep conductivity convention](https://meep.readthedocs.io/en/latest/Materials/) ·
[Meep normalization and reflection tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/).

| Vacuum λ (µm) | Film n+ik | Analytical R / T / A | Meep 320 cells/µm R / T / A |
| --- | --- | --- | --- |
| 1.55 | 2.00 + 0i | 0.036075 / 0.963925 / 0 | 0.036074 / 0.963926 / −4.70 × 10⁻⁹ |
| 1.45 | 2.05 + 0.10i, interpolated | 0.085240 / 0.632083 / 0.282677 | 0.085407 / 0.631176 / 0.283417 |
| 1.55 | 2.00 + 0.08i | 0.050988 / 0.723177 / 0.225834 | 0.051169 / 0.722301 / 0.226530 |

Across the three finest meshes the largest power-fraction difference was
0.000907 and the largest complex amplitude difference was 0.000513. For
the lossless control, an initial 1 µm PML left a residual power imbalance
of 2.15 × 10⁻⁶ on the finest mesh. Widening the PML to 2 µm, increasing
the run time and recording every Fourier update reduced that residual to
4.70 × 10⁻⁹. This residual did not decrease monotonically as the spatial
mesh was refined. It is reported with a dedicated absolute bound of 10⁻⁷
on each of the three meshes. The 0.002 bound on power and complex-amplitude
differences applies to every finest-mesh observable. These comparisons
qualify the specified planar cases. [Raw reference, device and boundary-control readings](verification/v010-thin-film-meep.json).

From the checkout, reproduce the 24 native simulations with the Meep
interpreter:

```bash
.local/meep-conda/bin/python scripts/verify_thin_film.py --output film-meep-check.json
```
