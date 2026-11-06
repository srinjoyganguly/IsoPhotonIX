# Stability of the finite-height ring wall update

The 24-cell/µm Lumerical acquisition exposed a late instability in the original `wall-tensor` operator. The last quarter of all three Hz probe histories grew to hundreds of field units after the pulse ended. The highest collected guided power was 1.741. Of 241 wavelengths, 224 failed the recording-window check, 127 failed passivity, 147 failed incident magnitude and 118 failed incident phase. An isolated 1527 nm sample passed the local checks, but the unstable evolution invalidates its scattering result too. We preserve the native arrays and their original mask for diagnosis. The SHA-bound stability decision withholds the whole spectrum, all phases and every return peak. The device accumulated 134,360 steps over 6,250,272 cells in 1,432.121 s of synchronized step scopes, with 1,432.321 s on the CUDA event clock. [Native acquisition](verification/v014-ring3d-lumerical-r24-t3200-wall-tensor.json) · [Stability decision and measured probe growth](verification/v014-ring-wall-stability.json).

## Shared staggered coupling

The original xy and yx rows sampled the local inverse dielectric tensor at different Yee locations. As a result the Ex→Ey and Ey→Ex interactions between the same two edges could differ. Their largest mismatch in the tested 24-cell curved-wall tile was 0.05860 in inverse-permittivity units. This violates a known sufficient condition for stability of anisotropic Yee discretizations. The measurement supports this mechanism without proving that every observed feature in the earlier coarser runs has that cause. [Werner and Cary, 2007](https://www.sciencedirect.com/science/article/pii/S0021999107002227) · [Werner, Bauer and Cary, 2013](https://arxiv.org/abs/1212.4857).

The control uses the symmetry principle from those papers, with a pair averaging and bounding rule specific to this implementation. The distinct `reciprocal-wall` model retains the exact ring and bus areas, physical 100 nm gaps, component-centered diagonal inverse permittivities and vertical mixture. Each Ex edge has at most four adjacent Ey partners. For each pair we average its two locally sampled off-diagonal coefficients, bound this average and divide by four. Both update directions use this one stored link. Source planes remain diagonal and use the original straight-lead mode construction. The native FDTDX curl, PML memories, magnetic update and source operations are retained.

For a pair with diagonal inverse entries $d_i,d_j$, let $m=1/\epsilon_{\max}$ and $M=1/\epsilon_{\min}$ in the lossless dielectric model. Clamp the averaged off-diagonal entry to a magnitude at most

$$
\min\!\left(\sqrt{(d_i-m)(d_j-m)},\sqrt{(M-d_i)(M-d_j)}\right).
$$

Each stored link is one quarter of the bounded value. The full discrete operator $B$ is symmetric. For a real test vector $u$, the magnitude of $2B_{ij}u_i u_j$ is at most $((d_i-m)u_i^2+(d_j-m)u_j^2)/4$. Summing at most four pairs at each edge yields $u^T B u\ge m\lVert u\rVert^2$. Applying the same argument to $MI-B$ yields $u^T B u\le M\lVert u\rVert^2$, up to coefficient rounding. Here $M\le1$, so the uniform-grid vacuum Courant bound is sufficient for the source-free periodic lossless interior. The PML and driven device receive separate numerical checks. The link bound changes some interface weights, so spectral accuracy still needs independent mesh and vendor tests.

## Bounded checks before the device acquisition

The 16 × 16 × 6 periodic tile uses curved-interface coefficients cropped from the actual Lumerical 24-cell mesh near a 45° wall. Its assembled sparse matrix has zero transpose discrepancy and eigenvalues from 0.08257365 to 1.00000000. Random E/H fields excited the whole tile. In 8,192 CPU steps the relative drift of the discrete leapfrog energy, $E_n^T B^{-1}E_n+H_n^T H_{n-1}$, was 1.07 × 10⁻⁷ against the declared 2 × 10⁻⁴ limit. A separate analytic rotated standing wave had maximum field error 9.09 × 10⁻⁷. With a homogeneous rotated tensor, native source and six PML faces, the specialized and independent dense updates agreed to 1.27 × 10⁻⁷ or better in E, H and both PML memories. [CPU energy, eigenvalue, wave and PML receipts](verification/v014-reciprocal-wall-cpu.json) · [Validation program](../scripts/validate_reciprocal_ring_wall.py).

The native worker screens the full time trace before reporting a spectrum. After source shutoff, rejection requires increasing amplitudes at all three probes in the last two quarters, fourth-quarter peaks exceeding eight times each probe’s first-half maximum, and collected power above 1.05 in at least 10% of wavelengths. That combination clears all usable wavelength flags while preserving the local mask. Passing this severe-instability screen still leaves the per-wavelength launch, duration, passivity and commercial-accuracy checks in force.

## Memory admission for fresh high-resolution stages

At 20 or 24 cells/µm the reciprocal worker checks the budget inside each uncached reference/device calculation. It verifies the numerical source hashes, installed numerical payload, CUDA driver, GPU model, allocator settings, physical fixture and mesh. It reads free memory for the selected GPU immediately before allocating the new stage. The larger of measured board-memory increase and allocator peak in each 16-cell role is scaled by total cells, with 40% margin and another 2048 MiB reserve. The greater role estimate must fit. Verified existing stages reuse their earned acquisition receipts without consulting the reservation. The budget’s SHA and actual available memory are stored on newly acquired stages.

A local developer can profile the same model on a different GPU using the registered FDTDX interpreter and a source tree that matches the worker. Serialize profiling and solver work on that card. Use the resulting JSON through `PHOTONICS_RING_MEMORY_PLAN` when running a fresh high-resolution job.

```bash
"$FDTDX_PYTHON" -m scripts.profile_ring_wall_memory --source "$PWD" --vendor lumerical \
  --model reciprocal-wall --base-mesh 16 --candidate-mesh 24 --output my-ring-budget.json
PHOTONICS_RING_MEMORY_PLAN="$PWD/my-ring-budget.json" "$FDTDX_PYTHON" \
  -m scripts.run_reciprocal_ring3d --vendor lumerical --device gpu --resolution 24 \
  --duration-um-c 3200 --ring-interface reciprocal-wall --checkpoint my-stages --output fresh-ring
```

The same independently compiled periodic stress control ran on the RTX 3080 Ti Laptop GPU. Relative energy drift was 1.31 × 10⁻⁷ across 8,192 steps, with 0.334 s synchronized stepping and 0.342 s of CUDA event elapsed time. The 16-cell role probes measured a conservative 8,372 MiB requirement for the 24-cell full ring, versus 12,501 MiB free at its outer admission check. The worker repeats that check before each newly computed stage. [GPU energy receipt](verification/v014-reciprocal-wall-gpu.json) · [Measured memory profile](verification/v014-ring-memory-lumerical-r24-reciprocal.json) · [Outer admission](verification/v014-ring-admission-lumerical-r24-reciprocal.json).

The 24-cell pilot exercised the refusal path. After recording the straight stage, its process retained GPU allocations and the board reported 8,235 MiB free against the 8,372 MiB requirement. The native worker withheld the device stage. A fresh process reopened the verified straight checkpoint, released the prior allocator pool and rechecked the same budget before proceeding. The normal reciprocal entry creates a temporary checkpoint when none is supplied. It retries once, and only after observing this specific post-reference refusal and verifying a newly written reference archive. Other failures propagate. [Original admission and recovery evidence](verification/v014-reciprocal-ring-admission-recovery.json) · [Native orchestration](../scripts/run_reciprocal_ring3d.py).
