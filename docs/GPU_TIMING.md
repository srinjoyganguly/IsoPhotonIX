# GPU time for calibrated runs

Open **Calibrated components → GPU timing records** for the selected preset. The table lists each incident-reference and device stage alongside its actual grid and time steps. Download the CSV to compare meshes. A completed local result export contains its original numerical report, raw arrays and a timing CSV when stage measurements are available. [All recorded stage uses](verification/v014-gpu-performance.csv) · [Source receipt hashes and unique acquisition totals](verification/v014-gpu-performance.json).

A ring can finish its physical acquisition while individual wavelengths fail the observation-window check. The failed run then offers **Export diagnostic spectrum, validity mask and replay**. The package identifies the numerical gate failure and preserves the valid-bin mask. A replay reports whether the same measurements and mask were reproduced.

| Field | What was measured |
| --- | --- |
| Cells and steps | Allocated cells including absorbing boundaries, and the actual electromagnetic steps advanced. Some runs stop early on their measured decay criterion. |
| Compile | Time spent explicitly lowering and compiling the stepping function. A warm compiler cache can shorten it. |
| Stepping | Sum of host elapsed times around native dispatch and synchronization on the requested GPU. Field sampling performed inside the compiled update is part of that workload. Setup, explicit compilation and host-side monitor analysis are outside these scopes. |
| GPU clock | Sum of elapsed CUDA event intervals bracketing the same completed batches on the selected device. The start event completes before dispatch. The end event follows synchronization. The interval includes GPU idle and host launch gaps inside a batch. |
| Kernel active | Requires independent kernel tracing. On this WSL driver an experimental CUPTI tracer completed a small matrix product but several native FDTD runs failed during tracing. The workbench uses elapsed timers for native runs and leaves exclusive kernel-active time unavailable. |
| Earlier timer | Original measured duration where an older run used a broader timing scope. Each row identifies which operations were included. |
| Origin | A completed stage can appear in several spectra. New timed acquisitions have distinct IDs. A cache hit reuses the original timing and contributes no new electromagnetic computation. Totals deduplicate those IDs. Blank numerical values mean that measurement is unavailable. |

**Interpreting the cost.** The purpose column distinguishes production stages, memory probes, notebook replays and interrupted attempts. A lower bound on completed steps appears separately when a process failed before writing a timer. Compare the same physical device, duration, grid, boundary treatment, monitor set, solver version and execution target. A refined mesh typically increases both cell count and the number of steps needed to cover the same physical duration. Throughput in million cell-steps per second uses the synchronized stepping timer. The ledger reports elapsed time for the measured work and does not infer past kernel-active or stepping time from operation counts. [CUDA event timing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html).

The GUI’s 16-cell polygon-area splitter illustrates why these durations are kept separate. Its three input ports required six optical stages. The CUDA event intervals totalled 70.43 s, while the local job elapsed time was 1,717.38 s during a shared-GPU campaign. Job elapsed time includes waiting for the exclusive device, compilation, setup and analysis. [Measured matrix and stage times](verification/v014-splitter-polygon-matrix-gui16.json).
