# sparsesom-roofline-addendum

[![arXiv](https://img.shields.io/badge/arXiv-2608.24067-b31b1b.svg)](https://arxiv.org/abs/2608.24067)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22245686.svg)](https://doi.org/10.5281/zenodo.22245686)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Instruction-level profiling addendum to the SparseSOM Phase 1 manuscript ([arXiv:2608.24067](https://arxiv.org/abs/2608.24067), §5.7): an
**Instruction Roofline** (Ding & Williams, *Instruction Roofline Model for GPUs*, PMBS
2019) and a **conventional FP32 roofline** of the best-matching-unit (BMU) phase of the
paper's three GPU implementations, from six Nsight Compute captures with an extended
counter set. The manuscript's §5.7 cites this repository; the main reproduction pipeline
is [`mongrolwarrior/sparsesom-paper1`](https://github.com/mongrolwarrior/sparsesom-paper1),
and the corpus is on Zenodo (concept DOI
[10.5281/zenodo.20770707](https://doi.org/10.5281/zenodo.20770707)).

## Why this is a separate repository

This measurement cannot run inside the standard reproduction container, by design.
Instruction-level profiling needs **Nsight Compute**, which the pinned image does not
carry (the `WITH_NCU=1` build argument adds it, ~1 GB), and GPU performance-counter
access, which needs the container run with **`--cap-add=SYS_ADMIN`** (or the host module
option `NVreg_RestrictProfilingToAdminUsers=0`) — an elevated capability many users
cannot or should not grant. The paper's reproduction (`repro --profile outline` / `full`)
therefore treats the roofline as skippable: the clean-room outline run passes 17/18
checks with the roofline check reporting SKIP, exactly as intended. Folding a
profiler-only path into that pipeline would make the standard reproduction depend on
privileges it deliberately does not require, for a result that is **diagnostic** (why the
fused kernel performs as it does) rather than load-bearing (none of the paper's
quantitative claims rest on it). Hence a deliberate split: the pipeline reproduces the
paper; this repository holds the profiler study.

## What was profiled — provenance

The **exact frozen builds the paper benchmarked**, on the same RTX 4090 host, same
corpus (`corpus.train.sbcsr`, 26,912,934 × 30,766, the authoritative 90/10 split):

- `sparsesom` (SparseBin.SOM) built from the tree released as
  [`SparseSOM`](https://github.com/mongrolwarrior/SparseSOM) commit `67f31fb`;
- `standardsparsesom` (cuSPARSE.SOM) built from the tree released as
  [`StandardSparseSOM`](https://github.com/mongrolwarrior/StandardSparseSOM) commit `dbdad1d`.

(The binaries were compiled from the pre-release working trees, which are byte-identical
to those release commits — the release squashed history, not content.)

Six captures, the same matrix as the paper's Tables 10–11 capture: {sbsom-bin fused,
ssom-feat, ssom-node} × {edge 128 **exact** (every launch), edge 256 **sampled** at
`NCU_CAP=60` launches and scaled per kernel by its own captured-launch count against the
6,571 tiles per epoch — the paper's published methodology}. One epoch, seed 0, fp16
codebooks on both sides. Profiler wall time ≈ 5.2 h (SASS-level counters replay each
kernel many times).

## Validation against the paper's frozen measurements

This capture must talk about the paper's kernels, so it first has to reproduce the
paper's numbers (Tables 10–11; frozen `roofline_phase_summary.csv` in
`sparsesom-paper1/frozen/2026-08-02/`):

| kernel set | DRAM TB @128² — this / frozen | DRAM TB @256² — this / frozen | kernel-time s @128² — this / frozen | @256² — this / frozen |
|---|---|---|---|---|
| sbsom-bin fused | 2.815 / 2.818 | 18.976 / 18.969 | 10.29 / 10.29 | 42.37 / 42.34 |
| ssom-feat phase | 3.338 / 3.103 | 13.633 / 13.847 | 8.39 / 8.12 | 63.18 / 62.79 |
| ssom-node phase | 3.874 / 3.843 | 22.716 / 22.859 | 70.33 / 70.25 | 306.5 / 306.3 |

Same DRAM-traffic sign crossover between 128² and 256², magnitudes within ~1–8 %. The
256² tile-sampling factors here are ×329–346 (19–20 tiles captured of 6,571) versus the
frozen run's ×22/×329; the agreement of the whole-phase totals across both sampling rates
is itself a validation of the scaling.

## Ceilings — measured on the device, not from a spec sheet

All roofs come from Nsight Compute's `peak_sustained` counters on this 4090 (SM clock
2.52 GHz), recorded in `data/roofline2_ceilings.csv`: **DRAM 983 GB/s · L2 5.0 TB/s ·
L1 165 TB/s · warp-issue 1,289 G warp-instructions/s**, plus FP32 peak **82.5 TFLOP/s**
(128 SMs × 128 lanes × 2 × measured clock). Conventional DRAM ridge: 84 FLOP/byte;
Instruction-Roofline DRAM ridge: 42 warp-instructions per 32-byte sector.

## Findings, each traceable to a file here

- **The fused binary kernel executes essentially no floating-point multiply**: 0.05 G
  FMUL+FFMA against 5,727 G FADD per epoch at 128² — one multiply per *sample* (the final
  distance), none per neuron (`data/roofline2_kernels.csv`, columns `op_fmul_G`,
  `op_ffma_G`, `op_fadd_G`). Its instruction stream is 53 % integer, 19 % bit, 12 %
  memory, ~8 % FP32 add; the `op_hadd_G` column is the FP16→FP32 conversions
  (`HADD2.F32`), not arithmetic. FP32 throughput: **0.67 % of peak**.
- **It reaches 30–46 % of the nearest ceiling at every level, and no more than 32 % of
  warp-issue peak** (`data/ceiling_utilisation.csv`): 32.3 / 30.0 % of issue, 27.8 /
  45.5 % of DRAM, 13 % of L2, 0.4 % of L1 at 128² / 256²; no execution pipe above 23 %
  (`pipe_*` columns in `roofline2_kernels.csv`). A kernel under **every** roof by 2–3× is
  **latency-limited**, not bandwidth- or compute-bound.
- **Why: occupancy.** The kernel compiles to **121 registers per thread**
  (`data/kernel_resources_cuobjdump.txt`); a 256-thread block takes 31 k of the SM's
  65,536 registers → 2 blocks/SM → 16 of 48 warps → **33 % theoretical occupancy**, while
  every codebook read misses L1 (L1 bytes ≈ L2 bytes in `roofline2_kernels.csv`), so too
  few warps are in flight to hide L2/DRAM latency. This is the price of the
  16-samples-per-tile design that shares each codebook read across the tile — it buys the
  DRAM saving the paper's Table 10 shows at 128² and pays in latency exposure.
- By-products: **node-major csrmm is L2-bandwidth-bound** (151 TB through L2 at 128²,
  17× feature-major's 8.9 TB, at ~46 % of L2 peak while DRAM sits at 4 %) — the paper's
  layout penalty as an access-pattern fact; and the **argmax read-back** amplifies DRAM
  13–16× through L1 at 2–4 % of issue — a latency-bound strided scan, which is why fusing
  it away wins even where the fused kernel moves more DRAM bytes.

## The hypothesis this tested — including the half that was wrong

Before the capture, the prediction (recorded 2026-08-24) was: *the conventional FLOP
roofline is uninformative for the fused kernel — it does ~no FMUL, sits far below the
ridge, and would be classified memory-bound while DRAM is only 29–45 % utilised; the
real limiter is instruction issue or L1/shared traffic.*

- **Confirmed:** ~zero FMUL (measured); 0.67 % of FP32 peak; 1.5–2 decades left of the
  84 FLOP/B ridge; "memory-bound" by classification at only 28–46 % of the DRAM roof.
  The conventional roofline (`figures/fig_conventional_roofline.*`) adds nothing beyond
  the paper's Table 11 for this kernel.
- **Refuted:** the fused kernel is **not** issue-bound (30–32 % of issue peak) and
  **not** L1/shared-bound (0.4 % of L1). It is under every ceiling — the signature of a
  latency/occupancy limit, established by the register count above. Only the Instruction
  Roofline (`figures/fig_instruction_roofline.*`) could show this: the conventional one
  cannot distinguish "at the memory roof" from "under all roofs".

## Contents

```
capture.sh                      the six ncu invocations, exactly as run (metric list inside)
aggregate2.py                   raw ncu CSVs -> per-kernel/phase tables (incl. 256² tile scaling)
plot_rooflines.py               the two figures from the aggregated tables
data/ncu2_raw_captures.tar.gz   the six raw Nsight Compute CSVs
data/roofline2_kernels.csv      per-kernel counters, op mix, intensities, pipe utilisation
data/roofline2_phase.csv        per-implementation phase totals
data/roofline2_ceilings.csv     the ncu-measured device ceilings
data/roofline2_points.csv       every point plotted in the figures
data/ceiling_utilisation.csv    % of each ceiling per kernel (the 30-46 % / 32 % numbers)
data/kernel_resources_cuobjdump.txt  register/shared usage (the 121 registers)
figures/fig_instruction_roofline.{png,pdf}
figures/fig_conventional_roofline.{png,pdf}
MANIFEST.sha256                 hashes of everything above (sha256sum -c MANIFEST.sha256)
```

To re-run the capture (not required to inspect any number above): build the pipeline
image with `--build-arg WITH_NCU=1`, run with `--gpus all --cap-add=SYS_ADMIN`, and
invoke `capture.sh 128` and `capture.sh 256 60` with the corpus path; then
`aggregate2.py` and `plot_rooflines.py`. Expect ~5 h of profiler time.

## Licence and data notices

MIT (see `LICENSE`), matching the five release repositories. The corpus the profiled
runs consumed is the anonymised MEDLINE/MeSH `.sbcsr` from the Zenodo record above:
data **courtesy of the U.S. National Library of Medicine**; a fixed 2026-baseline
snapshot that does not reflect the most current NLM data; NLM does not endorse or
recommend this work.


## Citation

This repository is the instruction-level profiling addendum to:

> Amos, A. J. (2026). *A Feature-Major Codebook for Memory-Efficient Sparse-Binary
> Self-Organizing Maps: Scaling a MEDLINE Atlas to 1.05 Million Neurons on a Single
> Consumer GPU.* arXiv:2608.24067 [cs.LG].
> <https://doi.org/10.48550/arXiv.2608.24067>

```bibtex
@misc{amos2026featuremajor,
  title         = {A Feature-Major Codebook for Memory-Efficient Sparse-Binary
                   Self-Organizing Maps: Scaling a MEDLINE Atlas to 1.05 Million
                   Neurons on a Single Consumer GPU},
  author        = {Amos, Andrew James},
  year          = {2026},
  eprint        = {2608.24067},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2608.24067}
}
```

Please cite the paper. To cite this repository specifically, use its archived release
on Zenodo: <https://doi.org/10.5281/zenodo.22245686> (concept DOI - always resolves to the
latest version; v1.0 is [10.5281/zenodo.22245687](https://doi.org/10.5281/zenodo.22245687)).
