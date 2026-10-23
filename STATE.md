# STATE — read this first

_Last updated: 2026-10-23, session 1. Branch: `claude/ncm-research`._

## Current phase
**Phase 0 (harness + baselines)** nearly done for the released checkpoints; Phase 1 draft written
(`docs/phase1_literature_and_candidates.md`); Phase-2 pilot P1 ready to launch on Colab.

## Environment facts that change the plan
- This agent runs in a **cloud container with no GPU** (4-core Xeon with AMX, 15 GB RAM), not on the
  RTX 4050 laptop. Measured: ~0.5 TFLOPS fp32, ~2-3 TFLOPS bf16 on large matmuls. One heavy CPU job at a
  time (two 4-thread jobs slow each other ~10x). Small pilots here; matched baselines on Colab.
- arXiv/stanford.edu/alphaxiv are unreachable (DNS): **the NCM paper text has not been read.** Facts come
  from code (vendored, upstream c702742), HF abstract, GitHub issues/PRs, and the independent repro
  (HF `velaia/ncm-reproduction`). User asked to drop the PDF into `docs/`.

## Established (measured unless marked)
- **Sudoku-hard reproduced: 86.30%** (paper 86.1). Easy/medium: see R004+ in log.md.
- Released TinyStories tokenizer: lookup-table encoder; ~lossless at full depth (184 ppm token errors;
  names 918 ppm, rare words 4588 ppm); PAD never reconstructed (packed training windows).
- Coarse codes behave like window bags-of-words (R003).
- Sudoku success falls with constraint-propagation depth (R004): 99% at 5-6 rounds -> 64% at >=11 -> 50%
  needing search. No board correct before the last level.
- NCM costs ~4.25x AR FLOPs per sequence at equal params (641 vs 128 positions); its edge is NFE (11 vs 128).
- Training code for upstream NCM is still unreleased (issues #2/#10).

## Live hypotheses
- H3 (P1): a large share of the NCM–AR NLL gap is within-level conditional total correlation (TC).
- H4 (C4): carrying the generator state across levels (trained with backprop through levels) raises the
  propagation depth NCM can handle (Sudoku rounds >= 9).
- H5 (C6): a contextual encoder removes the bag-of-words property of coarse codes.

## Killed ideas
- "Fix the lossy tokenizer" as a primary lever on TinyStories: full-depth error is 0.018%; not the bottleneck.
- Level-skipping early exit with the released Sudoku hierarchy: never fires (0% solved before last level).

## Compute
- Colab: 0 / 100 CU spent. Requested: `p1_tc_pilot` (~1.5 CU) and `c4_sudoku_pilot` (~3 CU) — awaiting launch (docs/colab_runs.md).
- CPU (container): ~1.5 CPU-h so far.

## Next 3 actions
1. Finish easy/medium Sudoku repro (running); then released TinyStories generator: per-level NLL, 5x256
   samples, full eval suite (reproduce genPPL 15.41 / entropy 3.98).
2. Level probes (`scripts/probe_levels.py`) on the released tokenizer; then train tok_s on CPU and run the
   same probes (lookup vs contextual encoder = C6) while Colab pilots run.
3. When P1/C4 results arrive: `scripts/analyze_p1.py`, compare C4 arms against docs/preregistration.md.
