# STATE — read this first

_Last updated: 2026-10-23, session 1._

## Current phase
**Phase 0 (harness + baselines)**, with Phase 1 literature search running in parallel.

## Environment facts that change the plan
- This agent runs in a **cloud container with no GPU** (4-core Xeon with AMX, 15 GB RAM), not on the
  RTX 4050 laptop. Measured: ~0.5 TFLOPS fp32, ~2-3 TFLOPS bf16 on large matmuls. Small-model pilots
  (≤ ~5M params, ≤ ~1 PFLOP) run here; the matched 25M-param baselines go to Colab.
- arXiv, stanford.edu, alphaxiv are unreachable (DNS) from here: **the NCM paper text has not been
  read.** Facts about NCM come from the released code (vendored at `src/ncmr/ncm_ref`, upstream
  c702742), the HF abstract, GitHub issues/PRs, and an independent reproduction (HF dataset
  `velaia/ncm-reproduction`). Ask the user for the PDF (drop into `docs/`).

## Key facts established so far (measured unless marked)
- Training code is still unreleased (issues #2/#10; authors say "shortly").
- Released TinyStories tokenizer: **encoder = token + position embedding lookup (enc_layers=0)** followed by
  GroupNorm; 12 levels [1,8,16,24,32,40,52,64,76,92,108,128], 4096-entry codebooks, dim 384; decoder =
  1 transformer block, no positional embedding. => coarse codes are quantized window-averages of
  context-free token embeddings. (code reading; verify with probes)
- Generator: GPT-2-small size, block-diagonal level mask, level 0 drawn from a stored unigram, 11 NFE.
- Sudoku test sets: all puzzles unique; easy/medium/hard share solution grids; difficulty = 41/46/51
  blanks; 85.5% of hard solvable by singles-propagation alone (mean 7.5 parallel rounds).
- Independent repro (not ours): Sudoku 99.4/96.4/86.3 vs paper 99.5/96.5/86.1; TinyStories genPPL
  15.98±0.28 vs 15.41. Paper timings need bf16 + KV cache + CUDA graphs (issue #9).

## Live hypotheses
- H1: NCM Sudoku success is predicted by constraint-propagation depth (rounds) per puzzle.
- H2: Released TinyStories coarse codes are ~permutation-invariant within windows (bag-of-words), so
  "concepts" carry no word-order information at coarse levels.
- H3: A large share of the NCM–AR quality gap is within-level conditional total correlation (TC),
  measurable as (factorized NLL − within-level-AR NLL) on the same tokenizer.

## Killed ideas (one line each)
_(none yet)_

## Compute
- Colab: 0 / 100 CU spent.
- Local CPU (cloud container): ~1 CPU-h so far (setup, audits, repro in progress).

## Next 3 actions
1. Finish Sudoku repro (3 x 2000 boards) + per-level trajectories; test H1.
2. Finish tokenizer per-level/per-class analysis; run within-window shuffle test for H2.
3. Write evaluation suite (2 scorers, entropy, repetition, MAUVE) and reproduce TinyStories genPPL.
