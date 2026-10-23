# STATE — read this first

_Last updated: 2026-10-23, session 1. Branch: `claude/ncm-research`. Repo is PUBLIC: commit nothing
identifying (no personal names/emails, no absolute local paths, no session links, no credentials)._

## Current phase
Phase 0 done for released checkpoints (Sudoku reproduced; TinyStories eval running). Phase 1 re-evaluated
after reading the paper: `docs/phase1b_reevaluation.md` (bets B1-B7). Phase 2: P1 and C4 running on Colab.

## Environment
- Cloud agent: no GPU (4-core Xeon w/ AMX; ~0.5 TFLOPS fp32, ~2-3 bf16 peak). One heavy CPU job at a time.
- Colab (user launches notebooks from `notebooks/`, see `docs/colab_runs.md`).
- Optional laptop GPU via a separate local session: `docs/laptop_gpu_runbook.md`.
- Paper: `docs/2610.14162v1.pdf` (read).

## Established (measured unless marked)
- **Sudoku reproduced:** easy 99.4 / medium 96.4 / hard 86.3 (paper 99.5 / 96.5 / 86.1), NFE 12.
- Sudoku success falls with constraint-propagation depth (99% at 5-6 rounds -> 64% at >=11 -> 50% needing
  search); no board correct before the last level.
- TinyStories tokenizer: lookup encoder; ~lossless (184 ppm token errors); coarse codes are window
  bags-of-words — with coarser levels held fixed, L1/L2 codes are 100% invariant to in-window shuffles (40-ex check).
- Released generator's teacher-forced code NLL: 870 nats/window (6.8 nats/token); 43% in the
  token-straddling levels (76/92/108 codes).
- Paper (not ours): TinyStories AR 8.89 vs NCM 15.41 Gen-PPL (data-matched training; AR used ~4x fewer
  FLOPs); conditioning corruption is NCM's biggest lever; more levels keep helping.
- NCM costs ~4.25x AR FLOPs per sequence at equal params.

## Live hypotheses / bets (see docs/phase1b_reevaluation.md)
- H3 (P1, running): within-level TC is a large share of the NCM–AR NLL gap.
- H4 (C4, running): carrying generator state across levels raises solvable propagation depth.
- B1 (top priority): self-correcting NCM — on-policy errors + re-quantized targets.
- B2: Jacobi decoding across levels (exact, adaptive) — training-free test queued on CPU.
- B7: equivalence-class soft targets (nuisance entropy). B3: CorEx/RSMI tokenizer (high risk).

## Killed / demoted
- Fixing tokenizer loss on TinyStories (0.018% error). Level-skipping early exit on released Sudoku hierarchy.
- C7 (importance resampling), C8 (lifting tokenizer), C11 (JEPA) demoted.

## Compute
- Colab: launched p1_tc_pilot (~1.5 CU est.) and c4_sudoku_pilot (~3 CU est.); actuals pending.
- CPU: ~3 CPU-h so far.

## Next 3 actions
1. CPU queue (running): B2 Jacobi on Sudoku-hard; released TinyStories eval suite; full invariance probe;
   level probes.
2. Implement NCM conditioning corruption + B1 (shared code path) + B7 soft targets; small-scale kill test configs.
3. When P1/C4 results arrive: analyze against docs/preregistration.md; update priors.
