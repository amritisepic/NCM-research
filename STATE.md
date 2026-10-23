# STATE — read this first

_Last updated: 2026-10-23, session 1. Branch: `claude/ncm-research`. Repo is PUBLIC: commit nothing
identifying (no personal names/emails, no absolute local paths, no session links, no credentials).
Goals, success bar and rules: `docs/brief.md`._

## Current phase
Phase 0 done for released checkpoints. Phase 1b re-planned for ~200 CU (`docs/phase1b_scaleup.md`: tiers
S / L (paper-scale fine-tunes) / M (from-scratch matched), waves 0-3). Phase 2: P1 and C4 pilots running on
Colab; wave 0 and wave 1 notebooks ready; wave 1 pre-registered (`docs/preregistration.md` W1).
Scale-limit change **approved** by the user (2026-10-23): tiers L/M use the released 88.6M generator /
36.4M tokenizer. `w0_calib_diag` launched on L4 (2026-10-23). P1 and C4 results arrived in
`results/colab/` (complete, all runs present) and are **not yet analyzed**.

## Environment
- Cloud agent: no GPU (4-core Xeon w/ AMX). One heavy CPU job at a time (pause others with SIGSTOP).
- Colab (user launches notebooks from `notebooks/`, see `docs/colab_runs.md`); prefer L4.
- Optional laptop GPU via a separate local session: `docs/laptop_gpu_runbook.md`.
- Paper: `docs/2610.14162v1.pdf` (read).

## Established (measured unless marked)
- Sudoku reproduced (99.4 / 96.4 / 86.3); success falls with propagation depth on all three sets (R012).
- Released TinyStories NCM: Gen-PPL 16.10 ± 0.59, H 3.971, MAUVE 0.878 (R010; paper 15.41).
- Coarse codes are window bags of words (R011, n = 1000).
- Level probes (R016): global attributes (dialogue, main character, content) saturate by L5-L6; the token-
  straddling levels L8-L10 carry near-chance content in their own codes (L7 and L11 do not).
- Released generator code NLL 870 nats/window, 43% in token-straddling levels (R008).
- Exact Jacobi across levels: Sudoku-hard -47% NFE at zero quality change (1.87x FLOPs); no saving on
  text training-free (R013).
- **R014:** the generator's own one-step errors at every level keep 98% of tokens (uniform corruption keeps
  43%); fine-level errors are near-equivalent codes; re-quantized labels lift the corrupted-reconstruction
  ceiling only modestly (92.9 -> 96.9% uniform, 99.40 -> 99.84% on-policy).
- Paper (not ours): AR 8.89 vs NCM 15.41 Gen-PPL; corruption is NCM's biggest lever; more levels help.
- NCM costs 4.25-4.66x AR FLOPs per sequence at equal params.

## Live hypotheses / bets (ranked; docs/phase1b_scaleup.md §5)
1. B1a on-policy corruption (one-step, two-pass) — tier-L factorial W1 (H6, H8).
2. B2-trained consistency for text Jacobi (criterion d) — WMT14, to build.
3. B6/C5 Sudoku repair passes (+ C4 carry if H4 holds) — tier L, to build after C4 reports.
4. C2 red-black within-level steps — conditional on P1 (H3) / wave-0 diagnostic.
5. B1b re-quantized labels (H7), B7 soft labels (H9) — in W1. Token-aligned schedule, B3/B4 — tier S.
- H3 (P1, results in, to analyze): within-level TC share of the NCM–AR gap. H4 (C4, results in, to analyze):
  carry raises solvable depth.

## Killed / demoted
- Fixing tokenizer loss on TinyStories (0.018% error). Level-skipping early exit on released Sudoku.
- C7, C8, C11 demoted. Training-free Jacobi on text (no saving).

## Compute
- Colab: p1_tc_pilot (T4, 2.9 h training wall, ~3.5 CU + setup) and c4_sudoku_pilot (T4, 4.8 h, ~5.7 CU +
  setup) done: ~10 CU spent. w0_calib_diag running (L4, est. 2.5 CU). Budget ~200 CU.
- CPU: ~8 CPU-h so far. No CPU job running.

## Next 3 actions
1. Analyze P1 (`scripts/analyze_p1.py` on `results/colab/p1_tc_pilot/`) and C4 (`results/colab/c4_sudoku_pilot/`,
   overall and by propagation depth) strictly against `docs/preregistration.md`; log R017/R018; apply the
   decision rules in `docs/phase1b_scaleup.md` §7 (C2 promotion; Sudoku carry). Note: P1's tokenizer is our
   tok_s, not the released one.
2. When w0 results arrive (`results/colab/w0_calib_diag/`): sanity-check the released reference (val NLL ~870,
   Gen-PPL ~16.1), set wave-1 step counts from measured throughput (cut steps, not seeds, if < 60% of the
   estimate), read the quality-loss-by-level curve, then ask the user to launch the six `w1_*` notebooks.
3. Build the Sudoku tier-L fine-tune (control / on-policy / repair passes / carry if H4 holds) and the WMT14
   consistency fine-tune (B2-trained, criterion d); smoke-test on CPU; add notebooks.
