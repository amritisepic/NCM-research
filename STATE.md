# STATE — read this first

_Last updated: 2026-10-23, session 1. Branch: `claude/ncm-research`. Repo is PUBLIC: commit nothing
identifying (no personal names/emails, no absolute local paths, no session links, no credentials)._

## Current phase
Phase 0 done for released checkpoints. Phase 1b re-planned for ~200 CU (`docs/phase1b_scaleup.md`: tiers
S / L (paper-scale fine-tunes) / M (from-scratch matched), waves 0-3). Phase 2: P1 and C4 pilots running on
Colab; wave 0 and wave 1 notebooks ready; wave 1 pre-registered (`docs/preregistration.md` W1).
**Pending user decision:** scale-limit change (tiers L/M use the released 88.6M generator / 36.4M tokenizer).

## Environment
- Cloud agent: no GPU (4-core Xeon w/ AMX). One heavy CPU job at a time (pause others with SIGSTOP).
- Colab (user launches notebooks from `notebooks/`, see `docs/colab_runs.md`); prefer L4.
- Optional laptop GPU via a separate local session: `docs/laptop_gpu_runbook.md`.
- Paper: `docs/2610.14162v1.pdf` (read).

## Established (measured unless marked)
- Sudoku reproduced (99.4 / 96.4 / 86.3); success falls with propagation depth on all three sets (R012).
- Released TinyStories NCM: Gen-PPL 16.10 ± 0.59, H 3.971, MAUVE 0.878 (R010; paper 15.41).
- Coarse codes are window bags of words (R011, n = 1000).
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
- H3 (P1, running): within-level TC share of the NCM–AR gap. H4 (C4, running): carry raises depth.

## Killed / demoted
- Fixing tokenizer loss on TinyStories (0.018% error). Level-skipping early exit on released Sudoku.
- C7, C8, C11 demoted. Training-free Jacobi on text (no saving).

## Compute
- Colab: p1_tc_pilot + c4_sudoku_pilot launched (est. 4.5-9 CU; actuals pending). Budget ~200 CU.
- CPU: ~5 CPU-h so far. `scripts/probe_levels.py --n 4000` still running (queue1).

## Next 3 actions
1. User: confirm scale-limit change; launch `w0_calib_diag` (L4), then the six `w1_*` arms (seed 0);
   `w2_M_ncm` / `w2_M_ar` any time.
2. When P1/C4 arrive: analyze vs preregistration; build Sudoku tier-L notebook (control / on-policy / repair
   passes / carry) and the WMT14 consistency fine-tune.
3. When w0 arrives: set wave-1 step counts from measured throughput; read the quality-loss-by-level curve.
