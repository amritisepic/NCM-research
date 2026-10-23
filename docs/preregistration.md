# Pre-registrations (written before the runs; do not edit after results arrive — append instead)

## P1 — How much of NCM's NLL gap to AR is within-level dependence? (registered 2026-10-23)
- **Setup:** one small NCM-style tokenizer (`configs/pilot/tok_s.yaml`, lookup encoder like released NCM,
  12 levels, 1024 codes, ~8.4M params). On its codes: NCM generator (`ncm_s`), two-stream factorized
  probe (`ts_fact_s`), two-stream within-level-AR probe (`ts_ar_s`); plus AR-token baseline (`ar_s`),
  same transformer body (d256 x 6 layers), FLOP-matched. Seed 0. Val = first 1000 packed val windows.
- **Hypothesis H3:** within-level conditional total correlation, TC = Σ_k (NLL_ts_fact,k − NLL_ts_ar,k),
  accounts for a large share (≥ 30%) of the gap NLL_NCM − NLL_AR (nats per 128-token window; NCM NLL
  includes level 0 under its unigram).
- **Metric:** TC (nats/window) and TC / gap, at the final checkpoint; TC per level; trend over training.
- **Counts as support:** TC ≥ 30% of the gap and TC increasing or stable over the last 3 evals.
- **Falsified if:** TC < 10% of the gap (then within-level refinement — C2, C7 — cannot close the gap and
  is deprioritized; the representation term becomes the target: C6, C1b, C8).
- **Sanity checks:** NLL_ts_fact within 5% of NLL_ncm (same information); tokenizer full-depth val
  accuracy ≥ 99%; per-level truncated-decode curve monotone (qualitative match to released NCM).

## C4 — Recurrent NCM on Sudoku-hard (registered 2026-10-23)
- **Setup:** our Sudoku tokenizer (`configs/sudoku/tok.yaml`) trained on 98k generated hard puzzles
  (51 blanks, unique solutions; 2k held out as val); two generators with identical config
  (`configs/sudoku/gen_s.yaml`, d256 x 6 layers, ~4.8M params, 20k steps x 64, seed 0): (a) NCM (no
  carry, trained via the sequential path, mathematically identical to upstream's parallel pass);
  (b) RNCM = (a) + carried hidden state across levels (zero-init projection), full backprop through levels.
  Same NFE (12) and parameters (+66k for the carry projection, also present but unused in (a)).
- **Hypothesis H4:** carrying state raises the propagation depth NCM can handle.
- **Metric:** greedy solve rate on our 1000-puzzle val set (selection) and, once at the end, on the 2000
  released hard test puzzles, overall and by propagation depth (≤6, 7-8, 9-10, ≥11, needs search).
- **Counts as support:** (b) − (a) ≥ +3 points overall on val AND ≥ +10 points on test puzzles with ≥ 9
  propagation rounds (n = 441).
- **Falsified if:** (b) − (a) < +1 point overall (then C4 is killed for Sudoku), or (b) fails to train.
- **Caveat:** one seed per arm; a positive result triggers a 3-seed confirmation before any claim.
