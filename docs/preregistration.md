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

## C6 — Does a contextual encoder remove the bag-of-words property of coarse codes? (registered 2026-10-23)
- **Setup:** `configs/pilot/tok_s.yaml` (lookup encoder, as NCM) vs `configs/pilot/tok_ctx1_s.yaml`
  (identical + 1 bidirectional transformer layer in the encoder; 9.22M vs 8.43M params). Seed 0, same
  data order, CPU. Probes: `scripts/probe_invariance.py` and `scripts/probe_levels.py` (--tokenizer).
- **Hypothesis H5:** with context, coarse codes become order-sensitive.
- **Metric:** % of level-1 codes unchanged under within-16-token-window shuffles, and of level-2 codes under
  within-8-token shuffles; per-affected-window change rate under single synonym swaps; full-depth recon.
- **Counts as support:** shuffle invariance drops by >= 20 points at L1 and at L2 relative to the lookup
  tokenizer, with full-depth reconstruction >= 99% for both.
- **Falsified if:** invariance within 5 points of the lookup tokenizer at both L1 and L2.
- **Why it matters:** if context breaks the bag-of-words structure, "semantic coarse levels" (idea 2) are
  reachable by architecture alone; P1-style TC measurements on both tokenizers then show whether that costs
  parallelizability (the trade-off C1b would optimize).

## W1 — Tier-L TinyStories objective factorial (registered 2026-10-23, before any run)
- **Setup:** the released 88.6M TinyStories generator and tokenizer (tokenizer frozen). Each arm fine-tunes
  for the same training FLOPs (~349 PFLOP = 4000 steps x 256 windows of plain teacher forcing): lr 5e-5
  (200 warmup, cosine to 5e-6), AdamW (0.9, 0.95), weight decay 0.1, dropout 0.1, same data order per
  seed (`configs/scale/L_ts_*.yaml`). Arms differ only in the training hierarchy:
  A0 released model, no training (reference); A1 control = uniform corruption (mean 0.5, linear 0.75 -> 0.25,
  clean labels); A2 uniform + re-quantized labels; A3 one-step on-policy codes + clean labels (3000 steps);
  A4 one-step on-policy + re-quantized labels (3000 steps); A5 two-pass on-policy + clean labels (2400
  steps); A6 A1 + soft labels (tau 0.25, 16 nearest codes).
- **Metrics:** primary Gen-PPL (GPT-2 large, upstream function) on 4 x 256 unconditional samples (seeds 42,
  0, 1, 2; paper protocol, 11 NFE), always with unigram entropy; co-primary MAUVE (1024 samples vs 1024
  human val windows, 5 quantization seeds). Secondary: TinyStories-33M Gen-PPL, rep-n, distinct-n,
  repeated words at distance 1 and 2, clean teacher-forced val NLL. Curves every 1000 steps (256 samples).
- **Hypotheses:** H6 (B1a) A3 < A1 on Gen-PPL with MAUVE not lower; H7 (B1b) A2 < A1 and A4 < A3;
  H8 (compounding) A5 < A3; H9 (B7) A6 < A1.
- **Screen (seed 0):** an arm advances to seeds 1-2 if Gen-PPL is >= 0.5 below A1 (the 4-group mean has
  SE ~0.3 from R010's per-group SD 0.59), entropy within 0.05 nats of A1, MAUVE not lower by > 0.01.
  A1 always gets seeds 1-2.
- **Claim (Phase 3 rule):** with 3 seeds, Gen-PPL gain > 2x the pooled seed SD, entropy within 0.05 nats
  (otherwise re-sampled at a temperature that matches entropy before comparing), MAUVE not worse beyond
  its SD. A tier-L claim is stated as "improves the released model at equal fine-tuning compute" until
  tier M confirms it from scratch.
- **Falsified:** no arm passes the screen -> B1 and B7 killed at tier L.
- **Sanity checks:** A0 clean val NLL ~870 nats/window (R008) and Gen-PPL ~16.1 (R010). If A1 ends > 1.0
  Gen-PPL worse than A0, the fine-tuning recipe itself damages the model: lower the lr and re-run A1 before
  reading any other arm.
