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

## W1a — amendment to W1: step counts from measured throughput (registered 2026-10-23, before any W1 run)
- Wave 0 measured 39.2 useful TFLOPS for the tier-L trainer on an A100-40GB (2.23 s/step at 256 windows;
  `results/colab/w0_calib_diag/calib_L__calib.json`), 33-52% of the planned rate. Per the plan's rule (< 60%: cut
  steps, not seeds), every arm runs 45% of its steps: control / requant / soft 1800, onpol / b1 1350, onpol2 1080
  (~157 PFLOP per arm, ~6 CU on an A100). Val NLL every 300 steps, sample curves every 600. Everything else in W1
  (arms, recipe, schedule shape incl. 200 warmup steps, metrics, screen, claim rule, sanity checks) is unchanged.
  On GPUs with >= 40 GB the notebook uses micro-batches of 64 instead of 32 (same 256-window optimizer step).

## P1b — re-test of H3 with a fair within-level-AR probe (registered 2026-10-23)
- **Why:** P1's TC estimate was negative (R018), which a true TC cannot be; the AR probe's previous-code
  embeddings started at N(0, 1) and buried its own-position latent.
- **Setup:** identical to P1's `ts_ar_s` (tok_s, `configs/pilot/ncm_s.yaml`, seed 0, 6000 x 32) except
  `--ts_prev_init zero`; run `ts_ar0_s`. TC = NLL(existing ts_fact_s) - NLL(ts_ar0_s), per level and total.
- **Thresholds:** as P1 (support: TC >= 30% of the 1059-nat gap and stable or rising over the last 3 evals;
  falsified: TC < 10%).
- **Validity condition:** final TC >= -5 nats in total and no level below -2 nats. If it fails, the probe
  approach is closed: P1 stays inconclusive and no further re-runs are made.

## S1 — Tier-L Sudoku: on-policy corruption and board-feedback repair passes (registered 2026-10-23, before any run)
- **Setup:** the released Sudoku-hard generator (25.5M) and tokenizer (frozen). Training data: our generated hard
  puzzles (`assets/sudoku_train/hard.npz`, 98k, validity-preserving augmentation); val = the first 1000 of the
  held-out 2000 (model selection and operating points); test = the 2000 released hard puzzles, scored once at
  the end. Recipe for every arm: 5000 steps x 256 (or fewer at equal FLOPs), lr 5e-5 (200 warmup, cosine to 5e-6),
  AdamW (0.9, 0.99), weight decay 0.01, dropout 0.1, same data order per seed (`configs/scale/L_sdk_*.yaml`,
  `scripts/train_sudoku_L.py`).
- **Arms:** A0 released, no training; A1 control = plain teacher forcing (the released config records no
  corruption); A1u = A1 + NCM's uniform conditioning corruption (mean 0.5, linear 0.75 -> 0.25, clean labels);
  A2 = conditioning codes replaced by the model's own one-step greedy predictions at A1u's rates, clean labels
  (3750 steps); A3 repair = A1's loss on half of each batch, and on the other half one parallel pass from each
  state of a pool of the model's own in-progress Jacobi decodes. In that pass the input is the puzzle plus a
  zero-initialized embedding of the board decoded from the previous pass (lr x20 for this embedding), and the
  label is the solution's canonical codes. States advance by that pass's argmax and are replaced by fresh
  puzzles when solved or after 16 passes.
- **Decoding:** D1 sequential greedy (12 NFE, paper protocol). D2 greedy Jacobi passes that stop at the first
  board satisfying the rules and the clues (verifier), at a fixed point, or at the cap c in {4, 8, 12, 16, 24};
  board feedback for A3 only (for A0-A2, D2 is training-free Jacobi with verifier stopping). D2-nocheck: the same
  without the verifier.
- **Operating point:** per arm, c* = the largest cap whose mean NFE on val is <= 12. The best control is the
  max over {A0, A1, A1u} of the D2 solve rate at its own c* on val. Verifier stopping alone adds ~4 points (R020),
  so no arm gets credit for it.
- **H10 (B6/C5):** A3 beats the best control by >= +3 points on the test set under D2 at mean NFE <= 12. Kill:
  < +3 points (plan, bet 3).
- **H11 (B1a on Sudoku):** A2's D1 solve rate beats the better of A1 / A1u. Kill: < +1 point.
- **Screen (seed 0, val n = 1000, SE ~1 point):** A2 or A3 advances to seeds 1-2 if >= +2 points over the
  relevant control on val. A1 (and A1u if it is the best control) always gets seeds 1-2.
- **Claim (Phase 3):** 3 seeds; mean gain > 2x the pooled seed SD and >= the kill threshold, on the test set.
  Reported with D1, D2, D2-nocheck, mean NFE, FLOPs and solve rate by propagation depth.
- **Sanity checks:** A0 D1 test 86.15% and D2 ~90.3% at ~2.4 passes (R020). If A1 ends > 2 points below A0 on
  val D1, the fine-tuning recipe damages the model: lower the lr and re-run A1 before reading any arm.
- **Caveat:** verifier stopping uses task knowledge that text tasks lack; D2-nocheck is reported for every arm.

## S1a — amendment to S1: step counts (registered 2026-10-23, before any S1 run)
- 5000 x 256 is ~83 PFLOP per arm. At the ~25 TFLOPS that wave 0 measured for a 25M generator on an A100
  (`calib_M__calib.json`), that is ~5 CU per arm, not the planned 1.3. Every S1 arm therefore runs half its
  steps: A1 / A1u / A3 2500, A2 1875 (val curves every 500). Nothing else in S1 changes.

## J1 — WMT14 self-consistency fine-tune for early exact Jacobi convergence (criterion d, text) (registered 2026-10-23, before any run)
- **Setup:** the released WMT14 De-En NCM (120M generator, 11 levels, greedy) and tokenizer (frozen). Training
  pairs: 400k sampled uniformly (seed 0) from the HF `wmt/wmt14` de-en training set in the released format
  (`scripts/prep_wmt14.py`); val = newstest2013 (3000, selection); test = newstest2014 (3003), scored once at the
  end. Recipe for both arms: batch 128, lr 5e-5 (200 warmup, cosine to 5e-6), AdamW (0.9, 0.95), weight decay
  0.01, dropout 0.1, same data order per seed (`configs/scale/L_wmt_*.yaml`, `scripts/train_wmt14_L.py`).
- **Arms:** J0 released, no training (w0: 11.07 exact passes, BLEU 22.63 on test). J1 control = teacher forcing
  on reference translations, 3000 steps. J2 consist = J1's loss on half of each batch; on the other half, the
  current model's greedy Jacobi trajectory is run to its fixed point (no grad), 4 input states are sampled
  uniformly from those before the fixed point, and one parallel pass from each is trained to output the fixed
  point. 2000 steps (equal training FLOPs, trajectory passes counted).
- **Decoding:** exact Jacobi (stop when no code changes; output identical to the model's own sequential greedy
  output, verified per sentence), fp32. NFE = passes including the one that detects convergence; sequential = 11.
- **H12 (B2-trained):** J2 reaches mean exact passes <= 8.25 on test (>= 25% fewer than 11) with BLEU >= 21.63
  (J0 - 1). This meets the text half of criterion (d). Kill: J2 saves < 25% NFE at < 1 BLEU loss, or J2's
  passes are not below J1's by >= 1.
- **Screen (seed 0, val):** J2 advances to seeds 1-2 if val passes <= 9.0 and val BLEU >= J0 val BLEU - 1.
  J1 gets seeds 1-2 if J2 advances.
- **Claim:** 3 seeds; passes <= 8.25 and BLEU loss < 1 vs J0 on test in the seed mean, and the pass reduction vs J1
  > 2x the pooled seed SD. Reported with the pass histogram, FLOPs (each Jacobi pass processes every level) and
  wall-clock. Per the brief, a result believed to meet criterion (d) goes to the user before more compute is
  spent on it.
- **Sanity checks:** J0 re-evaluation reproduces 11.07 passes / 22.63 BLEU. If J1 loses > 1 BLEU vs J0, the
  recipe damages the model: lower the lr and re-run J1 before reading J2.
