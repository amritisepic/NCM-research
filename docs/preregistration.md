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

## W1b — amendment to W1/W1a after the external review and the code audit (registered 2026-10-23, before any W1 run)
Why: `docs/reviews/triage_2026-10-23.md` (G5-G7, codex F1-F8). Everything not listed here is unchanged from W1/W1a.
- **Code fixes before launch (R024):** autocast no longer leaks after on-policy steps (F2; it would have scored the
  on-policy arms in bf16 and the uniform arms in fp32); two-pass policy codes condition on the sampled level 0 (F3);
  the corruption RNG is checkpointed (F1); resume and `--eval_only` refuse a mismatched run directory (F5); sample
  evaluation no longer reseeds the training RNG (F6). The on-policy arms' random draw order changed.
- **Evaluation protocol:** final eval on **16 x 256** samples (seeds 42, 0-14) instead of 4 x 256; sampling (generator
  and tokenizer) and scoring in fp32 with TF32 off (F8). A0 is re-evaluated in `w1_L_ts_ctrl` with exactly this
  protocol (`L_ts_released16`); w0's 4-group A0 stays as the sanity reference. All W1 runs on bf16 GPUs (A100 or L4).
- **Screen (seed 0):** an arm advances to seeds 1-2 if its Gen-PPL is >= **0.35** below any of its registered
  comparators, entropy within 0.05 nats and MAUVE not lower by > 0.01. Comparators: A2 vs A1; A3 vs A1; A4 vs A1 or
  A3; A5 vs A1 or A3; A6 vs A1. From sampling noise alone (per-group SD 0.59, 16 groups per arm, unpaired) the SE of
  a difference is 0.21: an arm with no true effect passes 5% of the time, a true 0.5 passes 76%, a true 0.7 passes
  95%. Training-seed variance is not in these figures (unmeasured until seeds 1-2), so the seed-0 screen is a
  selection step, not a test. A1 always gets seeds 1-2.
- **No arm passes:** reported as "B1 / B7 not supported at tier L (a sampling-noise-only gain of >= 0.7 Gen-PPL would
  have passed with ~95% probability)". B1 and B7 are deprioritized for wave 2; they are not declared false.
- **Step-count confound (A3, A4: 1350 steps; A5: 1080):** A1's sample curve runs every 270 steps (256 samples, the same
  seed at every point), giving A1 at 1080 and 1350. If A1 at the matched step is >= 0.5 Gen-PPL better than A1 at 1800,
  fewer steps alone helps: the on-policy arm's win is then credited to H6/H8 only if it also beats A1 at its matched
  step by >= 0.35 on the same 256-sample protocol (curve-point evaluation of the final checkpoint).
- **Conditional controls (registered now, run only if triggered, at seeds 0-2 with the triggering arm's seeds 1-2):**
  A1m = uniform corruption whose per-level rates are scaled so that its decoded-token survival equals A3's (measured on
  CPU with `scripts/repair_capacity.py`-style decoding, per-level corruption rates now logged), 1800 steps; triggered if
  A3 advances. H6 is attributed to policy matching only if A3 also beats A1m by >= 0.35 at seeds 0-2; otherwise the
  gain is reported as "milder corruption". A6s = A6 with its soft-label mass spread over 16 random codes (same weights);
  triggered if A6 advances. H9 is attributed to near-equivalent codes only if A6 beats A6s by >= 0.35.
- **Definitions made precise:** H7 tests a consistent re-quantized hierarchy (labels and the unmasked finer-level
  inputs are both re-quantized), not labels alone. H8 tests proposals conditioned on a full previous policy hierarchy,
  then mixed with clean codes by the corruption mask. A6's soft-label gap is the per-level mean nearest /
  second-nearest distance gap, frozen at the start of training from 512 val windows (was: per micro-batch, F7).
- **Claim rule addition:** the entropy-matching temperature (when a claim needs one) is chosen on sample seeds
  100-103 and the claim is evaluated on the 16 screen seeds. The FLOP-matching unit stays the generator-FLOP proxy
  (F4: omits tokenizer and hierarchy work, the same attention pricing in every arm); measured wall-clock per arm is
  reported with it.

## P1b — withdrawn before launch (2026-10-23)
- Its premise ("with zero-initialized previous-code embeddings the AR probe equals the factorized probe at step 0") is
  false: the two probes' attention masks differ (AR code tokens attend to earlier code tokens of their level), so the
  functions differ at step 0. The estimator has also never been checked on data with known TC. Not run; notebook
  removed. Replacement (P1c, to be registered once built): (1) CPU validation of the TC estimator on synthetic codes
  with known TC (independent, duplicated, Markov); (2) an AR probe that keeps the factorized mask and adds the previous
  same-level codes through a zero-initialized side channel, so it equals the factorized probe exactly at step 0,
  warm-started from `ts_fact_s` and trained next to an equally long continuation of `ts_fact_s`.

## S1b — amendment to S1 (registered 2026-10-23, before any S1 run)
- **Feedback ablation:** the repair arm A3 is also decoded with D2 when its board feedback is zeroed (`check_fb0`) and
  when it comes from another puzzle's board (`check_fbshuf`), at every cap, on val and test.
- **H10 credit:** A3's gain over the best control counts as learned repair only if A3's D2 solve rate exceeds its own
  `check_fb0` rate by >= 2 points on val at c*. Otherwise the gain is reported as an effect of training on the pool
  (extra exposure to the model's own hard states), not of feedback.
- **Reported with every D2 result:** cap-hit count (no convergence guarantee with feedback), p90 and max NFE (mean NFE
  already counts unsolved puzzles at the cap), and per-puzzle wrong -> right / right -> wrong transitions against A0
  sequential (from the saved per-puzzle results).
- **Criterion (d) claims** additionally need batch-size-1 wall-clock at the operating point on 256 puzzles and a cost
  figure that includes attention (the w0 FLOP ratios are position counts).

## J1a — amendment to J1 (registered 2026-10-23, before any J1 run)
- **Capped decoding:** every evaluation (J0, J1, J2; val and test) also reports the output after at most c passes for
  c in {6, 8, 9}: BLEU, mean passes, length ratio, and % identical to the exact-stop output. Test hypotheses and
  per-sentence pass counts are saved.
- **H12 comparator:** besides <= 8.25 mean exact passes and BLEU >= J0 - 1, J2's test BLEU must exceed the BLEU of J0
  and of J1 decoded at the cap ceil(J2's mean exact passes) (equal or more compute) by more than the 95% paired
  bootstrap interval (1000 resamples), and J2's length ratio must be within 0.02 of J0's. If J0 at cap 8 is already
  within 1 BLEU of J0, that is reported: a fixed cap is not an adaptive rule and does not meet (d) by itself, but J2
  must then beat it as above.
- **Criterion (d) claims** additionally need batch-size-1 wall-clock at the operating point on 256 sentences and an
  attention-inclusive cost (w0: each Jacobi pass costs 0.30 of a full sequential decode by position count).

## M1 — Tier M, design C: matched from-scratch NCM and AR baselines for criterion (b) (registered 2026-10-23, before any run)
- **Why:** the 15k / 70k tier-M plan costs 116.6 CU for its baselines (R022, R027). Design and changes follow the
  external review (`docs/reviews/triage_tierM_2026-10-23.md`).
- **Setup:** released TinyStories tokenizer (36.4M, frozen; not counted). NCM-c: generator from scratch, d512 x 6
  (~18.9M body), paper recipe with uniform conditioning corruption, 6000 x 256 windows, level-0 prior estimated from
  262k training windows (`configs/scale/M_ncm_c.yaml`). AR-c: same body (~23.1M with embeddings), FLOP-matched by the
  generator proxy (27,335 x 256 windows; `configs/scale/M_ar_c.yaml`), snapshots at 6000 (data-matched, secondary) and
  20500 (75%, curve point). Both evaluated with the W1b protocol (16 x 256 samples, fp32 sampling and scoring; GPT-2
  large primary, TinyStories-33M, entropy, repetition, length, MAUVE always reported).
- **Gate (seed 0, before any other tier-M spend):** G0 = Gen-PPL(NCM-c) - Gen-PPL(AR-c). Seeds 1-2 of both baselines
  run only if G0 >= max(2.4, 4 x the pooled training-seed SD of W1's control A1 at seeds 0-2). Otherwise criterion (b)
  is reported as "not measurable at this scale and budget" and tier M stops.
- **Candidate (registered later, after wave 1):** one method, 3 seeds, FLOP-matched to NCM-c by the proxy (fewer
  updates for costlier objectives), <= 42 NFE for 128 tokens with decoder work disclosed.
- **Criterion (b) claim:** the 95% interval of (P_NCM + P_AR)/2 - P_candidate (seed means, 3 seeds each) lies above 0,
  the gain over NCM-c exceeds 2x the pooled training-seed SD, and the same holds with the gap of a conditional AR-c15
  (1.5x AR steps, seed 0, `configs/scale/M_ar_c15.yaml`) run before the claim. Stated as conditional on the released
  tokenizer, at this body and budget, with proxy FLOPs and measured wall-clock.

## S1c — amendment to S1 after the second code audit (registered 2026-10-23, before any S1 run)
Why: `docs/reviews/codex_audit2.md` (A2-1, A2-2, A2-3, A2-6, A2-9); triage `docs/reviews/triage_audit2_2026-10-23.md`.
- **Data order:** every arm draws the same 256 rows per step from the training RNG; the repair arm trains its
  teacher-forcing half on the first 128 of them. Repair-pool refills come from a separate, checkpointed RNG, so the
  pool's puzzles are an exposure that the other arms do not have; this is recorded, not claimed equal.
- **Exact resume:** model, optimizer, data RNG, corruption RNG, repair pool and pool RNG are saved in one atomic
  checkpoint; a resume without them fails. A run manifest (config, training-data SHA-256, base config hash and
  checkpoint sizes, device) must match to resume; `--eval_only` refuses a directory with a checkpoint.
- **Reporting:** every cap also reports p90 / max NFE and the cap-hit count; per-puzzle stop / NFE / correctness are
  saved for every decoding mode; the operating cap c* is computed on val and stored with the run.
- **Precision:** evaluation in fp32 with TF32 off. On-policy proposals (A2) are computed under the training autocast
  (bf16 on A100/L4), i.e. from the bf16 policy, while inference decodes in fp32; recorded as part of A2's definition.

## J1b — amendment to J1/J1a after the second code audit (registered 2026-10-23, before any J1 run)
Why: `docs/reviews/codex_audit2.md` (A2-3 to A2-8, A2-10).
- **Teacher:** J2's trajectories are run with the fp32 greedy map (autocast off, TF32 off), the map that exact
  decoding is evaluated with, and always for K + 1 = 12 passes (stopped sources hold their codes), so the executed
  generator proxy equals the registered one (control 3000 x 384 = J2 2000 x 576 forward rows).
- **Degenerate sources:** a source whose all-zero start is already its fixed point has no pre-fixed-point state; it is
  dropped from that step's consistency loss and counted (`traj_degenerate`). If more than 1% of trajectory sources
  over the run are degenerate, this is reported and H12 is not read until a policy for them is registered.
- **Data order:** both arms draw one common batch of 128 per step; J2 uses rows 1-64 for teacher forcing and every
  4th of rows 65-128 as trajectory sources.
- **Comparators:** every evaluation keeps sentence-ordered exact hypotheses and capped hypotheses for every cap 1-12
  (val and test), so the J1a paired bootstrap at ceil(J2's mean passes) is always possible. Exact-vs-sequential
  disagreements are recorded per sentence; any disagreement is reported, and "exact" applies only where they agree.
- **Data:** `prep_wmt14.py` samples uniformly from the length-eligible pairs until exactly 400k are found, and reuses an
  archive only if its saved selection metadata matches. Run manifest and `--eval_only` guard as in S1c.

## W1d — eval-only diagnostic of the re-quantized-label arm (registered 2026-10-24, before the run; exploratory)
- **Why:** A2 (uniform corruption + re-quantized labels) is the only wave-1 arm better than the control on both registered
  metrics at equal entropy (Gen-PPL -0.31 ± 0.08 paired by sample seed, sampling noise only; MAUVE +0.034), but it missed
  W1b's screen (0.31 < 0.35) and is +0.13 ± 0.08 Gen-PPL worse than the released model (R030, R031). Before any
  confirmation run, establish whether the gain holds across sampling temperatures and where in the hierarchy it arises.
- **Setup:** no training. A0 (released), A1 (`L_ts_ctrl_s0/final.pt`), A2 (`L_ts_requant_s0/final.pt`): the seed-0 wave-1
  checkpoints. `scripts/diag_w1_frontier.py`, notebook `w1d_requant_diag`, A100, fp32 sampling and scoring, TF32 off.
  (1) Frontier: T in {0.8, 0.9, 1.0}, 8 groups x 256 unconditional samples per T (seeds 42, 0-6; W1b's sampler): Gen-PPL
  (GPT-2 large, TinyStories-33M), entropy, repetition, MAUVE (2048 samples vs 2048 val windows, 5 quantization seeds).
  (2) Oracle-intervention curve (R020's protocol): the canonical codes of levels 0..j of the first 1024 val windows are
  supplied and the model samples the rest at T = 1, for j in {-1, 3, 5, 7}; Gen-PPL (GPT-2 large) in 4 groups of 256.
  Windows and seeds are the same for every model, so model differences are paired.
- **Integrity check:** at T = 1, every group must reproduce the wave-1 result's Gen-PPL for the same seed (max |diff| <=
  0.01 on an A100). If it does not, the run is reported as not reproducing wave 1 and its numbers are not read.
- **Reading** (paired differences A2 - A1, mean over groups ± SE; exploratory, no claim):
  - Frontier: "A2 ahead" if A2's Gen-PPL point estimate is below A1's at every T, the mean of the three paired differences
    is at least 2 SE below 0, A2's entropy is within 0.05 nats of A1's at each T, and A2's MAUVE is not lower by > 0.01
    at any T. "Trade-off" if the Gen-PPL gain comes with lower entropy or reverses at some T. Otherwise "no difference".
  - Oracle curve: if A2 - A1 shrinks by at least half from j = -1 to j = 5, the gain arises while the model samples its
    own coarse levels; if at least half of it remains at j = 5 and j = 7, it is a fine-level effect, bounded by those
    levels' ~7% share of the oracle quality gap (R020).
- **Consequence:** only "A2 ahead" leads to registering a confirmation (W1c: A1 and A2 at fresh seeds, Phase-3 rule),
  whose A1 runs are the held A1 seeds (R033). Otherwise A2 is reported as a likelihood / MAUVE shift without a Gen-PPL
  gain and is not pursued. The oracle reading decides which magnification route (coarse-level repair vs fine-level
  rendering) any later W2 arm targets.
- **Cost:** ~2.5 CU (A100).

## J1c — amendment to J1 / J1a / J1b after the fourth code audit (registered 2026-10-24, before any J1 run)
Why: `docs/reviews/codex_audit4.md` (A4-1 to A4-5); triage `docs/reviews/triage_audit4_2026-10-24.md`.
- **Degenerate consistency sources** (J1b): still dropped from the loss and counted. Their student rows now run as
  zero-loss padding, so the executed generator proxy equals the registered one (J2: 2000 x 576 forward rows) in every
  step, including a step whose sources are all degenerate. Run-level source and degenerate totals are checkpointed,
  restored on resume and reported (`degenerate_pct`); J1b's 1% rule reads those totals.
- **NFE:** with WMT's static source prefix, convergence may be detected on pass 1, so a sentence whose all-zero start is
  already its fixed point costs 1 pass, not 2. This applies to J0, J1 and J2 alike. J0's sanity reference (11.07 passes,
  R020) was measured with pass-1 detection off, so J0's re-evaluation may differ by at most 1 pass times the share of such
  sentences; the difference is reported, not treated as a failed sanity check.
- **Evaluation records:** validation NLL is scored in fp32; every curve evaluation keeps ordered exact and capped
  hypotheses and per-sentence passes, and checks exact = sequential.
- **Run identity:** the run manifest identifies the base tokenizer and generator by SHA-256, not by file size.


## J2-F — frozen-teacher consistency arm, contingent on J2 missing its screen (registered 2026-10-24, before any J2-F run or code)
- **Timing, stated plainly:** the user decided to register J2-F as a contingent arm (run only if J2 misses the J1
  screen). This text was written after J2's seed-0 result had arrived and been analysed (R036: J2 missed the screen;
  about 9% of val and 16% of test sentences decode to one of two input-independent outputs). J2's miss triggers J2-F.
  The design is the one proposed before J2's result (`docs/literature_advice_2026-10-24.md`, "frozen versus
  moving-teacher Jacobi distillation"); nothing in it, and none of its criteria, was changed in response to J2's result.
  The question is narrower than J2's: does a fixed teacher remove the failure J2 showed?
- **Arm:** identical to J2 (J1 + J1a + J1b + J1c: data, common batch draw per step, rows 1-64 teacher forcing on
  references, every 4th of rows 65-128 as trajectory sources, 4 states per trajectory sampled uniformly before the fixed
  point, one parallel student pass per state, 2000 steps, recipe, seed handling, degenerate-source rule with zero-loss
  padding) except for one thing: the trajectories and fixed points come from a **frozen copy of the released
  generator** (J0), not from the current model. The teacher runs in eval mode, no grad, fp32 with TF32 off, always
  K + 1 = 12 passes (`full_budget`, `initial_fixed_ok` as J1c). Its targets are therefore J0's own greedy outputs, and
  its input states are J0's intermediate Jacobi states. The student is the current model, initialised from the released
  generator as in J2. The generator FLOP proxy is the same as J2's (2000 x 576 forward rows; the teacher has the same
  architecture), and the extra fp32 copy changes memory, not the proxy. Config `configs/scale/L_wmt_consist_frozen.yaml`
  (arm name `consist_frozen`), notebook `w2c_L_wmt_consist_frozen`; J0 and J1 are not re-run (their seed-0 results are
  `results/colab/w2c_L_wmt_ctrl/`).
- **Readings, unchanged from J1 / J1a / J1b / J1c:** the J1 screen (val passes <= 9.0 and val BLEU >= J0 val - 1 =
  22.55), H12 on test (<= 8.25 mean exact passes, BLEU >= J0 - 1, passes below J1's by >= 1), the J1a comparators at
  ceil(J2-F's mean passes) with the paired bootstrap and the 0.02 length-ratio rule, J1b's 1% degenerate rule, exact =
  sequential checks, and the 3-seed claim rule. If J2-F advances, J2-F and J1 get seeds 1-2 (J1's seeds 1-2 were
  contingent on J2 advancing; they now follow J2-F). Per the brief, a result believed to meet criterion (d) goes to the
  user before more compute is spent on it.
- **Diagnostic (reported for J0, J1, J2 and J2-F; not a criterion):** share of sentences whose output string is shared
  by >= 5 sentences (`scripts/analyze_j1.py`, "shared outputs"), mean passes on the remaining sentences, and the length
  ratio on them.
- **Prediction and what overturns it:** the moving teacher caused J2's collapse onto input-independent outputs, so J2-F
  has a shared-output share <= 1% on test (J0 and J1: 0%; J2: 16%). Overturned if J2-F's share is > 5%: then the
  collapse does not need a moving teacher. Secondary prediction (no claim attached): J2-F saves fewer passes than J2,
  because the student is trained on J0's states and decodes through its own. If J2-F's test passes are >= 10.0, the
  frozen-teacher route is reported as not reducing passes; if it passes the screen, it is read only through H12 and J1a.
- **Code gate:** a trainer option for the frozen teacher (the only change), with tests that (1) the teacher's fixed
  points equal J0's exact-decode outputs on a fixed batch, (2) the teacher's weights never change and receive no grad,
  (3) the executed proxy equals J2's, (4) the arm is in the run manifest and a run directory refuses a different arm.
  Review by PR (Codex) and a deep audit before launch (`CLAUDE.md`, tier 2).
- **Cost:** ~6.5 CU seed 0 (J2 took 70.6 A100-minutes); seeds 1-2 of J2-F and J1 ~22 CU if it advances (J1 took 51.5 A100-minutes).

## W2M — tier-L mechanism arms for the on-policy collapse (registered 2026-10-25, before any code or run; exploratory, criterion c candidate)
- **Why:** wave 1's on-policy arms collapsed (A3 / A4 / A5 Gen-PPL 41.4 / 36.8 / 32.2 vs A1 16.6; R030, R031). Leading
  account (b'): replacements drawn from the model's own distribution are plausible, hence undetectable, so the model
  learns to discount its own coarse codes. Competitors: (i) lost denoising coverage at A3's lower realized corruption
  rate, (ii) used-parent mismatch (proposals sampled under the canonical L0 while ~75% of windows then carry a unigram
  L0; R034), (iii) moving-policy feedback, (iv) the bf16 policy forward that only the on-policy arms had (R036). Design
  reconciled from `docs/reviews/codex_adversarial_onpolicy.md` and `docs/literature_advice_2026-10-24.md`; draft
  `docs/mechanism_arms_draft_2026-10-25.md`. Arm set and the flag input chosen by the user (all 8 arms, ~27 CU).
- **Common protocol:** released TinyStories NCM (generator and tokenizer, tokenizer frozen), `scripts/train_gen.py` as of
  PR 5 (fp32 policy forward with TF32 off; policy RNG separate from the training RNG), seed 0, W1's recipe (batch 256,
  lr 5e-5, 200 warmup, cosine to 5e-6, AdamW 0.9 / 0.95, weight decay 0.1, dropout 0.1, same data order) for **600
  steps in every arm** (equal updates; proposal forwards are reported separately, not matched). Canonical (clean)
  labels in every arm. Corruption rates as W1's A1 (mean 0.5, linear 0.75 -> 0.25 over levels 0..K-2, finest level
  never corrupted) unless stated. **25% of the rows of every batch are left clean** (chosen per row from the mask
  stream) in every arm, so that all-zero flags are trained. **Masks come from one stream** (their own generator,
  separate from replacement draws), so every arm with W1's rates corrupts exactly the same slots. Proposals and
  replacements draw from their own generator.
- **Arms** (`configs/scale/L_ts_m{0..7}_*.yaml`):
  - **M0 uniform:** NCM's uniform random replacement at the masked slots (control).
  - **M1 current policy:** A3's policy (the current model's one-step samples at T = 1 given the clean coarser levels,
    level 0 from the unigram) at the masked slots, fp32 policy forward.
  - **M2 frozen plausible-wrong:** at every masked slot a code sampled at T = 1 from a frozen copy of the released
    generator's one-step distribution given the clean coarser levels, **with the canonical code excluded** (level 0:
    the unigram with the canonical code excluded). Every masked slot is therefore wrong.
  - **M3 uniform-wrong:** every masked slot replaced by a uniformly random code different from the canonical one.
  - **M4 M2 + truthful flags:** as M2, plus a per-position flag on the generator's inputs: for the level-k input at
    position q, the share of the full-length span of q whose cumulative reconstruction contains a replaced ancestor
    (the mask propagated with the tokenizer's own up-sampling and the area down-sampling that builds the inputs).
    Flags signal the sampled mask, never `used != canonical`. The flag enters as `f * w` added to the code input
    vector, `w` a learned vector initialised to zero (step 0 equals the released model). Inference and every
    evaluation use all-zero flags.
  - **M5 M2 + shuffled flags:** as M4, but each row's flags are permuted within each level by an independent RNG (same
    count and level distribution, no positional information).
  - **M6 used-parent:** as M1, but the policy's proposals for levels >= 1 are sampled given the **level-0 codes actually
    used** (canonical or unigram per the mask) and the clean levels 1.., not given the canonical level 0.
  - **M7 uniform at A3's realized rates:** uniform replacement with per-level probability d_k / (1 - 1/V_k) on the
    corrupted rows, d_k = A3's mean logged per-level disagreement over its logged steps <= 600
    (`results/colab/w1_L_ts_onpol/L_ts_onpol_s0__metrics.jsonl`: 0.735, 0.691, 0.601, 0.505, 0.398, 0.304, 0.239, 0.103,
    0.171, 0.167, 0.145, 0), V_k the level's codebook size. Its masks differ from the others' by construction.
- **Evaluation:** curve at steps 100 / 300 / 600 (256 samples, val NLL); final at 600 with W1b's protocol (16 x 256 fp32
  samples, seeds 42 and 0-14; Gen-PPL under GPT-2 large (primary) and TinyStories-33M, entropy, MAUVE in fp32, bigram
  gain and adjacent duplicates per 1k as R031, clean val NLL per level in fp32). Every arm logs its realized per-level
  disagreement. Eval-only mechanism measures on the eight final checkpoints and the released model, same windows and
  seeds for every model (`scripts/mech_eval_w2m.py`): (a) coarse-context dependence: on 512 clean val windows, one
  code at level j (j = 1..6) changed to a uniformly random different code at a fixed random position, finer inputs
  rebuilt; mean KL(clean || perturbed) of the predictive distributions of levels > j, per j; (b) replacement
  detectability: a logistic probe on the generator's last hidden states separating contaminated from clean input
  positions, inputs corrupted plausible-wrong (as M2) and uniform-wrong (as M3) at W1's rates with zero flags, trained on
  384 windows, AUC on 128 held-out windows; (c) flag use (M4, M5): clean-label NLL per window on corrupted held-out
  windows with the true flags minus with zero flags.
- **Readings** (seed 0; D(X) = Gen-PPL(X) - Gen-PPL(M0), GPT-2 large, paired over the 16 sample seeds, ± SE):
  - Sanity: M0 within 1.0 of A0 (16.16). **Validity:** the mechanism readings below are made only if D(M1) >= 5.0; if
    not, it is reported that the collapse did not reproduce under this protocol (fp32 policy, 600 steps, clean rows)
    and the arms are read descriptively only.
  - Let D1 = D(M1). "Damaging" = D >= 0.5 D1; "healthy" = D <= 0.2 D1.
  - (b') predicts: M2 damaging, M3 healthy, M4 rescues (R4 = 1 - D(M4)/D(M2) >= 0.8, with flag use (c) <= -1 nat per
    window), M5 damaging (R5 = 1 - D(M5)/D(M2) < 0.5), M6 damaging, M7 healthy.
  - Overturned / weakened: M2 healthy while M1 damaging -> feedback (iii) or used-parent (ii) favoured over
    plausibility; M3 damaging -> plausibility not needed; R4 < 0.2 with flag use demonstrated -> detectability is not a
    sufficient remedy; R5 >= 0.8 -> a flag rescue is regime separation, not detection; M6 healthy -> the used-parent
    mismatch is sufficient; M7 damaging -> lost coverage is sufficient. Exploratory, not a criterion: probe AUC >= 0.90
    on plausible-wrong contexts would contradict "undetectable".
  - Anything else is reported as mixed. One seed: no reading establishes a mechanism; criterion (c) needs the deciding
    arms at seeds 1-2 under a new registration, which goes to the user first.
- **Code gate:** implementation on a `claude/wip-` branch with tests (mask stream identical across arms, 25% clean rows,
  replacements always wrong in M2 / M3 / M4 / M5, flags equal the propagated mask and are zero at inference, zero-init
  flag vector gives the released model's logits, shuffled flags preserve per-level counts, M6 conditions on the used
  L0, M7's rates, resume equals an uninterrupted run), Codex review and a deep audit before launch.
- **Cost:** ~27 CU (M0 / M3 / M7 ~2.9 each, M1 / M2 / M4 / M5 / M6 ~3.7 each, mechanism eval ~1); each run < 15 CU.

## W2Ma — amendment to W2M's readings after a pre-reading check of the flags and the probe (registered 2026-10-27, before any W2M result was in the repository)
- **Why:** R043. At W1's rates the truthful flags saturate: on corrupted rows 99.6% of flag slots are 1 (level 1 is
  constant within a row; >= 96.9% from level 2 on), and within-level shuffling changes 0.0014 per slot. Probe (b)'s
  labels are 99.7% positive, its negatives sit 91% at level 1, and a score that only knows the level reaches AUC 0.95.
- **M4 / M5:** at these rates M5 is not a positional control but a near-replicate of M4: both carry a row-level
  corruption indicator (and, at level 1, whether L0 was replaced). The (b') prediction "M5 damaging (R5 < 0.5)" is
  withdrawn. M4 rescuing M2 (R4 >= 0.8 with flag use (c) <= -1 nat per window) is read as "row-level provenance
  prevents the damage": evidence for hidden provenance at the row level, not for positional detection. R5 >= 0.8 is
  then expected and does not weaken (b'). |D(M4) - D(M5)| is reported as a lower bound on run-to-run variation (same
  seed, data and masks). If M5 has not run, the user may skip it.
- **Probe (b):** the pooled AUC and its L0-clean-windows variant are not read. Replacement, if the user approves the
  code change: a row-level detectability probe per level k = 1..6 (50% clean windows mixed in; label = window
  corrupted; logistic probe on level-k hidden states; held-out AUC), for plausible-wrong contexts with uniform-wrong
  contexts as the positive control. If the uniform-wrong AUC is < 0.75 at every level, the probe is reported as too
  weak to read. Positional detectability is not measurable at W1's rates (beyond level 2, corrupted windows have
  almost no clean positions). The fix to `scripts/mech_eval_w2m.py` goes through a PR, Codex review and a delta deep
  audit before `w2m_mech_eval` runs; if the eval runs first, (b) is reported as uninformative.
- **Validity gate:** if D(M1) < 5.0, the registered descriptive-only reading stands and no mechanism arm is
  replicated. A follow-up that separates W2M's three protocol changes (fp32 policy, 25% clean rows, 600 steps) is not
  pre-approved; it goes to the user with its cost.
- Unchanged: every arm, metric, threshold and reading not named here.

## S0 — within-level left-neighbour dependence of the released TinyStories generator; gate for the coupled sampler (registered 2026-10-27, before the probe was built or run)
- **Why:** `docs/plan_2026-10-27.md` section 4 (S0). Astra's direction 1 (a marginal-preserving coupled sampler) is funded
  only if neighbouring codes of one level carry dependence that NCM's factorized head misses. The head predicts every
  level-k code independently given the coarser levels: under block-diagonal attention no level-k position sees another
  level-k code.
- **Setup (laptop, no Colab):** the released TinyStories generator and tokenizer, frozen, eval mode, fp32 with TF32 off.
  4096 packed training windows (random story openings of the training stream, `TrainStream.batch`, numpy seed 0); the
  last 512 of them are a development split used only for early stopping. Reported on the first 1024 even-index val
  windows (`examples("val", 1024)`). Canonical codes from the tokenizer in fp32; teacher-forced hidden states after the
  final layer norm at every level-k position, k = 1..6.
- **Term (one per level k = 1..6, trained separately):** at position p >= 1 of level k, with h the frozen hidden state
  and n the frozen level-k codebook vector of the canonical left neighbour c*_{k,p-1}, the head's query z = proj(h)
  becomes z + U[(A h) ⊙ (B n)] + U0 (B0 n) (⊙ elementwise, rank r = 64); logits keep the head's own distance form
  and temperature. U and U0 start at zero, so step 0 is the released model exactly; A, B, B0 start random (torch seed
  0). Position 0 has no neighbour and is unchanged. AdamW (lr 1e-3, weight decay 0.01), batches of 64 windows, at most
  30 epochs over the 3584 training windows, stopping after 3 epochs without a lower development NLL; the best
  development epoch is kept. Nothing is tuned on the val windows.
- **Reported:** per level k = 1..6, the mean over val windows of the level's canonical NLL under the released head
  (nats per window) and the gain (released minus with the term; positive = the left neighbour helps), ± SE over
  windows, also per code.
- **Gate (plan section 4):** a coupled-sampler pilot (<= 5 CU) is registered only if (1) the gain summed over levels 1-5
  is >= 10 nats per window AND >= 5% of those levels' summed canonical NLL, and (2) E2's adjacent-pair interaction is
  positive for the released model: its mean over windows, summed over levels 3-6 and scored by GPT-2 large, has a 95%
  interval (± 1.96 SE, paired over windows) above 0. If (1) fails, direction 1 is dropped and reported without
  waiting for E2. Level 6 is reported, not gated.
- **Reading:** the gain is a lower bound on the within-level dependence the factorized head misses (one left
  neighbour, a rank-64 term, 3584 training windows).
- **Cost:** laptop, ~1-2 h; no Colab.

## D1 — batch-1 wall-clock gate for criterion (d) (registered 2026-10-27, before any D1 timing run)
- **Why:** J1a requires batch-1 wall-clock and an attention-inclusive cost for any (d) claim; the user kept that rule
  (R044). R042's val-only lead (J2-F + "stable x1": 7.80 passes, 29.1% below 11, BLEU -0.54 vs J0) is only a latency
  claim: each Jacobi pass processes every level. Plan `docs/plan_2026-10-27.md` section 4 (D).
- **Setup:** `scripts/bench_batch1.py`, notebook `d1_batch1_wallclock`; A100-40GB (the gate), L4 optional; eager PyTorch,
  no compilation or CUDA graphs (dispatch overhead is measured, identically for every decoder). Items: the first 256
  newstest2014 test sentences (sources tokenised and padded to 96 as the evaluation does) and the first 256 released
  Sudoku-hard puzzles. Batch 1; each item's latency runs from its token prefix on the GPU to its output text (WMT14,
  detokenised) or board (Sudoku), CUDA-synchronised before and after. Warm-up: every configuration on the first 16
  items, untimed. Then per item every configuration 3 times in a seeded random interleaved order; item latency = median
  of its 3; reported: mean (the gate), SE, median, p10, p90 over the 256 items. fp32 with TF32 off (primary: the
  precision of every quality number); bf16 autocast (secondary, never gated). No quality metric is computed on test.
- **Decoders:** sequential greedy (WMT14: level 0 from the source prefix, then levels 1-10, 11 calls; Sudoku: level-0
  queries, then levels 1-11, 12 calls); Jacobi exact (passes until no code changes, at most K + 1; WMT14 detects a
  fixed point from pass 1, J1c); Jacobi + stable x1 (exact, or stop at the first pass t >= 2 whose detokenised text
  equals pass t-1's; output that text; the per-pass tokenizer decode and detokenisation are timed); Sudoku Jacobi +
  verifier (exact, or the first pass whose decoded board satisfies the rules and the clues; R020). Each in two
  implementations: `eager`, the upstream code, which re-runs the source / puzzle prefix in every forward call (upstream
  `generate` re-encodes the 96-token source in each of its 11 calls: 1368 generator rows vs 408 for one Jacobi pass);
  and `cached` (`src/ncmr/cached_decode.py`), which runs the prefix once per item with its pad mask and every later
  call on the target rows only. The sequential side of every comparison is the faster implementation of J0's
  sequential decoder; the Jacobi side the faster implementation of its configuration.
- **Configurations:** WMT14: J0 sequential, J0 Jacobi exact, J0 + stable x1, J2-F + stable x1 (J2-F's seed-0
  `final.pt`). Sudoku: sequential, Jacobi exact, Jacobi + verifier.
- **Integrity (fp32; the gate is not read if any fails; at most 2 of 256 items may differ, for float-order near-ties
  between batch-128 and batch-1 decoding):** J0 Jacobi exact reproduces the first 256 saved test hypotheses and pass
  counts (`results/colab/w2c_L_wmt_ctrl/L_wmt_released__hyp_test.json`) and the sequential output; both
  implementations of every configuration give the same outputs and passes; J0 + stable x1 and J2-F + stable x1 equal
  the rule applied to their saved capped test hypotheses; the rule applied to the four arms' saved val outputs
  reproduces `results/j2f_stop_rule_val_s0.json` (passes exactly, BLEU exactly when the newstest2013 text is present);
  Sudoku: cached = eager sequential, Jacobi exact = sequential; repeats of an item give identical outputs.
- **Cost reported per configuration:** mean passes and pass histogram; generator rows per item; executed FLOPs per item
  (FlopCounterMode with math attention: matmuls, attention over every key of the dense masks, convolutions, heads,
  tokenizer decodes), and the generator blocks alone in `src/ncmr/flops.py`'s convention (24 d^2 + 4 d x keys per row
  and layer), both from per-call counts and the measured passes.
- **Gate (fp32, A100):** the text half of (d) stays alive only if mean latency(J2-F + stable x1) <= 0.75 x mean
  latency(J0 sequential), both at their faster implementation; the rule's val BLEU condition is already met (R042:
  -0.54). The paired bootstrap 95% interval of the ratio (1000 resamples of items) is reported, not gated. The Sudoku
  half is alive only if Jacobi + verifier <= 0.75 x sequential. Also reported: J0 + stable x1 and J0 Jacobi exact
  against J0 sequential (training-free).
- **Consequences:** text gate passes -> J1d (below) runs, then the user is asked before any claim (brief, criterion d).
  Fails -> (d) is reported as unreachable on text for this model family under the wall-clock rule, with the cost model
  and the Sudoku timing; no more Jacobi training.
- **Cost:** ~1.5 CU.

## J1d — seeds for J2-F + stable x1, contingent on D1's text gate (registered 2026-10-27, before D1 and before any seed)
- **Runs only if D1's text gate passes.** Arms: J2-F and J1 at seeds 1 and 2, exactly as registered (J1, J1a, J1b, J1c,
  J2-F; `configs/scale/L_wmt_consist_frozen.yaml`, `L_wmt_ctrl.yaml`), ~23 CU. Seed 0 is the existing run.
- **Rule, fixed now:** stable x1 as in D1 (cap K + 1 = 12), applied to each run's saved capped test hypotheses. Test
  (newstest2014) is scored once per seed, after all seeds have run.
- **Comparators per seed:** J0 + stable x1 and the same seed's J1 + stable x1; J0 and that J1 capped at ceil(J2-F +
  rule's mean passes), with J1a's paired bootstrap (1000 resamples) and the 0.02 length-ratio rule.
- **Claim (text half of d):** over seeds 0-2, the seed mean of J2-F + stable x1's passes <= 8.25 and its BLEU within 1
  of J0 exact; at every seed its BLEU exceeds both capped comparators by more than the 95% paired bootstrap interval;
  and each seed's checkpoint passes D1's latency gate (<= 0.75 x J0 sequential, fp32, A100) under D1's protocol on the
  same 256 sentences. A result believed to meet (d) goes to the user before more compute (brief).

## E1 — how much of NCM's code NLL is representation choice (registered 2026-10-27, before any E1 run; criterion (c) candidate C1)
- **Why:** plan section 4 (C1). Clean code NLL and sample quality have come apart in two domains (TinyStories A1 / A2 /
  A4, R030-R031; WMT14 J1 / J2-F, R042). If most fine-level code NLL is spent choosing among codes that decode to the
  same text, code NLL is the wrong selection metric and the dissociation is explained.
- **Quantities** (item x, canonical codes c*, reconstruction x^ = D(c*), cut k): N_k = -log Q(c*_{>=k} | c*_{<k})
  (teacher-forced); A_k = -log Q(c*_{<k}) (TinyStories level 0: the released unigram); rho_k = P(D(c*_{<k}, C_{>=k}) =
  x^), C_{>=k} ancestral samples from the model at T = 1 (level k given c*_{<k}, each finer level given the coarser
  sampled ones), estimated by Monte Carlo (rounds of 64, at most 2048 samples, stop once 20 hits are reached; Gumbel
  noise seeded per item and cut, the same for every model); T_k = -log rho_k; R_k = N_k - T_k (a lower bound on the nats
  spent choosing among codes with the same text); U_k = A_k + T_k (an upper bound on the text NLL). A hit is exact
  equality: TinyStories all 128 tokens; WMT14 the tokens up to the first EOS, pads dropped (the text the evaluation
  reads); Sudoku the board. fp32, TF32 off. `scripts/e1_equivalence.py`, notebook `e1_equivalence`.
- **Items and models:** TinyStories, the first 64 even-index val windows: A0 (released), A1, A2, A4 (wave-1 seed-0
  `final.pt`). WMT14, the first 128 newstest2013 (val) pairs as `prep_wmt14.py` stores them (targets <= 63 tokens), target
  given source: J0 (released), J1, J2-F (seed 0).
  Sudoku-hard, the first 128 released hard puzzles: the released model.
- **Cuts:** k = 8..K-1 on every item (registered). k = 5..7 on the first 8 items only, descriptive: a laptop
  prototype on 3 released TinyStories windows (R046) found 0-8 hits in 2048 samples at k = 5-7 (~95% of the
  cost) and 20-30 hits within 32-256 samples at k = 8.
- **Summaries per model and cut:** hit share; means ± SE over items of N_k, A_k and, when every item has a hit, T_k,
  R_k, U_k and R share = mean R / mean N. With zero-hit items: bounds R_lo (their R = 0) and R_hi (their T = log(n/3),
  the rule-of-three bound), and the median T if more than half hit. Paired over items, model minus base (A1; J0): dN_k
  and, when both have hits on every item, dT_k and |dT| / |dN|. The stopping rule biases each T_k by about 1/20 nat
  (inverse sampling), negligible against N_k.
- **Predictions:**
  1. Released TinyStories: R_8 >= 0.8 N_8 (read on R_lo if any item lacks a hit). Overturned if R_8 < 0.5 N_8 (read on
     R_hi). Criterion (c) replication (R044: domains and checkpoints): the same inequality in the released WMT14 and
     Sudoku-hard models; a model whose mean N_8 < 5 nats per item is reported, not read.
  2. A2 - A1, A4 - A1 (TinyStories) and J1 - J0 (WMT14): |dT_8| <= 0.2 |dN_8| (the NLL differences are representation
     choice). Seed-0 checkpoints: supporting evidence, not part of a (c) claim without seeds (R044). Overturned if dT_8
     carries >= 50% of dN_8 in two of the three comparisons.
  3. In each domain the models' rank order by mean U_8 (lower = better) equals their order by the registered quality
     metric while the order by mean N_8 does not. TinyStories: Gen-PPL (GPT-2 large, W1b protocol, seed 0) A0 16.16,
     A2 16.29, A1 16.60, A4 36.8; WMT14: val BLEU J0 23.55, J2-F 22.90, J1 22.82. T_8's order is reported beside it.
- **Integrity:** the sampler's level calls equal the teacher-forced logits (tests); each item reports whether x^ = x.
- **Cost:** ~2-3 CU (one notebook, A100 or L4). E2 is not run (R045: S0 failed, so E2 decides nothing).

## E1a — amendment to E1's analysis after the round-2 review (registered 2026-10-27, before any E1 run; analysis and readings only)
- **Why:** `docs/research_review_2026-10-27.md` section 2: -log of a stopped, capped hit rate has no finite expectation
  and an unknown bias; the rule-of-three endpoint is not a valid input to an aggregate claim; prediction 1's 0.8 is
  undemanding at hundreds of nats; R046's three pilot windows are already observed; there is no theorem that U must rank
  models like their quality metrics. E1's sampling, items, cuts and saved fields (`level_nll`, `mc[k] = [hits, samples]`
  per item) are unchanged; everything below is computed afterwards from the saved results.
- **Primary uncertainty:** per item and cut, the anytime-valid beta-mixture confidence sequence for rho (Bernoulli,
  alpha = beta = 1/2; valid at E1's batchwise stopping), intersected with [exp(-N_k), 1] (c* itself decodes to x^), at
  delta = 0.05 / (items x cuts read) per model (Bonferroni). Reported: T_k in [-log p_U, -log p_L] and R_k in
  [N_k + log p_L, N_k + log p_U]; mean bounds = means of the per-item endpoints (equal item weights). The registered
  point summaries are reported beside them; the "about 1/20 nat" bias statement is withdrawn.
- **Prediction 1 reading:** on the confidence bounds (lower bound of mean R_8 / mean N_8 for "holds", upper bound for
  "overturned"). Reported twice: all 64 registered windows, and the 61 untouched by the R046 pilot (the first three val
  windows excluded). Added (untouched windows): the lower bound of mean R_8 / mean N_8 exceeds 0.90 and at least 80% of
  windows have a T_8 interval no wider than 2 nats; overturned if the upper bound is below 0.80; inconclusive if more
  than 20% of windows are censored (no hit). Same rules for the released WMT14 and Sudoku-hard models.
- **Prediction 2:** "holds" only if the whole interval of dT_8 lies within +/-0.2 |dN_8| (endpoints propagated);
  absolute dT_8 also reported.
- **Prediction 3** (rank order by U_8 vs quality): reported descriptively, no longer read as a test.
- **Integrity:** per item, R_k should not increase with k and U_k should not decrease; violations beyond the intervals
  are reported as estimation or implementation problems.

## W2C — stage 1: M2 / M3 at fresh seeds 1-3 (registered 2026-10-28, before any code or run; criterion (c) candidate)
- **Why:** W2M seed 0 (R049) found plausible-wrong replacement (M2) damaging and uniform-wrong replacement in the same
  slots (M3) healthy, with canonical NLL barely distinguishing them (M2 - M3: +20.94 ± 0.37 Gen-PPL, 4.04 nats/window
  clean NLL; R055). One seed establishes nothing (R044). Approved by the user as part of the package (R056); stage 2
  (flag controls) is cut. Seed 0 stays exploratory and is not part of this reading.
- **Protocol:** W2M's, unchanged: configs `configs/scale/L_ts_m2_frozen_pw.yaml` and `L_ts_m3_uniform_wrong.yaml` as run at
  seed 0 (25% clean rows, W1's rates, one mask stream, canonical labels, 600 steps, W1's recipe, fp32 proposal forwards),
  `scripts/train_gen.py --seed s` for s = 1, 2, 3. Within a seed both arms see the same data order and the same masked
  slots. A100-40GB, as seed 0 (the notebook's `--micro_batch 64`). Final evaluation as W2M: 16 x 256 fp32 samples,
  sample seeds 42 and 0-14 (the same for every training seed), Gen-PPL under GPT-2 large (primary) and TinyStories-33M,
  entropy, MAUVE (fp32), adjacent duplicates per 1k and bigram gain (`scripts/w1_sample_stats.py`, R031), clean val NLL.
- **Quantities:** per seed s, D_s = G(M2_s) - G(M3_s), Gen-PPL under GPT-2 large, paired over the 16 sample groups
  (± SE); the same under TinyStories-33M; dNLL_s = clean val NLL(M2_s) - NLL(M3_s) (nats/window); dDup_s = duplicates
  per 1k, M2_s - M3_s. Pooled seed SD s_p = sqrt((var_s G(M2_s) + var_s G(M3_s)) / 2), sample variances over s = 1..3
  (the brief's "pooled seed SD").
- **Readings** (seeds 1-3 only):
  - **Damage replicates** if every D_s >= 10, mean D > 2 s_p, and every seed moves the same way under TinyStories-33M
    and duplicates per 1k (D_s > 0 and dDup_s > 0). **Overturned** if mean D < 5, or if the seed-mean M2 - M3 difference
    has the opposite sign (M2 better) under either scorer or MAUVE. Anything else: mixed.
  - **Dissociation:** holds if |mean dNLL| <= 10 nats/window; removed if mean dNLL > 20 (M2's clean NLL worse by more
    than 20 nats/window); between: not established.
  - "Damage replicates" + "dissociation holds" is the criterion-(c) candidate claim: under this continuation protocol,
    frozen-model plausible-wrong replacement damages generation that uniform-wrong replacement in the same slots does
    not, while canonical teacher-forced NLL barely distinguishes the two. If met, the user is told before any further
    compute is spent on it (brief).
- **Cost:** M2 ~51 and M3 ~30 A100-minutes per seed (seed 0, R049), ~245 A100-minutes in total (~22 CU at 5.4 CU/h;
  ~44 at R056's observed burn). Caps: 70 A100-minutes per M2 run, 45 per M3 run; every run < 15 CU. Waits for the
  user's compute purchase (R056).

## M3-0 — pilot: M3 without clean rows (registered 2026-10-28, before any code or run; exploratory)
- **Why:** W2M's control drifted (M0 21.71 vs A0 16.16; M3 21.69); prime suspect the 25% clean rows (R049). Astra A1
  (`docs/research_review_round3_2026-10-27.md`) proposes M3 without clean rows as the missing cell of a clean-fraction x
  replacement factorial; it replaces the earlier M0-without-clean-rows control (R055, R056).
- **Protocol:** M3 seed 0 (`L_ts_m3_uniform_wrong.yaml`) with `clean_row_frac: 0.0` (config
  `L_ts_m3_uniform_wrong_c0.yaml`), seed 0, 600 steps, A100-40GB. The mask stream draws the same uniforms as M3 seed 0
  (the clean-row key is still drawn), so every slot masked in M3 is masked here, plus the slots of the rows M3 left
  clean. W2M's final evaluation.
- **Reading:** I = G(M3, seed 0) - G(M3-0) (GPT-2 large; 21.69 minus the pilot; paired over the 16 sample groups).
  Prediction (Astra A1): I >= 3. I < 1 weakens the clean rows as the principal cause of the control's drift. Between:
  inconclusive. TinyStories-33M, MAUVE, duplicates and clean NLL are reported beside it; A0 (16.16) and W1's A1 (16.60;
  no clean rows, uniform replacement, 1800 steps) are descriptive references. Delta_0 = G(P_0) - G(U_0) needs M2
  without clean rows, which is not funded: no reading is registered for it.
- **Cost:** ~30 A100-minutes (~2.7 CU; ~5.4 at R056's burn); cap 45 A100-minutes.

## W2M-G — eval-only: shared context banks, crossed replay, root reliance, M4 flag routing (registered 2026-10-28, before any code or run; exploratory)
- **Why:** R053 / R055: the collapsed arms keep the control's canonical NLL, so the damage has to be located in which
  contexts generation visits and how the models respond to them. Astra A2 / A4 supply the measurements and predictions.
  W2M-G also decides the direction-1 kill (L1R below).
- **Models:** the released generator and W2M's seed-0 final checkpoints M0, M2, M3, M4 (`L_ts_m*_s0/final.pt`); M4 with
  zero flags unless stated. fp32, TF32 off, eval mode, for every forward, sample and score.
- **Sampler:** ancestral, level by level, T = 1, Gumbel-max: level 0 from the released unigram, then each level
  j = 1..11 from the model's level-j call (`src/ncmr/cached_decode.py`; a level attends to the prefix and itself, so its
  logits equal the teacher-forced ones, which the tests check). Noise: one generator per sample group, seeded with the
  group's seed and drawn in level order with fixed shapes, so every cell of a comparison sees the same level-0 codes and
  the same Gumbel noise (common random numbers). Samples are decoded, cut at EOS, pads dropped and re-encoded as in
  `scripts/train_gen.py`.
- **Banks (contexts):** clean: the first 512 val windows (`examples("val", 512)`, even-index validation items, as
  `scripts/mech_eval_w2m.py`); generated: 512 ancestral samples each from the released model, M2 and M3 (two groups of
  256, seeds 900 and 901, the same for every model). For (iv) also 512 from M4 (seeds 900 and 901) and the 512
  plausible-wrong corrupted windows of `w2m.corrupted_contexts` at seed 2026 (W1's rates; the mech eval's).
- **(i) Predictive divergence:** on each bank, teacher-forced on the bank's codes, the Jensen-Shannon divergence (nats)
  between M2's and M3's predictive distributions at every position of levels 2-5; per bank, J_b = the mean over levels
  2..5 of the per-level mean over positions and windows. Ratio r_b = J_b / J_clean for b in {released, M2, M3 samples};
  95% CI by a window bootstrap (2000 resamples, seed 0, both banks resampled). Prediction (Astra A2, "the model discounts
  generated-looking contexts"): **holds** if r_b >= 3 for all three generated banks with every lower CI bound > 1.2;
  **overturned** if every r_b's upper CI bound is <= 1.2; else mixed. Per-level J is reported.
- **(ii) Crossed replay:** prefix model P and suffix model Q in {M2, M3}: P samples levels 1..c-1, Q samples levels
  c..11 given them, at cuts c = 2, 4, 6 (prefix through L1, L3, L5). Cells: (M2, M2) and (M3, M3) (identical at every
  cut by the shared noise) and the two crossed cells at each cut: 8 cells, 2048 samples each (8 groups of 256, seeds
  0-7). Per cell: Gen-PPL (GPT-2 large, primary; TinyStories-33M), the mean over groups; the mean token log loss L = the
  mean over groups of the group's mean token NLL (= log of the group's Gen-PPL); entropy and adjacent duplicates per 1k
  (`src/ncmr/eval_text.py`). With Delta = L(M2, M2) - L(M3, M3) (notation prefix -> suffix), the prefix share
  pi_c = (L(M2 -> M3 at c) - L(M3, M3)) / Delta and the suffix share sigma_c = (L(M3 -> M2 at c) - L(M3, M3)) / Delta,
  interaction 1 - pi_c - sigma_c; SE by the jackknife over the 8 groups (the pairing is by group). Readings (GPT-2 large
  log loss; TinyStories-33M beside it): **early-state selection** (Astra A2's forecast) if pi_4 >= 0.5 and
  pi_6 >= 0.75; **conditional rendering** if sigma_6 >= 0.8 (Astra's overturning result: a healthy prefix through L5
  keeps >= 80% of the damage); else distributed. Sanity: G(M2, M2) and G(M3, M3) within 3 SE of W2M's 42.62 and 21.69
  (different noise, fewer samples); a miss is reported, and the shares are still computed within W2M-G.
- **Direction-1 run gate (L1R):** **passes** if pi_2 >= 0.25 under GPT-2 large log loss and pi_2 > 0 under
  TinyStories-33M. Why 0.25: the kill predicts >= 25% rescue from changing L1's targets alone; if the L0-L1 prefix that
  M2 samples carries less than a quarter of the contrast, the rescue would have to come through L1's effect on the
  response to later levels, which R055's numbers (0.42 nats apart at clean L1) make unlikely enough not to fund now.
- **(iii) L1 root reliance:** on the clean bank, for each window 8 roots drawn from the released unigram (generator seed
  2027, the same draws for every model; a draw may equal the true root), level 1's input rebuilt from the drawn root,
  level 1's canonical targets kept: rho = the mean over windows and draws of NLL_L1(drawn root) - NLL_L1(true root)
  (nats per window, 8 codes). Models: released, M0, M2, M3, M4. Channel-model prediction (R055: the posterior that a
  supplied root is canonical is 0.43 under M2, 0.63 under M3): rho(M2) < rho(M3). **Holds** if the paired (over
  windows) 95% CI of rho(M3) - rho(M2) lies above 0; **overturned** if it lies below 0; else neither.
- **(iv) M4 flag routing sweep:** a constant flag f in {0, 0.5, 1} on every input position of levels 1..11 (f * w added
  to every level's input). Generation: M4 at each f, 2048 samples (seeds 0-7, (ii)'s noise); rescue R(f) =
  (G(M2, M2) - G(M4, f)) / (G(M2, M2) - G(M3, M3)) per scorer, with (ii)'s pure cells. Diagnostics: canonical-label NLL
  per window at each f on the clean bank and on the corrupted bank (there also with the true flags); mean
  KL(P_f=0 || P_f) over the positions of levels 1..11 on the clean, corrupted and M4-sample banks. Prediction (Astra
  A4): **no** f recovers 80% of the gap. **Overturned** if some f in {0.5, 1} has R(f) >= 0.8 under both scorers,
  entropy >= M3's - 0.05 and duplicates per 1k <= M3's + 2 on seeds 0-7, and again on fresh seeds 8-15 (run only in
  that case); if it holds on seeds 0-7 only: mixed. Astra A4's four outcome patterns (misrouting, denoising only,
  ineffective use, no capacity bottleneck) are reported descriptively.
- **Cost:** ~30k samples, two scorers and the bank forwards, on an L4 (fp32 with TF32 off is no faster on an A100):
  ~60-90 L4-minutes (~2-2.5 CU; ~4-5 at R056's burn); cap 120 L4-minutes.

## E1b — continued E1 sampling on censored and wide items (registered 2026-10-28, before any code or run; criterion (c) candidate C1)
- **Why:** E1 (R054) missed its (c) bar because 18 WMT14 J0 pairs had no hit in 2048 samples at cut 8 (they carry 21%
  of N_8 into the lower bound at R = 0), not because of evidence against prediction 1. E1a's per-item beta-mixture
  confidence sequences are anytime-valid: by Ville's inequality they hold at any data-dependent stopping time, including
  a continuation decided after seeing the first samples. Registered after E1's result, which is stated here.
- **How E1 seeded its rounds** (`scripts/e1_equivalence.py`, `mc_hits`): one CUDA generator per (item i, cut k),
  seeded 1_000_003 * seed + 101 i + k (seed 0), drawn in rounds of 64 samples, each round drawing its Gumbel noise level
  by level with fixed shapes, stopping at 20 hits or 2048 samples. E1 ran on an L4 (torch 2.11.0+cu130).
- **Items:** cut 8 only, in the released models. WMT14 J0: every item whose E1 count is censored (0 hits) or whose E1a
  interval for T_8 (delta = 0.05 / (128 x 3); erratum 2026-10-28, before any E1b code ran: WMT14 has three registered cuts, 8-10, so E1a's delta is 0.05 / 384, as `scripts/analyze_e1.py` computes and R054 used; the counts below were computed with it) is wider than 2 nats: 18 + 40 = 58 of 128. TinyStories A0: the same rule
  (delta = 0.05 / (64 x 4)): 6 + 8 = 14 of 64. Sudoku has no censored item and no wide interval and is not continued.
  The code recomputes the lists from the saved E1 results.
- **Sampling:** per item, E1's generator and seed: first a replay in rounds of 64 up to E1's sample count n_E1, checking
  that it reproduces E1's hit count exactly (reported per item); then rounds of 512 until the item's T_8 interval is no
  wider than 2 nats or 2^15 = 32768 samples in total. Order: J0's items by index, then A0's. L4, fp32, TF32 off. If a
  replay does not reproduce E1's count (e.g. a changed CUDA library), the item's E1b stream is still a valid independent
  stream: it is read on its E1b counts alone (from sample 1) and flagged. An item's E1b counts replace its E1 counts at
  cut 8; every other item, cut and model keeps E1's.
- **Readings:** prediction 1 re-read for the three released models with E1's and E1a's rules and deltas unchanged
  (`scripts/analyze_e1.py`: E1's rule = lower bound of mean R_8 / mean N_8 >= 0.8; E1a's added rule on the 61 untouched
  TinyStories windows and on all WMT14 and Sudoku items). **E1's (c) replication bar is met** if E1's rule holds in all
  three released models after E1b (Sudoku's E1 reading, 0.993, stands); E1a's stronger rule is reported for each.
  Prediction 2 is not re-read (its comparison models are not continued). E1's readings stay as R054 reported them;
  E1b's are reported beside them.
- **Cost:** at most 72 items x 32768 samples; ~60-100 L4-minutes (~2-3 CU; ~4-6 at R056's burn); cap 150 L4-minutes,
  checked between items; items not reached keep E1's counts, and the reading says so.

## L1R — direction-1 kill: balanced L1 target reassignment (registered 2026-10-28, before any code or run; exploratory)
- **Why:** Astra round 3, direction 1 and A3: change only the context-target association at the first refinement while
  every corrupted input stays M2's. Gated (R055, R056) on W2M-G's crossed replay at cut 2; prior ~15-20%.
- **Arm M2R** (config `L_ts_m2r_frozen_pw_l1reassign.yaml`): exactly M2 at seed 0 (same data order, mask stream,
  replacement stream and micro-batch partition: A100-40GB, `--micro_batch 64`), except L1's targets on a selected set of
  root-corrupted rows. Before training, the run replays its own data, mask and replacement streams for all 600 steps
  (the frozen proposer's level >= 1 draws consume the replacement generator by shape only; the run checks the supplied
  roots against the replay at every micro-batch and stops on any difference) and records, for every row whose level-0
  code is replaced, its canonical root a_i, its supplied root u_i (a_i != u_i by construction) and its canonical L1
  block y_i (8 codes). Selection: the largest balanced subset of the multigraph with edges a_i -> u_i (every root has as
  many selected outgoing as incoming edges), solved as a maximum circulation (a linear program on a network matrix,
  integral); within each (a, u) pair the rows are drawn at random (seeded); within each root r the selected rows with
  u_i = r receive, by a random permutation, the blocks y_j of the selected rows with a_j = r. Only roots and masks enter
  the selection, never target content or a score. Unselected rows are unchanged.
- **CPU gate (code):** tests show that M2R's inputs, masks, replacement codes and its level >= 2 targets equal M2's at
  every step, that the L1 target multiset over the run is unchanged, that every reassigned row receives a block whose
  canonical root equals its supplied root, and that a resumed run equals an uninterrupted one; a CPU estimate of the
  coverage on the seed-0 data stream (simulated replacement draws) is reported. **Run-time gate:** coverage = selected
  / root-corrupted rows >= 0.80, else the run stops before training (a kill).
- **Run gate:** W2M-G (ii)'s direction-1 gate passes (pi_2 >= 0.25 under GPT-2 large log loss, > 0 under
  TinyStories-33M).
- **Reading** (seed 0; W2M's final evaluation; rescue r = (G(M2) - G(M2R)) / (G(M2) - G(M3)) with W2M's seed-0 M2 and M3,
  per scorer): prediction **>= 25% rescue under both scorers**, with no diversity loss explaining it (entropy >= M3's
  - 0.05, i.e. >= 3.94; MAUVE >= M2's 0.412; duplicates per 1k <= M2's). **Killed** if r <= 0.10 (GPT-2 large), if the
  coverage gate fails, or if the gain comes with entropy below 3.94. Between 0.10 and 0.25: reported, no expansion.
  Seeds 1-3 (~14 CU at run time) only after a new user decision. A rescue of a deliberately damaged model is not a
  quality advance over the healthy model and is not reported as one.
- **Cost:** ~55 A100-minutes (M2's 51 plus the replay; ~5 CU; ~10 at R056's burn); cap 75 A100-minutes. Waits for the
  run gate and the user's compute purchase.

## W2M-GR — W2M-G's localization at W2C's seeds 1-3 (registered 2026-10-28, after W2C stage 1's result (R062) and before any code or run; criterion (c) candidate, with W2C)
- **Why:** W2C stage 1 replicated M2's damage and the dissociation at seeds 1-3 (R062). That is a finding about which
  wrong codes damage generation. Criterion (c) asks for a finding about what the levels encode. The link to the levels is
  W2M-G's crossed replay (R058): the damage forms at levels 2-5 and is amplified by M2's own later levels, not at L1. That
  rests on seed 0 only, and its pattern was read after the run. W2M-GR replicates it with thresholds fixed now, before any
  seed 1-3 measurement exists. Approved by the user (2026-10-28).
- **Models:** for each training seed s in {1, 2, 3}: the released generator and W2C's final checkpoints M2_s and M3_s
  (`L_ts_m2_frozen_pw_s<s>/final.pt`, `L_ts_m3_uniform_wrong_s<s>/final.pt`). No M0 or M4 exists at these seeds.
- **Protocol:** W2M-G's, unchanged, restricted to (i), (ii) and (iii): the same sampler, the same noise (banks seeds 900
  and 901; replay seeds 0-7; roots seed 2027; bootstrap seed 0), the same clean bank, the same scorers, fp32, TF32 off,
  eval mode. (iii)'s models are the released generator, M2_s and M3_s. (iv), the M4 bank and the corrupted bank are not
  run. One run per training seed, each on an **L4** (as W2M-G; the Gumbel streams depend on the device). Seed 0 stays
  exploratory and is reported beside, never pooled.
- **Sanity (per seed):** G(M2_s, M2_s) and G(M3_s, M3_s) under GPT-2 large within 3 SE of W2C's final-eval G(M2_s) and
  G(M3_s) (R062). A miss is reported; the shares are still computed within W2M-GR.
- **Quantities (per seed, GPT-2 large log loss primary, TinyStories-33M beside):** W2M-G's shares pi_c (M2 prefix, M3
  suffix), sigma_c (M3 prefix, M2 suffix) and the interaction 1 - pi_c - sigma_c at cuts c = 2, 4, 6, jackknife SE over
  the 8 groups; (i)'s ratios r_b; (iii)'s rho(M3) - rho(M2) with its paired CI. Across seeds: the mean and seed SD of
  every share.
- **Readings (seeds 1-3 only):**
  - **Localization replicates** if, in every seed under GPT-2 large log loss: (a) pi_2 <= 0.10 (the L0-L1 prefix carries
    little), (b) pi_6 >= 0.30 (M2's prefix through L5 carries a substantial share), (c) sigma_6 <= 0.20 (M2's levels
    6-11 alone carry little), and (d) the interaction >= 0.20 at both cut 4 and cut 6 (M2's later levels amplify what
    its levels 2-5 produce); and (a)-(d) hold on the seed means under TinyStories-33M. **Overturned** if, on the GPT-2
    large seed means, pi_2 >= 0.25 (L1 carries the damage), or sigma_6 >= 0.50 (the late levels carry it), or the
    interaction is < 0.05 at both cuts 4 and 6 (additive, no amplification). Anything else: mixed. Seed 0's values
    (0.009, 0.511, 0.061, 0.469 / 0.428) set the pattern; the thresholds leave room for seed variation while still
    separating "L1", "late levels" and "levels 2-5 with amplification".
  - **(i) replicates** if W2M-G's (i) rule holds in every seed (every r_b >= 3 with every lower CI bound > 1.2).
  - **(iii) replicates** if the paired 95% CI of rho(M3_s) - rho(M2_s) lies above 0 in every seed.
  - W2M-G's own (ii) reading per seed (early-state selection / conditional rendering / distributed) is reported.
  - "Localization replicates", together with W2C's "damage replicates" and "dissociation holds", is the criterion-(c)
    candidate: under W2C's continuation protocol, plausible-wrong replacement damages generation through levels 2-5,
    amplified by the damaged model's own later levels, not through L1. If met, the user is told before any further
    compute is spent on it, and the claim audit covers it.
- **Code (to build, one PR with Codex review and a delta deep audit, since eval code changes):** `scripts/w2mg_eval.py`
  takes the arm set (M2, M3 only) and skips what needs M0 or M4; its sanity reference for seed s >= 1 comes from W2C's
  saved results; a script reads the three runs into the readings above. Tests check that the reduced run's (i)-(iii)
  equal the full run's on the same models (tiny CPU models), and the reading functions at their boundaries.
- **Cost:** per seed 8 cells x 2048 samples, three 512-sample banks, the bank forwards and (iii) for three models: ~20-25
  L4-minutes per seed (seed 0's full run took ~31 L4-minutes with 11 cells, four banks and (iv)); cap 40 L4-minutes per
  seed; ~2 CU at run-time rates for three seeds (~2.5 at the ~1.25x burn).

## W2M-GX — cross-seed and single-level replay on W2C's seed 1-3 checkpoints (registered 2026-10-28, after W2M-GR (R065) and deep audit 14 (R067), before any code or run; exploratory, supports the scoped W2C + W2M-GR result)
- **Why:** audit 14 (A14-1) left two open contributors to W2M-GR's prefix/suffix interaction: checkpoint
  compatibility (co-adaptation within one checkpoint) and M3's response to a foreign-prefix distribution. It also noted
  that whole-range cuts at 2, 4, 6 cannot say which of levels 2-5 matter. Approved by the user (2026-10-28).
- **Models:** the released generator; M2_s and M3_s for s in {1, 2, 3} (W2C's `final.pt`). Record each checkpoint's
  SHA-256 in the result.
- **Protocol:** W2M-G's sampler, scorers, precision and noise, unchanged:
  - 8 groups x 256 per cell, replay seeds 0-7, roots seed 2027.
  - Released-unigram roots shared across cells.
  - fp32, TF32 off, eval mode, one L4.

  New cells only; W2M-GR's cells are not re-run. "A_s -> B_t at cut c" means A_s samples levels 1..c-1 and B_t samples
  levels c..11.
  - (X) Cross-seed, cut 6, every ordered pair s != t: M2_s -> M3_t, M3_s -> M2_t, **M2_s -> M2_t** and
    **M3_s -> M3_t** (24 cells).
  - (S) Single-level swaps on seed s's own pair, for l = 1..5 (30 cells over three seeds):
    - M3_s everywhere except level l, which M2_s samples;
    - M2_s everywhere except level l, which M3_s samples.
  - (C) Cuts 3 and 5 on seed s's own pair: M2_s -> M3_s and M3_s -> M2_s (12 cells).
- **Quantities.** l is GPT-2 large mean log loss of the decoded text per group (TinyStories-33M beside).
  delta_s = l(M2_s, M2_s) - l(M3_s, M3_s), from W2M-GR's pure cells of seed s.
  - Same-arm compatibility penalty, for A in {M2, M3}, with delta_bar the mean of delta_s and delta_t:
    kappa_A(s, t) = [l(A_s -> A_t, 6) - (l(A_s, A_s) + l(A_t, A_t)) / 2] / delta_bar.
  - Cross-seed shares at cut 6: pi_6(s, t) = [l(M2_s -> M3_t) - l(M3_t, M3_t)] / [l(M2_s, M2_s) - l(M3_t, M3_t)].
    sigma_6(s, t) is defined likewise with M3_s -> M2_t.
  - Single-level effects on seed s:
    - e_up(l) = [l(M3 with M2 at l) - l(M3, M3)] / delta_s;
    - e_down(l) = [l(M2, M2) - l(M2 with M3 at l)] / delta_s;
    - pi_3, pi_5, sigma_3, sigma_5.
  - Jackknife SE over the 8 groups for every ratio.
- **Readings (seeds 1-3; GPT-2 large primary):**
  - **Compatibility minor** if every |kappa_M2(s, t)| <= 0.10 and every |kappa_M3(s, t)| <= 0.10.
  - **Compatibility material** if the mean of kappa_M2 over the six pairs is >= 0.25.
  - Otherwise compatibility is mixed.
  - **Cross-seed localization holds** if every pi_6(s, t) >= 0.30 and every 1 - pi_6(s, t) - sigma_6(s, t) >= 0.20.
  - It **fails** if the mean pi_6(s, t) < 0.15.
  - The single-level effects and cuts 3 / 5 are descriptive: no reading is registered. "Concentrated at level l" may be
    said only if e_up(l) and e_down(l) each carry >= 50% of the summed effect over l = 1..5 in every seed.
  - **Sanity:** every pure cell re-sampled for kappa must equal W2M-GR's pure cell of the same checkpoint within 3 SE.
    Alternatively, the pure cells may be taken from W2M-GR's results if the noise streams are provably identical. The
    code must state which.
- **Cost:** 66 cells x 2048 samples. W2M-GR ran 8 cells per seed in ~18.5 L4-minutes, so this is ~2.6 L4-hours, ~4.5
  CU at run time; cap 3.5 L4-hours. It may be split into one run per seed (~1.5 CU each) if that fits the notebooks'
  resume rules better.

## PD — marginal-preserving association kill: plausible-wrong codes reassigned across examples at L2-L5 (registered 2026-10-28, before any code or run; exploratory kill; Astra round 4 Direction 1)
- **Why:** audits 12 and 14 list the missing control. Does M2's damage need the *association* between a plausible
  wrong code and its own example, or only the non-uniform multiset of wrong codes? Approved by the user (2026-10-28).
- **Arms.** All use seed 0 and continue the released TinyStories generator under W2M's protocol: W1's per-level rates,
  one mask stream, 25% clean rows, the same data order.
  - **P:** M2 exactly (`L_ts_m2_frozen_pw.yaml`).
  - **D:** M2's proposals are computed as in P. Then, at levels 2-5 only, the proposed wrong codes of the masked slots
    are permuted across rows by a seeded derangement, within each batch, level and position.
    - A receiving slot whose canonical code equals the incoming code keeps its own proposal.
    - Every other level is exactly as in P. Batches, masks, clean rows and the proposal forward are identical to P's.
    - The derangement uses its own RNG stream, saved in the checkpoint, which no other stream consumes.
  - **U:** M3 (`L_ts_m3_uniform_wrong.yaml`), the same-step reference.
  - All three run **175 steps**, then W2M's final evaluation at step 175 (16 groups x 256, fp32).
- **Coverage:** the fraction of masked L2-L5 slots in D that receive another row's code, logged per level.
  - It must be >= 0.80 overall for any reading; below that the kill is **inconclusive**.
  - It is checked on CPU before launch, on the real data and masks for the first 20 steps.
- **Quantities:**
  - L_X = mean over the 16 groups of log Gen-PPL (GPT-2 large; TinyStories-33M beside).
  - r = (L_P - L_D) / (L_P - L_U).
  - Paired SEs over groups.
  - MAUVE, unigram entropy, adjacent duplicates per 1k.
- **Readings:**
  - **Pilot valid** iff L_P - L_U >= 0.15 nats/token under GPT-2 large. Otherwise it is **underpowered**, with no
    reading. A 600-step reference must not be substituted.
  - **Association matters (rescue)** iff all of: r >= 0.25 under both scorers; D's duplicates <= P's; D's MAUVE
    >= P's - 0.02; D's entropy >= U's - 0.05.
  - **Killed** iff r <= 0.10 under GPT-2 large, or if the rescue holds only with an entropy or MAUVE drop beyond those
    margins.
  - Otherwise: inconclusive.
  - A rescue supports example-conditioned association as a causal factor at fixed marginals. It does not separate
    semantic conflict from within-row coherence or embedding displacement.
- **If rescue:** P and D at fresh seeds 4-6, 600 steps (~28 CU at run-time rates), registered separately before launch.
- **Cost:** P and D ~15 A100-min each (M2 ran 50.7 min per 600 steps) and U ~9, plus three final evals. About 4.5 CU at
  run-time rates; cap 5 CU (60 A100-minutes total).

## E0 — middle-level erasure kill: zero the masked L2-L5 contributions instead of replacing them (registered 2026-10-28, before any code or run; exploratory kill; Astra round 4 Direction 3)
- **Why:** the only candidate method gate toward a paper-scale run (Astra round 4, Part C). Approved by the user
  (2026-10-28). The reviewer's estimate of P(gain) is about 25%.
- **Arms.** All use seed 0 and continue the released TinyStories generator for 400 steps: W1's per-level rates, one
  mask stream, no clean rows, the same data order.
  - **U0:** M0's uniform-corruption recipe (`L_ts_m0_uniform.yaml`, the upstream replacement convention), with
    `clean_row_frac: 0.0`.
  - **E0:** identical to U0, except at levels 2-5. There, the masked slots' code-embedding contributions are replaced
    by a fixed zero vector before the existing upsampling and convolution, instead of by a replacement code.
    - Targets, rates, masks, parameters and sampling depth are unchanged.
    - No flag or extra input is added.
    - At inference nothing changes: no slot is erased.
- **Evaluation:** W2M's final evaluation at step 400 (16 x 256, fp32, both scorers, MAUVE, entropy, duplicates).
- **Readings:** gain g = G(U0) - G(E0), GPT-2 large Gen-PPL, paired over the 16 groups.
  - **Method gate passes** iff all of: g >= 1.0; E0 better under TinyStories-33M; E0's entropy >= U0's - 0.05; E0's
    MAUVE >= U0's - 0.02; E0's duplicates <= U0's.
  - **Killed** iff g <= -0.5, or the gain holds only with a diversity loss beyond those margins.
  - Otherwise: inconclusive. An inconclusive result buys nothing.
- **If it passes:** E0 and U0 at fresh seeds 4-6, 600 steps (~16 CU), registered separately. No CU purchase unless that
  three-seed confirmation passes (Astra T1 / T2) and the user decides.
- **Cost:** ~20 A100-min per arm at 400 steps (M3 ran ~29 min per 600), plus two final evals: ~4.5 CU; cap 5 CU (55
  A100-minutes total).

## W2M-GXa and PDa — clarifications before any code result (registered 2026-10-28; raised by the Codex build, answered by Claude with the user's go-ahead)
- **W2M-GXa (sigma_6 cross-seed).** W2M-GX's "sigma_6(s, t) likewise with M3_s -> M2_t" was ambiguous. The interaction
  is only meaningful when pi and sigma come from the same model pair. For the pair (M2_s, M3_t):
  - pi_6(s, t) = [l(M2_s -> M3_t) - l(M3_t, M3_t)] / [l(M2_s, M2_s) - l(M3_t, M3_t)], as registered.
  - sigma_6(s, t) = [l(M3_t -> M2_s) - l(M3_t, M3_t)] / [l(M2_s, M2_s) - l(M3_t, M3_t)]. This uses the cell with M3_t's
    prefix and M2_s's suffix, which is among the 24 (X) cells.
  - The interaction is 1 - pi_6(s, t) - sigma_6(s, t). The readings are unchanged.
- **PDa (derangement rule).** W2M-GX's sibling PD said that a receiving slot whose canonical code equals the incoming
  code "keeps its own proposal". That breaks exact multiset preservation. Example: canonical [0, 1], proposals [1, 2];
  the swap gives [2, 1], and the fallback gives [2, 2]. The replacement rule within each (batch, level, position)
  group of masked slots at levels 2-5 is:
  - D uses a permutation pi of the group's slots: slot i receives proposal[pi(i)]. Any permutation preserves the
    multiset exactly.
  - pi must satisfy proposal[pi(i)] != canonical[i] for every i. Fixed points (pi(i) = i, the slot keeps its own
    proposal) are allowed; they are always valid, because M2's own proposal excludes the canonical code.
  - Construction, using only the derangement RNG stream:
    1. Draw a uniformly random derangement of the group (with n = 1, the identity).
    2. While some slot i collides (proposal[pi(i)] == canonical[i]), take the lowest-index colliding i. Among slots
       j != i, collect those whose swap of targets clears both collisions (proposal[pi(j)] != canonical[i] and
       proposal[pi(i)] != canonical[j]). If any exist, pick one uniformly and swap pi(i) and pi(j). Otherwise make i
       a fixed point: with k = pi^-1(i), set pi(k) = pi(i) and pi(i) = i.
    3. The loop terminates: a swap strictly reduces the number of colliding slots, and a fixed point is never a
       collision. The identity is always valid.
  - Coverage is the fraction of masked L2-L5 slots with pi(i) != i. The registered >= 0.80 gate is unchanged.
  - Tests: an exact multiset per group; no canonical code received; termination on adversarial groups (all canonical
    codes equal, n = 1, n = 2 with crossed codes); resume equality of the stream.

## E0a — clarification of E0's diversity kill (registered 2026-10-28, before any code result; raised by the Codex build)
- E0 says "killed iff g <= -0.5, or the gain holds only with a diversity loss beyond those margins". The terms mean:
  - **Gain:** any GPT-2 large improvement, g > 0. It need not meet the gate's g >= 1.0 or the TinyStories-33M
    condition. The clause exists so that a Gen-PPL improvement bought by lower diversity reads as killed, not
    inconclusive.
  - **Diversity loss beyond those margins:** E0's unigram entropy < U0's - 0.05, or E0's MAUVE < U0's - 0.02. This
    matches PD's kill clause.
- Duplicates are not part of the kill, since they have no registered margin; a higher duplicate rate still fails the
  gate.
- **Full rule:** killed iff g <= -0.5, or (g > 0 and (entropy < U0 - 0.05 or MAUVE < U0 - 0.02)). The gate passes as
  registered. Otherwise the result is inconclusive.
