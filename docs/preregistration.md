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

