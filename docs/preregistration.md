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
