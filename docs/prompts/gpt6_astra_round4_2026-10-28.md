# Research review, round 4 (for an external model), 2026-10-28

You reviewed this project three times (`docs/research_review_2026-10-26.md`, `docs/research_review_2026-10-27.md`, and
round 3, triaged in R055 / `docs/plan_2026-10-27.md` section 10). Since round 3 everything it queued has reported: D1,
the W2M mechanism eval, E1 and its continuation E1b, W2M-G, the no-clean-rows pilot M3-0, and W2C stage 1 (M2 / M3 at
fresh seeds 1-3). Two results now meet a pre-registered criterion-(c) bar and have survived a deep code audit each; a
third measurement (W2M-GR) decides whether one of them is a finding about the levels. About three weeks remain before
the write-up. This round has three parts:

- **Part A (hostile, most important):** try to break the two claim candidates (section 4) as a reviewer at a top venue
  would. Decide what each may say, what it must not say, and the cheapest test that would overturn it.
- **Part B (convert):** with ~96 CU, what single experiment programme most increases the value of the paper: consolidate
  (c), turn E1 into an efficiency result for (a) or (d), or something better? Ideas welcome, but every one must reach an
  experiment that fits section 6.
- **Part C (short):** the paper. What is its one-sentence contribution, and what is missing for it to stand?

Ground rules (as before): label every statement **[ours]** (a number from this brief), **[source]** (a published paper
or official code: authors, year, venue or arXiv id, and the setting a number comes from) or **[speculation]**. If you
are unsure a paper exists or says what you remember, say so; do not invent citations or numbers. Challenge our framing.
Prefer mechanisms over recipes: for each claim, say what quantity it predicts and how we would measure it. The repository
is public (github.com/amritisepic/NCM-research, branch `claude/ncm-research`: `STATE.md`, `log.md` R001-R065,
`docs/preregistration.md`, `docs/reviews/`); this brief is self-contained.

## 1. The object of study (recap)
Next-Concept Modeling (NCM; Kumbong, Khare et al., arXiv 2610.14162; code HazyResearch/ncm) generates text
coarse-to-fine. A frozen tokenizer (a VQ-VAE with multi-scale residual quantization, as in VAR / Infinity) maps each
128-token window to 12 levels of codes (sizes 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128; 4096 codes per level);
level k quantizes the residual the coarser levels leave; the decoder maps the summed up-sampled embeddings to tokens
(~99% token accuracy). A transformer generator predicts all of level k's codes in parallel and independently, given the
reconstruction of levels < k; level 0 comes from a unigram; 11 forward passes per window vs 128 for an AR model. Training
uses conditioning corruption: a share of the coarser input codes (mean 0.5, 0.75 at level 0 linearly to 0.25 at level
10) is replaced by uniform random codes, labels stay canonical. We work with the released checkpoints: TinyStories
generator 88.6M + tokenizer 36.4M, Sudoku-hard 25.5M, WMT14 De-En 120M (11 levels).

## 2. Success bar (any one, met precisely; pre-registered; a training-arm claim needs 3 fresh seeds, gain > 2x pooled SD)
- (a) Better than matched NCM at equal compute across the full suite with <= 1.5x NFE, or matched quality at <= 0.6x NFE.
- (b) Close >= half of the NCM -> AR gap with >= 3x fewer NFE than AR. (Cut: no candidate.)
- (c) A replicated mechanistic finding about what the levels encode (released-model properties: replication across
  domains / checkpoints; training effects: fresh seeds).
- (d) An adaptive-compute rule saving >= 25% NFE at < 1 point quality loss on Sudoku AND a text task, with a >= 25%
  batch-1 wall-clock saving and the attention-inclusive cost reported. (Closed on text for Jacobi decoding by D1.)

## 3. What happened since round 3 (all [ours])
- **D1 (R052):** the text gate fails: J2-F plus the output-stability stop takes 1.41x the batch-1 latency of J0's
  sequential decode (fp32, A100); Sudoku's verifier stop 0.44x. Criterion (d) is closed on text for Jacobi; J1d not run.
- **W2M mechanism eval (R053):** M4 uses its flags (-35 nats/window); plausible-wrong windows are detectable at the row
  level by every model (AUC >= 0.92 from level 2); the collapsed arms are not less dependent on coarse context when the
  context is clean. W2M's registered reading is final: mixed.
- **W2M-G (R058, eval-only, seed-0 checkpoints, registered, exploratory):**
  - (i) M2 and M3's predictive distributions (Jensen-Shannon, levels 2-5) diverge 0.0126 nats on real contexts, and
    6.34x / 6.86x / 5.53x more on samples from the released model / M2 / M3.
  - (ii) Crossed replay (prefix model samples levels 1..c-1, suffix model c..11; shares of the M2-M3 log-loss contrast,
    GPT-2 large; ± jackknife SE):

    | cut (prefix through) | pi (M2 prefix, M3 suffix) | sigma (M3 prefix, M2 suffix) | interaction |
    |---|---|---|---|
    | 2 (L1) | 0.009 ± 0.011 | 0.868 ± 0.019 | 0.123 |
    | 4 (L3) | 0.220 ± 0.015 | 0.311 ± 0.015 | 0.469 |
    | 6 (L5) | 0.511 ± 0.010 | 0.061 ± 0.009 | 0.428 |

    Read after the run: the damage forms at levels 2-5 and is amplified by M2's own later levels, which are nearly
    harmless on M3's prefixes; not at L1 (so the L1 target-reassignment kill, L1R, was not run).
  - (iii) M2's L1 leans 41% less on the root code than M3's (root reliance 2.00 vs 3.39 nats/window), as a channel model
    predicted, at no cost to generation.
  - (iv) No constant M4 flag rescues: f = 1 turns M4 into M2 (a regime switch); f = 0 is its best setting (rescue 0.53).
- **M3-0 (R059, seed 0):** M3 without the 25% clean rows returns to the released model's quality (Gen-PPL 16.50 vs 21.69
  with clean rows; released 16.16) while its canonical code NLL gets 106 nats/window worse (78% of it at L8-L11): the
  clean rows caused W2M's control drift.
- **E1 / E1b (R054, R060; deep audit 11, R061): representation vs text share of code NLL.** For one item, N_k = the
  canonical code NLL of levels >= k given the canonical levels < k; rho_k = the probability that the model's own samples
  of levels >= k decode to exactly the item's text (Monte Carlo, beta-mixture confidence sequences, Bonferroni per model);
  T_k = -log rho_k (text cost), R_k = N_k - T_k (cost of choosing among code paths that decode to the same text; R_k >= 0
  because the canonical path is one of them). At cut 8 (registered items; modelwise 95%; Monte Carlo error only):

  | released model | items | mean N_8 (nats) | R_8 / N_8 bounds | share of all code NLL in N_8 |
  |---|---|---|---|---|
  | TinyStories (per 128-token window) | 64 | 404 | [0.894, 0.997] | 46% |
  | WMT14 (per sentence) | 128 | 131 | [0.859, 0.981] | 46% |
  | Sudoku-hard (per puzzle) | 128 | 32 | [0.993, 1.000] | 74% |

  E1's registered (c) bar (lower bound >= 0.8 in all three models) is met; the 0.90 version only in Sudoku. Resampling
  items: 5% quantiles of the lower bound 0.819 (TinyStories), 0.800 (WMT14). Censoring: 5 TinyStories windows and 9 WMT14
  sentences had 0 hits in 32,768 samples. Prior art we know: marginal likelihood over subword tokenizations (Cao and
  Rimell 2021).
- **W2C stage 1 (R062; deep audit 12, R063): seeds 1-3, M2 frozen plausible-wrong vs M3 uniform-wrong**, W2M's protocol
  (continuation of the released TinyStories generator, 600 steps, 25% clean rows, W1's per-level rates, one mask stream so
  both arms corrupt the same slots; "plausible-wrong" = a draw at T = 1 from the frozen released model given the
  canonical coarser levels, canonical code excluded; the root from the released unigram, canonical root excluded):

  | seed | G(M2) | G(M3) | D (GPT-2 large, ± paired SE) | D (TinyStories-33M) | dNLL (nats/window) | dDup / 1k | dMAUVE |
  |---|---|---|---|---|---|---|---|
  | 1 | 39.79 | 21.43 | +18.36 ± 0.32 | +28.5 | +3.86 | +12.4 | -0.41 |
  | 2 | 41.31 | 21.80 | +19.51 ± 0.31 | +31.1 | +3.91 | +12.3 | -0.42 |
  | 3 | 40.56 | 21.80 | +18.76 ± 0.32 | +29.8 | +4.32 | +12.2 | -0.39 |
  | seed 0 (exploratory) | 42.62 | 21.69 | +20.94 ± 0.37 | +33.1 | +4.04 | +13.0 | -0.45 |

  Registered readings: damage replicates (pooled seed SD 0.56), dissociation holds (|mean dNLL| 4.03 <= 10; 0.53% of
  clean NLL, positive in every seed). All 16 sample groups favour M3 in every seed under both scorers. M2's samples have
  slightly higher entropy (+0.027 nats) and are ~1 token shorter. Audit 12: executed traces confirm identical batches,
  clean rows and masks across arms; no P0 / P1.
- **W2M-GR (registered at `733ea77` before any seed 1-3 measurement; eval-only on W2C's checkpoints; R064-R065):**
  W2M-G's (i)-(iii) at seeds 1-3. Localization replicates iff in every seed under GPT-2 large pi_2 <= 0.10, pi_6 >= 0.30,
  sigma_6 <= 0.20 and the interaction >= 0.20 at cuts 4 and 6, and the same on TinyStories-33M seed means.
  **Result: [TO FILL FROM R065 BEFORE SENDING: per-seed pi_2, pi_4, pi_6, sigma_6, interactions, (i) ratios, (iii)
  contrasts, the registered readings.]**
- Earlier facts that bear on Part B: oracle-intervention curve (supplying real levels 0..j and sampling the rest moves
  Gen-PPL 16.0 -> 8.6; levels 1-3 account for 48%, 4-5 for 39%, 6 for 5%, 7-11 for 7%). Decoder robustness: replacing
  every level but the finest by the generator's own one-step samples keeps 98.0% of tokens, by uniform codes 42.6%. On
  Sudoku, decoding with the finer levels simply left out solves 0% of boards before the last level (cell accuracy 76.0% at
  L9, 81.9% at L10, 98.4% at L11; R004), so "skip the fine levels" cannot mean "drop them".

## 4. The two claim candidates (Part A's object)
**C-E1 (drafted, awaiting the user, `docs/claim_e1_2026-10-28.md`):** "In each of the three released NCM models
(TinyStories, WMT14, Sudoku-hard), given the canonical codes of levels 0-7, at least 80% of the canonical code NLL of
levels 8 and finer is spent choosing among code paths that decode to the same output (lower 95% bounds 0.894, 0.859,
0.993 on the registered items; Monte Carlo error only)."

**C-W2C (wording after audit 12; not claimed):** "Under one continuation protocol for the released TinyStories NCM (600
steps, 25% clean rows, W1's per-level rates, one mask stream), replacing masked codes with wrong codes drawn from the
frozen released model given the canonical coarser levels caused additional generation degradation relative to
uniform-wrong replacement in the same slots: M2 - M3 Gen-PPL +18.9 under GPT-2 large on the mean of three training seeds
(each seed at least +18.36; pooled seed SD 0.56) and +29.8 under TinyStories-33M, MAUVE -0.41, adjacent BPE duplicates
+12.3 per thousand pairs, while canonical teacher-forced NLL was higher by only 4.0 nats/window (0.53%), positive in
every seed." If W2M-GR replicates, add: "the damage forms at levels 2-5 and is amplified by the damaged model's own
later levels, not at level 1."

Audit 12's limits on C-W2C: comparative only (M3 itself is 5.5 Gen-PPL above the released model, because of the clean
rows); no temperature-, frequency- or length-matched control, so model-specific plausibility is not shown to be the
unique cause; not equal compute (M2's proposal forward); one model, domain and protocol.

## 5. Our current thinking for Part B (challenge it)
- **Consolidate (c):** C-W2C + W2M-GR, plus one control that separates "plausible" from "non-uniform": e.g. M2 at a
  higher proposal temperature, or a frequency-matched (unigram-per-level) wrong code, 3 seeds (~25 CU per arm).
- **E1 -> efficiency:** given levels 0-7 the text is nearly decided (T_8 at most ~43 nats per TinyStories window, ~18
  per WMT14 sentence, ~0.2 per Sudoku puzzle), yet levels 8-11 take 4 of 11 sequential passes. Fill them cheaply rather
  than drop them: one parallel pass for all fine levels, a small fill network, or the model's argmax. Best case 3-4 of 11
  passes saved (0.64-0.73x NFE): short of (a)'s 0.6x unless level 7 also goes; (d) needs >= 25% on Sudoku and text plus
  wall-clock. Caveat: E1 measured T at canonical coarse levels; in generation they are the model's own.
- **Marginal-likelihood training** (train on equivalent code paths rather than the canonical one): removes ~90% of the
  fine-level target noise, but text hit rates are tiny (0 hits in 32,768 for some items) and self-generated targets
  echo our two feedback collapses (J2; the on-policy arms).
- **The no-clean factorial** (M2 and M3 at 0% clean rows, seeds 1-3, ~22 CU): would tell whether C-W2C holds without the
  drifted control.
- **Stop and write**, spending little more.

## 6. Constraints (hard)
- **Compute:** ~96 CU available after W2M-GR (Colab balance; A100-40GB ~5.3 CU/h, L4 1.71 CU/h; run-time estimates now
  hold within ~25%). A 600-step tier-L fine-tune of the 88.6M model costs ~2.7 CU (uniform) to ~4.6 CU (with proposal
  forwards) per seed. Any single run > 15 CU needs the user's approval.
- **Hardware besides Colab:** a laptop RTX 4050 (6 GB) for evaluation and small jobs; CPU.
- **Models:** fine-tune or evaluate the released checkpoints; no new tokenizer or full-size generator from scratch.
- **Process:** every comparison pre-registered (prediction, overturning result, reading), Codex code review plus a deep
  audit before any GPU run whose eval / training code changed (~1 day of latency per new piece of code), 3 fresh seeds
  for a training claim, fp32 evaluation. Null results are reported.
- **Timeline:** about three weeks to a write-up; at most two new pieces of code.

## 7. Part A: break the claims
1. **C-E1.** Is "representation choice" the right reading of R_8, or can a large R_8 arise from something else (the
   decoder's many-to-one map at the token-straddling levels; the sampler's independence within a level; censoring on the
   hardest items; conditioning on canonical rather than generated coarse levels)? Is it a rediscovery of Cao and Rimell
   (2021) or of known VQ redundancy results, and what makes it new for a coarse-to-fine code hierarchy? Does it say
   anything about what the levels encode beyond "the fine levels are mostly redundant", and what would make it a stronger
   (c) statement? What is the cheapest test that would overturn it (e.g. a domain or cut where it fails)?
2. **C-W2C.** Which confound would a reviewer press first: plausibility vs any non-uniform wrong distribution;
   displacement size (plausible-wrong codes are near the canonical code in embedding space, uniform-wrong ones are not);
   target conflict; the clean rows; equal steps but unequal compute; the scorers. For each: does W2C's data or W2M /
   W2M-G / M3-0 already answer it, and if not, the cheapest arm or eval that would. Is "clean likelihood barely
   distinguishes them" an interesting dissociation or an expected property of teacher-forced metrics under any
   context-distribution shift (exposure bias)? Is the whole thing a rediscovery of scheduled sampling's improperness
   (Huszár 2015) or of known exposure-bias results in masked / discrete-diffusion models, and what is new?
3. **The pairing.** Does W2M-GR's result (section 3) turn C-W2C into a finding about what the levels encode? If it is
   mixed or overturned, what may still be claimed?
4. For each claim: the strongest single sentence a hostile reviewer would write, and our best answer from the data.

## 8. Part B: convert
1. Rank the options in section 5 and any better ones you see by (value to the paper) x (probability of a clean result)
   / CU, within section 6. For the top one or two: the registered prediction with numbers, the result that kills it, arms
   / seeds / CU, and how it uses what we already have.
2. **E1 -> efficiency, specifically.** Is there a fill of levels 8-11 (or 7-11) that is both cheap and faithful enough,
   given E1's T_k, the decoder-robustness numbers and R004's truncated-decode failure? Derive the expected NFE and
   quality trade-off, say whether (a) or (d) is reachable, and give an eval-only kill test (<= 2 CU).
3. **One new idea, if you have it,** that turns E1 and W2C into a single account (for example: plausible-wrong
   corruption hurts because the fine levels' many-to-one redundancy makes "plausible" and "correct" indistinguishable
   at those levels), with the measurement on existing checkpoints that would test it.

## 9. Part C: the paper
The one-sentence contribution; the three figures that carry it; what a reviewer would say is missing; and what to cut.

## 10. Output format
- Part A first: per claim, a verdict (claim as worded / narrow it / hold), the narrowed wording if any, the reviewer's
  strongest sentence and our answer, and the cheapest overturning test with its prediction.
- Then Part B with a ranked list and at most two fully specified experiments; then Part C in at most 10 lines.
- End with one queue for the ~96 CU: item, CU, gate, what it decides.
- Dense: mechanisms, numbers, experiments; no general encouragement.
