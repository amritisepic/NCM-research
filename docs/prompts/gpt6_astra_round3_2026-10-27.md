# Research review, round 3 (for an external model), 2026-10-27

You reviewed this project twice (`docs/research_review_2026-10-26.md`, `docs/research_review_2026-10-27.md`). Since then
the eight-arm W2M screen has reported (seven arms; M5 skipped, as W2Ma allowed), S0 failed (the coupled sampler is
dropped), and the timing, equivalence and mechanism evaluations are about to run. This round has two parts:

- **Part A (narrow):** interpret W2M with your round-2 outcome table, and design the confirmation (W2C) at minimum cost.
- **Part B (broad, the more important part):** a fresh hostile evaluation, deep theory, an overview of the field, and
  **new ideas**: one or two research directions that could produce a significant, publishable result. Go
  theoretically deep and think outside the box. Start from first principles about what a coarse-to-fine discrete
  code hierarchy is, what its training-context distribution does to its conditionals, what parallel generation can and
  cannot represent, and what our anomalies are telling us, rather than from the list of known fixes. Borrow from
  wherever the right structure lives: information and coding theory, statistical decision theory, robust statistics,
  control (closed-loop stability), statistical physics, numerical analysis, optimal transport, game theory,
  linguistics, neuroscience. Ideas that look strange are welcome if you can say what they predict. If the right move is
  to change the question, the representation or the objective, say so and argue it. **Unconstrained theory first,
  feasibility second:** every direction you propose must eventually reach an experiment we can run under section 5, but
  do not let the budget narrow your thinking before you get there.

Ground rules (as before): label every statement **[ours]** (a number from this brief), **[source]** (a published paper
or official code: authors, year, venue or arXiv id, and the setting a number comes from) or **[speculation]**. If you
are unsure a paper exists or says what you remember, say so; do not invent citations or numbers. Challenge our framing.
Prefer mechanisms over recipes: for each claim, say what quantity it predicts and how we would measure it. The
repository is public (github.com/amritisepic/NCM-research, branch `claude/ncm-research`: `STATE.md`, `log.md` R001-R050,
`docs/preregistration.md`); this brief is self-contained.

## 1. The object of study (recap)
Next-Concept Modeling (NCM; Kumbong, Khare et al., arXiv 2610.14162; code HazyResearch/ncm) generates text
coarse-to-fine. A frozen tokenizer (a VQ-VAE with multi-scale residual quantization, as in VAR / Infinity) maps each
128-token window to 12 levels of codes (sizes 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128; 641 codes; 4096 per
level); level k quantizes the residual the coarser levels leave; each level's embeddings are linearly up-sampled,
passed through a kernel-3 residual convolution and summed; the decoder maps the sum to tokens (~99% token accuracy). A
transformer generator predicts all of level k's codes in parallel and independently, given the cumulative
reconstruction of levels < k (area-down-sampled to level k's length); level 0 comes from a unigram; 11 forward passes
per window vs 128 for an AR model. Training uses **conditioning corruption**: a share of the coarser input codes (mean
0.5, 0.75 at level 0 linearly to 0.25 at level 10) is replaced by uniform random codes, labels stay canonical. The paper
reports it as its largest ablation effect (Gen-PPL ~48 -> ~19 at 10k steps) [source]. Paper numbers [source]:
TinyStories AR 8.89 Gen-PPL at 128 NFE (data-matched, ~4x fewer training FLOPs), NCM 15.41 at 12 NFE, data 9.67;
Sudoku-hard NCM 86.1% at 12 NFE vs 81.3% for the best discrete-diffusion baseline at 16. We fine-tune the released
checkpoints: TinyStories generator 88.6M + tokenizer 36.4M, Sudoku-hard 25.5M, WMT14 De-En 120M (11 levels).

## 2. Success bar (any one, met precisely; pre-registered; a training-arm claim needs 3 fresh seeds, gain > 2x pooled SD)
- (a) Better than matched NCM at equal compute across the full suite with <= 1.5x NFE, or matched quality at <= 0.6x NFE.
- (b) Close >= half of the NCM -> AR gap with >= 3x fewer NFE than AR.
- (c) A replicated mechanistic finding about what the levels encode (for released-model properties, replication across
  domains and checkpoints counts; for training effects, fresh seeds).
- (d) An adaptive-compute rule saving >= 25% NFE at < 1 point quality loss on Sudoku AND a text task, with a >= 25%
  batch-1 wall-clock saving and the attention-inclusive cost reported.

Status: (c) is our primary target; (d) has one live gate (D1, below); (b) is cut (no candidate, and its AR baseline would
eat the budget); (a) has no candidate. **This bar is a floor, not a ceiling.** If you see a route to a more ambitious
result (for example a theory that predicts, before training, which corruption distributions help or destroy a
coarse-to-fine generator; a training principle that removes NCM's exposure bias and transfers to other multi-scale or
masked generators; or matching AR quality at a fraction of its NFE), propose a sharper criterion, say why it would
matter more, and estimate how reachable it is.

## 3. What we have measured (all [ours]; seed 0 unless stated)
Representation and likelihood
- Coarse codes behave like window bags of words; the token-straddling levels L8-L10 carry near-chance linearly readable
  content; L8-L11 hold 46.8% of clean code NLL on 63% of slots. Small-model code NLL is >= 1059 nats/window above the
  entropy of its own targets (an inequality); released clean code NLL 861.8 nats/window.
- The released generator's one-step disagreement with the encoder per level 0..11 (T = 1): 100, 98, 91, 81, 67, 55, 48,
  21, 45, 53, 55, 14%; from L5 on its sampled code is the encoder's nearest or second-nearest code. Decoder robustness:
  all levels but the finest replaced by uniform codes keep 42.6% of tokens; by the generator's own one-step samples,
  98.0%.
- Oracle-intervention curve: supplying the real codes of levels 0..j and sampling the rest moves Gen-PPL 16.0 -> 8.6;
  levels 1-3 account for 48% of that, 4-5 for 39%, 6 for 5%, 7-11 for 7% (an intervention, not an attribution).
- E1 prototype (3 windows): given the canonical levels 0..7, the model's own samples of levels 8-11 decode to exactly the
  same text with probability ~0.08-0.94; at cuts 5-7 hits are rare. The full E1 (released TinyStories, WMT14, Sudoku;
  confidence sequences, your round-2 amendment) is about to run.
- S0: adding the canonical left neighbour to the factorized head on frozen features gains 6.2 nats/window over levels
  1-5 (1.7% of those levels' NLL; 3.2 at level 1, 1.5 at level 2, < 0.8 beyond). The coupled sampler is dropped.

Wave 1 (released TinyStories model fine-tuned 1080-1800 steps at equal generator FLOPs; Gen-PPL GPT-2 large, 16 x 256
fp32 samples, ± over sample groups; MAUVE; clean code NLL)
| arm | training hierarchy | Gen-PPL | MAUVE | clean NLL |
|---|---|---|---|---|
| A0 | released, no training | 16.16 ± 0.11 | 0.827 | 861.8 |
| A1 | NCM's uniform corruption, continued | 16.60 ± 0.08 | 0.881 | 844.0 |
| A2 | uniform corruption + labels re-quantized against the corrupted residual | 16.29 ± 0.09 | 0.915 | 749.3 |
| A3 | corruption by the model's own one-step samples, canonical labels | 41.44 ± 0.43 | 0.496 | 885.7 |
| A4 | A3 + re-quantized labels | 36.81 ± 0.26 | 0.673 | 707.4 |
| A5 | A3 with two-pass samples | 32.17 ± 0.18 | 0.543 | 913.9 |
| A6 | A1 + soft labels over the 16 nearest codes | 17.46 ± 0.10 | 0.832 | 888.9 |
- The on-policy collapse keeps vocabulary and story plans, breaks local order (bigram gain 2.13 -> 1.71 nats/token,
  adjacent duplicates 6.9 -> 17.7 per 1k). A2's gain exists only at T = 1 (it reverses at T = 0.8). A1 trails A0 by
  0.69 with nothing supplied and by 0.00 once real levels 0..3 are supplied: continued training hurts coarse sampling.
- Sudoku CPU analogue: plausible wrong codes raise predictive entropy on clean contexts by 74% (uniform wrong: 5%) and
  clean NLL by 32 nats (vs 3), but there is no collapse (solve rates fall ~5 points in both arms).

W2M (600 steps, seed 0, W1's corruption rates, **25% of rows left clean in every arm**, one mask stream so arms corrupt
the same slots, fp32 policy forwards; D(X) = Gen-PPL(X) - Gen-PPL(M0), paired over 16 sample groups)
| arm | Gen-PPL | D ± SE | TS-33M | MAUVE | bigram gain | adj. dup/1k | clean NLL |
|---|---|---|---|---|---|---|---|
| M0 uniform | 21.71 | 0 | 18.73 | 0.865 | 2.054 | 8.3 | 753.8 |
| M1 current policy's one-step samples | 42.50 | +20.79 ± 0.27 | 51.80 | 0.468 | 1.698 | 18.2 | 751.2 |
| M2 frozen released model's samples, canonical excluded | 42.62 | +20.91 ± 0.35 | 51.74 | 0.412 | 1.691 | 21.3 | 757.9 |
| M3 uniform-wrong | 21.69 | -0.02 ± 0.10 | 18.65 | 0.865 | 2.059 | 8.3 | 753.9 |
| M4 M2 + truthful flags (row-level; flags saturate) | 31.58 | +9.87 ± 0.23 | 32.60 | 0.720 | 1.894 | 13.9 | 754.4 |
| M6 M1 with proposals under the level-0 code actually used | 42.18 | +20.47 ± 0.37 | 51.44 | 0.454 | 1.692 | 18.8 | 751.2 |
| M7 uniform at A3's lower realized rates | 22.07 | +0.36 ± 0.08 | 19.14 | 0.860 | 2.052 | 8.3 | 741.2 |
- Registered reading "mixed": M2 damaging, M3 healthy, M6 damaging, M7 healthy as predicted; M4's rescue R4 = 0.53
  (predicted >= 0.8; < 0.2 would have overturned). The sanity check fails: M0 is 5.55 above the released model.
- Curves (256 samples at steps 100 / 300 / 600): M0 19.25 / 22.55 / 22.69 while its clean NLL falls 804.6 -> 753.8;
  W1's control (no clean rows) stayed at 16.0-17.1 from step 270 to 1800. M4 tracks M2 at step 100 (37.5 vs 37.6),
  reaches 31.7 by 300, then stays; its learned flag vector ends at norm 0.15.
- Realized disagreement on corrupted rows from L2 on is far lower for M1 than M2 (L5 0.27 vs 0.50, L7 0.08 vs 0.40;
  equal at L0-L1), yet D(M1) = D(M2).
- **The collapsed arms keep the control's clean NLL** (within 4 nats), whereas W1's on-policy arm, trained without clean
  rows, was 42 nats worse than its control.

Decoding: fixed-point (Jacobi) iteration across levels
- Training-free exact Jacobi: Sudoku-hard 6.41 passes for identical output (-46.6% logical NFE); with a verifier stop
  2.37 passes and 90.3% solved (vs 86.15% sequential). WMT14 11.07 passes vs 11 (no saving); TinyStories no saving.
- WMT14 consistency fine-tuning: with the current model as its own teacher (J2), 16% of test sentences collapse onto two
  input-independent strings; with a frozen released-model teacher (J2-F) 0.3% do, but exact passes fall only 5.6%.
  J2-F's early-pass outputs improved a lot (test BLEU after <= 6 / 7 / 8 passes 20.70 / 21.57 / 22.02 vs the released
  model's 17.63 / 19.70 / 21.50). Val only: stopping when the decoded text repeats gives 7.80 passes at -0.54 BLEU. Each
  Jacobi pass processes every level: at 7.8 passes that is ~2.3x the positions of an eager sequential decode and ~6x
  those of a prefix-cached one.
- Continued teacher forcing on WMT14 lowers val code NLL by 43.5 nats/sentence and costs 0.73 BLEU, 92% of it n-gram
  precision.

## 4. Killed, considered, in flight
- Killed / demoted: on-policy corruption with canonical labels as a method; soft labels (A6); re-quantized labels (A2);
  carry across levels (Sudoku, from scratch); training-free text Jacobi; token-aligned schedules; level-skipping on
  Sudoku; consistency training for text Jacobi (J2, J2-F); the marginal-preserving coupled sampler (S0); tier-M
  from-scratch NCM vs AR; Sudoku training arms.
- Considered, not run: a parallelizability-regularized tokenizer; red-black relaxation within a level; the encoder as a
  fixed-point verifier; contextual / summary-supervised coarse levels; importance-resampled levels with a within-level AR
  energy; lifting-scheme tokenizers; successive-cancellation list decoding over levels; global plan tokens; exact AR
  sampling from NCM drafts; JEPA-style continuous coarse latents; your round-1 list (subtractive dither, bag then
  permutation, irreversibility-gated reopening, decoder-aware geometry, damage-matched donor corruptions, protecting
  coarse predictions from fine gradients, intervention-response maps).
- In flight, results unknown when you read this: D1 (batch-1 wall-clock for J2-F + the stop rule vs the fastest
  sequential decoder; you predicted < 10% pass), E1 (the exact-fiber decomposition of code NLL), the W2M mechanism eval
  (flag use for M4; per-level row-level detectability with a uniform-wrong positive control).
- **Held for this review:** W2C (fresh seeds), an M0 control without clean rows, and W2M-G (eval-only on the W2M
  checkpoints: an oracle curve with real coarse levels supplied, and predictive entropy when levels < k are the model's
  own samples vs real data).

## 5. Constraints (hard)
- **Compute:** project budget ~200 Colab compute units, ~94 left; this month's Colab balance ~77 CU (A100-40GB ~5.4
  CU/h, so ~14 A100-hours; L4 1.71 CU/h). A 600-step tier-L fine-tune of the 88.6M model costs ~2.7 CU (uniform
  corruption) to ~4.6 CU (with proposal forwards), final evaluation included. Any single run > 15 CU needs the user's
  approval. A training-arm claim needs 3 fresh seeds per arm. A direction that needs more than the remaining budget must
  say so, by how much, and why the result would justify it (the user raised the budget once).
- **Hardware besides Colab:** a laptop RTX 4050 (6 GB) for evaluation and small jobs; a 4-core CPU container (2 h jobs).
- **Models:** fine-tune the released checkpoints; a new 36M tokenizer or 88.6M generator from scratch is out of budget;
  small from-scratch models (<= ~25M generator, <= ~10M tokenizer) are affordable but ours were far below the released
  models' quality.
- **Process:** every comparison pre-registered (prediction, overturning result, reading), one axis at a time, matched
  data order and seeds, fp32 evaluation, code review plus a deep audit before any GPU run. Null results are reported.
- **Timeline:** weeks, not months (a write-up in about four weeks).

## 6. Part A: W2M and the confirmation
Proposed (held): W2M-G (~2 CU); M0 without clean rows (~2.7 CU); W2C stage 1, M2 and M3 at fresh seeds 1-3 (~22 CU);
stage 2, M4 and your random-row flag control at fresh seeds 1-3 (~27 CU), only if the mechanism eval shows flag use.
1. **What does W2M license, given the drifted control?** Your round-2 table says a substantially deteriorated M0 calls
   for an orthogonal factorial or descriptive findings only. Is M2 vs M3 interpretable on a drifted baseline? Which
   factorial is worth its cost (clean-row fraction x replacement distribution, at which seeds)? Should W2C keep the 25%
   clean rows (needed for M4)?
2. **The clean-NLL invariance.** Give the accounts that fit (discounting of contexts that look like model samples;
   detectability through cross-level inconsistency, a wrong coarse code beside canonical finer codes; damage confined
   to fine-level rendering; others), each with the eval-only measurement on existing checkpoints that separates it and
   a pre-registrable prediction. Is W2M-G the right instrument, and what controls does it need?
3. **W2C at minimum cost.** Arms, seeds, protocol and readings for the strongest (c) claim this budget allows. M2 and M3
   differ in displacement, target conflict and context statistics at once (your round-2 point): what single extra arm
   narrows that most?
4. **M4's partial rescue.** Limited flag capacity or training time, a clean-regime exposure bias at flag 0 (that branch
   is trained only on clean rows), regime-bias effects: which control distinguishes them most cheaply? Is the
   random-row flag control still the decisive one?
5. **Damage that does not scale with wrong slots.** What does M1 = M2 imply about where damage enters (levels 0-2?), and
   what is the cheapest test (replacement restricted to L0-L2 vs L3+)?

## 7. Part B: evaluation, theory, field, new ideas (spend most of your effort here)
1. **Hostile evaluation.** Which of our current conclusions, the W2M story above all, are weakest, and what is the
   cheapest test that would overturn each? Is anything we killed killed for the wrong reason? Is criterion (c), framed
   as "plausibility of the corruption, not feedback, parent mismatch or coverage, causes the collapse; clean likelihood
   cannot see it; provenance protects", publishable, and what would make a reviewer call it a rediscovery of scheduled
   sampling's inconsistency?
2. **Deep theory.** Give one account, derived rather than asserted, that explains as many of these jointly as it can,
   and says which it cannot:
   (i) plausible-wrong corruption destroys generation while leaving clean likelihood intact (with clean rows) and damages
   clean likelihood without them; (ii) uniform corruption is essential to NCM's generation, yet 25% uncorrupted rows
   degrade the uniform control (16.2 -> 21.7) while lowering its clean NLL by ~100 nats; (iii) damage does not scale with
   the number of wrong slots; (iv) row-level provenance halves the damage; (v) code likelihood and sample quality come
   apart across domains (A2, A4, J1, M0); (vi) consistency training with a moving teacher collapses onto
   input-independent fixed points, while a frozen teacher improves early-pass outputs without speeding exact
   convergence; (vii) little within-level pairwise dependence (S0) while coarse levels hold most of the quality gap.
   Specifically: what is the right theory of the *training-context distribution* of a coarse-to-fine generator, that
   is, which distribution over corrupted contexts (and which targets) yields conditionals that are right for the
   contexts the model actually meets at generation time? Is there an optimal or proper corruption, and can you derive
   it, or a computable surrogate, for NCM? What does it predict for the noise level, the noise distribution, the
   per-level schedule and the targets? Candidate lenses, not a checklist: exposure bias and DAgger-style
   distribution-matching guarantees; Huszár's scheduled-sampling improperness; denoising score matching and diffusion
   noise schedules; precision weighting of priors versus evidence (predictive coding); a channel-coding view in which
   corruption is the training channel and generation a different channel; closed-loop stability, with coarse sampling
   errors as disturbances and corruption as a robustness margin; the model and its own sampler as players in a game;
   calibration of trust under covariate shift; renormalization and multigrid views of the hierarchy. For each account,
   the measurable prediction that separates it from the others, and the existing or held experiment that would test it.
3. **Field overview (2024-2026), focused on what bears on these findings:** exposure bias and self-correction in
   discrete diffusion and masked generators (remasking, corrector samplers, on-policy or self-generated training
   contexts), train-test discrepancy in next-scale image generation (VAR / Infinity and their successors), the design of
   corruption or noise distributions for discrete denoising, likelihood vs sample-quality dissociations, Jacobi and
   consistency decoding after CLLMs, adaptive stopping. For each line: what it establishes at what scale and what it
   implies for us. Flag anything that already does what we might propose.
4. **New ideas, then synthesis.** First brainstorm widely: 8-12 genuinely new ideas, one or two lines each, with the
   theoretical reason each could work and a rough probability; at least half must not be variants of anything in
   sections 3-4 or your earlier reviews; include at least two high-risk ideas most researchers would dismiss, and say why
   they might be wrong to. Then pick **one or two breakthrough directions** and, for each, give:
   - the core idea in two sentences, the mechanism, and why it is not incremental;
   - the criterion it targets, (a)-(d) or a more ambitious one you define (section 2), the effect size you expect and
     your probability of success;
   - a kill experiment of <= 5 CU or CPU-only that tells us within days whether to continue, with its pre-registrable
     prediction and the result that would kill it;
   - the full plan to a replicated claim within ~90 CU (arms, seeds, metrics, CU per run), and what it would need if
     the budget were raised (how much, and why it would be worth it);
   - the strongest objection a reviewer would raise and how the design answers it;
   - how it uses or reinterprets what we already have (W2M, the oracle curve, J2 / J2-F, S0, E1, the Sudoku analogue).
   Rank the directions. If consolidating one mechanistic finding is the best use of the budget, argue that explicitly.
5. **What to stop doing.** Name parts of the current queue (including W2C as proposed) you would cut or change, and why.

## 8. Output format
- Part A first and short: at most 8 bullets of verdict, then per question the answer, a pre-registrable prediction
  (with a number) and the result that would overturn it.
- Then Part B, which should be most of your answer, with the deepest work in B2 and B4.
- End with one revised queue covering both parts for the ~77 CU this month: item, CU, gate, what it decides.
- Dense: derivations, mechanisms, numbers, experiments; no general encouragement. Being boldly wrong with a clear
  prediction beats being vaguely safe.
