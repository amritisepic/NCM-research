# Research review and direction-setting brief (for an external model), 2026-10-27

You are an independent senior researcher reviewing a small-compute project on non-autoregressive text generation.
I want three things from you: a hostile evaluation of what we have done, an overview of the relevant field, and one
(at most two) research directions that could plausibly produce a significant, publishable result within our compute.
Above all I want **new ideas**. Go theoretically deep and think outside the box: start from first principles about
what a coarse-to-fine discrete code hierarchy is, what parallel generation can and cannot represent, and what our
anomalies are telling us, rather than from the list of known fixes. Borrow from wherever the right structure lives:
information theory, coding theory, statistical physics, numerical analysis, control, optimal transport, game theory,
linguistics, neuroscience. Ideas that look strange are welcome if you can say what they predict. Do not limit yourself
to tweaks of our current experiments; if the right move is to change the question, the representation or the
objective, say so and argue it. The one hard requirement: every direction you propose must eventually reach an
experiment we can actually run under the constraints in section 5. Unconstrained theory first, feasibility second.

Ground rules for your answer:
- Separate three kinds of statement and label them: **[ours]** (a number from this brief), **[source]** (a published
  paper or official code: give authors, year, venue or arXiv id, and the exact setting a number comes from), and
  **[speculation]** (your own reasoning). If you are not sure a paper exists or says what you remember, say so; do not
  invent citations or numbers.
- Challenge our framing. If a result we treat as established looks weak, say why and what would settle it.
- Prefer mechanisms over recipes: for each claim, say what quantity it predicts and how we would measure it.
- The repository is public if you can browse: github.com/amritisepic/NCM-research, branch `claude/ncm-research`
  (`STATE.md`, `log.md` with entries R001-R040, `docs/preregistration.md`). You do not need it; this brief is meant to
  be self-contained.

## 1. The object of study: Next-Concept Modeling (NCM)
NCM (Kumbong, Khare et al., arXiv 2610.14162; code HazyResearch/ncm) generates text coarse-to-fine:
- A frozen tokenizer (a VQ-VAE with **multi-scale residual quantization**, as in VAR / Infinity for images) maps each
  128-token window to a 12-level code hierarchy with level sizes 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128
  (641 codes), 4096 codes per level. Level k's codes quantize the residual the coarser levels leave. Each level's
  embeddings are linearly up-sampled to full length, passed through a kernel-3 residual convolution, and summed; the
  decoder maps the sum back to tokens (~99% token accuracy).
- A transformer generator predicts all of level k's codes **in parallel**, conditioned on the cumulative
  reconstruction of levels < k (area-down-sampled to level k's length). Level 0 is drawn from a unigram. Sampling
  takes 11 forward passes for 128 tokens, against 128 for an autoregressive (AR) model.
- Training uses **conditioning corruption**: a share of the coarser codes in the input (mean 0.5, linear from 0.75 at
  level 0 to 0.25 at level 10) is replaced by uniformly random codes, and labels stay the clean (canonical) codes. The
  paper reports this as its largest ablation effect (Gen-PPL ~48 -> ~19 at 10k steps).
- The paper's own TinyStories numbers [source]: AR 8.89 Gen-PPL at 128 NFE, NCM 15.41 at 12 NFE, data 9.67
  (Gen-PPL = GPT-2 large perplexity of unconditional samples, always read with unigram entropy, ~3.96-4.01 here).
  The AR baseline is data-matched, so it used ~4x fewer training FLOPs (641 vs 128 positions). NCM's case is speed,
  not quality. Quality keeps improving with more levels (K up to 32). Typical artifacts are adjacent duplicates
  ("little little egg"), the signature of sampling neighbouring positions independently. On Sudoku-hard NCM gets 86.1%
  at 12 NFE vs the best discrete-diffusion baseline 81.3% at 16.
- Released checkpoints we fine-tune: TinyStories generator (88.6M) + tokenizer (36.4M), Sudoku-hard (25.5M, 16 codes
  per level), WMT14 De-En (120M, 11 levels).

## 2. Our success bar (any one, met precisely, with >= 3 seeds, gain > 2x pooled seed SD, pre-registered)
- (a) Better than matched NCM at equal compute across the full evaluation suite with <= 1.5x NFE, or matched quality
  at <= 0.6x NFE.
- (b) Close >= half of the NCM -> AR quality gap with >= 3x fewer NFE than AR.
- (c) A replicated mechanistic finding about what the levels encode.
- (d) An adaptive-compute rule saving >= 25% NFE at < 1 point quality loss on Sudoku AND one text task.
Logical NFE alone does not count for (d): it needs batch-1 wall-clock and an attention-inclusive cost.

This bar is a floor, not a ceiling. Treat it as negotiable upward: if you see a route to a more ambitious result (for
example matching AR quality at a fraction of its NFE, a general theory that predicts when parallel coarse-to-fine
generation works and when it must fail, or a training principle that transfers beyond NCM), propose a sharper success
criterion for it, say why it would matter more, and estimate how reachable it is.

## 3. What we have measured (all [ours]; seed 0 unless stated; details in `log.md`)
Representation and where quality is lost
- Coarse codes behave like window bags of words. The token-straddling levels L8-L10 carry near-chance linearly
  readable content yet ~43% of the generator's NLL; L8-L11 hold 46.8% of clean NLL on 63% of code slots.
- **Code NLL vastly exceeds text entropy.** Codes are a deterministic function of text, so H(codes) <= H(text). With
  small models (8.4M tokenizer, 5.5M generators): NCM code NLL 1232.7 nats per window vs an AR token model's 173.6, so
  >= 86% of NCM's NLL is model excess over the entropy of its own targets (an inequality; it does not split excess into
  within-level dependence vs approximation / optimization error). Released model's clean code NLL: 861.8 nats/window.
- The released generator's one-step disagreement with the encoder's codes (T = 1) per level 0..11: 100, 98, 91, 81,
  67, 55, 48, 21, 45, 53, 55, 14 %. From L5 on, its sampled code is the encoder's nearest or second-nearest code by
  distance: fine-level "errors" are near-equivalent codes; coarse-level ones are different content.
- Decoder robustness: replacing all levels but the finest with uniform random codes keeps 42.6% of tokens; with the
  generator's own one-step samples, 98.0%. Any single coarse level can be fully randomized at a cost of < 1% of tokens.
- **Oracle-intervention curve:** supplying the real codes of levels 0..j and sampling the rest moves Gen-PPL 16.0 ->
  8.6 (all levels real except the finest). Levels 1-3 account for 48% of that gap, 4-5 for 39%, 6 for 5%, 7-11 for 7%.
  This is an intervention curve, not an attribution.
- Released TinyStories model under our protocol (16 x 256 fp32 samples): Gen-PPL 16.16 ± 0.11, entropy 3.97,
  MAUVE 0.827.

Wave 1: training-time corruption variants (released TinyStories model fine-tuned 1080-1800 steps at equal generator
FLOPs; batch 256, lr 5e-5)
| arm | training hierarchy | Gen-PPL | MAUVE | clean NLL |
|---|---|---|---|---|
| A0 | released, no training | 16.16 ± 0.11 | 0.827 | 861.8 |
| A1 | continue NCM's uniform corruption | 16.60 ± 0.08 | 0.881 | 844.0 |
| A2 | uniform corruption + labels re-quantized against the corrupted residual | 16.29 ± 0.09 | 0.915 | 749.3 |
| A3 | corruption by the model's own one-step samples, clean labels | 41.44 ± 0.43 | 0.496 | 885.7 |
| A4 | A3 + re-quantized labels | 36.81 ± 0.26 | 0.673 | 707.4 |
| A5 | A3 with two-pass (compounded) samples | 32.17 ± 0.18 | 0.543 | 913.9 |
| A6 | A1 + soft labels over the 16 nearest codes | 17.46 ± 0.10 | 0.832 | 888.9 |
- **Clean NLL and sample quality dissociate:** A2 / A4 gain 113 / 154 nats, 88-98% of it at L8-L11, yet A4 has the
  best NLL and the second-worst samples.
- **The on-policy collapse:** every on-policy arm collapsed, although the model's own samples were the mildest
  corruption in token terms. Pooled vocabulary is as human-like as the control's, story-level plans are intact, but
  local order breaks (bigram gain 2.13 -> 1.71 nats/token, adjacent duplicates 6.9 -> 17.7 per 1k tokens). Clean-NLL
  damage sits at L3-L6. It appears by step 300 and does not grow. Two independent code audits found no defect.
- Leading account (b'): corruption drawn from the model's own distribution is plausible, hence undetectable, so the
  model learns to discount its own coarse codes (scheduled sampling's improperness, cf. Huszár 2015, made visible by
  residual quantization). A matched CPU test on Sudoku supports the hedging step: plausible wrong codes raise
  predictive entropy on clean contexts by 74% (uniform wrong: 5%), raise clean NLL by 32 nats (vs 3), and cut reliance
  on the true L6 code by 76%. But there is no Sudoku collapse; solve rates fall ~5 points in both arms. Live
  competitors: lost denoising coverage (A3's realized corruption rate is lower), a parent mismatch (proposals sampled
  under the canonical L0 while ~75% of windows then carry a unigram L0), moving-policy feedback, and a bf16 policy
  forward that only the on-policy arms had.
- **W1d (eval-only, A0 / A1 / A2):** A2's Gen-PPL gain over A1 exists only at T = 1 (A2 - A1 = +0.22 ± 0.08 at T = 0.8,
  +0.05 ± 0.10 at 0.9, -0.33 ± 0.12 at 1.0; entropies equal within 0.005). Given real levels 0..3, A2 renders the finer
  levels worse than A1 (+0.42). In the oracle protocol A1 trails the released model by 0.69 with nothing supplied and
  by 0.00 once levels 0..3 are real: the continued-training damage is in sampling the coarse levels.

Decoding: fixed-point (Jacobi) iteration across levels
- Training-free exact Jacobi (all levels updated in parallel each pass, iterate to the sequential decode's fixed
  point): Sudoku-hard 6.41 passes for identical output (-46.6% logical NFE, 1.87x by position count); with a verifier
  stop, 2.37 passes and 90.3% solved (vs 86.15% sequential). WMT14: 11.07 passes vs 11 (no saving), final output first
  reached at pass 9.58 on average. TinyStories: no saving.
- WMT14 consistency fine-tune (J2, CLLM-style, the current model as its own teacher on its Jacobi trajectories, 2000
  steps): val 9.36 passes, BLEU 18.53 (screen needed <= 9.0 passes and BLEU >= 22.55); test 8.89 passes at -6.6 BLEU.
  It found a degenerate solution: 16% of test sentences decode to one of two input-independent strings in ~3 passes; on
  the rest it saves 9% of passes. The released model capped at 9 passes already loses only 0.10 BLEU (18% fewer passes),
  at 8 passes 1.11 BLEU: that is the bar a learned rule must beat.

Other results
- Sudoku: success falls with propagation depth. Repair levels built from the finest Sudoku codebook fix <= 63% of
  failures even with an oracle. A recurrent NCM (state carried across levels, trained from scratch at 5.7M params)
  did not beat NCM.
- A small-model test of within-level total correlation (how much an AR order inside a level would help) was
  inconclusive (probe failure).

## 4. Killed, demoted and in flight
- Killed / demoted: on-policy corruption with canonical labels (destructive at tier L); soft equivalence-class labels
  (A6, +0.85 Gen-PPL); re-quantized labels (A2, gain only at T = 1); carry across levels on Sudoku from scratch;
  training-free Jacobi on text; token-aligned level schedules; level-skipping early exit on Sudoku.
- Considered in our candidate lists but not run: a parallelizability-regularized tokenizer; multigrid-style red-black
  relaxation within a level; the encoder as a fixed-point verifier; contextual / summary-supervised coarse levels;
  importance-resampled levels with a within-level AR energy; lifting-scheme (wavelet) tokenizers; successive-
  cancellation list decoding over levels (from polar codes); global plan tokens instead of window pyramids; exact AR
  sampling from NCM drafts via coupling and parallel verification; JEPA-style continuous coarse latents with a flow head.
- In flight: **J2-F** (WMT14 consistency with a frozen released-model teacher, ~6.5 CU, running now) and **W2M**
  (eight 600-step tier-L arms, ~27 CU, after a code audit): uniform, current policy, frozen plausible-wrong (canonical
  excluded), uniform-wrong, plausible-wrong + truthful corruption flags (a zero-initialised learned input vector),
  plausible-wrong + shuffled flags, used-parent proposals, uniform at A3's realized rates. Plus eval-only measures:
  coarse-context dependence (KL after perturbing one coarse code), a probe for whether replaced positions are
  detectable from hidden states, and flag use. Predictions are registered; one seed.
- Held: a tier-M from-scratch NCM vs FLOP-matched AR (d512 x 6, ~12 CU per seed pair); three Sudoku arms (~7.5 CU).

## 5. Constraints (hard)
- **Compute:** about 90 Colab compute units left after J2-F and W2M, i.e. roughly 17 A100-40GB hours. A 600-step
  tier-L fine-tune of the 88.6M model costs ~3-4 CU; the final 16 x 256 evaluation adds ~0.2 CU. Any single run
  > 15 CU needs explicit approval. Replication needs 3 seeds per arm, so a two-arm confirmation at tier L costs
  roughly 6 x the single-run price. CPU: a laptop and a 4-core cloud container (2 h jobs), fine for Sudoku-scale or
  eval-only work.
- **Models:** we fine-tune the released checkpoints; training a new 36M tokenizer or a new 88.6M generator from scratch
  is out of budget. Small from-scratch models (<= ~25M generator, <= ~10M tokenizer) are affordable but our earlier
  ones were far from the released models' quality.
- **Process:** every comparison is pre-registered before the run (predictions, the result that would overturn them,
  the reading), one axis at a time, matched data order and seeds across arms, fp32 evaluation. Code changes get an
  independent review and a deep audit before any GPU run. Null results are reported.
- Timeline: weeks, not months.

## 6. What I want from you
1. **Hostile evaluation.** Which of our conclusions are weakest, and what is the cheapest test that would overturn
   each? Is anything we killed killed for the wrong reason? Is the success bar itself well chosen for this compute,
   and is any criterion much more reachable than the others?
2. **Deep theory.** This is the most important part; spend most of your effort here. Give your best account of
   (i) why on-policy corruption collapses NCM while uniform corruption helps it, (ii) why clean code NLL and sample
   quality dissociate, (iii) what the gap between code NLL (~860 nats/window) and text entropy (<= ~174 nats/window)
   implies for how NCM should be trained or decoded, and (iv) what fundamentally limits a model that emits many
   positions per forward pass (how much dependence one parallel step can resolve, and whether the hierarchy moves that
   dependence or only hides it). Go past restating known results: derive, sketch a proof or a toy model, or construct
   a counterexample where you can. Candidate lenses, not a checklist: proper scoring rules and exposure bias, denoising
   and score matching, rate-distortion and the many-to-one map from codes to text, symmetry and gauge freedom in the
   codebook, mean-field vs Bethe-style approximations to the joint, multigrid and fixed-point iteration, belief
   propagation and decoding of concatenated codes, renormalization-group views of multi-scale structure, consistency
   models. For each account, state the measurable prediction that distinguishes it from the others, and say which of
   our in-flight W2M arms would test it.
3. **Field overview (2023-2026).** Coarse-to-fine and multi-scale generation (VAR, Infinity and bitwise
   self-correction, other residual-quantized text models), discrete diffusion and masked models for text (MDLM, SEDD,
   DUO and successors; remasking and self-correction samplers), exposure-bias remedies, Jacobi / consistency decoding
   (CLLMs and later), parallel and speculative decoding with exact verification, and learned or adaptive stopping. For
   each line: what it establishes at what scale, and what it implies for NCM. Flag anything that already does what we
   might propose.
4. **New ideas, then synthesis.** First, brainstorm widely: list 8-12 genuinely new ideas (at least half should not
   be a variant of anything in sections 3-4 or the field overview), one or two lines each, with the theoretical reason
   it could work and a rough probability. Include at least two high-risk ideas you think most researchers would
   dismiss, and say why they might be wrong to. Then pick **one or two breakthrough directions** and, for each, give:
   - the core idea in two sentences, the mechanism, and why it is not incremental;
   - which success criterion it targets: (a)-(d), or a more ambitious one you define (section 2), with the size of
     effect you expect and your probability of success;
   - a kill experiment of <= 5 CU (or CPU-only) that tells us within days whether to continue, with its pre-registrable
     prediction and the result that would kill it;
   - the full plan to a replicated claim within the remaining ~90 CU (arms, seeds, metrics, CU per run);
   - the strongest objection a reviewer would raise and how the design answers it;
   - how it uses or reinterprets what we already have (the oracle curve, the collapse, Jacobi on Sudoku, J2's failure).
   Rank the directions. If you think the best use of the remaining compute is to consolidate one mechanistic finding
   (criterion c) rather than chase a method, argue that case explicitly.
5. **What to stop doing.** Name the parts of our current queue (J2-F, W2M arms, the held tier-M and Sudoku runs) you
   would cut or change, and why.

Keep the answer dense: derivations, mechanisms, numbers, experiments. No general encouragement. Being boldly wrong
with a clear prediction beats being vaguely safe.
