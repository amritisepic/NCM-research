# Research review, round 4 (for an external model), 2026-10-28

You reviewed this project three times (`docs/research_review_2026-10-26.md`, `docs/research_review_2026-10-27.md`, and
round 3, triaged in R055 / `docs/plan_2026-10-27.md` section 10). Everything round 3 queued has now reported. Two results
meet a pre-registered criterion-(c) bar and have each survived a deep code audit; a third (W2M-GR, section 3) decides
whether one of them is a finding about the levels.

This round asks for **a wide, unconstrained brainstorm**, and for a hard look at where we stand against the field. It has
four parts, to be done in this order:

- **Part A — Frontier map (short):** where our findings sit relative to 2024-2026 work; what is new, what is a
  rediscovery, and which assumption of the field our results contradict.
- **Part B — Divergent brainstorm (most of your effort):** many ideas, generated without regard to our budget, from the
  anomalies in section 3. Rules in section 7.
- **Part C — Convergence:** pick two or three directions, then (and only then) fit them to our constraints (section 9).
- **Part D — Hostile review of the two claims, and the paper (short).**

Ground rules (as before): label every statement **[ours]** (a number from this brief), **[source]** (a published paper
or official code: authors, year, venue or arXiv id, and the setting a number comes from) or **[speculation]**. If you
are unsure a paper exists or says what you remember, say so; do not invent citations or numbers. Challenge our framing.
For every idea, say what quantity it predicts and how we would measure it. The repository is public
(github.com/amritisepic/NCM-research, branch `claude/ncm-research`: `STATE.md`, `log.md` R001-R065,
`docs/preregistration.md`, `docs/reviews/`); this brief is self-contained.

## 1. The object of study
Next-Concept Modeling (NCM; Kumbong, Khare et al., arXiv 2610.14162, October 2026; code HazyResearch/ncm) generates text
coarse-to-fine. A frozen tokenizer (a VQ-VAE with multi-scale residual quantization, as in VAR / Infinity) maps each
128-token window to 12 levels of codes (sizes 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128; 4096 codes per level);
level k quantizes the residual the coarser levels leave; each level's embeddings are up-sampled, passed through a
kernel-3 residual convolution and summed; the decoder maps the sum to tokens (~99% token accuracy). A transformer
generator predicts all of level k's codes in parallel and independently, given the cumulative reconstruction of levels
< k; level 0 comes from a unigram; 11 forward passes per window vs 128 for an AR model. Training uses conditioning
corruption: a share of the coarser input codes (mean 0.5, 0.75 at level 0 linearly to 0.25 at level 10) is replaced by
uniform random codes; the targets (the next level's canonical codes) are unchanged, so the model is never asked to
repair the corrupted codes themselves, only to predict finer codes in their presence. The paper reports this
corruption as its largest ablation effect (Gen-PPL ~48 -> ~19 at 10k steps) [source]. Paper numbers [source]:
TinyStories AR 8.89 Gen-PPL at 128 NFE, NCM 15.41 at 12 NFE, data 9.67; Sudoku-hard NCM 86.1% at 12 NFE vs 81.3% for the
best discrete-diffusion baseline at 16. We work with the released checkpoints: TinyStories generator 88.6M + tokenizer
36.4M, Sudoku-hard 25.5M, WMT14 De-En 120M (11 levels). Our searches found no follow-up work on NCM yet.

## 2. Success bar (a floor, not a ceiling)
Any one, met precisely, pre-registered; a training-arm claim needs 3 fresh seeds and a gain > 2x the pooled seed SD.
- (a) Better than matched NCM at equal compute across the full suite with <= 1.5x NFE, or matched quality at <= 0.6x NFE.
- (b) Close >= half of the NCM -> AR gap with >= 3x fewer NFE than AR.
- (c) A replicated mechanistic finding about what the levels encode.
- (d) An adaptive-compute rule saving >= 25% NFE at < 1 point quality loss on Sudoku AND a text task, with a >= 25%
  batch-1 wall-clock saving.

Status: (c) has two candidates (section 4); (a), (b) and (d) have none ((d) is closed on text for Jacobi decoding).
If you see a route to a more ambitious result, for example a principle that predicts which training-context
distributions help or destroy any iterative / multi-scale generator (masked diffusion LMs, VAR-style image models, NCM),
propose a sharper criterion, say why it would matter more, and estimate how reachable it is.

## 3. The anomalies (all [ours]; seed 0 unless stated). These are your raw material.
**Training contexts**
1. **Plausible-wrong vs uniform-wrong (W2C, seeds 1-3, replicated).** Continuing the released TinyStories model for 600
   steps with 25% clean rows and W1's per-level rates, one mask stream so both arms corrupt the same slots: replacing the
   masked coarse codes by draws from the frozen released model (given the canonical coarser levels, canonical code
   excluded) gives Gen-PPL 40.6 vs 21.7 for uniform-wrong replacement (GPT-2 large; +18.4 / +19.5 / +18.8 per seed,
   pooled seed SD 0.56; TinyStories-33M +29.8; MAUVE 0.45 vs 0.85; adjacent duplicates +12.3 per 1k), while clean
   teacher-forced code NLL differs by only +4.0 nats/window (0.53%, the same sign every seed). Every one of 16 sample groups
   per seed favours uniform-wrong.
2. **The model's own current samples do the same (W2M):** M1 (current policy's one-step samples) 42.50, M2 (frozen
   model) 42.62, M6 (M1 with proposals under the level-0 code actually used) 42.18, uniform-wrong 21.69, uniform at the
   on-policy arm's lower realized rates 22.07.
3. **Damage does not scale with the number of wrong slots:** M1's realized disagreement on corrupted rows from L2 on is
   far lower than M2's (L5 0.27 vs 0.50, L7 0.08 vs 0.40; equal at L0-L1), yet their damage is equal.
4. **Provenance helps, partially, and acts as a switch:** truthful row-level flags (M4) halve the damage (rescue 0.53);
   M4 uses them (-35 nats/window); a constant flag f = 1 on every input turns M4 into M2; f = 0 is its best setting.
5. **The damaged models treat generated contexts differently:** M2 and M3's predictive distributions (Jensen-Shannon,
   levels 2-5) differ 0.0126 nats on real contexts and 5.5-6.9x more on generated ones (W2M-G (i)).
6. **Where the damage forms (W2M-G (ii), crossed replay, prefix model samples levels 1..c-1, suffix model c..11; shares
   of the M2-M3 log-loss contrast):** M2's prefix through L1 carries 0.009; through L3 0.220; through L5 0.511; M2's
   suffix after L5 alone carries 0.061; the interaction is 0.47 / 0.43 at cuts 4 / 6. Damage forms at levels 2-5 and is
   amplified by M2's own later levels. **W2M-GR at seeds 1-3 (registered thresholds): [TO FILL FROM R065 BEFORE
   SENDING].**
7. **The channel prediction that was right and irrelevant:** M2's level 1 leans 41% less on the root code (as a
   Bayesian channel model of the corruption predicts), yet the L1 prefix carries ~1% of the damage.
8. **Clean rows:** without the 25% clean rows the uniform-wrong control returns to the released model's quality (16.50 vs
   21.69 Gen-PPL; released 16.16) while its clean code NLL gets 106 nats/window worse, 78% of it at L8-L11 (M3-0).
   Uniform corruption is essential to NCM [source], yet some uncorrupted training makes generation worse.
9. **Likelihood and quality come apart everywhere:** continued teacher forcing lowers WMT14 code NLL by 43.5
   nats/sentence and costs 0.73 BLEU (92% of it n-gram precision); labels re-quantized against the corrupted residual
   lower TinyStories code NLL by ~95 nats and help Gen-PPL only at T = 1 (they hurt at T = 0.8); continued uniform
   training (A1) trails the released model by 0.44-0.69 Gen-PPL with nothing supplied and by 0.00 once real levels 0-3
   are supplied.
10. **The first on-policy collapse (wave 1, no clean rows):** corruption by the model's own one-step samples with
   canonical labels: Gen-PPL 41.4 vs 16.6; it keeps vocabulary and story plans and breaks local order (bigram gain 2.13
   -> 1.71 nats/token; adjacent duplicates 6.9 -> 17.7 per 1k). A CPU Sudoku analogue: plausible wrong codes raise
   predictive entropy on clean contexts by 74% (uniform wrong: 5%) and clean NLL by 32 nats (vs 3), but no collapse.

**Representation**
11. **The fine levels mostly encode the tokenizer's arbitrary choice (E1 / E1b, three released models, registered,
   audited).** With N_k = canonical code NLL of levels >= k given canonical levels < k and rho_k = the probability that the
   model's own samples of levels >= k decode to exactly the item's text, the share (N_8 + log rho_8) / N_8 is at least
   0.894 (TinyStories, mean N_8 404 nats per window), 0.859 (WMT14, 131 nats per sentence) and 0.993 (Sudoku, 32 nats
   per puzzle) (lower 95% bounds). N_8 is 46% / 46% / 74% of all canonical code NLL. Some text items have 0 exact hits in
   32,768 samples.
12. **Per-level behaviour of the released generator:** one-step disagreement with the encoder per level 0..11 (T = 1):
   100, 98, 91, 81, 67, 55, 48, **21**, 45, 53, 55, **14**%; from L5 on its sampled code is the encoder's nearest or
   second-nearest code. Coarse codes behave like window-level bags of words; the token-straddling levels L8-L10 carry
   near-chance linearly readable content.
13. **Oracle-intervention curve:** supplying the real codes of levels 0..j and sampling the rest moves Gen-PPL 16.0 ->
   8.6; levels 1-3 account for 48% of that, 4-5 for 39%, 6 for 5%, 7-11 for 7%. The NCM -> AR gap (15.4 vs 8.9) is almost
   entirely a coarse-level sampling gap.
14. **Decoder robustness:** replacing every level but the finest by the generator's own one-step samples keeps 98.0% of
   tokens; by uniform codes 42.6%. On Sudoku, decoding with the finer levels left out solves 0% of boards before the
   last level (cell accuracy 76.0% at L9, 98.4% at L11).
15. **Little within-level dependence:** adding the canonical left neighbour to the factorized head gains 6.2
   nats/window over levels 1-5 (1.7% of those levels' NLL).

**Decoding**
16. **Jacobi (fixed-point) iteration across levels:** Sudoku exact convergence in 6.41 passes (-47%); with a verifier
   stop 2.37 passes and 90.3% solved vs 86.15% sequential (0.44x batch-1 latency); text: no saving. Consistency training
   with the current model as its own teacher collapses 16% of WMT14 test sentences onto two input-independent strings;
   with a frozen teacher, early-pass outputs improve a lot but exact passes fall only 5.6%, and at batch 1 the best text
   variant is 1.41x slower than sequential decoding.

## 4. The two claim candidates
**C-E1** (drafted, awaiting the user): in each of the three released NCM models, given the canonical levels 0-7, at
least 80% of the canonical code NLL of levels 8 and finer is spent choosing among code paths that decode to the same
output (anomaly 11).
**C-W2C** (wording after audit 12; not claimed): under one continuation protocol, plausible-wrong replacement causes
additional generation degradation relative to uniform-wrong replacement in the same slots while canonical
teacher-forced NLL barely distinguishes them (anomaly 1); if W2M-GR replicates, the damage forms at levels 2-5 and is
amplified by the damaged model's own later levels. Audit 12's limits: comparative only (the uniform-wrong arm is itself
5.5 Gen-PPL above the released model, because of the clean rows); no temperature-, frequency- or length-matched control;
M2 costs one extra forward per step; one model, domain and protocol.

## 5. Part A — Frontier map
Quick searches (not a novelty clearance) suggest:
- **Training corruptions in discrete diffusion LMs.** Recent work argues that uniform random replacements are a poor
  training signal because inference errors are plausible, model-like mistakes, and builds on that (e.g. LLaDA-2.1-style
  editing and "targeted remasking", arXiv 2605.26436; corrective diffusion LMs, arXiv 2512.15596). GIDD (arXiv
  2503.04482) finds that mixing uniform noise into masking enables self-correction. SUNDAE (Savinov et al. 2022) trains
  on the model's own unrolled samples. **Anomalies 1-3 point the other way under NCM's setup**, where the corrupted
  codes are context and the targets are other (finer) codes. Is this a contradiction of the field's premise, a
  difference of setup (correct-the-corrupted-token vs predict-finer-codes-despite-it), or both?
- **Non-canonical tokenizations.** For subword LMs the marginal over tokenizations is mostly close to the canonical
  probability (Geh et al., arXiv 2408.08541; Cao and Rimell 2021; Pohl et al., arXiv 2510.20208). Anomaly 11 says that
  in NCM's code hierarchy the canonical path is exponentially unlikely relative to the marginal at the fine levels
  (hundreds of nats). Is this known for residual-VQ / VAR-style tokenizers?
- **Exposure bias and teacher-forced metrics** (scheduled sampling, Bengio et al. 2015; its improperness, Huszár 2015;
  likelihood vs sample-quality dissociations): which of our results are expected consequences, and which are not?

Correct and extend this map. For each finding in section 3: **new / partly known / known**, with the closest work and
what exactly differs. Then answer directly: **is anything here past the public frontier, and what single result, if
added, would make it so?**

## 6. What not to repeat (unless you argue it was killed for the wrong reason)
Killed or demoted: on-policy corruption with canonical labels as a method; soft labels; re-quantized labels; carry across
levels (Sudoku, from scratch); training-free text Jacobi; token-aligned schedules; dropping the fine levels (anomaly 14);
consistency training for text Jacobi (moving and frozen teacher); the marginal-preserving coupled sampler (anomaly 15);
L1 target reassignment (anomaly 7); tier-M from-scratch NCM vs AR; Sudoku training arms; constant-flag routing.
Considered, not run: a parallelizability-regularized tokenizer; red-black relaxation within a level; the encoder as a
fixed-point verifier; contextual / summary-supervised coarse levels; importance-resampled levels with a within-level AR
energy; lifting-scheme tokenizers; successive-cancellation list decoding over levels; global plan tokens; exact AR
sampling from NCM drafts; JEPA-style continuous coarse latents; subtractive dither; bag-then-permutation; irreversibility-
gated reopening; decoder-aware geometry; damage-matched donor corruptions; protecting coarse predictions from fine
gradients; intervention-response maps; training on equivalent code paths (marginal likelihood over code paths); filling
the fine levels cheaply instead of sampling them. (You may build on any of these; say so when you do.)

## 7. Part B — Divergent brainstorm (do this before reading section 9)
Generate **at least 25 ideas**. Do not evaluate feasibility, cost or our budget in this part; that comes in Part C.
Quantity and range first; most ideas will be wrong, and that is expected.

**How to generate them.** Use each of these moves at least twice, and label each idea with the move(s) it used:
1. **Explain an anomaly.** Pick one or more items of section 3 and propose a mechanism that would produce it; then ask
   what else that mechanism must imply, and turn that implication into an idea (a method, a measurement, a theory).
   The strongest ideas usually explain two anomalies that look unrelated (e.g. 1 and 11; 3 and 6; 8 and 9; 13 and 16).
2. **Invert an assumption.** NCM assumes, among other things: coarse-to-fine order; a fixed, frozen tokenizer; uniform
   corruption of context with canonical targets; independent codes within a level; one sample per level; likelihood as
   the training objective; the same model for every level; text windows of fixed length. Negate one and follow it.
3. **Import a mechanism from another field** where the right structure already lives: information and coding theory
   (successive refinement, channel coding, list decoding, rate-distortion), statistical decision theory and robust
   statistics, control (closed-loop stability, disturbance amplification, observers), statistical physics
   (renormalization, annealing, replica methods), numerical analysis (multigrid, preconditioning, defect correction),
   optimal transport, game theory and mechanism design, linguistics and psycholinguistics, neuroscience (predictive
   coding, precision weighting), immunology (self / non-self discrimination), evolution (error thresholds, mutational
   robustness), economics (adverse selection: plausible-wrong signals are the "lemons"). Map the import to a concrete
   loss, sampler, representation or measurement.
4. **Change the question.** Propose a question we should be asking instead of "how do we make NCM better", whose
   answer would matter beyond NCM (masked diffusion LMs, VAR-style image models, multi-scale tokenizers, speculative or
   parallel decoding).
5. **Push to the extreme.** Take a quantity to 0 or infinity (corruption rate, number of levels, codebook size,
   clean-row share, proposal temperature, number of samples per level) and say what must happen there and what that
   reveals in between.
6. **Combine two findings into a product.** E.g. E1's redundancy x W2C's sensitivity; the oracle curve x Jacobi's
   Sudoku success; provenance flags x the regime switch.

**Quotas.** At least 8 ideas that are not variants of anything in sections 3, 6 or your earlier reviews. At least 4
high-risk ideas most researchers would dismiss, each with why they might be wrong to dismiss it. At least 3 ideas that
are theory or measurement rather than methods. At least 3 ideas whose payoff would transfer beyond NCM. At least 2 ideas
that would show something in the field's current practice (section 5) is wrong.

**For each idea, 3-5 lines:** the idea; the mechanism (why it could work); the anomalies it uses or explains; one
falsifiable prediction with a number or a sign; your probability that the prediction holds; what it would change if true.

## 8. Part C — Convergence
1. **Cluster** the ideas; name the 3-5 underlying themes.
2. **Rank** all ideas by (surprise if true) x (significance beyond NCM) x (probability) x (testability), in a table.
3. **Deep dive on the top two or three.** For each:
   - the core idea in two sentences, the mechanism derived rather than asserted, and why it is not incremental;
   - the criterion it targets, (a)-(d) or a sharper one you define (section 2), the effect size you expect and your
     probability of success;
   - a **kill experiment** that fits section 9 (<= 5 CU, eval-only or CPU preferred), with a pre-registrable prediction
     and the result that would kill it;
   - the full plan to a replicated claim within the budget, and separately what it would need with a raised budget (how
     much, and why it would be worth it);
   - the strongest objection a reviewer would raise and how the design answers it;
   - how it uses what we already have (checkpoints of W1, W2M, W2C at seeds 0-3; the E1 machinery; the crossed-replay
     sampler; J2 / J2-F; the Sudoku analogue).
4. **Then, against the best direction, argue the opposite:** is consolidating C-E1 and C-W2C and writing them up a
   better use of the time than any new direction? Say which you would do and why.

## 9. Constraints (for Part C only)
- **Compute:** ~96 CU after W2M-GR (Colab; A100-40GB ~5.3 CU/h, L4 1.71 CU/h; our estimates now hold within ~25%). A
  600-step tier-L fine-tune of the 88.6M model costs ~2.7 CU (uniform) to ~4.6 CU (with proposal forwards) per seed;
  eval-only passes over the released models ~1-2 CU. Any single run > 15 CU needs the user's approval. A direction that
  needs more must say how much more and why it is worth it (the user has raised the budget before).
- **Hardware besides Colab:** a laptop RTX 4050 (6 GB) for evaluation and small jobs; CPU.
- **Models:** fine-tune or evaluate the released checkpoints; no new 36M tokenizer or 88.6M generator from scratch;
  small from-scratch models (<= ~25M generator, <= ~10M tokenizer) are affordable but ours were far below the released
  models' quality.
- **Process:** every comparison pre-registered (prediction, overturning result, reading), one axis at a time, matched
  data order and seeds, fp32 evaluation, Codex review plus a deep audit before any GPU run whose code changed (~1 day of
  latency per new piece of code). Null results are reported.
- **Timeline:** about three weeks to a write-up; realistically one or two new pieces of code.

## 10. Part D — Hostile review and the paper (short)
1. For C-E1 and C-W2C: a verdict (claim as worded / narrow it / hold), the strongest single sentence a hostile reviewer
   would write, our best answer from the data, and the cheapest test that would overturn the claim.
2. For C-W2C specifically: which confound first (plausibility vs any non-uniform wrong distribution; displacement size
   in embedding space; target conflict; the clean rows; unequal compute; the scorers), and does W2M / W2M-G / W2M-GR /
   M3-0 already answer it?
3. The paper: its one-sentence contribution as things stand; the same sentence if your top direction succeeds; the
   three figures that carry it; what to cut.

## 11. Output format
- Part A: at most 15 lines plus the per-finding novelty table.
- Part B: the numbered idea list (>= 25), each labelled with its move(s), anomalies used, prediction and probability.
- Part C: the theme list, the ranking table, two or three deep dives, then the consolidate-vs-explore argument.
- Part D: at most 20 lines.
- End with one queue for the ~96 CU: item, CU, gate, what it decides.
- Dense: mechanisms, numbers, experiments; no general encouragement. Being boldly wrong with a clear prediction beats
  being vaguely safe.
