# Phase 1b — goals re-evaluated after reading the paper (2026-10-23)

## What the paper changes (facts from arXiv 2610.14162 v1, now in docs/)
1. **AR beats NCM on TinyStories by a wide margin.** Table 8: AR (sample, 128 NFE) Gen-PPL **8.89**
   (H 4.01); NCM (12 NFE) **15.41** (H 3.98); data 9.67 (H 3.96). The AR baseline uses the same
   architecture and the same number of updates and batch size as the NCM generator, i.e. it is
   data-matched and therefore used ~4x *fewer* training FLOPs (our R-analysis: 641 vs 128 positions).
   NCM's case is speed (NFE, wall-clock), not quality. Diffusion baselines (MDLM 17.9, DUO 19.6) are worse.
2. **Exposure bias is NCM's biggest measured lever.** Appendix F.3: training with uniformly random
   replacement of conditioning codes (clean labels kept) moves Gen-PPL from ~48 to ~19 at 10k steps.
   The authors: "Corruption based on the model's actual errors may better match inference conditions;
   we leave this to future work." Our pilot NCM did not have this; the matched baseline must.
3. **Quality is serial-depth-limited.** Appendix F.1.1: more levels (K = 4, 8, 16, 32) keeps lowering
   Gen-PPL, still improving at 32. Consistent with our Sudoku finding (success falls with propagation depth).
4. **Artifacts are adjacent duplicates** ("little little egg", "Sam and Sam", "school school"; Appendix J)
   — the signature of sampling neighbouring positions independently.
5. Stated future work: input-dependent schedules (adaptive compute), end-to-end tokenizer, fixing
   artifacts at level transitions.
6. Sudoku setup = S-FLM/FMLM+ (40/35/30 clues). Best baseline FMLM+ 81.3% hard at 16 NFE; NCM 86.1% at 12.

## Re-evaluated targets (ambitious, mapped to the success bar)
- **T1 (criterion b, ideally better):** on TinyStories, close most of the NCM→AR gap (15.4 → ≤ 11 in
  the paper's units; our matched-scale equivalents) at ≤ 20 NFE.
- **T2 (criterion a):** NCM-level quality at ≤ 0.6x NFE (≤ 7 passes).
- **T3 (criteria a + d, Sudoku):** hard ≥ 97% at ≤ 12 average NFE, with per-puzzle adaptive compute.
- **T4 (criterion c):** a replicated mechanistic account of what NCM levels encode and why it fails
  (bag-of-words coarse codes; serial-depth law; where the NCM−AR gap lives).

## New, riskier, higher-payoff bets
**B1. Self-correcting NCM: on-policy errors + re-quantized targets.** NCM's corruption teaches the model
to *ignore* noisy coarse codes (labels stay clean). Residual quantization lets us compute the *correct
repair*: given wrong codes at levels < k, the expert target for level k is the quantization of the true
residual z − ẑ_wrong. Training recipe (one extra no-grad pass): (1) teacher-forced pass, sample each
level's codes from the model (its real one-step errors); (2) build a corrupted hierarchy mixing sampled
and true codes; (3) recompute every later level's target by re-quantizing the true residual;
(4) train on corrupted inputs → repaired targets. Later levels learn to fix earlier mistakes instead of
compounding them. Precedent: image VAR "bitwise self-correction" (Infinity) with random bit flips and
re-quantized targets; on-policy errors for text NCM are new. Targets: T1, T2, adjacent duplicates.
Kill: small-scale TinyStories, 4 arms at equal compute (no corruption / NCM uniform+clean labels /
uniform+re-quantized / on-policy+re-quantized); B1 must beat NCM's recipe by > 2 seed-SD on Gen-PPL at
matched entropy and on MAUVE. Cost ~3 CU (or laptop GPU). Prior 45%. **Top priority.**

**B2. Fixed-point (Jacobi) decoding across levels, then consistency training.** Run all levels in one
parallel pass from a guessed hierarchy, update every level from the current earlier levels, repeat. With
fixed sampling noise this converges *exactly* to sequential decoding (the sequential solution is the
unique fixed point; level k is guaranteed correct after k+1 iterations), so stopping at the first
unchanged iteration is an exact, training-free adaptive-compute rule. If it converges in < K passes it
already meets criterion (d) on Sudoku with zero quality loss. Then train for fast convergence (B1's
robustness to wrong coarse codes should help; CLLM-style consistency loss). Targets T2, T3. Kill:
released Sudoku-hard checkpoint, greedy: mean iterations to fixed point must be ≤ 0.75 K. Cost ~0.3
CPU-h (no training). Prior 25% training-free, 20% for T2 after training. Trades FLOPs for latency.

**B3. Tokenizers whose coarse codes explain the fine levels' dependencies (CorEx / RSMI).** (outside
ML: real-space mutual-information renormalization group, Koch-Janusz & Ringel 2018; total-correlation
explanation, Ver Steeg & Galstyan 2014.) In renormalization, a good coarse variable captures what is
correlated with *distant* parts of the system, so that, given it, local details are nearly independent.
Loss: InfoNCE between each coarse code and distant windows (not its neighbours), plus the factorized
prior's NLL as a total-correlation proxy, on a contextual encoder. If it works, within-level sampling
becomes nearly exact (lower TC) *and* coarse levels become semantic (T1, T4). Kill: TC from the P1 probe
pair must drop ≥ 30% at ≥ 99% reconstruction, and the bag-of-words invariance must break. Cost ~4 CU.
Prior 15%. **High-risk bet.**

**B4. Global plan tokens instead of window pyramids.** Replace window averages by a 1D set of latent
tokens produced by cross-attention over the whole text, with nested dropout over the token order
(FlexTok-style, text). Coarse levels = global plan (characters, setting, events), not local bags of
words. Targets T4 and possibly T1. Kill: ≥ 99.9% reconstruction with ≤ 2x NCM's code count, and probes
showing global attributes decodable from the first levels. Cost ~4 CU. Prior 15%.

**B5. Exact AR sampling from NCM drafts (maximal coupling + parallel verification).** Fallback route to
T1: NCM drafts 128 tokens in 11 passes; an AR model scores the whole draft in one pass; accept the
longest coupled prefix, resample the first mismatch, repeat. Output distribution = AR exactly; passes =
11 + verification rounds. Meets criterion (b) only if the mean accepted run is ≥ ~4 tokens. A hybrid, not
an NCM improvement — reported separately. Kill: measured acceptance on released NCM drafts vs a small
TinyStories AR. Cost ~1 CPU-h + an AR model. Prior 30%.

**B6. Sudoku: Recurrent NCM + weight-tied full-resolution repair loop with halting.** Extends C4
(running on Colab): after the hierarchy, repeat a full-resolution repair pass trained with B1's
re-quantized targets, halting on confidence (decoder min-confidence flags failures with AUC 0.87) or a
valid board. Targets T3. Kill: depends on C4's result. Cost ~3 CU. Prior 30%.

**B7. Equivalence-class (soft) targets for the generator.** New measurement (R008): the released
generator's teacher-forced code NLL is 870 nats per 128-token window (6.8 nats/token; small TinyStories
AR models typically reach ~1-1.5 nats/token), and 43% of it sits in the three levels whose windows
straddle token boundaries (76, 92, 108 codes: 1.1-1.5 nats/code), which raise truncated-decode accuracy
only from 94% to 99%. Codes are a deterministic function of the text, so most of this is not text
entropy: it is the cost of predicting *which* of several near-equivalent codes the encoder happened to
pick ("nuisance entropy"), plus within-level dependence. Generation never needs the canonical code, only
one that decodes to good text. Train with soft targets over near-equivalent codes (softmax of negative
residual-to-codebook distance, temperature τ) and/or stochastic encoding (the upstream quantizer already
has `set_stochastic_encode`), so capacity goes to distinctions that change the text. Pairs naturally with
B1 (later levels repair sloppy intermediate choices). Kill: Gen-PPL/MAUVE at matched entropy and compute
vs hard targets. Cost ~2 CU. Prior 30%. Side test (cheap): a token-aligned schedule (only integer
tokens-per-code ratios) to see whether the straddling levels are what makes codes unpredictable.

## Dropped or demoted
- C7 (importance resampling with an AR energy): dominated by B1/B5 for the same cost. Demoted.
- C8 (lifting tokenizer): low prior and the paper shows more levels help; going to fewer, denser codes
  works against that. Dropped.
- C11 (JEPA coarse levels): unchanged, deprioritized.

## Order of work
1. Cheap, decisive, CPU now: B2 training-free on Sudoku; teacher-forced-coarse-levels sampling on the
   released TinyStories model (how much quality is lost at coarse vs fine levels); duplicate-rate metric.
2. Add NCM's conditioning corruption to our generator; implement B1 (both arms share code).
3. B1 kill test on GPU (Colab or the laptop). Then B6 and B3 depending on C4/P1 results.
