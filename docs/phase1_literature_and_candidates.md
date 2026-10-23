# Phase 1 — literature check and candidate architectures

_Status: draft v1 (2026-10-23). Priors will be updated after the Phase-0 TC measurement (R00x)._

## Plain-language glossary
- **NFE** (number of function evaluations): sequential forward passes of the generator per sample.
  AR on 128 tokens = 128; NCM on TinyStories = 11 (12 levels, level 0 drawn from a frequency table).
- **Total correlation (TC)** of a group of variables: how far their joint distribution is from the
  product of their individual distributions, in nats: TC = Σ_i H(x_i) − H(x_1..x_n). Zero iff independent.
- **Conditional TC of level k**, TC_k = Σ_i H(c_{k,i} | coarser levels) − H(c_k | coarser levels): the part
  of level k's information that lives in the *dependencies between its own codes*. A sampler that draws
  all codes of a level independently (NCM) cannot represent it; its best possible KL error is exactly
  Σ_k TC_k (Kang et al. 2025, as cited in the parallel-decoding literature below).
- **Bag of words**: a summary that records which words occur but not their order.

## What is already done (so we do not rediscover it)
**Coarse-to-fine / next-scale generation.** NCM (arXiv 2610.14162) is the only next-scale *text* model
found. Images: VAR (2404.02905); HMAR (Kumbong et al., CVPR 2025 — same first author as NCM) adds
masked multi-step generation *within* each scale and a Markov scale dependency; Markov-VAR
(2511.23334); scale skipping / early exit for VAR: SkipVAR (2506.08908), FreqExit, SparVAR
(2602.04361), SparseVAR (2507.20454). Audio/speech/motion next-scale models (2408.09027, 2409.11630,
2604.03799). => *within-level refinement* is an obvious extension (HMAR); *scale skipping* exists for images.

**Concept-level / latent LMs.** Large Concept Models (Meta 2024; sentence embeddings, diffusion/MSE);
CALM (2510.27688; autoencoder K tokens -> 1 vector, energy-score generative head, >99.9% recon);
NCP-ArchPreview (2609.10715); Synergy (2507.12769).

**End-to-end tokenization.** H-Net dynamic chunking (ICLR 2026), RL-learned tokenization (2602.13940),
hierarchical AR transformers (ICLR 2025), UNITE (2603.22283, images). None trains a *multiscale VQ*
tokenizer jointly with a *parallel* generator for text.

**Parallel-decoding error theory.** Conditional TC lower-bounds the KL of factorized parallel decoding
(Kang et al. 2025); serial-depth bounds for adaptive schedules (2608.25505); generation order vs
parallelization bias (2602.00286); within-block TC = extra error, fixed by a small local AR model
(CoDiLA, 2603.20216, ICML 2026); joint token commitment (2607.04469); query-efficient rejection
sampling (2610.22238); ParallelBench (2510.04767). All of this is about masked diffusion over a *fixed*
token representation; none treats the representation (tokenizer) as the lever that sets TC.

**Fixing within-step independence.** EDLM (2410.21357, ICLR 2025): AR-model energy + importance
resampling of parallel proposals; tensor-train joint heads (2607.03788); latent discrete diffusion
(2510.18114); dimensional-correlation distillation (2410.08709).

**JEPA for text.** LLM-JEPA (2025), VL-JEPA (2512.10942), JEPA-Reasoner, DLLM-JEPA; "The JEPA Paradox in
Language" (2026) reports collapse of squared-error latent prediction on text and argues objectives must
keep several completions. => JEPA coarse levels need a multimodal head (flow/energy), CALM-style.

**Recursive reasoning (Sudoku).** HRM 55.0% and TRM 87.4% on Sudoku-Extreme with 5-27M params; TRM's
largest gain came from full backpropagation through the recursion (56.5% -> 87.4%).

**Upstream NCM community PRs.** #12: rule-checker Sudoku early exit (no measurements reported); #7: FSQ +
"Jacobi" correction + coupled decoding (claims not supported by its own tests).

## Phase-0 facts that shape the candidates (ours, measured)
1. Released TinyStories tokenizer is ~lossless at full depth (184 ppm token errors) => quality gap is in
   *generating codes*, not reconstructing them. (R002)
2. Its encoder is a lookup table and coarse codes behave like window bags-of-words. (R003)
3. Sudoku: 85.5% of "hard" puzzles are solvable by singles propagation; mean 7.5 parallel rounds. (R000)

## Candidates
Each: mechanism → NCM weakness targeted → cheapest kill experiment → full-test cost → prior.
"(outside ML)" marks a mechanism imported from another field; "(new)" marks candidates that are not
extensions of the prompt's five ideas.

**C1. TC accounting, then a parallelizability-regularized tokenizer (PRT).** (extends idea 1, reframed)
For a lossless deterministic tokenizer, the best factorized generator's NLL is H(text) + Σ_k TC_k, so
"rate" in idea 1 is not the lever: the tokenizer controls only the *parallelization tax* Σ_k TC_k (plus
learnability). PRT adds β·E_soft[−log q_φ(c_k | c_<k)] to the tokenizer loss, with q_φ a small factorized
prior trained jointly and soft code assignments carrying gradient into a contextual encoder.
Targets within-level independence and the separately trained tokenizer. Kill (a): estimate TC_k as
NLL(factorized) − NLL(within-level AR) with two small generators on one tokenizer; kill PRT if
Σ_k TC_k < 10% of the NCM−AR NLL gap. Kill (b): PRT at small scale must cut Σ TC by ≥20% at ≤1% recon
loss. Cost: (a) ~3 CPU-h, (b) ~4 CPU-h; full ~25 CU. Prior: (a) informative 90%; PRT works 20%.
**This is the high-risk bet.**

**C2. Multigrid smoothing: red-black relaxation sweeps at high-TC levels.** (outside ML: multigrid PDE
solvers) A multigrid V-cycle alternates prolongation (coarse→fine) with smoothing (local relaxation
sweeps). NCM only prolongs. Add red-black Gauss–Seidel-style sweeps: resample even positions given odd
ones (and coarser levels), then odd given even, only at the levels where measured TC_k is largest.
Needs a generator also trained with random same-level context (mask-conditioned). Targets within-level
independence/repetition. Kill: at small scale, +1 sweep (2 NFE) at the top-3 TC levels must improve
gen-PPL at fixed entropy, or MAUVE, beyond 2 seed-SDs. Cost ~4 CPU-h; full ~10 CU. Prior 40%.
Novelty: modest (HMAR does within-scale masking for images); TC-targeted placement + text are new.

**C3. Encoder-as-verifier: fixed-point residual.** (new) The tokenizer encoder is deterministic, so a
sampled hierarchy is self-consistent iff Enc(Dec(c)) = c. Mismatch at coarse levels means the detail
did not realize the plan. Use it as a free per-sample quality score, for best-of-N among parallel
samples (same NFE, N× FLOPs), or to trigger selective resampling. Kill: on released-model samples,
|Spearman| between per-sample mismatch and scorer-LM NLL < 0.2. Cost ~1 CPU-h. Prior 30%.

**C4. Recurrent NCM: carry the generator's state across levels, train with backprop through levels.**
(outside ML: state observers / Kalman filtering carry the belief, not just the point estimate) NCM
discards the transformer state after each level and restarts from the committed codes z_hat. Pass level
k's final hidden states (area-upsampled, zero-initialised projection) into level k+1, and train by
unrolling the K levels with full backprop (prefix K/V cached once, so FLOPs ≈ standard training).
Same NFE, same params (+1 projection). Targets reasoning depth (Sudoku-hard) and plan coherence.
TRM's analogous change was its biggest win. Kill: small Sudoku model, hard solve rate +3 points over
matched NCM, 2 seeds. Cost ~2 CU per arm (Colab). Prior 35%.

**C5. Full-resolution refinement levels + halting (Sudoku).** (extends idea 5) Replace the late
resolution levels with repeated full-resolution residual levels so intermediate decodes are complete
boards; halt when a learned head or C3's fixed point says done. Targets fixed compute (criterion d).
Kill: intermediate full-res decodes must be exact for ≥30% of easy puzzles by mid-depth. Cost ~2 CU.
Prior 25%.

**C6. Contextual encoder + summary supervision ("semantic" coarse levels).** (idea 2) Our R003 shows
lookup-encoder coarse codes are bags of words. A 2-layer contextual encoder makes each latent
context-dependent; optional auxiliary loss predicts a frozen sentence-encoder embedding of each window
from its coarse code. Measure invariance probe (shuffle vs synonym), linear probes (main character,
topic), edit locality, generator NLL and TC (does semantics cost parallelizability?). Kill (semantic
claim): shuffle-invariance at L1/L2 not reduced vs lookup. Cost ~3 CPU-h per variant. Prior 50%
(semantics), 25% (better generation).

**C7. Importance-resampled levels with a within-level AR energy (EDLM on NCM).** (idea 4) Draw N
candidates per level from NCM's factorized logits (free), score them in one batched pass with C1(a)'s
within-level AR model, resample. +1 NFE per level. Kill: per-level effective sample size of the
importance weights at N=16 below 5% at the high-TC levels => cannot work at feasible N. Cost ~1 CPU-h
on top of C1(a). Prior 20%.

**C8. Lifting-scheme (critically sampled wavelet) tokenizer.** (outside ML: wavelets/lifting) NCM's
Laplacian pyramid uses 641 codes for 128 tokens (5× redundant), which costs 5× generator positions.
A learned lifting scheme (split even/odd, predict odd from even, update, recurse) gives 128 codes
total and 8 levels. Targets compute/NFE (criterion a's 0.6× NFE). Kill: ≥99.5% recon and generator
NLL within 5% of NCM at matched FLOPs. Cost ~4 CPU-h. Prior 15% (token embeddings are not smooth
signals, so details carry nearly everything).

**C9. Successive-cancellation list decoding over levels.** (outside ML: polar-code SC-list decoding with
a CRC check) Keep L candidate hierarchies through the coarse levels (batched: NFE unchanged, FLOPs ×L),
prune by generator log-prob plus a check (Sudoku rules, or C3's fixed point for text). Kill on the
released Sudoku-hard checkpoint: L=4 must add ≥1 point. Cost ~1 CPU-h. Prior 30% (Sudoku).

**C10. Serial-depth law for NCM on Sudoku.** (new; mechanistic) Is per-puzzle success predicted by the
puzzle's propagation depth relative to the number of levels? Free from Phase-0 trajectories; extension
trains small NCMs with 6/12/18 levels to see whether the solvable depth shifts with K. Kill: success flat
in propagation rounds. Cost: analysis free; extension ~3 CU. Prior 50% (dependence), 25% (clean law).

**C11. JEPA coarse levels with a flow/energy head.** (idea 3) Continuous coarse latents predicted by a
CALM-style energy head, discrete final level. Deprioritized: needs a multimodal head (JEPA paradox),
no exact likelihood to compare, highest cost at our scale. Prior 10%.

## Ranking (expected information per GPU-hour) and choice
1. C1(a) TC accounting — gates C1(b), C2, C7; cheap; directly answers "where is NCM's gap?".
2. C10 analysis — free from Phase-0 data.
3. C3, C9 — cheap on released checkpoints.
4. C6 — cheap, mechanistic (criterion c).
5. C4 — Colab; best shot at a large Sudoku gain.
6. C2 — best shot at criterion (b) if TC is large.
7. C1(b) PRT — high-risk, high-reward.

**Chosen for Phase 2:** C1(a)→{C2 or C7} (within-level dependence track), C6 (semantics track), and
C4 (high-risk/high-reward recurrence track; Sudoku first). Free/cheap side checks: C10, C3, C9.

## Prior art added after the external review (2026-10-23, R023)
Cited by the hostile review (`docs/reviews/gpt_hostile_review_2026-10-23.md`); all are known to us except Jacobi
Forcing, whose existence we did not check ourselves.
- **Corruption / exposure bias:** Bengio et al., scheduled sampling (2015): feeding model predictions during training
  helped sequence models; Huszár (2015): that objective can be statistically inconsistent. Qian et al., GLAT (2021):
  glancing (prediction-guided) training for parallel decoders. Han et al., Infinity (CVPR 2025): random bit flips in
  a bitwise hierarchical visual tokenizer with re-quantized finer targets ("bitwise self-correction"). **Bearing:**
  B1b / A2 is close to Infinity's mechanism (uniform, not on-policy, corruption with re-quantized targets); B1a's
  on-policy source is the scheduled-sampling idea applied across NCM levels. None predicts which W1 arm wins.
- **Jacobi / fixed-point decoding:** Santilli et al. (2023): training-free parallel (Jacobi) decoding for MT.
  Kou et al., CLLMs (2024): train intermediate Jacobi states to predict the fixed point, keeping an AR loss. Hu et al.,
  Jacobi Forcing (2025, unchecked): training on generated parallel trajectories. **Bearing:** J2 is CLLM-style
  consistency applied across NCM levels; the contribution, if any, is the level-structured version. CLLMs report a
  quality-speed trade-off, which is why J1a compares J2 with capped decoding of J0 / J1.
- **Iterative refinement / Sudoku:** Lee et al. (2018), iterative refinement for non-AR translation; Palm et al.,
  Recurrent Relational Networks (2018); Wang et al., HRM (2025); Jolicoeur-Martineau, TRM (2025). **Bearing:**
  repeated refinement solves hard Sudoku in other architectures; it does not predict gains for NCM's repair arm over
  the training-free verifier baseline (S1/S1b).
