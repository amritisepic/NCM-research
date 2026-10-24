# NCM research advice: on-policy corruption, residual hierarchies and Jacobi consistency

Date: 2026-10-24. Primary-source literature synthesis and experimental advice.
This report records the advice delivered after reviewing R030-R033; proposals labelled
**speculation** are not preregistration amendments or experimental results.

> **Publication note (2026-10-24).** The repository advanced after the report was drafted.
> [R034](../log.md#r034--w1d-diagnostic-built-audits-3-completed-4-and-5-mauve-precision-correction-2026-10-24)
> identifies a dropout-RNG confound in the R031 Sudoku comparison, so its matched-corruption
> interpretation requires a paired rerun. R034 also records a canonical-parent versus
> used-parent proposal mismatch, new context-reliance interventions, the W1 MAUVE bf16-feature
> correction, and the J1c amendment following audit 4. Read those updates alongside the
> historical assessment below; this document does not supersede the current log or registration.

The proposed mechanism is plausible, but the evidence currently supports **"plausible
corruption can suppress useful conditioning"** more strongly than **"that suppression
explains the text collapse."** A4 is the main unresolved counterexample.

Retain J2 as a bounded experiment, add a frozen-teacher comparator, and make the mechanism
test a controlled factorial. Also narrow "fine levels hold most of the NLL": the released
model's L8-L11 contribute **46.8% of clean NLL**, while containing **63.1% of predicted
code slots**. Their concentration of the *NLL improvement* is the stronger observation.

Evidence reviewed: the public branch's state, registrations, R030-R033, six W1 result
files, and the wave-0 intervention/Jacobi results. W1 uncertainty is over sampling groups,
not training seeds; R031 also records that clean-NLL evaluation used training autocast.
Those values warrant fp32 re-evaluation before interpreting small differences.
[Project records](../log.md#r030--wave-1-registered-analysis-w1--w1a--w1b-seed-0-colab-a100-40gb-user-run)

## 1. When does training on a model's own predictions hurt?

**Findings.** There are documented severe failures, including in the original
scheduled-sampling paper. There are also successful methods, but they differ substantially
in what remains paired with the altered input.

In the table, "unmeasured" means the source does not establish how accurately corruption
could be distinguished using the generator's available input. Absence of a replacement
flag does not establish indistinguishability.

| Paper and exact setting | Input changed or trained on | Target | Corruption signal | Reported effect |
|---|---|---|---|---|
| **Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks, 2015; MSCOCO captioning** | Previous words replaced by model predictions, gradually or always | Original reference caption | No replacement flag; detectability unmeasured | Baseline BLEU-4 **28.8**; scheduled sampling **30.6**; uniform scheduled sampling **29.2**; **always sampling 11.2**. CIDEr **89.5 -> 49.7** under always sampling. [Table 1](https://arxiv.org/pdf/1506.03099) |
| **How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?, 2015; two-variable theoretical analysis** | First variable replaced by an independently drawn model sample | Original second variable | Overlap is part of the construction | Objective is improper; in the full-replacement limit the optimum loses dependence and matches marginals. This is a theoretical result, not a measured collapse magnitude. [Section 3](https://arxiv.org/pdf/1511.05101) |
| **Scheduled Sampling for Transformers, 2019; IWSLT De-En and KFTT Ja-En** | First decoder pass supplies replacements for a second pass | Original translation | No origin flag; soft embedding mixtures also differ from ordinary tokens | No-backprop argmax: De-En test BLEU **29.62 -> 20.57**, Ja-En **19.46 -> 15.13**. Some mixture variants remain competitive; Gumbel-softmax alpha=10 reaches **20.87** Ja-En. [Table 2](https://aclanthology.org/P19-2049.pdf) |
| **Parallel Scheduled Sampling, 2019; CIFAR-10** | One parallel prediction/mixing pass; model-token probability p, exponential schedule, 100k warmup | Original image tokens | No origin flag; detectability unmeasured | Teacher forcing FID **52.39**; p=.5: **41.47**; p=1: **98.38**. IS **5.64 -> 6.42 -> 3.97**, respectively. [Table 3](https://arxiv.org/pdf/1906.04331) |
| **Input Perturbation Reduces Exposure Bias in Diffusion Models, 2023; ADM** | Extra Gaussian noise added to the usual noisy image; gamma=.1 | Original diffusion noise epsilon, **excluding** the extra perturbation | Noise level t supplied; extra realization unmarked | CIFAR-10, 1,000 sampling steps: FID **2.99 -> 2.76**; 80 steps: **3.63 -> 2.93**. FFHQ-128, 1,000 steps: **9.65 -> 2.98**. [Algorithm 3; Table 3](https://arxiv.org/pdf/2301.11706) |
| **MaskGIT, 2022; ImageNet-256** | Training masks real image tokens; inference retains confident predictions and re-masks others | Original masked image tokens | Explicit mask token; training context otherwise real | **227M**, **8 passes**, FID **6.18**, IS **182.1**. This is not an own-prediction-training ablation. [Table 1](https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf) |
| **Improved Masked Image Generation with Token-Critic, 2022; ImageNet-256** | Critic sees real tokens mixed with completions from a **fixed** MaskGIT generator | Binary original-versus-generated position mask | Critic learns origin detection; generator is held fixed | At 18 generator-plus-critic rounds, FID **6.56 -> 4.69**; recall **.48 -> .53**, precision **.79 -> .76**. Table labels cost **18x2**. [Algorithm 1; Table 1](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830070.pdf) |
| **Autoregressive Image Generation using Residual Quantization, 2022; 821M, ImageNet** | Codes sampled from a distribution based on distance to the **image's residual**, rather than from the generator | Sampled/soft tokenizer targets; later residuals depend on sampled earlier codes | No origin flag; detectability unmeasured | FID: baseline **14.06**; stochastic quantization **13.24**; soft labels alone **14.87**; both **13.11**. [Supplement B.3](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Lee_Autoregressive_Image_Generation_CVPR_2022_supplemental.pdf), [official quantization code](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqvae/quantizations.py#L372) |
| **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E), 2023; eight EnCodec quantizers** | AR first codebook; seven NAR residual stages. Training uses recorded-speech codes, inference predicted lower stages | Recorded-speech codec codes | No corruption procedure evaluated | In the NAR ablation with **ground-truth first-level codes**, removing both prompts yields WER **19.6**, phoneme prompt **3.0**, phoneme+acoustic prompts **2.8**. Architecture/prompt evidence, not evidence for own-prediction training. [Section 4.2.2; Table 4](https://arxiv.org/pdf/2301.02111) |
| **Visual Autoregressive Modeling (VAR), 2024; class-conditional ImageNet-256** | Clean coarse-to-fine token maps during teacher forcing | Canonical next-scale codes | No training corruption tested | The **2B** model reports FID **1.73**. No own-prediction corruption ablation in the original paper. [Paper](https://arxiv.org/html/2404.02905v2) |
| **Infinity, 2025 version; 5M image-text pairs, 512x512** | Random bit flips at earlier scales; residuals recomputed | **Re-quantized next-scale residual bits** | No origin flag; detectability unmeasured | At temperature=1, CFG=3: FID **9.76** baseline, **9.69** random flips alone, **3.48** bitwise self-correction. ImageReward **.52 -> .76**. [Section 3.4; Table 7](https://arxiv.org/html/2412.04431v2) |
| **CLLMs: Consistency Large Language Models, 2024; LLaMA2-7B, GSM8K** | Intermediate states of teacher Jacobi trajectories | Distribution under the trajectory's fixed-point prefix, plus teacher-generation AR loss | No replacement flag | Batch 1: **43.5 -> 132.4 tokens/s**, **3.0x**; accuracy **59.1 -> 56.4**. [Table 1](https://arxiv.org/html/2403.00835v4) |
| **Break the Sequential Dependency of LLM Inference Using Lookahead Decoding, 2024** | Inference-time Jacobi guesses generate candidate n-grams | Verified against the unchanged model | No training corruption | Up to **1.8x** on MT-Bench; up to **4x** with multiple-GPU strong scaling on code completion. Distribution preservation comes from verification. [Paper](https://arxiv.org/pdf/2402.02057) |
| **Knowledge Distillation for Visual Autoregressive Models (VarKD), 2026 preprint; ImageNet, LlamaGen-L -> B** | Student-generated suffix after a real prefix of 25-75% of sequence length; confidence-weighted teacher supervision | Teacher distribution **conditioned on the encountered context** | No origin flag; teacher confidence used | FID **6.51** undistilled, **4.92** supervised KD, **4.94** GKD, **4.58** VarKD. Adds confidence filtering and codebook grouping. [Tables 1-2](https://arxiv.org/html/2606.06078v1) |

Three distinctions matter:

- **Scheduled sampling retains the original future after changing its history.** The original
  paper also reports invalid parse trees under always sampling. Thus severe degradation is
  established prior art, although its training-from-scratch setting differs from the competent-
  checkpoint fine-tune here. [Section 4.2](https://arxiv.org/pdf/1506.03099)
- **RQ-Transformer and Infinity keep supervision anchored to the original continuous example
  while adapting the residual representation.** RQ-Transformer's stochastic choices come from
  tokenizer geometry, not the generator's rollout distribution. Its official implementation
  subtracts each sampled embedding before constructing the next residual.
- **Token-Critic and Lookahead avoid the specific inconsistent-label update.** One trains a
  separate detector; the other verifies proposals without changing model weights.

**Speculation.** The useful dividing line is whether the target remains appropriate for the
actual input state. "Training on predictions" combines denoising, imitation, distillation and
inconsistent history-future pairing. Their outcomes need not agree.

No primary source found matches this entire setting: residual text hierarchy, released
competent checkpoint, partial one-step self-sampling, mild reconstruction damage, and rapid
loss of local fluency. Infinity is the closest residual correction comparison; scheduled
sampling supplies the closest established failure mechanism.

## 2. Is the proposed mechanism right, and what is the cheapest discriminating experiment?

**Findings.** The Sudoku experiment establishes a meaningful first step: under the reported
matched wrong-code positions and rates, plausible replacements produce much larger
clean-context NLL and entropy increases. It does **not** establish the cause of text collapse.
The publication note above further qualifies the control claim following R034.

- Temperature-1 solve-rate changes are **-5.3 +/- 1.7 points** for plausible versus
  **-4.3 +/- 1.6** for uniform corruption; their reported difference is **-1.0 +/- 2.0**.
- Re-quantization reduces plausible corruption's clean-NLL increase from approximately
  **32 to 6.7 nats**, while text A4 still collapses.
- Lower average gradient norm in A3 argues against simple gradient explosion; it does not
  exclude harmful gradient direction or an abrupt objective change.
  [R031](../log.md#r031--wave-1-exploratory-integrity-checks-what-the-collapse-is-what-is-significant-2026-10-24)

**Speculation: a narrower mathematical account.** Consider a simplified frozen replacement
process. Let C be context and Y its target. With probability r, replace C by an independent
draw from its exact marginal. The optimal predictor under this altered training distribution is

$$
P_{\mathrm{train}}(Y\mid \widetilde C=c)
=(1-r)P(Y\mid C=c)+rP(Y).
$$

This formalizes conditioning suppression. But it assumes marginal-matched, independent
replacement. NCM samples conditionally on clean earlier levels, replaces multiple positions,
and exposes cumulative reconstructions. Therefore **"discount by roughly the corruption rate"
is a hypothesis, not a quantitative consequence of the scheduled-sampling analysis for this
hierarchy.**

Also, plausible individual codes need not form a plausible *joint hierarchy*.

| Alternative explanation - speculation | Prediction that distinguishes it |
|---|---|
| **Optimization shock / interference** | Lower learning rate, a gradual objective transition, or more clean examples substantially rescues quality without revealing replacement locations. Harm tracks parameter-update direction, not just distinguishability. |
| **Loss of robustness to broad corruption** | Training retains clean-context performance but becomes fragile specifically on fully sampled hierarchies; restoring a small uniform-corruption component rescues it. Clean-NLL damage makes this incomplete as a sole explanation. |
| **Moving-policy feedback** | Frozen-policy corruption is substantially safer than current-policy corruption. An early plateau does not exclude rapid arrival at a harmful equilibrium. |
| **Residual target incompatibility / insufficient repair capacity** | Damage tracks residual magnitude, decoder sensitivity or remaining repair capacity. Recomputed targets or coherent teacher continuations help even without a flag. |
| **Joint train/inference mismatch** | Individually plausible replacements produce incompatible cumulative sums. Coherent sequential prefixes are safer than independent one-step replacements at matched code error and decoded damage. |
| **Calibration change without lost dependence** | Entropy-matched sampling restores local fluency while controlled context interventions show unchanged dependence. Lower temperature alone is insufficient evidence because it also reduces diversity. |

### Assessment of the three proposed tests

**Speculation.**

1. **Own codes plus a replacement flag: valuable, but not decisive alone.** Add the same
   channel to every arm, including uniform corruption. Include shuffled flags, and enough
   completely clean examples that the all-zero inference flag is well trained. In NCM,
   propagate flags through the same scale/interpolation structure as the affected cumulative
   reconstruction. Successful rescue could reflect separation of training regimes; failure
   could reflect shared-weight interference or an ineffective flag.
2. **Plausible-wrong codes at uniform rates: valuable with exact matching.** Use identical
   masks and force both plausible and uniform replacements to be wrong. Excluding the correct
   code alters the policy distribution, so this tests *plausible wrong corruption*, not literal
   inference sampling.
3. **Uniform corruption at A3's lower rates: useful, but lower priority.** Match **per-level
   realized disagreement**, not the nominal replacement probability or a global mean. Policy
   draws can equal the original code; re-quantized labels introduce another reference for
   "changed."

**Cheapest discriminating experiment - speculation:** a short **2x2 factorial**:

- frozen plausible-wrong versus uniform-wrong replacements;
- truthful flags versus shuffled flags.

Use common masks, canonical targets, equal updates/data, identical architecture and a
clean-example mixture. Inspect at 0, 100, 300 and 600 steps. A short current-policy A3
reproduction is a validity check if the corrected trainer has not yet reproduced the effect.

The key measurement should be **dependence on context**, not just entropy: change one
coarse component while holding everything else fixed, and measure the response of subsequent
distributions and decoded local order. Alongside this, measure replacement-detection
performance from exactly the representation available to the generator. Low classifier
accuracy is only suggestive; high accuracy would directly weaken "undetectable."

What is missing from the original proposed tests is a **coherent-continuation target control**:
after altering a prefix, let a frozen teacher continue that prefix and supervise that
continuation. This separates inappropriate targets from prediction-derived inputs.

## 3. Why do likelihood and sample quality diverge?

**Findings.**

*A note on the evaluation of generative models* gives a particularly clear counterexample:
a mixture containing **1% good density and 99% bad density** can generate bad samples 99%
of the time while losing at most **log 100 approximately 4.61 nats per example** relative to
the good density. This is a construction, not an RVQ measurement.
[Section 3.2](https://arxiv.org/pdf/1511.01844)

NCM itself distinguishes canonical-code likelihood from text likelihood. Appendix A
marginalizes over hierarchies; its canonical assignment yields an ELBO whose gap is the
negative log posterior probability of that assignment. Corruption modifies that
clean-conditioning objective. Moreover, its TinyStories loss-weighting ablation moves the
coarse loss share **19% -> 63%**, yet 10k-step Gen-PPL stays about **18.9-19.0** through
alpha<=1, and reaches **19.4** at alpha=1.5.
[Appendices A and F.2.3](https://arxiv.org/html/2610.14162v1)

The raw released-model results give:

| Quantity | Recomputed value |
|---|---:|
| Predicted slots, L1-L11 | **640** |
| Slots, L8-L11 | **404 = 63.1%** |
| Clean NLL, L8-L11 | **403.28 / 861.84 = 46.8%** |
| Slots, L8-L10 | **276 = 43.1%** |
| Clean NLL, L8-L10 | **42.9%** |

Thus L8-L10's large aggregate loss is approximately proportional to their slot count.
The more distinctive result is that re-quantization improves this part of the loss
substantially while generation scarcely improves or deteriorates.
[Released result JSON](../results/colab/w1_L_ts_ctrl/L_ts_released16__result.json)

There is relevant equivalence-aware work. VarKD groups visual code embeddings and **sums
teacher/student probability within groups**, rather than insisting on the original token
ID. In its LlamaGen-L->B ablation, adding codebook relaxation after entropy weighting
improves FID **4.66 -> 4.60**. This establishes a useful approximate grouping objective,
not exact decoded-image likelihood.
[Equations 11-12; Table 2](https://arxiv.org/html/2606.06078v1)

But a universal likelihood-quality split is contradicted by other settings: Infinity reports
validation-loss correlations of **-.983, -.981 and -.979** with GenEval, ImageReward and
HPSv2 across **125M-4.7B** models. These are scaling correlations, not an objective-change
experiment like W1. [Section 4.5](https://arxiv.org/html/2412.04431v2)

**Speculation.** Several effects can coexist:

- Canonical code NLL rewards one encoding of the example; sampling quality concerns all
  generated encodings and their decoded outputs.
- Teacher-forced NLL evaluates real prefixes. Sampling visits model-produced prefixes.
- A small embedding correction can be difficult to predict by code ID yet have little
  decoder effect.
- Within-level independence can make jointly compatible detail hard to sample even when
  individual probabilities improve.
- Aggregate loss weights levels partly through their number of positions.

For a deterministic decoder D, the relevant output probability is

$$
P_D(x)=\sum_{\mathbf c:D(\mathbf c)=x}P_\theta(\mathbf c).
$$

For the paper's probabilistic decoder, use
$\sum_{\mathbf c}P_\theta(\mathbf c)P_\psi(x\mid\mathbf c)$.

Neither "sum probability over 16 nearest embeddings" nor average decoder reconstruction
loss automatically computes this marginal. Nearness is not equivalence, and averaging
negative log probabilities differs from taking the negative log of their average.

No primary evidence was found for the general numerical claim that RVQ fine levels hold
most likelihood cost and almost no perceptual value. Diminishing reconstruction returns are
known; that particular likelihood-quality allocation is not established by the sources checked.

## 4. What do multi-scale levels encode, and what could be new?

**Findings.**

**RQ-VAE provides direct truncation evidence.** For ImageNet validation with an **8x8x4**
code stack, decoding the first one, two, three and four depths gives reconstruction FID
**100.86, 22.74, 7.66, 4.73**. Reconstruction MSE falls
**.018 -> .014 -> .012 -> .010**. This supports progressive refinement, but the last
depth still changes rFID appreciably.
[Supplement Table A.2](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Lee_Autoregressive_Image_Generation_CVPR_2022_supplemental.pdf)

**Hierarchical VQ-VAE-2 shows progressive reconstruction, with a qualification.** Its
three-level reconstruction figure reports latent maps approximately **3072x, 768x and
192x** smaller than the original. The paper explicitly describes levels as extracting
complementary information, rather than every bottom level exclusively refining the top.
These are hierarchical latents, not the same construction as additive RVQ.
[Figure 3; Section 4](https://proceedings.neurips.cc/paper_files/paper/2019/file/5f8e2fa1718d1bbcadf1cd9c7a54fb8c-Paper.pdf)

**SoundStream directly tests removing quantizer stages.** A model trained at **18 kbps**
retains useful quality when decoded using fewer initial quantizers; quantizer dropout closes
the gap to bitrate-specific training. The scalable system spans **3-18 kbps**. This
demonstrates usable prefixes, not semantic specialization of particular residuals.
[Section V-C](https://arxiv.org/pdf/2107.03312)

**EnCodec shows diminishing perceptual returns.** On **48-kHz stereo music**, Table 4
reports MUSHRA **82.9 +/- 2.4 at 6 kbps**, **88.0 +/- 2.7 at 12 kbps**, and
**87.5 +/- 2.6 at 24 kbps**. The final doubling has no measured advantage within
uncertainty. This is a bandwidth intervention, not per-level NLL attribution.
[Table 4](https://arxiv.org/pdf/2210.13438)

**SpeechTokenizer is a stronger causal interchange precedent.** On VCTK, combining source
RVQ-1 with reference-speaker residual layers yields:

| Reference residuals supplied | WER | Speaker similarity |
|---|---:|---:|
| RVQ-2 | **2.6** | **.72** |
| RVQ-2:4 | **11.7** | **.80** |
| RVQ-2:8 | **35.4** | **.82** |

Residuals transfer speaker information but increasingly disturb content. Its first level
is explicitly shaped by HuBERT distillation; this is not evidence that ordinary RVQ
naturally disentangles semantics.
[Section 5.2, Table 5](https://arxiv.org/html/2308.16692v2)

**VAR demonstrates conditional editing**, forcing tokens outside a region and generating
inside it. That establishes useful spatial conditioning. Its original paper does not
supply a quantitative causal decomposition assigning specific semantic attributes to each
scale. [Section 6](https://arxiv.org/html/2404.02905v2)

**Speculation.** "Coarse structure, fine detail" is established. A stronger new NCM result
would identify **which dependency each level supports, under which prefix distribution**.

For example:

> Mid-level residuals bind local token order to an earlier lexical plan; plausible-prefix
> corruption selectively weakens that binding.

To support that statement, use controlled level swaps or replacement interventions,
measure lexical content and local order separately, and repeat under both correct and
sampled coarse prefixes. Replicate across training seeds and at least two released task
checkpoints.

The oracle curve is valuable, but its **86-87% contribution through L1-L5 is conditional
on supplying increasingly correct prefixes**. It cannot establish that later levels are
dispensable after a wrong plan.

## 5. Can consistency training make level-parallel Jacobi converge early?

**Findings.**

CLLM's results are encouraging but narrower than "quality-preserving acceleration" suggests:

- GSM8K accuracy **59.1 -> 56.4**, speed **3.0x**.
- Spider metric **70.0 -> 69.3**, speed **3.4x**.
- Its iteration-related profiling reports fast-forwarded prefix tokens per update:
  **1.1 -> 5.7** for 16-token Spider blocks, and **1.1 -> 2.8** for 16-token GSM8K
  blocks. These are **not mean full-block iteration counts**.
- Increasing the AR-loss weight from 1 to 10 changes GSM8K **51.3 -> 56.4**, while
  speedup changes **3.2x -> 3.0x**.
  [Tables 1-3 and 6](https://arxiv.org/html/2403.00835v4)

A later primary comparison, **Jacobi Forcing**, reports on HumanEval with
Qwen2.5-Coder-7B-Instruct: **1.03 tokens/forward** for ordinary Jacobi versus **4.01**
for its trained model; accuracy changes **87.8 -> 83.5**, with **3.86x** wall-clock
speedup. It uses progressive trajectory distillation and modified training attention.
[Table 1](https://arxiv.org/html/2512.14681v1)

There is also direct multi-scale precedent beyond Jacobi: **Distilled Decoding** converts
VAR-d16 from **10 steps, FID 4.19** to **one step, FID 9.94**, or **two steps, FID 7.82**.
It supplies Gaussian randomness through a deterministic noise-to-teacher-output mapping.
Thus multi-scale acceleration is demonstrated, with a substantial quality trade-off; it
does not establish early exact fixed-point convergence.
[Table 1](https://arxiv.org/html/2412.17153v2)

The WMT14 baseline has **11.0666 exact passes**, versus **9.5808** for first reaching the
final decoded output, on **3,003 sentences**. Its current position-count proxy is
**3.30x sequential**. At 8.25 unchanged full passes that proxy would still be
**2.46x sequential**. These are computed-position proxies, not attention-inclusive FLOPs.
[Raw WMT14 result](../results/colab/w0_calib_diag/b2_jacobi_wmt14_n3003.json)

**Speculation.** J2 is learnable in principle, especially because the source sentence can
disambiguate the target even when earlier codes are wrong. But failure is expected when:

1. **Current inputs omit information needed to predict the chosen fixed point.** The loss
   then rewards averaging or bypassing earlier levels.
2. **Teacher trajectories move too quickly.** J2 uses the current model, whereas original
   CLLM collects trajectories from a target teacher.
3. **Reference anchoring is insufficient.** Faster convergence can arise from simpler,
   shorter or less informative translations.
4. **Exact latent convergence remains dominated by harmless residual changes.** Better
   decoded outputs need not shorten the latent tail.
5. **The architecture cannot express the shortcut cheaply.** NCM's cumulative
   reconstruction differs from access to a full token history.
6. **Fewer calls fail to reduce latency.** Whole-hierarchy passes perform much more work
   per call.

For a triangular hierarchy, finite convergence follows from the dependency structure.
Fast convergence additionally requires useful levels to become correct before all their
predecessors settle. A consistency objective encourages that behavior; it does not guarantee it.

## 6. Argue against the plan, then choose one experiment under approximately 100 CU

**Findings.** The registered J1/J2 design already includes essential safeguards: reference
fine-tuning, exact-versus-sequential checks, saved capped hypotheses, length-ratio checks,
paired bootstrap comparisons and wall-clock accounting. Preserve these.
[J1/J1a/J1b](preregistration.md#j1--wmt14-self-consistency-fine-tune-for-early-exact-jacobi-convergence-criterion-d-text-registered-2026-10-23-before-any-run)

Sudoku already supplies a strong comparison for the same exact-stopping algorithm:
**6.406 versus 12 passes**, identical sequential output, **86.15%** solved. Verifier
stopping gives **2.3655 passes and 90.3%**, but changes the output-selection rule.
[Raw Sudoku result](../results/colab/w0_calib_diag/b2_jacobi_sudoku_hard.json)

**Speculation: the case against the plan.**

- The mechanism candidate is being treated as an explanation before demonstrating the
  causal link to text quality.
- A four-arm screen without uniform flag controls or a frozen policy may produce another
  ambiguous result.
- J2 has a plausible route to a call-count success that is a practical compute failure.
- "Equivalent-code likelihood" could consume effort without becoming either tractable or
  decision-useful.
- Dropping clean NLL entirely would discard a useful warning signal. Drop it as a sole
  selection metric; retain it as a diagnostic.
- The oracle curve does not justify spending less computation on fine levels without
  testing wrong-prefix conditions.
- Replicating improvements within the collapsed regime would answer a less important
  question than identifying a safe objective or meeting criterion (d).

### Recommended single experiment: frozen versus moving-teacher Jacobi distillation

**Speculation.** Prioritize one three-arm WMT14 experiment, targeting criterion **(d)**:

| Arm | Purpose |
|---|---|
| **J1:** registered reference fine-tune | Measures ordinary fine-tuning effects |
| **J2:** registered current-model trajectories and fixed points | Tests the existing proposal |
| **J2-F:** same loss structure, but freeze the released trajectory/fixed-point teacher | Tests whether stable supervision makes the shortcut safer |

Keep the reference-loss mixture, initialization and inference architecture matched between
J2 and J2-F. Count trajectory construction, teacher evaluation and backward work in the
compute budget. Register J2-F separately before results.

Use **exact fixed-point stopping on both WMT14 and Sudoku** for the main cross-task claim.
Sudoku's verifier can remain an additional result. This avoids requiring different
stopping certificates to carry the main comparison.

A proposed budget ceiling - not a runtime prediction - is:

| Stage | CU ceiling |
|---|---:|
| Released-model checks, trajectory collection/profile, stopping baselines | **6** |
| Three WMT14 arms, seed 0 | **15** |
| Seeds 1-2 for the comparison that passes the registered screen, including controls | **30** |
| Final evaluation, paired uncertainty, batch-1 timing and cost accounting | **14** |
| Unspent contingency | **35** |
| **Total cap** | **100** |

The unused allocation should remain unused if the seed-0 gate fails.

**Evidence for this bet:** the Sudoku half already has evidence, with runnable released
models, registered comparisons and a modest text target. CLLM supplies a direct objective
precedent; frozen supervision makes the experiment easier to interpret.

**Evidence against:** the most relevant papers show real quality losses, and the current
WMT14 implementation is expensive per pass. There is no primary evidence that a 120M
residual text hierarchy can learn this shortcut within the budget.

Success should require the registered **<=8.25 mean WMT14 passes**, **<1 BLEU loss**,
superiority to the capped controls, and replication. Report separately whether it also
saves wall time and actual computation. It would address criterion (d); it would not
establish (a) or (b).

## What we would have to see to change our minds

The following are **speculation: proposed decision thresholds**, not amendments to
existing registrations.

| Position | Observation that would overturn or materially revise it |
|---|---|
| **Undetectable plausible corruption is the dominant mechanism** | A detector using only generator-visible inputs reaches **AUC >=.90** on plausible versus clean contexts; or truthful flags remove **<20%** of excess Gen-PPL damage across three seeds despite demonstrated flag use and a trained clean branch. |
| **Conditioning suppression causes the local-order collapse** | Controlled coarse-context sensitivity remains within **10%** of the matched uniform control while Gen-PPL rises by **>=10**, or repairing the implicated levels removes **<20%** of the local-order deficit. That would weaken the proposed mediation. |
| **Moving-policy feedback is secondary** | Frozen-policy corruption removes **>=80%** of the excess Gen-PPL damage at matched realized error rates, while current-policy corruption reproduces it. |
| **J2 is a promising criterion-(d) route** | Apply its existing kill rules: it fails the required quality/pass trade-off, or does not reduce passes by **>=1 versus J1**. If J2-F succeeds while J2 fails, revise the teacher policy rather than dismissing consistency training. |
| **A J2 result is practically efficient** | At the accepted quality point, batch-1 latency is **no lower than sequential**, or attention-inclusive computation exceeds sequential. A logical-NFE result could remain, but a practical speedup claim would fail. |
| **On-policy corruption should be dropped altogether** | Reopen it if flagged corruption or context-appropriate teacher targets produce a replicated gain **>=0.7 GPT-2-large Gen-PPL**, exceeding **2x pooled training-seed SD**, with entropy within **.05**, MAUVE loss no worse than **.01**, and no contradictory second-scorer/local-order result. |
| **The original canonical-label on-policy recipe deserves more compute** | Reopen that specific recipe only if it reproducibly passes those quality checks after protocol fixes. Success of a different target or flagged regime would not rehabilitate A3 itself. |

The strongest immediate falsification opportunity is the interaction between **replacement
plausibility, truthful flags and target appropriateness**. The strongest bounded performance
bet is **stable-teacher Jacobi distillation with the existing capped-decoding and quality
controls intact**.
