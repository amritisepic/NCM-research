# Independent research review: residual dependence and parallel generation

Date: 2026-10-26. Advisory research synthesis, preserved from the original response. Proposed experiments, budgets, and queue changes are recommendations, not preregistration amendments or launch approvals.

**[speculation] Your best opportunity is to show that a residual hierarchy creates dependencies that its parallel sampler cannot represent—and then repair those dependencies without improving any individual code prediction.** That would explain several anomalies at once and support a result more interesting than another corruption recipe.

I would pursue two linked directions:

1. **A marginal-preserving joint sampler:** keep the released generator frozen and change only how its simultaneous categorical predictions are coupled.
2. **A causal account of corruption-induced collapse:** test whether hidden corruption provenance creates uncertainty over coherent alternatives, which independent sampling then converts into incoherent combinations.

I would protect enough compute to finish the mechanistic result even if the sampler fails. I would not spend the remaining budget establishing another weak from-scratch NCM–AR comparison.

I checked the local brief and the public branch’s log through R040 and registrations. This is a research assessment, not a new code audit or a reproduction of every reported number. **[ours]** below denotes your measurements; **[source]** denotes verified literature; **[speculation]** includes my deductions, proposed experiments, thresholds, costs, and subjective probabilities. Mathematical deductions are exact under their stated assumptions; their applicability to NCM remains a hypothesis.

**[speculation] Several of your strongest-sounding conclusions are substantially weaker than their wording.**

| Claim under pressure | What the evidence actually supports | Cheapest discriminating test |
|---|---|---|
| Coarse levels contain “concepts” or plans | **[ours]** They preserve some content and are insensitive to within-window order. **[speculation]** Averaging token embeddings can produce this mechanically, without semantic abstraction. | Compare sentences with identical words but reversed roles, negation scope, and attachment. Measure conditional code changes with earlier levels fixed; use position-aware probes alongside pooled probes. CPU. |
| Fine codes are near-equivalent nuisance variables | **[ours]** Many sampled alternatives are geometrically close and canonical finer levels rescue reconstruction. **[speculation]** Equivalence may require coordinated changes elsewhere in the hierarchy. | Compare single substitutions with independently combined substitutions and coherent re-quantized substitutions, matched for individual code distances. Measure reconstruction and downstream generated text separately. |
| One-step model corruption approximates inference | **[ours]** It preserves 98% of tokens in your intervention. **[speculation]** Its proposals saw clean ancestors, and the reconstruction retains strong canonical information. That is a forgiving experiment. | Repeat with genuinely ancestral proposals and sampled remaining levels. Separate proposal damage from the ability of canonical fine codes to overwrite it. |
| The oracle curve localizes the problem to coarse levels | **[ours]** Real coarse prefixes remove most of the Gen-PPL gap. **[speculation]** This identifies a successful intervention, not an additive contribution or a ceiling on improving fine levels. | A small factorial: clean/generated coarse prefix × original/modified fine sampler. Report interaction effects and log scorer loss, not percentages of a nonlinear perplexity difference. |
| The NLL gap establishes a representation problem | **[ours]** Small-model excess is at least about 1,059 nats/window. **[speculation]** It does not distinguish dependence, insufficient conditioning state, or inaccurate marginals. | Nested probes with identical initialization and optimization, adding either raw ancestor information or sibling information. Validate on synthetic known distributions first. Report predictive gains, not automatically “TC estimates.” |
| Undetectable plausible corruption explains collapse | **[ours]** Plausible corruption induces hedging; Sudoku does not exhibit the text collapse. **[speculation]** The missing link is why hedging becomes incoherent generation in one setting. | W2M plus a dependence intervention: change joint sampling while preserving marginals. A rescue without reversing hedging would be especially informative. |
| Carry, soft labels, and token-aligned schedules are dead | **[ours]** Particular implementations failed. **[speculation]** Carry failed in a weak regime; independent soft labels do not test joint equivalence classes; the oracle curve does not bound a redesigned tokenizer. | No immediate revival. Preserve narrow negative claims. Revisit only after a cheap diagnostic directly identifies the corresponding bottleneck. |
| Smaller Gen-PPL at similar unigram entropy means better language modeling | **[speculation]** Unigram entropy does not control sentence diversity, conditional dependence, or memorization. | Keep both scorers, MAUVE, repetition, length, training-neighbor overlap, and a blinded sample comparison. Include an independent evaluation seed bank. |

**[speculation] The largest methodological danger is treating sample-level precision as replication.** Sixteen sampling groups tightly characterize one trained checkpoint. They do not characterize training variation. Your large on-policy degradation is credible as a phenomenon; its precise cause and the small A2 differences remain much less secure.

**[source] Two NCM details further weaken architectural interpretations.** Its level-count ablation changes the tokenizer and total code workload, and runs without conditioning corruption. Its loss-weighting ablation already reports little generation improvement from shifting weight toward coarse levels. Neither “more levels help” nor “coarse levels matter” therefore establishes that additional serial reasoning or coarse-loss weighting is the solution. Kumbong, Khare et al., 2026, arXiv:2610.14162, Appendices F.1–F.2. [NCM paper](https://arxiv.org/html/2610.14162v1)

---

**[speculation] Start from the object itself: NCM is a redundant coordinate system, not necessarily a semantic hierarchy.**

Write the frozen encoder as

\[
C=E(X)=(C_0,\ldots,C_{K-1}),
\]

the cumulative reconstruction as

\[
Z_k=\sum_{\ell\leq k}A_\ell e_\ell(C_\ell),
\]

and the generator’s conditioning state as \(S_k=B_k Z_{k-1}\). The operators \(A_\ell\) include upsampling and residual convolution; \(B_k\) is the resize seen by level \(k\).

Three objectives are different:

- reconstruct \(X\) accurately from all codes;
- make each partial reconstruction useful;
- make the next level’s codes conditionally independent and easy to predict.

Optimizing the first two does not establish the third.

**[source] Successive refinement theory already distinguishes progressively improving reconstruction from achieving an optimal description at every stage.** Optimal successive refinement requires structural conditions on the rate–distortion solutions. Equitz and Cover, 1991, *IEEE Transactions on Information Theory*. [Successive Refinement of Information](https://isl.stanford.edu/~cover/papers/paper94.pdf)

**[speculation] NCM needs an additional property beyond successive refinement: computationally cheap conditional sampling of each refinement.** A representation can be excellent for compression or reconstruction and terrible for factorized generation.

Here is a particularly damaging counterexample.

Let \(X_1,X_2\) be independent fair bits. Store the coarse variable

\[
S=X_1+X_2,
\]

then residuals

\[
R_1=X_1-S/2,\qquad R_2=X_2-S/2.
\]

When \(S=1\), the residual pair is either

\[
(-1/2,+1/2)\quad\text{or}\quad(+1/2,-1/2).
\]

The original variables were independent. **The coarse representation has created a perfect residual anticorrelation.** Sampling the two correct residual marginals independently produces an invalid residual combination half the time conditional on \(S=1\), or one quarter of the time overall.

The coarse approximation reduces reconstruction error while making independent refinement harder.

This is a better starting point than “text has long-range dependencies.” Even independent source variables can acquire dependencies through residualization. NCM may repeatedly create constraints resembling

\[
\sum_i R_{k,i}\approx 0,
\]

or more general cancellation constraints determined by overlapping upsampling and convolution.

**Prediction:** conditional residual dependence should concentrate in combinations that preserve coarse aggregates or cancel reconstruction errors. Measure covariance and higher-order constraints of residual embeddings after conditioning on the parent state; compare canonical samples with independent samples from matched marginals. The relevant object is the *joint residual distribution*, not nearest-code distance.

That gives a concrete renormalization interpretation: coarse-graining does not simply remove detail. It changes the interactions among the remaining variables. A factorized next-level head discards those induced interactions.

---

**[speculation] The code-NLL gap has a useful exact decomposition—and it contains a third term your framing largely omits.**

For a generator with

\[
Q(C)=\prod_k\prod_i q_{k,i}(C_{k,i}\mid S_k),
\]

where \(S_k\) is a deterministic function of \(C_{<k}\),

\[
\begin{aligned}
\mathcal L(Q)-H(C)
={}&\sum_k \operatorname{TC}(C_k\mid S_k)\\
&+\sum_k I(C_k;C_{<k}\mid S_k)\\
&+\sum_{k,i}\mathbb E_{S_k}
D_{\mathrm{KL}}\!\left(
P(C_{k,i}\mid S_k)\,\|\,q_{k,i}(\cdot\mid S_k)
\right).
\end{aligned}
\]

The terms are:

1. **Dependence discarded inside a level.**
2. **Relevant ancestor information discarded by the conditioning state.**
3. **Error in predicting the available conditional marginals.**

The derivation is short: insert \(\sum_{k,i}H(C_{k,i}\mid S_k)\), collect each level’s total correlation, then use

\[
H(C_k\mid S_k)-H(C_k\mid C_{<k})
=I(C_k;C_{<k}\mid S_k).
\]

Include the root prior when applying the full identity. A cumulative sum followed by resizing need not be a sufficient statistic of the ancestor codes. It could be effectively sufficient; that is measurable, not guaranteed.

**[ours]** The small-model calculation is

\[
1232.7-173.6=1059.1,\qquad
1059.1/1232.7=85.9\%.
\]

**[speculation]** That is a lower bound on excess cross-entropy under the corresponding population interpretation. The finite validation estimate has sampling error. Applying the approximately 174-nat bound to the released model additionally requires the same text distribution, packing, and tokenization; re-score the AR model on the exact evaluation windows before presenting approximately \(861.8-173.6=688.2\) nats as a measured released-model bound.

The practical consequence is not “ignore fine codes.” It is:

> Determine which part of the excess is removable without changing the representation.

A sibling-conditioned probe targets the first term. An ancestor side channel targets the second. Better calibration or more training targets the third. These probes require different controls.

A positive held-out likelihood improvement from a trained sibling probe is useful evidence. It is not an unbiased estimate of total correlation unless the relevant models approximate their conditional distributions sufficiently well.

---

**[speculation] Clean code NLL and text quality can dissociate even with perfect measurement and no optimization bug.**

Let \(D(C)\) be the deterministic decoded text and

\[
Q_D(x)=\sum_{c:D(c)=x}Q(c).
\]

For a canonical code hierarchy \(c^*\) reconstructing \(x\),

\[
-\log Q(c^*)
=
-\log Q_D(x)
-\log Q(c^*\mid D(C)=x).
\]

Canonical-code training pays for two things:

- placing probability on the right decoded text;
- choosing the encoder’s particular representation of that text.

The second can be expensive without affecting decoded text probabilities.

More generally, the KL chain rule gives

\[
D_{\mathrm{KL}}(P_C\|Q_C)
=
D_{\mathrm{KL}}(D_\#P_C\|D_\#Q_C)
+
\mathbb E_{x\sim D_\#P_C}
D_{\mathrm{KL}}\bigl(P_C(\cdot\mid x)\|Q_C(\cdot\mid x)\bigr).
\]

This identity is exact for reconstructed texts \(D_\#P_C\). Identifying those with the original data distribution requires lossless reconstruction or an explicit approximation.

**[speculation] This supports a “decoder-equivalence” interpretation, but not a free symmetry claim.** Two nearby embeddings need not decode identically. Changing one code may require a compensating change at another level. Equivalence classes may be curved, disconnected, and context-dependent. Calling these directions “gauge freedom” is an analogy until you demonstrate output invariance.

This matters for A6. Suppose the valid representations of one text include only

\[
(c_1,c_2)=(a,b)\quad\text{and}\quad(b,a).
\]

Independent soft labels allocate probability to \((a,a)\) and \((b,b)\), which need not represent that text. **Softening each coordinate can destroy the joint equivalence class.** A6 therefore does not kill joint marginalization or coordinated alternative encodings.

**[ours]** A2/A4’s gains concentrate in late-level NLL while their sample outcomes differ sharply.

**[speculation]** Three explanations remain compatible:

- probability is redistributed within decoder-equivalent regions;
- better teacher-forced marginals coexist with worse ancestral state visitation;
- fine-level gradients damage the shared network’s coarse predictions.

The last has a cheap diagnostic. For an actual parameter update \(\Delta\theta\), estimate

\[
\Delta L_{\text{coarse}}\approx
\nabla_\theta L_{\text{coarse}}^\top\Delta\theta.
\]

Use the actual optimizer update, not raw gradient norm. If fine-level improvement systematically predicts positive coarse-loss changes, shared-parameter interference is real. It would explain why continued training can lower total NLL while damaging generation.

**[speculation] Gen-PPL adds another mismatch.** It evaluates samples under an external scorer; it is not \( -\log Q_D(X)\). Even a genuine improvement in model likelihood need not improve Gen-PPL, and a Gen-PPL improvement can result from narrowing the generated distribution.

---

**[speculation] The most convincing account of on-policy collapse is hidden channel identity followed by independent mode mixing.**

Let \(S\) be a clean conditioning state, \(Y\) its next-level target, and \(M\) indicate corruption. After corruption, the optimum of cross-entropy is

\[
P_{\rm train}(Y\mid\widetilde S=s)
=
a(s)P_0(Y\mid s)
+[1-a(s)]P_1(Y\mid s),
\]

where

\[
a(s)=P(M=0\mid\widetilde S=s).
\]

The posterior weight depends on the **whole context’s clean and corrupted densities**, not just the nominal replacement probability.

Two conditions are needed for harm:

1. clean and corrupted states overlap sufficiently that their provenance is ambiguous;
2. their appropriate target distributions disagree.

Plausibility alone is insufficient. Undetectable corruption whose targets remain compatible need not hurt.

**[source] This is related to the inconsistency of scheduled sampling, established by Huszár, 2015, arXiv:1511.05101.** Cross-entropy itself remains proper for the altered training joint; the problem is that this joint differs from the joint whose conditionals you want at inference. [Scheduled-sampling analysis](https://arxiv.org/abs/1511.05101)

**[speculation] A toy example shows why uniform replacement can be safer despite causing larger reconstruction damage.**

Let the clean context \(C\in\{0,1\}\) select one of two coherent continuations. Replace \(C\) with an independent draw from its own marginal with probability \(r\). Then

\[
P(Y=C_{\rm observed}\mid C_{\rm observed})=1-r/2.
\]

By contrast, replace it uniformly from a vocabulary of size \(V\), of which only two context values are plausible. At a plausible observed context, the posterior probability of no replacement is

\[
a=\frac{1-r}{1-r+2r/V}.
\]

For large \(V\), that is close to one. Grossly wrong contexts are easy to segregate; realistic wrong contexts contaminate the clean-context conditional.

This explains the direction of the hedge. It does not yet explain a collapse.

Now suppose the two coherent continuations are **AB** and **BA**. A joint posterior chooses one complete ordering. A factorized posterior independently mixes the two position marginals. Its duplicate probability is

\[
P(AA\text{ or }BB)=2a(1-a).
\]

For the independent replacement example with \(r=1/2\), \(a=3/4\), giving a **37.5% toy duplicate rate**. This is a deduction, not an NCM estimate.

For a coherent branch requiring \(m\) coordinated choices, independent sampling preserves branch consistency with probability

\[
a^m+(1-a)^m.
\]

Even moderate uncertainty becomes destructive when spread across many simultaneous decisions. A shared branch draw would preserve coherence while retaining exactly the same individual marginals.

**This is the missing bridge from hedging to collapse.** The problem may not be that the model becomes uncertain. It may be that the sampler cannot represent *what it is uncertain about*.

**[speculation] Why Sudoku need not collapse similarly:**

- clues sharply constrain the admissible global mode;
- uncertainty may be concentrated in decoder-insensitive code alternatives;
- greedy decoding may suppress some marginal uncertainty;
- the relevant dependency structure differs from local lexical ordering.

These are alternatives to test, not explanations established by the solve-rate difference.

**[speculation] Re-quantization cannot generally resolve this ambiguity.** It supplies the residual needed to reconstruct the original training example. If the observed coarse state plausibly describes a different example, the model still has to infer which original example generated it. Re-quantization fixes geometric bookkeeping; it does not restore lost information. A4 is therefore not a fatal counterexample to the ambiguity account.

**[speculation] The W2M readings should be more specific than “plausibility wins.”**

| Mechanism | Most diagnostic W2M comparison | What would genuinely weaken it |
|---|---|---|
| Hidden provenance plus incompatible targets | M2 versus M3; M4 versus M2 | Plausible-wrong is healthy, or truthful flags are demonstrably used but do not protect clean conditioning |
| Moving-policy feedback | M1 versus frozen M2, interpreted with their different wrong-code rates | Frozen plausible corruption remains similarly damaging |
| Canonical-versus-used parent mismatch | M6 versus M1 | Used-parent proposals remain damaging |
| Lost denoising coverage | M7 versus M0, using realized rates | Rate-matched uniform remains healthy |
| Precision-sensitive proposals | Reproduction of M1 under fp32 | Collapse remains, although that does not quantify the old bf16 contribution |
| Independent mode mixing | Not directly isolated by the eight arms | Joint coupling cannot improve decoded coherence despite demonstrably capturing the relevant dependence |

There are three important limitations.

First, **W2M changes precision and adds clean rows simultaneously** relative to the old collapse. If M1 becomes healthy, the experiment cannot identify which change rescued it.

Second, a low-AUC linear hidden-state probe does not establish undetectability. A high AUC demonstrates information is available; a low AUC may reflect a weak probe or a representation that already discarded it.

Third, inspect the flags before interpreting M4–M5. Ancestor contamination can saturate spatially. Under the nominal schedule, even a simplified path containing one ancestor at every corrupted level is entirely clean with probability only about

\[
\prod_{\ell=0}^{10}(1-r_\ell)\approx 2.66\times10^{-4}.
\]

Actual footprints differ, but this is a warning that late-level flags may mostly distinguish corrupted rows from clean rows. Report flag variance, saturation, and the fraction unchanged by shuffling. If truthful and shuffled flags are nearly identical, their comparison has little power to identify positional detection.

---

**[speculation] Parallel generation has a precise limitation, but “many tokens per pass” is not itself a fundamental impossibility.**

Given fixed state \(S\), independent categorical sampling produces

\[
Q(Y\mid S)=\prod_i q_i(Y_i\mid S).
\]

No amount of deterministic self-attention before those draws creates conditional dependence between the draws. Attention can coordinate the *probabilities*. It cannot coordinate independent realizations.

But there is no universal requirement to use one neural-network pass per dependent token. Shared randomness, structured sampling, or a sufficiently informative earlier latent can generate dependent outputs cheaply.

For example, \(m\) identical random bits have \(m-1\) bits of total correlation. One shared random bit generates them perfectly in parallel.

If a latent \(Z\) makes the outputs conditionally independent,

\[
\operatorname{TC}(Y\mid S)
=
\sum_i I(Y_i;Z\mid S)-I(Y;Z\mid S)
\leq (m-1)H(Z\mid S).
\]

This gives a necessary information budget for a shared latent. It is not sufficient.

A counterexample is a uniform distribution over even-parity binary strings. Its total correlation is only one bit, and for length at least three its pairs are independent. Yet every product-distribution component supported entirely on valid parity strings must be deterministic. An exact product-mixture representation consequently needs a latent identifying the full valid string.

Therefore:

- large TC can be solved by a tiny shared variable;
- small TC can conceal a difficult higher-order support constraint;
- pairwise dependency tests can miss the entire problem.

**[source] Zhang et al., 2026, arXiv:2602.00286, already distinguish forward total-correlation cost from reverse-KL incoherence in masked parallel decoding, with controlled Block-HMM and LLaDA arithmetic experiments.** This substantially reduces the novelty of merely saying “parallel sampling ignores dependence.” [Information-theoretic analysis](https://arxiv.org/abs/2602.00286)

**[speculation] Your distinctive question is whether residual quantization manufactures particular dependencies, and whether their structure permits cheap correction.**

The hierarchy helps when earlier levels expose a low-cost separator of downstream uncertainty. It merely relocates the problem when earlier codes themselves require difficult coordinated choices, or when residualization creates new cancellation constraints.

A better definition of a useful “concept” is therefore:

> A variable that makes the remaining generation problem easier to sample, at an acceptable cost to generate that variable.

Semantic interpretability is neither necessary nor sufficient.

---

**[speculation] Jacobi results concern dependency propagation and objective design, not necessarily iterative reasoning.**

With a fixed random seed for each categorical draw, NCM defines a triangular map

\[
F_k(C)=f_k(C_{<k}).
\]

After one Jacobi pass, the first generated level is correct; after two, the second is correct; and so on. Exact convergence follows by induction. It does not require a contraction argument.

A continuous relaxation’s Jacobian is strictly lower triangular and hence nilpotent. Its spectral radius can be zero while perturbations still require many passes to propagate through the dependency chain. Thus “make the map more contractive” is not automatically the right acceleration objective.

**[ours]** Sudoku sometimes reaches a correct intermediate board before converging to an incorrect sequential fixed point.

**[speculation]** That demonstrates that preserving the model’s fixed point and preserving task correctness are different goals. The verifier supplies information the fixed-point objective lacks.

**[ours]** J2 obtains some of its speed by mapping many sources to common strings.

**[speculation]** A constant map is an excellent consistency model: it converges immediately. Consistency alone rewards eliminating sensitivity to the source. J2-F tests whether anchoring the targets removes this degeneracy, but cannot guarantee useful speedup or eliminate student-state distribution shift.

For future analysis, measure both convergence and retained dependence on the source: source perturbation response, shared-output frequency, length, and teacher-output fidelity. A decrease in passes accompanied by loss of source sensitivity is a failure, however smooth the trajectory becomes.

---

**[speculation] The field has already occupied most straightforward extensions. The useful distinctions are below.** Numbers are included only where the setting is identifiable.

| Line | Verified result and scale | Implication for NCM |
|---|---|---|
| Visual next-scale generation | **[source]** Tian et al., NeurIPS 2024, arXiv:2404.02905: VAR’s 2B model reports FID 1.73 on class-conditional ImageNet-256. [VAR](https://arxiv.org/abs/2404.02905) | **[speculation]** Establishes viability of next-scale image generation, not conditional independence of text residuals. |
| Residual self-correction | **[source]** Han et al., CVPR 2025, arXiv:2412.04431: on the 5M-image, 512-resolution ablation, temperature 1 and CFG 3, FID is 9.76 baseline, 9.69 with random flips, 3.48 with bitwise self-correction. [Infinity, Table 7](https://arxiv.org/html/2412.04431v2) | **[speculation]** Re-quantized correction is established prior art. Its success is conditional on the representation and task; A2/A4 test an important transfer failure. |
| NCM itself | **[source]** Kumbong, Khare et al., 2026, arXiv:2610.14162: TinyStories GPT-2-large Gen-PPL is 15.41 at 12 reported NFE, versus sampled AR 8.89 at 128. [Table 8](https://arxiv.org/html/2610.14162v1) | **[speculation]** The central claim is a quality–speed trade-off at modest scale. Your implementation’s 11 transformer calls should be reconciled explicitly with the paper’s convention. |
| Other hierarchical representations | **[source]** Yu et al., 2023, arXiv:2305.07185, MEGABYTE: local/global byte modeling, including million-byte sequences. Hwang, Wang and Gu, 2025, arXiv:2507.07955, H-Net: learned dynamic chunks. Barrault et al./LCM team, 2024, arXiv:2412.08821: sentence-representation prediction, explored at 1.6B and scaled to 7B. [MEGABYTE](https://arxiv.org/abs/2305.07185), [H-Net](https://arxiv.org/abs/2507.07955), [LCM](https://arxiv.org/abs/2412.08821) | **[speculation]** Hierarchy is not synonymous with parallel sampling. These do not establish an NCM-style residual text hierarchy’s advantages. I did not verify another sufficiently close independent residual-text result to treat as direct supporting evidence. |
| Residual-code generation in audio | **[source]** Wang et al., 2023, arXiv:2301.02111, VALL-E: residual codec generation trained on 60K hours of English speech; its architecture combines an AR first stage with non-AR residual stages. [VALL-E](https://arxiv.org/abs/2301.02111) | **[speculation]** A useful precedent for making the coarse stage responsible for difficult dependence. Speech conditioning and acoustic distortion make direct numerical transfer inappropriate. |
| Principled discrete likelihood training | **[source]** Lou, Meng and Ermon, ICML 2024, arXiv:2310.16834: SEDD estimates discrete probability ratios. Sahoo et al., NeurIPS 2024, arXiv:2406.07524: MDLM simplifies masked diffusion training and supports semi-autoregressive sampling on language-modeling benchmarks. [SEDD](https://arxiv.org/abs/2310.16834), [MDLM](https://arxiv.org/abs/2406.07524) | **[speculation]** Proper objectives do not guarantee accurate large parallel sampling steps. NCM corruption is not automatically a valid reverse-diffusion process. |
| Uniform diffusion and consistency | **[source]** Sahoo et al., ICML 2025, arXiv:2506.10892: DUO uses a Gaussian–uniform diffusion connection and discrete consistency distillation; main language models are 170M on LM1B/OWT. Deschenaux, Gulcehre and Sahoo, 2026, arXiv:2602.21185: predictor–corrector samplers improve the OWT quality–entropy frontier. [DUO](https://arxiv.org/abs/2506.10892), [Ψ-samplers](https://arxiv.org/abs/2602.21185) | **[speculation]** “Uniform noise is crude” is not a useful argument. Noise geometry and the correctness of the associated sampling process matter. |
| Scaling beyond small benchmarks | **[source]** Nie et al., 2025, arXiv:2502.09992: LLaDA 8B trained from scratch. Ye et al., 2025, arXiv:2508.15487: Dream 7B uses AR initialization. Sahoo et al., 2026, arXiv:2602.15014: matched diffusion-family scaling to 1.7B finds likelihood ranking can disagree with practical sampling preference. [LLaDA](https://arxiv.org/abs/2502.09992), [Dream](https://arxiv.org/abs/2508.15487), [Scaling study](https://arxiv.org/abs/2602.15014) | **[speculation]** A TinyStories result needs a structural explanation or transfer experiment to matter beyond its benchmark. “Likelihood and sample quality differ” alone is already established. |
| Revision and block generation | **[source]** Wang et al., NeurIPS 2025, arXiv:2503.00307: ReMDM adds principled remasking to pretrained masked models. Arriola et al., ICLR 2025, arXiv:2503.09573: block diffusion combines diffusion within blocks with sequential blocks and KV caching. [ReMDM](https://arxiv.org/abs/2503.00307), [Block diffusion](https://arxiv.org/abs/2503.09573) | **[speculation]** Generic remasking, blockwise generation, and iterative correction are crowded. A residual-specific support constraint would be more distinctive. |
| Exposure-bias remedies | **[source]** Agarwal et al., ICLR 2024, *On-Policy Distillation of Language Models*: GKD gets teacher feedback on student-generated sequences. [GKD](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf) | **[speculation]** On-policy training is not inherently inconsistent. Supervising the encountered context differs fundamentally from retaining an incompatible original future. |
| Jacobi and learned consistency | **[source]** Santilli et al., ACL 2023: fixed-point MT decoding. Kou et al., ICML 2024, arXiv:2403.00835: CLLMs train rapid convergence to trajectory fixed points. Hu et al., 2025, arXiv:2512.14681: Jacobi Forcing extends trajectory training to Qwen2.5 7B coding/math models. [MT Jacobi](https://aclanthology.org/2023.acl-long.689/), [CLLMs](https://arxiv.org/abs/2403.00835), [Jacobi Forcing](https://arxiv.org/abs/2512.14681) | **[speculation]** J2’s broad method is established. Its collapse could support a useful objective-design analysis, but another consistency variant needs a compelling new mechanism. |
| Exact parallel verification | **[source]** Leviathan, Kalman and Matias, ICML 2023, arXiv:2211.17192: exact speculative sampling. Fu et al., ICML 2024, arXiv:2402.02057: Lookahead verification of Jacobi proposals. Chen, Liang and Liu, ICML 2026, arXiv:2602.06036: DFlash uses a parallel block-diffusion drafter. [Speculative decoding](https://arxiv.org/abs/2211.17192), [Lookahead](https://arxiv.org/abs/2402.02057), [DFlash](https://arxiv.org/abs/2602.06036) | **[speculation]** “Parallel drafts plus exact AR verification” is already occupied. NCM also has an awkward many-to-one code-to-text proposal probability; code likelihood cannot simply replace text proposal likelihood in an acceptance ratio. |
| Adaptive commitment and stopping | **[source]** Israel, Van den Broeck and Grover, 2025, arXiv:2506.00413: APD combines diffusion marginals with an auxiliary AR joint. Bao et al., 2025, arXiv:2509.25188: Learn2PD predicts whether tokens match final outputs. Lee et al., 2026, arXiv:2607.28166: confidence plus temporal agreement for global stopping on LLaDA/Dream. [APD](https://arxiv.org/abs/2506.00413), [Learn2PD](https://arxiv.org/abs/2509.25188), [C⁴](https://arxiv.org/abs/2607.28166) | **[speculation]** Confidence stopping and small learned filters are not fresh directions. Any NCM rule must beat capped decoding and account for its larger parallel passes. |

**[source] The closest older obstacle to claiming novelty for my sampler proposal is structured non-autoregressive decoding.** Sun et al., NeurIPS 2019, arXiv:1910.11555, already add efficient CRF structure to non-AR translation. [Fast Structured Decoding](https://arxiv.org/abs/1910.11555)

**[speculation] Consequently, the new claim cannot be “add pairwise interactions.”** It must be the causal separation obtained by fixing all marginals, identifying residual-induced dependence, and demonstrating when that dependence changes decoded quality.

---

**[speculation] Here are ten ideas worth considering before narrowing the programme.** Probabilities are subjective chances of a clear, useful pilot result under this budget—not probabilities of publication. “New” means new relative to your queue; I am not claiming exhaustive priority clearance.

| Idea | Mechanism, prediction, and affordable first experiment | Probability |
|---|---|---|---:|
| **1. Couple the draws, freeze the predictions** | Learn joint sampling with exactly the existing categorical marginals. Better text with unchanged marginal CE would isolate dependence. Frozen-feature pilot. | 35% |
| **2. Enforce residual conservation laws** | Learn which sums or projections of residual embeddings must cancel; sample codes subject to those constraints. Predict reduced off-manifold residual combinations before any text gain. CPU analysis, then a small head. | 30% |
| **3. Treat duplicates as an allocation failure** | A coarse bag predicts a lexical inventory; a small balanced allocation module assigns its mass to positions instead of drawing counts independently. Predict lower count variance and duplication at fixed lexical coverage. Start on synthetic bags and short TinyStories spans. | 25% |
| **4. Measure irreversibility, not uncertainty** | Estimate whether remaining codebooks can repair the current state, using oracle residual fitting or a convex relaxation. Spend computation reopening an ancestor only when repair is infeasible. Test reachability on saved failed trajectories. | 30% |
| **5. Decoder-aware geometry** | Train small adapters using distances induced by decoder changes, rather than Euclidean codebook distance. Predict that equal-distance substitutions differ greatly in functional damage. Start with finite-difference substitution tests. | 40% |
| **6. Match damage while rotating target conflict** | Construct donor corruptions with similar reconstruction damage and detectability but different compatibility with the retained future. Predict gradient damage tracks incompatibility, not visible corruption magnitude. Small frozen-state/short-update experiment. | 50% |
| **7. Protect coarse predictions from fine gradients** | Constrain fine-level updates against a small clean coarse-state reference set. Predict A1-like coarse deterioration disappears while some fine NLL improvement remains. A 100-step comparison can reject it. | 30% |
| **8. Subtractive dither for residual codes — high risk** | Shared, known dither can decorrelate quantization errors; compensate it during reconstruction. Test on a tiny frozen-codebook system before touching the released model. It may fail because the learned decoder depends on the original quantization geometry. | 10% |
| **9. Generate a bag, then a permutation — high risk** | Change the representation: lexical choice first, ordering as a structured permutation problem. Start with 16–32-token synthetic/templated language and a small model. This could expose whether syntax, rather than semantics, is the expensive shared variable. | 15% |
| **10. Map the response to small interventions** | Perturb a coarse code and measure how predictive distributions and decoded relations change across levels. Look for a few high-impact directions rather than level-average sensitivity. Predict those directions explain collapse better than level NLL totals. CPU/Sudoku first. | 45% |

**[speculation] The two high-risk ideas are dismissible for sensible reasons, but not for decisive ones.** Dither sounds inappropriate for learned language codes, yet quantization-induced dependence is exactly where it could help. Bag-plus-permutation sounds linguistically crude, yet your own coarse representation already behaves approximately like a bag; explicitly modeling the missing ordering problem may be more honest than repeatedly residualizing it.

I would not launch these ten experiments. They define alternatives if the two selected directions fail.

---

**[speculation] Ranked first: repair dependence while preserving every marginal.**

**Core idea:** freeze the tokenizer and generator, including its logits. Add a small structured sampler that changes which codes occur together while retaining each position’s original conditional distribution.

For a pair of positions with probabilities \(p(a)\) and \(q(b)\), learn compatibility scores \(A_\phi(a,b;s)\), then construct

\[
\Pi_\phi(a,b\mid s)
=u_a\,e^{A_\phi(a,b;s)}v_b
\]

with row sums \(p\) and column sums \(q\). Matrix scaling supplies \(u,v\). At zero compatibility, \(\Pi=pq\).

A practical first version should avoid \(4096^2\) tables:

- partition each codebook into a fixed, modest number of groups;
- couple group choices;
- sample the exact original conditional distribution within each selected group.

This preserves the full categorical marginals, apart from a measured numerical normalization tolerance. Start with disjoint neighboring pairs at L1–L5. All pairs can be sampled concurrently. Do not begin with a general loopy graphical model.

Train the small head on canonical training pairs using joint likelihood. The backbone remains frozen.

**Why it could matter:** the experiment changes dependence while holding marginal modeling fixed. The legacy factorized clean-code CE is exactly unchanged; the *actual joint sampler likelihood* changes and must be reported separately. Downstream states and unconditional output marginals can change, because earlier joint samples change.

**Why it could fail:** the harmful constraints may be higher-order, nonlocal, or already lost in the marginal distributions. Canonical pair dependence may also describe decoder-irrelevant bookkeeping.

**Proposed success criterion and expectation**

- Primary mechanistic target: criterion **(c)**, sharpened to *a reproducible causal effect of residual-code dependence at fixed conditional marginals*.
- Method target: beat both the frozen released model and an equal-training-compute continuation control on the full text suite, with the same transformer-call count and measured inference overhead at most 10%.
- **Expected useful effect if it works:** roughly 0.5–1.5 Gen-PPL improvement and at least 20% fewer adjacent duplicates.
- **Subjective probability:** 25% for that method-sized result; about 40% for a useful mechanistic result; below 10% for closing half the AR gap.

A more ambitious result would show that the same dependence intervention improves text and a second domain while preserving the original marginal predictors. That would support a general sampling principle rather than an NCM patch.

**Kill experiment: at most 5 CU**

1. Cache frozen training features and marginals for a bounded training subset.
2. Fit one small coupling head, with an independence initialization.
3. Evaluate joint predictive gain, marginal preservation, unconditional text, and batch-1 latency.
4. Include an inference ablation that destroys learned pair alignment while retaining its marginal constraints.

Pre-register:

- positive held-out joint log-score gain;
- at least **0.35 Gen-PPL** improvement at \(T=1\);
- no reversal into clear damage at \(T=0.9\);
- at least **15% duplicate reduction**;
- no material decline in MAUVE, length, lexical coverage, or the second scorer;
- marginal normalization error below a fixed numerical tolerance.

Kill the method if joint likelihood improves but decoded quality does not, or if the useful gain requires more than 10% measured latency overhead. That negative result would indicate that the captured dependence is nuisance, insufficient, or too expensive.

**Confirmation plan: approximately 36 CU, including the pilot**

| Component | Seeds/arms | Estimated CU |
|---|---|---:|
| Kill pilot | Exploratory | 5 |
| Marginal-preserving head | 3 fresh seeds, about 2 CU each | 6 |
| Matched small unconstrained structured head | 3 fresh seeds, about 2 CU each | 6 |
| Uniform continuation control | 3 matched seeds, about 3.5 CU each | 10.5 |
| Final text suite, alignment ablations, latency | All arms | 3.5 |
| Sudoku transfer with small heads and frozen backbone | 3 seeds; bounded screen/confirmation | 5 |
| **Total** | | **36** |

These are cost estimates, not throughput measurements. Charge feature extraction and all auxiliary computation. Match the declared training-FLOP budget explicitly; identical CU expenditure is not compute matching. Use the pilot to calibrate the feasible head-training budget before registering confirmation.

**Strongest reviewer objection:** “This is a small CRF, already known to help non-AR models.”

**Answer:** include the unconstrained structured-head comparator. The scientific claim is the controlled intervention: unchanged marginals, changed dependence, measured residual constraints, and downstream effects. If the only result is that a generic CRF helps, the novelty is weak.

**Connection to your results:** the oracle curve motivates testing L1–L5; the NLL/sample dissociation makes marginal preservation unusually informative; the collapse predicts stronger gains where uncertainty mixes incompatible local orders. J2 motivates keeping the generative backbone fixed so the model cannot obtain apparent improvement by deleting dependence on its input.

---

**[speculation] Ranked second, and safer: establish a causal law for corruption ambiguity and parallel incoherence.**

**Core idea:** distinguish uncertainty about *which coherent continuation is intended* from uncertainty independently attached to each output position. Test whether ambiguous corruption harms NCM chiefly because it increases the former while the sampler represents only the latter.

This is more ambitious than confirming “plausible noise hurts.” The desired result connects:

\[
\text{provenance ambiguity}
\longrightarrow
\text{loss of conditional specificity}
\longrightarrow
\text{incompatible joint samples}.
\]

The first arrow is close to scheduled-sampling theory. The second is where your representation and sampler can supply a new mechanistic result.

**Proposed success criterion and expectation**

Criterion **(c)**, strengthened to require:

- a controlled corruption intervention and rescue across three fresh training seeds;
- a pre-registered intermediate quantity that predicts the outcome;
- a second-domain test of the intermediate mechanism;
- explicit identification of where the full text-collapse mechanism does *not* transfer.

Expected effect, conditional on W2M reproducing collapse: truthful provenance information removes at least half the excess Gen-PPL and restores a substantial fraction of clean-context dependence. I would assign roughly **50% probability to a useful causal finding conditional on the W2M validity gate**, and substantially less to the complete parallel-amplification account.

**Kill experiment: CPU plus at most 2 CU**

Use W2M checkpoints and fresh validation examples before launching replication:

1. Check that flags provide a nontrivial intervention: saturation, spatial variance, true-versus-shuffled differences.
2. Measure clean-context dependence separately from output uncertainty and local-order coherence.
3. Fit the simplest two-mode mixture description on a development subset and test its predictions on untouched validation examples.
4. Use a synthetic residual construction with known joint probabilities to verify that the proposed measurements distinguish hidden-provenance hedging from ordinary marginal error.
5. If direction 1’s sampler is available, test whether coupling improves collapsed checkpoints **without** restoring their marginal confidence.

Kill the strong account if:

- the collapse does not reproduce under W2M;
- damaging arms do not lose useful conditional specificity;
- a genuine provenance intervention restores specificity but leaves local incoherence essentially unchanged;
- or the supposedly relevant joint uncertainty cannot be detected by a validated instrument.

Do not replace the failed account with “the mechanism is complicated” and continue spending.

**Confirmation plan: approximately 46 CU**

Assuming the W2M screen supports the intended contrast, run **four arms on three fresh seeds**:

- frozen plausible-wrong;
- uniform-wrong;
- plausible-wrong with truthful flags;
- plausible-wrong with shuffled flags.

At W2M’s estimated costs, training is approximately

\[
3(3.7+2.9+3.7+3.7)=42\ \text{CU}.
\]

Reserve approximately 4 CU for final evaluation, diagnostic work, and the bounded kill test. Use CPU Sudoku experiments for cross-domain mechanism checks.

If W2M instead points to parent mismatch or lost coverage, replace this set with the smaller decisive comparison and preregister that change before confirmation. Do not replicate irrelevant arms to preserve the original narrative.

**Strongest reviewer objection:** “You trained on mismatched labels and rediscovered scheduled sampling.”

**Answer:** the claim must exceed that observation. Demonstrate which uncertainty is introduced, how residual coding makes it consequential, why factorized sampling amplifies it, and why the same hedge has different consequences on Sudoku. If you cannot establish the amplification link, publish the narrower result honestly; do not describe a general collapse theory.

The existing oracle, corruption, and Jacobi results become complementary evidence: coarse states select continuations; late codes can conceal corruption in reconstruction; parallel sampling can break local coherence; fixed-point objectives can suppress dependence altogether.

---

**[speculation] I would allocate the remaining budget as a shared programme, not two independent promises.**

| Allocation | CU |
|---|---:|
| Marginal-preserving sampler programme | 36 |
| Mechanistic confirmation programme | 46 |
| Reserve for failed runs, evaluation, and cost error | 8 |
| **Maximum** | **90** |

All runs remain below 15 CU. All GPU work remains contingent on registration, independent review, and the required audit. The plans use three fresh confirmation seeds; the exploratory seed is not silently promoted into a confirmatory estimate.

If both pilots fail, stop. Spending the full budget is not an objective.

**[speculation] Criterion (c) is much more reachable than your other criteria, but currently too underspecified.** “Levels encode bags of words” is not a sufficiently strong endpoint. A useful version of (c) requires a quantitative mechanism, an intervention that changes it, a predicted behavioral consequence, and replication.

Criterion (a) is attainable for a modest method, but “better across the full suite” should mean a declared primary improvement with predeclared noninferiority margins elsewhere—not every noisy metric improving numerically. Any change to the formal bar should be agreed before confirmation.

Criterion (b) is currently expensive because the matched AR comparator itself consumes much of the budget. Also specify the gap scale: using the paper’s numbers, the arithmetic Gen-PPL midpoint is **12.15**, whereas the midpoint in log perplexity corresponds to approximately **11.70**. Those are different targets. Neither can be mixed with your evaluation protocol without re-evaluation.

Criterion (d) is closer in logical passes than your failed J2 might suggest, but less attractive in actual compute. **[ours]** WMT14 at cap 8 loses 1.11 BLEU; cap 9 loses 0.10. **[speculation]** A simple validation-selected rule allocating the ninth pass to a minority of examples is a mandatory baseline before further training. Its wall-clock and attention-inclusive cost may still disqualify it.

---

**[speculation] I would make the following queue decisions.**

- **Finish J2-F once.** It is already running and answers a concrete question about moving targets. If it prevents collapse but fails the quality–latency frontier, stop the consistency line. Do not launch another teacher or loss variant merely because the output is healthier.
- **Keep W2M as the bounded discovery screen.** Before interpreting it, add the inexpensive flag-saturation and probe-power diagnostics. Preserve its registered outcomes. If M1 does not reproduce collapse, isolate precision versus clean-row exposure before making causal claims.
- **Do not replicate all eight W2M arms.** Replicate only the deciding contrast, with fresh seeds and a fixed causal statement.
- **Cut the held tier-M from-scratch pair.** It buys a new scale-specific baseline rather than answering the strongest question exposed by the released checkpoints.
- **Cut the held Sudoku training arms.** Keep Sudoku as a cheap mechanistic and timing testbed. The weak recurrent result is not a general refutation, but there is no budgetary reason to reopen it now.
- **Stop using total code NLL as the principal method-selection signal.** Retain it as a diagnostic, separated into marginal prediction, joint dependence, and decoder relevance.
- **Stop expanding candidate lists after a failed screen.** One sampler pilot and one decisive mechanism programme are enough.

**[speculation] The central wager is that NCM often knows the plausible alternatives but draws incompatible pieces of them independently.** Your most valuable experiment is one that can prove that statement wrong while leaving the generator’s individual predictions untouched.
