# Research review, round 2 — 2026-10-27

- **[speculation] 1. Keep (c) primary, but demand an intervention.** A large canonical-code/marginal-text likelihood gap is an accounting identity plus a measurement; a mechanism needs a manipulation that changes the gap, dependence, or provenance while controlling competing explanations.
- **[speculation] 2. Repair E1 before interpreting it.** Its population quantity is sound. Its stopped hit-rate estimates, uncertainty, and claim that likelihood rankings should match generation metrics are not. The 80% threshold is extremely undemanding at hundreds of nats.
- **[speculation] 3. Withdraw my earlier positive-interaction prediction for E2.** Mode mixing can give either sign. Replace the sign gate with a marginal-preserving coupling intervention; keep the original factorial experiment as a decoder-response diagnostic.
- **[speculation] 4. M4 versus M2 cannot identify a provenance mechanism by itself.** Add a row-shuffled flag control if M4 rescues. Within-level shuffling supplies almost no control when flags saturate. Regime-dependent bias and provenance conditioning are overlapping explanations, not mutually exclusive ones.
- **[speculation] 5. Spend replication money on three fresh seeds of the deciding contrast.** Two new seeds plus a separately reported, selection-driving seed are not three independent confirmatory seeds. Preserve the existing registration's verdict; distinguish it from a stronger confirmation.
- **[speculation] 6. Give D1 its gate, then stop if it fails.** Require the fastest correct sequential implementation, include every tokenizer decode, and compare J2-F stopping against J0 stopping and matched-latency caps. A failure kills this implementation and operating point, not all text Jacobi methods.
- **[speculation] 7. Do not revive the coupled sampler by lowering S0's threshold after seeing its result.** A neighbor probe needs a context-only adaptation control before its gain can be called dependence. A new specificity experiment can earn a different, explicitly prospective pilot.
- **[speculation] 8. Keep the tier-M and Sudoku-training cuts.** Allocate the likely spare compute to causal controls, a third fresh seed, and one bounded intervention pilot. Do not fill the budget with another generation method before establishing what the existing diagnostics identify.

**[ours] Evidence boundary.** The supplied brief is the main empirical record. The remote repository additionally contains R044–R046: R044 approves domain/checkpoint replication for fixed released-model properties; R045 reports S0's 6.22 nats/window gain and failed gate; R046 reports a three-window E1 prototype and registrations including cached-prefix timing. These are repository-reported observations, not independently recomputed results in this review. I mark them explicitly when used. No W2M arm outcome was inspected. **[speculation]** All numerical forecasts below concern unobserved outcomes; proposed changes are recommendations, not silent amendments to existing registrations.

**1. Plan verdict: aim for an identified mechanism, not a collection of dissociations**

**[speculation]** Criterion (c) is the right target. The publishable claim should be narrower and stronger than “codes contain redundancy”: *which parts of the code loss measure canonical representation choice, which interventions expose harmful unresolved dependence, and when conditioning on corruption provenance prevents that harm*. These are separable claims; one convincing causal result is worth more than three suggestive plots.

**[speculation]** A hostile reviewer will dismiss (i) the existence of a many-to-one decoder, (ii) canonical likelihood disagreeing with BLEU, (iii) noisy inputs hurting training, and (iv) a learned stopping rule exploiting a post hoc favorable checkpoint. What is non-trivial is quantitative scope plus a falsifying control: hundreds of nats assigned to the same *exact* text under the actual ancestral model; a dependence-only intervention with unchanged marginals; or a rescue that requires informative rather than arbitrary regime labels.

**[ours]** The brief gives A4 the best clean NLL and poor samples, while its oracle-prefix experiments already show why fine-level improvements need not repair coarse sampling. **[speculation]** E1 cannot, by itself, turn that observation into a causal explanation of A4's failure. The title “why code likelihood misleads” is defensible as a measurement result; “why on-policy training collapses” needs W2C and the controls below.

**[speculation]** Reorder: cheap E1 estimator checks and W2M probe correction first; D1 next or concurrently when a GPU is available; read W2M; select one causal contrast; replicate it; only then spend on a method pilot. Under the repository's approved replication interpretation, E1 can establish properties of the three released checkpoints without pretending they are independently trained seeds. Generalizing a training intervention still needs training replication. Bootstrap windows are not substitute seeds.

**[speculation] Numerical prediction and reversal.** My probability of a defensible mechanistic result within the post-W2M budget is about 65%; of satisfying (d) under the fp32 timing rule, below 10%. I would reverse the prioritization if D1's paired latency-ratio interval lies wholly below 0.75 and an untouched evaluation confirms less than 1 BLEU loss. For (c), require a pre-specified causal effect of at least 50% rescue of the relevant damage, replicated at three fresh seeds, with gain exceeding twice pooled seed SD; report paired seed differences and all seeds as well. This is a proposed stronger standard, not a claim that the existing plan already meets it.

**2. E1: the identity is correct; its estimability and interpretation need correction**

**[speculation] Exact object.** Fix the source if conditional, the canonical prefix a = c*<k, suffix z* = c*>=k, and deterministic rendered output y = D(c*). Define the decoder fiber F_a(y) = {z : D(a,z) = y}. With q* = Q(z* | a),

\[
\rho_k=\sum_{z\in F_a(y)}Q(z\mid a),\qquad
R_k=\log\frac{\rho_k}{q^*}
    =-\log Q(z^*\mid a,D(a,Z)=y).
\]

**[speculation]** Thus R_k is exactly the model's posterior surprisal of the encoder-selected suffix, conditional on the prefix and decoded output. It is not the entropy of the deterministic encoder, and not an intrinsic number of “wasted bits” in the representation. The encoder's posterior is a point mass; the model's posterior need not be. A model can assign tiny probability to that one path while assigning substantial probability to its decoder fiber. There is no contradiction with H(C) <= H(X).

**[speculation]** A useful chain-rule identity makes the distinction sharper. Let P_C be the empirical/data distribution of deterministic encoder outputs and P_Y its pushforward through D. Then

\[
D_{KL}(P_C\Vert Q_C)=D_{KL}(P_Y\Vert Q_Y)
 +\mathbb E_{P_Y}D_{KL}(P_{C\mid Y}\Vert Q_{C\mid Y}).
\]

**[speculation]** When reconstruction is injective over the data under consideration, P(C | Y) is a point mass and the second term is the average full-fiber canonical surprisal. With reconstruction collisions it also contains the data posterior's entropy correction. Either way, E1 measures a model/data mismatch *inside decoder fibers*, not an independent source of data entropy. The conditional-at-cut version clamps a particular prefix and consequently measures only part of that mismatch.

**[speculation] Bounds and a useful invariant.** Let A_k = -log Q(a), L = A_k + N_k, and Q_Y be the full decoded-text distribution. Then

\[
0\le T_k\le N_k,\quad 0\le R_k\le N_k,\quad
-\log Q_Y(y)\le U_k=A_k+T_k=L-R_k\le L.
\]

**[speculation]** More precisely,

\[
R_{\rm full}=L+\log Q_Y(y)
=R_k-\log Q(C_{<k}=a\mid Y=y)\ge R_k.
\]

**[speculation]** R_k is a lower bound on the *full* gap because the prefix is fixed. Exact token equality makes the fiber definition precise; it is not the reason for this lower bound. For a single model/window, R_k must be non-increasing as k increases, and U_k non-decreasing. This follows because the posterior event “canonical prefix of length k” becomes smaller as the prefix grows. T_k itself need not be monotone. Violations beyond simultaneous estimation uncertainty expose an implementation or inference problem.

**[speculation]** The bound is on y = reconstructed text, not necessarily the original x. Report reconstruction failures separately. For WMT, equality after EOS/pad removal defines a rendered-string distribution, which is legitimate, but not the same random variable as all padded token IDs. Apply one fixed rendering map to canonical and sampled outputs. A stochastic tokenizer decoder would require summing Q(c)P_D(y | c), not counting deterministic equality; the proposed derivation assumes the actual evaluation decoder is deterministic.

**[speculation] Why the 80% threshold is weak.** From the brief's 861.8 nats and 46.8% share, mean N_8 is approximately 403.3 nats. R_8 >= 0.8N_8 requires only T_8 <= 80.7, or a fiber probability around 10^-35 at that illustrative N. An observable hit frequency with 2,048 samples resolves probabilities many orders of magnitude larger. Any statistically credible hit can therefore establish a spectacular-looking R fraction without accurately estimating a very rare probability. The informative quantities are the *absolute* T, coverage of windows, prefix cost A, and whether model differences remain inside fibers.

**[ours; repository update]** R046 already reports cut-8 prototype probabilities roughly 0.08–0.94 in three windows, with no N_k recorded. **[speculation]** “Cut 8 often has hits” is now observed pilot information, not a fresh prediction. Exclude those three windows from a genuinely prospective extension, or mark them as pilot-included in the registered descriptive analysis. Do not present a new prediction calibrated on them as preregistered before evidence.

**[speculation] Hit-rate estimation.** At fixed n, H/n is unbiased for rho, but -log(H/n) is not. For every rho < 1 it has infinite unconditional expectation because H = 0 has positive probability. In the high-count regime, ignoring that zero event, the delta-method bias and variance are approximately

\[
\operatorname{bias}(\widehat T)\simeq\frac{1-\rho}{2n\rho},\qquad
\operatorname{var}(\widehat T)\simeq\frac{1-\rho}{n\rho}.
\]

**[speculation]** With exact stopping at r successes, the stopping time tau has a negative-binomial law. r/tau is upward-biased for rho; (r-1)/(tau-1) is unbiased for rho when r > 1 and there is no cap. The estimator

\[
\widehat T_{\rm IBS}=H_{\tau-1}-H_{r-1}
=\psi(\tau)-\psi(r)
\]

**[speculation]** is unbiased for -log rho under that uncapped stopping scheme. Differentiating the normalization of the negative-binomial distribution proves the digamma identity. This is not a justification for the current capped, batched estimator: stopping only after batches overshoots r, and censoring at 2,048 changes the law. At r = 20 the characteristic log-scale standard error is roughly 0.22 nats in a rare-event regime; the bias of the naive stopped log estimate is of order 1/r, with neither its sign nor its exact value established by saying “about 1/20.”

**[source]** Bas van Opheusden, Luigi Acerbi and Wei Ji Ma, 2020, *PLOS Computational Biology*, DOI 10.1371/journal.pcbi.1008483, develop inverse-binomial log-likelihood estimation for discrete simulator outcomes. Their result supports using the appropriate stopping-law estimator; it does not validate a hard-capped, batch-overshooting substitution. [Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008483).

**[speculation] Recommended registration amendment.** Keep adaptive sampling, but make confidence bounds primary. For Bernoulli hit count h at time t, a beta-mixture likelihood ratio is

\[
M_t(p)=\frac{B(h+\alpha,t-h+\beta)}
 {B(\alpha,\beta)p^h(1-p)^{t-h}},\quad \alpha=\beta=1/2.
\]

**[speculation]** Under any fixed candidate p, this is a nonnegative mean-one martingale. The set M_t(p) < 1/delta is an anytime-valid confidence set, including at batchwise stopping times. Intersect its interval hull with [exp(-N_k),1], since the canonical suffix is always in the fiber. If the resulting interval is [p_L,p_U], report

\[
T_k\in[-\log p_U,-\log p_L],\qquad
R_k\in[N_k+\log p_L,N_k+\log p_U].
\]

**[source]** Steven Howard, Aaditya Ramdas, Jon McAuliffe and Jasjeet Sekhon, 2021, *Annals of Statistics*, arXiv:1810.08240, develop time-uniform confidence sequences and mixture constructions. The displayed Bernoulli specialization is the proposed implementation here; no empirical NCM claim comes from that paper. [Paper](https://arxiv.org/abs/1810.08240).

**[speculation]** Allocate error probability across the finite set of model/window/cut bounds used for a joint claim, or derive a separately valid aggregate procedure. Average window bounds with the registered equal window weights. Keep canonical masses and candidate sums in log space: exp(-N) can underflow precisely where its bound matters. Do not average only hit windows, weight by their draw counts, or replace uncertain T with a point estimate and then bootstrap as if known. For checkpoint differences, propagate both endpoints; declare the proposed equivalence only when the entire interval for delta T lies inside +/-0.2|delta N|. Report an absolute difference as well when delta N is near zero. Window uncertainty, Monte Carlo uncertainty, and training-seed variability are different quantities.

**[speculation] Zero hits are not infinite loss.** With zero hits in a *fixed* 2,048 draws, the one-sided 95% binomial bound is rho <= 1 - 0.05^(1/2048), about 0.00146; hence T >= 6.53. Together with the canonical-path bound, T lies in [6.53,N], and R in [0,N-6.53], before simultaneous-coverage adjustment. This is often almost uninformative about hundreds of nats. The registration's rule-of-three endpoint is an approximate single-interval statement, not an exact value to insert into an aggregate confidence claim. Hit windows also need uncertainty. For adaptive reporting, use the confidence sequence consistently.

**[speculation] Cheap bounds when hits vanish.** Use a finite, deduplicated set B of verified exact-text suffixes containing z*. Compute each full ancestral suffix probability by teacher forcing its *own* earlier sampled levels, not the canonical earlier levels. Then

\[
\rho_k\ge\sum_{z\in B}Q(z\mid a),\qquad
R_k\ge\log\sum_{z\in B}\exp[N_k+\log Q(z\mid a)].
\]

**[speculation]** This lower bound is deterministic once the candidates are verified. Candidates can come from compensated code edits, local search, or re-quantization; using x for inference is legitimate here because this is likelihood evaluation, not unconditional generation. A thousand equally probable alternatives buy only log(1000) nats. A dramatic bound needs either far more mass or alternatives much more likely than the canonical path. Do not mistake finding many near-identical codes for finding substantial probability mass.

**[speculation]** A normalized proposal g(z | a,y) gives the unbiased importance estimate E_g[1_F Q/g]. A defensive mixture g = epsilon Q + (1-epsilon)g_local ensures support and bounds weights by 1/epsilon. The proposal density, mixture atoms, and full-path probabilities must be computable; self-normalizing the weights estimates a different quantity here. This can improve efficiency, but cannot guarantee discovering disconnected fibers. Appending the canonical path to ordinary hit counts is invalid.

**[speculation]** The variational alternative is log rho >= E_r[log Q(z|a)-log r(z)] for a normalized r supported entirely on the exact fiber. Optimizing r over a finite verified set gives precisely the log-sum bound above. A distribution that spills outside the hard fiber has a negative-infinite hard-event ELBO. Replacing equality by exp(-beta distance) makes optimization easier but changes the estimand until a justified limit is taken. This is why annealing is not a free solution to coarse-cut rarity.

**[source]** Radford Neal, *Annealed Importance Sampling*, arXiv:physics/9803008 (1998 preprint), constructs weighted estimates through intermediate distributions and Markov transitions. **[speculation]** For this project it would require tractable base normalization, valid transition kernels, and evidence of mode coverage. Do not fund an AIS implementation before the finite-candidate and direct-sampling bounds fail. [Paper](https://arxiv.org/abs/physics/9803008).

**[speculation] Equivalence and prior art.** Exact rendered output should remain primary. Report token-distance balls as secondary rate-distortion curves if exact equality is too brittle. “Same GPT-2 score” is not a useful semantic equivalence: unrelated or equally bad texts collide. A coarser deterministic partition yields a valid likelihood of that partition's cell, but no longer the likelihood of the text. The surrogate cannot retroactively count as evidence for exact fiber mass.

**[source]** Kris Cao and Laura Rimell, 2021, EMNLP, arXiv:2109.02550, evaluate pretrained subword language models by marginalizing over tokenizations and compare sampling estimators. This is direct prior art for canonical-path versus surface-string likelihood, although subword segmentations have a more explicit inverse map than neural residual codes. [Paper](https://aclanthology.org/2021.emnlp-main.161/).

**[source]** Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho and Wook-Shin Han, 2022, CVPR, arXiv:2203.01941, introduce RQ-VAE/RQ-Transformer for image generation. **[speculation]** This supplies residual-quantization context, not verified precedent for your exact R_k/T_k split. I did not verify a prior multi-scale VQ paper reporting this same empirical decomposition; that is a limited search result, not a novelty guarantee. Codec redundancy and dependence between stages are also not the same as probability mass over exactly identical decoder outputs. [Paper](https://arxiv.org/abs/2203.01941).

**[speculation] What the outcomes mean for training.** Large R with small T says canonical suffix prediction is a poor *text-likelihood surrogate at that prefix*. It does not prove the gradients spent on it harmed generation. Large R plus strong improvement in prefix loss could still improve text. Small R with large N says the model is truly poor at recovering the reference output under that prefix, or the exact rendered event is extremely selective. A2/A4 having delta T comparable to delta N would directly weaken the claim that their fine-level NLL gains are merely within-fiber redistribution.

**[speculation]** The correct class loss is -log sum_{z in F} Q(z|a), whose gradient averages whole-path score gradients under the posterior on the fiber. Independent nearest-code soft labels instead optimize a product-like marginal objective; they can place mass on combinations outside the fiber. Two valid paths (A,B) and (B,A) can produce invalid combinations (A,A) and (B,B). **[ours]** A6's +0.85 Gen-PPL therefore rejects that recipe. **[speculation]** It does not reject a joint fiber loss. Nevertheless, with large R, posterior inference and coarse-level quality are the bottlenecks; another fine-level loss should not be the first new training run.

**[speculation] Numerical prediction and reversal.** On untouched TinyStories windows, predict a simultaneous lower bound on mean R_8/mean N_8 above 0.90, with at least 80% of windows obtaining a T_8 interval of width <= 2 nats within 2,048 draws. This forecast incorporates R046 and is stronger than the already-informed 80% claim. A valid upper bound below 0.80 overturns it; too many censored windows make it inconclusive. Retain the registered delta-T equivalence tests, but remove the implication that U must rank models like BLEU or Gen-PPL: data likelihood, a loose likelihood bound, and the quality of sampled outputs have no such ordering theorem. Report the registered ranking result without promoting it to a mechanistic necessity.

**3. E2: interactions are real, but their sign does not identify mode mixing**

**[speculation]** Let L_ab be damage after applying replacement a at position i and b at position j, with a,b in {0,1}. The factorial contrast I = L_11-L_10-L_01+L_00 measures non-additivity of the entire intervention-to-output map. Overlapping decoder footprints, thresholds, saturation, suffix responses, and genuine incompatible choices can all produce it. Common random numbers reduce variance; they do not remove these causal pathways.

**[speculation] Counterexample to my earlier prediction.** Suppose AB and BA are valid, AA and BB invalid, with loss 0 for valid and 1 for invalid. Start at AB and flip either symbol. L_00=0, L_10=L_01=1, L_11=0, so I=-2. A shared mode is exactly what independence destroys, yet the interaction is negative. Conversely, a decoder threshold that fails only after two benign-looking perturbations gives I>0 without any multimodal data distribution. Far pairs can interact through attention and later levels even when their immediate convolutional footprints are disjoint.

**[speculation] Register E2 as two experiments, not one.** E2a measures intervention responses. E2b tests whether changing dependence while preserving each position's marginal changes damage. Neither alone identifies the true conditional data joint; “mode mixing” needs evidence of coherent alternatives, not merely a non-additive loss.

**[speculation] E2a design.** Use 128 validation windows selected before looking at outcomes, levels 3–6, A0/A1/A3. For each window/level select one adjacent pair and one far pair by a fixed seeded rule, excluding boundaries; far separation at least half the level length. Record actual upsampling/convolution support overlap rather than treating “far” as independent. Draw eight replacement pairs from each model's actual conditional marginals; retain identity draws in the primary analysis. Reuse each pair in all four cells. A secondary wrong-only analysis must be labeled as a different intervention and match displacement and realized-change rates across models.

**[speculation]** Run two descendant conditions: (i) all unedited codes, including finer levels, clamped to canonical values, which measures the frozen decoder's direct response; (ii) finer levels regenerated ancestrally in every cell, with common randomness, which measures the total response including correction or amplification. In condition (ii), cell 00 also regenerates the suffix. Do not compare it to a fully canonical baseline. Keep all other current-level positions canonical to make the local intervention well-defined; separately acknowledge that this is a clean-context assay, not a free-running sample distribution.

**[speculation] Metrics and analysis.** Primary loss: fixed-scorer token cross-entropy, in nats per emitted token, aggregated consistently across cells; use a fixed 128-token TinyStories rendering to avoid variable-length denominators. Do not use exponentiated perplexity as the factorial loss. Secondary: token Hamming distance to canonical reconstruction, adjacent duplicate counts, and the independent TinyStories scorer. Even token-summed losses can contain decoder interactions; additive scoring removes only one source of nonlinearity. Report the signed interaction and a noise-corrected squared interaction, using two independent four-draw halves and averaging I_first I_second. Bootstrap at story/window level, not at resampling-draw level. Report levels separately and pre-specify their equal-weight aggregate.

**[speculation]** This is 98,304 cell outputs for the ancestral condition before controls, plus the cheaper clamped condition. Time a fixed 16-window pilot first; cap total expenditure at 3 CU. If the target sample count does not fit, report a precision failure or register a smaller follow-up before seeing its effects. Do not call the sample count “CPU-only” merely because no model is trained.

**[speculation] E2b: a sharper, small intervention.** Restrict two positions to their top two codes under the frozen conditional head and explicitly renormalize. This is a restricted-support experiment, not a claim about the full 4,096-way distribution. Evaluate the four cell costs l_00,l_01,l_10,l_11 with the same suffix protocol. Let P be the independent 2x2 probability table. All tables

\[
P_\theta=P+\theta\begin{pmatrix}1&-1\\-1&1\end{pmatrix},\quad
-\min(P_{00},P_{11})\le\theta\le\min(P_{01},P_{10})
\]

**[speculation]** have exactly the same marginals. Their expected loss changes by theta(l_00+l_11-l_01-l_10). Thus the factorial contrast is a *directional derivative along a dependence-only intervention*, not a universal positive signature. Selecting theta with the correct sign reduces damage whenever that interaction is exploitable and both directions are feasible. This is the precise bridge E2 was missing.

**[speculation]** First estimate the oracle transport ceiling: minimum expected loss over feasible theta versus theta=0. With exact cell expectations this bounds the opportunity in this coupling family. With Monte Carlo costs, select theta on one set of suffix draws and evaluate on independent draws; that estimates the selected oracle-assisted policy's gain, not a certified ceiling. The oracle can use the cell costs, so it is not a deployable method. A prospective pilot learns the sign and magnitude from frozen context features on training windows; compare independent, learned coupling, and context-shuffled coupling on untouched windows. Fix the candidate sets and marginals across all three. Evaluate with a second scorer and repetitions to guard against fitting the primary scorer. Context-specific success beyond the shuffled control separates useful dependence from plain marginal broadening, because broadening is impossible under these constraints.

**[speculation]** To specifically call it mode mixing, additionally show paired coherent alternatives and degraded recombinations at identical conditioning. Nearest-neighbor windows with merely similar prefixes do not provide such a control. If exact-condition coherent donors cannot be found, use the narrower claim “some decoder errors are caused by the coupling of otherwise fixed marginals.” That is already non-trivial and experimentally identifiable.

**[speculation] Numerical prediction and reversal.** I no longer predict a positive E2a sign, nor zero interaction for far pairs. For an E2b pilot I would predict a held-out, learned-coupling reduction of at least 0.02 nats/token in primary scorer loss and at least 10% in adjacent duplicates, with identical marginals and no entropy collapse; confidence roughly 25%. A 95% upper bound below 0.005 nats/token improvement and below 5% duplicate reduction kills this local coupling family. An oracle gain with no held-out learned gain says the interaction exists but is not accessible to this controller. Given R045, this pilot is currently unfunded unless the new specificity tests in question 7 pass; do not reinterpret the failed S0 gate as a pass.

**4. J2-F: earlier useful outputs and exact convergence are different objectives**

**[speculation]** For a fixed source and fixed sampling noise, write the hierarchical update as F_k(c) = f_k(c_<k). Jacobi iteration is triangular: after t iterations the first t predicted levels have their sequential fixed-point values, by induction. The worst-case dependency-chain length remains unchanged by fitting a consistency loss within the same architecture. Training can weaken effective dependencies, improve initialization, and bring early iterates into the correct decoder region without shortening the last nonzero chain.

**[speculation]** The linear analogy is e^(t+1)=J e^t with strictly lower-triangular J. Its eigenvalues are all zero, yet a long nilpotent chain can require many iterations. Reducing intermediate error magnitudes helps an output metric long before exact state equality changes. In a discrete decoder, small latent changes can also be invisible until a boundary is crossed. Predicting early text quality from an asymptotic contraction rate is therefore the wrong analysis.

**[ours]** J2-F improves early capped BLEU substantially, but exact test passes fall only 5.6%. Frozen teaching reduces shared test outputs from 16.2% for J2 to 0.3%. **[speculation]** This supports the frozen-teacher *package* preventing collapse in this seed; it does not prove moving targets were necessary. Both the trajectory distribution and target map changed. A necessity claim would require separating those two interventions and replication; that is not worth the current budget.

**[source]** Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng and Hao Zhang, 2024, ICML, arXiv:2403.00835, train CLLMs for consistency along Jacobi trajectories. Their work motivates the recipe, not a theorem that the NCM hierarchy's exact convergence must accelerate. [Paper](https://arxiv.org/abs/2403.00835).

**[speculation]** Consecutive decoded-text equality is an empirical approximate stopping rule, not a fixed-point certificate. An unobserved coarse correction can be propagating while two successive decoder outputs coincide, then alter the text later. Requiring another stable pass reduces some such cases but supplies no general guarantee. **[ours]** Only 80% of J2-F stopped outputs equal its exact output, while stopped val BLEU is 23.02 versus its exact 22.90. **[speculation]** Exact convergence is consequently not even a monotone quality target here.

**[speculation] Registration.** Freeze the rule now: identical deterministic rendered text at consecutive passes t>=2, the registered exact-state stop if earlier, and cap K+1; explicitly define EOS and pad handling. Save every capped output and the stop reason. Seed 0's val result remains exploratory. Apply the unchanged rule once to the untouched test evaluation of each new training seed; do not choose a new patience, cap, or rendering rule per seed. Timing on known test sources without quality scoring is distinguishable from tuning on test BLEU, but do not optimize the implementation to those sentences.

**[speculation]** J0 plus the same stop is the decisive training-free rule comparator; J1 plus the same stop is the continued-training comparator. Also compare fixed caps on a frontier selected using validation only and matched by measured latency, not just pass count. J0 sequential is the absolute speed baseline. A cap computed from the test set's observed mean stopping count makes the comparator test-adaptive; pre-specify that computation if retaining the current registration, and add a validation-selected cap as the clean comparison. Three fresh seeds give three confirmatory replications; two fresh seeds plus selected seed 0 should be described transparently as such.

**[speculation] First-principles timing prediction.** With 7.8 passes, the 25% speed gate requires

\[
\frac{t_{\rm Jacobi}+t_{\rm decode/stop}}{\bar t_{\rm sequential\ step}}
\le\frac{0.75\cdot11}{7.8}=1.058.
\]

**[speculation]** In an intentionally optimistic model t_seq=a+b and t_J=a+3.278b, where a is invariant dispatch/weight-read overhead and b is position-dependent work, this requires b/(a+b)<=0.0254 before the extra decoder work. More than 97% of average sequential-step time must behave like invariant overhead in that toy model. Attention, projection shapes, and dense masked work make the true calculation less favorable or at least more complicated. The 120M weights occupy about 480 MB in fp32 and 240 MB in bf16, so weight traffic can matter at batch 1; that observation alone cannot establish the required overhead fraction.

**[speculation]** My forecast against the fastest correct baseline is an fp32 J2-F-stop latency ratio of 1.2–2.5 and a bf16 ratio of 0.9–1.8; these are engineering priors, not measured confidence intervals. fp32 with TF32 disabled is less likely to hide the added arithmetic than bf16 on tensor cores. I assign less than 10% probability to passing the fp32 <=0.75 gate. L4 versus A100 may change the tradeoff; neither platform's result substitutes for the registered primary hardware.

**[ours; repository update]** R046's D1 now includes prefix-cached variants on both sides. **[speculation]** That corrects the brief's most important baseline weakness. Source hidden states can be reused only where the attention mask and conditioning make them independent of target levels; test exact logits and outputs with the relevant masks and positional conventions. Report eager and cached timings separately and use the faster correct implementation on each side, as registered. Include tokenizer decoding, synchronization, detokenization, stopping, and the actual dense or sparse attention work. Batch-1 speed, throughput, and FLOPs are separate results.

**[ours; repository update]** D1's registration gives 96 source positions, 1,368 eager sequential generator rows, and 408 rows per eager Jacobi pass. **[speculation]** These imply 312 target rows: a fully prefix-cached sequential decode uses approximately 96+312=408 rows, while 7.8 cached Jacobi passes use 96+7.8*312=2,529.6 rows, about 6.2 times as many. This assumes one prefix prefill and every target row on each Jacobi pass, as the described implementation does; verify against executed counters. The often-quoted 2.3x ratio is against the eager recomputing sequential baseline, not against cached sequential work. The preceding 3.278x-per-step toy model describes the eager comparison and is already optimistic; it must not be reused as the cached cost ratio.

**[speculation] Numerical prediction and reversal.** D1 is predicted to fail the fp32 text gate. A measured ratio <=0.75 overturns the point prediction; an upper paired 95% bound below 0.75 makes the reversal robust. Do not silently change the registered point-estimate gate to a confidence-bound gate: report both. A latency-only gain at 2.3x positions is publishable as a latency/compute tradeoff if it survives caching and quality evaluation. It is not a general compute saving; criterion (d), under R044, can still be met if actual wall time and quality satisfy its rule and attention-inclusive costs are disclosed.

**5. W2M under W2Ma: identify the intervention, not the story attached to it**

**[speculation]** Let M identify a clean versus corrupted training row, S its observed coarse context, and Y the clean target. Proper log-loss training on the corrupted mixture learns

\[
P(Y\mid S=s)=a(s)P(Y\mid s,M=0)
 +(1-a(s))P(Y\mid s,M=1),\quad a(s)=P(M=0\mid S=s).
\]

**[speculation]** This objective is proper for its training mixture. Its problem is that inference wants a different conditional, and a plausible replacement can disagree with the clean target while appearing to be a valid context. When the two target conditionals disagree in regions shared by clean and corrupted contexts, the mixture hedges. Uniform corruption can be easier to route away from the clean conditional. This account requires *both* context overlap and target conflict. Detectability alone is insufficient: an undetectable corruption that preserves Y's conditional causes no such conflict.

**[speculation]** With saturated flags, the extra input is almost M times one learned vector. That is a conditional bias selecting a regime. Twenty-five percent clean rows can train the unflagged regime while flagged rows absorb conflicting supervision. Calling this “regime bias” does not refute provenance conditioning: it is one implementation of it. What remains unproved is whether the information in the flag, rather than extra parameters, optimization asymmetry, or generic gradient routing, caused the rescue.

**[speculation]** M2 versus M3 isolates the frozen plausible-wrong versus uniform-wrong replacement distributions under the registered pipeline, not plausibility in isolation: displacement, target conflict, and context statistics all change together. M4 versus M2 tests adding the propagated indicator under that pipeline. M5 contributes almost no positional-information control; a same-seed difference between two slightly different recipes is not a lower bound on training-seed variance.

**[speculation] Add the deciding control only if rescue appears.** Use a row-level indicator with the same 25/75 frequency and identical zero-initialized vector architecture, but permute indicator assignments across rows independently of corruption and targets. Fix its RNG separately so it does not alter masks, data order, or proposal draws. This asks whether *informative association with the corruption regime* matters. Run the same clean-inference flag convention in the main comparison, with both flag settings reported as an intervention.

**[speculation]** A cheap complementary control is an always-on learned bias, used consistently at train and test, which tests generic added capacity. Do not train always-on and evaluate always-off: that introduces an artificial shift. An inverted truthful coding, with the inference flag inverted too, tests whether the result depends on the special zero-vector clean branch. None of these alone distinguishes every optimization pathway; truthful versus random-row flags is the minimum necessary information control. A positional provenance claim would require a new sparse-corruption regime with informative non-saturated flags, which the current wave does not supply.

**[speculation] Outcome decision table.** Use the registered damage and rescue quantities, with Gen-PPL accompanied by entropy, both scorers, repetitions, MAUVE, and clean-context dependence. The patterns below are not exclusive; combine their licensed claims rather than forcing one winner.

| Pattern | Deciding replication | Claim licensed; claim not licensed |
|---|---|---|
| **[speculation]** M1 does not reproduce registered damage >=5 | No W2C mechanism confirmation; first localize the changed protocol | The historical collapse is not reproduced under this recipe. It does not establish its cause. Separate fp32, clean-row fraction, and duration prospectively if worth the cost. |
| **[speculation]** M2 bad, M3 healthy, M4/M5 rescue | M2, M3, M4; add random-row flags as the fourth arm if claiming information matters | Plausible-wrong corruption is more harmful here; truthful regime conditioning rescues. Only failure of random-row rescue supports an information-specific account. No positional detectability claim. |
| **[speculation]** M2 bad, M3 healthy, M4 fails despite verified flag use | M2 versus M3 | Replacement-distribution-specific harm, with this flag conditioning insufficient. This weakens the simple “provide provenance and the problem vanishes” account. |
| **[speculation]** M2 and M3 similarly bad; M0 healthy | M0, M2, M3 after integrity checks | Wrong-only corruption or another shared difference matters. Uniform-wrong excludes the canonical code, so do not assume M0 and M3 are identical, but a large gap deserves replication before a plausibility story. |
| **[speculation]** M1 bad, M2 healthy | Current versus frozen proposals with canonical exclusion and parent choice held identical | Supports policy drift only after those confounds are removed. The existing M1/M2 contrast alone cannot isolate moving-policy feedback. |
| **[speculation]** M6 rescues M1 | M1 versus M6 | Using the actually selected L0 parent is sufficient to prevent this damage under this recipe. It does not prove every parent mismatch is fixed or exclude mixture-target conflict. |
| **[speculation]** M7 is as bad as M1 | M0 versus a rate-matched uniform control using the *new* realized rates | Reduced corruption coverage can reproduce harm. Historical A3 rates are not automatically M1's rates after fp32 and clean-row changes. |
| **[speculation]** M7 healthy, M1 bad | M1 versus matched-rate uniform, if rate coverage remains the deciding hypothesis | Rules out the tested rate schedule as a sufficient explanation; not all notions of denoising coverage. |
| **[speculation]** M4 and M5 differ substantially | Re-run with literally identical row flags and controlled RNG; then independent seeds | Evidence of sensitivity or an implementation issue, not recovered positional information. Their flag difference is too small to support the planned strong comparison without further evidence. |
| **[speculation]** M4 and random-row flags both rescue | M2, truthful flags, random-row flags | Extra conditioning/optimization can protect the model without truthful provenance. Retract the information-specific explanation. |
| **[speculation]** Several controls rescue, or M0 itself deteriorates substantially | Choose an orthogonal factorial or stop at descriptive findings | Mechanisms may coexist. Do not select the best narrative from eight arms and call the same seed a confirmation. |

**[speculation] Is the new detectability probe estimating a(s)?** Only with careful definitions. At level k define S_k as the exact input visible to the generator, excluding the supplied flag. Let p_0k and p_1k be its clean-row and corrupted-row distributions. With clean-row prior pi_0=0.25,

\[
a_k(s)=\frac{\pi_0p_{0k}(s)}{\pi_0p_{0k}(s)+(1-\pi_0)p_{1k}(s)}.
\]

**[speculation]** If a balanced classifier returns eta(s)=P_bal(clean|s), then a(s)=eta/(3-2eta), assuming calibrated class probabilities and the same class-conditional distributions. AUC is invariant to this prior change and does not estimate a(s). A hidden-state logistic probe estimates, at best, P(clean|h(S)); information may already have been discarded by h. A weak probe's failure cannot prove Bayes-level undetectability.

**[speculation]** Row regime and realized corruption are also different labels. Some nominally corrupted rows have unchanged relevant ancestors at early cuts. Keep row-regime labels if estimating the mixture above, and separately report the probability of no effective input change per level. Do not relabel only those cases and still use the 25/75 prior. The near-saturation of propagated flags says little about how easily S alone distinguishes these distributions.

**[speculation] Probe design.** Fit on training windows, calibrate on a disjoint training-development group, and score a fixed held-out group; group all variants of a story/window together before splitting. Use the same clean windows for M2/M3 corruption and several independently seeded corruption draws, with uncertainty clustered by window. Report each level separately, balanced log-loss, Brier score, AUC, calibration, and the distribution of prior-corrected a on clean inputs. Use released-model features as a common measurement instrument and adapted-model features only as a secondary analysis. Remove the flag from probe inputs. The uniform-wrong positive control tests instrument sensitivity; it does not establish adequate power for every plausible-wrong feature.

**[speculation]** The most discriminating quantity is overlap *where target conflict is high*. Stratify clean-context reliance and prediction changes by held-out detectability score; predefine strata on development data. A high AUC can coexist with a small overlapping region that produces most destructive gradients. Conversely, ambiguous provenance with matching target conditionals predicts no hedging. Claims about a(s) need calibrated probabilities and conditional-target disagreement, not an AUC threshold alone.

**[speculation] Numerical prediction and reversal.** Conditional on M1 reproducing and M2 causing at least 5 Gen-PPL points of damage relative to M0, predict truthful row flags remove at least 50% of that excess while random-row flags remove at most 20%, with matching direction in repetitions and clean-context reliance across three fresh seeds. This is the decisive new prediction; confidence about 50% conditional on reaching this branch. Rescue >=50% with random-row flags overturns the claim that truthful information is necessary for the observed rescue. No truthful rescue despite demonstrated flag use weakens this simple mechanism. For the probe, predict plausible-wrong balanced log-loss exceeds uniform-wrong by at least 0.10 nat at the pre-specified aggregate of levels 3–6; a well-calibrated reverse ordering overturns the proposed detectability ordering, not every target-conflict account.

**6. The WMT dissociation has several ordinary explanations**

**[ours]** J1 gains 43.5 nats of val code NLL while losing 0.73 val BLEU; J0 and J1 have the same reported test mean length ratio, 0.930, and duplicates rise from 17.1 to 21.0 per 1,000 words. **[speculation]** Equal mean length weakens a simple global-shortening explanation but does not exclude sentence-level EOS errors, length-conditioned effects, or compensating short and long tails.

| Account | Measurement that separates it |
|---|---|
| **[speculation]** Within-fiber redistribution | E1's paired delta R, delta T, and delta prefix cost A, retaining censoring. Large delta R alone does not establish why BLEU fell. |
| **[speculation]** Coarse sampling or calibration damage | Per-level canonical NLL plus own-context entropy/reliance; a crossed replay using J0/J1 coarse prefixes with J0/J1 fine renderers. This is a conditional intervention, not attribution to a unique layer. |
| **[speculation]** Fine-tuning subset or reference-style shift | Stratify NLL and BLEU changes by source/target length, vocabulary, repetition, and similarity to the 400k subset; compare to a held-out sample from the same subset distribution. A new shuffled full-pool control is needed to make domain shift causal, not just correlated. |
| **[speculation]** EOS/padding effects | Paired length errors, brevity penalty versus n-gram precision, and losses stratified by true length/padding fraction. Fine code positions need not correspond one-to-one to padding tokens; do not label them “padding loss” from index alone. |
| **[speculation]** Reference likelihood versus decoding risk | Fixed validation-only temperature/cap or small search sweep, tested once after freezing; compare reference likelihood with free-running repetition. Improved probability of references need not improve the argmax or sampled output's BLEU. |
| **[speculation]** Seed noise or continued-training overfit | Matched fresh training seeds and paired sentence bootstrap. The existing single-seed effect cannot distinguish these by itself. |

**[speculation]** A simple likelihood counterexample needs no redundant code: raising the probability of a reference from 0.20 to 0.25 improves its NLL, while transferring more probability to a bad competitor can change the decoded winner adversely. Sequence metrics further weight errors differently from token/code likelihood. Consequently, E1's proposed U-ranking prediction is an optional empirical observation, not the discriminating test of representation choice.

**[speculation] Numerical prediction and reversal.** Predict at least half of J1's corpus-BLEU deficit remains in the n-gram-precision component when both systems' corpus brevity penalties are set to one for this diagnostic. A loss that largely disappears—less than 25% remains—would redirect effort toward length/EOS behavior. This is a metric decomposition, not a causal length correction; sentence-level length strata must accompany it. Fund these measurements before any new WMT training control.

**7. S0: preserve its failed gate; correct what it can establish**

**[speculation]** If the baseline were the true P(C_i|S), and the augmented head recovered the true P(C_i|S,C_j), the expected log-loss gain would equal I(C_i;C_j|S). With a frozen approximate head, the measured gain also includes calibration and approximation improvements. An added branch can help even if the neighbor adds no information: it can exploit a near-constant neighbor feature or correlate with information already in S. The observed gain is therefore not automatically a lower bound on conditional mutual information.

**[ours; repository update]** R045 reports 6.22 nats/window across levels 1–5, 1.7% of 365.5 nats, and the registered gate fails. Level 1 contributes 3.24 and level 2 contributes 1.54. **[speculation]** Keep the failed verdict. The assertion that only level 1 gains more than one nat/window is inconsistent with the reported level-2 number. More fundamentally, neither a small gain nor a failed low-rank probe upper-bounds the remaining joint dependence.

**[speculation]** Parity is decisive: three uniform binary variables with C_3=C_1 XOR C_2 have zero pairwise mutual information but one bit of joint constraint. A left-neighbor test can miss all of it. Conversely, a large neighbor gain may live entirely within an exact-text fiber and be irrelevant to generation. Predictive information, useful coupling, and canonical-code likelihood are three different tests.

**[speculation] A new specificity test, not a revised old gate.** On the same frozen features, compare (i) the original frozen head, (ii) a trainable context-only adapter with comparable capacity and tuning, (iii) the neighbor adapter, and (iv) a sham-neighbor adapter. Shuffling neighbors globally is imperfect because it changes their relation to S; report that limitation and use context-stratified or conditional resampling where feasible. The context-only model is indispensable. Add a two-neighbor interaction probe and synthetic parity data as a known-power control; failure to learn a planted signal invalidates a negative interpretation. Preserve the existing good initialization: zero output factors, random input factors, rather than zeroing both sides of a bilinear product.

**[speculation] Prospective thresholds.** I would no longer use “5% of total code NLL” as the scientific threshold: it scales with the nuisance quantity E1 is questioning. For a newly registered specificity study on untouched windows, require at least 2 nats/window incremental gain over the best context-only/sham control across levels 1–5, a paired 95% lower bound above zero, and recovery of at least 80% of the planted synthetic gain. This is a different estimand from R045's total gain, not permission to reclassify 6.22 as a success. Do not use positive E2 interaction as a gate. A method pilot additionally needs a nontrivial marginal-preserving loss-reduction ceiling and a context-predictable coupling on held-out data.

**[speculation] Numerical prediction and reversal.** Predict the context-controlled incremental gain is below 2 nats/window at levels 3–5 combined, and any useful pair effect is concentrated at levels 1–2; confidence about 60%, informed by R045. An incremental gain above 2 at levels 3–5 with passing planted-signal controls overturns that localization forecast. If the 95% upper bound for the levels-1–5 incremental gain is below 2 with adequate power, kill this cheap neighbor family. An inconclusive interval or a failed planted-signal control warrants no claim about higher-order dependence. Given the completed failed S0 run, the default remains no coupled-sampler training.

**Revised queue**

**[speculation]** Use 93 CU as the conservative post-W2M planning envelope from the brief and reconcile it with the live balance before launch. The following are proposed caps, not measured runtime promises. Every new implementation still needs the repository's review/audit process and a prospective registration. No individual proposed run exceeds 15 CU. The main path totals 92.5 CU including a 15-CU reserve; untriggered allocations remain unspent. The D1-success branch substitutes for the method branch rather than being added to it.

| Item | CU cap | Gate | What it decides |
|---|---:|---|---|
| **[speculation]** E1 estimator amendment, finite-fiber toy checks, R046 pilot labeling | CPU | Before reading full E1 outcomes | Correct stopping/censoring coverage, probability accounting, and monotonicity. |
| **[speculation]** D1, including cached/eager baselines and all stop costs | 1.5 | Existing fp32 gate; report its paired interval | Whether text latency warrants any further consistency training. |
| **[speculation]** E1 released-model study plus selected checkpoint contrasts | 4 | Pilot runtime and interval-width accounting; retain every window | Quantitative exact-fiber gap across domains; seed-0 contrasts remain supporting evidence. |
| **[speculation]** W2Ma per-level calibrated probe and matched diagnostics | 0.5 | Uniform-wrong instrument control; no pooled AUC | Context overlap and its relation to clean-context damage. |
| **[speculation]** New S0 specificity/power controls | 0.5 or CPU | Existing S0 remains failed; new estimand registered | Whether neighbor information adds beyond adaptation, and where. |
| **[speculation]** Revised E2a/E2b measurement | 3 | Only if resolving a live causal claim; 16-window runtime pilot | Non-additivity versus a dependence-only effect; no positive-sign gate. |
| **[speculation]** W2C: three deciding arms, three fresh seeds | 31.5 | W2M validity and a discriminating outcome; about 3.5 CU/run including evaluation allowance | Replicated replacement/provenance or parent/rate mechanism. Use two arms when sufficient. |
| **[speculation]** Random-row flag control, three fresh seeds | 10.5 | Truthful flags rescue and an information-specific claim is pursued | Whether truthful regime information is required beyond the extra branch. |
| **[speculation]** One coupling intervention pilot | 5 | New specificity and marginal-preserving ceiling gates pass; otherwise spend zero | Accessible dependence-only quality gain, not merely canonical NLL gain. |
| **[speculation]** Coupling confirmation: two arms, three fresh seeds | 21 | Pilot meets pre-specified held-out quality/cost thresholds | A method result beyond the mechanistic finding. |
| **[speculation]** Reserve | 15 | Re-evaluation, claim audit, one pre-specified discriminating control | Protects inference and reproducibility rather than adding another candidate. |
| **[speculation]** Alternative if D1 passes: J2-F/J1 at three fresh seeds | About 34.5, replacing the 26-CU coupling pilot/confirmation allocation plus 8.5 CU of reserve | Robust timing evidence and unchanged stopping rule | Three fresh confirmations of the text half of (d), alongside the Sudoku result. |
