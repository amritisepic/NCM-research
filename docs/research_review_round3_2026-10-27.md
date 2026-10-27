# Research review, round 3 — 2026-10-27

Advisory research assessment, preserved from the response. Proposed experiments, thresholds, budgets, success criteria and queue changes are recommendations, not preregistration amendments or launch approvals. Bibliographic names identify published sources; no private identities or local paths are included.

## Part A — W2M and the cheapest defensible confirmation

**[speculation] My verdict is to confirm the replacement-distribution effect, replace the proposed flag-confirmation stage with a more identifying intervention, and treat the clean-row change as a factorial question—not an explanation already established.** The most promising broader question is: **which joint distribution of contexts and targets are we teaching, and which conditional distribution does generation actually require?**

**[ours] Evidence boundary:** I used the supplied brief, both earlier reviews, the public registrations and log through R050, and the saved W2M result JSONs. Repository recomputations are identified below. **[speculation]** All forecasts, thresholds, budgets and probabilities below are proposals. Mathematical deductions carry this label too: their validity under stated assumptions does not establish their applicability to NCM. This is a research assessment, not a new production-code audit.

- **[speculation] M2–M3 remains interpretable:** it estimates the effect of changing replacement distribution **within the 25%-clean, 600-step protocol**. The drifted control limits transport to the released operating point; it does not invalidate that matched contrast.
- **[speculation] Keep “mixed” as W2M’s registered verdict.** The strong provenance-rescue prediction failed. Do not retrospectively replace its 80% threshold with 50%.
- **[speculation] Retire “plausibility causes collapse” as the main claim.** Plausibility, displacement, conditional target conflict and training-context overlap remain entangled.
- **[speculation] Replace “clean likelihood cannot see it” with “canonical teacher-forced NLL barely distinguishes these checkpoints.”** Equal cross-entropy does not imply equal conditionals, and a few nats per window can accommodate enormous changes in generated probability mass.
- **[speculation] The minimum confirmation costs approximately 21.9 CU:** M2 and M3, unchanged at 25% clean rows, three fresh seeds each.
- **[speculation] A full clean-fraction × replacement-distribution factorial costs approximately 43.8 CU.** Fund its second half only if the no-clean pilot makes the clean-row mechanism worth claiming.
- **[speculation] Do not automatically spend another approximately 27 CU on flags.** First use existing M4 for a routing diagnostic; then prefer a target-pairing intervention that holds the corrupted inputs fixed.
- **[speculation] Keep S0’s failed gate and D1’s existing gate.** Neither should be reopened by changing thresholds.

### A1. What the drifted control licenses

**[ours] Repository recomputation:** M2–M3 is **20.936 ± 0.374 Gen-PPL**, paired SE over the 16 sampling groups. M2–M3 clean NLL is **4.035 nats/window**. These characterize particular checkpoints, not training variability.

**[speculation]** The defensible statement is:

> Under this continuation protocol, frozen-model wrong-code replacement produces substantially worse generation than uniform-wrong replacement, despite similar canonical teacher-forced loss.

**[speculation]** M2 shows that moving-policy feedback is **unnecessary for this failure**. M6 shows that correcting the tested L0-parent mismatch is **insufficient to prevent it**. M7 shows that its particular historical lower-rate schedule is **insufficient to reproduce it**. None establishes that feedback, other parent inconsistencies or coverage have zero contribution.

**[speculation]** Use **M3 without clean rows**, rather than M0 without clean rows, for the held pilot. It becomes the missing control cell of a clean factorial:

| [speculation] | Uniform-wrong | Frozen plausible-wrong |
|---|---|---|
| 0% clean rows | \(U_0\) | \(P_0\) |
| 25% clean rows | \(U_{25}\), existing M3 recipe | \(P_{25}\), existing M2 recipe |

**[speculation]** Define \(\Delta_c=G(P_c)-G(U_c)\), and the interaction \(I=\Delta_{25}-\Delta_0\). The interaction answers whether clean rows change the replacement-distribution effect; \(G(U_{25})-G(U_0)\) answers whether they degrade the uniform-wrong control. Historical W1 is supporting evidence, not a factorial cell.

**[speculation] Prediction:** \(U_0\) improves Gen-PPL by **at least 3 points** over \(U_{25}\), and \(\Delta_0\) remains **at least 10 points**.
**[speculation] Overturning result:** improvement below **1 point** weakens clean rows as the principal explanation of control drift; \(\Delta_0<5\) makes the collapse strongly protocol-dependent and defeats a clean-fraction-independent claim. Intermediate outcomes remain inconclusive.

**[speculation]** Preserve 25% clean rows in the minimum confirmation. Removing them now would confirm a different intervention. Flags require exposure to their inference setting, but that is not a reason to make flag training the primary scientific question.

### A2. What “clean-NLL invariance” could mean

**[ours] Repository recomputation:** M2’s rounded per-level losses exceed M3’s by only **1.69 nats across L1–L5**, and by at most **0.57 nat at any individual level**. Thus this pair does not conceal a large coarse-loss deterioration cancelled by fine-level improvement.

**[speculation]** Nevertheless, similar losses on the canonical labels do not establish similar distributions over alternatives. W2M-G should measure distributions and state visitation, not only entropy.

| Account | [speculation] Separating measurement on existing checkpoints | [speculation] Prospective prediction and overturn |
|---|---|---|
| The model discounts generated-looking contexts | Evaluate M2 and M3 on the **same** clean, released-model, M2 and M3 prefix banks; measure predictive JS divergence and response to controlled ancestor changes | M2–M3 divergence on generated banks is **≥3×** that on clean banks at L2–L5. A ratio **≤1.2**, with adequate precision, overturns this localization |
| Aggregate NLL misses changed alternatives | Compare full predictive distributions, top-code rankings and conditional calibration on clean inputs | Detectable clean-input distribution changes despite small CE differences would overturn the stronger claim that clean conditionals were preserved |
| Cross-level seams reveal corruption | Compare coherent ancestral prefixes with mosaics assembled from different trajectories, matched for per-level code frequencies | Mosaic/coherent discrimination exceeds **0.80 AUC** and predicts suppression of ancestor response. A well-powered **≤0.60 AUC** plus unchanged response weakens the seam account |
| Damage is chiefly early-state selection | Cross M2/M3 prefix generators and suffix generators at fixed cuts | Supplying real levels through L3 removes **≥50%**, and through L5 **≥75%**, of the scorer-loss contrast. Retaining **≥80%** after L5 overturns this forecast |
| Damage is chiefly conditional rendering | Give both renderers identical generated prefixes | Most damage remains attributable to which suffix generator completes the shared prefix; a healthy M3 suffix repairs it |
| Calibration or temperature, rather than lost ordering information | A small, frozen validation temperature comparison with both scorers, MAUVE, repetition and diversity | Predict **≥10 Gen-PPL points** remain after the best permitted calibration. Removing nearly all damage without diversity loss would substantially change the story |

**[speculation]** W2M-G is therefore the right starting instrument, with four additions:

1. **Crossed replay:** distinguish a bad distribution of prefixes from a bad response to the same prefixes.
2. **Full ancestral prefixes:** one-step replacements beside canonical levels are a separate distribution.
3. **Predictive divergence and intervention response:** entropy alone cannot distinguish confident error, useful uncertainty and context neglect.
4. **Log scorer loss for decomposition:** retain registered Gen-PPL, but calculate rescue fractions on mean token log loss too. Perplexity differences are nonlinear.

**[speculation]** Do not score the original canonical suffix against a generated prefix and call that the “correct generation target.” That measures recovery of the original example after intervention. A generated prefix may legitimately warrant a different continuation.

### A3. Minimum-cost W2C and the most useful extra arm

**[speculation]** Run M2/M3 at fresh seeds **1–3**, 600 updates, unchanged masks, clean fraction, data order, optimizer and evaluation. Keep seed 0 exploratory. Require the registered gain criterion, report every seed and paired differences, and distinguish sampling uncertainty from training uncertainty.

**[speculation] Prediction:** mean M2–M3 damage **≥10 Gen-PPL**, matching deterioration under the second scorer and in adjacent duplication; mean canonical-NLL difference **≤10 nats/window**.
**[speculation] Overturning result:** mean damage **<5**, a sign reversal in the main quality measures, or substantial deterioration of clean likelihood that removes the purported dissociation.

**[speculation] The most identifying extra arm changes targets while keeping the corrupted inputs fixed.** My preferred first version is **balanced target reassignment at L1**, detailed in B4:

- Retain M2’s exact corrupted input hierarchy.
- On a balanced subset of rows whose root was replaced, reassign the **whole eight-code L1 target block** from another training row whose canonical root equals the root actually supplied.
- Use a permutation, preserving the L1 target multiset exactly.
- Leave all other targets and every input unchanged.

**[speculation]** This directly manipulates the context–target association while fixing replacement distribution, displacement, wrong-slot count, input detectability and target marginals. It is more identifying than another “damage-matched” replacement distribution. Its limitation is equally important: it tests the first refinement, not all levels.

**[speculation] Prediction:** with **≥80% coverage** of root-corrupted rows, it removes **≥25%** of M2–M3 generation damage.
**[speculation] Overturning result:** **≤10% rescue**, despite verified pairing and coverage, kills this first-refinement intervention as the funded direction. It does not refute target conflict at later levels.

### A4. M4’s partial rescue

**[speculation]** A vector norm of 0.15 establishes neither insufficient capacity nor adequate capacity. Its effect depends on input scale, normalization and downstream sensitivity. The plateau also weakens—but does not eliminate—a simple “train longer” explanation.

**[speculation]** First evaluate the existing model with prescribed flag values **0, 0.5 and 1**, separately on clean contexts, M2-corrupted contexts and shared ancestral contexts. Treat this as a diagnostic; do not select on test.

**[speculation]** The outcomes separate useful possibilities:

- A nonzero flag improves free generation while hurting clean-reference likelihood: evidence that the inference branch is misrouted.
- True flags improve corrupted-context loss but no flag improves generation: the branch learns denoising, without supplying the desired continuation conditional.
- Changing flags barely changes logits: ineffective use remains plausible.
- Large logit changes without rescue: lack of capacity in the flag vector is not the immediate bottleneck.

**[speculation] Prediction:** no fixed flag setting recovers **80%** of the M2–M3 gap while retaining the other quality measures.
**[speculation] Overturning result:** such a setting exists on validation and replicates untouched, making inference routing a much larger explanation than I currently expect.

**[speculation]** Random-row flags remain necessary **if the claim is that truthful information is required**: same flag architecture, marginal flag frequency and extra RNG stream, but flags independent of corruption regime. They are not decisive between limited capacity, insufficient training and clean-branch exposure bias.

**[speculation] Prediction if this control is funded:** random flags rescue **≤20%**, truthful flags **≥50%** on fresh seeds. Random-flag rescue **≥50%** defeats information-specific attribution. I would not fund this approximately 27-CU branch before the cheaper routing test and target-pairing pilot.

### A5. Why M1 ≈ M2 does not establish a wrong-slot dose law

**[speculation]** It establishes that **total wrong-slot count is an inadequate scalar predictor across these arms**. It does not establish that damage is independent of corruption dose. Early sensitive errors may already saturate the effect; later differences may be decoder-insensitive; M1’s endogenous proposals also differ qualitatively.

**[speculation]** The clean localization experiment uses two hybrids, retaining the original masks everywhere:

- plausible-wrong at **L0–L2**, uniform-wrong at **L3+**;
- uniform-wrong at **L0–L2**, plausible-wrong at **L3+**.

**[speculation]** Leaving the other levels clean would change coverage and create another confound. The hybrids complete a level-group factorial with M2 and M3.

**[speculation] Prediction:** the early hybrid reproduces **≥70%** of M2–M3 damage; the late hybrid **≤30%**.
**[speculation] Overturning result:** the late hybrid is at least as damaging as the early hybrid. Start with crossed checkpoint replay; two short hybrid pilots should replace another pilot allocation, not silently enlarge the queue.

---

## Part B — The research question worth pursuing

### B1. Hostile evaluation: the current story is too strong

**[speculation]** The weakest claim is the compound sentence:

> Plausibility, rather than feedback, parent mismatch or coverage, causes collapse; clean likelihood cannot see it; provenance protects.

**[speculation]** Each clause currently exceeds its evidence.

| Current conclusion | [speculation] Strongest objection | [speculation] Cheapest overturning test |
|---|---|---|
| Plausibility is causal | “Plausible” labels several changing properties; no isolated intervention | Fixed-input target reassignment; alternatively the level-group hybrids |
| Feedback is ruled out | Frozen proposals show feedback is unnecessary, not causally irrelevant | Preserve the narrower claim; no new run needed |
| Parent mismatch is ruled out | M6 changes L0 conditioning only; mixed higher-level histories remain | Coherent ancestral versus one-step/mosaic replay |
| Clean likelihood is blind | Similar cross-entropy can conceal large probability rearrangements | Predictive-distribution comparisons and tail-stratified clean losses |
| Clean rows caused control drift | W1/W2M differ in schedule and other implementation history | The matched 0/25 factorial |
| Provenance protects | The intervention provides regime-conditioned computation and misses its strong prediction | Existing flag routing sweep; random-row flags only if pursuing this claim |
| Little within-level dependence explains S0 | A restricted neighbor probe does not bound total or higher-order dependence | Preserve the failed practical gate, retract the architectural inference |
| The coarse oracle curve identifies where errors originate | Real prefixes change both state distribution and downstream difficulty | Cross prefix source and suffix renderer |
| E1 explains generation degradation | Exact-fiber mass under canonical prefixes does not locate free-generation failures | Crossed rollout plus E1 checkpoint contrasts |
| Lower Gen-PPL is sufficient improvement | A model can lower scorer loss by reducing useful diversity | Fixed scorers plus MAUVE, sequence repetition, coverage and blinded samples |

**[speculation] A particularly cheap counterexample undermines the word “invariance.”** Let \(P\) be the data distribution and let \(R\) put probability only outside its support. Then

\[
Q=(1-\epsilon)P+\epsilon R
\quad\Longrightarrow\quad
\mathbb E_P[-\log Q]-H(P)=-\log(1-\epsilon).
\]

**[speculation]** Allocating **half** the samples to \(R\) costs only **0.693 nat per entire sequence**. Two models can have identical data NLL while placing that extra mass on very different outputs. This is an illustrative distributional calculation, not an estimate of NCM’s bad-sample fraction.

**[speculation]** Consequently, “only four nats” is not a strong distributional equivalence claim. Canonical-path redundancy makes interpretation still harder, but is not necessary for the dissociation.

**[source]** Scheduled-sampling inconsistency is established prior art: Ferenc Huszár, 2015, arXiv:1511.05101. More directly, Gengze Zhou, Chongjian Ge, Hao Tan, Feng Liu and Yicong Hong, 2025, arXiv:2512.06421, explicitly study unstable student forcing in scale-wise image generation and conflicting supervision on generated contexts. [Scheduled sampling](https://arxiv.org/abs/1511.05101), [scale-wise training dynamics](https://arxiv.org/html/2512.06421v1).

**[speculation]** A reviewer would reasonably call the current headline a rediscovery. A stronger contribution would establish:

1. a quantitative prediction of **which predictive distinctions a training channel suppresses**;
2. why this changes across levels and between unconditional text and source-conditioned tasks;
3. an intervention changing only context–target association;
4. fresh-seed replication and an explicit account of what canonical NLL misses.

**[speculation]** That can satisfy a substantial version of criterion (c). “Corruption sometimes hurts” cannot. Nor does a training pathology automatically establish what levels encode: the claim must connect the intervention to specific information or functional roles at particular levels.

**[speculation]** Several killed ideas were killed too broadly, but should remain unfunded. S0 kills the proposed cheap neighbor route, not all joint sampling. A6 kills independent softening under its recipe, not joint probability over equivalent code paths. A2 does not invalidate residual retargeting generally. J2-F kills this route to exact-pass acceleration, while demonstrating useful early-output improvement. Scope the negatives correctly without resurrecting their budgets.

### B2. A derived account: a hierarchy learns an experiment, not just a conditional

#### The representation is a sequence of coordinates and decisions

**[speculation]** Write

\[
C=E(X),\qquad
Z_{<k}=\sum_{\ell<k}A_\ell e_\ell(C_\ell),\qquad
S_k=B_kZ_{<k}.
\]

**[speculation]** The full prefix \(C_{<k}\) defines increasing information as levels accumulate. The compressed state \(S_k\) need not: addition and resizing can discard which ancestor produced which contribution. Moreover, the tokenizer’s partial reconstruction need not be a posterior mean, a semantic summary, or a sufficient statistic for the next level.

**[speculation]** Three desiderata are distinct:

- low final reconstruction distortion;
- informative partial reconstructions;
- refinements whose distribution is easy to generate with the available parallel computation.

**[speculation]** Residual quantization directly addresses the first. It can help the second while making the third harder. The correct question is not whether coarse codes “contain concepts,” but whether they expose a useful separator of downstream uncertainty at an affordable generation cost.

**[speculation]** There is also no unique correct probability distribution on code hierarchies. Any distribution \(\Pi(C)\) whose decoder pushforward matches the desired text distribution is a candidate. The canonical encoder distribution is one such choice, approximately accounting for reconstruction error. A different distribution over equivalent codes could be much easier for the generator.

#### What corruption actually teaches

**[speculation]** At a particular level, let \(A\) be the clean ancestor information, \(Y\) the canonical target and \(U\) the corrupted visible state. A corruption kernel \(K\) defines the training joint

\[
R_K(u,y)=\sum_a P(a,y)K(u\mid a,y).
\]

**[speculation]** With unrestricted categorical prediction, cross-entropy learns

\[
q_K^*(y\mid u)=R_K(y\mid u).
\]

**[speculation]** With a factorized head it learns the individual marginals \(R_K(Y_i\mid u)\). Cross-entropy is proper for this training joint. The problem is that this joint need not describe the continuation task performed at generation.

**[speculation]** This yields an exact criterion. For a chosen desired continuation law \(P^\dagger\) and deployment state distribution \(\mu\), a training joint is conditionally correct when

\[
R_K(Y_i\mid u)=P^\dagger(Y_i\mid u)
\quad \mu\text{-almost everywhere},
\qquad
\mu\ll R_{K,U}.
\]

**[speculation]** The support condition prevents completely untrained deployment states. The conditional condition prevents training on the right states with the wrong futures. Neither guarantees correct joint sampling when the head factorizes dependent targets.

**[speculation]** Off the support of the chosen data/code distribution, \(P^\dagger(Y\mid u)\) is not identified by clean examples. There is no uniquely “correct” original suffix for an arbitrary generated prefix. You must specify a continuation policy, a restoration task, or another extension. This is why distribution matching alone cannot solve the problem.

**[source]** DAgger queries an expert on states visited by the learner: Stéphane Ross, Geoffrey Gordon and Drew Bagnell, AISTATS 2011. GKD applies teacher feedback to student-generated sequences: Rishabh Agarwal et al., ICLR 2024, arXiv:2306.13649, across summarization, translation and arithmetic. Neither justifies attaching an unchanged original future to every modified prefix. [DAgger](https://proceedings.mlr.press/v15/ross11a.html), [GKD](https://arxiv.org/abs/2306.13649).

#### A sharper result: realistic replacement can erase precisely the predictive signal

**[speculation]** Consider a finite clean context \(A\), with corruption independent of the target conditional on \(A\). Define the reverse channel

\[
W_{ua}=P(A=a\mid U=u),\qquad
m_y(a)=P(Y=y\mid A=a).
\]

Then

\[
R_K(Y=y\mid U=u)=(Wm_y)(u).
\]

**[speculation]** Corruption applies a **smoothing operator to the conditional probabilities**. It does not merely perturb examples. Under reversibility, predictive components aligned with eigenvectors of \(W\) are attenuated by their eigenvalues.

**[speculation]** A simple case makes the consequence explicit. Replace \(A\) by an independent draw from its own marginal with probability \(r\). Then

\[
R_K(Y\mid U=u)
=(1-r)P(Y\mid A=u)+rP(Y).
\]

**[speculation]** The context is perfectly plausible, but its predictive contribution is reduced by \(1-r\). No within-level dependence is required; \(Y\) can be a single categorical variable.

**[speculation]** There is a useful impossibility statement behind this. Suppose a target-independent corruption preserves the context marginal and its reverse channel is irreducible. Preserving every clean conditional requires \(Wm_y=m_y\). The only harmonic functions of a finite irreducible stochastic matrix are constant. Therefore exact preservation requires \(Y\) to be independent of the mixed context component.

**[speculation]** In plain terms: **a channel that genuinely mixes predictive context states cannot generally preserve their original continuation conditionals while retaining the original labels.** This is a standard Markov-operator deduction, not a claim of a new theorem.

**[speculation]** Uniform vocabulary replacement behaves differently when most uniform codes lie outside the clean context distribution. At a plausible observed context, the posterior probability that it came from an uncorrupted example can remain high. This can preserve predictive distinctions on the useful part of the state space, despite much greater geometric displacement.

**[speculation]** The relevant quantity is therefore not “plausibility” alone. It is **how the corruption’s reverse channel mixes states with different appropriate futures**.

#### Why the clean rows change the interpretation

**[speculation]** For clean-row fraction \(\alpha\),

\[
R_\alpha(u,y)
=\alpha P_0(u,y)+(1-\alpha)P_K(u,y),
\]

and

\[
q_\alpha^*(y\mid u)
=a(u)P_0(y\mid u)+(1-a(u))P_K(y\mid u),
\]

\[
a(u)=
\frac{\alpha p_0(u)}
{\alpha p_0(u)+(1-\alpha)p_K(u)}.
\]

**[speculation]** Clean inputs can have \(a(u)\approx1\), while generated inputs have \(a(u)\approx0\). The model can preserve clean-reference performance and behave differently on its generated contexts. Cross-level inconsistency can help identify these regimes even when individual replacement codes are plausible.

**[speculation]** This explains a possible mechanism for clean-NLL protection. It does **not** establish that M2 actually learned such a partition. The shared-bank divergence and response measurements decide that.

**[speculation]** There is also a sharp limitation: in the simple marginal-preserving channel example, adding clean rows reduces smoothing and moves the Bayes conditional toward the clean one. It does **not** explain why adding clean rows degrades uniform-control generation. Any account claiming to derive that observation from target ambiguity alone is incomplete.

**[speculation]** To explain the control drift, the account needs finite-capacity optimization, deployment-state coverage, or both. One plausible version is that uniform-corruption training provides a useful robustness neighborhood; allocating updates to clean rows improves canonical accuracy while weakening the model’s response outside that neighborhood. Another is shared-parameter interference: improvements in high-volume fine losses alter a few consequential coarse decisions.

**[speculation]** These predict different measurements. The first predicts greater sensitivity or poorer calibration on generated/perturbed states after adding clean rows. The second predicts adverse coarse-risk changes from actual fine-loss optimizer updates. Neither is established by falling total NLL.

#### Why one bad coarse decision can dominate hundreds of wrong fine codes

**[speculation]** Wrong-slot count is a Hamming distance. Generation quality depends on an entirely different geometry.

For a small state perturbation \(\delta\), predictive KL is locally

\[
D_{\mathrm{KL}}(q(\cdot\mid s)\Vert q(\cdot\mid s+\delta))
\approx \tfrac12\delta^\top G(s)\delta,
\]

where \(G\) is the input-space Fisher metric induced by the predictive distribution.

**[speculation]** Two equally sized perturbations can have very different effects because they point in different directions. Further, downstream stages can amplify, compensate for, or ignore them. Later near-equivalent codes may change many indices while barely changing the text.

**[speculation]** A finite-horizon decision identity makes the relevant weighting precise. Let \(P\) be a reference generator, \(Q\) another generator, \(g\) a fixed terminal text cost, and \(V_k^P(h)\) the expected terminal cost after prefix \(h\), completed with \(P\). Then

\[
J(Q)-J(P)
=
\sum_k
\mathbb E_{h\sim Q}
\left[
\mathbb E_{c\sim Q_k(\cdot\mid h)}V_{k+1}^P(h,c)
-
V_k^P(h)
\right].
\]

**[speculation]** This follows by telescoping the reference continuation values. It weights a changed prediction by its downstream consequence **on prefixes actually visited by \(Q\)**. Clean NLL weights canonical labels on data prefixes instead.

**[speculation]** Crossed prefix/suffix generation is a practical approximation to these quantities. It provides a more meaningful localization than either slot counts or the oracle curve alone. Its contributions depend on the chosen reference and intervention order; they are not intrinsic attributions to a level.

**[speculation]** The control-theoretic analogy is consequently **finite-horizon disturbance amplification**, not an infinite unstable feedback loop. Saturation is possible: once an early error enters a destructive region, additional later errors may add little measurable damage.

#### What parallel generation can and cannot represent

**[speculation]** Conditional on a fixed state,

\[
Q(C_k\mid S_k)=\prod_i q_i(C_{k,i}\mid S_k)
\]

cannot represent residual dependence between the realized draws. Deterministic attention coordinates the probability vectors; it does not correlate independent random draws.

**[speculation]** That does not imply one neural pass per dependent token is necessary. An earlier latent, shared randomness or a structured sampler can sometimes coordinate many outputs cheaply. The difficulty depends on the structure of the dependence, not merely its amount.

**[speculation]** S0 therefore has two legitimate implications: the tested neighbor augmentation has a small gain, and it does not justify the proposed sampler expenditure. It does not establish that the true joint is nearly factorized.

**[speculation]** More importantly, the channel-smoothing example above produces harmful conditional learning with **one output per level**. Parallel factorization is not necessary for this failure. My earlier emphasis on independent mode mixing as the missing bridge was too strong.

**[source]** Shaorong Zhang et al., 2026, arXiv:2602.00286, already distinguish model error from factorization error and show that forward-KL diagnostics can miss severe reverse-KL incoherence, using controlled Block-HMM and LLaDA arithmetic settings. The generic dependence argument is occupied territory. [Information-theoretic analysis](https://arxiv.org/abs/2602.00286).

#### Why code likelihood and text quality separate

**[speculation]** For deterministic decoding, let \(Q_D(x)=\sum_{c:D(c)=x}Q(c)\). For a canonical hierarchy \(c^*\) decoding to \(x\),

\[
-\log Q(c^*)
=
-\log Q_D(x)
-\log Q(c^*\mid D(C)=x).
\]

**[speculation]** The second term measures insistence on the encoder’s selected path within the exact decoder fiber. Improving that term need not improve text probability. E1 measures a conditional version with a fixed prefix, so it cannot identify the entire free-generation probability.

**[speculation]** There are thus three separate dissociations:

1. canonical-path likelihood versus decoded-text likelihood;
2. data-prefix performance versus generated-prefix performance;
3. model likelihood versus external scorer loss or BLEU.

**[speculation]** A2, A4, J1 and M0 may involve different mixtures of all three. A single E1 result cannot explain them all.

**[source]** Marginalizing alternative representations is not new: Kris Cao and Laura Rimell, EMNLP 2021, arXiv:2109.02550, study marginal likelihood over subword tokenizations in pretrained English and German models. NCM’s potentially enormous, learned decoder fibers and their training consequences are the distinctive opportunity. [Marginal likelihood over tokenizations](https://arxiv.org/abs/2109.02550).

#### Why J2 collapses and J2-F improves early outputs without converging much faster

**[speculation]** A moving-target consistency objective admits a dangerous family of solutions: destroy input dependence and make outputs agree. A constant function is perfectly consistent. The supervised term may oppose this, but consistency itself does not distinguish a useful fixed point from an uninformative one.

**[speculation]** A frozen teacher supplies an external anchor. It can improve approximate early outputs without shortening the dependency chain that determines exact codes.

**[speculation]** With fixed sampling randomness, NCM’s level update is triangular:

\[
F_k(C)=f_k(C_{<k}).
\]

Correct prefixes propagate through Jacobi iterations by induction. A smooth relaxation has a strictly lower-triangular, hence nilpotent, Jacobian. Its spectral radius is already zero; “make it contractive” is not a sufficient acceleration prescription.

**[speculation]** What matters is the strength and length of transient dependency paths, and whether the remaining code changes affect decoded output. E1 could explain why exact code convergence is an excessively strict endpoint. D1 decides whether exploiting that fact helps real latency.

#### What this account explains—and where it must earn its keep

| Observation | [speculation] Account | [speculation] Discriminating prediction |
|---|---|---|
| Plausible-wrong hurts generation while clean loss stays close | Wrong context–target joint plus deployment-state weighting; possible regime separation | Generated-bank divergence/response changes substantially exceed clean-bank changes |
| Without clean rows, clean NLL also worsens | Loss of an explicit clean anchor, with overlap or parameter sharing | In the factorial, M2–M3 NLL difference rises from **≤10** to **≥15 nats**; **≤5** at 0% weakens this explanation |
| Uniform corruption helps, but clean rows hurt its continuation | Robustness/optimization trade-off, **not derived from simple Bayes smoothing** | No-clean control recovers **≥3 Gen-PPL** and generated-state robustness improves |
| Wrong-slot count fails | Direction, level, downstream value and saturation matter | Early/late hybrids separate by the A5 thresholds |
| Flags partially rescue | Conditional routing separates objectives; inference regime remains mismatched | Flag effects depend on the context bank, not just vector norm |
| Likelihood and quality separate | Fibers, occupancy and evaluation objectives differ | E1 plus crossed replay separate these terms |
| Moving-teacher collapse | Unanchored self-consistency can erase source dependence | J2’s response to source changes is markedly smaller than J2-F’s |
| S0 small, coarse oracle effect large | Conditional error and high downstream leverage need no large neighbor dependence | A target-association intervention can improve generation without adding joint sampling |

**[speculation]** This is one coherent account with several measurable components, not one proven cause. It does not derive the observed effect sizes, establish that clean rows caused drift, or explain why this optimizer reaches a particular solution.

#### Is there an optimal corruption?

**[speculation]** There is no universally optimal distribution of replacement codes. The design problem is over **context–target joints**, subject to deployment coverage, model capacity and computation.

A useful abstract objective is

\[
\min_{R}
\sum_k
\mathbb E_{u\sim\mu_Q}
\left[
w_k(u)\,
D\!\left(P^\dagger_k(\cdot\mid u),R_k(\cdot\mid u)\right)
\right]
+\lambda\,\mathrm{EstimationCost}(R),
\]

subject to adequate support and a declared target construction.

**[speculation]** Here \(w_k\) represents downstream consequence, rather than number of code slots. This is a research formulation, not a computable oracle.

**[speculation]** It suggests four concrete principles:

- **Noise level:** increase corruption only while useful deployment-state coverage outweighs conditional bias and estimation variance.
- **Noise distribution:** measure which predictive distinctions it mixes, not merely code distance or model probability.
- **Per-level schedule:** allocate corruption by deployment mismatch and downstream sensitivity, not a fixed linear rate alone.
- **Targets:** decide whether the task is restoration of the original example or continuation of the presented state. Re-quantization corrects residual geometry; it does not resolve this semantic choice.

**[source]** Denoising models obtain guarantees by pairing their learned posterior with an appropriate generative process. Yoshua Bengio, Li Yao, Guillaume Alain and Pascal Vincent, 2013, arXiv:1305.6663, analyze denoising-based generative chains. Jacob Austin et al., NeurIPS 2021, arXiv:2107.03006, explicitly design discrete transition kernels and their reverse models. NCM’s corrupted-ancestor training is not automatically such a matched reverse process. [Denoising generative chains](https://arxiv.org/abs/1305.6663), [D3PM](https://arxiv.org/abs/2107.03006).

### B3. The relevant field, 2024–2026

**[speculation]** These are the lines that most constrain novelty or change the experimental design. The cited empirical results are the authors’ results, not replications here.

| Line | [source] What it establishes and where | [speculation] Consequence for this project |
|---|---|---|
| Principled discrete objectives | Aaron Lou, Chenlin Meng and Stefano Ermon, **ICML 2024**, arXiv:2310.16834, SEDD; Subham Sahoo et al., **NeurIPS 2024**, arXiv:2406.07524, MDLM. Probability-ratio and masked-denoising objectives on language-modeling benchmarks. [SEDD](https://arxiv.org/abs/2310.16834), [MDLM](https://arxiv.org/abs/2406.07524) | Proper training objectives do not eliminate finite-step factorization error. Conversely, NCM cannot borrow their consistency guarantees merely because both use corruption. |
| Next-scale image generation | Keyu Tian et al., **NeurIPS 2024**, arXiv:2404.02905, VAR, establishes effective next-scale ImageNet generation. Jian Han et al., **CVPR 2025**, arXiv:2412.04431, Infinity: its **5M-image, 512², temperature-1/CFG-3 ablation** reports FID **9.76 baseline, 9.69 random flips, 3.48 bitwise self-correction**. [VAR](https://arxiv.org/abs/2404.02905), [Infinity Table 7](https://arxiv.org/html/2412.04431v2) | Residual retargeting is prior art, with a strong vision result. Your failed transfer needs a structural account of the text representation and supervision, not a generic claim that retargeting is ineffective. |
| Student forcing in next-scale models | Gengze Zhou et al., **2025**, arXiv:2512.06421, SAR: unstable naïve student forcing, staggered rollouts and altered supervision; reported **5.2% FID reduction** for FlexVAR-d16 on ImageNet-256 after **10 epochs, 5 hours on 32 A100s**. [SAR](https://arxiv.org/html/2512.06421v1) | This is particularly close prior art. A new claim must identify the channel/target mechanism more tightly. Its equations and pseudocode also differ in how they describe supervision, so copying the recipe requires checking official implementation. |
| Remasking and learned correction | Guanghan Wang et al., **NeurIPS 2025**, arXiv:2503.00307, ReMDM, derives remasking samplers and tests language, discrete images and molecules. Jaeyeon Kim et al., **2025/2026**, arXiv:2510.01384, PRISM, learns token-quality scores, evaluated on Sudoku, **170M unconditional text**, and **8B LLaDA code**. [ReMDM](https://arxiv.org/abs/2503.00307), [PRISM](https://arxiv.org/abs/2510.01384) | Generic correction, remasking and learned error detection are already occupied. NCM needs a residual-specific intervention or a transferable training principle. |
| Noise families and correctors | Dimitri von Rütte et al., **ICML 2025**, arXiv:2503.04482, GIDD, combines masking and uniform transitions. Sahoo et al., **ICML 2025**, arXiv:2506.10892, DUO, connects Gaussian and uniform diffusion. Justin Deschenaux, Caglar Gulcehre and Sahoo, **2026**, arXiv:2602.21185, develop predictor–corrector samplers, evaluated on OWT and CIFAR-10. [GIDD](https://arxiv.org/abs/2503.04482), [DUO](https://arxiv.org/abs/2506.10892), [Ψ-samplers](https://arxiv.org/abs/2602.21185) | “Uniform noise is unrealistic” is not an objection. What matters is the trained experiment and the corresponding generation dynamics. |
| Training on generated states | Agarwal et al., **ICLR 2024**, GKD; Phuong Tuan Dat, Qi Li and Xinchao Wang, **2026**, arXiv:2607.04428, dOPSD, uses later denoising states as privileged information on Dream/LLaDA reasoning and code. [GKD](https://arxiv.org/abs/2306.13649), [dOPSD](https://arxiv.org/abs/2607.04428) | On-policy training is not intrinsically doomed. The target’s relationship to the encountered state is central. “Use generated contexts with a teacher” is not a new direction. |
| Likelihood versus generation | Sahoo et al., **2026**, arXiv:2602.15014, scale masked, uniform and interpolating diffusion to **1.7B**, finding that likelihood rankings can disagree with practical sampling preference. Patrick Pynadath, Jiaxin Shi and Ruqi Zhang, **2026**, arXiv:2604.02718, analyze evaluation around GPT-2-small-scale models and propose generative frontiers. [Scaling study](https://arxiv.org/abs/2602.15014), [evaluation analysis](https://arxiv.org/abs/2604.02718) | Dissociation alone has little novelty. Also, unigram entropy is not sequence entropy and cannot turn Gen-PPL into a complete distributional evaluation. |
| Jacobi after CLLMs | Siqi Kou et al., **ICML 2024**, arXiv:2403.00835, CLLMs, trains rapid fixed-point prediction. Lanxiang Hu et al., **2025**, arXiv:2512.14681, Jacobi Forcing, trains on generated trajectories. Its **Qwen2.5-Coder-7B, HumanEval, single-B200 appendix comparison** reports **3.95×** speed with multiblock decoding, at **83.5 versus 87.8** accuracy. [CLLMs](https://arxiv.org/abs/2403.00835), [Jacobi Forcing](https://arxiv.org/html/2512.14681v1) | Real acceleration exists, but with different causal structure, caching and quality trade-offs. It does not rescue an NCM implementation processing every scale per pass. |
| Adaptive generation and stopping | Marianne Arriola et al., **ICLR 2025**, arXiv:2503.09573, Block Diffusion, combines sequential blocks, parallel denoising and caching. Chia-Ming Lee et al., **2026**, arXiv:2607.28166, C⁴, combines confidence and temporal agreement on LLaDA/Dream. [Block Diffusion](https://arxiv.org/abs/2503.09573), [C⁴](https://arxiv.org/abs/2607.28166) | Stable text, confidence and adaptive commitment are established tools. D1’s end-to-end timing remains essential; reported decoding-loop speedups are not automatically comparable. |
| Representation optimized for generation | Alan Baade et al., **2026**, arXiv:2610.20533, RePR, learns latent representations using generative information differences, with ImageNet and text-to-image experiments. [RePR](https://arxiv.org/abs/2610.20533) | Reconstruction-optimal and generation-friendly representations are already being separated. A new tokenizer proposal needs a precise advantage beyond “make latents easier to model.” |

**[speculation]** I found no verified paper in this search establishing the specific conjunction of **fixed corrupted inputs, exactly preserved target marginals, manipulated context–target pairing, and a replicated effect on a residual text hierarchy**. That is a promising distinction, not an exhaustive novelty clearance.

### B4. New ideas before narrowing

**[speculation]** Probabilities below are subjective chances of a useful, discriminating pilot under the available resources—not publication probabilities. “New” means structurally different from the existing queue and earlier reviews; it does not claim global priority.

| Idea | [speculation] Mechanism and first prediction | Pilot probability |
|---|---|---:|
| **1. Reassign futures, preserve both marginals — new** | Hold corrupted states and target counts fixed; alter only their coupling. If training the wrong continuation is causal, compatible pairing improves generation without changing corruption severity. Exact-root permutation first. | 35% |
| **2. Invert the training channel — new, high risk** | Treat learned denoising conditionals as blurred conditional probabilities \(Wm\). Recover selected predictive components by regularized inversion. Predict measurable attenuation before training; test the discrete root where \(W\) is tractable. | 15% |
| **3. Corrupt with a joint-preserving Markov kernel — new** | Construct corruption invariant under \(P(A\mid Y)\), rather than merely plausible under \(P(A)\). This preserves the context–target joint in the ideal case. Test exact finite distributions, then an approximate released-model kernel. | 30% |
| **4. Audit the hierarchy as a martingale of evidence — new** | For terminal observables \(f(X)\), calibrated forecasts \(E[f\mid C_{\le k}]\) obey the tower property. Systematic forecast drift identifies incompatible refinement beliefs; test lexical counts, negation and relations. | 40% |
| **5. Change temporal persistence of corruption — new** | Compare fixed versus resampled replacements at the same empirical marginal channel. A difference isolates optimization and gradient-noise effects from population target ambiguity. Cached-corruption short pilot. | 30% |
| **6. Check Bayes reciprocity across levels — new** | Fit forward and reverse conditional probes. Their probability ratios should agree with a common joint; inconsistent cycles expose incompatible local beliefs. Test whether violation predicts bad rollouts better than NLL. | 35% |
| **7. Train on exact decoded-text events — extension of E1** | Optimize probability of the whole decoder fiber, rather than a selected code path. Predict useful gradients after canonical fine-level loss has become mostly representation choice. | 25% |
| **8. Generate a syntax program before lexical realizations — new, high risk** | A balanced grammar derivation can expose independent subproblems in logarithmic depth. A small controlled-language model tests whether choosing operations/relations is a better separator than an averaged lexical bag. | 15% |
| **9. Preserve source information during consistency training — extension** | Constrain sensitivity to controlled source distinctions while shortening trajectories. Predict removal of constant-output collapse without assuming exact convergence accelerates. | 25% |
| **10. Make sampling match the denoising experiment — new** | On a small model, alternate a known corruption with its correctly trained posterior instead of using the denoiser as an unmatched ancestral conditional. Predict recovery of the target distribution where one-pass continuation fails. | 30% |

**[speculation]** Ideas 2 and 8 deserve consideration despite obvious objections. Channel inversion can produce negative probabilities and amplify estimation noise, but the damaging component may be low-dimensional and identifiable at the first levels. Grammar programs sound like retreating to handcrafted linguistics, but a representation that exposes conditional independence may be more valuable than one whose intermediate reconstructions merely look semantic.

**[speculation]** Neither warrants training a new large model. The relevant first tests are a tractable root-channel calculation and a small synthetic grammar with known dependencies.

---

### Direction 1 — Rank first: identify and repair the context–target coupling

**[speculation] Core idea:** Treat corruption as construction of a statistical joint, not as a choice of noise vocabulary. Establish a predictive account of which context distinctions it destroys, then change the pairing between fixed contexts and fixed targets to test that account causally.

**[speculation]** The potential breakthrough is a rule for diagnosing harmful training contexts before committing to a full run. The intervention is useful even if it merely rescues a deliberately damaged model: it identifies something a replacement-distribution comparison cannot. It becomes a competitive method only if it also improves the healthy baseline.

#### The exact first experiment

**[speculation]** At L1, the visible unconditional state is determined by the root code and fixed architectural processing. This gives a rare place where “compatible with the supplied context” can be made exact.

**[speculation]** For each root-corrupted training row, record canonical root \(a_i\), supplied root \(u_i\), and its complete L1 target block \(y_i\). Build a directed multigraph with edges \(a_i\to u_i\). Select a large balanced circulation, so each root has equal selected incoming and outgoing counts. Reassign target blocks within this subset so row \(i\) receives a block from a row \(j\) with \(a_j=u_i\).

**[speculation]** This construction preserves:

- every corrupted input and every mask;
- the exact multiset of L1 target blocks;
- every other target;
- model capacity, optimizer, update count and inference procedure.

**[speculation]** Choose assignments randomly within root classes, using no target content or evaluation score. Selection using only roots and masks avoids preferentially selecting “easy” target blocks within a root. Report the unmatched fraction; leave those rows unchanged.

**[speculation]** This is locally compatible supervision at the first refinement. It does not make later corrupted conditionals proper, and it cannot be transferred to WMT by swapping targets across different source sentences.

#### Ambition, effect size and probability

**[speculation]** The primary target is **(c)**: a replicated account of how early codes function as predictive evidence, with a target-association intervention. A sharper proposed criterion is:

> A pre-training channel model predicts the direction of conditional-information change, and changing only the context–target coupling removes at least 25% of generation damage across three fresh seeds.

**[speculation]** Forecast: **5–10 Gen-PPL points of rescue**, without additional inference calls. Probability of meeting that intervention criterion: **35%**. Probability of a broader, publishable predictive mechanism after the factorial and diagnostics: approximately **45%**. Probability this immediately beats healthy NCM sufficiently for (a) or (b): **below 10%**.

**[speculation]** These are different bets. A collapse rescue must not be advertised as a quality advance over the healthy model.

#### Kill experiment: at most 5 CU

**[speculation]** First, CPU preprocessing must achieve **≥80% matching coverage** and verify exact input/target-multiset invariants. Use a finite toy problem to verify the channel-smoothing calculation and reassignment.

**[speculation]** Then run the reassigned arm from the released checkpoint at seed 0, replaying M2’s protocol, with a **5-CU total cap**, including preprocessing/proposal overhead and evaluation.

**[speculation] Prediction:** **≥25% rescue** of M2–M3 damage, matching direction under both scorers, with no deterioration in repetition or MAUVE that explains the gain.
**[speculation] Kill:** **≤10% rescue**, failure of the matching-coverage gate, or improvement attributable to reduced diversity. Between 10% and 25%, report the result but do not automatically fund expansion.

**[speculation]** The small clean L1-NLL difference is a real objection to this bet. The experiment earns its place because it is unusually identified, not because the existing evidence guarantees the first refinement is the culprit.

#### Replication within the current envelope

| [speculation] Component | CU |
|---|---:|
| M2/M3 at 25% clean, seeds 1–3 | 21.9 |
| Reassigned-target arm, seeds 1–3, conditional on pilot | 13.8 |
| Optional 0%-clean M2/M3 factorial cells, seeds 1–3 | 21.9 |
| Pilot, shared diagnostics and reserve | Covered by the final 77-CU queue |

**[speculation]** The minimum causal package is the first two rows plus diagnostics. The second factorial half is justified only if the clean-row mechanism becomes part of the main claim. All training effects require the stated three fresh seeds and gain exceeding twice pooled SD.

**[speculation]** A larger programme of approximately **130–140 CU**, or **36–46 CU beyond the roughly 94 remaining**, could add a second released unconditional model family and multiple prospectively chosen channel strengths. That would be worth considering only after the exact pairing intervention succeeds; the additional value is a transferable prediction, not a larger NCM ablation table.

#### Strongest reviewer objection

**[speculation]** “You restored an appropriate training association after deliberately breaking it. This is expected.”

**[speculation]** The answer must be quantitative and level-specific: why the original channel was useful in one form and destructive in another; which predictive components were suppressed; why matched clean loss missed the effect; and whether the channel model predicted an unobserved intervention. Without that prediction, this remains a controlled pathology study.

**[source]** Optimal-transport pairing itself is also prior art. Daegyu Kim et al., 2024, arXiv:2403.05069, use approximate transport in diffusion training for image generation. The proposed contribution cannot be “use transport”; it is the controlled causal estimand and the residual-hierarchy mechanism. [Approximate transport for diffusion](https://arxiv.org/abs/2403.05069).

#### How it uses the existing results

**[speculation]** W2M supplies the large phenomenon and matched channels. M1≈M2 motivates testing a saturated early bottleneck. The oracle curve motivates early-level localization but does not prove it. S0 permits a prediction that requires no new joint sampler. E1 warns against using total canonical NLL as the mechanism score. Sudoku supplies a boundary condition: external clues can reduce ambiguity about the desired future, so unconditional-text results need not transfer unchanged. J2/J2-F warn against letting the target generator move while attempting identification.

---

### Direction 2 — Rank second: learn probability of decoded text rather than canonical residual codes

**[speculation] Core idea:** Use E1 to turn exact decoder equivalence from an evaluation observation into a training objective. Train the generator to allocate probability to all code continuations producing the correct rendered output, while retaining the original coarse-prefix objective.

**[speculation]** This changes what the generator is asked to learn. It is stronger than nearest-code soft labels because it respects coordinated alternatives and exact decoder behavior.

#### The objective and estimator

**[speculation]** At a fixed canonical prefix \(a\), let

\[
\rho_\theta(x\mid a)
=
\sum_{z:D(a,z)=x}Q_\theta(z\mid a).
\]

Then

\[
\nabla_\theta[-\log\rho_\theta(x\mid a)]
=
-\mathbb E_{Z\sim Q_\theta(\cdot\mid a,D(a,Z)=x)}
[\nabla_\theta\log Q_\theta(Z\mid a)].
\]

**[speculation]** An exact accepted ancestral sample from the current model supplies an unbiased conditional score estimate. No derivative through discrete sampling or the frozen decoder is needed.

**[speculation]** This does not make sampling cheap. Expected rejection cost is \(1/\rho\), and the smallest-probability training cases can dominate runtime. E1 must therefore provide the cost gate.

**[speculation]** Start at cut 8 on a prospectively defined training pool with a measured high-hit regime. Keep the sampling target current, retain all selected training items, and stop the experiment if its runtime/acceptance gate fails. Do not silently skip hard items, substitute canonical targets on failures, or call a finite candidate-set likelihood the full fiber likelihood.

#### Why it might matter—and why it might not

**[speculation]** If much of fine-level CE trains a model-specific preference for the encoder’s path, removing that demand could release shared capacity and reduce harmful gradients. But if fine-level path choice is harmless, or the useful gradient is nearly zero, the method may reduce to discarding a loss term.

**[speculation]** Therefore the indispensable comparator is **omit the fine canonical loss**, not only ordinary continuation training.

**[speculation]** This direction targets (c), with a possible modest quality improvement. A sharper result would show that exact-output training improves conditional text probability beyond simple loss omission while canonical NLL becomes worse.

**[speculation] Forecast:** **0.5–1.5 Gen-PPL improvement** over a matched healthy continuation control if the method works; **20% probability** of a useful generation result. A mechanistic objective-comparison result is somewhat more likely, approximately **35%**. It is not a credible near-term claim to close half the AR gap.

#### Kill experiment: at most 5 CU

**[speculation]** Use three short matched-update arms:

1. canonical losses;
2. remove the selected fine-level canonical losses;
3. replace those losses by exact-fiber score gradients.

**[speculation]** Keep the same coarse examples and objective, preserve every selected training item, and measure actual proposal/decode costs. Use separate validation windows for the outcome.

**[speculation] Prediction:** fiber training improves held-out conditional exact-text loss by **≥0.2 nat/window** beyond omission, without degrading coarse canonical NLL by more than **2 nats/window**.
**[speculation] Kill:** improvement below **0.05 nat**, unstable rejection cost, or no distinguishable benefit over omission. If the selected stratum cannot be handled within the cost cap, the affordable version is dead even if the population objective is attractive.

#### Full plan below 90 CU

| [speculation] Component | CU cap |
|---|---:|
| Cost/gradient pilot | 5 |
| E1-informed training-pool preparation and estimator checks | 3 |
| Three fresh seeds: canonical, omission, fiber; approximately 2.7/2.7/4.6 CU per seed triplet | 30 |
| Additional equal-training-compute control, three fresh seeds, if claiming a method gain | 13.8 |
| Held-out fiber estimates, full sample evaluation and timing | 5 |
| Reserve | 6 |
| **Total** | **62.8** |

**[speculation]** The equal-update comparison identifies the objective intervention; the additional compute-matched control is needed for a training-efficiency claim. The inference procedure remains unchanged, but extra training work must be reported.

**[speculation]** A **130–150 CU programme**, **36–56 CU above the remaining 94**, could add replicated WMT training and investigate earlier cuts where posterior sampling is much harder. That expansion is justified only if exact-event training beats loss omission on the first domain.

#### Strongest reviewer objection

**[speculation]** “This is latent-variable marginal likelihood, and the improvement is just coarse-loss reweighting.”

**[speculation]** Both are serious. The identity is not novel. The answer is an effective estimator for this decoder’s exact fibers, measured scope, and a result beyond the omission control. If omission performs equally well, report that and abandon the stronger method claim.

**[speculation]** E1 is central to this direction. A2/A4 motivate separating canonical-path improvement from output probability. S0 explains why changing the loss may be more useful than adding a neighbor model. The oracle curve warns that fine-only gains may have a low ceiling. J2-F suggests that decoded-output quality and exact-code convergence can separate, but does not prove fiber training will improve either.

**[speculation] My ranking is therefore:** fund the context–target intervention first; retain exact-fiber training as the alternative if E1 is strong and the first intervention fails its inexpensive gate. Do not fully fund both this month. Consolidating a well-identified mechanistic finding is preferable to assembling several unreplicated rescues.

### B5. What I would stop or change

- **[speculation] Cut automatic W2C stage 2.** Flag replication is a substantial expenditure for a claim whose strong prediction already failed and whose interpretation remains broad.
- **[speculation] Replace the no-clean M0 pilot with no-clean M3.** It fits the deciding factorial.
- **[speculation] Expand W2M-G’s design, not its checkpoint count.** Prioritize common context banks and crossed continuations for M2/M3; add M4 only for the routing question.
- **[speculation] Stop treating wrong-slot count, total NLL or probe AUC as sufficient mechanism statistics.**
- **[speculation] Keep the coupled sampler, new large tokenizer, tier-M AR comparison and further Sudoku training off the queue.**
- **[speculation] Stop text-Jacobi training if D1 fails.** E1 may still make exact-code convergence scientifically interesting, but it does not pay the latency bill.
- **[speculation] Do not fund richer target transport after an exact-root failure without new evidence.** Otherwise the proposed kill experiment is not a kill experiment.
- **[speculation] Keep temperature and stopping decisions on validation, freeze them, and report nulls.** Existing test observations cannot become prospective gates.

## Revised queue for this month

**[speculation]** These are conditional allocation caps using the brief’s cost estimates, not launch approvals or amendments. The maximum is **77 CU**; untriggered allocations remain unspent. Every new implementation still needs registration, review and deep audit. No individual proposed run exceeds 15 CU.

| Item | CU cap | Gate | What it decides |
|---|---:|---|---|
| **[speculation]** Registration, replay invariants, target-permutation toy checks and audits | CPU | Before new GPU work | Whether the comparisons implement the intended estimands |
| **[speculation]** D1 | 1.5 | Existing registered timing gate | Whether text-Jacobi latency remains alive |
| **[speculation]** E1 across released domains/checkpoints | 3.0 | Valid uncertainty/censoring; runtime cap | Scope of exact decoder equivalence; feasibility of direction 2 |
| **[speculation]** Mechanism evaluation plus revised W2M-G | 2.5 | Common context banks; coherent-prefix controls | Conditional change versus state visitation; early selection versus rendering |
| **[speculation]** M3 with 0% clean rows, seed-0 pilot | 2.7 | Same 600-step protocol otherwise | Whether the clean-row factorial merits confirmation |
| **[speculation]** W2C: M2/M3 at 25% clean, fresh seeds 1–3 | 21.9 | Integrity checks; fixed analysis | Replicated replacement-distribution effect and NLL dissociation |
| **[speculation]** W2C: M2/M3 at 0% clean, fresh seeds 1–3 | 21.9 | No-clean pilot improves control by ≥3 Gen-PPL, and clean-row mechanism will be claimed | Clean-fraction main effect and interaction |
| **[speculation]** One direction-1 kill experiment | 5.0 | ≥80% exact-root matching; invariant checks pass | Whether changing target association rescues damage |
| **[speculation]** Reassigned-target arm, fresh seeds 1–3 | 13.8 | Pilot rescues ≥25%, with quality controls | A causal intervention beyond descriptive W2M replication |
| **[speculation]** Protected reserve | 4.7 | Re-evaluation, runtime overrun or one essential discriminating check | Protect the final claim rather than add another recipe |
| **[speculation] Total maximum** | **77.0** | All optional gates pass | One coherent mechanistic programme |
