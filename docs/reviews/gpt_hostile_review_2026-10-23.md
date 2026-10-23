# External hostile review (GPT), 2026-10-23 — verbatim

Reviewed commit `1c7185c` of `claude/ncm-phase2-analysis-finetune-pkh64d` (before W1a, P1b, S1, S1a and J1 were
registered). Triage and actions: `docs/reviews/triage_2026-10-23.md`, log R023.

Prompt (summary): hostile review of the P1 (R018), C4 (R019) and w0 (R020) verdicts; the W1 design; the planned
Sudoku tier-L and WMT14 arms; and the related literature. Numbered findings with severity and a check of at most
1 GPU-hour each.

---

Reviewed the requested branch at cached commit `1c7185c`; remote refresh failed. Headline results below match committed JSON unless flagged. “Fatal” means fatal to the inference, not the project.

1. Major — P1 is inconclusive; the proposed explanation is unproven.
−57.3 nats cannot establish low total correlation: fitted cross-entropy differences mix dependence with approximation and optimization error. Initialization is a plausible explanation, not a demonstrated cause. Moreover, zero previous-code embeddings do not make the AR probe equal the factorized probe: their attention masks still differ. P1b is referenced but absent from the reviewed preregistration.
Check, CPU minutes: copy identical weights into both probes, zero previous-code embeddings, disable dropout, and compare logits on one batch. Then verify the estimator on synthetic independent versus duplicated codes before another pilot.

2. Fatal — The entropy bound is valid; its mechanistic interpretation is not.
For deterministic encoding, H(C) ≤ H(X) ≤ CE_AR. The reported values imply an estimated excess fraction of 85.915%, rounding to 86%; finite validation estimates do not establish a literal population guarantee of “≥86%.” R008’s earlier assertion that determinism implies equality is wrong without invertibility.
Crucially, this excess includes irreducible error from factorization, not just difficulty learning canonical labels. TC could dominate it. Example: eight copies of one fair bit have one bit of joint entropy, eight bits of optimal factorized cross-entropy, and 87.5% excess entirely explained by TC. The bound also does not transfer the numerical percentage to the released model.
Check, CPU seconds: enumerate that duplicated-bit distribution and calculate its joint entropy, marginal cross-entropy and TC.

3. Major — C4 fails its registered screen; broad mechanistic falsification overreaches.
The observed −1.95 points on released hard supports “carry hurts under this training recipe.” It does not establish that carry cannot raise propagation depth in a competent model. Lower training CE with worse decoding also fits overfitting or optimization differences, besides exposure bias.
However, “almost no power” is itself overstated: zero successes among 441 deep puzzles gives an approximately 0.68% one-sided upper 95% bound on RNCM’s solve probability. A +10-point improvement would have been detectable. The limitation is the tested regime, not merely sample size.
Check, ≤1 GPU-hour: reevaluate existing checkpoints with clean versus generated conditioning, reporting held-out CE, clue violations and cell accuracy by depth. This distinguishes general degradation from degradation specific to rollout.

4. Major — w0 establishes an oracle intervention curve, not an attribution or an upside bound.
Recalculation gives 86.34% of the PPL reduction at L1–L5; L8–L10 contribute 0.293 PPL, 3.96%, rather than ~0.2/3%. L7–L11 contribute 7.27%, rather than 5%.
Supplying progressively more real-text codes changes the conditional task and suppresses downstream interactions. Small fine-level effects under correct coarse codes do not imply small effects after erroneous coarse plans. Nor does this bound what a retrained level schedule could achieve. The 48%-versus-50% promotion decision is below the diagnostic’s stated uncertainty. R016’s pooled linear probes do not resolve these issues.
Check, ≤1 GPU-hour: on 128 windows and three noise seeds, repair L8–L10 after both sampled and oracle coarse plans, using paired fixed Gumbels. Report scorer cross-entropy differences and bootstrap intervals.

5. Major — W1 can screen recipes; it cannot reliably distinguish H6–H9 at the stated threshold.
SE≈0.30 is for one mean, not its difference from A1. With independent arms, SE of the difference is 0.417. A 0.5 improvement is only 1.20 SE; under a normal approximation, an identical-quality arm passes that PPL gate about 11.5% of the time. Approximately 1.04 PPL is needed for 80% power at one-sided 5%, before other gates. Pairing might help, but its covariance is unmeasured.
The contrasts also disagree with the screen: H8 requires A5 versus A3; H7 requires A4 versus A3. Advancing everything against A1 can discard supporting evidence. Failure to pass is not falsification.
Check, CPU minutes: simulate the screen using SD=0.59 and several plausible paired correlations; calculate power for each registered contrast, including selection across arms.

6. Major — FLOP matching is defensible for a budget comparison, but confounds mechanism.
The 4000/3000/2400-step arms differ in examples, optimizer updates, cumulative weight decay and warmup fraction. Equal estimated FLOPs therefore tests complete training procedures, not corruption alone. Re-quantization and tokenizer work are also incompletely represented in the generator-only accounting.
Fine-tuning a released checkpoint is a fair test of improvement from that checkpoint at fixed adaptation compute, as the preregistration correctly limits it. It is not a decisive test of from-scratch objectives. A null can reflect adaptation time or the inherited basin. I could not verify that the released checkpoint was actually converged.
Check, ≤1 GPU-hour: run two short branches for identical updates/data and LR schedules, alongside the compute-matched comparison; plot changes against both updates and measured GPU time.

7. Major — W1 does not isolate policy matching, target repair or diversity preservation.
Uniform and on-policy replacement produce different effective corruption rates and latent damage. Thus an on-policy win could mean “less destructive noise.” R014 measures predictions under clean contexts, not the compounded inference distribution. W1’s T=1 matches its initial evaluation temperature, but entropy-matched resampling changes that relationship. Two-pass policy generation also retains gold L0 internally before substituting sampled L0 afterward.
Re-quantization changes subsequent inputs as well as labels (`used = c_lab`), so H7 is not a labels-only intervention. Soft labels can help through generic regularization. Finally, per-sample unigram entropy can remain unchanged when every output repeats the same story; five MAUVE quantization seeds do not measure sampling uncertainty.
Check, ≤1 GPU-hour: log per-level replacement rates/distances; compare corruption sources at matched damage on 256 windows. Add CPU corpus-duplication and shuffled-neighbor soft-label controls. Fix temperature selection on validation data before final evaluation.

8. Fatal — The proposed Sudoku kill condition can reward an already-solved problem.
Training-free verifier stopping already achieves 90.3% at 2.3655 passes, versus 86.15% sequential. Consequently, +3 points over sequential can pass without training.
The baseline must include the frozen model and every fine-tuned control under the same verifier, cap and initialization, with quality–NFE curves. Repair additionally changes feedback, state distribution and repeated exposure to difficult puzzles. A gain cannot automatically be assigned to learning repair. R021’s 88% on 250 generated puzzles does not establish distribution equivalence.
Check, ≤1 GPU-hour: evaluate frozen/control/repair on the same held-out 500 puzzles with verifier on/off and feedback present/zero/shuffled. Count newly solved sequential failures; include failed puzzles in mean NFE and report tail NFE. Register the comparator before training.

9. Major — WMT14 needs stronger controls than frozen sequential decoding alone.
Verified baselines are 11.0666 exact passes, BLEU 22.626, and 10.5884 stable-output passes, BLEU 22.639. Also test frozen capped Jacobi: eight passes might already satisfy the quality tolerance.
Require ≤8.25 passes relative to the 11-pass sequential baseline, and <1 BLEU loss against the original model. Include an equal-compute ordinary fine-tune. Exact Jacobi equality after training means equality to the new model’s sequential output; it does not preserve the original model. Faster convergence could arise from shorter, more generic translations or weakened source dependence.
Check, ≤1 GPU-hour: evaluate frozen and fine-tuned checkpoints at caps 6/8/9/full, with exact/stable stopping. Save hypotheses, lengths and per-sentence pass counts; bootstrap BLEU and inspect paired translation changes.

10. Major — Verifier halting legitimately counts toward criterion (d), but current accounting overstates execution savings.
The brief does not require learned halting. A checker using only rules and clues is a legitimate adaptive rule; no solution access is needed. Sudoku’s half is already satisfied operationally by the proposed stopping rule; text remains necessary.
However, w0 derives stopping times from trajectories, and the shared decoder still forwards stopped examples within the batch. Mean logical NFE therefore does not demonstrate realized GPU savings. Reported “FLOPs ratios” use position ratios, omitting attention and decoder/checker costs. Board feedback also removes the triangular dependency that guarantees convergence to the sequential NCM output.
Check, ≤1 GPU-hour: replay 256 examples with active-batch compaction or batch size one; count processed examples, generator/decoder calls and wall time. Test feedback decoding for cycles and cap hits.

11. Minor — Corruption and re-quantized repair have substantial precedent; their transfer remains uncertain.
Bengio et al., Scheduled Sampling (2015) reported benefits; Huszár (2015) showed its objective can be statistically inconsistent. Lee et al. (2018) established parallel iterative refinement; Qian et al., GLAT (2021) used prediction-guided training. Most directly, Han et al., Infinity (CVPR 2025) corrupts hierarchical bits and re-quantizes subsequent targets.
These support plausibility, not a predicted W1 winner. Infinity supports B1b’s mechanism, specifically; its corruption is not on-policy.
Check, CPU minutes: map each cited method’s corruption source, conditioning and target construction against A1–A6. Separate inherited mechanisms from the specific NCM hypothesis.

12. Major — Jacobi consistency training is established prior art, with an explicit quality–speed tradeoff.
Santilli et al. (2023) established training-free fixed-point translation decoding. Kou et al., CLLMs (2024) trained intermediate Jacobi states toward fixed-point predictions while retaining an AR quality loss. Hu et al., Jacobi Forcing (2025) trained on generated parallel trajectories.
These predict that training can improve convergence, but not that NCM will meet 8.25 passes within one BLEU. Consistency with a poor endpoint can simply accelerate poor output.
Check, ≤1 GPU-hour: compare endpoint consistency versus adjacent-state consistency on one fixed small trajectory pool, retaining the same quality-preserving loss and compute budget.

13. Minor — Recursive Sudoku refinement is supported; this particular repair arm is not predicted to win.
Palm et al., Recurrent Relational Networks (2018), Wang et al., HRM (2025), and Jolicoeur-Martineau, TRM (2025) demonstrate repeated reasoning/refinement. Their state representations, supervision and difficulty distributions differ from NCM. They neither validate C4’s carry implementation nor predict gains over the existing verifier baseline.
Check, ≤1 GPU-hour: on held-out training-style puzzles, measure wrong→correct and correct→wrong transitions by repair age, with verifier disabled. Establish that additional learned passes repair failures beyond merely retaining transient solutions.

All cited papers’ existence was verified. The stated w0 SE≈0.4 and “converged model” characterization were not independently substantiated by the reviewed result files.

(Links in the original: arXiv 1506.03099, 1511.05101, ACL D18-1149, ACL 2021.acl-long.155, arXiv 2412.04431,
ACL 2023.acl-long.689, arXiv 2403.00835, 2512.14681, 1711.08028, 2506.21734, 2510.04871.)
