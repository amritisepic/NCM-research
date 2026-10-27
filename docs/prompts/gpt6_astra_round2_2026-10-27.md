# Research review, round 2 (for an external model), 2026-10-27

You reviewed this project on 2026-10-26 (`docs/research_review_2026-10-26.md`). Since then the frozen-teacher WMT14
run (J2-F) has finished, we checked W2M's design against your warnings, and we re-planned around your review. This
round is narrower than the last. I want a hostile review of the new plan and rigorous design help for four
experiments, not a new survey. Go deep where the plan rests on an assumption; be brief where it does not.

Ground rules, as before: label every statement **[ours]** (a number from this brief), **[source]** (a published paper
or official code: authors, year, venue or arXiv id, and the setting a number comes from) or **[speculation]**. If you
are unsure a paper exists or says what you remember, say so; do not invent citations or numbers. The repository is
public (github.com/amritisepic/NCM-research, branch `claude/ncm-research`: `log.md` R042-R043,
`docs/plan_2026-10-27.md`, `docs/preregistration.md` W2M and W2Ma), but this brief is self-contained.

## 1. Minimal recap (all [ours] unless marked)
- NCM generates a 128-token window coarse-to-fine: a frozen residual-quantized tokenizer gives a 12-level code
  hierarchy (TinyStories: 1, 8, 16, ..., 128 codes; 641 per window; 4096 codes per level); a transformer predicts all
  codes of level k in parallel, independently given the cumulative reconstruction of levels < k. WMT14 De-En: 11
  levels, target given the source as a prefix. Released checkpoints: TinyStories 88.6M generator + 36.4M tokenizer,
  WMT14 120M, Sudoku-hard 25.5M.
- Success bar (any one, >= 3 seeds, gain > 2x pooled seed SD, pre-registered): (a) beat matched NCM on the full suite
  at <= 1.5x NFE; (b) close half the NCM -> AR gap at >= 3x fewer NFE; (c) a replicated mechanistic finding about
  what the levels encode; (d) an adaptive-compute rule saving >= 25% NFE at < 1 point quality loss on Sudoku AND a
  text task, with batch-1 wall-clock and an attention-inclusive cost.
- Anomalies you already know: released TinyStories clean code NLL 861.8 nats/window, 46.8% of it in L8-L11; continued
  training lowers code NLL but not quality (A1 control 844.0 nats, Gen-PPL 16.60 vs released 16.16; A2 re-quantized
  labels 749.3 / 16.29; A4 on-policy + re-quantized 707.4 / 36.81, the best NLL with the second-worst samples); the
  on-policy collapse (A3 41.4 Gen-PPL; local order broken, plans intact); exact Jacobi across levels: Sudoku-hard
  6.41 passes, verifier stop 2.37 passes and 90.3% solved vs 86.15% sequential; WMT14 11.07 passes (no saving).
- Cost model: one Jacobi pass runs every level. By position count (no attention term), a WMT14 Jacobi pass is 0.298
  of a whole eager sequential decode (sequential: 11 level steps, each recomputing the source prefix), i.e. ~3.3x an
  average sequential step. Sudoku: 0.29.

## 2. New results
**J2-F (WMT14, frozen released-model teacher, seed 0; R042).** Same recipe as J2 (CLLM-style consistency on half of
each batch, teacher forcing on the other half, 2000 steps), except that trajectories and fixed points come from a
frozen copy of the released model instead of the current model.

| | val passes | val BLEU | test passes | test BLEU | test len ratio | val NLL | shared outputs (test) |
|---|---|---|---|---|---|---|---|
| J0 released | 10.92 | 23.55 | 11.07 | 22.61 | 0.930 | 301.4 | 0% |
| J1 teacher forcing only, 3000 steps | 11.53 | 22.82 | 11.62 | 21.81 | 0.930 | 257.9 | 0% |
| J2 moving teacher | 9.36 | 18.53 | 8.89 | 16.02 | 0.750 | 284.1 | 16.2% |
| J2-F frozen teacher | 10.29 | 22.90 | 10.39 | 22.13 | 0.922 | 283.4 | 0.3% |

- Registered verdicts: J2-F misses the screen (val passes <= 9.0 needed) and the kill condition holds (5.6% fewer
  test passes). The registered prediction holds: J2's collapse onto two input-independent strings needed the moving
  teacher. One seed each.
- Exploratory: training moved quality to early passes without moving exact convergence. Test BLEU after at most c
  passes, J2-F / J0 / J1: c = 6 20.70 / 17.63 / 14.73; c = 7 21.57 / 19.70 / 17.65; c = 8 22.02 / 21.50 / 20.22;
  c = 9 22.10 / 22.51 / 21.56; c = 1 10.16 / 0.31 / 0.12.
- Exploratory, val only (test untouched): stopping at the first pass whose decoded text equals the previous pass's
  gives J2-F 7.80 mean passes (29% below 11), output identical to its exact output for 80% of sentences; J0 9.02
  (18%), 88%; J1 9.76 (11%), 92%. BLEU not yet computed. At 7.8 passes this is ~2.3x the sequential decode's
  positions.
- Fine-tuning tax: teacher forcing alone (J1) lowers val NLL by 43.5 nats/sentence and costs 0.73 val BLEU; adjacent
  duplicated words per 1k rise 17.1 (J0) -> 21.0 (J1) -> 24.4 (J2-F).

**W2M pre-reading check (R043).** W2M: eight 600-step tier-L TinyStories arms (M0 uniform, M1 current policy, M2
frozen plausible-wrong, M3 uniform-wrong, M4 = M2 + truthful flags, M5 = M2 + within-level shuffled flags, M6
used-parent policy, M7 uniform at A3's realized rates), W1's corruption rates (0.75 at L0 linearly to 0.25 at L10),
25% clean rows, fp32 policy forward. Using the repository's own flag code at those rates: on corrupted rows 99.6% of
flag slots are 1 (level 1 is constant within a row, >= 96.9% from level 2, 100% from level 5); truthful and shuffled
flags differ by 0.0014 per slot. The registered detectability probe pools all levels: its labels are 99.7% positive,
negatives sit 91% at level 1, and a score that only knows the level reaches AUC 0.95 (threshold for "contradicts
undetectable" was 0.90). Amendment W2Ma (before any W2M result): M5 is read as a near-replicate of M4; an M4 rescue
is read as row-level provenance; the pooled probe is not read and is replaced, if approved, by a per-level
clean-vs-corrupted-window probe with uniform-wrong contexts as the positive control.

## 3. The plan you are reviewing (`docs/plan_2026-10-27.md`)
~120 CU left (~93 after W2M; an A100-hour is ~5.4 CU). Primary target criterion (c); (d) gets one gate.
- **E1 (eval-only, ~2 CU):** for window x, canonical codes c*, reconstruction x^ = D(c*), cut level k:
  `N_k = -log Q(c*_{>=k} | c*_{<k})`; `rho_k = P(D(c*_{<k}, C_{>=k}) = x^)` with `C_{>=k}` sampled ancestrally at T = 1
  (Monte Carlo, <= 2048 samples per window, stop at 20 hits); `T_k = -log rho_k`; `R_k = N_k - T_k` (nats spent choosing
  the encoder's code among codes that decode to the same text; exact token equality, so a lower bound); text-NLL upper
  bound `U_k = -log Q(c*_{<k}) + T_k`. Models: TinyStories released, A1, A2, A4; WMT14 J0, J1, J2-F; Sudoku released.
  Predictions: released `R_8 >= 0.8 N_8`; for A2 - A1, A4 - A1, J1 - J0, `|dT_8| <= 0.2 |dN_8|`.
- **E2 (exploratory):** cost of independent draws without a joint model. Levels 3-6, models A0 / A1 / A3: damage from
  resampling one position, its neighbour, both independently, and a far pair; finer levels resampled with common
  random numbers. Your mode-mixing bridge predicts a positive adjacent interaction, larger in A3, ~0 for far pairs.
- **W2C (14-28 CU):** after W2M, the deciding contrast at 2 fresh seeds (seed 0 reported separately). If (b') holds:
  M2, M3, M4 (optionally M1).
- **D1 (~1.5 CU):** batch-1 latency, A100 and L4, fp32 (quality precision) and bf16: WMT14 J0 sequential, J0 Jacobi
  exact, J0 and J2-F Jacobi + stability stop; Sudoku sequential vs verifier stop. The text half of (d) stays alive only
  if J2-F + stop <= 0.75x J0 sequential latency in fp32 and the rule's val BLEU is within 1 of J0's. Then (only then)
  J2-F and J1 seeds 1-2 (~23 CU). Our prior that text passes: ~20%.
- **S0 (CPU), gate for your coupled sampler:** held-out gain from adding the canonical left-neighbour code to the
  frozen factorized head (zero-initialised low-rank bilinear term), levels 1-6. Pilot (<= 5 CU) only if levels 1-5
  gain >= 10 nats/window and >= 5% of their NLL, and E2's adjacent interaction is positive.
- Cut: tier M, Sudoku training arms, A1 seeds, J2/J2-F seeds.

## 4. Questions (answer in this order; spend most effort on 2, 3 and 5)
1. **Plan verdict.** Is (c) the right primary target for this budget? What would you cut, add or reorder? Which
   planned result would a hostile reviewer dismiss as trivial, and what would make it non-trivial? Is there a better
   use of the ~50 CU the likely path leaves unallocated that aims at a stronger claim?
2. **E1, rigorously.** Is `R_k` the right object, and is its interpretation sound when the canonical encoder is
   deterministic (H(codes) <= H(text)) but the model cannot reproduce its choice? Estimator: bias of `-log` of a hit
   rate, the stop-at-20-hits rule, windows with zero hits, and a tractable bound for coarse cuts where hits vanish
   (importance sampling centred on c*, annealed IS, a variational bound?). Is exact token equality the right
   equivalence, or should it be scorer-level? Prior art: marginal likelihood over tokenizations for language models,
   redundancy in residual codecs, anything that already reports this split for multi-scale VQ. What outcome would be
   surprising, and what would each outcome imply for how NCM should be trained (e.g. a loss over equivalence classes,
   given that independent soft labels over 16 nearest codes hurt: +0.85 Gen-PPL)?
3. **E2, identifiability.** Do overlapping footprints, the nonlinearity of the damage metric and resampling of finer
   levels make the interaction uninterpretable? Give the design you would register (damage metric, positions, sample
   counts, controls) and the result that would separate mode mixing from plain marginal broadening.
4. **J2-F and criterion (d).** Explain why frozen-teacher consistency improves early-pass outputs but barely speeds
   exact convergence (your triangular-map argument predicts what here?). Is the stability stop principled, and is
   J0 + the same rule the decisive comparator? From first principles, predict D1 for a 120M model at batch 1 on an
   A100 in fp32 and bf16 (Jacobi pass ~3.3x an average sequential step's positions). What is the fair sequential
   baseline (prefix recomputed vs cached)? Is a latency-only (d) claim worth making at 2.3x the compute?
5. **W2M under W2Ma.** With saturated flags, what can M2 / M3 / M4 still distinguish, and could an M4 rescue come from
   something other than provenance (for example the learned flag vector acting as a regime bias that the clean rows
   calibrate)? For each plausible outcome pattern of the eight arms, name the contrast to replicate and the exact claim
   it would license. Is per-level row-level detectability the right estimate of your a(s) = P(clean | observed
   context)?
6. **The WMT dissociation.** J1 lowers val NLL by 43.5 nats and loses 0.73 BLEU. Which accounts besides representation
   choice fit (fine-tuning on a 400k subset, reference style, length), and which measurement separates them?
7. **S0 as a gate.** Is a left-neighbour bilinear probe adequate, given your own parity counterexample (pairwise
   independence with a higher-order constraint)? Thresholds you would register instead.

## 5. Output format
- First, at most 10 bullets: the verdict on the plan and the changes you would make, ranked.
- Then, for each question: the answer, the pre-registrable prediction (with a number), and the result that would
  overturn it.
- End with a revised queue: item, CU, gate, and what it decides. Dense; no general encouragement. Being boldly wrong
  with a clear prediction beats being vaguely safe.
