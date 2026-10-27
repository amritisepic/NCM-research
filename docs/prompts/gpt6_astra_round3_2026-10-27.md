# Research review, round 3 (for an external model), 2026-10-27

You reviewed this project twice (`docs/research_review_2026-10-26.md`, `docs/research_review_2026-10-27.md`). Your
round-2 outcome table for W2M can now be applied: the eight-arm screen has reported (seven arms; M5 skipped as W2Ma
allowed). This round is narrow: interpret W2M, then design the confirmation that will carry the project's main claim
(criterion (c)) at minimum cost. Same ground rules: label **[ours]** / **[source]** / **[speculation]**, no invented
citations or numbers, predictions with numbers and the result that would overturn them. The repository is public
(github.com/amritisepic/NCM-research, branch `claude/ncm-research`, `log.md` R045-R050, `docs/preregistration.md` W2M
and W2Ma); this brief is self-contained.

## 1. Setup (recap)
W2M fine-tunes the released 88.6M TinyStories NCM generator for 600 steps per arm, seed 0, batch 256, lr 5e-5, canonical
labels, W1's corruption rates (0.75 at L0 linearly to 0.25 at L10), **25% of rows left clean** in every arm (added so
that all-zero flags are trained), one mask stream (every arm with W1's rates corrupts the same slots), fp32 policy
forwards. Final evaluation: 16 groups of 256 unconditional samples (seeds 42, 0-14), fp32; D(X) = Gen-PPL(X) -
Gen-PPL(M0) under GPT-2 large, paired by sample seed. Arms: M0 uniform replacement; M1 the current model's one-step
samples (A3's on-policy corruption); M2 samples from a frozen copy of the released model with the canonical code
excluded (every masked slot wrong); M3 uniform-wrong; M4 = M2 + truthful flags (a zero-initialised learned vector
added in proportion to the flag; flags 0 at inference); M6 = M1 with proposals sampled under the level-0 code actually
used; M7 uniform at A3's lower realized rates. Before the run we found (R043) that the flags saturate at these rates
(99.6% of flag slots on corrupted rows are 1), so M4 carries a row-level indicator, not positional information.

## 2. Results (all [ours], seed 0)
| arm | Gen-PPL | D (± SE over 16 sample groups) | TS-33M | entropy | MAUVE | bigram gain | adj. dup/1k | clean val NLL |
|---|---|---|---|---|---|---|---|---|
| A0 released | 16.16 | | | 3.97 | 0.827 | 2.144 | 6.6 | 861.8 |
| M0 uniform | 21.71 | 0 | 18.73 | 3.989 | 0.865 | 2.054 | 8.3 | 753.8 |
| M1 current policy | 42.50 | +20.79 ± 0.27 | 51.80 | 4.026 | 0.468 | 1.698 | 18.2 | 751.2 |
| M2 frozen plausible-wrong | 42.62 | +20.91 ± 0.35 | 51.74 | 4.018 | 0.412 | 1.691 | 21.3 | 757.9 |
| M3 uniform-wrong | 21.69 | -0.02 ± 0.10 | 18.65 | 3.992 | 0.865 | 2.059 | 8.3 | 753.9 |
| M4 M2 + truthful row flags | 31.58 | +9.87 ± 0.23 | 32.60 | 4.019 | 0.720 | 1.894 | 13.9 | 754.4 |
| M6 used-parent policy | 42.18 | +20.47 ± 0.37 | 51.44 | 4.021 | 0.454 | 1.692 | 18.8 | 751.2 |
| M7 uniform at A3's rates | 22.07 | +0.36 ± 0.08 | 19.14 | 3.993 | 0.860 | 2.052 | 8.3 | 741.2 |
- Registered reading: validity passes (D(M1) >= 5); M2 damaging, M3 healthy, M6 damaging, M7 healthy as predicted; M4's
  rescue ratio R4 = 1 - D(M4)/D(M2) = 0.53 (prediction >= 0.8 not met; < 0.2 would have overturned) -> "mixed". The
  sanity check fails: M0 is 5.55 above A0 (registered: within 1.0). Flag use (clean-label NLL with true minus zero
  flags) is not yet measured; the per-level row-level detectability probe (W2Ma) has not run.
- Training curves (256 samples at steps 100 / 300 / 600): M0 19.25 / 22.55 / 22.69, clean NLL 804.6 -> 753.8; W1's
  uniform control (no clean rows, 1800 steps) stayed at 16.0-17.1 from step 270 on, NLL 844-855. M1 40.8 / 39.1 / 41.1;
  M2 37.6 / 38.0 / 41.4; M4 37.5 / 31.7 / 32.5 (tracks M2 until the flag is learned); final flag-vector norm 0.15.
- Realized disagreement on corrupted rows, L0..L10: M2 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25; M1 0.75
  0.70 0.59 0.49 0.37 0.27 0.20 0.08 0.15 0.14 0.12. D(M1) = D(M2) despite far fewer wrong slots from L2 on.
- In W1 (no clean rows), the on-policy arm's clean NLL was 42 nats above its control. Here the collapsed arms are within
  4 nats of M0's: they still predict real data as well as the control while their samples collapse.
- Other numbers since round 2: S0 failed (coupled sampler dropped); J1a bootstrap J2-F - J0 at cap 11 = -0.48 [-0.84,
  -0.12]; J1's WMT BLEU loss is 92% n-gram precision with the brevity penalty set to 1 (your prediction held).

## 3. What we propose (`docs/plan_2026-10-27.md` section 9; ~77 CU left this month; A100 ~5.4 CU/h)
- **W2M-G** (eval-only, ~2 CU) on the W2M checkpoints: (i) oracle-intervention curve, real levels 0..j supplied and the
  rest sampled, j in {-1, 1, 3, 5}; (ii) per-level predictive entropy when levels < k are the model's own samples vs
  real data. Our candidate account predicts real coarse levels rescue M1 / M2 far more than M0, and own-sample entropy
  >> real-context entropy in M1 / M2 only.
- **M0 without clean rows** (seed 0, ~2.7 CU): does dropping the clean rows return the control to ~16-17?
- **W2C stage 1:** M2 and M3 at fresh seeds 1-3 (~22 CU). **Stage 2:** M4 and your random-row flag control at fresh
  seeds 1-3 (~27 CU), only if the mech eval shows flag use.
- Already queued: D1 (batch-1 wall-clock gate for criterion (d)), the fixed mech eval, E1 (with an analysis-only
  amendment adopting your confidence sequences).

## 4. Questions (spend most effort on 1-3)
1. **What does W2M license, given the drifted control?** Your round-2 table says a substantially deteriorated M0 calls
   for an orthogonal factorial or descriptive findings only. Is the M2-vs-M3 contrast interpretable when both sit on a
   drifted baseline? Which factorial is worth its cost (clean-row fraction x replacement distribution, at which seeds)?
   Should W2C keep the 25% clean rows (needed for M4) or not?
2. **Account for the clean-NLL invariance.** Collapsed arms predict real data as well as the control. Give the
   accounts that fit: discounting of contexts that look like model samples, detectability through cross-level
   inconsistency of corrupted rows (a wrong coarse code next to canonical finer codes), damage confined to fine-level
   rendering, or others. For each, the eval-only measurement on existing checkpoints that separates it, with a
   pre-registrable prediction. Is W2M-G as proposed the right instrument, and what controls does it need?
3. **W2C design at minimum cost.** Arms, seeds, protocol and readings for the strongest (c) claim this budget allows.
   Is "M2 vs M3 at three fresh seeds" sufficient for a plausibility claim, given that M2 and M3 differ in displacement,
   target conflict and context statistics at once (your round-2 point)? What single extra arm would most narrow it?
4. **M4's partial rescue (R4 = 0.53).** Candidates: limited flag capacity or training time (the flag is learned
   between steps 100 and 300, then flat), a clean-regime exposure bias at flag 0 (the flag-0 branch is trained only on
   clean rows), regime-bias effects. Which measurement or control distinguishes them most cheaply? Is your random-row
   flag control still the decisive one?
5. **Why does damage not scale with wrong slots?** M1 corrupts far fewer slots than M2 from L2 on, yet does equal damage.
   What does that predict about where the damage enters (levels 0-2?), and what is the cheapest test (for example M2
   with replacement restricted to L0-L2, or to L3+)?

## 5. Output format
- First, at most 8 bullets: the verdict on W2M and the changes you would make to section 3, ranked.
- Then per question: the answer, the pre-registrable prediction (with a number), and the result that would overturn it.
- End with a revised queue for ~77 CU: item, CU, gate, what it decides. Dense; no general encouragement.
