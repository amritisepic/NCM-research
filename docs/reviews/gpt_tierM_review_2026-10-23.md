# External design review of the tier-M re-plan (GPT, 2026-10-23)

Saved verbatim as returned, except: two links to local files were replaced by their repo-relative paths
(`results/colab/w0_calib_diag/calib_M__calib.json`, `results/colab/w0_calib_diag/calib_M_ar__calib.json`), and the
LaTeX was kept as plain text. Triage: `docs/reviews/triage_tierM_2026-10-23.md`.

---

**The original plan fails the budget. A smaller test is affordable; a measurable ≤25M gap is unestablished.**

The saved rates imply **24.93 CU/NCM seed** and **13.95 CU/AR seed** at 15k/70k steps. Three seeds of just those
baselines cost **116.6 CU**, before evaluation or an improved method.

Costs below assume A100-40GB at **5.4 CU/h**: CU_N = 0.001662 S_N, CU_A = 0.0001993 S_A. These use the saved NCM
timing (`results/colab/w0_calib_diag/calib_M__calib.json`) and AR timing
(`results/colab/w0_calib_diag/calib_M_ar__calib.json`).

1. **Concrete designs**

All use **128-token packed windows, batch 256, training seeds 0/1/2**, the same corpus and frozen tokenizer. Totals
include **NCM baseline + AR + improved method, three seeds each**, assuming the method costs the same as NCM.

| Design | Body | NCM / AR steps | Matching | CU/seed NCM / AR | Nine-run CU |
|---|---|---:|---|---:|---:|
| A: shorten existing models | 512 × 8, ≈25.2M | 5k / 23.3k | FLOPs | 8.31 / 4.64 | **63.8** |
| B: prioritize equal exposure | 512 × 8, ≈25.2M | 7k / 7k | Data exposure | 11.63 / 1.39 | **74.0** |
| C: smaller body | 512 × 6, ≈18.9M | 6k / 27.34k | FLOPs | **7.65† / 4.28†** | **58.8†** |

- **A:** tests improvement during short training. Cannot establish behavior near convergence; only one-third of the
  planned NCM updates.
- **B:** tests equal-data behavior. NCM receives **4.66× AR's generator FLOPs**; cannot support an equal-compute claim.
- **C:** leaves evaluation and failure reserve within 70 CU. Cannot establish results at the larger body or paper
  scale. **†Costs project the repo's analytic FLOPs through measured throughput; fresh calibration is required.**

For A/C, save AR's data-exposure checkpoint too. An intermediate checkpoint on a longer cosine schedule is not an
independently optimized short-run baseline.

"25M" needs correction: the existing body is **25.2M**, and AR is approximately **29.4M including embeddings**. C's AR
is approximately **23.1M**. None of these totals includes the frozen **36.4M tokenizer**.

2. **Is criterion (b) measurable?**

Define G = P_N − P_A > 0. Half closure requires P_X ≤ (P_N + P_A)/2.

**The repo supplies no measured ≤25M Gen-PPL gap.** The paper's gap is **6.52**, requiring **3.26** improvement; that
is an external 88M result. R020's approximately **7.38-point oracle-reconstruction gap is also not an AR comparison**.

R010's **0.59 SD measures generation groups from one checkpoint**, not training seeds. If training SD were also 0.59,
the brief's gain > 2σ rule would require improvement > 1.18, hence **G > 2.36** for exactly half closure. Actual
training variance is unknown.

With that hypothetical SD and three independent fits per arm, the SE of (P_N + P_A)/2 − P_X is approximately
**0.42 PPL**. Require its confidence interval to support half closure; a point estimate of 50% is insufficient. More
generated samples reduce sampling noise, not training variance.

3. **Spurious-win mechanisms**

- **Tokenizer:** changing tokenizer capacity, training exposure, or reconstruction quality between NCM arms changes
  the experiment. A shared released tokenizer supports a conditional generator claim, not an entirely from-scratch
  system claim. Re-estimate level-0 frequencies from permitted training data rather than borrowing the released
  generator's prior.
- **Under-trained AR:** a weak AR raises the halfway target and makes success easier. Use the complete FLOP-matched
  endpoint, suitable learning-rate schedules, and checkpoint curves.
- **Compute accounting:** the repo's FLOP proxy omits tokenizer/hierarchy work and undercounts dense masked attention.
  Equal proxy FLOPs require qualification.
- **Scorer choice:** preregister GPT-2-large as primary; report TinyStories-33M regardless of direction.
  Reconstruction scoring well does not establish attainable AR quality.
- **Entropy mismatch:** lower temperature, repetition, or shorter outputs can reduce Gen-PPL. Compare at matched
  entropy, with fixed EOS handling, repetition, length and MAUVE.

4. **Recommendation and kill condition**

Choose **C**, after calibration. Train all three baseline pairs first: approximately **35.8 CU†**. Then fund one
candidate, three seeds; allocate approximately **10 CU** beyond training for calibration, evaluation and failures.
Give expensive objectives fewer updates to match compute—e.g. nominally **4.5k one-pass on-policy updates** versus 6k
baseline, subject to full-cost accounting.

**Kill this criterion-(b) attempt before candidate training if** the baseline gap is not reliably positive, half the
gap fails the brief's two-pooled-training-SD threshold, or AR improves by **≥0.35 Gen-PPL during its final quarter of
training**. The last condition means the budget has not established a stable comparator. At 128 output tokens,
success also requires **≤42 NFE**, with decoder work disclosed.

Reviewed cached branch snapshot **`b073e9a`**; live GitHub verification failed. **70–90 CU remaining and the current
billing rate were not independently verified.** Table costs exclude evaluation; smaller-body and candidate costs
remain projections.
