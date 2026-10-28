<!-- Paste everything below this comment into Codex (desktop app, or `scripts/review/codex_local.sh audit audit12
docs/prompts/codex_audit12_w2cclaim.md` from a checkout of the audited commit). Work in a new branch `codex/audit12` made
from the audited commit, and commit only the audit doc and its tests there. -->
Deep audit 12 before any criterion-(c) claim rests on W2C stage 1 (registered "W2C" in `docs/preregistration.md`).
Follow "Deep audit protocol" and "Code Review Rules" in AGENTS.md. Push only `docs/reviews/codex_audit12.md` and
`tests/test_audit12_*.py`; change no other file. If your sandbox cannot run the repository's Python, say so in the audit
doc and still write the tests; they will be run for you. The recomputation needs only `json` and `statistics`.

## Why
R062 reports W2C stage 1's registered readings as "damage replicates" and "dissociation holds": M2 (frozen-model
plausible-wrong replacement) minus M3 (uniform-wrong replacement in the same slots) is +18.36 / +19.51 / +18.76
Gen-PPL (GPT-2 large) at training seeds 1-3, while clean teacher-forced NLL differs by +4.03 nats/window on the seed
mean. That is believed to meet criterion (c) of `docs/brief.md`. The trainer delta (M0-M7) and
`scripts/analyze_w2c.py` were source-reviewed in audit 10 (no P0 / P1; Python did not run there). This audit's job is
(1) an executed recomputation, (2) the matched-comparison and evaluation-integrity checks the claim depends on, and
(3) whether the proposed claim says more than the evidence.

## Subject
Files: `scripts/analyze_w2c.py` (W2C part), `scripts/w1_sample_stats.py` (`--set w2c1`: adjacent duplicates per 1k),
`scripts/train_gen.py` (the `mech` corruption path, the final evaluation and what it writes to `result.json`),
`src/ncmr/w2m.py` (M2 / M3 replacement, the mask stream, clean rows, `realized`), `src/ncmr/eval_text.py` (Gen-PPL per
group, MAUVE, entropy, duplicates), `src/ncmr/data.py` (val split). Configs: `configs/scale/L_ts_m2_frozen_pw.yaml`,
`configs/scale/L_ts_m3_uniform_wrong.yaml`. Notebooks: `notebooks/w2c1_L_ts_m2.ipynb`, `notebooks/w2c1_L_ts_m3.ipynb`
(or however the repository names them; the run commands are in each run's `*__provenance.jsonl`). Existing tests:
`tests/test_r056_build.py`, `tests/test_w2m*.py`, `tests/test_audit10_*.py`.
Inputs: `results/colab/w2c1_L_ts_m2/` and `results/colab/w2c1_L_ts_m3/` (`*__result.json`, `*__config.yaml`,
`*__run_manifest.json`, `*__provenance.jsonl`, `*__samples_final.json`, `*__metrics.jsonl`, `_meta.json`); seed 0
for context only: `results/colab/w2m_L_ts_m2/`, `results/colab/w2m_L_ts_m3/`. Outputs: `results/w2c_analysis.json`,
`results/w2c1_sample_stats.json`. Log entry whose numbers to recompute: R062 (every number), and the seed-0 numbers it
quotes from R049 / R055.

## Spec (what correct means)
The W2C registration is the spec; W2M's protocol (registered "W2M", "W2Ma") applies unchanged. What it leaves implicit:
- **Per seed s.** G(arm) is the final-eval Gen-PPL under GPT-2 large over 16 sample groups (sample seeds 42 and 0-14,
  256 samples each, fp32); D_s = G(M2_s) - G(M3_s), paired over the 16 groups (SE = SD of the 16 group differences /
  4). The same under TinyStories-33M. dNLL_s = clean val NLL (nats per window, summed over levels, val split = even-index
  validation items, identical items for both arms). dDup_s = adjacent duplicates per 1k from
  `scripts/w1_sample_stats.py --set w2c1` (registered source), M2_s - M3_s. MAUVE in fp32.
- **Pooled seed SD.** s_p = sqrt((var_s G(M2_s) + var_s G(M3_s)) / 2), sample variances (ddof 1) over s = 1..3.
- **Readings** (seeds 1-3 only; seed 0 never pooled): damage replicates iff every D_s >= 10, mean D > 2 s_p, every
  D_s(TinyStories-33M) > 0 and every dDup_s > 0; overturned iff mean D < 5 or the seed-mean M2 - M3 is negative under
  either scorer or MAUVE (MAUVE higher is better, so "M2 better" means mean dMAUVE > 0); else mixed. Dissociation holds
  iff |mean dNLL| <= 10; removed iff mean dNLL > 20; else not established.
- **Matched comparison.** Within a seed, M2 and M3 must see the same data in the same order, the same clean rows and
  the same masked slots (one mask stream); the arms differ only in the replacement code. M2's extra frozen-model
  proposal forward must not consume any RNG stream that M3 also uses. Evaluation cadence and the final evaluation
  must not differ by arm; sampling and scoring run in fp32 with TF32 off in both.

## Questions
1. **Recompute** every number in R062 from the raw `result.json` / sample files with your own code: G(M2_s), G(M3_s),
   D_s and paired SEs under both scorers, the "16 of 16 groups M2 > M3" statements, dNLL_s and its per-level
   split, dMAUVE, entropy differences, dDup_s (recompute duplicates per 1k from `*__samples_final.json` with your own
   tokenisation-consistent counter, and say whether it matches `w1_sample_stats`'s definition), s_p, 2 s_p, the
   readings, the A100-minutes from provenance. List every disagreement.
2. **The readings as coded.** Does `analyze_w2c.py` implement the spec above exactly: ddof, pairing by group index
   (are the 16 per-group lists in the same seed order in every file?), the TinyStories-33M and duplicate conditions,
   the MAUVE sign in "overturned", the absolute value in "holds", seeds 1-3 only? Write boundary tests for each
   threshold if `tests/test_r056_build.py` misses any.
3. **Matched comparison, executed where possible.** Do M2_s and M3_s draw identical batches, clean rows and masked
   slots? Evidence available: `realized_disagreement` in each `result.json` is identical between arms within a seed
   to 4 decimals at every level; the training `metrics.jsonl` loss curves; the code. Is equal `realized_disagreement`
   sufficient evidence of identical slots, or could it coincide with different slots? Can M2's proposal forward
   (dropout, sampling at `policy_temp`, any `torch` RNG call) shift the data, mask or clean-row streams relative to M3?
   A CPU test on a tiny model that runs both arms for a few steps and compares the batches, masks and clean-row
   indices would settle it.
4. **What M2 and M3 replace with.** At a masked slot, does M3 draw uniformly over codes != canonical, and does M2 draw
   from the frozen released model's distribution at T = 1 conditioned on what, excluding the canonical code? Is every
   masked slot wrong in both arms (disagreement on corrupted rows equals the mask rate)? Does M2's proposal see the
   corrupted or the canonical coarser levels? State precisely what "plausible-wrong" means as implemented; the claim
   must use that meaning.
5. **Evaluation integrity.** Is the final evaluation identical across arms and seeds: same 16 sample seeds, same
   number of samples, fp32 sampling, TF32 off, no autocast leak from training into evaluation (R024's defect class),
   GPT-2 large and TinyStories-33M scoring identical, MAUVE reference set and features identical, clean val NLL on the
   same items in fp32? Two notebooks ran at different docs-only SHAs (`6be0690`, `8bc61b8`): confirm that
   `src scripts configs assets notebooks` are identical to `5ad74b5` at both, and that both runs' manifests pin the same
   base checkpoint and data hashes.
6. **Alternative explanations the claim must survive.** For example: M2's damage is a generic effect of any
   non-uniform wrong distribution (would a frozen-model sample at a higher temperature, or a unigram-frequency wrong
   code, do the same?); the Gen-PPL gap is a length or entropy artifact (M2 entropy +0.027); GPT-2 large rewards
   something M3 does rather than penalising M2 (TinyStories-33M and MAUVE agree in direction); the NLL metric is
   insensitive by construction (does clean val NLL include the levels and positions where the damage forms, R058?).
   For each, say whether W2C's data can rule it out, and if not, what the claim must not say.
7. **The proposed claim.** Does anything in the code, the registration or the numbers contradict, or fail to
   support, this wording? "Under one continuation protocol for the released TinyStories NCM (600 steps, 25% clean
   rows, W1's per-level corruption rates, one mask stream), replacing masked codes with plausible-wrong codes sampled
   from the frozen released model damages generation, while uniform-wrong replacement in the same slots does not:
   M2 - M3 Gen-PPL (GPT-2 large) +18.9 on the mean of three training seeds (each seed >= +18.4; pooled seed SD 0.56;
   TinyStories-33M +29.8, MAUVE -0.41, adjacent duplicates +12.3 per 1k), while canonical teacher-forced NLL differs by
   only +4.0 nats per window (0.5%), the same sign in every seed." In particular: does "does not damage" overstate M3
   (both arms sit above the released model's 16.2 Gen-PPL because of the clean rows, R059)? Is this a finding "about
   what the levels encode" (criterion (c)) without the level localization, which is one seed (R058)? What may it not
   be read as (other domains, from-scratch training, other corruption rates, on-policy corruption)?

## Out of scope
Style, naming, performance tuning; M0, M1, M4-M7, W2M-G, M3-0 and L1R except where a question above needs them; the
resume path (every W2C run was one uninterrupted session, `had_ckpt: false`).
