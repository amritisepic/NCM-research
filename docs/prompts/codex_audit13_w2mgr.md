<!-- Paste everything below this comment into Codex (desktop app, or `scripts/review/codex_local.sh audit audit13 <this
brief>` from a checkout of the audited commit, the head of PR 18 / branch `claude/wip-w2mgr`). Work in a new branch
`codex/audit13` made from the audited commit, and commit only the audit doc and its tests there. -->
Deep audit 13 (delta) before launching the `w2mgr_eval` notebook (registered "W2M-GR" in `docs/preregistration.md`, the
last section). Follow "Deep audit protocol" and "Code Review Rules" in AGENTS.md. Push only
`docs/reviews/codex_audit13.md` and `tests/test_audit13_*.py`; change no other file. If your sandbox cannot run the
repository's Python, say so in the audit doc and still write the tests; they will be run for you.

## Why
W2M-GR replicates W2M-G's localization (R058: the M2 - M3 generation damage forms at levels 2-5 and is amplified by M2's
own later levels, not at L1) on W2C's seed 1-3 checkpoints, with thresholds registered before any measurement. Its
reading, together with W2C's (R062, audited in 12), would be a criterion-(c) candidate. The W2M-G code was audited in
audit 10 (source review; no P0 / P1). PR 18 changes that eval code: a reduced arm set, a new sanity reference, a new
analysis script and an L4-only notebook. Eval code changed since the last deep audit, so CLAUDE.md requires this audit
before launch.

## Subject
Delta: `git diff 733ea77..HEAD` (PR 18). Files: `scripts/w2mg_eval.py` (`--arms`, `parse_arms`, `banks_for`,
`rho_models`, `w2c_reference`, `run`, `check_manifest`, `protocol_block`, `analysis_ii`, `analysis_iii`),
`src/ncmr/w2mg.py` (`sanity_ii`), `scripts/analyze_w2mgr.py` (new), `notebooks/make_notebooks.py` (`RUNS["w2mgr_eval"]`,
`GUARD_GPU`) and the generated `notebooks/w2mgr_eval.ipynb`, `tests/test_w2mgr.py`, `tests/test_r056_build.py`.
Unchanged code the delta relies on: the rest of `scripts/w2mg_eval.py` and `src/ncmr/w2mg.py` (sampler, noise, cells,
banks, JS, bootstrap, root reliance, shares, jackknife; audited in 10), `scripts/train_gen.py`'s `final.pt` format.
Existing tests: `tests/test_w2mg.py`, `tests/test_audit10_*.py`. Inputs the run will read: W2C's final checkpoints on the
user's Drive (not in the repo) and `results/colab/w2c1_L_ts_m{2,3}/*__result.json` (in the repo). No log entry's
numbers depend on this code yet: there is nothing to recompute except the sanity references (W2C's G values, R062).

## Spec (what correct means)
The W2M-GR registration is the spec; W2M-G's protocol applies unchanged to (i), (ii), (iii). What it leaves implicit:
- With `--arms m2,m3` the run must compute (i), (ii) and (iii) exactly as the W2M-G path would on the same models: the
  same cells (pure and crossed at cuts 2, 4, 6; 8 groups of 256, seeds 0-7), the same banks (released, M2, M3; seeds 900,
  901), the same clean bank, roots (seed 2027), bootstrap (seed 0), scorers, fp32 with TF32 off, eval mode. Only (iv), the
  M4 cells, the `m4f0` bank, the corrupted bank and M0 / M4 in (iii) are dropped.
- The default arm set must leave W2M-G's computation unchanged.
- Sanity for seed s >= 1: G(M2_s, M2_s) and G(M3_s, M3_s) under GPT-2 large within 3 SE of W2C's final-eval G(M2_s),
  G(M3_s); a miss is reported, not fatal.
- `scripts/analyze_w2mgr.py`: localization replicates iff in every seed 1-3, under GPT-2 large log loss, pi_2 <= 0.10,
  pi_6 >= 0.30, sigma_6 <= 0.20, interaction (1 - pi - sigma) >= 0.20 at both cuts 4 and 6, and the same four conditions
  hold on the TinyStories-33M seed means; overturned iff on the GPT-2 large seed means pi_2 >= 0.25, or sigma_6 >= 0.50, or
  the interaction < 0.05 at both cuts 4 and 6; else mixed. (i) replicates iff W2M-G's (i) rule holds in every seed; (iii)
  replicates iff the paired 95% CI of rho(M3_s) - rho(M2_s) lies above 0 in every seed. Seed 0 never pooled.
- A run directory refuses another protocol, seed, arm set, checkpoint, asset, val data, device or runtime. Every seed runs
  on an L4.

## Questions
1. Does the reduced path compute (i)-(iii) bit-for-bit as the full path on the same models (noise streams, cell order,
   bank order, bootstrap draws, (iii)'s shared root draws when fewer models are passed to `root_reliance`)? Can dropping
   M0 / M4 change any RNG consumption or any batch composition that (i)-(iii) depend on?
2. Is the default W2M-G path unchanged apart from the recorded `arms` key? Would a resumed seed-0 W2M-G run directory
   now be refused (manifest gained `arms`), and does that matter?
3. `w2c_reference`: right files, right key, right seed; can a seed-1 run silently use seed 0's or W2M's reference? Is the
   reference read before anything is written?
4. `analyze_w2mgr.py`: each condition exactly as registered (inclusive / strict, every seed vs seed mean, which scorer,
   both cuts vs either); the EPS = 1e-12 tolerance in comparisons (does it move any registered boundary in a way that
   matters?); validation of each per-seed file (could a seed-0, full-arm, incomplete, non-L4 or tampered result be
   accepted?); shares recomputed from the cells. Write boundary tests for anything `tests/test_w2mgr.py` misses.
5. The notebook: the command, seeds, run-dir names that the analyzer expects after the collect cell
   (`results/colab/w2mgr_eval/w2mgr_eval_s<s>__result.json`), the GPU guard (whole-word "L4"), provenance and resume
   behaviour inherited from the template, and the release cell. Is anything needed by a registered reading not saved?
6. Precision and evaluation protocol (AGENTS.md): any autocast or TF32 path the reduced run enters differently?

## Out of scope
Style, naming, performance tuning; W2M-G's (iv); the W2C training runs (audit 12); anything audit 10 covered that the delta
does not touch.
