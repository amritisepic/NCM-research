Deep audit 5 before `w1d_requant_diag` (registered W1d in `docs/preregistration.md`; eval-only, exploratory). Follow
"Deep audit protocol" and "Code Review Rules" in AGENTS.md. Push only `docs/reviews/codex_audit5.md` and
`tests/test_audit5_*.py`; change no other file.

## Why
W1d decides whether a ~20 CU confirmation of the re-quantized-label arm (A2) is registered. It adds new sampler and
evaluation code (`scripts/diag_w1_frontier.py`) and an fp32 path in the shared MAUVE featurizer, so CLAUDE.md's Tier-2
rule applies before launch. Tier-1 review (PR 7, two rounds) is triaged in `docs/reviews/triage_pr7.md`.

## Subject
Head of PR 7. Files: `scripts/diag_w1_frontier.py`, `src/ncmr/eval_text.py` (`mauve_features`, `evaluate_samples`),
`notebooks/make_notebooks.py` (the `w1d_requant_diag` entry and the provenance-record change), `tests/test_w1d_diag.py`,
`tests/test_notebook_provenance.py`. Reference behaviour: `scripts/train_gen.py` (`draw_samples`, `sample_metrics`) and
`scripts/tf_levels_diag.py` (R020's oracle curve). Wave-1 result files under `results/colab/w1_*/`.

## Spec (what correct means)
- Frontier: exactly W1b's sampler (`train_gen.py`'s `draw_samples`), groups of 256 with seeds 42, 0-6, at T = 0.8 / 0.9 /
  1.0; samples re-encoded as in `train_gen.py`; Gen-PPL with GPT-2 large and TinyStories-33M; MAUVE featurized in fp32
  against the first 2048 val windows (even-index split), 5 quantization seeds. fp32 and TF32 off throughout.
- Reproduction check: T = 1 runs first; every requested seed must have a wave-1 group and match it within 0.01 Gen-PPL,
  with finite values; otherwise the command exits non-zero before anything else runs.
- Oracle curve: R020's protocol (canonical levels 0..j of the first 1024 val windows, Gumbel-max sampling of levels j+1..
  at T = 1, one seed per j shared by every model) at j = -1 / 3 / 5 / 7, scored in groups of 256.
- A1 / A2 load the wave-1 `final.pt` strictly, with its level-0 unigram; A0 is the released model.
- Pairing: the same seeds and windows for every model. No absolute path in any output (`result.json`,
  `samples_final.json`, `provenance.jsonl`, `_meta.json`).

## Questions
1. Can a wrong checkpoint, a different sampler, a reordered seed list or another GPU type pass the reproduction check?
   Is there any path where T = 1 reproduces but T = 0.8 / 0.9 sample differently from `draw_samples` at those
   temperatures?
2. Is every forward on the path (generator, tokenizer decoder, both Gen-PPL scorers, MAUVE featurizer for samples and
   references) fp32 with TF32 off? Any autocast or cached bf16 reference left (the reference cache name)?
3. Does the oracle sampler equal `tf_levels_diag.py`'s `sample_from` (levels, offsets, Gumbel draw order, per-j seed)?
   List every protocol difference from R020 that makes A0's new curve not directly comparable with R020's.
4. Does loading `final.pt` reproduce the evaluated wave-1 model exactly (buffers, dropout settings, level-0 unigram)?
5. Does the provenance change (command templates instead of expanded commands) keep every protection of PR 3
   (refusal of mixed code, unprovenanced checkpoints) and the collection step working for all notebooks?
6. Is the reading rule in W1d implementable from the saved outputs (paired differences per group, SE per temperature and
   per j), and is anything it needs missing from `result.json`?

## Out of scope
Style, naming, performance tuning, the trainers.
