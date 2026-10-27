<!-- Paste everything below this comment into Codex (desktop app, or `scripts/review/codex_local.sh audit audit9
docs/prompts/codex_audit9_d1e1.md` from a checkout of the audited commit). Work in a new branch `codex/audit9` made
from the audited commit, and commit only the audit doc and its tests there. -->
Deep audit. Follow '## Deep audit protocol' and '## Code Review Rules' in AGENTS.md. Audited commit: dcdb89092bccb967402461bda297337e99587400
(head of `claude/wip-e1-d1`, PR 14). Audit id: audit9. This repository is PUBLIC. Write no absolute local paths,
usernames, emails, session links or credentials into any file or into your final message; refer to files by
repo-relative path. Work on a new branch `codex/audit9` made from the audited commit; commit there and push it.
If your sandbox cannot run Python, say so in the audit doc and still write the tests; they will be run for you.

Deep audit 9 before the notebooks `d1_batch1_wallclock`, `e1_equivalence` and the fixed `w2m_mech_eval` (registered
"D1", "E1" and amendment "W2Ma" in `docs/preregistration.md`). Push only `docs/reviews/codex_audit9.md` and
`tests/test_audit9_*.py`; change no other file.

## Why
D1 decides whether criterion (d)'s text half stays alive (and whether ~23 CU of J1d seeds run): J2-F + a stable-text
stop must take <= 0.75x J0 sequential's batch-1 latency in fp32. A biased benchmark (a handicapped sequential
baseline, an untimed decode, a decoder that is not the evaluated one) would decide it wrongly. E1 is the primary
criterion-(c) candidate: whether most fine-level code NLL is representation choice. W2Ma replaces W2M's confounded
probe (R043). All three are new eval code; no training code changed.

## Subject
Head of `claude/wip-e1-d1`. Files: `src/ncmr/cached_decode.py` (new), `scripts/bench_batch1.py` (new),
`scripts/e1_equivalence.py` (new), `scripts/mech_eval_w2m.py` (W2Ma), `scripts/w2m_flag_power.py` (new),
`scripts/s0_neighbour_probe.py` (new; already run, R045), `notebooks/make_notebooks.py` and
`notebooks/{d1_batch1_wallclock,e1_equivalence,w2m_mech_eval}.ipynb`, tests `tests/test_d1_bench.py`,
`tests/test_e1_equivalence.py`, `tests/test_mech_eval_rowprobe.py`, `tests/test_w2m_flag_power.py`,
`tests/test_s0_probe.py`, and the PR 14 triage `docs/reviews/triage_pr14.md`. Upstream reference:
`src/ncmr/ncm_ref/models/next_concept.py` (`generate`, `_decode_level_logits`, `forward`, `_run_stack`),
`src/ncmr/jacobi_train.py`. Log entries to recompute: R045 (from `results/s0_neighbour_gain.json`), R043's numbers
(from `results/w2m_flag_power.json`), and R042's stable-rule numbers (from the saved `hyp_val.json` files and
`results/j2f_stop_rule_val_s0.json`; passes only unless you have the newstest2013 references).

## Spec (what correct means)
The registrations D1, J1d, E1, S0 and W2Ma in `docs/preregistration.md` are the spec. What they leave implicit:
- **Cached decoders compute the upstream function.** For the released WMT14 (predict_level0, padded 96-token source,
  rotary positions, QK-norm, untied heads), Sudoku-hard (query_level0, 91-token puzzle) and TinyStories (level 0 given,
  1-token prefix) generators: `seq_greedy(cached=True)` returns `generate(temperature=0)`'s codes and
  `cached_parallel_logits` returns `parallel_logits`'s logits, up to float rounding. The prefix attends only to itself
  and gets rotary positions 0..P-1 in every upstream call; padded prefix keys are masked for every query row (upstream's
  own `use_kv_cache=True` path drops this mask, which is why it is not used).
- **Timed region.** From the item's token prefix on the GPU to its output: WMT14 detokenised text, Sudoku board. For
  Jacobi + stable x1 the per-pass tokenizer decode and detokenisation are inside; the final output is not decoded a
  second time (`memo`). The prefix cache is built inside the timed region. CUDA synchronisation before and after each
  item; warm-up excluded; configurations interleaved per item in a seeded random order; item latency = median of 3.
- **Stop rules.** Exact: as `jacobi_decode` (WMT14 pass-1 detection on, J1c; Sudoku off); stable x1: the first pass
  t >= 2 whose detokenised text equals pass t-1's, else the exact stop, cap K + 1; verifier: as R020 /
  `train_sudoku_L.evaluate`. `stable_stop` on saved capped hypotheses must equal the online rule.
- **Integrity.** J0 exact = the saved test hypotheses and passes; stable-rule outputs = the rule on saved capped test
  hypotheses (J0, J2-F); cached = eager; at most 2 of 256 mismatches; the gate is not read otherwise.
- **FLOPs.** Executed FLOPs per item from FlopCounterMode (math attention) per call type times the calls made;
  generator blocks in flops.py's convention (24 d^2 + 4 d keys per row and layer, dense keys).
- **E1.** N_k / A_k from teacher-forced logits (fp32); the sampler draws level k from the model's distribution given
  c*_{<k} and each finer level given the sampled coarser ones (ancestral, T = 1); hits as registered per domain; Gumbel
  noise seeded per (item, cut), identical across models; cuts >= 8 on every item, 5-7 on the first 8; summaries and
  bounds (R_lo / R_hi, rule of three) as registered; WMT14 items are `prep_wmt14.py`'s val arrays (target given source).
- **W2Ma.** Row-level probe per level 1..6: 50% clean windows, the same clean rows and masked slots for plausible-wrong
  and uniform-wrong contexts (built once from the released generator), label = window corrupted, feature = mean level-k
  hidden state after ln_f, train windows 0-383 / AUC on 384-511; "too weak" iff uniform-wrong AUC < 0.75 at every
  level; the pooled probe kept under `probe_pooled`, marked not read; (a) and (c) unchanged.

## Questions
1. Do the cached decoders equal the upstream ones on the real released WMT14 config (not only the tiny test model):
   rotary tables for prefix and target rows, the pad bias on every query row, level-0 read-off with padding, the
   level-encoding offsets, and `_pad_vocab`? Any case where cached and eager differ beyond float rounding?
2. Is D1's comparison fair to both sides? Anything inside the timed region on one side but not the other (tokenizer
   decodes, detokenisation, host-device copies, prefix embedding, warm-up effects, the order of configurations,
   autocast weight-cast caching in the bf16 run)? Does the gate pick the faster implementation of each side as
   registered, and does an integrity failure block the gate?
3. Do the integrity checks compare like with like (batch-128 saved outputs vs batch-1 decoding; capped hypotheses vs
   per-pass texts; pass counts), and would they catch a wrong checkpoint for J2-F?
4. Is the FLOP accounting what the code executes (per-pass decodes for stop rules, the prefix pass for cached
   variants, level-0 calls), and are the generator-block numbers in flops.py's convention?
5. Does E1's sampler draw from the model's ancestral distribution (including the expanded prefix cache, the z_hat
   clone, the first level's shared logits) and are N_k, A_k, T_k, R_k, U_k and the paired dN / dT computed as
   registered? Is the hit test right per domain (EOS / pad handling for WMT14, board for Sudoku)? Does anything
   differ between models other than the weights (windows, seeds)?
6. Does `scripts/mech_eval_w2m.py` implement W2Ma exactly, with identical inputs for every model, in fp32 TF32 off,
   and are (a) and (c) unchanged?
7. Recompute R045's table and the gate verdict from `results/s0_neighbour_gain.json`, R043's numbers from
   `results/w2m_flag_power.json`, and R042's stable x1 / x2 passes for the four arms from the saved val hypotheses.
8. Do the notebooks run the registered commands (sizes, cuts, checkpoints on Drive) and save what the readings need?

## Out of scope
Style, naming, performance tuning, anything not run by the listed notebooks; the W2M training arms.
