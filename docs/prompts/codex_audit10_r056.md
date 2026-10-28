<!-- Paste everything below this comment into Codex (desktop app, or `scripts/review/codex_local.sh audit audit10
docs/prompts/codex_audit10_r056.md` from a checkout of the audited commit). Work in a new branch `codex/audit10` made
from the audited commit, and commit only the audit doc and its tests there. -->
Deep audit. Follow '## Deep audit protocol' and '## Code Review Rules' in AGENTS.md. Audited commit: __SHA__
(head of `claude/wip-r056-package`, PR __PR__). Audit id: audit10. This repository is PUBLIC. Write no absolute local
paths, usernames, emails, session links or credentials into any file or into your final message; refer to files by
repo-relative path. Work on a new branch `codex/audit10` made from the audited commit; commit there.
If your sandbox cannot run Python, say so in the audit doc and still write the tests; they will be run for you.

Deep audit 10 (a delta audit) before the notebooks `w2mg_eval`, `w2p_L_ts_m3_c0`, `e1b_continue`, `w2c1_L_ts_m2`,
`w2c1_L_ts_m3` and `l1r_L_ts_m2r` (registered "W2C", "M3-0", "W2M-G", "E1b" and "L1R" in `docs/preregistration.md`,
the last five sections). Push only `docs/reviews/codex_audit10.md` and `tests/test_audit10_*.py`; change no other file.

## Why
W2C is the route to a criterion-(c) claim from a training arm: M2 / M3 at fresh seeds with W2M's protocol unchanged.
W2M-G locates the M2 damage (crossed replay) and gates the L1R kill. E1b continues E1's Monte Carlo on censored items
under an anytime-valid confidence sequence, the cheapest route to a (c) claim. L1R changes the trainer (a new arm
that must equal M2 in every input and differ only in some L1 targets). Audits 8 and 9 covered W2M's trainer and E1;
this audit covers what changed since: everything in `git diff b1b5fc5..__SHA__`.

## Subject
Files: `src/ncmr/w2m.py` (arm m2r, refactored frozen-wrong draws), `src/ncmr/l1r.py` (new: replay, circulation LP,
assignment), `scripts/train_gen.py` (m2r replay, root check, target reassignment, plan sha in the checkpoint),
`configs/scale/L_ts_m2r_frozen_pw_l1reassign.yaml`, `configs/scale/L_ts_m3_uniform_wrong_c0.yaml`,
`scripts/l1r_coverage.py`, `src/ncmr/w2mg.py` and `scripts/w2mg_eval.py`, `scripts/w2mg_gate.py` (new),
`scripts/e1b_continue.py` and `scripts/analyze_e1b.py` (new), `scripts/analyze_w2c.py` (new), `scripts/w1_sample_stats.py`,
`notebooks/make_notebooks.py` and the generated notebooks, their tests, and the PR's triage file. Reference code:
`scripts/e1_equivalence.py` and `scripts/analyze_e1.py` (audited in 9; unchanged), `src/ncmr/cached_decode.py`,
`scripts/mech_eval_w2m.py`. Numbers to recompute: none are logged yet; recompute the registration's item counts for
E1b (58 J0, 14 A0) from `results/colab/e1_equivalence/*.json` and the W2M seed-0 numbers the registrations quote
(M2 42.62, M3 21.69, M2 - M3 20.94 ± 0.37, 4.04 nats clean NLL; entropy, MAUVE, duplicates) from
`results/colab/w2m_L_ts_m*/` and `results/w2m_sample_stats.json`.

## Spec (what correct means)
The five registrations are the spec. What they leave implicit:
- **W2C / M3-0.** No trainer change for M0-M7: the existing arms' draws, orders and code paths are bit-for-bit as at
  `b1b5fc5` (W2M ran there, audits 8). M3-0 differs from M3 only in `clean_row_frac: 0.0`, and its masks are a superset
  of M3's at seed 0 (same uniforms). Seeds 1-3 change data order, masks and replacement draws through `--seed` only;
  the final evaluation's sample seeds are fixed.
- **L1R.** The replay reproduces the training run's roots exactly: same data stream (a fresh `default_rng(seed)`),
  same mask stream, same replacement generator consumption (torch.multinomial with num_samples = 1 consumes its
  generator by shape only; the run checks the roots at every micro-batch), same micro-batch partition; the replay draws
  nothing from the global RNG (dropout) or the training data_rng. The circulation is maximal and balanced; selection
  uses only roots and masks; receivers get donor blocks whose canonical root equals their supplied root; the L1 target
  multiset over the run is unchanged; inputs, masks, replacement codes and every other target equal M2's. Resume
  recomputes the plan and fails closed on a different sha. Coverage < 0.80 stops before training.
- **W2M-G.** fp32, TF32 off, eval mode everywhere. The sampler draws each level from the model's level-j call, equal
  to teacher-forced logits; the noise is shared across cells (common random numbers), so pure cells are identical at
  every cut. Gen-PPL and log loss per group as `src/ncmr/eval_text.py` computes them; shares and the gate as
  registered; the flag sweep adds f * w to every level-1..11 input; banks and seeds as registered (clean bank =
  even-index val windows only; nothing reads test).
- **E1b.** The replay reproduces `mc_hits` exactly for the same seed, round size and device; the continuation extends
  the same generator; the stopping rule is the registered width rule or 32768 samples; the reading substitutes only the
  continued items' cut-8 counts and uses E1a's deltas and rules unchanged; replay mismatches are read on E1b's own
  counts and flagged.
- **Notebooks.** The new entries run exactly the registered commands; the L1R notebook runs the gate first and refuses
  any GPU but an A100 with > 39 GB; every notebook ends with a cell that releases the runtime after the results are
  copied; provenance and the refuse-to-resume-under-changed-code checks are unchanged.

## Questions
1. Can the L1R replay ever differ from training (a draw between replay and training that consumes a generator, the
   `torch.chunk` partition at the last micro-batch, the tokenizer encode in a different precision or mode, the device
   of the generators, resume from a mid-run checkpoint)? Does the root check catch every difference before a
   gradient step?
2. Is the circulation truly maximal and the assignment free of any dependence on target content? Can a receiver get
   its own block, or a block whose root differs from its supplied root? Is the multiset preserved exactly, including
   rows split across micro-batches and steps?
3. Does any change in `src/ncmr/w2m.py` or `scripts/train_gen.py` alter M0-M7 (the frozen-wrong refactor, the new
   checkpoint extra, `_run_key`, the manifest)? Would a W2C seed resumed under this commit equal an uninterrupted run?
4. W2M-G: are crossed cells really "prefix model through level c-1, suffix model from c", with the same noise as the
   pure cells? Is the JS / root-reliance / KL arithmetic as registered (levels, positions, pooling, nats)? Are the
   bootstrap and jackknife what the registration says, and do the readings use the stated thresholds and scorers?
5. E1b: does the selection reproduce E1a's intervals (delta, exp(-N) intersection, width in nats)? Is the
   continuation's CS evaluated at the same delta as the reading? Does the cumulative time cap or a resume change any
   item's counts?
6. analyze_w2c: are D_s paired over the 16 groups, s_p the pooled seed SD as registered, and the readings' boundary
   cases right?

## Out of scope
Style, naming, performance tuning, anything not run by the listed notebooks.
