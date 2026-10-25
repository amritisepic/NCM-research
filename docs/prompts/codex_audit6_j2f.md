Deep audit 6 before `w2c_L_wmt_consist_frozen` (registered J2-F in `docs/preregistration.md`, with J1 / J1a / J1b / J1c).
Follow "Deep audit protocol" and "Code Review Rules" in AGENTS.md. Push only `docs/reviews/codex_audit6.md` and
`tests/test_audit6_*.py`; change no other file.

## Why
J2 (moving teacher) missed its screen with 16% of test sentences decoded to two input-independent strings (R036).
J2-F tests whether a frozen teacher removes that failure; it is the only remaining text candidate for criterion (d).
It adds a trainer arm, so CLAUDE.md's Tier-2 rule applies before launch. Tier-1 review: PR 10.

## Subject
Head of PR 10 (branch `claude/wip-j2f`). Files: `scripts/train_wmt14_L.py` (arm `consist_frozen`: the `teacher` block,
`consist_loss(..., teacher_net)`, the training loop call), `configs/scale/L_wmt_consist_frozen.yaml`,
`notebooks/make_notebooks.py` and `notebooks/w2c_L_wmt_consist_frozen.ipynb`, `tests/test_j2f_frozen_teacher.py`,
`tests/test_audit4_protocol.py`. Reference: J2's arm in the same file (audited in audit 4) and its seed-0 results
`results/colab/w2c_L_wmt_consist/`, analysed by `scripts/analyze_j1.py` (R036). Log entry to recompute: R036.

## Spec (what correct means)
- J2-F equals J2 in everything except the teacher: same common batch of 128 per step (rows 1-64 teacher forcing, every
  4th of rows 65-128 as trajectory sources), 4 states per trajectory drawn uniformly strictly before the fixed point,
  one student pass per state, zero-loss padding rows for degenerate sources (J1c), 2000 steps, same recipe and seed
  handling, generator FLOP proxy 2000 x 576 forward rows.
- Teacher: a copy of the released generator, made before any checkpoint is restored into the student; eval mode, no
  grad, never updated, not checkpointed (rebuilt from the base files, which the manifest pins by SHA-256); trajectories
  in fp32 with TF32 off, `full_budget` (K + 1 = 12 passes), `initial_fixed_ok`. Its fixed points are therefore the
  released model's exact-decode outputs, which J0's evaluation decodes.
- The student (`net`) is in train mode for its consistency pass, as in J2. Evaluation is J2's (`evaluate`, fp32).
- The run manifest records the arm; a run directory refuses J2's config and vice versa; a resumed run equals an
  uninterrupted one (data RNG, state draws, dropout RNG, optimizer, run-level degenerate totals).

## Questions
1. Can the teacher ever differ from the released generator (copied after `maybe_resume`, shared parameters or buffers
   with the student, a mode or dropout flip, a cached module such as the tokenizer that training mutates, device or
   dtype differences)? On resume, is it rebuilt identically?
2. Does anything the frozen teacher changes alter the RNG streams relative to J2 (dropout draws, the state draws), so
   that seed-0 J2-F does not see J2's data and draws where the trajectories coincide?
3. Is the executed FLOP proxy and the logged `traj_*` / `degenerate_pct` accounting the same as J2's? Note any memory
   or wall-clock risk on an A100-40GB from the second 120M fp32 copy.
4. Do the new tests exercise production code (the trainer's own blocks and functions), and would each fail if its
   property broke? Name any registered J2-F code-gate property left untested.
5. Is everything J2-F's registered readings need (screen, H12, J1a comparators, the shared-output diagnostic in
   `scripts/analyze_j1.py`) saved by this arm, and can `scripts/analyze_j1.py` be pointed at J2-F's outputs without
   changing what it computes for J0-J2?
6. Recompute R036's registered numbers from `results/colab/w2c_L_wmt_*/` and `results/j1_analysis_s0.json`; list any
   disagreement.

## Out of scope
Style, naming, performance tuning, arms and notebooks other than J2-F.
