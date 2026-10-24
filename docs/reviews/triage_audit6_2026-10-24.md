# Triage of Codex deep audit 6 (J2-F frozen-teacher arm before its launch)

Source: `docs/reviews/codex_audit6.md`, `tests/test_audit6_*.py` (local Codex CLI on PR 10 head `dd64aa8`; branch
`codex/audit6`). The audit's suite was re-run first on the audited code: **3 passed, 7 strict xfails**, as reported.
Both findings reproduce. Fixes are on PR 10 (`claude/wip-j2f`).

| # | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| A6-1 | P1: the WMT14 run manifest is checked only when `ckpt.pt` exists, so a run directory interrupted before its first checkpoint accepts another arm, recipe or base generator and overwrites the manifest. | **accept** | `test_recorded_run_without_checkpoint_refuses_changed_identity` (6 cases) xfails on `dd64aa8`, passes with the fix. | The audit's patch: check whenever a checkpoint or a manifest exists. Same pattern exists in `scripts/train_gen.py` and `scripts/train_sudoku_L.py`: to fix before their next runs (not J2-F's path). |
| A6-2 | P1: `scripts/analyze_j1.py` cannot select J2-F while keeping the J0-J2 readings. | **accept** | `test_registered_frozen_analysis_keeps_the_original_three_arms` xfails on `dd64aa8`. | The audit's patch (`--candidate J2-F`; J2's shared-output mask kept for J0-J2, J2-F's own mask for its rest). Default output re-checked identical to `results/j1_analysis_s0.json`. Added `test_candidate_j2f_keeps_j2_mask_for_original_arms` (kills a mutant the audit's test missed). |
| gate 1 | Registered gate (1) is tested only on a tiny model, not on the released WMT14 weights. | **accept** | Answer 4 of the audit. | Launch-time check `--teacher_check`: before training, the frozen teacher must reproduce J0's first 128 saved exact hypotheses and pass counts (`{R}/L_wmt_released/hyp_val.json`), else exit; notebook passes it. Tests for accept / refuse (weights, pass counts, missing file); mutation-checked. |
| memory | ~1.13 GB extra resident (teacher generator plus its deep-copied tokenizer); no OOM shown. | **accept** (note) | Answer 3. | Reported; A100-40GB as registered. |
| R036 wording | "exact excludes them": aggregates are all-sentence; only the exactness claim excludes the 3 disagreeing sentences. | **accept** | Answer 6. | Wording correction for the log. |
| R036 numbers | All registered and exploratory numbers recomputed; no disagreement. | **accept** | Answer 6 tables. | None. |

Ported tests: the two strict-xfail markers removed, nothing else changed; they fail (7) on the audited scripts and pass
on the fix. Full suite 160 passed on torch 2.6 and 2.11.
