# Triage of Codex deep audit 5 (the W1d diagnostic, before its launch)

Source: `docs/reviews/codex_audit5.md`, `tests/test_audit5_*.py` (local Codex CLI on `4df7d13`, branch `codex/audit5`).
The audit could not execute Python in its sandbox, so its tests were run here first, on the audited code: **13 passed,
3 strict xfails**, each failing on its intended assertion. Verdicts: **accept**, **partial**, **reject**.

| # | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| verdict | No P0/P1 defect in the registered notebook's default path; the frontier, oracle kernel, precision, checkpoint loading and paired statistics implement the spec. All 20 generated notebooks keep PR 3's guards with the template records. | **accept** | Consistent with the PR 7 tests (sampler equal to `train_gen.py`'s, oracle canonical-level and pairing tests, registration consistency). | None. |
| A5-1 | P2: the reproduction helper truncates with `zip`, so a missing current group, a requested seed hidden by truncation or a duplicated seed can still report `ok`. | **accept** | The three strict xfails fail on `4df7d13` (`reproduction([10.0], [0, 999], ...)` returned ok). Not reachable with the notebook's eight distinct seeds. | Codex's patch: `ok` requires one score per distinct requested seed (`valid_groups`) and computes missing seeds from the requested set. Markers removed; 101 tests pass on torch 2.11 and 2.6. |
| numbers | Historical arithmetic agrees except R020 slips already corrected in R023. | **accept** | As R023 / R024 recorded. | None. |
| limits | No A100 replay, no load of the wave-1 `final.pt` files, no scorer rerun. | **accept** | True: those exist only on Drive / Colab. | The notebook's T = 1 reproduction gate (fail-closed, PR 7) is the run-time check that the right checkpoints and sampler were used. |
