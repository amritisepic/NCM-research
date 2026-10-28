# Triage of Codex deep audit 13 (delta: PR 18, W2M-GR eval code)

Source: `docs/reviews/codex_audit13.md`, `tests/test_audit13_{analysis,numbers,boundaries}.py` (local Codex CLI,
gpt-6.1-sol / ultra, on `a949044`; branch `codex/audit13` at `bd0e0ae`; brief `docs/prompts/codex_audit13_w2mgr.md`).
The first attempt stopped at the Codex usage limit with no output; the retry after the reset completed. Python did not
run in the audit's sandbox. Audit suite re-run here on the audited commit first: 31 passed, 4 xfailed (strict), i.e.
all four A13-1 probes fail on `a949044` as they should. Ported to PR 18 (`b550b3b`) with the xfail marker removed: they
pass after the fix.

| # | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| A13-1 | P1: `scripts/analyze_w2mgr.py` `check_result` checks counts, seeds, cell names and the advertised `n_groups` / `n`, but not the protocol block's `clean_bank`, each cell's saved `spec`, or the actual group vectors. A result declaring a test-split clean bank, a crossed cell at the wrong cut, replay seeds 8-15, or four score groups with consistent statistics is accepted and read. | **accept** | The four strict xfails fail on `a949044` (reproduced). The evaluator itself does not write such results (the audit says so too), but the analyzer's stated job is to withhold a reading from unregistered runs, as in A11-1. | PR 18 `b550b3b`: `REGISTERED` now also pins `clean_bank`, `cells`, `sampler` and `ci`; `expected_cell_spec` compares every cell's spec; `check_cell_scores` requires eight finite paired group Gen-PPLs and log losses per scorer, log loss = log Gen-PPL, and the stored mean / SD / SE equal to theirs; `summarise_seed` classifies on share points recomputed from the cells (admission's 1e-9 tolerance is no longer a second decision boundary; the stored jackknife SEs are kept). New tests: the admission fields equal what `scripts/w2mg_eval.py`'s `protocol_block` and `cell_specs` write (so real results are accepted), ten declared departures, two layers no other check covers (a constant log-loss shift; seven consistent groups), and recomputed shares. Eight mutations of the fix, each killed. |
| Q2 note | The default W2M-G path now records `arms`, so an unfinished pre-PR-18 W2M-G run directory would be refused (exact manifest comparison). | **accept** (note) | No such directory is in use: W2M-G finished (R058). | None. |
| Q4 note | `EPS = 1e-12` widens inclusive bounds and narrows the strict `< 0.05` overturn bound by 1e-12. | **accept** (note) | Far below any resolvable difference between estimated shares; the audit's boundary tests characterise it. | None. |
| numbers | W2C's six sanity references (G(M2_s), G(M3_s)) and R062's paired contrasts recompute exactly. | **accept** | Agrees with R062 and audit 12. | None. |

No P0. Everything else in the delta (reduced path identical to the full one on the retained computations, sanity
reference, notebook wiring and L4 guard, precision) was found correct by source review.
