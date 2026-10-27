# Triage of Codex deep audit 9 (D1, E1, W2Ma before launch)

Source: `docs/reviews/codex_audit9.md`, `tests/test_audit9_{cached,numbers,protocol}.py` (Codex desktop on `dcdb890`,
branch `codex/audit9` at `bb9e60c`; brief `docs/prompts/codex_audit9_d1e1.md`). Audit suite re-run here first on the
audited code (torch 2.6 CPU): 19 passed, no xfails, as reported. The audit reports no P0 / P1 defect, so there are no
strict xfails to see fail on the audited commit.

| # | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| Q1-Q6, Q8 | Not findings: cached decoders equal upstream (full-size random-weight WMT14 config: equal codes, logits within 1.1e-6; released Sudoku hash-pinned); D1 timed region, interleaving, gate selection and integrity blocking; FLOP / decode counts; E1 ancestral sampler (independent full-recompute oracle agrees exactly); W2Ma; notebook wiring. | **accept** (note) | Audit answers 1-6, 8; the ported tests pass here. | Tests ported unchanged. Limits (from the audit): CPU only; released WMT14 / TinyStories weights, GPU kernels and timing, torch 2.11 not run. D1's own integrity checks on the A100 cover the released WMT14 weights. |
| Q7-a | R043: L0 replaced in 75.65%, written 75.6% (and the level-1 flag share inherits it). | **accept** | `results/w2m_flag_power.json`; already noted in R045 and `scripts/w2m_flag_power.py`. | None further. |
| Q7-b | R045: level 4 gain per code is 0.0155, written 0.016. | **accept** | `results/s0_neighbour_gain.json` `levels.4.gain_per_code` = 0.015451. | Corrected in R048 (log is append-only). Verdict unchanged. |
| Q7-c | R045: "only level 1 gains > 1 nat per window" is false; level 2 gains 1.54. | **accept** | `levels.2.gain_mean` = 1.5419. | Corrected in R048. Gate verdict (fail on both parts) unchanged. |

Port notes: the three audit test files are ported unchanged and pass on the PR head (19 passed, 26 s).
