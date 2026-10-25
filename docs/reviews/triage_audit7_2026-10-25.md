# Triage of Codex deep audit 7 (the J2-F delta after audit 6)

Source: `docs/reviews/codex_audit7.md`, `tests/test_audit7_gate.py` (local Codex CLI on `b4673ab`, branch `codex/audit7`;
brief `docs/prompts/codex_audit7_j2f_delta.md`). Scope: `dd64aa8..81a6979` (audit 6's fixes and the launch-time teacher
check merged in PR 10).

| # | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| none | No new P0/P1. The launch check draws no random numbers and changes no training state; `evaluate(model=None)` is unchanged for every existing call; the comparison uses J0's first batch of 128 in order and fails closed on a missing or differing file; A6-1 / A6-2 match audit 6's patches. | **accept** | Codex's sandbox could not run Python, so its probes were run here: `tests/test_audit7_gate.py` (5) with the WMT14 / J2-F suites, 72 passed on torch 2.6 and 2.11 CPU. | Tests ported unchanged (no xfails: no defect). |
| limits | The check is a finite behavioural test (128 sentences), not proof of model identity; a stale J0 file with identical content would pass; A100 kernel behaviour unverified. | **accept** (note) | Answers 2-3. | Identity on resume rests on the manifest's base hashes; J2-F runs on the A100 type J0 ran on. |
| arithmetic | R036 numbers re-checked from the sentence-ordered records: no disagreement. | **accept** | Summary tables. | None. |
