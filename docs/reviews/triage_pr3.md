# Triage of Codex's review of PR 3 (notebook provenance)

Reviewed commit `257234e`. Verdicts: **accept** (finding correct, acted on), **partial**, **reject** (with evidence).

| Comment id | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| 4365630299 | P1: a run directory whose `ckpt.pt` predates provenance recording has no `provenance.jsonl`, so `started` is empty and the code-version check is skipped; the resume then runs under possibly changed code and the new history records only the current SHA. | **accept** | Reproduced: `tests/test_notebook_provenance.py::test_refuses_checkpoint_without_provenance` (a checkpoint without provenance) fails on `257234e` (the training command runs) and passes after the fix. Every wave-1 run directory on Drive is in exactly this state. | The run cell now refuses a checkpoint without provenance unless `ALLOW_CODE_CHANGE = True` is set on purpose. Regression test added; mutation-checked (removing the new condition makes it fail). Collection still tolerates missing history. |
