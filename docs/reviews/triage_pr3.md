# Triage of Codex's review of PR 3 (notebook provenance)

Reviewed commit `257234e`. Verdicts: **accept** (finding correct, acted on), **partial**, **reject** (with evidence).

| Comment id | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| 4365630299 | P1: a run directory whose `ckpt.pt` predates provenance recording has no `provenance.jsonl`, so `started` is empty and the code-version check is skipped; the resume then runs under possibly changed code and the new history records only the current SHA. | **accept** | Reproduced: `tests/test_notebook_provenance.py::test_refuses_checkpoint_without_provenance` (a checkpoint without provenance) fails on `257234e` (the training command runs) and passes after the fix. Every wave-1 run directory on Drive is in exactly this state. | The run cell now refuses a checkpoint without provenance unless `ALLOW_CODE_CHANGE = True` is set on purpose. Regression test added; mutation-checked (removing the new condition makes it fail). Collection still tolerates missing history. |
| 4365642733 | P1 (round 2, on `d420142`): the check compares only the first session's commit. Started at A, resumed under B with the override, back at A with the override off: `started[0] == GIT_SHA`, so the B-trained checkpoint resumes silently. | **accept** | Reproduced: `test_refuses_return_to_first_commit_after_override` (start records A and B, a checkpoint, current commit A) fails when the loop checks only the first session and passes with the fix. | The cell now requires every earlier session's commit to have identical code under `src/ scripts/ configs/ assets/` (or the override). A mixed run therefore needs the override at every later resume, which is the intent. |

Round 3 was requested only to review the accepted fix on the new head; neither finding is disputed.
