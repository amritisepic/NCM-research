# Agent rules (Codex and Claude)

Read `STATE.md` first, then `docs/brief.md` (goals, success bar, honesty rules). Every run is logged in
`log.md`; registrations are in `docs/preregistration.md`. Colab notebooks clone branch `claude/ncm-research`
and pull its latest code at start, so anything merged there can change a running wave.

**This repository is public.** Never write absolute local paths, usernames, personal names or emails, session
links or credentials into any file, commit message, PR or review comment. Refer to files by repo-relative path.

## Code Review Rules
Report P0 (results would be invalid) and P1 (blocks a trustworthy registered comparison or a reproducible
continuation). Every finding: repo-relative file:line, why it is wrong, and a test that would expose it.
These are the defect classes earlier audits found here (`docs/reviews/`); check them first.

### Precision and evaluation protocol
- Autocast or other context managers must not be re-entered or leak state across calls (R024: an autocast
  leak put on-policy arms' evaluation in bf16 while uniform arms scored in fp32).
- Sampling, scoring and teachers used as targets run in fp32 with TF32 off, identically across arms.

### Resume and reproducibility
- Every RNG (data, corruption, pool, sampling) and every piece of training state is in the one atomic
  checkpoint and restored without drawing extra data; a resumed run must equal an uninterrupted one.
- A run directory must refuse a config, base checkpoint or arm that differs from its manifest.

### Matched comparisons
- Arms with the same seed see the same data in the same order; evaluation cadence must not change training.
- FLOP / NFE accounting matches what the code actually executes (no early-stopping teacher counted at full
  budget); proxies are labelled as proxies.

### Evaluation integrity
- Splits: val = even-index, test = odd-index validation items (`src/ncmr/data.py`); nothing tunes on test.
- Per-item outputs needed by a registered analysis (capped hypotheses, per-puzzle records) are saved.
- The code implements the registration and its amendments in `docs/preregistration.md` exactly.

## Deep audit protocol (when asked for an audit)
- Do not change production code. Write `docs/reviews/codex_audit<N>.md`: per finding, an id, P1/P2, locations,
  the failure, and a proposed patch as a diff (not applied).
- Add CPU tests in `tests/test_audit<N>_*.py` that demonstrate each defect, marked `xfail(strict=True)`, so they
  start passing only when the defect is fixed. Use `tests/conftest.py` and the hash-pinned assets.
- Recompute any number the audited log entries report from `results/**.json`; list disagreements.
- Finish with a summary table: id, priority, file, one-line issue.
