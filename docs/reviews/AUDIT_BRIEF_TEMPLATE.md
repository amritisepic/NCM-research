Deep audit <N> before <wave / run names>. Follow "Deep audit protocol" and "Code Review Rules" in AGENTS.md.
Push only `docs/reviews/codex_audit<N>.md` and `tests/test_audit<N>_*.py`; change no other file.

## Subject
Commit <sha>. Files: <repo-relative paths>. Log entries whose numbers to recompute: <R-ids>.

## Spec (what correct means)
Point to the registration in `docs/preregistration.md` (<section>) and state anything it leaves implicit:
shapes, which tensors get gradients, which split each metric uses, what a resume must restore, how FLOPs / NFE
are counted.

## Questions
1. <specific doubts, e.g. does a resumed repair-arm run draw the same batches as an uninterrupted one?>
2. <...>

## Out of scope
Style, naming, performance tuning, anything not run by the listed notebooks.
