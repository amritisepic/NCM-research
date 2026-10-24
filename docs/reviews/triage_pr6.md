# Triage of Codex's review of PR 6 (wave-1 exploratory scripts)

Reviewed commit `8903805`. Verdicts: **accept** (finding correct, acted on), **partial**, **reject** (with evidence).
The adversarial review's ADV-1 (fixed in this PR) is triaged in `docs/reviews/triage_adversarial_onpolicy_2026-10-24.md`.

| Comment id | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| 4367731857 | P1: `scripts/w1_sample_stats.py:107` and its `--read` message print absolute local paths (the default `--out` is built from the checkout root), as does `scripts/onpolicy_repro.py:200` when given an absolute `--out`; stdout ends up in pasted logs. | **accept** | Reproduced by reading the code: both scripts print `a.out` verbatim; the result files themselves carry no paths. No registered comparison depends on it, but the public-repository rule covers anything that may be pasted into `log.md`, and the fix is small. | `shown()` in both scripts prints a repo-relative path, or only the file name outside the repository (or on another drive). `tests/test_exploratory_print_paths.py` runs `w1_sample_stats.main()` on stand-in data in its default and absolute `--out` forms and checks every output-path `print` in `onpolicy_repro.py`; each of the three print sites was mutation-checked (reverting it fails the test). |
