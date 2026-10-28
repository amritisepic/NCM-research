# Triage of Codex deep audit 10 (R056 package before launch)

Source: `docs/reviews/codex_audit10.md`, `tests/test_audit10_{e1b,numbers}.py` (local Codex CLI on `5f0c689`, branch
`codex/audit10` at `855ebad`; brief `docs/prompts/codex_audit10_r056.md`). The audit could not run Python in its
sandbox. Audit suite re-run here first on the audited commit (torch 2.6 CPU): 30 passed, 4 xfailed (strict), i.e. every
strict xfail fails on `5f0c689` as it should. Ported without the xfail markers to the PR branch: they pass after the
fixes below.

| # | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| A10-1 | P1: E1b resumes under changed inputs and runtime: the resume check compares the protocol and the selection only, so changed saved E1 rows (e.g. N_8), released weights, data, GPU or torch let finished items stand beside items computed under other inputs. | **accept** | The three A10-1 tests fail on `5f0c689` (strict xfails). Codex's PR review found the same class in round 2 (id 4396397569, fixed in `96f9eca` for models, data and E1 files); the audit adds the runtime. | `96f9eca` pins the released checkpoint files, the WMT14 pairs, TinyStories' validation stories and the saved E1 files; now also the E1 file digest inside each model's selection (`build_plan`), and the GPU and torch passed by `main` are compared with the saved ones before any item runs (a resume on another GPU type or runtime is refused). This reverses the PR-review triage's choice not to pin the device: AGENTS.md requires a resumed run to equal an uninterrupted one, and W2M-G pins it too. The A10-1 tests pass. |
| A10-2 | P1: an interrupted item's elapsed time is never written, so the registered 150-minute cumulative cap is renewed by every interruption, and `not_reached` / `complete` go stale between items. | **accept** | `test_interrupted_session_time_is_charged_before_another_item_can_start` fails on `5f0c689` (strict xfail): after 149 + 5 minutes and an interrupt, the resumed session starts two more items. | `run` writes minutes and status after every item, on any exit (`finally`) and, inside an item, every 30 s of sampling (`continue_item(on_round=...)`), so a hard kill (a Colab disconnect cannot be caught) loses at most 30 s instead of the audit's proposed fail-closed marker, which would block every resume after a disconnect. Tests: the audit's A10-2 test and `test_elapsed_time_reaches_disk_inside_an_item_before_a_hard_kill`; removing the final checkpoint fails the former, removing the in-item checkpoint fails the latter (mutations killed). |
| numbers | Recomputed W2M seed-0 numbers (M2 - M3 20.936 ± 0.374, 42.62 / 21.69, 4.035 nats) and E1b's selection (J0 58, A0 14, S0 0) agree with the registration; the registration printed J0's delta as `128 x 4` (three registered cuts: `128 x 3`). | **accept** | Matches our own computation. | Registration erratum already pushed to `claude/ncm-research` (`7d91f87`) before this audit's result. |

No P0 / P1 was found in the L1R replay and assignment, the M0-M7 trainer delta, W2M-G, analyze_w2c or the notebooks
(source review; the GPU paths were not executed by the audit).
