Deep audit 7 before `w2c_L_wmt_consist_frozen`: the delta after deep audit 6. Follow "Deep audit protocol" and
"Code Review Rules" in AGENTS.md. Push only `docs/reviews/codex_audit7.md` and `tests/test_audit7_*.py`; change no
other file.

## Why
Audit 6 audited `dd64aa8`. Its two P1s were fixed with its own patches, and a launch-time check was added that changes
the evaluation function, so CLAUDE.md's Tier-2 rule applies to that delta before launch.

## Subject
`git diff dd64aa8 81a6979 -- scripts/train_wmt14_L.py scripts/analyze_j1.py notebooks/make_notebooks.py
notebooks/w2c_L_wmt_consist_frozen.ipynb tests/` (merged as PR 10). Triage of audit 6:
`docs/reviews/triage_audit6_2026-10-24.md`.

## Spec (what correct means)
- `evaluate(..., model=None)` must behave exactly as before for every existing call (J0 / J1 / J2 / J2-F curve and final
  evaluations: same model, same precision, same mode afterwards, same outputs); with `model=teacher` it decodes with the
  frozen teacher and leaves it in eval mode.
- `check_teacher` (J2-F gate 1): before any training step, the frozen teacher's exact decode of the first 128 val
  sentences (one batch, as J0's evaluation decoded them) must equal J0's saved `hyp_val.json` hypotheses and pass
  counts, else the run exits before training; it must draw no random numbers and change no state that training or the
  checkpoint uses; on resume it runs again identically.
- A6-1's guard and A6-2's analyzer as in audit 6's patches.

## Questions
1. Can `check_teacher` change any RNG stream, module mode, autocast / TF32 state, or the data order, so that J2-F's
   training differs from what it would be without the check?
2. Is the comparison sound: same sentences, order, batch composition, precision and map as J0's saved decoding? Can it
   pass while the teacher differs from J0, or fail spuriously on the registered GPU type?
3. Does the notebook pass the right file, and does a missing or stale J0 file fail closed?
4. Any regression in the A6-1 / A6-2 fixes or their tests?

## Out of scope
Everything audit 6 covered and found correct.
