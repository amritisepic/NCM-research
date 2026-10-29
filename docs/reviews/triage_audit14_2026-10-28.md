# Triage of Codex deep audit 14 (claim audit: W2C + W2M-GR, R065's numbers and the combined wording)

Source: `docs/reviews/codex_audit14.md`, `docs/reviews/codex_audit14_numbers.json`,
`tests/test_audit14_{claim,numbers,protocol}.py`. The audit ran on the local Codex CLI (gpt-6.1-sol / ultra) at
`3b601dc`, on branch `codex/audit14` at `5b1f4ed`, from brief `docs/prompts/codex_audit14_w2cgrclaim.md`.

How the files were committed: Codex's closing message hit the usage limit after the five audit files were written.
They are complete; the summary table is present. Its sandbox could not commit, so they were committed unchanged on its
branch by hand.

Python ran in the audit's sandbox. The audit suite re-run here on the audited commit: 30 passed, 4 xfailed (strict).
These are the two claim-scope guards and the two margin guards, and they fail on `3b601dc` as they should.

Ported to `claude/ncm-research` with the xfail markers removed: 34 pass after R067 and the prompt note. The
neighbouring suites (W2M-G, W2M-GR, audits 12-14) pass: 320 tests. One earlier invocation of that suite printed a crash
traceback from the pytest process with no failing test; an identical rerun was clean.

| # | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| A14-1 | P1 (claim scope): R065's Status and section 4 of the Astra round-4 prompt read crossed replay as an origin. They say "the damage forms at levels 2-5" and "not at L1", and "later levels alone carry ~5%" drops the M3-prefix condition. Whole-range replay gives conditional responses; it does not separate damaged-prefix content from M3's response to a foreign-prefix distribution or from checkpoint compatibility, and it says nothing about code semantics. | **accept** | Reproduced from the cells: the L2-5 switch moves 0.511 of the contrast given M2's L1 and M3's levels 6-11, but 0.828 given M3's L1 and M2's levels 6-11. M2's L1 moves 0.016 on an M3 continuation and 0.119 on an M2 one. No cut at 3 or 5, no single-level replay, and no same-arm cross-seed replay exist. The audit's two claim guards fail on `3b601dc`. | R067 supersedes R065's claim wording with the conditional-replay wording and its limits. A dated correction note in section 4 of the sent prompt (original kept). `docs/claim_w2c_2026-10-28.md` uses the corrected wording. Guards ported and passing. |
| margins | R065's printed margins "pi_2 <= 0.025" and "interactions >= 0.416" round inward: seed 2's pi_2 is 0.025342 and seed 3's cut-6 interaction is 0.415719. | **accept** | `results/w2mgr_analysis.json` per-seed values. The table entries are correctly rounded. The two strict xfails fail on `3b601dc`. | R067 states pi_2 < 0.026 and interactions > 0.415. Tests ported and passing. |
| provenance | R065 says every share, ratio and contrast was recomputed from the cells' per-group Gen-PPL. Only the replay shares were. (i) and (iii) use saved means; their bootstrap CIs need per-window arrays that the notebook's collector does not copy. | **accept** | `scripts/w2mg_eval.py` saves the arrays in the run's cache on Drive; `results/colab/w2mgr_eval/` holds only results and provenance. | Corrected in R067. Archiving the cache arrays with hashes is proposed to the user, as a copy from Drive. |
| sanity z | The z uses the replay SE alone, which is not a two-estimate test. | **accept** (note) | It is a fixed-reference check, as registered. With both SEs in quadrature: -1.46 / -0.39, -1.30 / +0.22, -0.33 / -0.56. | Stated in R067. No reading changes. |
| ckpt identity | W2C never recorded its `final.pt` hashes, so W2M-GR's checkpoint hashes cannot be authenticated against W2C's. | **accept** (limit) | `scripts/train_gen.py` writes the checkpoint without a digest. Names and metrics agree; no mismatch is shown. | Stated in R067. A trainer change to record final hashes would go through PR, Codex review and a deep audit; not urgent. |
| (c) | On the strict reading of `docs/brief.md` ("what the levels encode"), W2C + W2M-GR does not establish criterion (c). Counting it would be an interpretive decision, not a stronger result. | **accept** | The replay shows which level ranges, under intervention, carry a generation failure. It does not show what those levels represent. | R067. The decision goes to the user (`docs/claim_w2c_2026-10-28.md`). |
| wording B | The external reviewer's wording is sounder if "localizes" means a conditional sampler-range contrast. | **accept** (note) | — | The claim document offers the audit's precise version of B as an option. |
| gap 5.5 | The prompt's "5.5 Gen-PPL above the released model" is correct (21.677 - 16.164 = 5.512). R059's 5.19 is M3 against M3-0, not against the released model. | **accept** (note) | Recomputed. | Claim document: the clean-row cause rests on the seed-0 M3-0 intervention. |

No P0. No implementation defect was demonstrated. The registered readings (localization, (i), (iii) replicate; (ii)
distributed) stand, and every R065 table entry reproduces.
