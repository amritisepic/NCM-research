# W2M-GX build validation

Registration: `docs/preregistration.md`, W2M-GX and W2M-GXa. No experiment has been launched.

The evaluator loads all six W2C final checkpoints in one resumable run. It runs the 66 new cells and six
re-sampled pure sanity cells. Kappa uses the re-sampled pure means; its denominator and every cross-seed,
single-level and cut baseline use W2M-GR's original pure group vectors, as registered. W2M-GXa's sigma uses
the reverse cell M3_t -> M2_s with the same M2_s / M3_t pure baselines as pi.

One run avoids duplicating checkpoints/cross-seed cells across per-seed runs. Expected runtime including
the six sanity cells is about 2.8 L4-hours, about 4.8 CU at 1.7 CU/hour. A cumulative 210-minute alarm
covers checkpoint loading, evaluation and cached replays across resumed sessions. An unclosed session
is conservatively charged through the next start. On the cap failure path, the notebook collects partial
metrics and provenance, marks the result incomplete and unassigns the runtime. Finished cell caches
remain in Drive `runs/`. The collector also retains per-window JS/rho arrays for independent reanalysis.

## Executed verification

- Focused W2M-GX and original W2M-GR regression: 190 passed in 93.20 seconds.
- Subsequent pure notebook/cap/admission checks: 26 passed, 2 deselected in 18.39 seconds.
- No trainer, corruption rule or training RNG/state was changed.
- Diagnostic full CPU suite after the interrupted attempt: 40 failed, 1029 passed, 32 warnings in 553.32s
  (0:09:13); no skipped tests. All 40 failures were the existing notebook invariant forbidding any release
  before the terminal cell. The cap-failure cleanup now appears only in GX's generated run cell, and the
  invariant admits that single guarded exception. Executed tests prove it releases only after a failed
  command and exhausted cap, with partial results and provenance already copied; a successful command
  or a non-cap failure does not release early. All other notebooks retain the terminal-cell-only rule.
- New interrupted-session cap regression: 1 failed in 2.05s before the fix. An exhausted unclosed session
  now persists its charged time before refusing resume, so the notebook recognizes and cleans up the cap.
- Post-fix cap, release-order, notebook and provenance checks: 55 passed, 76 deselected in 4.00s.
- Final full CPU suite on the settled code: **1072 passed, 32 warnings in 617.66s (0:10:17)**; no
  skipped tests. Command: `python -m pytest tests --ignore=tests/_deps -q`, with CUDA disabled and
  `tests/_deps`, `tests` and `src` on `PYTHONPATH`. The warnings arise in existing degenerate synthetic
  metric fixtures. Test assets were available; no download or experiment run was needed.
- Tested tree: `0d3cb05c48a8d98d97c3fc586f21b2d14431aa65`. Sanitizing the three unpublished commit
  identities left this tree unchanged. The later validation-record commit changes this document only.

## Behavior mutations

Each mutation was applied temporarily, its named CPU test returned a failing assertion, and the original
source bytes were restored. Bytecode was invalidated before the mutant run and after restoration.

| Mutation | Test in `tests/test_w2mgx.py` | Outcome |
| --- | --- | --- |
| Minor compatibility cutoff 0.10 -> 0.09 | `test_reading_compatibility_minor_boundary_every_arm_pair` | caught |
| Material compatibility mean cutoff 0.25 -> 0.26 | `test_reading_compatibility_material_mean_boundary` | caught |
| Localization pi cutoff 0.30 -> 0.31 | `test_reading_localization_holds_boundaries_every_pair` | caught |
| Localization failure mean cutoff 0.15 -> 0.14 | `test_reading_localization_fails_strict_mean_boundary` | caught |
| Concentration half-sum -> third-sum | `test_reading_concentration_half_summed_effect_both_directions_every_seed` | caught |
| Pure-cell sanity margin 3 SE -> 4 SE | `test_reading_sanity_exact_three_se_and_beyond` | caught |
| Sigma reverse pair M3_t -> M2_s replaced by M3_s -> M2_t | `test_gxa_sigma_uses_reverse_seed_pair_with_same_baselines` | caught |
