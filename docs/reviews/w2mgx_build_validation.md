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
- The full CPU suite is pending its serialized execution slot.

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
