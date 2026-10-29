# PD/PDa build validation

Specification: the final PD and PDa sections of `docs/preregistration.md`. This is build validation, not a scientific
pilot result. The implementation uses the amended exact permutation rule; no registered thresholds were selected.

The executed tiny CPU trainer compares P and D data, masks, clean rows, unpermuted proposals and corresponding RNG
states; checks exact masked per-position multisets and unchanged levels outside L2-L5; checks coverage by row identity;
and compares the complete resumed model, optimizer, every RNG and coverage state against uninterrupted training.
The run-identity tests execute the production config/manifest guards for changed arm, config and base weights.
Adversarial permutation groups cover singleton, crossed two-row codes and all canonical codes equal.

Focused CPU suite: `tests/test_pd.py`, `tests/test_pd_analysis.py`, `tests/test_pd_launch.py`: **36 passed in 29.76s**.
The affected existing trainer subset first gave **45 passed, 1 failed in 157.01s**: its historical config guard counted
newly added PD configs as changes to pre-existing configs. It now applies `--diff-filter=MDRT`, preserving checks for
modifications, deletions, renames and type changes while allowing new registered configs. The complete suite is the
final check of that change. No test skip was introduced.

## Temporary reading mutations

Each mutation below was applied to `scripts/analyze_pd.py`, its named boundary test was run and failed, and the
original source was restored in a `finally` block. All twelve were detected; none remain applied.

| Reading or quantity | Exact temporary mutation | Test in `tests/test_pd_analysis.py` | Observed |
|---|---|---|---|
| Pilot validity | `gap >= 0.15` to `gap > 0.15` | `test_pd_validity_includes_point15_and_refuses_below` | 1 failed |
| GPT-2 rescue | `r_gpt2 >= 0.25` to `r_gpt2 > 0.25` | `test_pd_rescue_includes_every_registered_equality` | 1 failed |
| TinyStories rescue | `r_ts33m >= 0.25` to `r_ts33m > 0.25` | `test_pd_rescue_includes_every_registered_equality` | 1 failed |
| Duplicate margin | `dup_d <= dup_p` to `dup_d < dup_p` | `test_pd_rescue_includes_every_registered_equality` | 1 failed |
| MAUVE margin | `mauve_d >= mauve_p - 0.02` to `mauve_d > mauve_p - 0.02` | `test_pd_diversity_kill_only_beyond_registered_margin[mauve_d-0.78]` | 1 failed |
| Entropy margin | `entropy_d >= entropy_u - 0.05` to `entropy_d > entropy_u - 0.05` | `test_pd_diversity_kill_only_beyond_registered_margin[entropy_d-3.95]` | 1 failed |
| Kill boundary | `r_gpt2 <= 0.10` to `r_gpt2 < 0.10` | `test_pd_kill_includes_point10_but_not_above` | 1 failed |
| Coverage gate | `coverage >= 0.80` to `coverage > 0.80` | `test_pd_coverage_includes_point80_and_gates_any_reading_below` | 1 failed |
| Inconclusive | final `reading = "inconclusive"` to `reading = "rescue"` | `test_pd_rescue_requires_point25_under_both_scorers` | 2 failed |
| Underpowered priority | `elif not valid:` to `elif not valid and False:` | `test_pd_validity_includes_point15_and_refuses_below` | 1 failed |
| Same-step reference | `steps != 175` to `steps not in (175, 600)` | `test_pd_refuses_unregistered_or_inconsistent_inputs[steps]` | 1 failed |
| Mean log loss | mean group log loss to log mean group PPL | `test_pd_means_group_log_ppl_before_averaging_and_pairs_contrasts` | 1 failed |

## Notebook budget behavior

The new notebook specs enforce per-arm 25/25/15 A100-minute caps and a shared cumulative 60-minute cap, including
final evaluation and resumed sessions. Unclosed budget reservations are charged in full; this deliberately refuses
to spend an unverified remaining allocation after an abrupt runtime loss. A cap exit collects available outputs and
releases the Colab runtime before preserving the incomplete failure status. The executed cap-path regression checks
that collection and release both happen before the exception.

The CPU coverage report and final complete-suite summary are recorded separately when those checks finish. No GPU
training or launch clearance is implied by this build.

## CPU coverage memory bound

The initial real-model CPU coverage attempt completed no step or report before memory/pagefile pressure exhausted
local disk space; it was interrupted and provides no coverage evidence. The coverage-only frozen eval forward now
uses eight-row chunks, concatenating the logits before the original 64-row canonical exclusion, softmax and sampling
draw. Training's P/D/U micro-batch partition and all root/fine draw shapes and order remain unchanged. The report
identity records `cpu_forward_batch`. The tiny-model regression compares split/unsplit logits at fp32 precision,
checks identical sampled codes and replacement RNG states, and records the unchanged complete fine-draw shape.
The updated coverage/launch focused tests gave **5 passed in 11.03s**. No simulated probabilities or reduced step
count were used.
