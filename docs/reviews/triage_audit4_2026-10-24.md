# Triage of Codex deep audit 4 (WMT14 tier-L trainer before the J1 launch)

Source: `docs/reviews/codex_audit4.md`, `codex_audit4_numbers.json`, `tests/test_audit4_*.py` (Codex desktop, run by the
user on `c91126a`; final version `665c24b` on `codex/audit4`). The audit's suite was re-run first on the audited code:
**18 passed, 7 strict xfails**, each failing on its intended assertion, as the report states. Every finding below
reproduces. Verdicts: **accept** (finding correct, acted on), **partial**, **reject**. Fixes are in PR 8; the J1
registration is amended as J1c (`docs/preregistration.md`).

| # | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| A4-1 | P1: `val_nll` scores under the training autocast (bf16 / fp16), not fp32. | **accept** | `test_val_nll_generator_forwards_are_fp32` xfails on `c91126a` (tiny model: -0.059 nats/sentence). Same class as audit 3's A3-1. | Autocast disabled for the scoring forward (TF32 is already off on CUDA in this trainer). |
| A4-2 | P1: the run manifest identifies the base tokenizer and generator by byte size, so same-size different weights pass. | **accept** | `test_manifest_refuses_changed_same_size_base_weights` xfails for both files. The audit-2 triage had accepted size-only; that is superseded. | SHA-256 of both files in the manifest. |
| A4-3 | P1: dropping a degenerate source also drops its four student rows, while the FLOP proxy still charges them (576 - 12d rows executed vs 576 reported). | **accept** | `test_realized_generator_rows_match_registered_proxy[1]` and `[16]` xfail. | Codex's patch: dropped sources run as zero-loss padding rows, so every step executes the registered rows; an all-degenerate step returns an exact zero loss. Audit 2's degenerate-source test is adapted to this contract (no training state is ever a fixed point; only zero padding reaches the student). J1c. |
| A4-4 | P2: with WMT's static prefix, an all-zero start that is already the fixed point is charged two detection passes. | **accept** | `test_initial_fixed_point_needs_one_detection_pass` xfails. | Opt-in `initial_fixed_ok` in `jacobi_decode` for the WMT teacher and evaluation; the default (off) is kept for Sudoku's feedback decoding, whose prefix changes after pass 1. J1c records the effect on J0's 11.07-pass sanity reference. |
| A4-5 | P2: curve evaluations keep only aggregates (no ordered hypotheses, no per-sentence passes, no sequential check). | **accept** | `test_curve_evaluation_persists_ordered_comparator_inputs` xfails. | Per-step `hyp_val_step<N>.json` (ordered exact and capped hypotheses, per-sentence passes) with the sequential check on; the notebook collects them (guarded for a missing run directory). |
| draft note | The draft report (not the final) noted that `traj_stats`, which J1b's 1% degenerate rule needs, was an interval logging buffer, reset at each log and not checkpointed. | **accept** | True of `c91126a`: a time-budget stop or crash loses the counts since the last log. | Run-level totals kept in `traj_stats` (never reset), saved in every checkpoint, restored on resume, reported in `result.json` (`degenerate_pct`). `tests/test_wmt_run_stats.py`, mutation-checked. |
| numbers | All R020 / R028 WMT numbers recompute; the R028 "up to 28%" shortfall is consistent with the proxy algebra (27.8%). | **accept** | Recomputed by the audit's numbers test. | Ported: the snapshot's count of `results/*.json` (62 at the audited commit) is excluded in the working tree, which has more result files; every number recomputed from a named file still is compared. |

Ports of the audit's tests, beyond removing the seven strict-xfail markers: the `traj_stats` equality now compares the
interval keys and checks the run-level totals separately; the restore-to-first-step RNG check accepts the restore as an
assignment (its return value is now used). With all fixes: **103 passed** on torch 2.11.0+cpu and on 2.6.0.
