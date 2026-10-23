# Triage of the second code audit (log R028)

Source: `codex_audit2.md` (audited `b397656`, the post-R024 code; the four subject files had not changed since).
The audit's suite was re-run here before any change: 12 passed, 13 strict xfails, as reported.

| # | Finding | Verdict | Action |
|---|---|---|---|
| A2-1 | Repair pool saved separately and non-atomically; restoring constructs a pool first (draws data); final save omits the pool. | **accept** | Pool and pool RNG in the atomic checkpoint (`save_checkpoint()`); restored pool draws no data; a repair checkpoint without them fails. |
| A2-2 | Sudoku corruption RNG reseeded on resume instead of restored. | **accept** | Saved and restored; missing state fails. |
| A2-3 | Auto-resume can relabel an arm or contaminate A0 / J0. | **accept, modified** | Run manifest (config, data SHA-256, base config hash, base checkpoint byte sizes, device) checked before anything is overwritten; `--eval_only` refuses a directory with a checkpoint. Modified: full SHA-256 of the released .pt files replaced by byte sizes (they are read from Drive; 0.5-1.1 GB per run start) and torch version not part of the identity (a Colab image update between sessions must not block a resume; it is still recorded in result.json). |
| A2-4 | J2 teacher ran under the bf16 training autocast, not the fp32 map exact decoding uses. | **accept** | Teacher in autocast-off fp32, TF32 off on CUDA (also for all evaluation, as in wave 0). Costs J2 extra wall-clock (fp32 trajectories), reported. |
| A2-5 | A source whose all-zero start is already fixed would be trained on its fixed point. | **accept, modified** | The audit's patch raises an error; that would end a 4-CU run on one pathological sentence. Instead such sources are dropped and counted, and the policy is registered (J1b, with a 1% flag). |
| A2-6 | Same seed does not give the same data order across arms (WMT 128 vs 64 + 16 rows; Sudoku pool refills on the training RNG). | **accept** | One common batch per step in every arm; separate pool RNG (S1c, J1b). |
| A2-7 | WMT FLOP match holds only for the upper bound; early-stopping teachers spend up to 28% less. | **accept** | Teacher runs the full K + 1 passes (`jacobi_decode(full_budget=True)`); result field renamed `generator_flops_proxy_total`. |
| A2-8 | `prep_wmt14.py` can return fewer than n pairs or reuse a stale archive. | **accept** | Walks the full permutation (same pairs as before when enough are eligible: checked on a 20k smoke archive), asserts n, saves and checks selection metadata. Dataset revision pinning: not done (noted as provenance gap). |
| A2-9 | S1b tails missing; c* not recorded. | **accept** | p90 / max NFE / cap hits per cap, per-puzzle records for every mode, c* from val stored. |
| A2-10 | Capped hypotheses discarded; comparator cap may be missing; sequential disagreement only a percentage. | **accept, modified** | All caps 1-12 evaluated and saved (val and test). Modified: a sequential disagreement is recorded per sentence instead of raising, so a rare argmax tie cannot destroy a finished run's evaluation. |
| P1c-1 | TC estimator calibrated on synthetic codes with known TC (independent, duplicated, two Markov chains), 3 seeds each. | **accept** | All 12 within 3.1e-5 nats of the truth (tolerance 0.05). Supports basic instrument behaviour only (one 4-code level, constant context); P1c step 2 (side-channel probe on real codes) still to build and register. |

Tests: the audit's file now runs against the working tree (`AUDIT2_REF` pins a commit). Strict xfails removed;
nine tests adapted to the fixed interfaces (noted in the file). 25 / 25 pass on the working tree; against `b397656`
14 fail, so they still discriminate. Whole suite: 59 passed.
