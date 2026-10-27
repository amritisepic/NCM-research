# Triage of Codex's review of PR 14 (D1 benchmark, E1 estimator, W2Ma probe)

Reviewed commit `b37a11a`. Verdicts: **accept** (finding correct, acted on), **partial**, **reject** (with evidence).

| Comment id | Finding | Verdict | Evidence | Action |
|---|---|---|---|---|
| 4390707645 | P1: `integrity_wmt` compares each WMT14 Jacobi implementation only with the saved batch-128 outputs, so eager and cached could each differ on 2 different items (4 eager/cached disagreements) and the gate still reads; D1 requires the implementations to agree. | **accept** | Reproduced on `b37a11a`: with eager wrong on items {0, 1} and cached on {2, 3}, every list in `integrity_wmt` has <= 2 entries and `gate_ok` is True. An independent internal review found the same gap. | Direct eager-vs-cached comparisons (outputs and passes) for J0 exact, J0 + stable x1 and J2-F + stable x1, each held to the same tolerance. Test `test_gate_needs_deterministic_repeats_and_direct_implementation_agreement` (fails on `b37a11a`, passes now; mutation removing the comparison is killed). |
| 4390707659 | P1: the 2-item tolerance also covers `nondeterministic`, so an item whose timing repeats disagree can still pass the gate; D1 requires identical repeats. | **accept** | `gate_ok({"nondeterministic": [0]})` was True on `b37a11a`. The registration's tolerance is for batch-128 vs batch-1 near-ties, not for repeats. | `gate_ok` requires an empty `nondeterministic` list. Same test as above (mutation restoring the old rule is killed). |
| 4390707670 | P1: stable / verifier FLOPs charge one tokenizer decode per pass, but a fixed point at pass t > 1 returns before decoding and the output reuses pass t-1's decode, so t-1 decodes run; executed cost is overstated. | **accept** | Code path: `jacobi_stop` returns on a fixed point before calling `decode` / `check`; `memo` returns the previous decode for the output. The overcount is against Jacobi (conservative for the gate) but D1 registers executed FLOPs. | `decodes_made` counts the decodes that run (stop reason recorded per item). Test `test_counted_decodes_match_what_a_fixed_point_stop_runs`: whole-configuration FlopCounterMode counts equal `flops_per_item` per item, with a fixed-point stop present (fails with the old count; mutation killed). |

Also fixed in this round, from an internal blind review (no Codex comment): E1 excludes items whose reconstruction x^ is not a
valid output (a Sudoku board that does not parse would otherwise count unparseable samples as hits), reports A_k's SE,
uses the registered sizes as defaults, and records the R_hi clamp (zero-hit T = min(log(n/3), N_k)); D1 labels a gate
verdict computed on a GPU other than the A100. Tests in `tests/test_e1_equivalence.py`.
