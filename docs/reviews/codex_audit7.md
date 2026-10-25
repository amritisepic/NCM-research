# Deep audit 7: J2-F delta after audit 6

Audited commit: **`b4673ab44449c49747e2a07dedac1167c3f393d5`**.
Scope: the requested `dd64aa8..81a6979` delta in
`scripts/train_wmt14_L.py`, `scripts/analyze_j1.py`,
`notebooks/make_notebooks.py`, the frozen-arm notebook and their tests.
Specification: `docs/prompts/codex_audit7_j2f_delta.md`, J1/J1a/J1b/J1c/J2-F
in `docs/preregistration.md`, and the audit-6 triage. Prior audited behavior
outside this delta was not reopened. Only this report and audit7 tests were
added; the explicit allowlist leaves `log.md` unchanged.

**No new P0/P1 finding was established. Executable verification remains
pending.** The source review and independent result arithmetic are complete,
but neither the new CPU probes nor the existing regression suite could be
executed in this audit environment. There is no new test-pass receipt and no
released-WMT14/A100 replay claim. No production patch is proposed or applied.

## Verification and its limits

The review traced the actual loader, frozen-copy construction, resume call,
evaluation call sites, Jacobi implementation, tokenizer and training loop.
The manifest and analyzer fixes were also compared with audit 6's proposed
diffs. Independent JSON arithmetic was recomputed using Node.

Python is absent from the shell's command search path. The existing virtual
environment's launcher points to an unavailable interpreter; the cached
standalone-runtime directory is empty. Obtaining a temporary interpreter
failed because the download connection was blocked. Consequently, no pytest
pass, xfail, mutation result or numerical CPU reproduction is claimed here.

`tests/test_audit7_gate.py` supplies positive CPU probes for:

- Successful, mismatching and missing-file gate calls, including a surrounding
  bf16 autocast context: RNGs, module state/modes/gradients, optimizer, data RNG,
  trajectory totals and precision settings must remain unchanged.
- Equality of subsequent losses and training state with the check omitted,
  including a checkpoint round-trip and another check after resume.
- Direct output and post-call-mode equality with `dd64aa8`'s actual evaluation
  implementation, including sequential decoding, caps and multiple batches.
- The first-128, single-batch comparison against a larger J0 evaluation.
- The shared decoding path with `tests/conftest.py`'s hash-pinned released
  Sudoku fixture. This exercises shared NCM/VQ code, not released WMT14 parity.

These are regression probes, not demonstrated-defect tests. There are no
strict xfails because no new defect was established. The next verification
step, with dependencies and the pinned fixture installed, is:

```text
python -B -m pytest -p no:cacheprovider tests/test_audit7_gate.py tests/test_j2f_frozen_teacher.py tests/test_audit6_teacher.py tests/test_audit6_analysis.py tests/test_audit6_gates.py tests/test_analyze_j1.py tests/test_audit4_protocol.py tests/test_wmt_run_stats.py -q
```

## Answers to the audit questions

### 1. Training-state isolation

No state-changing path was found in a successful launch check.
`scripts/train_wmt14_L.py:310` runs the check after resume and before entering
the training loop or its autocast context. `check_teacher` reads ordered text
and calls `evaluate` with the frozen teacher; it never calls `batch`, samples
trajectory states, updates trajectory counters, or touches the optimizer.
`evaluate` uses no-grad, fresh autocast-disabled contexts, eval-mode forwards
and greedy argmax decoding. The tokenizer is frozen/eval. The executed prefix,
VQ reconstruction and decoder paths contain no random draw or training-buffer
update. CUDA TF32 is disabled at process setup (`:71-73`); the check does not
toggle it.

The teacher is already eval when passed to `evaluate`. Saving its mode and
restoring `m.train(was)` (`:238-261`) therefore leaves it eval and does not call
`net.train()` on the student. The student, its independent NumPy data stream,
and the checkpointed Torch stream have no identified mutation. Each
unfinished resume with the notebook's command repeats this check on the
rebuilt released teacher, after restoring training RNGs.

The CLI flag remains optional, and a checkpoint already at the configured
step limit skips the check. The supplied notebook always passes the flag;
every invocation that can execute another training step rechecks it. A direct
CLI invocation omitting the flag does not provide this gate's evidence.

This is source evidence, with the dynamic isolation/resume probes supplied
but unexecuted. Startup duration and GPU allocator/kernel caches can change;
they are not the saved training RNG or model state, and this review does not
claim identical timing.

### 2. Decode comparison and existing evaluation calls

`scripts/train_wmt14_L.py:302` selects the first 128 validation sources and
references in their existing order. The default evaluation batch size is
128, so these are exactly J0's first batch. Source tokenization, truncation,
padding, prefix embeddings, padding mask, `K + 1` Jacobi budget,
`initial_fixed_ok=True`, VQ reconstruction and detokenization are unchanged.
Both use fp32 evaluation with process TF32 disabled. The frozen generator and
its copied tokenizer were reconstructed before loading a student checkpoint;
the original frozen tokenizer still supplies the same prefix and token map.

Disabling the sequential verification and capped-output decoding in the check
does not change Jacobi's exact inputs or stopping logic. In J0 those operations
follow the exact decode of the batch. They use greedy/eval paths as well.
The gate separately compares all 128 hypothesis strings and pass counts
(`:303-307`), so matching text with different convergence counts is rejected.

For existing calls with `model=None`, the selected model is still `net`.
Every existing curve evaluation follows training; final evaluation follows
`val_nll`, which leaves `net` in train mode, including J0's zero-step path.
Thus restoring the entry mode leaves those calls in the same train mode as
the old unconditional `net.train()`. Their precision contexts and output
calculations are unchanged. An arbitrary caller entering with an eval-mode
student would now retain eval mode, but no existing call does so.

The first-128 comparison is a finite behavioral check. Different weights or
behavior on other inputs can pass if these hypotheses and counts coincide;
that is not proof of full model identity. Base-file digests provide the
separate resume identity check. No A100 execution was available to establish
whether a library/kernel difference could cause a spurious mismatch on the
registered hardware; the gate fails closed if such a mismatch occurs.

### 3. Notebook input and stale/missing files

`notebooks/make_notebooks.py:158` and the checked-in frozen notebook pass
`{R}/L_wmt_released/hyp_val.json`. The control notebook's J0 command writes
that raw file under the common Drive runs root. The collected result files
with `__` prefixes are a different export and are correctly not passed here.

A missing file exits at `scripts/train_wmt14_L.py:299-300`, before training.
Short records or differing first-128 hypotheses/pass counts cannot reach the
required equality counts. Invalid JSON or absent keys also abort the process.
A stale file with differing checked content therefore fails closed. The raw
JSON has no source/checkpoint provenance to authenticate: a stale file whose
checked content is identical passes. That is the evidence boundary of the
specified comparison, not an established defect in its implementation.

### 4. Audit-6 repairs and their tests

A6-1's guard at `scripts/train_wmt14_L.py:93-98` matches audit 6's proposal:
either an existing checkpoint or manifest triggers identity validation before
overwriting the manifest. The six ported pre-checkpoint cases retain their
changed-arm, recipe and same-size-base-file checks and their manifest
preservation assertion.

A6-2's analyzer change matches audit 6's proposal. The default candidate still
selects J2. Selecting J2-F uses its values for screen, H12, cap comparators,
bootstrap labels, degenerate share and proxy comparison, while retaining J0,
J1 and J2. Historical rest-of-corpus diagnostics still use J2's shared-output
mask; J2-F's own rest uses its own mask. The additional
`tests/test_analyze_j1.py::test_candidate_j2f_keeps_j2_mask_for_original_arms`
uses distinguishable pass counts to catch mask substitution that the original
audit-6 synthetic test could miss. No regression was identified by inspection;
the retained and new tests were not rerun here.

## Log and result arithmetic

R037 is the log entry for this delta. Its review identifiers, test counts and
process claims have no corresponding experiment JSON from which to recompute
them. In particular, its historical full-suite claim is not independently
verified by this audit. There is no J2-F experiment result to recompute.

As a backward-compatibility cross-check, sentence-ordered data under
`results/colab/w2c_L_wmt_ctrl/` and `results/colab/w2c_L_wmt_consist/` were
independently compared with `results/j1_analysis_s0.json` and R036. All six NFE
means/histograms, 72 capped pass means, capped hypothesis equality rates,
sharing, duplication, rest-of-corpus pass means and final-curve agreement
agreed. Representative recomputations:

| Quantity | Recomputed |
|---|---:|
| J0 val/test mean passes | 10.916000 / 11.068931 |
| J1 val/test mean passes | 11.527333 / 11.622045 |
| J2 val/test mean passes | 9.357000 / 8.890110 |
| J2 test saving / BLEU loss | 19.180819% / 6.589668 |
| J0 cap-9 BLEU loss / saving | 0.099403 / 18.315018% |
| J0 cap-8 BLEU loss / saving | 1.108609 / 27.278782% |
| Shared J2 test sentences / share | 487 / 16.217116% |
| J2 passes on shared / remaining test sentences | 2.928131 / 10.044118 |
| J1 and J2 generator FLOP proxy total | 83,837,908,942,848,000 each |
| J2 trajectory / degenerate sources | 32,000 / 0 |
| J1/J2 result wall seconds | 3071.235334 / 4201.913802 |
| Provenance elapsed total / CU at 5.4 per hour | 132 minutes / 11.88 |

No numerical disagreement was found. Matching reference translations are
absent, so BLEU and length-ratio scalars were checked against stored results,
not freshly rescored; no paired BLEU bootstrap was run. Audit 6's qualification
still applies: aggregate results include the recorded exact/sequential
disagreement sentences; only the exactness claim excludes them.

## Summary

| ID | Priority | File | One-line issue |
|---|---|---|---|
| None | — | — | No new P0/P1 established; CPU execution and released-WMT14/A100 replay remain unverified. |
