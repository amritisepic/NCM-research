# Deep audit 8: W2M before launch

Audited revision: `2ede453bac101319b2871a873d275e45850d280e`, the supplied
`claude/wip-w2m` head. Publication branch: `codex/audit8`, based directly on that
revision. Audit date: 2026-10-26. Scope: the W2M registration, implementation,
eight arm notebooks, mechanism notebook, PR 5 text-trainer changes, PR 12 triage,
and R039 arithmetic. This document is the run record: the explicit publication
restriction permits only this report and `tests/test_audit8_*.py`, so `log.md`
is not edited. No production patch below is applied.

## Verdict

**Two P1 precision deviations remain. Fix them before clearing W2M.** The
proposal forwards themselves correctly disable autocast and TF32 and restore
the student's modes. The defects are elsewhere: final MAUVE still uses the
shared bf16 feature path, and W2M's canonical target encoder inherits training's
enabled TF32. Neither defect establishes an explanation for wave 1's collapse.

The original W2M suite passes all 77 parameterized cases. Additional probes
exercise production boundaries rather than relying on test names. In-memory
repairs make all three strict-xfail cases pass; production files remain intact.
R039's substantive arithmetic and registered reading agree, with two rounding
disagreements listed below.

## A8-1 — P1: W2M final MAUVE violates the registered fp32 protocol

**Locations:** `scripts/train_gen.py:329`, `scripts/train_gen.py:333`,
`scripts/train_gen.py:337`; defaults at `src/ncmr/eval_text.py:94` and
`src/ncmr/eval_text.py:123`.

W2M explicitly requires MAUVE in fp32. `_sample_metrics` calls both
`mauve_features` and `evaluate_samples` without their precision arguments.
Their defaults enable autocast for MAUVE features. Casting those features back
to float after the scorer forward does not recover fp32 features. The
`sample_metrics` wrapper turns off TF32 and preserves RNG, but does not override
this explicitly enabled inner autocast. Every W2M final evaluation takes this
path because its config has `sample_eval.mauve: true`.

The reference cache also uses W1's precision-agnostic filename. Merely changing
the generated features to fp32 would allow the old bf16 human features to be
reused, producing a mixed-precision comparison. W1's default must remain
available for its existing recipe; W2M needs explicit precision selection and
a distinct reference cache.

**Exposing test:**
`tests/test_audit8_precision.py::test_w2m_mauve_forwards_are_fp32_and_do_not_reuse_amp_references`,
parameterized for a cold cache and an existing legacy cache. It executes the
actual `_sample_metrics`, `evaluate_samples`, and `mauve_features` functions,
with a tiny CPU scorer replacing downloaded weights. Both cases strictly
xfail: the observed scorer output is bf16 under enabled autocast. The legacy
case also supplies a sentinel old reference that the corrected path must not
reuse. Both cases pass after an in-memory precision/cache repair.

**Proposed patch, not applied:**

```diff
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@
     ref = None
+    fp32_mauve = corr.mode == "mech"
     if mauve:
-        cache = os.path.join(ROOT, "runs", "cache", f"mauve_ref_val_{len(ids)}.npy")
+        tag = "fp32_" if fp32_mauve else ""
+        cache = os.path.join(ROOT, "runs", "cache", f"mauve_ref_val_{tag}{len(ids)}.npy")
@@
-            ref = mauve_features([bpe.decode(cut_at_eos(r)) for r in examples("val", len(ids))], device)
+            ref = mauve_features([bpe.decode(cut_at_eos(r)) for r in examples("val", len(ids))],
+                                 device, amp=not fp32_mauve)
@@
-    res = evaluate_samples(reenc, bpe.decode, groups, scorers=scorers, ref_feats=ref, device=device)
+    res = evaluate_samples(reenc, bpe.decode, groups, scorers=scorers, ref_feats=ref,
+                           device=device, mauve_amp=not fp32_mauve)
```

The old-arm suite currently asserts AST equality for `_sample_metrics`. This
conditional repair changes that AST. When implementing it, replace that one
structural assertion with a behavioral check preserving W1's AMP feature and
legacy-cache behavior; retain the other old-arm comparisons.

## A8-2 — P1: the canonical target teacher inherits enabled TF32

**Locations:** `src/ncmr/w2m.py:182` through `src/ncmr/w2m.py:184`, called at
`scripts/train_gen.py:249`; TF32 is enabled at `scripts/train_gen.py:122` through
`scripts/train_gen.py:124`.

`MechArm.canonical` disables autocast but leaves both TF32 flags enabled. Thus
the frozen tokenizer's encoder, learned quantization down-sampler, distance
matrix multiplication, and residual reconstruction can use TF32 when producing
the canonical codes that become training labels. In particular, disabling
autocast alone does not turn TF32 matmul or convolution into full fp32 arithmetic.
This violates AGENTS.md's precision rule for teachers used as targets. The
registered mechanism script, by comparison, disables both flags before encoding
its canonical windows.

This is common to M0-M7 and is a target-precision deviation, not an observed
arm-specific leak. The CPU probe establishes the missing boundary; this audit
does not claim to have measured changed code assignments on an A100 or L4.

**Exposing test:**
`tests/test_audit8_precision.py::test_canonical_teacher_disables_tf32_and_restores_the_caller`.
It uses `tests/conftest.py`'s released Sudoku fixture, checks the asset digests,
and observes the real encoder and quantizer while the caller deliberately
enables TF32 and bf16 autocast. Both observations are
`(autocast=False, matmul_tf32=True, cudnn_tf32=True, grad=False)` instead of
the required all-false tuple. Caller-state restoration succeeds. This is a
strict assertion-only xfail; the proposed boundary makes it pass in memory.

**Proposed patch, not applied:** restrict the change to W2M so the existing W1
hierarchy path is not silently changed.

```diff
--- a/src/ncmr/w2m.py
+++ b/src/ncmr/w2m.py
@@
     @torch.no_grad()
     def canonical(self, x):
-        with torch.autocast(device_type=self.device, enabled=False):
+        with fp32_eval(self.tok, self.device):
             return self.tok.vq.z_to_idx(self.tok._encode_pre_quant(x))
```

## Answers to the audit questions

1. **Proposal precision and state:** no bf16 proposal forward or R024 context
   re-entry was found. `MechArm._logits` enters fresh `fp32_eval`, including
   reconstruction and prefix creation. Current-policy dispatch uses `mech.fwd`;
   the frozen arm uses its own copied model. `Trainer.ctx` is a property that
   constructs a fresh context. The audit observes actual output dtype, autocast,
   TF32, grad mode and per-module modes for M1/M2/M6, and performs a student
   backward inside the same outer AMP region. It passes in eager mode and with
   `torch.compile(..., backend="eager")` dispatch. Registered configs have
   `compile: false`; CUDA Inductor kernels were not executed.

2. **Data, masks, clean rows:** the data generator is independent. One full-batch
   mask plan is drawn before the micro-batch loop and sliced by actual row count.
   Draw count is arm-independent, including M7: its rates threshold the same
   uniforms, and its selected clean rows match M0-M6. Replacement and shuffle
   generators never advance the mask generator. Existing tests check seed-0
   cross-arm slot equality, exactly 25% clean rows, seed/step variation and M7.
   Added tests run the real loop at accumulation factors 1/2/4 for every arm and
   compare consecutive plans and mask-generator states. The mask invariant
   passes. This does not assert whole-training bit identity across partitions:
   proposal draw layout, dropout and floating reductions can differ; the run
   identity correctly includes `grad_accum`.

3. **Replacement semantics:** M2/M4/M5 zero the canonical unigram entry at L0
   and set the canonical higher-level logit to negative infinity before
   sampling. Other tied codes remain eligible. M3 samples a nonzero modular
   offset. M0/M7 sample the full uniform vocabulary; M1/M6 do not exclude their
   canonical codes. M6 substitutes only the actually used L0 into its proposal
   context; levels 1+ remain canonical. M1 conditions on the fully canonical
   hierarchy. The labels supplied to `build_hierarchy` are canonical in every
   arm. No replacement/conditioning defect was found.

4. **Flags and optimizer:** the sampled masks determine the support, including
   the residual-convolution halo fixed in PR 12. Cumulative support is reduced
   by the same area bins as the tokenizer's inputs. The hand-computed fraction
   oracle and the independent tokenizer-change support test both pass, for
   linear and learned L0 up-sampling. M5 preserves the per-row, per-level
   multiset using its own generator. Zero initialization reproduces the
   released logits. Ordinary val NLL, generation, policy and curve/final
   evaluation do not call the flag module. Its one-dimensional parameter is
   assigned zero weight decay by `Trainer`; the registration does not explicitly
   prescribe decay for this new vector, and this matches the existing optimizer's
   treatment of vectors. No decay-related defect is established.

5. **Resume:** all eight arm continuation tests pass on CPU, including learned
   `w`, optimizer state, data order and disagreement accumulators. All three
   W2M generator states are saved in the same atomic checkpoint as the model;
   missing W2M state is refused. The frozen copy is taken before `maybe_resume`;
   the negative control copying it afterward demonstrably diverges. An added
   test checks distinct storage for parameters, persistent/nonpersistent buffers,
   the tokenizer held in the model's tuple, and an attention-cache tensor.
   Config/arm and hash-based base/data/runtime guards include the pre-first-
   checkpoint manifest case fixed by PR 11. Real val/sampling cadence tests
   preserve the next losses, parameters, flags, accumulators and all measured
   RNG states.

6. **Old arms:** without `masks`/`canonical`, the hierarchy matches the saved
   pre-W2M implementation exactly for clean, uniform, on-policy, re-quantized
   and soft-label paths. The old step functions have identical ASTs; matched
   uniform/on-policy loops and resumes have identical weights, losses and
   corruption/policy states. This comparison is against `38a08fa`'s pre-W2M
   behavior, already containing PR 5. The saved corruption and trainer function/
   loop snapshots were independently checked against that commit's ASTs.
   Original wave-1 proposals before PR 5
   intentionally used different precision and sampling RNG; bit identity to
   those original runs is not claimed.

7. **Mechanism measures:** the script selects even-index val windows through
   `examples("val", ...)`, builds canonical windows, perturbations and both
   corrupted-context sets once from the released model, then reuses them across
   all nine models. It disables TF32 globally. The KL is clean-to-perturbed,
   excludes target levels <= j, and rebuilds finer inputs. An independent
   `kl_div(log p_perturbed, p_clean)` oracle passes. Hidden states are taken
   after `ln_f`, excluding the prefix; probe forwards use zero flags and the
   registered 384/128 window split. The orchestration probe verifies held-out
   slicing, and the NLL oracle verifies true-flags minus zero-flags, including
   for M5. PR 12's batched KL and hidden-only tests pass. No new mechanism
   formula defect is established. Full-size A100/L4 memory behavior, including
   the initial 512-window proposal construction, was not measured on CPU.

8. **Notebooks and saved readings:** the eight notebooks and mechanism notebook
   exactly equal their generator's output. Each arm has its own config, seed 0,
   600 steps, curve samples at 100/300/600, and final seeds 42 and 0-14 with
   256 samples each. Notebook execution records source and checkpoint lineage
   and refuses changed/unknown code; existing provenance tests pass. Collection
   retains result/metrics/config/manifest/provenance files and final sample IDs
   with seed groups. Per-group Gen-PPL supports paired D(X), disagreement has
   both row denominators, clean val NLL has per-level values, and the mechanism
   JSON is collected. Adjacent word duplicates are in final metrics. **Bigram
   gain is not computed by the W2M notebook/trainer.** Saved sample IDs and the
   training stream suffice to compute it afterward using R031's fixed estimator;
   `scripts/w1_sample_stats.py` currently enumerates only W1 arms. This is a
   missing derived output, not a loss of the raw evidence or a reason to rerun
   training. Complete that registered reading before reporting W2M results.

9. **Test quality:** the brief's 43 tests are 28 arm, 9 loop and 6 old-arm test
   functions. PR 12 adds four functions, yielding 47 functions / 77 parameterized
   cases at the audited commit. Most execute production methods, or AST-extracted
   production functions and the actual loop on tiny upstream tokenizers/models.
   Some are structural checks, not behavioral guarantees. In particular:
   - The named proposal-precision test never inspects forward dtype or TF32.
     In-memory mutation preserving eval/mode/RNG behavior but re-enabling bf16
     inside the proposal boundary **still passes that test**. The audit adds
     direct forward observations and a same-region backward.
   - `Harness.state()["torch_rng"]` reads the current global RNG when called.
     Comparing `ref.state()` with `h.state()` after the second run therefore
     compares the same live RNG twice. The analogous resume RNG assertion is
     also vacuous. Other model/loss/state assertions remain useful. Audit probes
     capture RNG snapshots immediately after each independent run.
   - Plan slicing was tested directly, but the original suite did not compare
     actual loops at different accumulation factors. The audit adds that check.
   - The harness stubs val NLL and sampling, so its original resume/loop tests
     could not detect evaluation side effects. Audit probes run the real val
     and sampler, replacing only external LM scoring.
   - The original suite does not independently exercise mechanism KL direction,
     probe split/wiring or flag-use sign, nor the registered fp32 MAUVE gate.
     Added metric/orchestration probes and A8-1 cover these. M0/M6's allowance
     for canonical hits is established by source, not a dedicated original
     support test. CUDA execution and a full-size GPU continuation remain
     outside this CPU validation.

10. **R039:** recomputation and disagreements follow.

## R039 recomputation

R039 and its result files are absent from the audited commit. They are available
in the later, pinned results commit
`daf2b57ee5bd9296ccd200fb71c03cb32517ef80`. The inputs are
`results/colab/w1d_requant_diag/w1d_A{0,1,2}__result.json`; their SHA-256 values
are pinned in `tests/test_audit8_numbers.py`. That test reads the working-copy
files if available or the pinned Git objects, without copying them into this
audit branch. A checkout lacking both skips these numerical checks explicitly.

For each comparison, subtract paired GPT-2 large group scores and compute the
sample SD of the differences divided by sqrt(n): n=8 sample seeds for the
frontier, n=4 window groups for the oracle. For the frontier average, first
average the three temperature differences within each seed, then compute SE
across the eight seeds; the 24 values are not independent replicates.

| T | A0 | A1 | A2 | Paired A2 - A1, mean +/- SE | Entropy A1 / A2 | MAUVE A1 / A2 |
|---|---:|---:|---:|---:|---:|---:|
| 0.8 | 13.397313 | 13.603587 | 13.819600 | +0.216013 +/- 0.077312 | 3.949281 / 3.945324 | 0.921179 / 0.919429 |
| 0.9 | 14.341506 | 14.793709 | 14.842592 | +0.048883 +/- 0.100007 | 3.954590 / 3.955186 | 0.909382 / 0.932138 |
| 1.0 | 16.021221 | 16.560449 | 16.228253 | -0.332196 +/- 0.122250 | 3.961931 / 3.963821 | 0.896268 / 0.930031 |

The paired frontier average is **-0.022433 +/- 0.075626**. TinyStories-33M's
A2-A1 means are +0.418767, +0.278728, -0.243656. A2-A0 means are +0.422287,
+0.501086, +0.207032. Entropy differences are -0.003957, +0.000596, +0.001890;
MAUVE differences are -0.001751, +0.022756, +0.033763. All agree with R039's
reported precision and inequalities. The frontier reading remains "trade-off."

| Oracle j | A0 | A1 | A2 | Paired A2 - A1, mean +/- SE | Paired A1 - A0, mean +/- SE |
|---|---:|---:|---:|---:|---:|
| -1 | 15.963326 | 16.655482 | 16.519350 | -0.136132 +/- 0.094423 | +0.692156 +/- 0.105379 |
| 3 | 12.380133 | 12.384100 | 12.800359 | +0.416259 +/- 0.164584 | +0.003966 +/- 0.093372 |
| 5 | 9.514811 | 9.429094 | 9.536802 | +0.107707 +/- 0.028533 | -0.085717 +/- 0.013642 |
| 7 | 8.918435 | 8.866707 | 8.889443 | +0.022736 +/- 0.007749 | -0.051728 +/- 0.013534 |

**Disagreements with `log.md` at the R039 results commit:**

- At j=5, A0 is 9.514811, which rounds to **9.51**, not 9.52.
- At j=3, paired A2-A1 SE is 0.164584, which rounds to **0.16**, not 0.17.
  The latter appears both in the table and its exploratory paragraph.

All other table entries agree to their printed precision. The j=-1 gain is
1.4417 SE, agreeing with "1.4 SE"; A2-A1 reverses sign with real levels 0..5.
No substantive registered interpretation changes. The T=1 groups also match
the original wave-1 saved scores exactly, independently of the stored gate's
boolean: maximum absolute difference is 0.0 for every arm, with tolerance 0.01.
The diagnostic has eight frontier groups of 256 and four oracle groups of 256,
matching 2048/1024 sample totals.

The three result `wall_s` fields sum to 1899.716 seconds (31.662 minutes).
Provenance spans 00:08:03-00:40:55 UTC on 2026-10-25 (32.867 minutes), all at
`81a697970b9400a2b82393c4f337d66f2f2b2746`, consistent with R039's approximate
33-minute runtime and cited source. The JSONs do not establish billed CU,
cumulative CU spent, the 200-CU budget, or completion of seeds 1-2; those claims
are not independently recomputable from these diagnostic results.

## Execution and limits

The default sandbox could not launch the repository's Python. Authorized host
execution could, using the existing project environment with CPU PyTorch 2.6.0
and OmegaConf 2.3.1; the missing pytest dependency was installed there. Tests use
four CPU threads through the unchanged `tests/conftest.py`. The released assets
were checked against `tests/audit_assets.json`. No downloads of model weights,
GPU training, or full-size mechanism evaluation were performed.

Final combined validation: **144 passed, 3 strict xfailed, no skips**, in
164.10 seconds. This includes all 77 existing W2M cases, the text-trainer and
notebook provenance regressions, and audit 8's **30 passing / 3 strict-xfail**
cases. The final run includes the frozen-copy storage probe. The expected
failures are the two MAUVE cache cases (A8-1) and canonical TF32 boundary (A8-2).

Commands (from the audit checkout; the final validation combined these lists):

```sh
python -m pytest tests/test_w2m_arms.py tests/test_w2m_old_arms.py tests/test_w2m_train_loop.py tests/test_w2m_pr12.py -q
python -m pytest tests/test_audit8_precision.py tests/test_audit8_protocol.py tests/test_audit8_numbers.py -q -rx
python -m pytest tests/test_audit3_protocol.py tests/test_notebook_provenance.py -q
```

The source audit also checked the registered configs, notebook collection,
shared precision/resume helpers and PR 12 patches. Proposed fixes were tested
only in memory and not written to production. The three expected failures use
`xfail(strict=True, raises=AssertionError)`, so imports and runtime errors do
not count as a demonstrated defect. GPU numerical equivalence and torch 2.11
behavior remain unverified here.

## Design observations, not findings

As registered, one training seed is descriptive. Paired sample-seed SE estimates
generation noise, not training-seed uncertainty. A flag rescue can reflect
regime separation as well as positional detection, and common corruption
contexts support comparison without settling causality. W2M already states
these limits and requires a new registration and deciding-arm replications
before criterion (c).

## Finding summary

| ID | Priority | File | Issue |
|---|---|---|---|
| A8-1 | P1 | `scripts/train_gen.py:329` | W2M MAUVE uses bf16 features and a legacy precision-agnostic reference cache. |
| A8-2 | P1 | `src/ncmr/w2m.py:183` | Canonical target encoding disables autocast but inherits enabled TF32. |
