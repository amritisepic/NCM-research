# Deep audit 9: D1, E1 and W2Ma before launch

Audited revision: `dcdb89092bccb967402461bda297337e99587400`, the supplied
`claude/wip-e1-d1` head for PR 14. Publication branch: `codex/audit9`, created
directly from that revision. Audit date: 2026-10-27.

This document is the run record. The explicit publication restriction permits
only this report and `tests/test_audit9_*.py`, so `log.md` is not edited.
Production code, registrations, notebooks and earlier tests are unchanged.

## Verdict

**No P0 or P1 defect found in the requested evaluation paths.** The three
issues accepted in `docs/reviews/triage_pr14.md` are fixed at the audited
revision: direct eager/cached comparisons, deterministic repeats, and the
executed tokenizer-decode count at a fixed point. The existing 61 subject
tests pass. Additional audit tests exercise sampled histories, exception
cleanup, timing boundaries, the main-program integrity gate, released-size
WMT14 generator settings, notebook wiring and independent result arithmetic.

This is a source and CPU audit, not a measured D1 result or a full released-WMT14
checkpoint certification. The released Sudoku weights are checked through the
unchanged `released_sudoku` fixture and `tests/audit_assets.json`. The WMT14
configuration was read from the public release, but downloading its weights
failed for lack of disk space; the incomplete download was removed. Its
configuration SHA-256 was
`6401af2a14d41c6582c3181a0b683e8970c1eeefe8cf7afaf74f1597da3fcd72`.
The full-size WMT14 test therefore uses random weights. A100 execution and
D1's saved-output integrity checks remain necessary before reading its gate.

There are no defect tests to mark `xfail(strict=True)`, and no production
patches to propose. The added tests are positive audit checks. Small log
arithmetic and wording disagreements are listed below, without promoting them
to launch-blocking code findings.

## Answers to the audit questions

### 1. Cached functions and the upstream implementation

`src/ncmr/cached_decode.py:58`, `src/ncmr/cached_decode.py:91`,
`src/ncmr/cached_decode.py:114` and `src/ncmr/cached_decode.py:145` implement
the same supported functions as the eager paths on inspection and in the
executed comparisons:

- Prefix rows attend only to prefix rows. RoPE is applied before keys enter
  the cache; target calls supply the per-level or full target table, and
  upstream `SelfAttention` starts its rotary slice at the cached length P.
  It does not rotate the stored prefix keys again. See
  `src/ncmr/ncm_ref/models/common.py:103` and
  `src/ncmr/ncm_ref/models/next_concept.py:94`.
- The prefix pad bias is broadcast to every prefix and target query row,
  including the target-only sequential calls. This avoids upstream
  `generate(use_kv_cache=True)`'s missing pad mask. The benchmark uses
  `use_kv_cache=False` for its eager baseline.
- WMT14 level 0 uses the same `_level0_slice` with the pad mask, rather than
  position 95 irrespective of source length. Head indices, level-encoding
  offsets and per-level rotary tables use `_lvl_first` consistently.
- `cached_parallel_logits` calls `_pad_vocab` for level 0 and every remaining
  head, as `_run_stack` does. The sequential and ancestral calls use each
  head's native vocabulary, as upstream sequential generation does. All
  released WMT14 heads have 4096 entries, including level 0.
- `finally` clears the model's temporary attention caching state after both
  normal completion and injected exceptions. The retained `PrefixCache.kv`
  tensors survive that cleanup; later calls concatenate fresh target keys
  without modifying those retained tensors.

The full-size random WMT14 generator has 12 layers, width 768, 12 heads,
4096-way untied heads, QK norm, RoPE base 100000, the released 11-level layout,
a 96-token prefix and pad id 151643. Its VQ uses 384-dimensional codebooks and
the released learned-L0 cascade/residual-convolution settings. Unused tokenizer
stacks are omitted. On a source with 13 real tokens and 83 pads, cached and
eager sequential codes agree exactly; parallel logits have maximum absolute
difference **1.1324882507324219e-6**. The original suite also tests several
source lengths and pad-embedding perturbations on small models. Released
Sudoku sequential/parallel/stop comparisons pass with hash-verified weights.

No difference beyond float rounding was demonstrated. Actual released WMT14
and TinyStories weight equivalence, GPU kernels and rare greedy near-ties
remain outside the performed validation. The small TinyStories tests cover
its given level 0, one-token prefix and tied heads.

### 2. D1 timing and comparison fairness

The timed functions in `scripts/bench_batch1.py:80` and
`scripts/bench_batch1.py:98` start with the token prefix already on the device
and include prefix embedding, prefix-cache construction when selected,
generation, tokenizer reconstruction and the required host output conversion.
WMT14 detokenization is inside these functions. Stable-text decoding runs
inside each pass. `memo` reuses the last decoded result at a stable, verifier
or fixed-point stop; the final output is not decoded twice.

`run_timing` at `scripts/bench_batch1.py:123` excludes warm-up, synchronizes
before and after every timed invocation, interleaves configurations in an
item/repeat-seeded random order and takes the median of three repeats. An
independent event-trace test checks the complete call order and timing
boundaries in both precision contexts. The production main chooses the
minimum **mean item latency** over eager/cached implementations independently
for the sequential baseline and each Jacobi configuration
(`scripts/bench_batch1.py:428`). A main-program test verifies this selection.

Both loaded models and the shared tokenizer have gradients disabled before
timing. Thus the secondary bf16 run does not give one configuration a special
persistent autocast weight-cast cache: frozen weights do not qualify for that
leaf/requires-grad cache, and the two models inherit the same freezing policy.
The fp32 context is explicitly disabled-autocast; both TF32 flags are disabled.
Warm-up shares the same precision scope as measurement, and that scope exits
normally. No leaked or re-entered context was found.

These observations do not establish an A100 latency ratio. In particular,
calling the result a pass on CPU, another GPU or bf16 would ignore the explicit
qualification attached by the script. Only the registered fp32 A100 reading
can trigger J1d.

### 3. Integrity checks and wrong-checkpoint detection

The sources are the same first test items, padded to 96 as
`train_wmt14_L.evaluate` does. Saved batch-128 exact text and per-item NFE are
compared with batch-1 exact text and NFE. Stable outputs are compared with the
stable rule applied to each arm's own saved capped trajectory; this does not
compare a stable output with an unrelated exact output. Exact stopping uses
WMT14's pass-1 detection and leaves it off for Sudoku. Online and saved stable
rules agree in the original suite's synthetic fixed-point/stable/cap cases.

At `scripts/bench_batch1.py:287`, eager and cached implementations are compared
directly on outputs and passes in addition to their saved-output comparisons.
At `scripts/bench_batch1.py:320`, any nondeterministic item blocks the reading.
The main additionally requires the validation rule-reproduction boolean at
`scripts/bench_batch1.py:426`; a deliberately false boolean produces
`not read (integrity)` even when the mocked latency ratio is 0.5.

The added main-program test substitutes J0's saved stable trajectory for J2-F
on both implementations. **253 of the first 256** output/pass pairs differ
from J2-F's saved rule; the gate is not read. This establishes detection of a
concrete wrong checkpoint, not a guarantee against every possible weight file
with observationally identical outputs. The actual checkpoint digest is saved.
The registered two-item tolerance remains a per-comparison allowance; it is
not extended to repeats. The triage's disjoint eager/cached mismatch test passes.

### 4. FLOPs and rows

`generator_block_flops` at `scripts/bench_batch1.py:185` uses
`block_fwd(d,L,q,k) = L*q*(24*d*d + 4*d*k)` with dense key lengths. It prices
cached target rows against P plus target keys and includes one prefix-only
pass. WMT14's sequential level-0 prefix-only call is present. Its generator
row counts are:

| Configuration | Generator rows per item |
|---|---:|
| WMT14 sequential eager | 1368 |
| WMT14 sequential cached | 408 |
| WMT14 Jacobi eager, t passes | 408 t |
| WMT14 Jacobi cached, t passes | 96 + 312 t |

These are row counts, not an attention-inclusive FLOP estimate by themselves.
The separate executed-cost units come from `FlopCounterMode` with math
attention and disabled autocast, including heads and VQ/tokenizer convolutions.
Sequential units count the entire sequential function. Cached Jacobi units
add the prefix once and count the cached pass t times. Both eager and cached
parallel passes execute their level-0 head as counted.

`decodes_made` at `scripts/bench_batch1.py:208` correctly charges t-1 tokenizer
decodes for a stable/verifier fixed point at t > 1, and t otherwise; exact and
sequential output decoding is once. This matches the actual `memo` path.
The original whole-configuration FLOP test and the independent block-convention
test pass. FLOPs are reported for the calls made, rather than treating logical
NFE as saved total compute. They do not claim to count every scalar operation
or memory transfer; the registered matmul/attention/convolution convention is
the stated unit.

### 5. E1 distribution, quantities and pairing

`sample_from_cut` at `src/ncmr/cached_decode.py:243` builds the canonical
coarse reconstruction, computes the first level's shared conditional logits,
expands the prefix cache and **clones** the expanded reconstruction before
updating it. Every subsequent call conditions on the previously sampled
coarser codes. The Gumbel-max expression has the correct sign and T=1.

The added independent oracle uses full `parallel_logits` calls on each sampled
history and independently draws each level's Gumbels. It agrees exactly with
the cached sampler's codes and RNG state for TinyStories-like and padded
96-token WMT14-like models, and for the actual pinned Sudoku model. It checks
that canonical codes and stored prefix keys/values are unchanged. This extends
the original low-temperature and teacher-forced-logit tests to stochastic
ancestral trajectories and multiple expanded rows.

`Domain.level_nll` at `scripts/e1_equivalence.py:74` reads canonical
teacher-forced fp32 log probabilities, with TinyStories' released unigram for
level 0. Suffix and prefix sums give N_k and A_k. `summarise` and `paired` at
`scripts/e1_equivalence.py:134` and `scripts/e1_equivalence.py:164` implement
the registered T/R/U decomposition, SE over items, ratio of mean R to mean N,
zero-hit R bounds, and paired mean dN/dT. The zero-hit upper-R calculation
clamps its lower T bound to N, as the canonical suffix is itself a possible
hit. Invalid Sudoku reconstructions receive zero samples and are excluded,
explicitly recorded in the result protocol and PR14 triage.

The WMT14 key truncates at the first EOS and drops pads. TinyStories compares
all 128 token ids; Sudoku compares parsed boards. Reconstruction exactness is
saved per item. Model weights are restored between models; items and tokenizer
stay fixed. Seeds depend on item and cut, not model; round sizes and vocabulary
shapes match across models. Earlier stopping uses a prefix of the same noise
stream. WMT14 reads `val_src` and `val_tgt` from the prepared validation arrays,
including their <=63-target-token eligibility restriction, rather than decoding
newstest2014 sources or inventing a target format. No test-set tuning was found.

### 6. W2Ma

`scripts/mech_eval_w2m.py:47` and `scripts/mech_eval_w2m.py:66` build plausible-
wrong and uniform-wrong row contexts once from the released proposer, with the
same random clean rows and slot masks. Exactly half the 512 rows are clean.
The label is `~plan.clean`; the assertions reject masked clean rows and
nominally corrupted rows without a masked slot. The contexts and perturbations
are reused for every scored model.

`pool_level` and `row_probe_aucs` at `scripts/mech_eval_w2m.py:80` and
`scripts/mech_eval_w2m.py:86` slice the target layout for levels 1..6, average
post-ln_f hidden states over positions, train on windows 0..383 and score AUC
on 384..511. `fit_probe_auc` standardizes from training features only. The
uniform-wrong control is too weak exactly when no level reaches 0.75. Flags
are zero for these hidden-state forwards. All these details are exercised by
the 16 parameterized row-probe tests in the original subject suite.

The pooled probe is preserved under `probe_pooled` with `not read` status.
The calls implementing (a) context KL and (c) true-minus-zero flag NLL, their
contexts and their held-out flag-use slice are unchanged by this amendment.
The additional row-context construction occurs before the model loop and does
not replace those contexts. Main disables both TF32 flags; canonical targets,
proposals and hidden-state forwards use the existing fp32 boundaries. No
training-arm change was audited or made.

### 7. Result recomputation

Inputs are the audited commit's result JSONs. `tests/test_audit9_numbers.py`
pins the normalized Git-blob SHA-256 digests of all seven inputs and implements
its own stable rule, independently of `stable_stop`.

**R042, all 3000 validation items per arm:**

| Arm | Exact mean passes | Stable x1 | Stable x2 | x1 output equals exact |
|---|---:|---:|---:|---:|
| J0 | 10.916000000 | 9.024000000 | 10.045000000 | 88.2667% |
| J1 | 11.527333333 | 9.759000000 | 10.707333333 | 92.1667% |
| J2 | 9.357000000 | 7.130666667 | 8.313000000 | 81.0333% |
| J2-F | 10.284666667 | 7.799333333 | 9.083333333 | 80.0333% |

Both stable-rule means match `results/j2f_stop_rule_val_s0.json` exactly.
Relative to sequential's 11 calls, x1 savings are 17.963636%, 11.281818%,
35.175758% and 29.096970%, respectively. The reported R042 values agree at
their printed precision. newstest2013 references are absent from this audit
environment, so BLEU was **not** independently recomputed. The stored BLEU
numbers are not treated as fresh measurements.

**R045, from `results/s0_neighbour_gain.json`:**

| Level | Canonical NLL +/- SE | Gain +/- SE | Gain / NLL | Gain per code | Epochs / best |
|---|---:|---:|---:|---:|---:|
| 1 | 55.900823 +/- 0.206450 | 3.241912 +/- 0.113836 | 5.799399% | 0.405239 | 9 / 6 |
| 2 | 80.686761 +/- 0.421018 | 1.541931 +/- 0.060211 | 1.911009% | 0.096371 | 5 / 2 |
| 3 | 84.818365 +/- 0.625899 | 0.769992 +/- 0.051444 | 0.907813% | 0.032083 | 5 / 2 |
| 4 | 77.967134 +/- 0.768091 | 0.494431 +/- 0.029759 | 0.634153% | 0.015451 | 4 / 1 |
| 5 | 66.119236 +/- 0.825250 | 0.169106 +/- 0.013812 | 0.255759% | 0.004228 | 4 / 1 |
| 6 | 64.081181 +/- 0.681220 | 0.256994 +/- 0.021756 | 0.401044% | 0.004942 | 4 / 1 |

For levels 1..5, summing the level means gives base 365.492319000885 and gain
6.217371253297. The saved per-window float32 aggregate base is
365.492318987846, a 1.3e-8 rounding difference; its saved paired gain SE is
0.148914789608. Gain share is 1.701094915076%. The gate fails both the
10-nat and 5% requirements (5% requires 18.274616 nats). Saved runtime is
280.1 seconds and head reproduction error is 0.0, agreeing with R045's
rounded descriptions. Epoch minima, per-code gains, percentages and differences
of NLL means recompute from the JSON. Underlying per-window vectors are not
saved there, so the listed SEs and actual training runtime cannot be
independently re-estimated from raw observations.

**R043, from `results/w2m_flag_power.json`:**

| Quantity | Recomputed/saved value |
|---|---:|
| Corrupted rows | 4096 - 1024 = 3072 |
| L0-replaced corrupted rows | 2324 / 3072 = 75.651042% |
| Flags equal 1, all corrupted-row slots | 99.582976% |
| Flags equal 1 at level 2 | 96.936035% |
| Mean absolute truthful/shuffled difference, corrupted rows | 0.001437929 |
| Same difference, all rows | 0.001078447 |
| Contaminated probe positions | 99.683607% |
| Negatives per 640-position window | 2.024902 |
| Negatives situated at level 1 | 91.439593% |
| AUC, indicator of not-level-1 | 0.952313721 |
| Same AUC, L0-clean windows | 0.957203984 |

The existing simulation tests execute the mask/footprint/shuffle calculations
and reproduce the JSON to their stated tolerances. The simulation is explicitly
a mask-geometry calculation, rather than measured tokenizer-weight behavior.
Its no-clean-row probe variant intentionally omits the clean-row key draw,
as documented in `scripts/w2m_flag_power.py`; it has the same distribution,
not the identical RNG stream of the mechanism script. Its two-score level-1
indicator is a sufficient confounding example. Using every level's contamination
rate gives still higher saved AUCs, 0.992868841 and 0.998162806.

**Disagreements with the log:**

1. `log.md:992`: L0 replacement 75.651042% rounds to **75.7%**, not 75.6%.
   This was already acknowledged in R045 and the flag-power script. Level 1's
   equal-1 slot percentage inherits that same rounding correction.
2. `log.md:1033`: level 4 gain per code is 0.015450958, which rounds to
   **0.015** at three decimals, not 0.016.
3. `log.md:1040`: "only level 1 ... gains > 1 nat per window" is false:
   **level 2 also gains 1.541931 nats**. The preceding table already shows it.

No gate verdict changes. Other table entries agree at their displayed
precision. No Colab billing or cumulative-budget claim is inferable from
these three result files.

### 8. Notebooks and saved evidence

The three committed notebooks equal `make_notebooks.make` exactly. The audit
checks that correspondence without rewriting generated files.

- D1 invokes WMT14 with `L_wmt_consist_frozen_s0/final.pt` from the common run
  root and the prepared validation-text JSON, plus the Sudoku benchmark.
  Defaults supply n=256, 16 warm-up items, three repeats, seed 0 and
  fp32/bf16. The notebook specifies A100 for its gate.
- E1 invokes TinyStories A0/A1/A2/A4 on 64 items, WMT14 J0/J1/J2F on 128
  prepared validation pairs, and released Sudoku on 128 puzzles. Each command
  uses rounds of 64 and eight descriptive coarse-cut items. Script defaults
  give cuts 5..K-1, full cuts from 8, 2048 samples, 20 hits and seed 0.
  The checkpoint directory names agree with the registered seed-0 runs.
- W2Ma invokes the mechanism script with seed 0; its defaults give 512 windows,
  a 384-window probe training split and evaluation seed 2026. The notebook
  documents the `--only` change if M5 was skipped; without that user edit,
  the default command still requires every checkpoint.

The shared collection cell copies each run's `result.json` and provenance.
E1 results contain per-item level NLL, reconstruction exactness and hit/sample
counts, so the registered summaries and paired differences are recoverable.
D1 saves latency and pass vectors, stop histograms, counts, checkpoint identity
and integrity disagreements. Mechanism results contain both row-control AUCs
per model/level, the too-weak boolean, unread pooled probe, KL and flag-use NLL.
The notebooks rely on already prepared WMT14 arrays and existing training
checkpoints on the common Drive; they do not retrain those models. No notebook
was launched and no PR was merged during this audit.

## Execution and limitations

The default shell could not launch the project's Python. Host execution used
the existing Python 3.11 environment with PyTorch 2.6.0+cu124, on CPU; CUDA
initialization was unavailable. `tests/conftest.py` sets four CPU threads.
The local Sudoku assets passed the unchanged SHA-256 checks. The default
bundled Python was available for standard-library JSON arithmetic, but lacked
PyTorch. No new Python package installation was needed.

Executed commands, from this branch:

```sh
python -m pytest tests/test_d1_bench.py tests/test_e1_equivalence.py tests/test_mech_eval_rowprobe.py tests/test_w2m_flag_power.py tests/test_s0_probe.py -q
python -m pytest tests/test_audit9_numbers.py tests/test_audit9_cached.py tests/test_audit9_protocol.py -q
```

The subject suite passed **61 tests** in 323.07 seconds. The added audit suite passed **19 tests** in 32.50 seconds, with no skips
or expected failures: **80 passing tests** across the two final runs. A preliminary audit-only run had
one test-harness setup error: the tokenizer's `n_layers` argument is a head
count and cannot be zero. The test setup was corrected to the released value
6; this was not a production defect. The corrected full-size WMT14 test passed
in 25.71 seconds. Existing one-item paired-SE tests emit NumPy ddof warnings;
registered model comparisons use many items.

Full released WMT14/TinyStories weights, GPU numerical behavior, GPU timing,
Colab's PyTorch 2.11 behavior and live Drive checkpoint availability were not
validated. A successful source/CPU audit cannot authorize reading an integrity-
failing D1 result, spending J1d compute before D1 passes, or treating E1's
single training-seed comparisons as a replicated training result.

## Finding summary

| ID | Priority | File | One-line issue |
|---|---|---|---|
| None | — | Requested evaluation scope | No P0/P1 defect demonstrated; execution limits and log disagreements are documented above. |
