# Deep audit 5: W1d re-quantized-label diagnostic

Subject: `4df7d13197c32993398398fcf9e53b563b50e7e1` (PR 7 head).
Date: 2026-10-24. Scope: the five requested files, their sampling/scoring
dependencies, W1d, and the R020/R023/R030/R031 results used to motivate it.
This report is the audit run record; the explicit file restriction leaves
`log.md` untouched. No production changes or proposed patches are applied.

## Verdict and verification limits

**No P0/P1 defect was established in the registered notebook's default path.**
One P2 input-validation defect remains in the reproduction helper (A5-1).
The inspected code implements the specified frontier, oracle sampling kernel,
precision, checkpoint loading and paired output statistics. This is a source
audit and independent result-arithmetic check, not a successful A100 replay.

**Python tests could not execute in this environment.** The available virtual
environment's launcher points to an unavailable base interpreter; Python is
also absent from the command search path. An attempted temporary standalone
runtime download failed to connect. Therefore there is no claimed pytest pass,
observed XFAIL, model-forward measurement, or GPU reproduction in this audit.
The newly supplied strict xfails need execution before treating them as
experimentally demonstrated regressions. The saved wave-1 `final.pt` files and
the full TinyStories model/scorer assets are not in this checkout, so their
actual weights were not independently loaded or rescored.

The checks that did execute used PowerShell JSON arithmetic, Git source
inspection and notebook JSON parsing. All 20 generated notebooks preserve the
template record and PR 3 guards. Historical result arithmetic agrees except
for R020 slips already corrected in R023, listed below.

## A5-1 - P2: reproduction silently drops requested groups

**Locations:** `scripts/diag_w1_frontier.py:85`,
`scripts/diag_w1_frontier.py:87`, `scripts/diag_w1_frontier.py:91`;
group construction at `src/ncmr/eval_text.py:138`.

`dict(zip(sorted(seeds), per_group))` truncates to the shorter input and
collapses duplicate seeds. Missing seeds are then searched only in that
already-truncated dictionary. Consequently, with reference seed 0 scoring
10.0, `reproduction([10.0], [0, 999], expect)` returns `ok=True` even though
requested seed 999 has neither a score nor a reference. `[0, 1]` with just
the seed-0 score also passes despite missing a requested current group.
`reproduction([10.0], [0, 0], expect)` accepts duplicate requested seeds.
These conclusions follow directly from evaluating the Python expressions;
they were not executed with a Python interpreter in this environment.

There is also a CLI route to duplicate groups: `--seeds 0 1 2 3 4 5 6 6`.
`evaluate_samples` produces seven distinct groups, while the helper accepts
the seven resulting scores against a truncated seed list. Duplicating the
same 256 texts preserves the last group's corpus score up to rounding, so
this can report reproduction without eight independent sampling groups.

**Why P2:** this needs overridden arguments or a malformed helper input.
The notebook requests eight unique seeds, `draw` returns every sample, and
`evaluate_samples` returns one score per distinct group; no default-path route
to a missing score was found. This does not establish a W1d launch blocker.
It does weaken the helper's documented guarantee to check every requested
seed, and makes a future dropped-group regression fail open.

**Test:** `tests/test_audit5_reproduction.py` supplies missing current groups,
an unmatched requested seed hidden by truncation, and duplicate seeds. Three
cases have `xfail(strict=True, raises=AssertionError)`. Import or fixture errors
cannot count as expected failures. Passing controls cover unique seed
reordering, NaN, infinity and an excessive score difference. These are pure
score-matcher tests; released model weights would add no coverage. They use
normal `tests/conftest.py` discovery without downloading or substituting assets.

**Proposed patch, not applied:**

```diff
diff --git a/scripts/diag_w1_frontier.py b/scripts/diag_w1_frontier.py
--- a/scripts/diag_w1_frontier.py
+++ b/scripts/diag_w1_frontier.py
@@
     ours = dict(zip(sorted(seeds), per_group))
+    valid_groups = len(per_group) == len(seeds) == len(set(seeds))
     pairs = {s: [ours[s], theirs[s]] for s in sorted(ours) if s in theirs}
-    missing = sorted(s for s in ours if s not in theirs)
+    missing = sorted(set(seeds) - set(theirs))
@@
     return {"pairs": pairs, "missing_seeds": missing, "max_abs_diff": diff, "tol": tol,
-            "ok": bool(pairs) and not missing and finite and diff <= tol}
+            "valid_groups": valid_groups,
+            "ok": valid_groups and bool(pairs) and not missing and finite and diff <= tol}
```

## Answers to the six audit questions

### 1. Frontier and reproduction gate

`scripts/diag_w1_frontier.py:38` through the decoder is the same sampling
procedure as `scripts/train_gen.py:221`: a dedicated generator for L0
multinomial draws, global Torch RNG seeded once per group for `net.generate`,
256-row batches, identical EOS prefix and generation arguments, and identical
decoder/first-EOS/PAD processing. Re-encoding at diagnostic line 161 matches
`train_gen.py:260`. The intended temperature argument is the only sampling
difference. There is no separate lower-temperature branch that could pass
T=1 and switch samplers at T=0.8/0.9. The helper assumes evaluation mode; main
sets it before any draw, matching the reference sampler's evaluation mode.

Reordering unique seeds is safe: `evaluate_samples` sorts group labels and
the matcher sorts the requested/reference seed lists. A5-1 concerns missing
scores and duplicate seeds, not reordering. The default eight seeds all have
wave-1 reference groups. Nonfinite matched values or a difference above 0.01
make `ok=False`; main writes `reproduction_failed` and exits nonzero before
lower temperatures or oracle sampling (`scripts/diag_w1_frontier.py:155`,
line 167). The notebook stops after a nonzero command.

**The check tests numerical agreement, not identity.** A wrong checkpoint,
different sampler or different GPU can pass if every checked Gen-PPL happens
to agree within 0.01. Neither GPU type nor an expected checkpoint hash is
compared. The loaded checkpoint's digest and current GPU are recorded, but
the expected result has no trusted checkpoint digest to match. This is a
limit of the registered numerical rule, not evidence that a particular
substituted model passes. The three notebook commands supply `--expect`;
the standalone CLI permits omitting it and overriding tolerance/settings.
Such an invocation cannot be certified as the registered W1d run merely
because the script finishes.

"T=1 first" means first sampling/evaluation temperature: references are
featurized before that loop, and T=1's second scorer and MAUVE run before its
reproduction verdict. A failure still prevents T=0.8/0.9 and the oracle.

### 2. Precision and MAUVE cache

On the standalone main path all floating model forwards use fp32 weights
and TF32 is disabled for CUDA matmul and cuDNN at diagnostic line 123.
Generator inference, tokenizer prefix encoding/VQ and decoder are inside
fresh disabled-autocast contexts (lines 45 and 60). The oracle canonical
encoder and all generator passes share that boundary. No reusable context
object is re-entered, and both contexts exit before scoring.

Both scorer models are loaded with `torch_dtype=torch.float32`
(`src/ncmr/eval_text.py:78`). `gen_ppl` and upstream
`generation_perplexity` do not enable autocast; main has no enclosing AMP
context. Thus both Gen-PPL scorers run in fp32 on this path. The shared
`gen_ppl` helper is not itself a guard against arbitrary external callers'
autocast; that does not introduce AMP into this standalone command.

Reference features explicitly use `amp=False` (diagnostic line 150), and
sample features use `mauve_amp=False` (line 163), forwarded at
`src/ncmr/eval_text.py:148`. Each featurizer batch creates its own disabled
autocast context at eval-text line 110. The default cache is
`runs/cache/mauve_ref_val_2048_fp32.npy`; it cannot accidentally reuse W1b's
`mauve_ref_val_2048.npy` or R020's `mauve_ref_test_512.npy`. Five quantization
seeds are used. References come from `examples("val", 2048)`, whose
`src/ncmr/data.py:46` split selects even-index validation stories.

The cache name is a precision/split/count distinction, not content-hash or
GPU provenance. No evidence of a wrongly populated fp32 cache was available.
W1b's historical MAUVE used AMP; its numerical value should not be expected
to reproduce with W1d's new fp32/2048 protocol.

### 3. Oracle equivalence and R020 comparability

`scripts/diag_w1_frontier.py:57` matches
`scripts/tf_levels_diag.py:57` in level lengths, cumulative offsets,
canonical prefix levels 0..j, L0 unigram when j=-1, EOS embedding, full
block-diagonal forwards, logit slices, uniform tensor shapes, clamping,
Gumbel-max sign and draw order. Division by temperature is exactly division
by 1 here. Both sample in batches of 128, strip PAD/cut EOS, re-encode, and
use a single generator per j seeded `seed * 100 + j + 1` (default j+1).
Each model receives the same first 1024 validation windows and per-j seeds.

Differences from the historical R020 reported curve are:

| Aspect | R020 | W1d |
| --- | --- | --- |
| Windows | First 512 val windows | First 1024 val windows |
| Primary score | One pooled corpus Gen-PPL | Mean of four 256-window corpus Gen-PPLs |
| Precision enforcement | No explicit TF32/autocast guards in that script | Explicit fp32 and TF32 off |
| j points | Every j from -1 through 11 | -1, 3, 5, 7 |
| Other oracle metrics | Both LMs and MAUVE; 512 test reference features with default AMP | GPT-2 large only; no oracle MAUVE |

The sampling batch size and seed formula do not change. On the same
arithmetic/runtime, the first 512 windows use the same random-draw prefix;
the new run adds four more sampling batches. A mean of group PPLs is not
the pooled PPL: exponentiation and unequal token counts prevent that
identity. Hardware/runtime and actual TF32 defaults of the old run are
not pinned by its oracle JSON. These facts prevent reading differences
between the historical R020 point and W1d A0 as model effects. Compare
A0/A1/A2 within W1d; R020's coarse/fine shares remain historical context.

### 4. Loading final.pt

For the specified released-init arms, main reconstructs the same released
generator/tokenizer configuration as `train_gen.py:95`, then calls
`net.load_state_dict(st["model"])` with default `strict=True`
(diagnostic line 132). This loads persistent buffers as well as parameters,
including position/level indices and the attention mask. Nonpersistent
buffers are rebuilt from the same released configuration; no relevant
training-time mutation of those was found. The tokenizer is frozen during
wave 1 and remains the separately loaded released tokenizer.

`train_gen.py:332` stores the generator state and L0 unigram before final
evaluation. Diagnostic line 133 restores that checkpoint's L0, not the
bundle's initial unigram. Generator and tokenizer are both in eval mode;
dropout is inactive, including SDPA's explicit training-mode check in
`src/ncmr/ncm_ref/models/common.py:129`. No wave-1 dropout/config change
outside the state dict was found. The saved `model_cfg` is irrelevant for
these released-init arms; scratch init is rejected.

This establishes source compatibility, not actual checkpoint identity or
observed equality on the absent final.pt files. Strict loading rejects
missing/unexpected tensors, not a different arm with compatible shapes;
the numerical gate and recorded digest provide separate evidence.

### 5. Notebook provenance and collection

`notebooks/make_notebooks.py:281` retains refusal of output `ckpt.pt` without
start provenance. Lines 287-293 compare today's code against **every** prior
session SHA over `src`, `scripts`, `configs`, `assets`; a nonzero git-diff
status also fails closed. The intentional `ALLOW_CODE_CHANGE` override is
unchanged. Replacing the stored command with its original template does
not change these checks.

Lines 296-299 store the template, seed and added-data/microbatch flags.
Collection at lines 315-323 still parses original `CMDS`, independently of
stored provenance. Every current run-directory template uses its supported
`{R}` and `{S}` keys. All 20 generated notebooks include the new record,
the all-session comparison and the unprovenanced-checkpoint guard.

W1d reads old wave-1 final.pt files as evaluation inputs; PR 3's guard
protects resuming an output checkpoint and does not establish those inputs'
training provenance. This does not regress PR 3's resume protection.
Templates avoid expanded paths in new `provenance.jsonl`/`_meta.json` records.
W1d `result.json` omits ckpt/run_dir/expect arguments and uses only checkpoint
basenames plus a digest; `samples_final.json` contains IDs/group labels.
For the supplied notebook commands, none of those new fields embeds an
absolute path. Arbitrary user strings or preexisting legacy provenance are
not sanitized; the change does not retroactively rewrite old files.

### 6. Reading rule and saved outputs

`result.json` is sufficient for the registered reading with default settings.
Both frontier scorers save per-group scores and args save the requested
seeds. Array order is `sorted(args.seeds)`, not sampling order. For each
temperature define `d[t,g] = PPL_A2[t,g] - PPL_A1[t,g]`; report its mean and
`sample_sd(d[t,:]) / sqrt(8)`. For the mean over temperatures, first compute
`q[g] = mean_t(d[t,g])`, then `mean(q)` and `sample_sd(q) / sqrt(8)`.
Treating all 24 temperature/group pairs as independent would discard the
shared-seed pairing and is not the requested analysis.

Oracle arrays similarly retain four GPT-2 group scores per j: compute
paired differences and `sample_sd / sqrt(4)`. Groups map to consecutive
window ranges 0-255, 256-511, 512-767, 768-1023. Pooled entropy and MAUVE
values support their point-estimate conditions. Five `mauve_per_seed`
values measure quantization variation, not sampling SE. No required
group-score field is missing. Oracle text IDs are not saved, which limits
later text inspection but does not block the registered group-statistic
reading. No W1d result or scientific verdict is inferred before the run.

## Recomputed historical numbers

Read the `*__result.json` files under the six `results/colab/w1_*/`
directories, `results/colab/w0_calib_diag/*.json`,
`results/w1_sample_stats.json`, and `results/onpolicy_repro/*.json`.
Curve/resume checks additionally used the uploaded metrics JSONL.
`tests/test_audit5_numbers.py` makes the central R030/W1d and corrected
oracle arithmetic repeatable without models. These positive tests are not
marked xfail: no arithmetic defect was found.

R030 group means and sampling SEs (`sample_sd / 4`), recomputed from all
16 stored groups:

| Arm | Gen-PPL | SE | MAUVE | Clean NLL | Entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 16.164383 | 0.112107 | 0.827189 | 861.838523 | 3.970222 |
| A1 | 16.604238 | 0.077857 | 0.881099 | 843.992656 | 3.962499 |
| A2 | 16.289748 | 0.092180 | 0.914693 | 749.253773 | 3.962645 |
| A3 | 41.436881 | 0.429714 | 0.495774 | 885.667102 | 4.026335 |
| A4 | 36.805781 | 0.255182 | 0.673064 | 707.404344 | 4.037413 |
| A5 | 32.168815 | 0.177945 | 0.543029 | 913.889391 | 3.989512 |
| A6 | 17.455435 | 0.102006 | 0.832429 | 888.888031 | 3.958134 |

All agree with R030's rounding. Clean NLL remains the historical AMP
measurement identified in audit 3; arithmetic agreement does not make it
fp32. Screen gains and entropy/MAUVE conditions reproduce the logged
verdicts, including A4/A5's nominal passes against A3, not A1.

R031/W1d motivation: A2-A1 = **-0.314490762 +/- 0.079560644** paired SE;
A2-A0 = **+0.125364698 +/- 0.075727871**; A2-A1 entropy = **+0.000145763**,
MAUVE = **+0.033593693**. A2 misses 0.35. A1's step-1080/1350/1800 curve
is 16.486431/16.705433/16.299354, confirming the step-confound rule does
not fire.

R020 oracle gap = **7.381254373**. Its recomputed shares are L0 **1.5272%**,
L1-3 **47.8083%**, L4-5 **38.5326%**, L1-5 **86.3409%**, L6 **4.8579%**,
L7-11 **7.2739%**, L6-11 **12.1319%**, and L8-10 **3.9637%**
(**0.292573868** Gen-PPL). R020's released reference is 16.354139,
group SD 0.513380, TinyStories PPL 12.005006, entropy 3.975937,
MAUVE 0.887641, quantization SD 0.021477, rep-2 0.100183. The oracle
JSON records 541.651687 seconds.

Other R020 stored metrics agree: Sudoku exact 6.406 passes, 46.6167%
logical saving and position-cost ratio 1.871059, solve rate 86.15%; verifier
2.3655/90.3%; stable 2.917/88.2%. TinyStories 11.939453 passes; WMT14
11.066600 exact and 10.588412 stable, BLEU 22.625980/22.639256.
The position ratio is a proxy, as corrected in R023, not measured dense
FLOPs or wall-clock speedup.

R031's recomputed NLL changes: A2-A0 -112.584750, of which L8-11
-99.249994 (88.1558%); A4-A0 -154.434180, of which L8-11 -150.749996
(97.6144%); A3-A0 +23.828578 with L3-6 +15.269997; A5-A0 +52.050867
with L8-10 +30.529999. Text-statistic claims also agree: JS A3/A5/A1
0.010811/0.007324/0.009391; bigram gain A1 2.131294 versus A3 1.714425;
adjacent token duplicates 6.913686 versus 17.706196 per 1k; position-band
deficits -0.325015/-0.430510/-0.400550/-0.445691/-0.441246.

R031's Sudoku CPU arrays agree with the logged paired estimates: identical
25.8604% replacements; uniform/plausible mean losses 0.307552/0.395267;
clean NLL 43.191834 -> 45.891323/75.207001; plausible L8-10 damage
+25.801001. Greedy changes -4.6667 +/- 1.4686 and -1.0000 +/- 1.5290
percentage points; contrast +3.6667 +/- 1.5872. Temperature-1 changes
-4.3333 +/- 1.5816 and -5.3333 +/- 1.6743; contrast -1.0000 +/- 1.9745.
Entropy 54.812400 -> 57.797900/95.199800 (+5.4468%/+73.6830%).
Re-quantized plausible corruption gives NLL +6.734981, entropy 66.504599
(+21.3313%), greedy -3.3333 +/- 1.5542 and sampled -4.3333 +/- 1.5816.

**Disagreements:** no new disagreement in R030/R031/W1d motivation. R020's
historical 87% for L1-5 should be 86.3%; 5% for L7-11 should be 7.3%;
approximately 0.2/3% for L8-10 should be 0.292574/4.0%. These are already
corrected by R023, not new untriaged findings. Hardware identities, charged
CU, unrecorded setup/restart time, historical test counts and software-fix
claims cannot be independently recomputed from result JSON; no new runtime
or cost measurement was made here.

## Running the supplied tests

With a working CPU Python environment and the existing dependencies:

```sh
python -B -m pytest -p no:cacheprovider tests/test_audit5_reproduction.py tests/test_audit5_numbers.py tests/test_w1d_diag.py tests/test_notebook_provenance.py -q -rx
```

Expected for the unpatched subject: **three strict xfails** for A5-1; the
other assertions should pass. This is an expectation, not an observed test
result. No network, model download or training is performed by the added
tests. Model-based follow-up should use `tests/conftest.py` and verify the
released assets against `tests/audit_assets.json`, as the existing fixture
does. Neither fake models nor an unavailable asset should be mistaken for
an observed replay of wave-1 checkpoints.

| ID | Priority | File | One-line issue |
| --- | --- | --- | --- |
| A5-1 | P2, nonblocking | `scripts/diag_w1_frontier.py:85` | Score/seed zip can drop requested groups or collapse duplicate seeds while reporting reproduction |
