# Deep audit 12: W2C stage 1 and the proposed claim

Audited commit: `fc2a7a53252e43d12d3c85d2febe003cbf381663`.
Branch: `codex/audit12`. Specification: W2C at
`docs/preregistration.md:589-616`, with W2M/W2Ma unchanged, and the explicit
definitions in `docs/prompts/codex_audit12_w2cclaim.md`. This audit covers
the six seed-1--3 runs and the seed-0 context quoted in R062, not the resume
path or the other experimental arms. Production code and results are unchanged.

## Assessment

**No P0 or P1 was demonstrated.** Executed independent arithmetic reproduces
R062's numerical results at its displayed precision. The registered readings
are **damage replicates** and **dissociation holds**. Executed CPU traces of
the production training loop support the within-seed matched comparison;
precision, group-order and validation-split checks also pass.

The proposed sentence needs a narrower interpretation before publication:
M2 causes **additional degradation relative to matched uniform-wrong M3**.
It does not establish that M3 causes no degradation relative to the released
model. W2C replicates a training-signal effect in one model/domain/protocol;
the localization to levels 2--5 is still exploratory seed-0 evidence. Meeting
W2C's registered candidate bar is not, by itself, a replicated account of
what particular levels encode under `docs/brief.md` criterion (c).

There are no new defect regressions to mark `xfail(strict=True)` and no
production fix diff to propose. The added tests are positive integrity checks,
including a counterexample to treating aggregate disagreement as a slot trace.

## Independent recomputation

`tests/test_audit12_numbers.py` reads hash-pinned raw result, sample and
provenance files. Its arithmetic uses Python's standard-library JSON,
`statistics` and direct counters, independently of `analyze_w2c.py`.
`tests/test_audit12_readings.py` separately compares those calculations with
the production analyzer and saved `results/w2c_analysis.json`. Text digests
pin Git's LF bytes while allowing Windows checkout newline conversion.

Gen-PPL is the mean of sixteen group perplexities, not the perplexity of one
pooled corpus. SE below is sample SD of sixteen **paired differences** divided
by four. In every seed, **16/16 differences are positive under each scorer**.
Seed 0 is shown only for context and never enters a seed-1--3 aggregate.

| Seed | GPT-2 large M2 | GPT-2 large M3 | Difference | Paired SE | TS-33M M2 | TS-33M M3 | Difference | Paired SE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.790341372 | 21.425413127 | 18.364928245 | 0.324665470 | 46.817954242 | 18.271583547 | 28.546370696 | 0.498391660 |
| 2 | 41.307911921 | 21.801928440 | 19.505983481 | 0.305636949 | 49.888702445 | 18.787211907 | 31.101490537 | 0.528939931 |
| 3 | 40.560382771 | 21.803127275 | 18.757255496 | 0.320420410 | 48.603866418 | 18.820796349 | 29.783070069 | 0.573873046 |
| Mean, 1--3 | 40.552878688 | 21.676822947 | 18.876055741 | -- | 48.436841035 | 18.626530601 | 29.810310434 | -- |
| 0, exploratory | 42.622016036 | 21.685751521 | 20.936264515 | 0.373978485 | 51.735472682 | 18.651327880 | 33.084144802 | 0.719039801 |

The pooled seed SD is **0.558212709953** and twice that is
**1.116425419906** (sample variances, ddof 1). The sample SD of the three
paired seed differences is a different quantity, **0.579730011699**.
The GPT-2 gap is **87.0794%** of M3's seed mean, reproducing R062's +87%.

| Seed | Clean NLL M2 | Clean NLL M3 | dNLL | MAUVE M2 | MAUVE M3 | dMAUVE | Entropy M2 | Entropy M3 | dEntropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 758.085195313 | 754.223429688 | 3.861765625 | 0.445764204 | 0.859527500 | -0.413763296 | 4.010441205 | 3.982220144 | 0.028221061 |
| 2 | 757.599726563 | 753.690023438 | 3.909703125 | 0.439256940 | 0.858211590 | -0.418954650 | 4.012648638 | 3.986551379 | 0.026097259 |
| 3 | 757.513132813 | 753.190617188 | 4.322515625 | 0.452723106 | 0.843830559 | -0.391107453 | 4.015322250 | 3.988406343 | 0.026915908 |
| Mean, 1--3 | 757.732684896 | 753.701356771 | 4.031328125 | 0.445914750 | 0.853856550 | -0.407941800 | 4.012804031 | 3.985725955 | 0.027078076 |
| 0, exploratory | 757.885382813 | 753.850031250 | 4.035351562 | 0.411807388 | 0.865397811 | -0.453590423 | 4.017823560 | 3.991797691 | 0.026025869 |

NLL units are nats/window; entropy units are nats. dNLL's seed SD is
**0.253312299531**. The mean difference is **0.534871%** of M3's clean NLL,
consistent with the quoted 0.5%. It has the same positive sign in all seeds;
the effect is small, not zero. Entropy was also independently recomputed by
decoding and re-encoding all eight sample sets with the hash-checked released
BPE vocabulary/merges, then counting each sample's token frequencies.

Saved per-level dNLL, M2 minus M3, is as follows. Entries are already rounded
to .01 by `scripts/train_gen.py:302` (represented as float32 JSON numbers).
They cannot reconstruct the independently accumulated full-precision total
exactly. Level 0 is a fixed prior, not a trained/scored target here.

| Level | Seed 1 | Seed 2 | Seed 3 | Mean, 1--3 | Seed 0 |
|---|---:|---:|---:|---:|---:|
| 1 | .42 | .43 | .41 | .420000 | .42 |
| 2 | .33 | .31 | .38 | .339999 | .33 |
| 3 | .14 | .15 | .28 | .190000 | .23 |
| 4 | .22 | .19 | .20 | .203334 | .29 |
| 5 | .41 | .47 | .47 | .450001 | .42 |
| 6 | .63 | .65 | .65 | .643333 | .57 |
| 7 | .54 | .53 | .50 | .523333 | .51 |
| 8 | .46 | .50 | .49 | .483335 | .47 |
| 9 | .16 | .15 | .23 | .180000 | .09 |
| 10 | .26 | .23 | .42 | .303329 | .38 |
| 11 | .29 | .29 | .29 | .290001 | .30 |

This reproduces R062's **seed-mean** .18--.64 per level, not that range in
every individual seed. Rounded level differences sum to 3.859991074,
3.900001526 and 4.319999695, versus the unrounded totals above. Seed 0 sums
to 4.010004044; its L1--L5 subtotal is 1.690002441 and its maximum individual
level difference is .569999695 at L6, reproducing the relevant R055 context.

For duplicates the registered source is `scripts/w1_sample_stats.py:76-98`
and `:139-141`: truncate each saved ID sequence to 128, count equal adjacent
**BPE IDs**, and divide by valid adjacent **pairs**, pooling over samples.
There is no cross-sample pair and no re-encoding. These saved IDs have already
been cut at EOS and PAD removed. The independent integer counts below match
`results/w2c1_sample_stats.json` and the seed-0 W2M statistics to numerical
roundoff. This is intentionally different from `eval_text.word_dups`, which
counts lowercased regex words per thousand words on decoded text.

| Seed | M2 duplicate count / pairs | M3 duplicate count / pairs | M2 / 1k | M3 / 1k | dDup / 1k |
|---|---|---|---:|---:|---:|
| 1 | 10558 / 515745 | 4203 / 519326 | 20.471356969 | 8.093182317 | 12.378174652 |
| 2 | 10546 / 515844 | 4236 / 519384 | 20.444165290 | 8.155815350 | 12.288349940 |
| 3 | 10721 / 515410 | 4462 / 519188 | 20.800915776 | 8.594189388 | 12.206726388 |
| Mean, 1--3 | -- | -- | 20.572146012 | 8.281062352 | 12.291083660 |
| 0, exploratory | 10964 / 515513 | 4310 / 519374 | 21.268134848 | 8.298451598 | 12.969683251 |

Provenance elapsed minutes: M2 **51.85 / 50.11 / 50.10**, M3
**30.32 / 28.32 / 28.33**, total **239.03 A100-minutes**. These reproduce
R062's one-decimal times and 239.0 total. At the stated 5.3 CU/hour this is
**21.1143 CU**, or **26.3929** after the stated approximate 1.25 multiplier.
Those CU figures are estimates, not independently measured billing or idle
runtime usage. Each run has one start/end record, `had_ckpt: false`, exit 0,
step 600, and the stated A100-SXM4-40GB. `_meta.json` agrees with the session
records. The relevant seed-0 M2/M3 elapsed times are 51.77/30.34 minutes.

**Recomputation disagreements:** none in R062's reported numbers at their
stated precision. The proposed wording's exact inequality "each seed >=
+18.4" is too strong: the minimum is **18.364928245**. Say "each seed at
least +18.36" or "about +18.4 or more." This is a rounding/wording correction,
not a changed registered reading. Differences between rounded level sums and
total NLL, or between the two duplicate definitions, are explained above and
are not hidden numerical disagreements.

## Registered readings and pairing

`scripts/analyze_w2c.py:48-88` implements paired sample SD with ddof 1,
pooled sample variances with ddof 1, every GPT-2 difference >=10, mean >2s_p,
every TS-33M and duplicate difference >0, overturn at mean <5 or TS mean <0
or MAUVE mean >0, |mean dNLL| <=10, and removed only at mean dNLL >20.
An overturn takes precedence over replication if both fire.
`scripts/analyze_w2c.py:167-207` loads only seeds 1,2,3 and requires all six
results and all six registered duplicate statistics before issuing a reading.

Every condition is satisfied here: minimum D=18.364928245 >=10,
18.876055741 >1.116425420, all TS/duplicate differences positive, no sign
reversal, and |4.031328125| <=10. The saved analysis agrees with independent
calculations. Existing `tests/test_r056_build.py` already tests all requested
threshold boundaries: 10 inclusive, 2s_p strict, TS/duplicate zero, mean 5,
MAUVE/TS reversal signs and precedence, +/-10 inclusive, +20 strict, and large
negative dNLL. No missing threshold test was identified. New tests explicitly
keep a tempting seed-0 input out of the registered pool.

The **sample blocks** are saved in `[42,0,...,14]` order, each 256 samples.
The **score vectors** are in **`[0,...,14,42]`** order because
`src/ncmr/eval_text.py:138-144` sorts the group keys. The same code/config
applies in all six runs; an executed group-labelled probe verifies that
ordering. Pairing score vectors by index is valid. Confusing these two orders
would mislabel individual groups, but does not affect this paired analysis.

## Matched training and replacement semantics

`scripts/train_gen.py:74-77` creates a dedicated NumPy data generator from
the training seed. `src/ncmr/w2m.py:151-153` creates distinct mask, replacement
and flag generators. `:171-185` draws exactly 25% clean rows by ranking
full-batch keys, then draws all mask uniforms from the mask generator.
`scripts/train_gen.py:405-411` draws a full batch and full mask plan before
slicing into four micro-batches. M2 and M3 configs are otherwise identical
apart from arm; all six saved configs equal the registered config after
setting `train.seed=s` and `grad_accum=4` (micro-batch 64, total batch 256).

The executed test instruments the **actual production training loop** for
four optimizer steps at each seed 1,2,3, with training dropout enabled. It
compares every batch, canonical target, clean-row index, full/micro mask and
the final data/mask/global dropout RNG states. They match exactly across arms.
Proposals leave the global Torch state unchanged. These are CPU/tiny-model
checks of the path and stream isolation, not a replay of the six GPU trainings.

All six saved `realized_disagreement` dictionaries match within each seed:
153600 training rows and 38400 clean rows, all twelve level entries equal
to their four-decimal reporting precision. This is consistent evidence,
**not sufficient evidence of identical slots**: `src/ncmr/w2m.py:264-282`
retains only aggregate counts. The new counterexample permutes corrupted rows'
masks, obtains different slots with exactly the same `realized()` output,
and passes. Matching slots is supported by code and executed traces, not
deduced from this lossy summary. Loss curves also cannot identify slots.
For context, their first/final losses are M2 2.3372/2.0543,
2.3628/2.0528, 2.3487/2.0403; M3 2.1986/2.1606,
2.2400/2.1696, 2.2128/2.1571. The differences are expected with different inputs.

**M3:** `src/ncmr/w2m.py:225-233` draws a uniform integer offset from
1,...,V_k-1 and adds it modulo V_k. This is a bijection onto every code except
the canonical one. An exhaustive offset-support test verifies that mapping;
no statistical frequency test is needed. The finest level is never masked.

**M2:** the frozen proposer is a deep copy of the **released** generator,
made before training/restoration (`scripts/train_gen.py:182-186`;
`src/ncmr/w2m.py:163-168`). Dropout is off and parameters frozen. At level 0,
draw from the released unigram after removing the example's canonical root
(`:189-196`); this root draw does not use `policy_temp`. At levels >=1,
the single parallel proposal forward sees the example's **canonical clean
coarser hierarchy**, not corrupted coarser levels (`:235-241`). It removes
the canonical logit, softmaxes at T=1, and draws from the remaining distribution
using only `repl_gen`. The proposal/teacher forwards are fp32 with TF32 off
under `fp32_eval`; they cannot advance data, mask, clean-row or training-dropout
streams. Training then rebuilds its inputs with the masked replacements.

M2's extra proposal forward really executes and is included in the analytic
generator-only FLOP proxy (`scripts/train_gen.py:170-190`, four forward-equivalent
units versus M3's three). Both run 600 steps as registered; this is not an
equal-compute comparison or an actual hardware-FLOP measurement.

Thus "plausible-wrong" means **frozen-model-weighted wrong codes conditioned
on canonical coarser levels**, with a context-free canonical-excluded root.
There is no typicality, top-k or minimum-likelihood filter, and no guarantee
of plausibility under the corrupted context actually supplied to training.
It is neither an on-policy rollout nor sequentially self-consistent sampling.
Both replacements are wrong at every masked slot: the executed hierarchy
checks find `(used != canonical) == mask` exactly, including on hash-pinned
released tokenizer/puzzle inputs. Disagreement among corrupted rows therefore
equals realized mask rate; across all rows it includes the 25% clean dilution.

## Evaluation integrity

`scripts/train_gen.py:449-455` uses the same final evaluation for both arms:
1000 fixed even-index validation windows for clean NLL, 16 sample seeds
`[42,0,...,14]`, 256 samples each, ancestral T=1, 11 NFE/sample, both scorers,
and 4096-vs-4096 MAUVE. Validation runs every 100 training steps; curve samples
at 100/300/600 use the same sample seed within each paired training seed.
Final sample seeds/counts are identical across all training seeds and arms.

`scripts/train_gen.py:327-336` disables TF32 for the entire sampling/scoring
call and forks/restores training RNG, including exceptions. The trainer's
`ctx` property creates fresh autocast instances (`src/ncmr/train.py:86`);
the executed nested/repeated-context probe shows no R024 leak.
`scripts/train_gen.py:287-302` disables autocast and TF32 for canonical
teacher-forced NLL and scores every position at trainable levels 1--11.
`src/ncmr/data.py:42-47` selects val=even and test=odd; the executed item-order
probe confirms this, including the packed-window construction. It does not
mix odd-index test stories into val windows. No test-set tuning is present
in this W2C path.

`src/ncmr/eval_text.py:78` loads both scorer models in fp32/eval mode.
`scripts/train_gen.py:349-362` uses fp32 MAUVE sample/reference features and
a separate `mauve_ref_val_fp32_4096.npy` cache, referencing the first 4096 val
windows. MAUVE averages five quantization seeds 0--4. Existing executed
precision probes confirm that the bf16 reference cache is not reused.

Git diff and tree-object checks confirm **`src`, `scripts`, `configs`,
`assets`, `notebooks` are identical** at `5ad74b5`, M2's `6be0690`, and M3's
`8bc61b8`. Changes outside those trees include documentation/results.
All six run manifests are structurally identical, including the five base
asset hashes, three data hashes, Torch `2.11.0+cu130` and CUDA `13.0`.
In particular the generator is
`0acaf4f795390f9f7ccc8f84047867619a1d00e291681e299b855d7e563be6ff`
and validation data is
`f4723c2b0e4c36ac792e609dc50ddf7879ba4e33b7d794bc4ad44b952fde5c8d`.
The manifest also agrees on tokenizer, BPE vocabulary/merges, training bytes
and training offsets. Notebook/provenance commands match the two named configs.

These checks establish the same intended evaluation, not a fresh full-size
GPU replication. Scorer revision hashes and cached MAUVE feature bytes are
not published in these manifests/results. No scorer/reference-feature mismatch
was demonstrated, but their byte identity across notebooks cannot be certified
from saved scalars alone. No scorer weights were downloaded for this audit.

## Alternative explanations and permitted claim

| Explanation | What W2C establishes | What remains open / claim limit |
|---|---|---|
| Any nonuniform wrong distribution would do this | Frozen-model-weighted wrong differs from uniform-wrong at matched slots and rates. | No higher-temperature, frequency-matched or unigram-wrong control. Do not identify model-specific plausibility as the unique cause. |
| Length or entropy drives the metric gap | Entropy rises .027078 nats; re-encoded M2/M3 lengths are 126.7485/127.7495, 126.7666/127.7566, 126.6528/127.7073 (mean difference -1.0151 tokens). | No length/entropy-matched intervention. Excludes decreased marginal entropy as the description, not all length/entropy mediation or conditional-entropy effects. |
| GPT-2 rewards a peculiarity of M3 | Every paired group agrees under both scorers; MAUVE and adjacent duplicates agree on direction in every seed. | Stronger robustness evidence, not human evaluation or an absolute quality scale. Do not say all repetition measures worsen: M2's `rep_2` is about .0139 lower. |
| Clean NLL omits the affected levels | Clean NLL includes all positions of L1--L11, including R058's implicated L2--L5. Small positive differences occur throughout. | It conditions on canonical inputs and omits behavior under generated contexts. Do not claim zero clean predictive effect, likelihood invariance, or metric insensitivity by omitted levels. |
| M3 itself does not damage generation | M3 is the matched comparison for the additional M2 effect. | M3's mean 21.6768 exceeds released A0's 16.1644; seed-0 M3-0 supports a clean-row explanation for control drift (R059), not the full factorial across seeds. |
| This establishes what the levels encode | The hierarchy's generation response depends reproducibly on the training replacement distribution. | Stage 1 does not localize the effect. R058's L2--L5 / interaction account remains one seed. Do not call that localization replicated or identify individual codes' semantics. |

Suggested wording, which changes the proposed sentence's causal comparison:

> Under one continuation protocol for the released TinyStories NCM (600
> steps, 25% clean rows, W1's per-level rates and one mask stream), replacing
> masked codes with wrong codes drawn from the frozen released model given
> canonical coarser levels caused additional generation degradation relative
> to uniform-wrong replacement in the same slots. Across three training seeds,
> M2 minus M3 Gen-PPL was +18.9 under GPT-2 large (each seed at least +18.36;
> pooled seed SD 0.56) and +29.8 under TinyStories-33M, with MAUVE -0.41 and
> adjacent BPE duplicates +12.3 per thousand pairs. Canonical teacher-forced
> NLL was higher by only 4.0 nats/window (0.53%), positive in every seed.

Explain the root exception and "plausible-wrong" definition alongside that
sentence. Report this as W2C's registered **candidate** mechanistic finding,
with seed-0 localization as supporting evidence. A replicated claim about
particular levels would require replication of the localization/response
measurements; this audit neither changes the success criterion nor authorizes
new compute. No extension is established to other domains, released models,
from-scratch training, rates, clean fractions, continuation lengths, or
on-policy corruption. No benefit to generation architecture or adaptive
compute is established by this comparison.

## Execution and limits

The repository virtual-environment launcher could not find its original base
interpreter. A managed Python 3.11 interpreter could load the installed
repository packages, so **Python and CPU tests did run**. Runtime for this
audit was Torch `2.6.0+cu124`, exercising CPU operations, rather than the
Colab runs' Torch 2.11/CUDA 13. Default external pytest temporary directories
were not writable; explicit workspace-local temporary directories worked.
No dependency installation, training run or scorer download was needed.

Added files:

- `tests/test_audit12_numbers.py`: 22 checks, including eight full BPE entropy
  replays and eight independent duplicate counts, seed-0 context, aggregates
  and provenance minutes.
- `tests/test_audit12_readings.py`: 11 checks of pinned inputs, paired SE,
  pooled sample SD, saved analysis/readings and seed-0 exclusion.
- `tests/test_audit12_matched.py`: 7 checks of actual training-loop traces,
  proposal conditioning/precision, exact M3 support, released pinned assets
  and aggregate-slot collisions.
- `tests/test_audit12_evaluation.py`: 5 checks of six-run manifests/protocols,
  sorted group order, RNG/TF32 restoration and clean-NLL precision/split/coverage.

The existing `tests/conftest.py` CPU setup and hash-checked released Sudoku
fixture are reused. Released BPE assets are checked against the run manifest's
digests. The pinned Sudoku tokenizer/puzzles test uses a tiny head with the
text convention (predict L1 onward); it does not claim a Sudoku replication
of the TinyStories effect. Large checkpoints, the 600-step GPU histories,
MAUVE arrays, GPU precision/error bounds and scorer inference were not replayed.

Reproduction commands, with the repository's dependencies and pinned assets
available, are:

```sh
python -m pytest tests/test_audit12_numbers.py tests/test_audit12_readings.py tests/test_audit12_matched.py tests/test_audit12_evaluation.py -q -p no:cacheprovider --basetemp .tmp/audit12-final
python -m pytest tests/test_r056_build.py tests/test_w2m_arms.py tests/test_w2m_train_loop.py tests/test_audit8_precision.py -q -p no:cacheprovider --basetemp .tmp/audit12-existing
```

Final combined runs: **45 passed in 167.29 seconds** for the added audit
suite; **150 passed in 112.34 seconds** for the existing subjects above.
The latter emits 20 warnings from deliberately tiny sample-stat fixtures
(empty position bands and zero-probability JS intermediates); no test fails.
No strict xfail or skipped test was needed in these runs. Earlier failed
launcher/temp-directory attempts are not counted as passing tests.

Local Git staging is denied because the worktree's Git metadata is outside
the writable sandbox. The connected GitHub service also rejected tree creation:
"MCP tool call requires approval, but approval policy is never." No audit
commit or remote branch update was made. Only this audit document and the
four permitted test files are left as working-tree changes; scratch and local
assets are excluded. The public-content scan found no identifying information,
absolute local paths, credentials or links in the five new artifacts.

## Finding summary

| ID | Priority | File | One-line issue |
|---|---|---|---|
| -- | -- | Audited scope | No P0/P1 demonstrated; registered numerical readings reproduce, with the claim limits above. |
