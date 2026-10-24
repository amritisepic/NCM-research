# Adversarial analysis of the wave-1 on-policy explanation

2026-10-24. This is a mechanism challenge, not a production-code audit. It is
also the run record: the requested file restriction takes precedence over adding
a log entry. No production code, registration, log, result file, or checkpoint
was changed. Source examined: research results at `416cb82`, exploratory script
at `1a496e2`, and audit 3. Levels below are zero-indexed.

## Verdict

**(b') survives the attempted falsification, with stronger Sudoku evidence for
context discounting; it is not yet the explanation of W1.** The new fixed-input
test finds a 76% reduction in L6's downstream target-NLL advantage and an 81%
reduction in scale-normalized logit response after plausible-wrong training.
It does not identify why that dependence was lost, or establish that the same
mediator caused the text collapse. The strongest competitors are inconsistent joint conditioning,
loss of denoising coverage, and a stable policy feedback loop. None is excluded
by the arguments in R031. The Sudoku comparison has a reproduced dropout-RNG
confound and an especially consequential change to the distribution of errors.

Previously missing was a measurement of coarse-context reliance. Higher entropy
on clean contexts alone is not evidence that the model uses those contexts less.
The inference-only experiment below attacks that intermediate claim directly.
Its numeric record, including hashes and seeds, is included at the end.

## What the reported evidence actually establishes

Recomputation from the saved JSON agrees with R030: A1 / A3 / A4 / A5 Gen-PPL
is 16.6042 / 41.4369 / 36.8058 / 32.1688. The A4 improvement removes
18.649% of A3's excess **PPL above A1**. That is a descriptive ratio, not an
estimate that target mismatch caused at most 19% of the damage. PPL is nonlinear;
re-quantization changes labels, unmasked finer inputs, gradients, and the evolving
policy together. There is no clean mediation intervention here.

The local text damage is real, but preserved pooled unigram frequencies and
30 inspected stories do not establish that every coarse plan is intact. The
R014 98% versus 43% token survival uses retained canonical finest codes: it is
not a measurement of the perturbation seen on a generated hierarchy. R020's
small fine-level contribution under *real* coarse codes cannot bound the value
of fine levels after a wrong coarse plan. Both limitations matter specifically
for the proposed coarse/fine explanation.

Clean-NLL damage at L3-L6 is consistent with context discounting, shared-weight
interference, incorrect compensation for residuals, and changes to probability
calibration. Lower training loss and gradient norm do not imply smaller or safer
AdamW updates: the optimizer rescales coordinates, and an update's direction
relative to the original task matters. No update-norm or gradient-alignment
measurement was saved.

R031's claim that AMP affects every arm alike therefore cannot cause an arm
difference is too strong. The **validation** path is shared; the AMP **policy
proposal** path is present only in on-policy arms. Audit 3's A3-1 still permits
an interaction through the training distribution. This is an unexcluded
competitor, not evidence that precision caused a 25-PPL gap. The old context
state leak is fixed; the older-PyTorch no-grad cache hazard is not demonstrated
in W1's recorded 2.11 runtime. Audit 3 also cannot recover exact checkpoint
lineage. These are reasons to require a clean replication, not to assert a bug.

## Strongest competing explanations

| Explanation | Why the existing data fit | Prediction that distinguishes it from a simple detectability account |
| --- | --- | --- |
| **Joint-context mismatch, especially level 0** | A one-step proposal is drawn using the canonical hierarchy, then inserted into a different, independently masked hierarchy. Individually likely codes need not form a likely joint hierarchy. A5 conditions on a previous sampled hierarchy and can reduce that mismatch. Local residual errors and A4's failure are expected. | Generate each proposal from the *actual used* coarser reconstruction, with masks/rates held fixed. Rescue despite unflagged plausible codes would contradict plausibility alone being sufficient. An L0-only intervention can rescue while higher-level replacements stay unchanged. |
| **Lost denoising coverage / shortcut adaptation** | A3 changes about 0.18 of slots, A1 about 0.31, and their perturbation geometries differ. Fine-tuning can forget how to handle inference-time errors it no longer sees. Shared weights can lose this skill and worsen clean NLL simultaneously. A5 has more effective corruption and better samples. | Rate-matched lower-rate uniform corruption can collapse too; adding a small severe-uniform component can rescue without revealing replacement flags. Frozen on-policy corruption can remain damaging. |
| **Stable policy feedback** | Early damage and a plateau are exactly compatible with rapid convergence to a bad self-consistent fixed point. A3/A4/A5 all update their own proposal distributions. A5 changes the feedback map. | A frozen released teacher is benign while the updating teacher is destructive, at matched data, masks, optimizer updates, and proposal precision. A flag need not stop feedback. Plateau timing does not discriminate. |
| **Residual-coordinate interference / repair limits** | The generator sees downsampled sums of residual contributions, not an explicit history of discrete codes. Different histories can collide in this representation. Nearest-residual labels minimize a local quantization distance, not final decoder error or autoregressive consistency. Re-quantization can improve canonical code NLL while generated residuals remain incompatible. | Oracle target re-quantization still leaves reconstruction error for some used histories; giving the generator history or the actual residual helps more than a binary provenance flag. Dependence can stay strong while fine predictions compensate for the wrong residual. Effects track residual geometry rather than policy probability. |
| **Ordinary calibration / optimization interference** | Entropy can rise without loss of contextual dependence. A4's strong clean NLL and poor generation already defeat a universal clean-entropy account. Same step counts are not same optimization paths, and AdamW can amplify small conflicting gradients. | Temperature adjustment improves sampling without restoring coarse-context sensitivity, or plausible-code training preserves/increases sensitivity after scale normalization. Frozen-policy damage remains. |
| **Remaining implementation/runtime interaction** | Policy precision and RNG behavior differ from the uniform path; exact historical lineage is missing. No current defect is proved to explain W1. | Collapse disappears in a faithful replication with fp32/TF32-off proposals, isolated RNGs, immutable manifests, and identical optimizer partitions. That would remove the need for a latent-code mechanism for the historical result. |

These are not mutually exclusive. In particular, representation aliasing can
produce the ambiguity invoked by (b') without making code-level plausibility
the relevant variable. A detector has access to the **joint conditioning
representation**, not just the marginal frequency of a replacement code.

### The level-0 interaction is a large, specific hole

In `scripts/train_gen.py:163-175`, the first proposal pass sees canonical L0,
but the returned L0 is a fresh unigram draw. `build_hierarchy` then independently
masks levels (`src/ncmr/corruption.py:95-100`). At the specified rates,
75% of windows replace L0; each of the eight L1 positions is selected with
probability 70%. Thus about **52.5% of window/L1-position pairs** combine an
L1 proposal made under canonical L0 with an attempted unigram L0 replacement
(before allowing coincidental code equality). At inference L1 is sampled under
the L0 that is actually used. The distributions are not "exactly" the same.

This is the registered one-step recipe, not a newly discovered indexing bug.
The canonical-policy identity tests cannot exclude it: those tests intentionally
remove the wrong-parent condition. A5's second pass changes this relationship
and also changes effective corruption and update count. Its smaller damage
therefore cannot specifically identify detectability.

The cleanest discriminator is an unflagged one-step arm whose L1 proposal sees
the used L0, with the rest of the masking recipe unchanged. A full used-parent
proposal ablation is stronger but more expensive. Simply keeping L0 canonical
in training creates a new inference mismatch; it is a weaker test.

## Sudoku reproduction: matching and confounds

The following assessment refers to `scripts/onpolicy_repro.py` at `1a496e2`.

| Quantity | Assessment |
| --- | --- |
| Training puzzles, augmentation, order | Matched: independent NumPy RNG, same seed and batch calls. |
| Nominal mask slots and rates | Matched **between the two wrong-code arms**: both take the `onpolicy` hierarchy path and only masks consume `gen`; replacement draws use another generator/global RNG. This does not extend to ordinary `uniform`, which also consumes `gen` for random codes. |
| Realized changed slots | Matched for `uniform_wrong` versus `plausible_wrong`: both force disagreement with canonical labels at every masked slot; finest level is untouched. Saved logged rates agree. |
| Initialization and initial optimizer state | Same released weights and empty AdamW state, same hyperparameters and LR schedule. Subsequent optimizer moments differ as a consequence of different gradients; they should not be artificially held equal. |
| Training dropout | **Not matched.** Plausible sampling consumes global Torch randomness before the training forward; uniform-wrong replacements use `rep_gen`. |
| Step-0 reference | Shared numerically, not copied: all saved runs have identical clean NLL 43.1918335, greedy solution vector (88.3333%), roundtrip results, and available entropy 54.8124 nats/puzzle. Sampled runs share the T=1 reference, 87.0%. Step-0 equality does not fix later dropout. |
| Evaluation effects on training RNG | Greedy generation uses argmax; T=1 sampling is inside `fork_rng`. The saved 100/200-eval variants have identical final NLL/greedy outcomes, consistent with isolated evaluation. |

### ADV-1 — P1: the claimed matched-choice comparison also changes dropout

**Location:** `scripts/onpolicy_repro.py:118-128`, training forward at `:131`.
The plausible arm's `Categorical.sample()` advances the global generator; the
uniform-wrong arm's `randint(..., generator=rep_gen)` does not. Dropout therefore
differs even before the first update. This blocks the strong claim that only the
wrong code changed. It does **not** establish that dropout explains the large
NLL gap; the comparison can still be suggestive. Both arms still have the same
dropout distribution: different draws are an uncontrolled source of single-seed
variance, not a demonstrated directional bias. The P1 concerns the asserted
controlled comparison, not the validity of every saved metric or a cause of W1.

**Exposing test:**
`tests/test_adv_onpolicy_mechanism.py::test_sudoku_wrong_arms_share_dropout_rng`
executes the actual loss-function sampling branches and asserts equal ambient
RNG state at hierarchy construction. It strict-xfails at the intended assertion.
Use a dedicated policy generator, and restore the same per-step dropout state
for each arm in a paired rerun. This is the same defect class as audit 3 A3-2,
in a different experimental script. No production patch was applied.

### Excluding the canonical code is a substantive distribution intervention

For policy probability `p(c | h)`, the forced-wrong law is

`q(j | h,c) = p(j | h) / (1-p(c | h))`, for `j != c`.

It explicitly uses the clean target `c`. Natural policy errors occur at a
location with probability `1-p(c | h)`; forced-wrong corruption chooses locations
independently of that error probability. It consequently overweights easy,
high-confidence locations. At locations with error probabilities 0.01 and 0.80,
the natural errors favor the latter 80:1, while forced errors weight them 1:1.
The accompanying unit test verifies this simple distinction.

Uniform-wrong also excludes the canonical code, so exclusion alone is not an
unmatched treatment. The **interaction with a sharply concentrated policy** is:
conditioning away its dominant mode magnifies extremely rare alternatives at
otherwise almost deterministic slots. Those alternatives may be less recognizable
than random codes, but they are not draws from the ordinary inference law.
Equal counts do not match error locations, residual displacement, conditional
entropy, or the difficulty of recovering the original label.

The new released-model diagnostic quantifies this distortion on the same 16
puzzles used below. Expected changed slots under the nominal masks are **2.614%
for natural policy draws versus 25.776% for forced-wrong draws**, about 9.86x.
Mean original-policy NLL of a forced-wrong replacement is **7.633 nats**, versus
**1.332** for a natural policy error (both weighted by the recipe's mask rates).
At L1, 81.25% of the forced-wrong mass has original probability below 0.001;
at L4, all of it does, with mean original NLL 37.84. Mean canonical probability
at L4 rounds to 1.0 in fp32. At the tested L6, 40.55% of forced-wrong mass is
below 0.001. These are local released-policy measurements, not W1 error rates.
They decisively reject calling this an unchanged inference error distribution;
they do not by themselves prove that the next-level model can detect the errors.

That supplies an explanation for the Sudoku NLL gap without (b'): repeated
near-boundary, target-selected errors create conflicting local supervision and
change calibration, even though the puzzle still specifies one solution and
coarse-code dependence need not decrease. Higher corrupted-input loss does not
measure detectability; it can measure this target difficulty or geometry.

### What 200 steps do and do not show

There are 3,200 puzzle exposures, substantial reuse under augmentation, and
only 180 post-warmup updates with cosine decay. At 100/200 steps, uniform-wrong
NLL is 47.8995/45.8913, plausible-wrong 77.2404/75.2070, and re-quantized
plausible-wrong 58.0077/49.9268. The clear early NLL/entropy separation is real;
convergence, asymptotic damage, feedback causality, and equivalence to W1 are
not established. The downward trajectories and declining LR make a plateau
particularly uninformative about irreducible ambiguity. A no-corruption
fine-tuning control and a frozen-proposal control are absent from these saved
wrong-code results.

Recomputed endpoints, all original training seed 0:

| Arm | Clean NLL | Clean entropy, nats/puzzle | Greedy solve % | T=1 solve % |
| --- | ---: | ---: | ---: | ---: |
| Step 0 | 43.1918 | 54.8124 | 88.3333 | 87.0000 |
| Uniform wrong | 45.8913 | 57.7979 | 83.6667 | 82.6667 |
| Plausible wrong | 75.2070 | 95.1998 | 87.3333 | 81.6667 |
| Plausible wrong, re-quantized | 49.9268 | 66.5046 | 85.0000 | 82.6667 |
| Ordinary on-policy samples | 46.1320 | 63.5767 | 85.0000 | Not run |

The natural on-policy arm's small clean-NLL change versus the forced-wrong
arm's large change is already a warning against treating them as replications
of the same treatment. Its changed-slot rate differs too, so that observation
alone does not separate exclusion from dose.

On T=1 solves the plausible-minus-uniform gap is just -1 point; the earlier
reported paired SE is about 2 points. There is no demonstrated solve-rate
collapse or clear between-arm sampling disadvantage. Sudoku's unique solution
is determined by the prompt; its corruption creates a harder computational
problem, not the literal missing information of unconditional text. The
cross-domain extrapolation is especially weak for A4, whose Sudoku NLL/entropy
damage is largely repaired while its text generation remains bad.

## Cheapest direct attack on the proposed mediator

Use the released, hash-checked Sudoku model and the saved original fine-tuned
weights. No new training is needed. On 16 generated held-out puzzles, draw
forced-wrong codes **once from the released policy**, using seeds 0, 1, 2.
Mask source level L6 at its training rate (0.45),
leave the other codes canonical, and
give every evaluated model the identical counterfactual code array. Measure
only predictions at strictly finer levels:

1. The canonical-target NLL advantage of the true context over the counterfactual
   context, in nats/puzzle. Larger advantage means more functional reliance on
   that true coarse context for this task.
2. KL between the clean-context and counterfactual output distributions.
3. Squared change in centered logits divided by clean centered-logit squared
   size. This is invariant to a common logit scale multiplier, so a pure global
   temperature change cannot by itself explain its direction.

Before execution, the experiment specified the following falsifier of the
**Sudoku context-discount interpretation**: the plausible-trained arm has
*more*, rather than less, reliance than the uniform-trained arm at **L6**
in both target-NLL advantage and normalized logit response.
A mixed or null result is inconclusive. This is a necessary-mediator test, not
an assertion that a finite Sudoku experiment can falsify the historical text
cause. The latter requires a controlled text intervention.

The script also computes the original policy probability of forced-wrong codes
and the expected natural versus forced changed-slot rates. It uses the fixture
in `tests/conftest.py`, checks every released asset SHA-256, records the training
archive and saved-weight hashes, runs fp32 with TF32 off, and writes only this
report. Original weight files are optional private run artifacts, not new public
assets; their hashes make this reuse explicit. Without them, the same runner
still measures the exclusion distribution on the released model.

The saved weight files are identified by their hashes in the new record; the
older aggregate JSON did not bind those hashes to its training history. Their
names and architecture agree with the original experiment, but this analysis
does not independently reconstruct that lineage or rescore all 300 puzzles.
The sensitivity measurement is a local intervention on otherwise canonical
hierarchies, not a measurement of complete generated trajectories. Increased
sensitivity can also mean brittleness to wrong inputs; it would contradict a
simple uniform discount account, not prove that the model uses coarse codes well.

```sh
python -B -m pytest -p no:cacheprovider tests/test_adv_onpolicy_mechanism.py -q -rx
python -B tests/test_adv_onpolicy_mechanism.py --run --weights <saved-weight-directory> --n 16 --cpu_seconds 500
```

The weight directory must contain `uniform_wrong_s0.pt` and
`plausible_wrong_s0.pt`. This directory
argument is a placeholder, not a committed local path. The default pytest run
does not start inference or training.

Setup passes used **1,147.90625 CPU seconds** (480.6875, 242.4375, 53.5, 371.28125), without
finishing an arm comparison. Dense attention made the 64-puzzle design too
costly. The final design uses 16 puzzles, just source L6 (before the L8-L10
entropy rise), source-level masks at training rates, and just the two trained
checkpoints needed for the contrast. Seeds and metrics are unchanged; the
falsifier is evaluated at L6 alone. These changes preceded any between-model
results. The completed run has a separate 500-CPU-second cap;
all abandoned work counts toward the task's 30-CPU-minute budget. These puzzles
are a subset of the original validation set, not an independent replication.
The smaller sample limits precision and generalization.

### Result: the discount prediction survives this test

Original training seed **0**; counterfactual-code seeds **0, 1, 2**; mask seeds
**104729, 104730, 104731**. Errors below are SE across the **16 paired puzzles**,
after averaging counterfactual seeds; they are not training-seed uncertainty.
Actual L6 mask fractions are 45.14%, 44.44%, and 47.92%, identical across models.

| Checkpoint | Clean NLL | Clean entropy | Finer-target NLL advantage of true L6 | Scale-normalized logit response |
| --- | ---: | ---: | ---: | ---: |
| Released | 42.010 | 55.364 | 17.387 ± 1.660 | 0.012516 ± 0.000612 |
| Uniform-wrong trained | 45.695 | 59.482 | 17.165 ± 1.663 | 0.012100 ± 0.000594 |
| Plausible-wrong trained | 73.601 | 95.450 | 4.141 ± 0.630 | 0.002294 ± 0.000354 |

Paired plausible-minus-uniform NLL advantage is **-13.023 ± 1.196 nats/puzzle**;
normalized response changes by **-0.009805 ± 0.000479**. Output KL per finer
code is 0.05456 / 0.05219 / 0.006924 for released / uniform / plausible.
The prespecified falsifying direction did not occur. Compared with uniform,
plausible training reduced the NLL advantage **75.9%**, output KL **86.7%**, and
normalized response **81.0%**. This is more direct support for a context-discount
effect than entropy alone, and it remains after removing a common logit scale.

This result limits the pure-calibration alternative: a single global temperature
change cannot explain all of it. It does **not** separate detectability from
forced rare-error training, residual geometry, feedback during the original
training, or the original dropout draws. Nor does it test text, L0 mismatch,
other source levels, or re-quantized training. A substantial loss of reliance is
now measured in this checkpoint; its proposed cause and relevance to W1 remain
open. Do not claim a new three-seed replication from the three intervention seeds.

Completed inference used **275.84 CPU seconds**. Including all stopped setup
passes, the measured diagnostic processes used **1,423.75 CPU seconds (23.73
CPU-minutes)**; imports and the small checks add only a little. No GPU or new
training was used. Asset hashes and per-seed statistics follow; source revisions
are given above.

The exact loss-function check reads the local exploratory script when present,
otherwise its Git object at `1a496e2`. A clone without that object must fetch
`claude/wip-w1-exploratory` first. The result recomputation reads ancestor
`416cb82`, so a shallow clone also needs that history.

Verification of the final checks: **1 passed, 1 strict xfailed**, Torch 2.11 CPU.
The expected failure is ADV-1's equality assertion, not an import or asset error.

## What the planned tier-L arms must decide

Register the contrasts before seeing their results. Use the same update count,
LR schedule, data, proposal precision, and dropout streams for causal comparisons;
report extra proposal compute separately. Compare clean and rollout contexts,
per-level entropy, and causal coarse-context sensitivity alongside Gen-PPL,
entropy, and MAUVE. One training seed can challenge a large effect but cannot
establish a replicated mechanism.

| Planned arm | Prediction of the strong (b') account | Result that would overturn or sharply weaken it |
| --- | --- | --- |
| On-policy + replacement flag | Making corruption observable substantially prevents the generation damage and restores reliance on unflagged plausible context. | Collapse remains close to unflagged A3 even though the model demonstrably reads a correct, sufficiently expressive provenance signal; a capacity-matched flag control succeeds and precision/masking are correct. This refutes detectability as a sufficient remedy. |
| Plausible-wrong at uniform rates | Inference-like plausible alternatives remain damaging at matched changed slots; live policy feedback is unnecessary. | A fixed, demonstrably inference-like proposal bank is benign at equal effective dose while live A3 collapses. That favors feedback or proposal/used-parent mismatch. A benign target-excluded arm alone is weaker if its proposals are actually implausible. |
| Uniform corruption at A3's lower rates | Lower dose alone should not recreate the large collapse. | This arm reproduces most of the local damage, PPL gap, and reliance change. That makes loss of denoising coverage sufficient and defeats the claim that plausible replacements are needed. |

The converse pattern—flag rescues, fixed plausible corruption damages, lower-rate
uniform is healthy—would substantially strengthen (b'). It would still not
uniquely establish it: flags also separate mixed domains/coordinate histories,
and frozen teacher codes can still be conditioned on the wrong used parent.
An actual-used-L0 proposal ablation is the cheapest remaining tie-breaker.

For the flag arm, signal **replacement attempts** from the sampled mask, not
`used != canonical`: the latter leaks target correctness. The generator sees
cumulative residual inputs, so expose enough source-level mask history to
identify contaminated ancestors, and verify flag use by toggling it on identical
inputs. Give inference a prespecified flag convention. A new, ignored binary
embedding and a negative result would not falsify anything. A rescue is also
not automatic evidence of detecting wrong codes: it may teach two separate
conditional models, with all-zero inference flags selecting the clean one.

The proposed lower-rate uniform arm is **not the registered W1b A1m**: W1b
matches decoded-token survival, whereas this mechanism test matches per-level
canonical disagreements. Register it under a new name. A4's logged
`corrupted_frac` is disagreement with *re-quantized labels*, so it cannot be
used as the canonical-error rate. Rates and their location correlations should
be estimated from A3's saved trajectory, rather than just its final rounded mean.

Finally, the scheduled-sampling connection is a useful analogy, not a derivation
of this latent-code mechanism. The primary paper establishes improperness of a
scheduled-sampling objective; it does not establish this experiment's proposed
detectability, residual-quantization, or constant discount-rate claims
([primary paper](https://arxiv.org/abs/1511.05101)). For this mixture, the optimal
predictor is `P(target | used context, prompt)`: the posterior probability of
corruption depends on both component densities, location, and context. It is
not generally the nominal mask rate. Even in a one-mask simplification, it is
`r q(h) / ((1-r) p(h) + r q(h))`, where `p` and `q` are the clean and replaced
**joint context** densities. Equality of code marginals does not imply `p=q`.
Even `p=q` would not make the corrupted target independent of the used context:
the proposal and target still share canonical ancestors. Separate that narrower, falsifiable
statement from the broader observation that corrupted-input training can hurt.

<!-- ADV_ONPOLICY_NUMBERS -->

## Complete numeric record

```json
{
  "protocol": "inference-only; fixed released-policy wrong codes; clean generated validation tail",
  "torch": "2.11.0+cpu",
  "precision": "CPU fp32; TF32 off; eval mode",
  "threads": 1,
  "n_puzzles": 16,
  "counterfactual_seeds": [
    0,
    1,
    2
  ],
  "source_levels": [
    6
  ],
  "mask_seed_offset": 104729,
  "counterfactual_changed_fraction_by_seed_source": {
    "0:6": 0.4513888955116272,
    "1:6": 0.4444444477558136,
    "2:6": 0.4791666567325592
  },
  "original_training_seed": 0,
  "training_steps_executed": 0,
  "data_sha256": "ca975491994394c9a695c4017b621831cc1ab451f2fb6f950fd9364441565b31",
  "released_asset_hashes": {
    "config.yaml": {
      "url": "https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/main/sudoku/hard/config.yaml",
      "sha256": "8d58c49e6fe52e60191e06df4061937d5bb84f97c5ffbcd93ed9830c32f8df2f"
    },
    "generator.pt": {
      "url": "https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/main/sudoku/hard/generator.pt",
      "sha256": "9e0bfe44abc1ca3930b7a63cd56cd0b9564b95e29edfe12dd0903207b89df8b8"
    },
    "puzzles.pickle": {
      "url": "https://huggingface.co/datasets/hazyresearch/ncm-datasets/resolve/main/sudoku/hard/puzzles.pickle",
      "sha256": "dd7b6763e5594094a1366ac8423f87fc9431056721c61df52ce3490023f17d5e"
    },
    "tokenizer.pt": {
      "url": "https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/main/sudoku/hard/tokenizer.pt",
      "sha256": "3bc6aecad8e15820f9d9bb09750e3c59bdf91ca953f1f11dbd109aaed1d80c0b"
    }
  },
  "natural_expected_changed_fraction_all_slots": 0.026142045855522156,
  "forced_wrong_expected_changed_fraction_all_slots": 0.25776398181915283,
  "forced_wrong_original_policy_nll": 7.633216857910156,
  "natural_error_original_policy_nll": 1.3320413827896118,
  "exclusion_by_level": [
    {
      "level": 0,
      "p_canonical_mean": 0.8126976490020752,
      "forced_wrong_expected_original_policy_nll": 4.139737129211426,
      "forced_wrong_mass_with_original_p_below_0_001": 0.1527235060930252,
      "natural_policy_error_rate": 0.187302365899086
    },
    {
      "level": 1,
      "p_canonical_mean": 0.9974139928817749,
      "forced_wrong_expected_original_policy_nll": 15.211175918579102,
      "forced_wrong_mass_with_original_p_below_0_001": 0.8125000596046448,
      "natural_policy_error_rate": 0.0025859232991933823
    },
    {
      "level": 2,
      "p_canonical_mean": 0.9458880424499512,
      "forced_wrong_expected_original_policy_nll": 7.147031784057617,
      "forced_wrong_mass_with_original_p_below_0_001": 0.537116527557373,
      "natural_policy_error_rate": 0.05411193519830704
    },
    {
      "level": 3,
      "p_canonical_mean": 0.9423980116844177,
      "forced_wrong_expected_original_policy_nll": 7.680362701416016,
      "forced_wrong_mass_with_original_p_below_0_001": 0.6203384399414062,
      "natural_policy_error_rate": 0.05760206654667854
    },
    {
      "level": 4,
      "p_canonical_mean": 1.0,
      "forced_wrong_expected_original_policy_nll": 37.83670425415039,
      "forced_wrong_mass_with_original_p_below_0_001": 1.0,
      "natural_policy_error_rate": 0.0
    },
    {
      "level": 5,
      "p_canonical_mean": 0.9227914810180664,
      "forced_wrong_expected_original_policy_nll": 7.591390609741211,
      "forced_wrong_mass_with_original_p_below_0_001": 0.5106827020645142,
      "natural_policy_error_rate": 0.0772085040807724
    },
    {
      "level": 6,
      "p_canonical_mean": 0.9151279926300049,
      "forced_wrong_expected_original_policy_nll": 6.228515148162842,
      "forced_wrong_mass_with_original_p_below_0_001": 0.4055359661579132,
      "natural_policy_error_rate": 0.08487206697463989
    },
    {
      "level": 7,
      "p_canonical_mean": 0.8844965696334839,
      "forced_wrong_expected_original_policy_nll": 5.808847427368164,
      "forced_wrong_mass_with_original_p_below_0_001": 0.34955617785453796,
      "natural_policy_error_rate": 0.11550340056419373
    },
    {
      "level": 8,
      "p_canonical_mean": 0.8875560164451599,
      "forced_wrong_expected_original_policy_nll": 5.628402233123779,
      "forced_wrong_mass_with_original_p_below_0_001": 0.33929261565208435,
      "natural_policy_error_rate": 0.11244383454322815
    },
    {
      "level": 9,
      "p_canonical_mean": 0.8750871419906616,
      "forced_wrong_expected_original_policy_nll": 4.774895668029785,
      "forced_wrong_mass_with_original_p_below_0_001": 0.25282880663871765,
      "natural_policy_error_rate": 0.12491291761398315
    },
    {
      "level": 10,
      "p_canonical_mean": 0.8683470487594604,
      "forced_wrong_expected_original_policy_nll": 4.233728408813477,
      "forced_wrong_mass_with_original_p_below_0_001": 0.16960583627223969,
      "natural_policy_error_rate": 0.13165298104286194
    },
    {
      "level": 11,
      "p_canonical_mean": 0.9750558137893677,
      "forced_wrong_expected_original_policy_nll": 11.569426536560059,
      "forced_wrong_mass_with_original_p_below_0_001": 0.7731972932815552,
      "natural_policy_error_rate": 0.024944255128502846
    }
  ],
  "models": {
    "released": {
      "weights_sha256": null,
      "clean_nll": {
        "mean": 42.009817004203796,
        "se_across_puzzles": 1.9115282154260809
      },
      "clean_entropy": {
        "mean": 55.36435008049011,
        "se_across_puzzles": 1.4319520357258513
      },
      "by_source": {
        "6": {
          "target_nll_advantage": {
            "mean": 17.387474834918976,
            "se_across_puzzles": 1.6604117156356932
          },
          "output_KL_per_code": {
            "mean": 0.05456019251141697,
            "se_across_puzzles": 0.004044211081754853
          },
          "scale_normalized_response": {
            "mean": 0.01251610938925296,
            "se_across_puzzles": 0.0006115056814610301
          }
        }
      },
      "by_seed_and_source": [
        {
          "seed": 0,
          "source_level": 6,
          "target_nll_advantage": {
            "mean": 17.401581823825836,
            "se_across_puzzles": 1.9927383275844026
          },
          "output_KL_per_code": {
            "mean": 0.054745536763221025,
            "se_across_puzzles": 0.005304250689389077
          },
          "scale_normalized_response": {
            "mean": 0.012989600945729762,
            "se_across_puzzles": 0.0010463458571910145
          }
        },
        {
          "seed": 1,
          "source_level": 6,
          "target_nll_advantage": {
            "mean": 17.377534061670303,
            "se_across_puzzles": 2.2421911680496653
          },
          "output_KL_per_code": {
            "mean": 0.05272075335960835,
            "se_across_puzzles": 0.005949554213028204
          },
          "scale_normalized_response": {
            "mean": 0.012517389579443261,
            "se_across_puzzles": 0.0012663760653897538
          }
        },
        {
          "seed": 2,
          "source_level": 6,
          "target_nll_advantage": {
            "mean": 17.38330841064453,
            "se_across_puzzles": 2.254752197744666
          },
          "output_KL_per_code": {
            "mean": 0.05621428819722496,
            "se_across_puzzles": 0.006580817881950277
          },
          "scale_normalized_response": {
            "mean": 0.012041337322443724,
            "se_across_puzzles": 0.001197475245733884
          }
        }
      ]
    },
    "uniform_wrong": {
      "weights_sha256": "7fc7c1eb10586d2a0fae0a9b5215ca96e375684e88479ed16242502a27f9c17b",
      "clean_nll": {
        "mean": 45.69475960731506,
        "se_across_puzzles": 1.5426741341945516
      },
      "clean_entropy": {
        "mean": 59.482439279556274,
        "se_across_puzzles": 1.4429792053928907
      },
      "by_source": {
        "6": {
          "target_nll_advantage": {
            "mean": 17.16460481286049,
            "se_across_puzzles": 1.66260627677525
          },
          "output_KL_per_code": {
            "mean": 0.05218574812170118,
            "se_across_puzzles": 0.003934759299510143
          },
          "scale_normalized_response": {
            "mean": 0.012099552754079923,
            "se_across_puzzles": 0.0005935597380031262
          }
        }
      },
      "by_seed_and_source": [
        {
          "seed": 0,
          "source_level": 6,
          "target_nll_advantage": {
            "mean": 17.46067449450493,
            "se_across_puzzles": 2.0587758079756786
          },
          "output_KL_per_code": {
            "mean": 0.052627852419391274,
            "se_across_puzzles": 0.005248895435632646
          },
          "scale_normalized_response": {
            "mean": 0.012647813011426479,
            "se_across_puzzles": 0.0011403597722362926
          }
        },
        {
          "seed": 1,
          "source_level": 6,
          "target_nll_advantage": {
            "mean": 17.157131671905518,
            "se_across_puzzles": 2.2690140205772726
          },
          "output_KL_per_code": {
            "mean": 0.05119350960012525,
            "se_across_puzzles": 0.005873887174709444
          },
          "scale_normalized_response": {
            "mean": 0.012192848807899281,
            "se_across_puzzles": 0.0012476182575420435
          }
        },
        {
          "seed": 2,
          "source_level": 6,
          "target_nll_advantage": {
            "mean": 16.876008704304695,
            "se_across_puzzles": 2.1893659788358732
          },
          "output_KL_per_code": {
            "mean": 0.05273588007548824,
            "se_across_puzzles": 0.006662730143760893
          },
          "scale_normalized_response": {
            "mean": 0.01145799647201784,
            "se_across_puzzles": 0.0011329326782925475
          }
        }
      ]
    },
    "plausible_wrong": {
      "weights_sha256": "6904ea432c36c451e6520eeb25a424ec17797227321ed089ac0de363a9f560ae",
      "clean_nll": {
        "mean": 73.60108184814453,
        "se_across_puzzles": 2.7070291788914767
      },
      "clean_entropy": {
        "mean": 95.45015144348145,
        "se_across_puzzles": 2.4343376669295442
      },
      "by_source": {
        "6": {
          "target_nll_advantage": {
            "mean": 4.1414708867669106,
            "se_across_puzzles": 0.6300498411795974
          },
          "output_KL_per_code": {
            "mean": 0.006924359935510438,
            "se_across_puzzles": 0.001357712558445351
          },
          "scale_normalized_response": {
            "mean": 0.0022942220202821773,
            "se_across_puzzles": 0.000353883138758372
          }
        }
      },
      "by_seed_and_source": [
        {
          "seed": 0,
          "source_level": 6,
          "target_nll_advantage": {
            "mean": 4.04888160340488,
            "se_across_puzzles": 0.6236087486447123
          },
          "output_KL_per_code": {
            "mean": 0.006845811389212031,
            "se_across_puzzles": 0.001974162298730098
          },
          "scale_normalized_response": {
            "mean": 0.0024292878260894213,
            "se_across_puzzles": 0.0006524402674984971
          }
        },
        {
          "seed": 1,
          "source_level": 6,
          "target_nll_advantage": {
            "mean": 5.048604667186737,
            "se_across_puzzles": 1.3974963572023582
          },
          "output_KL_per_code": {
            "mean": 0.007907111314125359,
            "se_across_puzzles": 0.0031147287364884043
          },
          "scale_normalized_response": {
            "mean": 0.0022005324353813194,
            "se_across_puzzles": 0.00046692442433355556
          }
        },
        {
          "seed": 2,
          "source_level": 6,
          "target_nll_advantage": {
            "mean": 3.3269264064729214,
            "se_across_puzzles": 0.413075763689833
          },
          "output_KL_per_code": {
            "mean": 0.006020157408784144,
            "se_across_puzzles": 0.0007508990199080769
          },
          "scale_normalized_response": {
            "mean": 0.0022528457775479183,
            "se_across_puzzles": 0.0003314244414046343
          }
        }
      ]
    }
  },
  "paired_plausible_minus_uniform": {
    "6": {
      "target_nll_advantage": {
        "mean": -13.023133963346481,
        "se_across_puzzles": 1.1957816111431963
      },
      "output_KL_per_code": {
        "mean": -0.045261388178914785,
        "se_across_puzzles": 0.003000236051497501
      },
      "scale_normalized_response": {
        "mean": -0.009805330773815513,
        "se_across_puzzles": 0.0004792167818824335
      }
    }
  },
  "cpu_s": 275.84375,
  "wall_s": 292.40243750001537,
  "original_recomputation": {
    "uniform_wrong_t1_s0": {
      "args": {
        "arm": "uniform_wrong",
        "steps": 200,
        "batch": 16,
        "lr": 0.0001,
        "warmup": 20,
        "eval_every": 200,
        "n_eval": 300,
        "seed": 0,
        "threads": 10,
        "sampled_eval": true
      },
      "evals": [
        {
          "step": 0,
          "nll": 43.19183349609375,
          "solved_pct": 88.33333333333333,
          "solved_t1_pct": 87.0,
          "entropy_nats_per_puzzle": 54.81239987164736
        },
        {
          "step": 200,
          "nll": 45.89132308959961,
          "solved_pct": 83.66666666666667,
          "solved_t1_pct": 82.66666666666667,
          "entropy_nats_per_puzzle": 57.79789961129427
        }
      ]
    },
    "plausible_wrong_t1_s0": {
      "args": {
        "arm": "plausible_wrong",
        "steps": 200,
        "batch": 16,
        "lr": 0.0001,
        "warmup": 20,
        "eval_every": 200,
        "n_eval": 300,
        "seed": 0,
        "threads": 10,
        "sampled_eval": true
      },
      "evals": [
        {
          "step": 0,
          "nll": 43.19183349609375,
          "solved_pct": 88.33333333333333,
          "solved_t1_pct": 87.0,
          "entropy_nats_per_puzzle": 54.81239987164736
        },
        {
          "step": 200,
          "nll": 75.20700073242188,
          "solved_pct": 87.33333333333333,
          "solved_t1_pct": 81.66666666666667,
          "entropy_nats_per_puzzle": 95.19979989528656
        }
      ]
    },
    "plausible_wrong_requant_t1_s0": {
      "args": {
        "arm": "plausible_wrong_requant",
        "steps": 200,
        "batch": 16,
        "lr": 0.0001,
        "warmup": 20,
        "eval_every": 200,
        "n_eval": 300,
        "seed": 0,
        "threads": 10,
        "sampled_eval": true
      },
      "evals": [
        {
          "step": 0,
          "nll": 43.19183349609375,
          "solved_pct": 88.33333333333333,
          "solved_t1_pct": 87.0,
          "entropy_nats_per_puzzle": 54.81239987164736
        },
        {
          "step": 200,
          "nll": 49.926815032958984,
          "solved_pct": 85.0,
          "solved_t1_pct": 82.66666666666667,
          "entropy_nats_per_puzzle": 66.50459920614958
        }
      ]
    },
    "onpol_sample_s0": {
      "args": {
        "arm": "onpol_sample",
        "steps": 200,
        "batch": 16,
        "lr": 0.0001,
        "warmup": 20,
        "eval_every": 100,
        "n_eval": 300,
        "seed": 0,
        "threads": 10,
        "sampled_eval": false
      },
      "evals": [
        {
          "step": 0,
          "nll": 43.19183349609375,
          "solved_pct": 88.33333333333333,
          "entropy_nats_per_puzzle": 54.81239987164736
        },
        {
          "step": 100,
          "nll": 48.4531135559082,
          "solved_pct": 84.33333333333334,
          "entropy_nats_per_puzzle": 64.26899907365441
        },
        {
          "step": 200,
          "nll": 46.13202667236328,
          "solved_pct": 85.0,
          "entropy_nats_per_puzzle": 63.57669906690717
        }
      ]
    },
    "shared_step0_greedy_nll_roundtrip_entropy": true,
    "w1": {
      "A0": {
        "genppl": 16.164382846075473,
        "genppl_se": 0.11210748562766881,
        "val_nll": 861.8385234375,
        "mauve": 0.8271892459587334
      },
      "A1": {
        "genppl": 16.604238305949796,
        "genppl_se": 0.077856908526092,
        "val_nll": 843.99265625,
        "mauve": 0.8810994988033372
      },
      "A2": {
        "genppl": 16.28974754404967,
        "genppl_se": 0.09217999958295824,
        "val_nll": 749.2537734375,
        "mauve": 0.914693191713515
      },
      "A3": {
        "genppl": 41.43688138357142,
        "genppl_se": 0.4297139260039023,
        "val_nll": 885.6671015625,
        "mauve": 0.495773658187321
      },
      "A4": {
        "genppl": 36.80578145639008,
        "genppl_se": 0.2551821056207577,
        "val_nll": 707.40434375,
        "mauve": 0.6730636816949088
      },
      "A5": {
        "genppl": 32.16881464361296,
        "genppl_se": 0.17794518901235035,
        "val_nll": 913.889390625,
        "mauve": 0.5430287649565455
      },
      "A6": {
        "genppl": 17.455434813395414,
        "genppl_se": 0.10200590927836059,
        "val_nll": 888.88803125,
        "mauve": 0.8324290762210979
      }
    },
    "A4_fraction_A3_excess_PPL_removed": 0.18649242904613472
  }
}
```
