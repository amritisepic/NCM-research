# Deep audit 14: W2C + W2M-GR claim

Audited commit: `3b601dcd4d39ad9b69fdc9c509165542ea151260`.
Audit branch: `codex/audit14`. Scope: R065, including its seed-0 row;
W2M-GR's registered analysis and identity evidence; R063's W2C wording;
the combined candidate and limits in the round-4 prompt; wordings A and B.

**One P1 claim-scoping finding; no P0 or new P1 implementation defect
demonstrated.** The registered readings stand: localization replicates,
(i) replicates, (iii) replicates, and W2M-G's own (ii) is distributed in
every seed under both scorers. The saved analysis reproduces exactly.
The proposed claim needs narrowing before publication: these results
establish conditional responses to whole-range sampler interventions, not
an intrinsic or temporal attribution of damage or a description of what
individual levels encode. Meeting the registered numerical pattern is
distinct from establishing criterion (c)'s encoding interpretation.

Two literal printed margin bounds need nonblocking rounding corrections.
All R065 table entries agree at printed precision. Independent
reconstruction of bootstrap intervals and independent authentication of
the W2C-to-replay final-checkpoint link are unavailable from the committed
artifacts; the precise limits are below. Neither is an observed mismatch.
Only this report, its numbers JSON and audit14 tests were added. No
production code, log, registration, notebook or result was changed.

## A14-1 — P1: candidate turns conditional replay effects into an origin/encoding claim

**Locations:** `log.md:1611` and `log.md:1632-1639` (R065 title/Status);
`docs/prompts/gpt6_astra_round4_2026-10-28.md:137-145` (candidate A and its
limits). The same interpretation also occurs in the prompt at lines
78-89. The relevant instrument is `src/ncmr/w2mg.py:81-114` and its shares
at `src/ncmr/w2mg.py:274-299`.

**Failure.** "The damage forms at levels 2-5" and "not at L1" promote
conditional sampler effects to a unique origin in the hierarchy. "Later
levels ... alone carry ~5%" omits the M3-prefix condition. The numbers
justify positive interaction in decoded-text log loss, but do not identify
an intrinsic prefix defect, exclude a foreign-prefix response, separate
damage from checkpoint compatibility/co-adaptation, or reveal code
semantics. This blocks a trustworthy publication of candidate A and an
unqualified assertion that it establishes the brief's criterion (c).
It does not invalidate the registered reading named "localization
replicates."

The measured counterexamples to an unconditional allocation are concrete:

- The seed-mean GPT-2 large pi_6 - pi_2 is **0.511346**. This is the
  effect of switching L2-5 jointly from M3 to M2, **given M2 L1 and M3
  L6-11**. Both cells have identical M2-generated L1 for a group. This
  contrast is identifiable despite nonadditivity; what is unjustified is
  interpreting it as an unconditional or individual-level share.
- sigma_2 - sigma_6 is **0.828003** for the same joint switch **given M3
  L1 and M2 L6-11**. Different surrounding samplers yield different effects.
- sigma_2 is **0.881209**: M2 L2-11 on M3 L1 retains about 88% of the
  contrast. Switching L1 to M2 on an M2 continuation has effect
  **1 - sigma_2 = 0.118791**, versus pi_2 = **0.015916** on an M3
  continuation. The small pi_2 does not make L1 irrelevant in every context.
- At cut 6, switching the suffix to M2 adds **0.053206** on M3 L1-5 and
  **1 - pi_6 = 0.472738** on M2 L1-5. Their difference is interaction
  **0.419533**. "Amplifies" can describe this specific conditional
  difference; it does not identify its unique mechanism.

There are no cuts at 3 or 5 and no isolated single-level interventions.
"L2-5 taken together, under these surrounding samplers" is supported.
The cells cannot say which of those levels contributes, or that damage
first appears there as generation unfolds. An M3 suffix measures M3's
response to the M2-generated prefix distribution. That is a valid
intervention, but is not an independent readout of how damaged that prefix
is. A foreign-prefix distribution may be unfamiliar without being outside
M3's mathematical support. Same-arm, cross-training-seed cells such as
M2_s -> M2_s' were not run, so compatibility within a checkpoint remains
an unresolved contributor to the interaction.

**Exposing tests:** `tests/test_audit14_claim.py` independently reconstructs
these conditional contrasts from hash-pinned cells. Its two assertion-only
strict xfails guard the actual R065 and prompt candidates against omitting
the conditional/scorer interpretation, foreign-prefix response,
checkpoint compatibility and semantic-attribution limits. They permit a
later explicit append-only R065 correction. These are documentation
regressions, not a claim that numerical code can test semantics.

**Proposed patch, not applied:** revise the active prompt and append a
correction to the historical log. The registration remains historical and
its operational readings remain unchanged. Apply the same scoping to the
prompt's earlier anomaly description.

```diff
--- a/docs/prompts/gpt6_astra_round4_2026-10-28.md
+++ b/docs/prompts/gpt6_astra_round4_2026-10-28.md
@@
-registers (anomaly 1); the damage forms at levels 2-5 (M2's prefix through L5 carries ~53% of the log-loss contrast,
-through L1 ~2%) and is amplified by the damaged model's own later levels (interaction ~0.42-0.44), which alone carry ~5%
+registers (anomaly 1); crossed replay reproduces a conditional hierarchy response: M2's prefix through L5, rendered by M3,
+carries ~53% of the pure-model mean-log-loss contrast, versus ~2% through L1. M2's levels 6-11 on M3's prefix through L5
+carry ~5%, with positive prefix/suffix interactions of ~0.42-0.44
@@
-because of the clean rows); no temperature-, frequency- or length-matched control; M2 costs one extra forward per step;
+with a clean-row effect supported by the seed-0 M3-0 intervention); no temperature-, frequency- or length-matched control;
+no marginal-preserving reassignment control; M2 costs an extra batch-equivalent proposal forward per step;
@@
-ranges).
+ranges). These are scorer-defined output effects, not an established semantic attribution or a demonstration of what
+the levels encode. Foreign-prefix response and checkpoint compatibility/co-adaptation remain open. Roots are shared
+released-unigram draws. Scorer revisions and MAUVE feature bytes were not pinned in the published manifests.
--- a/log.md
+++ b/log.md
@@ append after R065
+
+## R066 — Audit 14 corrections to R065's candidate claim and numerical wording
+R065's claim wording is superseded by a conditional replay interpretation. Across training seeds 1-3, M2's L1-5
+prefix rendered by M3 carries 52.7% of the pure-model GPT-2 large mean-log-loss contrast, versus 1.6% through L1.
+M2 L6-11 on M3 L1-5 carries 5.3%, with positive prefix/suffix interactions 0.444 / 0.420 at cuts 4 / 6.
+These whole-range interventions share released-unigram roots and noise. Foreign-prefix response and checkpoint
+compatibility/co-adaptation remain open. Semantic attribution and what the levels encode are not established.
+The registered reading "localization replicates" stands; criterion (c) is not established on a semantic reading.
+Corrected numerical statement, with margins: pi_2 < 0.026 vs 0.10, pi_6 >= 0.517 vs 0.30, sigma_6 <= 0.067 vs 0.20,
+interactions > 0.415 vs 0.20. R065's table values are correctly rounded. Shares and jackknife SEs are independently
+reconstructible from group Gen-PPL; (i) ratios use saved JS means, and (iii) differences use saved rho means.
+Their saved bootstrap CIs cannot be independently reconstructed from the committed summaries alone.
```

Use the next free log-entry number if R066 has already been assigned when
the correction is applied. No correction above has been applied by this audit.

## Independent recomputation and R065 discrepancies

The independent implementation is in `tests/test_audit14_numbers.py`.
Its arithmetic uses `json`, `math` and `statistics`, without repository
analysis functions or NumPy. SHA-256 pins cover the raw result files,
provenance, metadata and reference results, normalized to Git's LF bytes.
`docs/reviews/codex_audit14_numbers.json` contains full-precision results
for every seed/scorer/cut, all jackknife SEs, seed means/sample SDs, the
seed-0 row, ratios, root summaries, sanity calculations, times and drift.
Copied measurements are explicitly labelled `stored`.

For each scorer and cell, take the log of **each group's** Gen-PPL, then
the mean of those eight logs. This is not log of mean Gen-PPL. With pure
means a and b and crossed means x and y, calculate delta=a-b,
pi=(x-b)/delta, sigma=(y-b)/delta, interaction=1-pi-sigma. Recompute all
three shares leaving each paired group out, then use
sqrt((7/8) sum_i(theta_-i - mean(theta_-i))^2) for the jackknife SE.
Seed SD is the sample SD of seeds 1-3, ddof 1; seed 0 is never pooled.

The following tables show independently recomputed point estimates with
jackknife SE in parentheses. All requested sigma_2 and sigma_4 values are
included, although R065 does not print them.

| GPT-2 large seed | Cut | pi (SE) | sigma (SE) | Interaction (SE) |
|---|---:|---:|---:|---:|
| 1 | 2 | .011112 (.003441) | .886667 (.021912) | .102221 (.022111) |
| 1 | 4 | .226080 (.017757) | .328846 (.017900) | .445074 (.021117) |
| 1 | 6 | .540267 (.014560) | .041503 (.010953) | .418230 (.012773) |
| 2 | 2 | .025342 (.008584) | .882851 (.018260) | .091807 (.024552) |
| 2 | 4 | .231548 (.018764) | .323843 (.018072) | .444609 (.022943) |
| 2 | 6 | .524160 (.010526) | .051191 (.009593) | .424649 (.016535) |
| 3 | 2 | .011294 (.009546) | .874109 (.018994) | .114597 (.016580) |
| 3 | 4 | .211100 (.022510) | .347866 (.015167) | .441034 (.027485) |
| 3 | 6 | .517358 (.016056) | .066923 (.014403) | .415719 (.012439) |
| 0, exploratory | 2 | .009084 (.010565) | .868142 (.018728) | .122775 (.014482) |
| 0, exploratory | 4 | .220083 (.014862) | .310877 (.014577) | .469040 (.018279) |
| 0, exploratory | 6 | .511046 (.010500) | .061066 (.009158) | .427889 (.014035) |

| TinyStories-33M seed | Cut | pi (SE) | sigma (SE) | Interaction (SE) |
|---|---:|---:|---:|---:|
| 1 | 2 | .004702 (.006022) | .892303 (.023693) | .102995 (.023822) |
| 1 | 4 | .195700 (.013723) | .317411 (.016026) | .486889 (.017889) |
| 1 | 6 | .494572 (.012979) | .023919 (.011480) | .481509 (.013684) |
| 2 | 2 | .018464 (.006229) | .884182 (.017640) | .097355 (.022711) |
| 2 | 4 | .205227 (.014401) | .318792 (.017017) | .475980 (.018082) |
| 2 | 6 | .495203 (.010047) | .045140 (.007715) | .459657 (.015133) |
| 3 | 2 | .012031 (.008608) | .870416 (.019374) | .117553 (.017030) |
| 3 | 4 | .184463 (.021080) | .341390 (.010028) | .474147 (.023239) |
| 3 | 6 | .479025 (.015913) | .056348 (.011297) | .464627 (.010290) |
| 0, exploratory | 2 | .011926 (.009849) | .878817 (.019045) | .109256 (.015486) |
| 0, exploratory | 4 | .197561 (.012043) | .312057 (.015890) | .490382 (.016711) |
| 0, exploratory | 6 | .487232 (.009952) | .055258 (.008941) | .457510 (.014167) |

| Scorer, seeds 1-3 only | Cut | pi mean (seed SD) | sigma mean (seed SD) | Interaction mean (seed SD) |
|---|---:|---:|---:|---:|
| GPT-2 large | 2 | .015916 (.008164) | .881209 (.006438) | .102875 (.011409) |
| GPT-2 large | 4 | .222909 (.010586) | .333518 (.012675) | .443572 (.002211) |
| GPT-2 large | 6 | .527262 (.011765) | .053206 (.012829) | .419533 (.004605) |
| TinyStories-33M | 2 | .011732 (.006886) | .882300 (.011064) | .105967 (.010422) |
| TinyStories-33M | 4 | .195130 (.010394) | .325864 (.013464) | .479006 (.006889) |
| TinyStories-33M | 6 | .489600 (.009164) | .041802 (.016471) | .468598 (.011454) |

**Every disagreement/correction:**

1. `log.md:1625-1626`: the literal bound **pi_2 <= .025** is false:
   seed 2 is **.02534171744770981**. Its three-decimal table entry .025
   is correct. Use pi_2 < .026 or approximately .025.
2. The same sentence's **interactions >= .416** is false: seed 3, cut 6
   is **.415718759007632**. Its rounded .416 is correct. Use > .415 or
   approximately .416. The other literal margins are valid:
   min pi_6=.5173584548755764 >= .517 and
   max sigma_6=.0669227861167916 <= .067. All are far inside the
   actual registered thresholds. Two strict xfails expose the printed
   inequality errors and allow an append-only correction.
3. `log.md:1614-1615` says every share, ratio and contrast below was
   recomputed from cells' per-group Gen-PPL. Only the replay shares and
   contrasts come from those values. (i) uses saved JS bank means;
   (iii) uses saved root-reliance means. Their CIs require uncommitted
   per-window arrays. This is an inaccurate provenance statement, not
   a demonstrated wrong ratio or CI. The proposed append corrects it.

No other R065 numerical disagreement was found at printed precision.
The seed-0 row, rounded mean/SD row, ratio ranges, rho values/bounds,
sanity z values, minutes and approximate CU all agree. "Within ~.03"
is reasonable for the selected table values (largest absolute departure
about .03134 for TinyStories sigma_6); it is not a bound on every
unprinted share: GPT-2 sigma_4 at seed 3 differs from seed 0 by .03699.

## (i), (iii), timing and the limits of reconstruction

For (i), independently average the four saved per-level JS means at
levels 2-5, then divide each generated-bank mean by the clean-bank mean.
The table's CI endpoints are **transcribed from the raw results**, not
independently bootstrapped.

| Seed | J_clean, nats | Released bank ratio [stored CI] | M2 bank ratio [stored CI] | M3 bank ratio [stored CI] |
|---|---:|---:|---:|---:|
| 1 | .012030003 | 6.41235 [6.15717, 6.67784] | 6.97071 [6.68833, 7.28254] | 5.53207 [5.30055, 5.77426] |
| 2 | .012341591 | 6.25352 [6.00139, 6.51401] | 6.83787 [6.55122, 7.14106] | 5.47883 [5.26051, 5.70910] |
| 3 | .012567683 | 6.16943 [5.92129, 6.42907] | 6.74523 [6.46544, 7.04643] | 5.38429 [5.16224, 5.62553] |
| 0, exploratory | .012583365 | 6.33711 [6.08372, 6.62517] | 6.85837 [6.57079, 7.16868] | 5.53198 [5.29412, 5.78131] |

Thus 6.2-7.0x, 5.4-5.5x, .0120-.0126 and the saved minimum lower bound
**5.1622421570**, printed 5.16, agree. Every ratio >=3 and every saved
lower bound >1.2 in all three seeds.

For (iii), subtract saved rho means independently; the individual rho
means, their marginal SEs and the paired bootstrap endpoints remain saved
measurements. Rho is the change in canonical L1 NLL when roots are drawn
from the released unigram, averaged over eight draws and 512 clean windows;
a draw may equal the true root. It is not mutual information or a
measurement of generated-prefix semantic content.

| Seed | Released rho | M2 rho | M3 rho | Independently subtracted M3-M2 | Stored paired 95% CI |
|---|---:|---:|---:|---:|---:|
| 1 | 3.043031 | 1.994935 | 3.376898 | 1.381963 | [1.273266, 1.500963] |
| 2 | 3.043031 | 2.014004 | 3.416041 | 1.402038 | [1.289682, 1.525654] |
| 3 | 3.043031 | 2.046001 | 3.427491 | 1.381490 | [1.270197, 1.499948] |
| 0, exploratory | 3.043031 | 1.995397 | 3.386043 | 1.390646 | [1.280170, 1.515068] |

Every saved paired interval lies strictly above zero. R065's 1.38 / 1.40 /
1.38, its two-decimal CIs, released 3.04, M2 1.99-2.05 and M3 3.38-3.43
are correct. Seed-0 M0/M4 are outside the claim and were not pooled.

`scripts/w2mg_eval.py:318-347` saves per-window JS and rho arrays in its
run's `cells` cache, but `notebooks/make_notebooks.py:451-463`'s collector
copies selected top-level files only. The committed W2M-G/GR directories
contain results and provenance, without those arrays, banks, cell samples
or cache. Different window distributions can have the same saved means
and different bootstrap intervals; exact rebootstrap is impossible from
these summaries, regardless of interpreter or compute. The CI algorithms
themselves match the registration by source and CPU tests: 2000
percentile-bootstrap draws, seed 0, independent window resampling of both
banks for (i), and paired window resampling of rho differences for (iii).
Archive these cached arrays and their hashes to make that part of the
claim independently reanalyzable. This audit does not label copied bounds
as recomputed or certify their GPU-level derivation.

The three provenance files each contain one start/end pair at `46b0470`,
NVIDIA L4, fresh run (`had_ckpt: false`), exit 0. `_meta.json` agrees
exactly. Saved elapsed minutes are **18.88 / 18.50 / 18.11**, sum **55.49**:
R065's 18.9 / 18.5 / 18.1 and 55.5 agree. UTC differences, recorded only
to seconds, are 18.8833 / 18.5000 / 18.1167 minutes, consistent with the
separately rounded elapsed counter. At 1.71 CU/hour the runtime estimate
is **1.581465 CU**, hence ~1.6; this is not independently measured billing.

## Registration, fixed admission and model identity

The last W2M-GR section in `docs/preregistration.md` is the numerical
specification; W2M-G's retained (i)-(iii) apply unchanged. Inspection and
the existing boundary tests establish:

- GPT-2 large must satisfy pi_2 <= .10, pi_6 >= .30, sigma_6 <= .20,
  and interaction >= .20 at **both** cuts 4 and 6, in **every** seed 1-3.
  TinyStories-33M must satisfy these on its **seed means**. It is not
  required to pass in every TinyStories seed, although all three do here.
- Overturn uses GPT-2 seed means: pi_2 >= .25, or sigma_6 >= .50, or
  interactions < .05 at **both** cuts. Overturn takes precedence. Otherwise
  the result is mixed. Inclusive versus strict comparisons match, except
  the documented 1e-12 localization tolerance for floating point noise;
  that tolerance has no remotely material effect on these results.
- (i) requires every r_b >=3 and every lower bound strictly >1.2 in every
  seed; its own overturn rule requires all upper bounds <=1.2.
  (iii) requires every paired CI strictly above zero; touching zero does
  not hold. Neither reading depends on the replay share thresholds.
- W2M-G's (ii) is early-state selection iff pi_4 >=.5 and pi_6 >=.75;
  conditional rendering iff sigma_6 >=.8, both if both conditions hold,
  otherwise distributed. Independent arithmetic gives **distributed** for
  both scorers at seeds 0,1,2,3. Seed 0 is reported beside, never pooled.
- No (iv), M0/M4 or corrupted bank enters W2M-GR. The clean bank is the
  first 512 windows built from even-index validation items, not test. Sampling/scoring and
  target forwards remain fp32, TF32 off, eval mode. This audit found no
  changed precision path or context-manager reuse in this delta.

A13-1 is fixed in the audited code. `git diff b550b3b HEAD` is empty for
the analyzer, evaluator, W2M-G module and W2M-GR notebook. The run commit
`46b0470` and PR 18 merge `17a4bf4` have identical `src`, `scripts`,
`configs`, `assets`, `notebooks` trees. All three real results pass the
fixed `check_result` for their actual seed and reject a different seed.
The new protocol tests mutate the actual seed-2 result to a test bank,
wrong cut, other replay seeds and consistently recomputed four-group
metrics; all four are rejected. This is acceptance for the right declared
protocol, not merely acceptance of internally consistent shares.

Each admitted result has exactly the eight registered cells, eight finite
paired group Gen-PPL/log-loss entries under both scorers, 256 samples per
group, 2048 per cell, correct means/sample SD/SE, replay seeds 0-7 and bank
seeds 900/901. The recorded GPU is exactly NVIDIA L4. All five released
asset hashes and the validation hash equal exploratory W2M-G seed 0 and
all six W2C manifests. In particular the released generator hash is
`0acaf4f795390f9f7ccc8f84047867619a1d00e291681e299b855d7e563be6ff`
and validation hash is
`f4723c2b0e4c36ac792e609dc50ddf7879ba4e33b7d794bc4ad44b952fde5c8d`.

**Final-checkpoint identity limit:** the W2M-GR results record six distinct
M2/M3 checkpoint SHA-256 values and the intended seed-specific run names.
W2C's committed result, manifest, config, provenance and metadata do **not**
record their written `final.pt` hashes. `scripts/train_gen.py:443-446`
writes those files without publishing a digest. The checkpoints themselves
are absent. Therefore equality to independently W2C-recorded final hashes
cannot be checked; there are no such reference digests in this checkout.
Plausible agreement of metrics and correct filenames are supporting
evidence, not a cryptographic identity proof. Do not mistake the shared
**released base** hash for the six **trained final** hashes. To close this
gap, publish hashes attested from the original W2C output files, then
compare with W2M-GR. No hash mismatch has been demonstrated.

Calling both analyzer entry points on the committed inputs, then
`json.dumps(..., indent=1)`, equals `results/w2mgr_analysis.json` **byte
for byte after CRLF-to-LF normalization**, as well as structurally. No
result file was rewritten. Independent arithmetic also agrees with all
saved share points and jackknife SEs to floating point roundoff; equality
of the repository's own output is a separate check from that arithmetic.

## Cell semantics and consistency with W2C

Every cell begins with **L0 sampled from the same released-model unigram**,
loaded at `scripts/w2mg_eval.py:203`; M2 and M3 do not choose separate roots.
The single device-local generator for each group is seeded by that group's
seed; it draws roots first, then level-ordered Gumbel noise with fixed
shapes. Thus roots and noise are common across all eight cells (and across
training-seed runs on the same L4/runtime). Levels 1..c-1 use the prefix
checkpoint and levels c..11 use the suffix checkpoint. "M2's prefix through
L1" means the shared released-unigram L0 followed by M2-generated L1, not
an M2-specific learned root. Pure cells also use that released root prior.

W2C final evaluation used 16 x 256 groups on A100 with sampling seeds
42,0-14; its sorted score vectors correspond to 0-14,42. W2M-GR used
8 x 256 on L4, seeds 0-7, its cached Gumbel sampler. Equal integer seeds
across those GPU/sampler paths do not establish paired generated samples.

| Seed | Replay M2 / W2C M2 | Replay M3 / W2C M3 | R065 replay-SE z, M2 / M3 | Replay delta / W2C delta, mean log loss |
|---|---:|---:|---:|---:|
| 1 | 38.775157 / 39.790341 | 21.349794 / 21.425413 | -1.664848 / -.495295 | .596036 / .618744 |
| 2 | 40.431446 / 41.307912 | 21.860950 / 21.801928 | -1.449380 / +.256466 | .614518 / .638973 |
| 3 | 40.359964 / 40.560383 | 21.689077 / 21.803127 | -.415571 / -.855364 | .620666 / .620469 |
| 0, exploratory | 41.901965 / 42.620000 | 21.469922 / 21.690000 | -.961937 / -1.063353 | .667892 / not used |

The seed-0 reference is W2M's **rounded registered constants**, not W2C.
All z values round exactly as R065 prints. Replay minus W2C PPL contrasts
are **17.425363 vs 18.364928**, **18.570497 vs 19.505983**,
**18.670887 vs 18.757255**. All pure means pass the registered 3-SE
sanity. There is no discrepancy large enough here to contradict a common
model/contrast interpretation, subject to the final-hash limitation above;
this is not a formal equivalence test.

`src/ncmr/w2mg.py:324-333` and `scripts/analyze_w2mgr.py:151-155` divide
by the **replay cell's SE alone**, computed as sample SD of eight PPLs /
sqrt(8). That reproduces the registered descriptive check if the saved
W2C estimate is treated as a fixed reference. It is not a calibrated
two-estimate z: W2C's 16-group mean also has uncertainty. Under independence
the appropriate denominator would be sqrt(SE_replay^2 + SE_W2C^2);
with coupling, include minus twice the covariance. Independence or exact
pairing cannot be certified here. As an explicitly assumption-dependent
sensitivity calculation, using both SEs gives M2/M3 z values
**-1.461/-.393**, **-1.296/+.218**, **-.328/-.559**. It changes no sanity
pass. Do not treat the replay-only z as evidence of statistical equivalence
or of a 3-sigma discrepancy test between two population means.

Quoting **+18.9 Gen-PPL** beside **52.7% of mean-log-loss contrast** is
acceptable if their units and evaluations are explicit. W2C's D_s is a
difference of arithmetic means of group PPL; the replay share is a ratio
of differences of mean **log** PPL. Nonlinearity and the new sample set
mean that .53 x 18.9 is **not** an attributable Gen-PPL increment. The
log-loss shares are not fractions of a code-defined quantity. Agreement
from TinyStories-33M supports scorer robustness, not an intrinsic
representation metric or code semantics.

## Wording judgments, alternatives and criterion (c)

R063's W2C sentence (`log.md:1575-1585`) remains supported: it defines
frozen released-model proposals given canonical coarser levels, including
the root exception, and gives the comparative generation effect and small
positive NLL effect. "Plausible-wrong" is that operational distribution,
not a guarantee of plausibility in corrupted training contexts. "Barely
registers" must mean **+4.0 nats/window, .53%, positive in every seed**,
not zero, equality or statistical equivalence. No extra clean-NLL claim
is established by replay. R063's final "until localization replicates"
does not logically make replication sufficient to identify encoding
semantics.

**A:** the comparative first clause is supported with R063's definition
and continuation scope. The origin/temporal clause overstates; "alone"
needs the M3-prefix qualifier, and "amplifies" needs its operational
conditional-interaction interpretation. Use instead:

> Under the registered continuation recipe for the released TinyStories
> NCM, frozen-proposer wrong replacement worsened generation relative to
> same-slot uniform-wrong replacement across three training seeds
> (GPT-2 large Gen-PPL +18.9), while canonical teacher-forced NLL increased
> by only 4.0 nats/window (.53%). Crossed replay reproduced a conditional
> hierarchy response: M2's L1-5 prefix rendered by M3 carried 52.7% of the
> pure-model GPT-2 large mean-log-loss contrast, versus 1.6% for M2 L1.
> M2 L6-11 on M3 L1-5 carried 5.3%, with positive prefix/suffix
> interactions of .444 and .420 at cuts 4 and 6.

Accompany it with: shared released-unigram roots and noise; whole-range
interventions, not individual-level or semantic attributions; foreign-prefix
response and same-checkpoint compatibility remain possible contributors.

**B:** substantially sounder, provided "localizes a middle-level prefix
contrast" means a conditional sampler-range contrast rather than code
content. A more precise version is:

> Under this continuation recipe, frozen-proposer wrong replacements
> worsen measured generation outcomes versus same-slot uniform-wrong
> replacements; across three seeds, crossed replay reproduces a
> conditional contrast involving the L2-5 sampler block and a positive
> prefix/suffix interaction in decoded-text log loss.

Neither version should be read as determining the semantics of individual
levels, proving that damage originates in L2-5, or identifying a unique
mechanism of the interaction. The defensible result is a **replicated
intervention finding about where two differently trained checkpoints'
samplers and responses diverge in ways that affect decoded generation**.
It is mechanistic at that operational level. On the strict reading of
`docs/brief.md:25`, **what the levels encode is not established**, so this
audit does not certify criterion (c). Calling this sufficient for a broader
hierarchical-response interpretation of (c) would require an explicit
interpretive decision, not a stronger empirical claim or a silent change
of success criterion. No new compute is authorized by this report.

| Alternative/limit | Can W2C + W2M-GR rule it out? | What the claim must concede |
|---|---|---|
| Frequency, temperature, length mediation | No matched intervention for any of these. | Do not say frozen-model plausibility uniquely causes the gap or these mediators are excluded. |
| Unequal compute | M2 executes an extra frozen proposal forward; it has the same optimizer steps and isolated data/mask/dropout streams. | This is the registered recipe comparison, not an equal-compute or efficiency result. Extra inference is not evidence of extra optimizer updates. |
| Any example-conditioned nonuniform wrong distribution | No marginal-preserving association-randomization control. | Model-specific plausibility versus nonuniformity, marginals or context association remains open. |
| Replay instrument | Whole-range crossed replay only, two scorers. | Scorer agreement does not provide another intervention instrument or semantic attribution. |
| Clean rows/control drift | Seed-0 M3-0 supports a clean-row effect; no complete M2/M3 x clean-fraction factorial across seeds. | Comparative to M3, not released-model harm uniquely due to M2; no generalization of this gap to 0% clean rows. |
| Foreign prefixes and checkpoint compatibility | No same-arm cross-seed replay control. | Cannot uniquely distinguish damaged-prefix content, learned response to unfamiliar prefixes, or checkpoint co-adaptation. |
| Scope and measurement | One released model, one text domain, 600-step continuation, W1 rates, 25% clean rows; external scorer/MAUVE artifacts not fully pinned. | No from-scratch, other-domain, other-rate, architecture-quality, human-quality or adaptive-compute conclusion. |

The prompt's **5.5 Gen-PPL above released** is correct. W2C M3 seed-mean
21.6768229473 minus released 16.1643828461 is **5.5124401013**. Seed-0
M3 minus released is **5.5213686753**. R059's **5.1895155093** instead
compares seed-0 M3 with M3-0 (16.496236) and is not the released-model
gap. "Because of the clean rows" should say that the causal clean-row
evidence is the seed-0 intervention; M3-0 returns **near** released quality,
not numerically identical quality. The section-4 limits list should retain
audit12's scorer-revision/MAUVE-feature pinning limit and add the
marginal-preserving control, conditional scorer units, foreign-prefix/
compatibility ambiguity and absence of individual-level/semantic attribution.

## Execution and reproducibility

Python did run: a managed Python 3.11.16 interpreter could use the already
installed repository dependencies (NumPy 2.4.6, pytest 9.1.1, Torch
2.6.0+cu124, CPU execution). The worktree has no working local virtual
environment launcher. No packages, scorer weights or model checkpoints
were downloaded; no training or GPU replication was attempted. The
unchanged `tests/conftest.py` supplies CPU setup, and the new tests pin
the committed numerical assets rather than requiring released binary
checkpoints.

Final new suite: **30 passed, 4 strict xfailed in .47 seconds**. The xfails
are the R065/prompt claim-scope guards and the two inward-rounded margin
inequalities. The numbers JSON exactly equals a fresh call to the
independent arithmetic implementation. Source compilation, JSON parsing
and the scan for identifying information, absolute paths, emails and
session/credential links passed for all five new artifacts.

Existing W2M-G/W2M-GR and audit13 suites: **249 passed, 4 deselected in
78.82 seconds**. The four deselected checks require absent released
TinyStories generator/tokenizer/BPE/validation binaries: sampler versus
released teacher forcing, real validation-window selection, released
environment loading and real-asset identity hashing. The new tests still
check the committed manifests' and results' pinned hashes. These CPU tests
do not replay the full-size checkpoints or certify GPU inference.

Reproduction commands with the repository's dependencies installed:

```sh
python -m pytest tests/test_audit14_claim.py tests/test_audit14_numbers.py tests/test_audit14_protocol.py -q -p no:cacheprovider
python -m pytest tests/test_w2mg.py tests/test_w2mgr.py tests/test_audit13_analysis.py tests/test_audit13_boundaries.py tests/test_audit13_numbers.py -q -p no:cacheprovider --basetemp .audit14-verification -k 'not test_sampler_matches_teacher_forcing_on_the_released_generator and not test_real_val_windows_are_the_first_512_even_index_items and not test_run_identity_hashes_the_released_assets_val_data_device_and_runtime and not test_load_env_reads_the_four_checkpoints_and_m4_flag'
```

With the released assets present, omit the four exclusions. Initial
missing-parent temporary-directory errors and missing-binary failures are
not counted as successful checks. An existing test rejects any displayed
source path containing the text `tmp`; a workspace temporary directory
without that substring lets it run successfully without changing code.

Publication is blocked. Local staging is denied because this worktree's
Git metadata is outside the writable sandbox. The connected GitHub
service rejected creation of the tree containing exactly the five allowed
audit artifacts: "MCP tool call requires approval, but approval policy is
never." No audit commit, remote tree or branch update was made. Temporary
test output was removed; only the five permitted artifacts remain as
working-tree changes.

## Finding summary

| ID | Priority | File | One-line issue |
|---|---|---|---|
| A14-1 | P1 | `log.md:1632`; `docs/prompts/gpt6_astra_round4_2026-10-28.md:140` | Conditional whole-range replay does not establish the proposed intrinsic/temporal origin or encoding claim. |
