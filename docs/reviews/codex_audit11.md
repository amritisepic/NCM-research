# Deep audit 11: E1's claim and analysis

Audited commit: `8bc61b8394756e04e08e7418f9b36ac0e687268e`.
Branch: `codex/audit11`. Scope: the files and questions in
`docs/prompts/codex_audit11_e1claim.md`, with E1/E1a/E1b and its erratum
as the specification. Production code and saved results were not changed.

## Assessment

The saved evidence supports E1's registered cut-8 replication reading.
Independent calculations reproduce the post-E1b lower shares:
TinyStories **0.893622195644** on all 64 windows and **0.887817258555** on
the 61 untouched windows, WMT14 **0.859141120696**, and Sudoku
**0.993431732451**. Every required lower bound exceeds 0.8. E1a's stronger
rule holds only for Sudoku.

One P1 is reported: the E1b analyzer validates a selection header without
validating the records against that selection or reconciling missing records.
It can issue a successful registered reading on an off-protocol input and
incorrectly say every selected item was continued. The actual pinned result
has 72 unique, correctly selected records and passes the missing checks when
checked independently; this defect does **not** change its numerical reading.
No P0 or material confidence-inversion defect was demonstrated.

The confidence statements concern Monte Carlo uncertainty for these fixed
items, checkpoints, canonical prefixes and output keys. They do not establish
a population confidence bound, explain free-generation quality, or cover all
levels' code NLL. Machine floating-point sampling is an approximation to the
real-arithmetic categorical model; the evidence and limits of that approximation
are stated below rather than treated as a measured GPU error bound.

## A11-1 — P1: validate the continuation records, not just their header

**Locations:** `scripts/analyze_e1b.py:53-63`, `:71-85`, `:94-108`,
and `:122-129`.

`check_selection` verifies the censored/wide lists and source digest only for
models present in `selection`. It never checks that each record belongs to
that plan. `substitute` ignores the record's domain, accepts negative Python
indices, and has no duplicate identity check. Its E1-count comparison reads
the already modified copy: a second record whose `e1` equals the first
record's `final` passes. Removing a model's selection header also bypasses
that model's source-digest check while its records are still substituted.
The supplied sampling protocol is copied into output without validation.

Separately, the analyzer does not require selected items to partition into
continued records and `not_reached`. Removing a still-censored J0 record
from the released result without changing `complete: true` or the empty
`not_reached` leaves the bar met and prints "every selected item was
continued". An extra unselected J0 item, a wrong domain, a `-1` index in
place of J0 item 127, or a declared temperature of 0.5 is likewise accepted
by the relevant control flow. These are violations of the registered
replacement/disclosure rules, not statistical grounds for rejecting an
honestly disclosed partial continuation.

**Why P1:** this is the program emitting the registered success flag and
its integrity account. It does not enforce which pinned item streams the
claim is based on. An upstream partial or misassembled result can silently
be described as the registered, complete comparison. A digest of the E1
source alone cannot validate an E1b record's identity or protocol.

**Tests:** `tests/test_audit11_integrity.py` has seven strict expected-failure
cases: four record/header identity mutations, chained duplicate substitution,
an undisclosed omission, and a changed temperature. The ordinary partial-run
test preserves a successful reading with `complete: false` and a correctly
listed unreached item, as the registration requires. These tests were written
but could not be executed in this sandbox; no pytest outcome is claimed.

**Proposed patch, not applied:** validate before substituting, compare against
immutable source counts, and derive/check the completion account. This retains
the registered permission to read a disclosed partial run.

```diff
--- a/scripts/analyze_e1b.py
+++ b/scripts/analyze_e1b.py
@@
 def substitute(rows, items, model):
     """A copy of `rows` with the continued items' final E1b counts at cut 8. `items`: the e1b result's records."""
     out = copy.deepcopy(rows)
+    seen = set()
     for r in items:
         if r["model"] != model:
             continue
-        cur = out[r["item"]]["mc"][str(CUT)]
+        i = r["item"]
+        if type(i) is not int or not 0 <= i < len(rows) or i in seen:
+            raise ValueError("invalid or duplicate continuation item")
+        seen.add(i)
+        cur = rows[i]["mc"][str(CUT)]
         if list(cur) != list(r["e1"]):
             raise ValueError(f"{model} item {r['item']}: e1b's E1 counts {r['e1']} differ from the saved E1 file's {cur}")
         out[r["item"]]["mc"][str(CUT)] = [int(r["final"][0]), int(r["final"][1])]
     return out
@@
+def validate_records(e1b, files):
+    if e1b["protocol"] != B.PROTOCOL:
+        raise ValueError("unregistered E1b protocol")
+    domains = {m: d for m, d, _ in B.DOMAINS}
+    expected = {}
+    for model, selection in e1b["selection"].items():
+        if model not in domains:
+            raise ValueError("unknown continued model")
+        for kind in ("censored", "wide"):
+            for i in selection[kind]:
+                key = (model, i)
+                if key in expected:
+                    raise ValueError("duplicate selected item")
+                expected[key] = kind
+    done = set()
+    for r in e1b["items"]:
+        model, i = r["model"], r["item"]
+        key = (model, i)
+        if (type(i) is not int or key not in expected or key in done
+                or r["domain"] != domains.get(model)
+                or r["kind"] != expected[key]):
+            raise ValueError("continuation record differs from pinned plan")
+        rows = files[domains[model]]["rows"][model]
+        if not 0 <= i < len(rows) or list(r["e1"]) != list(rows[i]["mc"][str(CUT)]):
+            raise ValueError("continuation source counts differ")
+        final = r["final"]
+        if (len(final) != 2 or any(type(x) is not int for x in final)
+                or not 0 <= final[0] <= final[1]
+                or not r["e1"][1] <= final[1] <= B.CAP):
+            raise ValueError("invalid continuation counts")
+        done.add(key)
+    raw_missing = e1b.get("not_reached", [])
+    if any(len(x) != 2 or type(x[1]) is not int for x in raw_missing):
+        raise ValueError("invalid not_reached entry")
+    missing = {tuple(x) for x in raw_missing}
+    if (len(missing) != len(raw_missing) or missing != set(expected) - done
+            or type(e1b.get("complete")) is not bool
+            or e1b["complete"] != (not missing)):
+        raise ValueError("inconsistent continuation completion account")
@@
     check_selection(e1b, files, digests)
+    validate_records(e1b, files)
     items = e1b["items"]
```

This validation detects inconsistent evidence; it is not authentication of
arbitrarily fabricated counts. The pinned files were separately checked for
unique identities, membership, domain, original counts, integer final counts,
sample limits and the empty missing-item partition, with zero errors.

## Confidence sequence, stopping and aggregation

`scripts/analyze_e1.py:71-133` implements the specified Beta(1/2,1/2)
mixture. Terms with zero coefficients are omitted, so S=0 and S=n do not
produce `0 * infinity`. The n=0 set is the full unit interval. The two root
searches are in log p and log(1-p); neither exponentiates the NLL floor.
At the empirical rate the mixture is at most one, below 1/delta. Each
nontrivial outward tail diverges, so doubling finds a bracket. No saved case
exposes an incorrect bracket or early exit.

Two independent calculations were used: log-odds inversion with log-gamma,
and a compensated finite-product evaluation using the half-integer beta
identities. In particular, start with
`log B(1/2,n+1/2) - log B(1/2,1/2) = sum(j=1..n) log(1-1/(2j))`, then
advance S using the beta recurrence and symmetry. This does not reuse the
production log-gamma subtraction or its root coordinates.

All **2,880** saved E1 per-item/cut intervals agree within
**5.003e-11 nats**, the saved endpoints being rounded to ten decimal places.
They include 189 zero-hit and 1,016 all-hit cases, S=0..64, n=64..2048,
and N up to **953.094329834**. The 72 E1b final pairs reach n=32768 and
S=83, with 14 remaining zero-hit pairs. Their independently recomputed
aggregate bounds agree as well. The Python tests additionally compare the
current mixture and roots with an independent implementation for every
saved count and for n=0, S=0 and S=n at n=32768.

Literal outward rounding is not guaranteed: the bisection returns the
midpoint of its bracket. When adjacent doubles cause `break`, that midpoint
can round to the inside endpoint by one ULP. It cannot produce a material
narrowing on these inputs; the error is far below 1e-6 nats. Returning the
outside bracket endpoint would make this rounding explicitly conservative.

The registered confidence sequence controls all sample times on each fixed
Bernoulli stream. Therefore rounds of 64, the 20-hit stop, the cap, adaptive
selection after seeing E1's aggregate results, and later 512-sample rounds
do not require another error allocation when continuing those streams.
Different items' selection decisions may depend on one another: on the
event all original sequences cover at all times, choosing any of their times
preserves coverage. The union bound requires no independence between items,
cuts or models sharing common random numbers. Actual round sizes change
noise consumption order, not the intended Bernoulli distribution.

All 72 replays match, on the same recorded GPU/library and source code.
The registered fallback to a genuinely restarted stream after a mismatch
would need a separate error-control argument: choosing between an original
and a new stream after observing E1 is not automatically covered by the
original stream's one Ville event. A union over two streams would ordinarily
need an error allocation. That branch was not used in these results; this
audit does not certify its general reused-delta guarantee.

The per-model deltas are exactly:

| Model family | Registered item/cut family | Delta |
|---|---:|---:|
| TinyStories | 64 x 4 | 0.0001953125 |
| WMT14 | 128 x 3 | 0.0001302083333333333 |
| Sudoku | 128 x 4 | 0.00009765625 |

E1b retains these deltas, including WMT14's registered erratum. The first
three excluded TinyStories windows do not reduce its denominator. The
descriptive cuts 5-7 use the same delta but are outside this simultaneous
registered family; their bounds must not be presented as extending that
family's 95% coverage.

On the coverage event, averaging per-item R endpoints and dividing by the
fixed mean N is valid. This is the ratio of means, not the mean of per-item
ratios. E1's `>=0.8` / `<0.5`, E1a's `>0.90`, width `<=2` on `>=80%`,
censoring `>20%`, and the mean-N `<5` not-read condition are implemented
with the registered strict/non-strict inequalities. No released model is
near the not-read threshold. E1a is registered for the 61 untouched
TinyStories items and all WMT14/Sudoku items; its all-64 TinyStories output
is correctly marked descriptive.

The conjunction that all three released models exceed 0.8 is a level-0.05
intersection-union test: if any model fails the target, passing all models
requires falsely passing that model. No further three-model correction is
needed for that conjunction. Each displayed numerical bound is modelwise
95%; a statement that all three stronger numerical bounds simultaneously
cover is different and is not given 95% coverage by this calculation.

## Sampler, canonical floor, split and pilot

`src/ncmr/cached_decode.py:229-262` constructs the canonical coarse
reconstruction, uses the same level inputs/masks/positions as teacher forcing,
and conditions finer calls on sampled ancestors. The first conditional logits
are shared, but each row/position receives distinct noise entries; subsequent
draws do too. Expanded prefix caches are read-only and the expanded mutable
reconstruction is cloned. Sampling is at T=1 without filtering. The model
is in eval mode; fp32 and both TF32-off settings match the registration.
No round-dependent hit rule or autocast context leak was found.

`scripts/e1_equivalence.py:51-92` computes canonical codes and
teacher-forced level NLL with those weights and prefix inputs. Decoding is
deterministic argmax. `Domain.key` compares all 128 TinyStories token ids;
for WMT14 it cuts at first EOS and drops pads; for Sudoku it parses the
board. The same key is applied to the canonical decoded target and each
sample. The canonical code path is consequently a hit in the mathematical
sampler, yielding rho >= exp(-N). Input reconstruction mismatches do not
invalidate this: the target is x-hat, not the original x. All actual Sudoku
targets parse; there are no excluded invalid targets in its 128-item reading.

The structural equality is not a measured bound on machine arithmetic.
JSON contains neither cached/teacher logits nor every sampled code. fp32
matrix kernels can differ with shape, and finite-grid Gumbel-max is not an
exact real-valued categorical sampler. For example, float32 uniform noise
has a finite upper tail, so sufficiently suppressed categories can have
different machine probabilities. The zero clamp also changes the extreme
lower tail. No actual L4 worst-case discrepancy can be recovered here.
Audit 9's random-weight WMT14 discrepancy of about 1.13e-6 is not a uniform
bound on these released checkpoints.

A useful sensitivity calculation, explicitly conditional: if the maximum
per-logit discrepancy is epsilon, log-softmax changes by at most 2 epsilon
per position. The cut-8 suffixes have 404 TinyStories positions and 168
WMT14 positions, giving canonical path log-probability differences at most
808 epsilon and 336 epsilon nats. Epsilon=1e-4 would mean 0.0808 and
0.0336 nats, compared with mean-R margins above 0.8 mean-N of **37.8561**
nats on all TinyStories items, **35.2249** on untouched TinyStories, and
**7.7445** on WMT14. This is a scale comparison, not a measured epsilon or
a proof bounding all finite-Gumbel effects. All 72 saved E1b teacher-NLL
cross-check differences are exactly zero. There is no evidence of a
reading-changing numerical error, but an unconditional GPU error bound is
not established by the saved files.

Largest cut-8 N values are A0 **621.753181458**, J0 **294.792739868**,
and S0 **51.776858330**. The overall maximum 953.094329834 at A0 cut 5
would underflow if its probability were formed directly; the implemented
log-space intersection handles it.

`src/ncmr/data.py:42-68` selects even-index validation stories for val
and odd-index stories for test, then packs windows using the entire selected
split. `examples("val",3)` and `examples("val",64)` therefore have the
same first three windows, even when a short starting story is continued
into later even-index stories. Slicing to the first n stories before
packing would have broken this identity; that is not what the code does.
R046 explicitly describes the first three val windows, so indices 0-2 are
the supported pilot exclusion. No retained pilot result/hash independently
certifies its inputs or counts. "Untouched" means registered window indices
3-63, not a demonstrated absence of overlapping story/token content with
the packed pilot windows.

R046's logged cut-8 counts imply rho=20/256, 24/224, 30/32, namely
0.078125, 0.1071428571, 0.9375, and T=2.549445, 2.233592, 0.064539 nats.
The logged arrays use 32-sample rounds; their differences from E1's
64-sample-round counts are expected and do not identify different windows.
The prototype's ~190 seconds, ~95% time allocation and all hit counts cannot
be independently recomputed from result JSON because none was retained.

## Integrity and provenance

`check_saved` checks arithmetic consistency with the run's own summary and
paired functions; it does not independently validate the statistical method,
item identities, registered row/cut completeness or checkpoint provenance.
The independent tests reimplement that arithmetic rather than rely on it.

The consecutive-cut checks in `scripts/analyze_e1.py:269-299` have the
right directions: R_lo(k+1) > R_hi(k), or U_hi(k+1) < U_lo(k), exposes an
incompatible increase of R / decrease of U. R+U equals total canonical NLL,
so these two checks are algebraically redundant. Overlapping intervals can
hide violations; point estimates are descriptive; only consecutive present
cuts are tested. They do not prove distributional equality or a correct
decoder. Empty intersections are explicitly listed. Replacing one by the
floor point is not the literal empty set, but on the assumed valid floor
that event already entails CS noncoverage, so it does not enlarge the
noncoverage event. No actual empty intersection occurs.

| Model | Point cut pairs with two hits | All adjacent pairs | x-hat = x |
|---|---:|---:|---:|
| A0 | 189 | 216 | 63/64 |
| A1 | 191 | 216 | 63/64 |
| A2 | 185 | 216 | 63/64 |
| A4 | 192 | 216 | 63/64 |
| J0 | 253 | 280 | 100/128 |
| J1 | 255 | 280 | 100/128 |
| J2-F | 254 | 280 | 100/128 |
| S0 | 408 | 408 | 128/128 |

Every point R-rise/U-fall count, beyond-interval violation count and empty
intersection count is **zero**. Saved summary/paired recomputation agrees.

The E1 provenance records `2a27ae9`, NVIDIA L4, torch `2.11.0+cu130`.
TinyStories has two starts and one successful end: the first start at
20:31:40 precedes the restart at 20:54:20; the second session ends at
22:15:32. WMT14 and Sudoku each have one complete session. Thus one
complete session per domain agrees with the log, alongside the disclosed
lost TinyStories attempt. Relevant E1 production files have an empty Git
diff between `dcdb890` and `2a27ae9`.

E1b records `b9eeec2`, NVIDIA L4, torch `2.11.0+cu130`, one session
05:04:33-07:18:24. Relevant E1b/analysis/dependency files have an empty Git
diff between `5ad74b5` and `b9eeec2`. The TinyStories and WMT14 E1 raw
digests equal both their selection and identity pins. Source equality is
not a rerun of the GPU computation or independent weight authentication.
The four raw inputs and two saved analysis outputs are hash-pinned in
`tests/test_audit11_recompute.py`.

## Independent recomputation of R054

Equal item weights; T and N are in nats. E1/E1a columns on nonreleased
checkpoints are applications of the same numerical rules for reference,
not additional registered prediction-1 tests. The all-64 A0 E1a reading
is descriptive. Digits below expose the unrounded computation.

| Model / set | Items | Censored | Mean N8 | R-share bounds | Mean T8 bounds | Width <=2 | E1 / E1a |
|---|---:|---:|---:|---|---|---:|---|
| A0 all | 64 | 6 | 404.349150 | [0.875339522, 0.997472813] | [1.021866, 50.406358] | 50/64 | holds / neither |
| A0 untouched | 61 | 6 | 401.115780 | [0.868488339, 0.997480711] | [1.010526, 52.751402] | 48/61 | holds / neither |
| A1 | 64 | 5 | 393.340779 | [0.894900255, 0.997664749] | [0.918549, 41.340016] | 51/64 | holds / neither |
| A2 | 64 | 7 | 301.952146 | [0.841984875, 0.995933581] | [1.227864, 47.713006] | 48/64 | holds / neither |
| A4 | 64 | 3 | 248.524381 | [0.912868907, 0.996341471] | [0.909234, 21.654201] | 51/64 | holds / neither |
| J0 | 128 | 18 | 130.949573 | [0.763057482, 0.984365316] | [2.047355, 31.027521] | 70/128 | neither / neither |
| J1 | 128 | 15 | 97.940216 | [0.779881143, 0.980229324] | [1.936344, 21.558488] | 69/128 | neither / neither |
| J2-F | 128 | 13 | 114.003851 | [0.809043652, 0.982382055] | [2.008514, 21.769759] | 72/128 | holds / neither |
| S0 | 128 | 0 | 32.227332 | [0.993431732, 0.999963672] | [0.001171, 0.211678] | 128/128 | holds / holds |

A0 untouched censoring is 9.836066%, narrow share 78.688525%; J0 censoring
14.0625%, narrow share 54.6875%. These reproduce 9.8%, 78.7%, 14.1% and
54.7%. No registered prediction-1 verdict is overturned. E1's original
three-domain conjunction is false because J0 is neither.

Prediction 2, all nats, independently propagating opposite T endpoints:

| Comparison | Mean dN +/- SE | dT interval | +/-0.2 abs(dN) | Reading |
|---|---|---|---:|---|
| A2-A1 | -91.388634 +/- 3.278933 | [-40.112151, 46.794456] | 18.277727 | neither |
| A4-A1 | -144.816398 +/- 2.513991 | [-40.430782, 20.735651] | 28.963280 | neither |
| J1-J0 | -33.009357 +/- 1.077950 | [-29.091177, 19.511133] | 6.601871 | neither |
| A0-A1, descriptive | 11.008370 +/- 1.956021 | [-40.318150, 49.487809] | 2.201674 | neither |
| J2-F-J0, descriptive | -16.945721 +/- 0.781486 | [-29.019008, 19.722404] | 3.389144 | neither |

Overall prediction 2 is neither; no comparison carries half. Prediction 3
uses interval midpoints for U where censoring prevents a full point mean:

| Domain | U midpoint order (nats) | N order | Quality order |
|---|---|---|---|
| TinyStories | A4 475.074616 < A2 479.637285 < A1 484.456918 < A0 493.252957 | A4 < A2 < A1 < A0 | A0 < A2 < A1 < A4 |
| WMT14 | J1 159.516653 < J2-F 171.271429 < J0 173.344094 | J1 < J2-F < J0 | J0 < J2-F < J1 |

TinyStories T-lower order is A4 < A1 < A0 < A2; T-upper and T-midpoint
orders are A4 < A1 < A2 < A0. WMT14's T-lower/upper/midpoint orders are
all J1 < J2-F < J0. Sudoku's point U mean is 11.549028. These orders are
descriptive; overlapping U intervals do not establish true pairwise order.
The registered quality numbers are external inputs to this analysis; E1 JSON
cannot independently remeasure Gen-PPL or BLEU.

Exploratory statements:

| Quantity | Independent value |
|---|---|
| TinyStories cut-8 lower-share range | 0.841984875 to 0.912868907 |
| WMT14's 18 censored items' share of total N8 | 21.272651% |
| Their mean N8 versus all J0 items | 198.090277 versus 130.949573 |
| A0's 6 censored items' share of total N8 | 12.088776% |
| Their mean N8 versus all A0 items | 521.395864 versus 404.349150 |
| Cut-5 TinyStories counts | all 4 x 8 descriptive items: 0/2048 |
| Their per-item T lower endpoint | 5.068559664 nats |
| 20/448, TinyStories delta, rho interval | [0.01206066949, 0.10841079109] |
| Width of that T interval | 2.195977930 nats |

| TinyStories model | Cut 9 lower share | Cut 10 | Cut 11 |
|---|---:|---:|---:|
| A0 | 0.955572568 | 0.951760270 | 0.956276857 |
| A1 | 0.975408932 | 0.951340304 | 0.954625472 |
| A2 | 0.901275534 | 0.935825361 | 0.935123037 |
| A4 | 0.947129862 | 0.959694434 | 0.957912413 |

All are >=90%. Sudoku lower shares at cuts 5 through 11 respectively are
0.989176857, 0.993596409, 0.993991325, 0.993431732, 0.992328124,
0.988707122, **0.937784515**. Their minimum rounds to 93.8%.
The censored J0 items alone leave at most 78.727349% of N8 available to
the aggregate lower bound at E1's budget, explaining why 0.8 was impossible
with those zero-hit items. E1b's ability to improve that bound follows from
optional continuation of the same confidence sequences, not a new delta.

## Independent recomputation of R060 and analysis reuse

`readings` builds cut-8 intervals with the original file's registered-cut
delta, then calls the same summary/rule functions. With unchanged counts
its prediction-1 object reproduces E1's reading. For valid inputs,
`substitute` deep-copies the rows and changes only the named model's
continued cut-8 counts: all other levels, counts and models remain intact.
The missing rejection/accounting cases are A11-1.

Mean N8 is unchanged from the preceding table. S0 is unchanged in every
field. The complete post-E1b reading is:

| Released set | Items | Censored before -> after | R-share bounds after | Mean T8 bounds after | Width <=2 before -> after | E1 / E1a after |
|---|---:|---|---|---|---|---|
| A0 all | 64 | 6 -> 5 | [0.893622196, 0.996717211] | [1.327393, 43.013775] | 50/64 -> 57/64 | holds / neither |
| A0 untouched | 61 | 6 -> 5 | [0.887817259, 0.996695588] | [1.325452, 44.998268] | 48/61 -> 54/61 | holds / neither |
| J0 | 128 | 18 -> 9 | [0.859141121, 0.980659247] | [2.532663, 18.445410] | 70/128 -> 109/128 | holds / neither |
| S0 | 128 | 0 -> 0 | [0.993431732, 0.999963672] | [0.001171, 0.211678] | 128/128 -> 128/128 | holds / holds |

After censoring shares are 7.8125%, 8.196721%, 7.03125%, and zero.
Narrow shares are 89.0625%, 88.524590%, 85.15625%, and 100%. These
reproduce R060's rounded percentages. Both TinyStories sets pass E1;
the untouched lower bound remains below 0.90, as does WMT14's.

| Continued model | Censored + wide selected | Width stops | Cap stops | Final samples, including replay |
|---|---:|---:|---:|---:|
| J0 | 18 + 40 = 58 | 39 | 19 | 911424 |
| A0 | 6 + 8 = 14 | 7 | 7 | 245696 |
| Total | 72 | 46 | 26 | 1157120 |

All 72 replay counts match, every saved NLL cross-check difference is zero,
and no item is unreached. A correctly disclosed `complete: false` run may
still meet the bar: unreached items keep E1's counts. Completeness is not
an additional registered statistical threshold. An item outside the selected
plan or an undisclosed missing item must instead be rejected; the current
analyzer accepts those cases.

## Disagreements, unavailable measurements and execution

No scientific table value, registered verdict, delta, reconstruction count,
monotonicity count or continuation count disagrees at the log's stated
rounding precision. The following qualifications prevent treating every
printed phrase as an exact independently measured inequality:

1. R054's completed provenance sessions last 81.19 + 50.51 + 1.46 =
   133.16 minutes. Adding the interrupted start-to-restart interval of
   22 minutes 40 seconds gives **155.8267 minutes**, agreeing with ~156.
   At the repository's stated 1.71 CU/hour this is **4.440 CU**, approximately
   4.4 rather than ~4.3. Actual billed CU cannot be inferred from these files.
2. R054's "at least 93.8%" Sudoku statement is a rounded description of
   **93.778451%**. Likewise cut-5 "T >=5.1" rounds **5.068560**. If those
   decimals are meant as literal exact lower bounds, use "about 93.8%"
   and "about 5.1 nats", or give the unrounded endpoints. This does not
   change a registered reading.
3. R060's 133.8 minutes agrees with provenance process time **133.84**,
   not the result driver's **132.787437** minutes or the item-wall sum
   **132.414362**. They measure different scopes. Provenance time implies
   **3.8144 CU**, or **7.6289** under the doubled estimate, matching 3.8/7.6.
   The 150-minute cap is not exceeded.
4. R046 has no retained pilot JSON. Its logged ratios and window-index
   interpretation can be checked, but its counts, timing and physical inputs
   cannot be independently remeasured from `results/**.json`.
5. R060's scope sentence at `log.md:1477` should explicitly say **the
   suffix NLL of levels 8 and finer conditional on canonical levels 0-7**.
   Its surrounding E1 definitions supply that scope; the sentence alone
   can be read as all code NLL, which these results do not support.

No runnable `python`, `python3`, `py`, or project virtual environment was
available in this sandbox. Consequently neither the repository's Python
suite nor the new Python tests was executed; no CPU/GPU inference or
package installation is claimed. Independent JavaScript computations
checked **6,590** saved numerical values, with maximum absolute difference
**5.003e-11**, and separately checked all 72 record identities and the
completion partition. Git/provenance inspection and source review were
also performed. This is not a substitute claim that pytest passed.

`tests/test_audit11_recompute.py` provides a standard-library-only
recomputation runnable directly as:

```sh
python tests/test_audit11_recompute.py
```

It pins inputs, independently recomputes mixture intervals, original point
summaries/paired blocks, selections, replacements, rules, P2/P3 and integrity
counts. Under pytest it also compares current production roots with the
independent implementation. In a repository Python environment run:

```sh
python -m pytest tests/test_audit11_recompute.py tests/test_audit11_integrity.py tests/test_analyze_e1.py tests/test_e1b.py -q
```

The seven A11-1 cases are marked `xfail(strict=True)`; fixes should make
them XPASS until their expected-failure markers are removed. Existing
`tests/conftest.py` supplies the CPU setup; new tests use pinned result blobs
and require no unpinned model downloads.

## Claim wording and limits

The proposed sentence is supported with an explicit fixed-item/Monte Carlo
qualification. A suitable wording is:

> On the registered validation items in each of the three released NCM
> models (TinyStories, WMT14 and Sudoku-hard), conditional on canonical
> levels 0-7, at least 80% of the canonical suffix code NLL at levels 8 and
> finer is representation choice among code paths decoding to the same
> output. Modelwise 95% lower confidence bounds on mean R8 / mean N8 are
> 0.894, 0.859 and 0.993, respectively (Monte Carlo uncertainty only).
> TinyStories also passes on the 61 registered nonpilot windows, with
> lower bound 0.888.

The target output is the canonical reconstruction x-hat under each domain's
registered key; it is not always the input x. These are fixed released
checkpoints and fixed item sets. The denominator is the conditional
canonical **suffix** NLL. The result does not say 80% of all code NLL is
redundant, bound the marginal text NLL of free generation, quantify quality
improvement, establish the same share on a population of windows, or give
simultaneous 95% coverage to all three stronger numerical endpoints.
It meets the registered cross-domain replication rule, whose unit is the
released model/domain; it is not a multi-training-seed replication claim.

## Finding summary

| ID | Priority | File | One-line issue |
|---|---|---|---|
| A11-1 | P1 | `scripts/analyze_e1b.py` | Records, protocol and missing-item accounting are not validated against the pinned continuation plan. |
