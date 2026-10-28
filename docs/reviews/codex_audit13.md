# Deep audit 13: W2M-GR delta

Audited commit: `a949044933a7a033b959ecf58f8208d67932e90d`.
Base: `733ea77`; requested branch: `codex/audit13`.
Scope: the PR 18 delta, its unchanged W2M-G dependencies, the W2M-GR
registration, and W2C's six saved sanity-reference inputs. No production
code, notebook, registration, log or result file was changed.

**One P1; no P0 found.** The reduced evaluator preserves the retained
computations by source inspection. The new analyzer's admission check can
accept explicitly unregistered inputs and produce a registered reading.
Fix A13-1 before relying on that reading. This audit does not establish that
the current evaluator actually generates the malformed inputs below.

## A13-1 — P1: result admission ignores the declared bank and replay design

**Locations:** `scripts/analyze_w2mgr.py:43-49` (partial registered schema),
`scripts/analyze_w2mgr.py:211-235` (protocol and cell checks), and
`scripts/analyze_w2mgr.py:247-254` (shares checked using unchecked vectors).
The accepted results reach registered readings at
`scripts/analyze_w2mgr.py:312-331`.

`check_result` checks the top-level counts/seeds, cell names, and advertised
`n_groups`/`n`. It never checks `protocol.clean_bank` or a cell's saved
`spec`, and never requires eight actual per-group score entries. Its share
recomputation takes the means of whatever vectors are supplied. Thus all
of these declared departures pass the current checks:

- `protocol.clean_bank = 'examples("test", 512)'`, despite the registered
  clean validation bank and the prohibition on reading/tuning on test;
- `cells.m2to3_c2.spec.cut = 4`, while the filename/key still says cut 2;
- that cell's `spec.seeds = [8, ..., 15]`, despite the required common
  replay seeds 0–7;
- four actual score groups in every cell, with means, SEs, shares and sanity
  consistently recomputed from those four groups, while the advertised
  counts remain eight groups and 2048 samples.

With other checks satisfied, `analyze_w2mgr` marks these inputs available
and makes localization/(i)/(iii) readings. The wrong-cut case misattributes
levels; the wrong-seed case loses the declared pairing; the four-group case
uses an incomplete registered measurement; the test-bank case admits an
explicit split violation. This defeats the new analyzer's stated role of
withholding a reading for nonregistered or incomplete runs. These are
detectable contradictions already present in the JSON, not a demand to
authenticate arbitrarily fabricated measurements.

**Exposing tests:**
`tests/test_audit13_analysis.py::test_declared_nonregistered_replay_is_rejected_before_any_reading`
has four strict assertion-only xfails, one per departure. Each first checks
an accepted control, then passes three saved files through the actual
analyzer and requires rejection of seed 2 and no localization reading.
The control uses the hash-pinned real seed-0 scores with metadata adapted
solely for this schema test; it is explicitly not a seed-1–3 measurement.
All six W2C sanity sources are also hash-pinned. The tests are unexecuted
here; the failure conclusion is from the source paths above.

**Proposed patch, not applied:** complete the fixed protocol fields,
compare the actual per-cell specs, and validate the group vectors before
the existing share check. Use shares recomputed from validated vectors for
classification, so the admission tolerance is not a second decision
boundary. This patch keeps the standalone analyzer dependent on NumPy
only. The existing stored finite jackknife SEs remain reported; their
independent recomputation can be added as a further consistency check.

```diff
--- a/scripts/analyze_w2mgr.py
+++ b/scripts/analyze_w2mgr.py
@@
 REGISTERED = {"n_clean": 512, "group_n": 256, "bank_seeds": [900, 901], "replay_seeds": list(range(8)), "cuts": [2, 4, 6],
@@
               "scorers": {"gpt2-large": "gpt2-large", "TinyStories-33M": "roneneldan/TinyStories-33M"}}
+REGISTERED.update({
+    "clean_bank": 'examples("val", 512)',
+    "cells": {"groups_per_cell": 8, "samples_per_group": 256, "crossed_cuts": [2, 4, 6],
+              "cut_meaning": "prefix model samples levels 1..c-1, suffix model c..11"},
+    "sampler": "ancestral, T=1, Gumbel-max, one generator per group seeded with the group seed (common random numbers)",
+    "ci": "percentile bootstrap over windows, both banks resampled",
+})
@@
 def _finite(x):
     return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

+
+def expected_cell_spec(key):
+    if key in ("m2_m2", "m3_m3"):
+        prefix = suffix = key[:2]
+        kind, cut = "pure", None
+    else:
+        pair, value = key.rsplit("_c", 1)
+        prefix, suffix = {"m2to3": ("m2", "m3"), "m3to2": ("m3", "m2")}[pair]
+        kind, cut = "crossed", int(value)
+    return {"key": key, "kind": kind, "prefix": prefix, "suffix": suffix,
+            "cut": cut, "f": None, "seeds": list(range(8)), "group_n": 256}
+
+
+def check_cell_scores(cell):
+    bad = []
+    for sc in SCORERS:
+        metric = cell.get("genppl_" + sc)
+        if not isinstance(metric, dict):
+            bad.append(f"missing {sc} metrics")
+            continue
+        ppl, loss = metric.get("per_group"), metric.get("log_loss_per_group")
+        if not all(isinstance(v, list) and len(v) == CELL_GROUPS
+                   and all(_finite(x) for x in v) for v in (ppl, loss)):
+            bad.append(f"{sc}: need eight finite paired group scores")
+            continue
+        if any(x <= 0 for x in ppl):
+            bad.append(f"{sc}: nonpositive Gen-PPL")
+            continue
+        if not np.allclose(np.log(ppl), loss, atol=1e-12, rtol=0):
+            bad.append(f"{sc}: log losses differ from the group Gen-PPL")
+        sd = float(np.std(ppl, ddof=1))
+        expected = {"mean": float(np.mean(ppl)), "sd": sd, "se": sd / math.sqrt(CELL_GROUPS)}
+        for q, value in expected.items():
+            if not _finite(metric.get(q)) or not math.isclose(metric[q], value, rel_tol=1e-12, abs_tol=1e-12):
+                bad.append(f"{sc}: {q} differs from the eight group scores")
+    return bad
+
@@
     for k in CELL_KEYS:
         c = cells.get(k)
-        if isinstance(c, dict) and (c.get("n_groups") != CELL_GROUPS or c.get("n") != CELL_N):
+        if not isinstance(c, dict):
+            bad.append(f"cell {k}: not an object")
+            continue
+        if c.get("spec") != expected_cell_spec(k):
+            bad.append(f"cell {k}: spec differs from the registered replay")
+        if c.get("n_groups") != CELL_GROUPS or c.get("n") != CELL_N:
             bad.append(f"cell {k}: n_groups {c.get('n_groups')}, n {c.get('n')} != {CELL_GROUPS}, {CELL_N}")
+        bad.extend(f"cell {k}: {why}" for why in check_cell_scores(c))
@@
 def summarise_seed(r):
@@
-    sh = r["ii"]["shares"]
+    # Retain the stored jackknife SEs; classify using recomputed points.
+    sh = {sc: {str(c): dict(r["ii"]["shares"][sc][str(c)]) for c in CUTS} for sc in SCORERS}
+    for sc in SCORERS:
+        ll = {k: r["cells"][k]["genppl_" + sc]["log_loss_per_group"] for k in CELL_KEYS}
+        for c in CUTS:
+            values = point_shares(ll["m2_m2"], ll["m3_m3"], ll[f"m2to3_c{c}"], ll[f"m3to2_c{c}"])
+            sh[sc][str(c)].update(zip(("pi", "sigma", "interaction"), values))
```

## Answers to the audit questions

1. **Reduced versus full computation.** The retained eight cell specs are
   the full run's first eight, with identical order, cuts, groups and seeds
   (`scripts/w2mg_eval.py:229-234`). Every sampling call creates its own
   seeded generator and fixed level shapes (`src/ncmr/w2mg.py:94-115`);
   removing later M4 calls cannot advance those streams. The retained banks
   remain released, M2, M3, each at seeds 900/901, before the omitted M4 bank.
   JS uses the same clean/generated arrays, model pair and window batches;
   each bootstrap call initializes seed 0 independently. Root reliance
   draws its roots once before iterating models
   (`src/ncmr/w2mg.py:221-237`), with its own seed-2027 generator; its
   per-model forward batches do not combine models. Removing M0/M4 cannot
   alter retained roots or batch composition. The existing tiny CPU tests
   compare retained cells and (i)/(ii)/(iii) outputs exactly and compare
   saved bank, JS and rho arrays elementwise. Those tests were read, not run;
   GPU bitwise equality is not measured by this audit.

2. **Default path and legacy resume.** For the intended seed-0 W2M-G path,
   sampling and analysis arithmetic/order remain unchanged; `arms` is added
   to the protocol and manifest. Old manifests lacking `arms` are refused
   by exact comparison (`scripts/w2mg_eval.py:466-476`). This is a safe
   compatibility break, not silent mixed computation. An unfinished old
   directory needs the original code to resume or a fresh directory under
   this code; do not delete its manifest to reuse cached cells. The existing
   completed exploratory seed-0 result remains readable beside W2M-GR and
   is never pooled. Full-arm runs requested at seeds >= 1 now use W2C sanity
   references too; the registered original W2M-G run is seed 0.

3. **Sanity reference.** `w2c_reference` constructs the correct arm/seed
   filenames, reads `final_eval["genppl_gpt2-large"]`, requires a positive
   finite value, and has no seed-0/W2M fallback
   (`scripts/w2mg_eval.py:88-105`). `run` reads both references before making
   output directories or writing manifests/cells (`scripts/w2mg_eval.py:483`).
   Loading the model environment happens earlier but writes no run output.
   The supplied six repository files' run/arm/seed metadata and group means
   agree with their filenames. The sanity miss remains nonfatal. The reader
   relies on those repository file contents; the analyzer validates the
   source path and self-consistency, not the reference against a second
   independent copy of W2C.

4. **Registered readings.** Localization uses every GPT-2-large seed,
   TinyStories-33M seed means, both interaction cuts, and the registered
   overturn disjunction on GPT-2-large seed means. Seed 0 is separate.
   (i) requires all three banks in every seed, `r >= 3` and `lo > 1.2`;
   (iii) requires `lo > 0` in every seed. The inclusive/strict operators
   match apart from the explicit `EPS = 1e-12`: inclusive bounds expand
   by EPS, and strict interaction `< 0.05` becomes `< 0.05 - EPS`.
   This changes mathematical decisions within that narrow band, far below
   the numerical resolution justified for these estimated shares; no
   consequential threshold change or separate P1 is established. The added
   boundary tests characterize that band and test adjacent float values at
   the strict (i)/(iii) endpoints. Shares are checked against cell means,
   but only after the incomplete admission checks described in A13-1.
   Existing guards reject an honestly labeled seed-0, full-arm, incomplete,
   non-L4 or changed-stored-share result. A13-1 covers affirmative protocol
   contradictions they miss. These checks do not cryptographically prove
   checkpoint identity or protect against wholly fabricated JSON.

5. **Notebook.** The command passes `--arms m2,m3`, seeds are `[1, 2, 3]`,
   and run directories are `w2mgr_eval_s<s>`. Collection produces exactly
   `results/colab/w2mgr_eval/w2mgr_eval_s<s>__result.json`, as the analyzer
   expects. The guard matches the whole alphanumeric token `L4`; `L40`,
   `L40S`, T4 and no GPU fail. It precedes the data-download/run cells.
   The setup still clones/pulls the research branch; inherited provenance
   records code, command template, seed, device and session outcomes, and
   compares earlier-session code on resume. Eval manifests additionally
   pin protocol, seed, arms, checkpoints, release/data digests and runtime.
   The final cell releases the runtime after successful collection.
   The generated notebook has the expected seven cells and no executed
   outputs. Cell group losses, shares/SEs, ratios/CIs, rho/CIs and sanity
   are in collected result JSONs. Samples, banks and raw JS/rho arrays remain
   in the run's `cells` directory on Drive; the collect template does not
   copy them into `results/colab`. No quantity required by the registered
   reading is missing. Printed time caps are instructions rather than an
   automatically enforced cumulative timer, as in the unchanged W2M-G
   template; this audit does not certify cap enforcement.

6. **Precision and checkpoint format.** No changed autocast lifetime,
   context reuse, TF32 or scorer branch was found. Sampling and bank/root
   forwards enter fresh fp32 evaluation contexts; the driver disables TF32
   before these operations. Removing models introduces no new precision
   path. `train_gen.py:444-446` writes the model state under `model`, which
   `load_env` reads into a released-model copy and puts in eval mode. M2/M3
   require no flag state. No training/RNG-checkpoint changes are in scope.

## Recomputed references and disagreements

No W2M-GR measurements or dependent log numbers exist yet. The requested
recomputation is W2C's final GPT-2-large Gen-PPL sanity references. Independent
PowerShell arithmetic over the six saved 16-group arrays gives:

| Seed | G(M2) | G(M3) | Paired M2 − M3 | Paired SE |
|---|---:|---:|---:|---:|
| 1 | 39.790341371937075 | 21.425413127057226 | 18.364928244879852 | 0.32466546963889104 |
| 2 | 41.30791192128558 | 21.801928440382500 | 19.505983480903080 | 0.30563694905006000 |
| 3 | 40.560382770864110 | 21.803127274556942 | 18.757255496307170 | 0.32042041047383496 |

Each reference equals its stored `final_eval` mean within `8e-15`. The
references, contrasts and SEs round to R062's printed values. **No
disagreement found.** Sources are
`results/colab/w2c1_L_ts_m2/L_ts_m2_frozen_pw_s{1,2,3}__result.json` and
`results/colab/w2c1_L_ts_m3/L_ts_m3_uniform_wrong_s{1,2,3}__result.json`;
all six digests are pinned in `tests/test_audit13_numbers.py`.

## Execution, artifacts and limits

Neither Python nor its launcher is available on PATH. The existing project
virtual-environment launcher cannot find its configured base interpreter.
**Repository Python and pytest could not run in this sandbox.** No passing
or observed xfail result is claimed. CPU tests use the shared
`tests/conftest.py` setup and hash-pinned repository JSON assets; no model
download is needed for the new tests. The report's proposed production
diff is unapplied and unexecuted. CUDA, private Drive checkpoints and an
end-to-end Colab session were not available for verification.

Run in the repository's usable Python environment:

```sh
python -m pytest tests/test_audit13_analysis.py tests/test_audit13_numbers.py tests/test_audit13_boundaries.py -q -rx
python -m pytest tests/test_w2mgr.py tests/test_w2mg.py tests/test_r056_build.py tests/test_notebook_provenance.py -q
```

At this commit, the four A13-1 probes should be xfails. After the proposed
fix, strict xfail should report XPASS until those markers are removed in
the fix's regression suite. The positive numerical and boundary controls
should pass. Existing audit-10 tests target unchanged, separately scoped
defects; they are not new W2M-GR findings.

Only this report and `tests/test_audit13_*.py` were written. Local Git
staging was denied because this worktree's Git metadata is outside the
writable sandbox. The connected GitHub create-tree action was also denied:
it requires approval, but this session's approval policy is `never`.
**No commit or push was made.** The requested files remain in the worktree
on `codex/audit13`; that branch still points at the audited commit.

## Finding summary

| ID | Priority | File | One-line issue |
|---|---|---|---|
| A13-1 | P1 | `scripts/analyze_w2mgr.py:211` | Analyzer accepts explicitly wrong clean splits, replay specs and incomplete group vectors. |
