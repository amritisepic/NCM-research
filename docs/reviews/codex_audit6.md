# Deep audit 6: J2-F before launch

Audited commit: **`dd64aa897385813aa74e43fd8d24b480fd5760e0`**, PR 10 head.
Date: 2026-10-24. Registration: J1, J1a, J1b, J1c and J2-F in
`docs/preregistration.md`. Scope is the requested frozen-teacher trainer,
config, notebook generator/notebook, tests, analysis compatibility and R036.
This report records the audit; the explicit file allowlist leaves `log.md`
unchanged. No production file or proposed production patch was modified.

**Two P1 findings prevent clearing the complete registered workflow.** A6-1
violates the run-identity gate before the first checkpoint. A6-2 leaves the
registered J2-F analysis unavailable through the analysis CLI. Neither finding
shows that the frozen teacher itself is wrong, nor invalidates J2's recorded
negative result. No P0 was established.

## Verification

CPU, Python 3.11.16, Torch 2.11.0+cpu, four Torch threads:

```text
python -B -m pytest tests/test_audit6_teacher.py tests/test_audit6_analysis.py -q
7 xfailed

python -B -m pytest tests/test_j2f_frozen_teacher.py tests/test_audit4_protocol.py \
  tests/test_wmt_run_stats.py tests/test_analyze_j1.py tests/test_audit6_gates.py -q
46 passed
```

Runs disabled pytest's cache and used fresh temporary directories. The
hash-pinned released Sudoku assets were read from an existing shared cache:
only `tests/conftest.py`'s asset root was redirected in memory, after checking
that its digest specification equals this checkout's `tests/audit_assets.json`.
Production functions came from the audited checkout. No asset files or
dependencies were added to the checkout. The two new tiny-model resume/RNG
tests also passed on Torch 2.6 CPU.

All seven strict xfails permit only `AssertionError`; dependency, fixture and
runtime errors cannot masquerade as demonstrated defects. A6-1's proposed
guard was substituted into extracted source in memory; its six cases passed
with the xfail markers removed. A6-2's proposed analyzer was exercised from a
temporary file: its unmarked CLI regression passed and its default analysis
remained exactly equal to `results/j1_analysis_s0.json` after JSON normalization.
These proposal checks do not apply patches to production.

Limits: no CUDA/A100 execution, peak-memory measurement, released WMT14
checkpoint replay, fresh corpus BLEU scoring or paired BLEU bootstrap. The
released WMT14 assets and sentence-ordered reference translations are absent
from this checkout. Sudoku fixtures verify the shared implementation and
copying behavior; they are not WMT14 quality evidence.

## A6-1 — P1: an existing manifest is ignored before the first checkpoint

**Locations:** `scripts/train_wmt14_L.py:91-95`.
Existing coverage: `tests/test_j2f_frozen_teacher.py:153-182`.

The manifest is checked only if `ckpt.pt` exists. A preemption before step 250
can leave `run_manifest.json` and logged training steps without a checkpoint.
The next invocation accepts J2's config in a J2-F directory, and vice versa,
then overwrites the manifest. A changed learning rate or changed base-generator
content is also accepted. Previous metrics stay in place and new metrics append
under a different identity. This contradicts both AGENTS.md and J2-F's explicit
directory-refusal gate. It is inherited code exposed by the new arm, not a
claim that restoring an existing checkpoint accepts the wrong arm.

**Test:** `tests/test_audit6_teacher.py::test_recorded_run_without_checkpoint_refuses_changed_identity`
executes the trainer's actual manifest block, creates the state left by a
pre-checkpoint interruption, and changes the arm, recipe or same-size base
generator. Both starting arms are covered: six observed strict xfails.
The original manifest must also remain unchanged upon refusal. Existing tests
created a checkpoint before testing identity, so they missed this case.

**Proposed patch, not applied:**

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -88,9 +88,9 @@
             "base_config_sha256": digest(os.path.join(_base, "config.yaml")),
             "base_sha256": {n: digest(os.path.join(_base, n)) for n in ("tokenizer.pt", "generator.pt")},   # audit 4, A4-2
             "device": dev}
-if os.path.exists(ckpt_path) and not a.calib_steps:
+if not a.calib_steps and (os.path.exists(ckpt_path) or os.path.exists(manifest_path)):
     if not os.path.exists(manifest_path) or json.load(open(manifest_path)) != manifest:
-        sys.exit(f"{a.run_dir} holds a checkpoint of a different (or unrecorded) run: use a fresh run directory")
+        sys.exit(f"{a.run_dir} holds a different (or unrecorded) run: use a fresh run directory")
 if not a.calib_steps:
     json.dump(manifest, open(manifest_path, "w"), indent=1)
 OmegaConf.save(cfg, os.path.join(a.run_dir, "config.yaml"))
```

## A6-2 — P1: the registered analysis cannot select J2-F

**Locations:** `scripts/analyze_j1.py:22-26`, `:90-92`, `:138-158`, `:163-171`.

The CLI accepts a root, output and references, but no candidate arm. Its input
mapping and screen/H12/J1a target are fixed to J2; adding a frozen run next to
the existing runs has no effect. Replacing J2's files would relabel J2-F as J2
and remove the actual moving-teacher comparison. Thus the registered J2-F
screen, H12, capped comparators and four-arm shared-output reading cannot be
obtained by pointing the existing program at the new outputs.

Simply rebinding J2 is insufficient: the historical exploratory results apply
J2's shared-output mask to all three original arms. That mask must stay in
place for J0-J2; J2-F's remaining-sentence diagnostic needs its own mask.
All required hypotheses and pass counts are saved by the trainer, so this is
an analysis integration defect, not lost experiment data or a wrong teacher.

**Test:** `tests/test_audit6_analysis.py::test_registered_frozen_analysis_keeps_the_original_three_arms`
runs the production CLI on four distinguishable synthetic arms. It requests
J2-F, checks its screen/H12/cap and shared-output entries, and requires the
original three arms and their historical remaining-sentence statistics to
stay unchanged. The audited CLI rejects the candidate argument: one observed
strict xfail. The selector below is a proposed interface; an alternative repair
must expose equivalent selection and preservation guarantees.

**Proposed patch, not applied:**

```diff
--- a/scripts/analyze_j1.py
+++ b/scripts/analyze_j1.py
@@ -24,6 +24,7 @@
     "J1": ("w2c_L_wmt_ctrl", "L_wmt_ctrl_s0", 3000),
     "J2": ("w2c_L_wmt_consist", "L_wmt_consist_s0", 2000),
 }
+FROZEN_ARM = ("w2c_L_wmt_consist_frozen", "L_wmt_consist_frozen_s0", 2000)
 SEQ = 11                          # sequential NFE (11 levels)
 H12_PASSES, BLEU_LOSS = 8.25, 1.0   # J1 H12: <= 8.25 mean exact passes on test, BLEU >= J0 - 1
 SCREEN_PASSES = 9.0               # J1 screen (val): passes <= 9.0 and BLEU >= J0 val - 1
@@ -35,7 +36,7 @@
 
 
 def load_arm(root, arm):
-    d, prefix, steps = ARMS[arm]
+    d, prefix, steps = FROZEN_ARM if arm == "J2-F" else ARMS[arm]
     r = json.load(open(os.path.join(root, d, f"{prefix}__result.json"), encoding="utf-8"))
     assert r["steps"] == steps, f"{arm}: {r['steps']} steps, registered {steps}"
     out = {"run": r["run"], "steps": r["steps"], "gpu": r.get("gpu"), "torch": r.get("torch"),
@@ -87,12 +88,18 @@
     return {"diff": bleu(hyp_a, refs) - bleu(hyp_b, refs), "ci95": [float(lo), float(hi)], "n_resamples": n}
 
 
-def analyse(root, refs=None):
+def analyse(root, refs=None, candidate="J2"):
+    if candidate not in ("J2", "J2-F"):
+        raise ValueError(f"unknown candidate {candidate!r}")
     A = {a: load_arm(root, a) for a in ARMS}
-    J0, J1, J2 = A["J0"], A["J1"], A["J2"]
+    if candidate == "J2-F":
+        A[candidate] = load_arm(root, candidate)
+    J0, J1, J2 = A["J0"], A["J1"], A[candidate]
     res = {"arms": {a: {k: v for k, v in x.items() if k not in ("val", "test")} |
                     {s: {k: v for k, v in x[s].items() if not k.startswith("_")} for s in ("val", "test")}
                     for a, x in A.items()}}
+    if candidate != "J2":
+        res["candidate"] = candidate
 
     # registered: J1 sanity checks
     res["sanity"] = {
@@ -100,7 +107,7 @@
                             "pass1_share_pct": 100.0 * J0["test"]["hist"][1] / J0["test"]["n"]},
         "J1_bleu_loss_vs_J0": {s: J0[s]["bleu"] - J1[s]["bleu"] for s in ("val", "test")},
         "J1_recipe_ok": all(J0[s]["bleu"] - J1[s]["bleu"] <= BLEU_LOSS for s in ("val", "test")),
-        "degenerate_pct_J2": J2["degenerate_pct"], "degenerate_ok": J2["degenerate_pct"] <= DEGEN_LIMIT,
+        f"degenerate_pct_{candidate}": J2["degenerate_pct"], "degenerate_ok": J2["degenerate_pct"] <= DEGEN_LIMIT,
         "flop_proxy_equal": J1["generator_flops_proxy_total"] == J2["generator_flops_proxy_total"],
         "exact_ne_sequential": {a: {s: A[a][s]["exact_ne_sequential_idx"] for s in ("val", "test")} for a in A},
     }
@@ -121,41 +128,52 @@
     res["h12_test"]["kill"] = not (saving >= 25.0 and res["h12_test"]["bleu_ok"]) or not res["h12_test"]["below_J1_ok"]
     # registered: J1a comparators at the cap ceil(J2's mean exact passes)
     cap = math.ceil(t["passes"])
-    res["j1a"] = {"cap": cap, "J2_bleu": t["bleu"],
+    res["j1a"] = {"cap": cap, f"{candidate}_bleu": t["bleu"],
                   "J0_at_cap": J0["test"]["capped"][cap], "J1_at_cap": J1["test"]["capped"][cap],
-                  "J2_minus_J0_at_cap": t["bleu"] - J0["test"]["capped"][cap]["bleu"],
-                  "J2_minus_J1_at_cap": t["bleu"] - J1["test"]["capped"][cap]["bleu"],
+                  f"{candidate}_minus_J0_at_cap": t["bleu"] - J0["test"]["capped"][cap]["bleu"],
+                  f"{candidate}_minus_J1_at_cap": t["bleu"] - J1["test"]["capped"][cap]["bleu"],
                   "len_ratio_diff_vs_J0": t["len_ratio"] - J0["test"]["len_ratio"],
                   "len_ok": abs(t["len_ratio"] - J0["test"]["len_ratio"]) <= LEN_TOL,
                   "J0_cap8_bleu_loss": J0["test"]["bleu"] - J0["test"]["capped"][8]["bleu"],
                   "J0_cap8_within_1": J0["test"]["bleu"] - J0["test"]["capped"][8]["bleu"] <= BLEU_LOSS}
     if refs and refs.get("test"):
         hc = {a: A[a]["test"]["_cap"][str(cap)] for a in ("J0", "J1")}
-        res["j1a"]["bootstrap"] = {f"J2_vs_{a}": paired_bootstrap(t["_hyp"], hc[a], refs["test"]) for a in hc}
+        res["j1a"]["bootstrap"] = {f"{candidate}_vs_{a}": paired_bootstrap(t["_hyp"], hc[a], refs["test"]) for a in hc}
     else:
         res["j1a"]["bootstrap"] = "not computed: reference translations not supplied"
 
     # exploratory: input-independent outputs, length, duplication, passes on the remaining sentences
     ex = {}
     for s in ("val", "test"):
-        mask, top = shared_outputs(J2[s]["_hyp"])
+        mask, top = shared_outputs(A["J2"][s]["_hyp"])
         words = {a: np.array([len(h.split()) for h in A[a][s]["_hyp"]]) for a in A}
         ex[s] = {"shared_output_pct": {a: 100.0 * shared_outputs(A[a][s]["_hyp"])[0].mean() for a in A},
                  "J2_top_shared": top,
-                 "J2_passes_on_shared": float(J2[s]["_nfe"][mask].mean()) if mask.any() else None,
+                 "J2_passes_on_shared": float(A["J2"][s]["_nfe"][mask].mean()) if mask.any() else None,
                  "passes_on_rest": {a: float(A[a][s]["_nfe"][~mask].mean()) for a in A},
                  "words_mean": {a: float(w.mean()) for a, w in words.items()},
                  "words_ratio_vs_J0_on_rest": {a: float(words[a][~mask].sum() / words["J0"][~mask].sum()) for a in A},
                  "dup_adjacent_per_1k_words": {a: dup_per_1k(A[a][s]["_hyp"]) for a in A},
                  "identical_to_J0_pct": {a: 100.0 * np.mean([x == y for x, y in zip(A[a][s]["_hyp"], J0[s]["_hyp"])])
-                                         for a in ("J1", "J2")}}
-        ex[s]["saving_on_rest_pct"] = {a: 100.0 * (1 - p / SEQ) for a, p in ex[s]["passes_on_rest"].items()}
+                                         for a in (("J1", "J2", "J2-F") if candidate == "J2-F" else ("J1", "J2"))}}
+        if candidate == "J2-F":
+            frozen = A[candidate][s]
+            frozen_mask, frozen_top = shared_outputs(frozen["_hyp"])
+            ex[s][f"{candidate}_top_shared"] = frozen_top
+            ex[s][f"{candidate}_passes_on_shared"] = (
+                float(frozen["_nfe"][frozen_mask].mean()) if frozen_mask.any() else None)
+            ex[s]["passes_on_rest"][candidate] = (
+                float(frozen["_nfe"][~frozen_mask].mean()) if (~frozen_mask).any() else None)
+            denominator = words["J0"][~frozen_mask].sum()
+            ex[s]["words_ratio_vs_J0_on_rest"][candidate] = (
+                float(words[candidate][~frozen_mask].sum() / denominator) if denominator else None)
+        ex[s]["saving_on_rest_pct"] = {a: 100.0 * (1 - p / SEQ) if p is not None else None for a, p in ex[s]["passes_on_rest"].items()}
     # exploratory: does the 500-sentence curve subset of J2's last curve point equal the final eval's first 500?
     d, prefix, steps = ARMS["J2"]
     cv = json.load(open(os.path.join(root, d, f"{prefix}__hyp_val_step{steps}.json"), encoding="utf-8"))
     n = len(cv["hyp_val"])
-    ex["curve_subset_matches_final"] = (cv["hyp_val"] == J2["val"]["_hyp"][:n]
-                                        and list(cv["nfe_val"]) == J2["val"]["_nfe"][:n].tolist())
+    ex["curve_subset_matches_final"] = (cv["hyp_val"] == A["J2"]["val"]["_hyp"][:n]
+                                        and list(cv["nfe_val"]) == A["J2"]["val"]["_nfe"][:n].tolist())
     res["exploratory"] = ex
     return res
 
@@ -164,11 +182,12 @@
     ap = argparse.ArgumentParser()
     ap.add_argument("--root", default="results/colab")
     ap.add_argument("--out", default="results/j1_analysis_s0.json")
+    ap.add_argument("--candidate", choices=("J2", "J2-F"), default="J2")
     ap.add_argument("--val_refs")
     ap.add_argument("--test_refs")
     a = ap.parse_args()
     rd = lambda f: [l.rstrip("\n") for l in open(f, encoding="utf-8")] if f else None
-    res = analyse(a.root, {"val": rd(a.val_refs), "test": rd(a.test_refs)})
+    res = analyse(a.root, {"val": rd(a.val_refs), "test": rd(a.test_refs)}, candidate=a.candidate)
     os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
     json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
     sc, h, j = res["screen"], res["h12_test"], res["j1a"]
@@ -176,7 +195,7 @@
           f"BLEU {sc['val_bleu']:.2f} (>= {sc['bleu_floor']:.2f}: {sc['bleu_ok']}) -> advance {sc['advance']}")
     print(f"H12 (test): passes {h['passes']:.3f}, saving {h['nfe_saving_pct']:.1f}%, BLEU loss {h['bleu_loss_vs_J0']:.2f}, "
           f"below J1 by {h['passes_below_J1']:.2f} -> kill {h['kill']}")
-    print(f"J1a cap {j['cap']}: J2 - J0 {j['J2_minus_J0_at_cap']:.2f}, J2 - J1 {j['J2_minus_J1_at_cap']:.2f}, "
+    print(f"J1a cap {j['cap']}: {a.candidate} - J0 {j[f'{a.candidate}_minus_J0_at_cap']:.2f}, {a.candidate} - J1 {j[f'{a.candidate}_minus_J1_at_cap']:.2f}, "
           f"len diff {j['len_ratio_diff_vs_J0']:.3f}")
     ex = res["exploratory"]["test"]
     print(f"exploratory (test): shared outputs {ex['shared_output_pct']}, passes on rest {ex['passes_on_rest']}")
```

## Answers to the six audit questions

1. **Teacher identity and resume.** `scripts/train_wmt14_L.py:98-101` loads the
   released files and freezes the tokenizer. Lines 124-128 deep-copy the
   generator before `maybe_resume` at line 135. The copy owns independent
   parameters and buffers, retains device/dtype, is eval/no-grad, is excluded
   from the optimizer, and is absent from `modules`. The generator holds its
   tokenizer in a tuple (`src/ncmr/ncm_ref/models/next_concept.py:80-81`): the
   deepcopy duplicates that tokenizer too, while `net.train()` does not reach
   the original tuple-held tokenizer. Both stay frozen/eval; executed `encode`
   and hierarchy reconstruction paths do not update their codebooks. No
   mutable tokenizer state was found that changes the teacher's map. The
   original tokenizer supplies the prefix, identically to the copied one.
   Autocast is off at teacher decoding, the loader creates fp32 tensors, and
   CUDA TF32 is disabled at lines 69-71. The saved base SHA-256 digests pin
   reconstruction on resume, subject to A6-1. The new CPU round-trip checks a
   rebuilt teacher, student/optimizer state, RNGs and nonzero degenerate totals.

2. **RNG and matching.** Deepcopy and eval teacher decoding consume no random
   numbers. For coincident trajectories/degenerate masks, the first-step test
   observes equal loss, parameter updates, data RNG and Torch RNG in J2/J2-F.
   The NumPy batch RNG is independent of teacher/state/dropout work, so both
   arms always see the same common 128-row draws at equal step indices.
   State draws use `torch.rand(gk, 4)` only for retained sources at line 212.
   If the teachers eventually produce different degenerate masks, `gk` differs
   and later state/dropout draws can diverge even for a source whose trajectory
   coincides. This follows J2's existing skip-degenerate recipe; it is not an
   extra RNG draw introduced by copying the teacher. J1c already discloses that
   padding changes dropout draws relative to removing rows.

3. **Executed proxy and accounting.** The common batch, first 64 TF rows and
   sources 65, 69, ..., 125 are unchanged. The shared `consist_loss` executes
   12 teacher passes over 16 sources, uniformly draws four strictly pre-fixed
   states per retained source, and runs 64 student rows including padding.
   The proxy is `3 * (64 + 64) + 12 * 16 = 576` forward rows per step,
   `2000 * 576` total, equal to J2 and `3000 * 384` for J1. It remains a
   generator proxy, not physical FLOPs or tokenizer/attention-inclusive claim
   cost. `traj_passes` measures detection passes, not all executed teacher
   passes; `traj_states_before_fp` and `traj_degenerate` use the same logic as
   J2. Logging resets interval counters only; checkpointed run counters produce
   `degenerate_pct`. The teacher is frozen/eval while the student consistency
   pass remains in train mode.

   Memory exceeds just the second 120M generator: source architecture
   `src/ncmr/ncm_ref/config/eval/wmt14.yaml` gives 119,963,904 generator parameters
   (479,855,616 bytes) and 1,531,776 buffer bytes; the copied tokenizer adds
   144,002,816 parameters (576,011,264 bytes) and 69,746,696 buffer bytes.
   Approximately **1,127,145,352 extra tensor bytes (1.13 GB, 1.05 GiB)** are
   resident, excluding allocator overhead, autocast caches and workspaces.
   There is no teacher gradient/optimizer state or saved activation graph.
   No evidence establishes an A100-40GB OOM; peak headroom and wall time remain
   unmeasured. Deepcopy adds startup work, and the extra tokenizer remains
   resident. Evaluation was already a significant cost in J2; the proxy does
   not predict its runtime.

4. **Tests and remaining coverage.** The original new tests execute production
   `consist_loss`, loop, constructor, arm/proxy and manifest blocks through AST
   extraction. Target/update tests construct a teacher with a local helper,
   but a separate test runs the real constructor and checks eval/freeze and
   parameter independence. Constructor ordering and model-only checkpoint
   membership have structural assertions. The row-count tests instrument the
   actual decoder with synthetic logits; the padding test checks real upstream
   loss and gradients with dropout disabled. These assertions would catch
   ignored `teacher_net`, student-as-teacher, a shallow parameter copy, missing
   eval/freeze, reduced teacher budget, missing padding, wrong arm/proxy or
   checkpointed teacher. Their manifest assertion covers only directories
   with a checkpoint; A6-1 demonstrates the missed case.

   `tests/test_audit6_gates.py` adds dynamic frozen resume including nonzero
   run totals, coincident-map RNG equality, and hash-pinned released-model
   parameter/buffer/tokenizer storage isolation. Still untested numerically:
   registered gate (1) on a fixed batch from the **released WMT14** checkpoint
   against J0 exact-decode outputs. The original tests establish it using a
   tiny WMT-style upstream model, not the released WMT weights. CUDA TF32,
   device-specific kernels and A100 memory are also untested. No independent
   mutation campaign was performed in this audit; claims of sensitivity above
   describe which assertions observe each property, plus the two demonstrated
   counterexamples and validated proposed repairs.

5. **Saved readings and analyzer.** Final val/test records contain ordered exact
   and all 1-12 capped hypotheses and per-sentence passes, BLEU/length ratios,
   exact/sequential disagreement indices, degenerate totals and the proxy.
   Every curve saves the corresponding ordered hypotheses and pass counts.
   `notebooks/make_notebooks.py:147-162` selects the frozen config and seed 0;
   its export at line 324 includes final and curve hypotheses, manifest and
   result. The checked-in notebook agrees. These are sufficient inputs for
   screen/H12/J1a and shared-output analysis when matching references are
   supplied for bootstrap. **The existing analyzer cannot select J2-F; A6-2
   supplies a backward-compatible proposal.** Claim-only batch-1 timing and
   attention-inclusive cost remain separate required measurements, as in J1a.

6. **R036 recomputation.** The tables below use the saved run records. The
   production analyzer's entire JSON-normalized output exactly equals
   `results/j1_analysis_s0.json`. Sentence-order NFE means, histograms, cap means,
   sharing, duplicates, curve agreement and word counts were also recomputed.
   BLEU/length scalars were compared to saved results, not freshly scored
   without the unavailable references. No rounded numerical disagreement was
   found; the qualifications after the tables matter.

## R036 arithmetic

Inputs: `results/colab/w2c_L_wmt_ctrl/`,
`results/colab/w2c_L_wmt_consist/`, and `results/j1_analysis_s0.json`.

| Arm | Val passes | Val BLEU | Test passes | Test BLEU | Test length ratio |
|---|---:|---:|---:|---:|---:|
| J0 | 10.916000 | 23.554662 | 11.068931 | 22.612264 | 0.930454 |
| J1 | 11.527333 | 22.821645 | 11.622045 | 21.810821 | 0.929534 |
| J2 | 9.357000 | 18.533666 | 8.890110 | 16.022596 | 0.749900 |

| Registered quantity | Recomputed value / reading |
|---|---|
| J0 sanity vs R020 | 11.068931 vs 11.07 passes; 22.612264 vs 22.63 BLEU; no pass-1 sentences |
| J1 BLEU loss, val/test | 0.733016 / 0.801443; both below 1 |
| J2 trajectory sources / degenerate | 32,000 / 0; 0% |
| J1/J2 total generator proxy | Both 83,837,908,942,848,000 FLOPs |
| Exact/sequential disagreements | J1 test index 1294; J2 val indices 531, 1118; others empty |
| Val screen | 9.357 > 9; 18.533666 < floor 22.554662; fails both |
| Test H12 | Saving 19.180819%; BLEU loss 6.589668; passes below J1 by 2.731935; killed |
| J1a cap | ceil(8.890110) = 9 |
| Cap-9 J0 | BLEU 22.512861; passes 8.985348 |
| Cap-9 J1 | BLEU 21.562753 |
| J2 minus cap-9 J0/J1 BLEU | -6.490265 / -5.540157 |
| J2 minus J0 length ratio | -0.180555; outside 0.02 |
| J0 cap-8 | Loss 1.108609 BLEU; passes 7.999334; saving 27.278782%; outside 1 BLEU |
| J0 cap-9 | Loss 0.099403 BLEU; saving 18.315018% |
| Paired BLEU bootstrap | Not computed: matching references absent |

| Exploratory quantity | Recomputed value |
|---|---|
| Shared val outputs | 260/3000 = 8.666667%; two strings, counts 233 and 27 |
| Shared test outputs | 487/3003 = 16.217116%; two strings, counts 428 and 59 |
| Passes on shared val/test sentences | 3.015385 / 2.928131 |
| First test pass emits most-shared string | 352 sentences |
| J0/J1 shared strings emitted at least 5 times | None |
| Remaining test sentences | 2516; J2 passes 10.044118, saving 8.689840%; matched J0 passes 11.040541 |
| J2/J0 words on remaining test sentences | 42,658 / 46,839 = 0.910737; 8.926322% fewer |
| Adjacent duplicates per 1k words, J0/J1/J2 | 17.118607 / 21.021021 / 25.624557 |
| J2 curve passes, steps 500/1000/1500/2000 | 9.814 / 9.758 / 9.684 / 9.638 |
| J2 curve shared percentages | 0 / 1.8 / 1.6 / 2.8 |
| J1 curve passes, steps 500 to 3000 | 11.344 / 11.438 / 11.436 / 11.448 / 11.522 / 11.454 |
| J2 final curve subset | Exact hypotheses and passes equal final val's first 500; BLEU 24.193896 vs full-val 18.533666 |
| Result wall seconds, J1/J2 | 3071.235334 / 4201.913802; ratio 1.368151 |
| Provenance elapsed, J0/J1/J2 | 594 / 3091 / 4235 seconds; total 7920 seconds = 132 minutes |
| CU at recorded 5.4 CU/hour | 11.88, agreeing with approximately 11.9 |

The two timing clocks differ: internal result walls sum to 130.582377 minutes,
while provenance includes launcher/setup and totals 132. J2-F's motivating
70.6-minute J2 cost uses provenance (70.583333 minutes), while R036's 4202
seconds uses the result clock. They are consistent.

R036's phrase **"exact excludes them"** should not be read as numerical
filtering: the saved aggregate metrics and analyzer include the disagreement
indices. Excluding them would give J1 test passes 11.621919 and J2 val passes
9.356237. J1c requires reporting disagreements and restricting the *exactness
claim* to agreeing sentences, not recomputing all aggregate BLEU/pass metrics
on a filtered corpus. The exclusions do not change any registered verdict;
label the published aggregates as all-sentence results.

The R036 Sudoku addendum was also checked against
`results/onpolicy_repro/{uniform,plausible}_wrong_adv1_s0.json` and its prior
comparison files, without auditing those trainers:

| Quantity | Recomputed |
|---|---|
| Run recipe | 200 steps, batch 16, lr 1e-4, warmup 20, seed 0, 300 evaluation puzzles, 8 threads |
| Baseline / uniform / plausible clean NLL | 43.191833 / 45.891323 / 75.648827; increases 2.699490 / 32.456993 |
| Prior plausible clean NLL | 75.207001 |
| L8-L10 excess NLL | 24.617999 nats |
| Baseline / uniform / plausible summed entropy | 2.087200 / 2.334300 / 3.162700; prior plausible 3.146800 |
| Uniform / plausible logged-loss mean | 0.307552 / 0.398905; mean of 21 logged values, not last-step loss |
| Final uniform greedy / sampled solve rate | 83.666667% / 82.666667% |
| Greedy change uniform / plausible, points ± paired SE | -4.666667 ± 1.468611 / -5.000000 ± 1.502506 |
| Greedy plausible minus uniform change | -0.333333 ± 1.529956; prior +3.666667 ± 1.587179 |
| Sampled change uniform / plausible, points ± paired SE | -4.333333 ± 1.581550 / -6.333333 ± 1.558830 |
| Sampled plausible minus uniform change | -2.000000 ± 1.762991; prior -1.000000 ± 1.974475 |
| Per-level corrupted fractions | Identical at corresponding logged steps |
| Completed-run internal wall | 1468.5 + 1871.3 = 3339.8 seconds = 0.927722 hours |

All these rounded assertions agree with R036. Historical suite counts,
process-failure anecdotes, review assertions and approximate total CPU time
have no corresponding machine-readable result record sufficient to recompute
them; they were not treated as freshly verified experiment numbers.

## Summary

| ID | Priority | File | One-line issue |
|---|---|---|---|
| A6-1 | P1 | `scripts/train_wmt14_L.py:91` | A recorded run accepts a different identity when its first checkpoint is absent. |
| A6-2 | P1 | `scripts/analyze_j1.py:90` | Analysis cannot select J2-F while retaining the original J0-J2 readings. |
