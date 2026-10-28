# Deep audit 10: R056 package delta

Audited revision: `5f0c689fe8bce24677023f9b826f7efceb86a6e1`.
Production delta: `b1b5fc5..4fcf520e707dc8d853b93c339e83f3202da8ef43`.
The audited revision differs from that production head only in
`docs/prompts/codex_audit10_r056.md`; this resolves the two revisions in the
request without changing the production scope. Branch: `codex/audit10`.
Audit date: 2026-10-27.

This report is the audit run record. The explicit restriction permits only
this report and `tests/test_audit10_*.py`, so `log.md` is not edited.
Production code, notebooks, registrations and existing tests are unchanged.

## Verdict

**Two P1 findings, both in E1b's continuation bookkeeping.** No P0 was
identified. E1b must refuse changed continuation inputs and preserve elapsed
time through interruptions before its registered continuation can be trusted.
The missing identity guard is the same defect class accepted and repaired
for W2M-G in `docs/reviews/triage_pr16.md`; that repair does not cover E1b.

No additional P0/P1 was established in the L1R replay/assignment, M0-M7
trainer delta, W2M-G measurements, W2C analysis or requested notebook wiring.
Those conclusions are source review, not an executed certification of the
GPU paths. Python could not run in this sandbox; see execution limits below.

## A10-1 — P1: E1b resumes under changed inputs and runtime

**Locations:** `scripts/e1b_continue.py:162-171`,
`scripts/e1b_continue.py:218-241`, `scripts/e1b_continue.py:261-262`;
downstream substitution at `scripts/analyze_e1b.py:52-63` and
`scripts/analyze_e1b.py:101-102`.

The resume check identifies a run only by its protocol and selected item
lists. It does not identify the source E1 JSON contents, released model,
tokenizer, item data, device or runtime. The saved GPU and torch fields are
overwritten by `res.update(extra or {})` after the check. The notebook's
unchanged-code guard cannot detect any of these changes.

For example, keep E1's hit counts and selected indices but change a selected
row's `level_nll[8]`. `build_plan` produces the same manifest, so a partial
E1b run resumes. Its already completed items used the old stopping floor;
its remaining items use the new one. Analysis accepts the completed records
because it checks only their original hit/sample counts and selection, then
substitutes them into the current E1 NLL rows. Likewise, replacing the WMT
data or released weights can change the canonical reconstruction and the
Bernoulli event being sampled while completed old items are silently reused.

This is distinct from the registered permission for E1b's replay to differ
from E1 under another CUDA library. Such a replay can estimate the same
model/item event from sample 1. It does not authorize changing the event,
model or source intervals within one E1b continuation. Recording
`nll_max_abs_diff_vs_e1` after sampling is not a guard: the reading does not
inspect it. Changed device/runtime also prevents equality with the original
E1b stream, even where the target distribution is intended to remain the same.

**Exposing tests:**

- `tests/test_audit10_e1b.py::test_resume_refuses_changed_e1_floor_inputs_even_when_selection_is_unchanged`
  uses the hash-pinned real J0 result, changes one selected NLL input in a
  temporary copy and exercises the actual `build_plan` and `run` functions.
- `tests/test_audit10_e1b.py::test_resume_refuses_a_changed_runtime_before_drawing_another_item`
  has two cases, changing the actual `torch` or `gpu` metadata supplied by
  `main`, while keeping the selection unchanged.

All three are strict, assertion-only expected failures. They are written but
unexecuted here. They check refusal before another item is sampled, rather
than relying on a replay-mismatch flag after work has already begun.

**Proposed patch, not applied:** hash source E1 files as part of the
selection manifest; pin all E1b asset/data/runtime identity before skipping
completed work. Store digests and logical labels only, never local paths.
The following concrete patch also makes the three probes pass in principle;
that has not been executed in this environment.

```diff
--- a/scripts/e1b_continue.py
+++ b/scripts/e1b_continue.py
@@
-        with open(os.path.join(e1_dir, fn), encoding="utf-8") as f:
+        source = os.path.join(e1_dir, fn)
+        with open(source, encoding="utf-8") as f:
             data = json.load(f)
@@
-        sel[model] = {"domain": dom, "delta": delta, "n_items": len(rows), "censored": cens, "wide": wide, "e1_file": fn}
+        sel[model] = {"domain": dom, "delta": delta, "n_items": len(rows),
+                      "censored": cens, "wide": wide, "e1_file": fn,
+                      "e1_sha256": E1.digest(source)}
@@
-    manifest = {"protocol": PROTOCOL, "selection": sel}
+    manifest = {"protocol": PROTOCOL, "selection": sel, "identity": extra or {}}
@@
-        for key in ("protocol", "selection"):
+        for key in manifest:
             if res.get(key) != json.loads(json.dumps(manifest[key])):
                 raise SystemExit(f"{path}: {key} differs from this run's; refusing to resume")
     else:
-        res = {"protocol": manifest["protocol"], "selection": sel, "items": [], "minutes": 0.0}
+        res = {**manifest, "items": [], "minutes": 0.0}
@@
 # ------------------------------------------------------------------------------------------------ models
+def run_identity(data_path, only, device):
+    from ncmr.data import DATA_DIR
+    from ncmr import ncm_ref_loader as loader
+    hashes, hit_ids = {}, {}
+    for model, domain, _ in DOMAINS:
+        if model not in only:
+            continue
+        art = os.path.join(ROOT, "artifacts", E1.ART[domain])
+        names = ["config.yaml", "tokenizer.pt", "generator.pt"]
+        if domain == "tinystories":
+            names += ["bpe_vocab_8192.pickle", "bpe_merges_8192.pickle"]
+        for name in names:
+            hashes[f"{domain}/{name}"] = E1.digest(os.path.join(art, name))
+        data = (os.path.join(DATA_DIR, "valid_bpe_8192.pickle")
+                if domain == "tinystories" else data_path)
+        hashes[f"{domain}/items"] = E1.digest(data)
+        if domain == "wmt14":
+            from tokenization.hf_tokenizer import HFTokenizer
+            cfg = loader.OmegaConf.load(os.path.join(art, "config.yaml"))
+            pre = HFTokenizer.from_config(cfg.tokenizer.pre_tokenizer)
+            hit_ids[domain] = {"eos": pre.eos_token_id, "pad": pre.pad_token_id}
+    return {"input_sha256": hashes, "hit_key_ids": hit_ids, "device": device,
+            "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
+            "torch": torch.__version__, "cuda": torch.version.cuda}
+
 def make_runner_factory(data_path, e1_dir, device):
@@
-              extra={"gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu", "torch": torch.__version__})
+              extra=run_identity(a.data, only, device))
```

This deliberately refuses legacy E1b state lacking identity. It pins E1b
sessions to one another; it cannot retroactively establish original E1
asset identity, because the released E1 records do not contain those hashes.
Before accepting a mismatched replay, verify the original asset/item lineage
and investigate a material recomputed-NLL discrepancy. Identical NLLs alone
would not prove identical items or weights. Runtime drift versus E1 remains
permitted when the underlying model and event remain verified.

## A10-2 — P1: interrupted work is lost from the cumulative time cap

**Locations:** `scripts/e1b_continue.py:178-200`, especially the item call at
line 189 and elapsed-time writes at lines 194-200.

`minutes` is persisted only after a whole item succeeds or the function
returns normally. An interruption during model loading or an item bypasses
both writes. If an item completes at cumulative minute 149 and the next item
uses five minutes before `KeyboardInterrupt`, the next session loads 149
minutes and starts another item under the 150-minute cap. Repeated failures
can renew this allowance repeatedly. The registered 150-minute cap is
documented by the driver as cumulative across sessions and is treated that
way in the audit requirement. Items reached under that cap determine
which cut-8 counts replace E1's. The extra items therefore depend on whether
the run was interrupted.

There is related stale status: per-item writes do not refresh `not_reached`
or `complete`. A fresh partial file may lack both fields; analysis defaults
the missing fields to an empty remainder and completion. A resumed partial
file can retain older status. The existing interruption test throws without
advancing its clock and does not expose the lost-time case.

This finding concerns the registered cap and reproducible continuation; it
does not assert that an anytime-valid confidence sequence becomes invalid
merely because sampling ran longer.

**Exposing test:**
`tests/test_audit10_e1b.py::test_interrupted_session_time_is_charged_before_another_item_can_start`
uses the actual driver with a deterministic clock. One item consumes 149
minutes, the second consumes five and raises `KeyboardInterrupt`. A resumed
call under the same 150-minute cap must sample nothing and retain 154
minutes. On inspection, the audited implementation instead starts the two
unfinished items from its saved 149 minutes. This strict, assertion-only
expected failure is written but unexecuted here.

**Proposed patch, not applied:** finalize elapsed time on catchable
interruptions, refresh status at every checkpoint, and fail closed after an
unfinalized hard termination. The in-flight marker prevents an uncatchable
runtime loss from silently renewing the budget; its elapsed time would need
reconciliation before continuation. Periodic time checkpoints could reduce
that manual recovery requirement.

```diff
--- a/scripts/e1b_continue.py
+++ b/scripts/e1b_continue.py
@@
     res.update(extra or {})
+    if res.get("session_in_flight", False):
+        raise SystemExit("previous E1b session ended without final time accounting; "
+                         "reconcile cumulative minutes before resuming")
     done = {(r["domain"], r["model"], r["item"]) for r in res["items"]}
@@
     def cumulative():
         return prior + (clock() - t0) / 60.0

-    for entry in plan:
-        key = (entry["domain"], entry["model"], entry["item"])
-        if key in done:
-            continue
-        if cumulative() >= max_minutes or (max_items is not None and processed >= max_items):
-            break
-        if entry["domain"] not in runners:
-            runners[entry["domain"]] = runner_for(entry["domain"])
-        t = clock()
-        rec = {"domain": entry["domain"], "model": entry["model"], "item": entry["item"], "kind": entry["kind"],
-               "e1": entry["e1"]}
-        rec.update(runners[entry["domain"]](entry))
-        rec["wall_s"] = clock() - t
-        res["items"].append(rec)
-        done.add(key)
-        processed += 1
+    def checkpoint(in_flight):
         res["minutes"] = cumulative()
+        res["not_reached"] = [[e["model"], e["item"]] for e in plan
+                              if (e["domain"], e["model"], e["item"]) not in done]
+        res["complete"] = not res["not_reached"]
+        res["max_minutes"] = max_minutes
+        res["session_in_flight"] = in_flight
         _write(path, res)
-    res["minutes"] = cumulative()
-    res["not_reached"] = [[e["model"], e["item"]] for e in plan if (e["domain"], e["model"], e["item"]) not in done]
-    res["complete"] = not res["not_reached"]
-    res["max_minutes"] = max_minutes
-    _write(path, res)
+    checkpoint(True)
+    try:
+        for entry in plan:
+            key = (entry["domain"], entry["model"], entry["item"])
+            if key in done:
+                continue
+            if cumulative() >= max_minutes or (max_items is not None and processed >= max_items):
+                break
+            if entry["domain"] not in runners:
+                runners[entry["domain"]] = runner_for(entry["domain"])
+            t = clock()
+            rec = {"domain": entry["domain"], "model": entry["model"],
+                   "item": entry["item"], "kind": entry["kind"], "e1": entry["e1"]}
+            rec.update(runners[entry["domain"]](entry))
+            rec["wall_s"] = clock() - t
+            res["items"].append(rec)
+            done.add(key)
+            processed += 1
+            checkpoint(True)
+    finally:
+        checkpoint(False)
     return res
```

## Answers to the audit questions

1. **L1R replay.** The replay and training loop use the same `torch.chunk`
   partition, fresh data RNG seeded with the run seed, dedicated mask and
   replacement generators, and the same fp32/eval/TF32-off canonical boundary.
   Fine-draw skipping preserves the registered shape-based multinomial
   consumption assumption. Replay does not draw from the global dropout RNG
   or the training data RNG. The frozen proposer is made before resume.
   Model, optimizer, data and global RNG state are restored before the
   isolated replay; the mechanism RNG state is restored afterward, then the
   plan SHA is checked. The runtime guard compares row keys, canonical/supplied roots and
   canonical L1 blocks for root-corrupted rows before that micro-batch's
   student forward. It does not compare every fine replacement or unselected
   tensor, nor pre-validate every future donor before an earlier receiver
   uses its replayed block. No current code path causing a mismatch was found.
   CUDA multinomial behavior and actual resumed GPU equality remain untested.

2. **Circulation and assignment.** The LP minimizes negative total flow with
   per-arc capacities and exact flow conservation. The network constraints
   yield an integral vertex; rounded integrality, bounds and balance are
   checked. Row selection uses only roots and its separate seeded RNG.
   Each selected donor is used once, receiver counts balance per root and
   `a_donor == u_receiver` is checked. Since `a_i != u_i`, a selected receiver
   cannot donate to itself. Unselected blocks remain unchanged, so the full
   run's L1 multiset is preserved, including assignments across steps and
   micro-batches. Existing tests cover these invariants; they were not run here.

3. **M0-M7 and W2C.** The frozen-wrong refactor retains the original draw
   operations and ordering for existing arms. L1R imports and conditional
   branches introduce no draws for M0-M7. Existing checkpoint extra/RNG
   handling remains in the atomic training checkpoint; M2R adds its plan SHA.
   No changed old-arm data/mask/replacement path was identified. M3-0 differs
   only in clean fraction and thresholds the same mask uniforms, giving the
   required mask superset. Final sample seeds remain fixed across training
   seeds. Bitwise GPU equality is a source-backed expectation, not a measured
   result of this audit.

4. **W2M-G.** Crossed dispatch uses the prefix model for levels below the cut
   and suffix model from the cut; each cell starts the same group generator
   and consumes fixed shapes in level order. Pure cells are reused. Level
   calls use the existing cached decoder. Fresh fp32 evaluation contexts and
   script-level TF32 disabling cover sampling, canonical encoding and scorer
   calls. Group log loss is the log of each group's corpus Gen-PPL. JS uses
   levels 2-5, with equal weighting of each level's position/window mean;
   ratios bootstrap both banks. Shares and their paired delete-one-group
   jackknife match the formulas. Root reliance sums eight L1 code losses
   per window and pairs windows across models. The flag sweep adds the
   constant vector at all levels 1-11, uses the registered banks/seeds and
   conditionally runs the fresh-seed confirmation. Gate/read boundaries match
   the registration. PR16's W2M-G identity fix includes the relevant release,
   validation-data, checkpoint and runtime digests; no further P1 was found.

5. **E1b.** The replay uses the same item seed and 64-sample rounds; the
   continuation extends that generator in 512-sample rounds, with the last
   round capped at total 32768. Selection and stopping use E1a's intersected
   log-space CS and the same delta as reading. A replay mismatch is flagged
   and read from the replay's own sample-1 counts. Substitution changes only
   continued cut-8 entries. The identity and interruption defects are A10-1
   and A10-2. The registration's WMT denominator disagrees with its own count;
   numerical details follow.

6. **W2C analysis and notebooks.** `analyze_w2c` subtracts the aligned 16-group
   vectors before estimating SE and pools the two arms' sample variances
   across training seeds 1-3 exactly as registered. Damage, dissociation,
   M3-0 and L1R boundaries match their inequalities. If MAUVE reverses while
   the other damage conditions hold, the script gives the registration's
   overturned clause precedence; this conservative overlap resolution is
   stated in its docstring. The six new notebook commands match the requested
   runs. L1R executes the gate before training and has the A100/>39-GB guard;
   the run template adds micro-batch 64. Every generated notebook ends with
   release after collection. Provenance and changed-code refusal templates
   are unchanged. Time caps outside E1b are mostly notebook instructions;
   this audit does not certify automatic enforcement of every printed cap.

## Recomputed numbers and disagreements

There are no new W2C/M3-0/W2M-G/E1b/L1R run results to recompute. The requested
baseline inputs are pinned by SHA-256 in `tests/test_audit10_numbers.py`.
The arithmetic was performed from the saved JSONs using PowerShell and
JavaScript, without running the production analyses.

For GPT-2 large, the 16 paired M2-minus-M3 group scores give
**20.936264515 +/- 0.373978485 SE**. Their group means are **42.622016036**
and **21.685751521**. Clean validation NLL differs by **4.035351562 nats per
window**. These round to the registration's 20.94 +/- 0.37, 42.62, 21.69
and 4.04. TinyStories-33M's paired contrast is **33.084144802 +/-
0.719039801 SE**. No disagreement with those printed numbers was found.

| Arm | GPT-2 Gen-PPL | TS-33M Gen-PPL | Clean NLL | Entropy | MAUVE | Bigram gain | Adjacent BPE duplicates/1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 21.710005 | 18.727564 | 753.818680 | 3.988993 | 0.865153 | 2.053709 | 8.256600 |
| M1 | 42.503421 | 51.801998 | 751.167836 | 4.026306 | 0.467740 | 1.697789 | 18.215080 |
| M2 | 42.622016 | 51.735473 | 757.885383 | 4.017824 | 0.411807 | 1.691359 | 21.268135 |
| M3 | 21.685752 | 18.651328 | 753.850031 | 3.991798 | 0.865398 | 2.058693 | 8.298452 |
| M4 | 31.577034 | 32.601384 | 754.372227 | 4.018795 | 0.719970 | 1.894307 | 13.874852 |
| M6 | 42.179520 | 51.436201 | 751.194000 | 4.021290 | 0.453869 | 1.691747 | 18.753709 |
| M7 | 22.068546 | 19.135264 | 741.235609 | 3.993303 | 0.860142 | 2.052358 | 8.346847 |

Gen-PPL/NLL/entropy/MAUVE come from
`results/colab/w2m_L_ts_m*/`; bigram gain and adjacent BPE duplicates come
from `results/w2m_sample_stats.json`. BPE duplicates are distinct from the
word duplicate statistic in `eval_text.py`: M2/M3's latter values are
20.141881/11.157305 per 1k. M5 has no result and is not imputed.

Independent beta-mixture interval recomputation, including the exp(-N)
intersection, gives:

| Model | Items | Full registered cuts | E1a delta | Censored | Additional wide | Selected |
|---|---:|---|---:|---:|---:|---:|
| J0 | 128 | 8, 9, 10 | 0.05/384 | 18 | 40 | 58 |
| A0 | 64 | 8, 9, 10, 11 | 0.05/256 | 6 | 8 | 14 |
| S0 | 128 | 8, 9, 10, 11 | 0.05/512 | 0 | 0 | 0 |

**Disagreement:** `docs/preregistration.md:698` prints J0's denominator as
`128 x 4`. J0 has three full registered cuts, so unchanged E1a uses
`128 x 3`. The literal four-cut denominator selects **59**, not 58, J0
items: item 43 (21 hits/128 samples) has interval width **1.983493 nats**
under E1a and **2.009488 nats** under the printed denominator. The code and
the printed count of 58 follow the unchanged-E1a instruction in the request.
Correct the denominator in the registration before launch; this is an
internally inconsistent specification, not an additional code P1 or a reason
to silently change E1a's reading delta.

## Execution and limits

Neither `python` nor a Python launcher is available on PATH. Invoking the
existing project virtual environment also fails because its configured base
interpreter is unavailable. Sandbox approval is disabled; no host execution,
dependency installation or model download was attempted. **No pytest command
ran, and no passing/xfail runtime result is claimed.**

The added CPU suite contains four strict assertion-only expected-failure
cases for the findings and thirty positive numerical/input-digest checks.
It uses the existing `tests/conftest.py` setup; numerical source files are
hash-pinned. Controller probes need no released model. The existing
hash-checked `released_sudoku` fixture remains unchanged; released-model,
CUDA and end-to-end notebook tests were not executed. Existing subject tests
and earlier audits were read to check prior precision, RNG, checkpoint,
comparison and output-retention defect classes first.

Run where Python and repository test dependencies are available:

```sh
python -m pytest tests/test_audit10_e1b.py tests/test_audit10_numbers.py -q -rx
python -m pytest tests/test_e1b.py tests/test_l1r.py tests/test_w2mg.py tests/test_r056_build.py tests/test_notebook_provenance.py -q
```

The second command also needs the released assets used by the existing
subject tests. Proposed production diffs above have not been applied or
executed. This audit does not claim full-size memory, timing, GPU numerical
equivalence or correctness of hypothetical changed-library replay counts.

Local staging was denied because the worktree's Git metadata is outside the
writable sandbox. The connected GitHub API also requires approval for
publication, while this session's approval policy is never. No audit commit
or remote branch update was made. Only the three permitted working-tree
files were written. Whitespace checks and a scan of the new files for local
paths, contact details and links found no issue.

## Finding summary

| ID | Priority | File | One-line issue |
|---|---|---|---|
| A10-1 | P1 | `scripts/e1b_continue.py:162` | Resume accepts changed E1 inputs, assets, data and runtime under the same selection. |
| A10-2 | P1 | `scripts/e1b_continue.py:189` | Interrupted item time is dropped, renewing the registered cumulative sampling budget. |
