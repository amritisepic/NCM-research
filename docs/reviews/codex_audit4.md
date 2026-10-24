# Audit 4: WMT14 tier L before wave 2c

Subject: **`c91126a6bcf229721ef59379a27999b1e9c8f0ba`**,
`claude/ncm-research` at the requested point. Audit date: 2026-10-24.
Scope: `scripts/train_wmt14_L.py`, `scripts/prep_wmt14.py`,
`src/ncmr/jacobi_train.py`, `src/ncmr/train.py`, both `configs/scale/L_wmt_*.yaml`,
and the WMT `w2c_*` entries and collection code in `notebooks/make_notebooks.py`;
J1, J1a and J1b. Production files were not edited. Existing unrelated work was
preserved and excluded from the audit commit. The explicit file allowlist takes
precedence over the general instruction to append this run to `log.md`.

**Hold launch for the three P1 findings below.** The teacher, gradient path,
training-state resume and source conditioning pass the bounded CPU checks.
Two P2 findings concern an edge-case NFE and missing intermediate evaluation
records; final evaluation records are present. None of these findings establishes
that an unlaunched WMT experiment has already produced invalid results.

Tests: `python -B -m pytest tests/test_audit4_protocol.py tests/test_audit4_numbers.py -q`.
The subject produces **18 passed, 7 strict xfails**. Adding
`tests/test_audit2_launch.py` produces **37 passed, 7 strict xfails**.
All xfails are restricted to `AssertionError`; import/setup/runtime failures do
not count as demonstrated defects. `AUDIT4_REF=c91126a` pins extracted subject
functions; the default uses the working tree so a repair makes its strict xfail
XPASS. Shared imports must also come from the subject checkout when pinning a
different checkout. Existing audit-2 tests are regression evidence, not an attempt
to reconstruct the historical R028 test transcript.

All five cumulative diffs below were also parsed and applied **in memory only**;
the notebook collection string was syntax-checked. With expected-failure markers
removed for that proposal check, all **23 protocol cases passed**. This checks
the proposed repair against the counterexamples; it does not apply it or replace
integration and CUDA testing.

Runtime: Torch **2.6.0+cu124**, CPU, four Torch threads. The small upstream
WMT-style network uses 11 levels, a source-prefix level-0 head, dropout 0.1,
64-token targets and EOS/PAD. A second check uses the released weights verified
by `tests/conftest.py` against `tests/audit_assets.json`; those are shared-primitive
fixtures, not WMT14 quality measurements. The released 120M WMT weights and a
CUDA training step were not tested. A CUDA initialization warning was emitted;
all tests explicitly used CPU.

## Findings and proposed patches

Locations below refer to the subject. P1 blocks the registered comparison or its
reproducible continuation; P2 is a bounded edge/reporting defect. Patches are
**proposals, not applied**, cumulative in the order printed. They require normal
integration, review and CUDA validation before launch.

### A4-1 — P1: validation NLL uses training precision

**Location:** `scripts/train_wmt14_L.py:251-252` (called at `:290` and `:306`).

`val_nll` enters a fresh `trainer.ctx`, which enables bf16 on CPU/A100/L4 or
fp16 on older CUDA devices. `.float()` converts already rounded logits; it does
not rescore the generator in fp32. This is the WMT counterpart of audit 3's
text-trainer scoring issue. It violates the required fp32 scoring protocol and
makes a continuation across bf16/fp16 hardware change the reported validation
quantity. BLEU and exact decoding have a separate autocast-off region and are
not implicated by this finding.

**Test:** `test_val_nll_generator_forwards_are_fp32` observes enabled autocast
at the actual generator forward. The tiny model reports 548.315185546875
nats/sentence under the current path versus 548.3743286132812 in fp32, a
-0.05914306640625 difference. This is an execution counterexample, not an estimate
of the error on the released model.

**Proposed patch:**

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -248,7 +248,7 @@
     for s in range(0, len(vs), 100):
         z, pm = prefix(vs[s:s + 100])
         idx = canon(vt[s:s + 100])
-        with trainer.ctx:
+        with torch.autocast(device_type=dev, enabled=False):
             lg = parallel_logits(net, z, idx, pm).float()
         tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), idx.reshape(-1), reduction="sum").item()
     net.train()
```

### A4-2 — P1: a different base checkpoint of the same size passes the manifest

**Location:** `scripts/train_wmt14_L.py:83-89`, especially `:85`.

The config and `.npz` training archive are content-hashed, but `tokenizer.pt` and
`generator.pt` are identified only by byte size. Two valid Torch state files with
different same-shaped weights can have identical sizes. The real manifest guard
accepts both replacements in the CPU counterexample. A changed frozen tokenizer
therefore changes inputs, canonical targets and trajectory targets during an
apparently compatible continuation. Different base generator weights can also
give the two arms different starting models without detecting that difference.

The size-only relaxation was explicitly accepted in the audit-2 triage and S1c /
J1b. This is not a claim that the code diverges from that amendment: it satisfies
its weak size check but fails the stronger AGENTS.md and present audit requirement
to refuse a different base checkpoint. Resolve that conflict before launch.

**Test:** `test_manifest_refuses_changed_same_size_base_weights`, parametrized
over both checkpoint files, serializes different valid weights with equal file
length and executes the actual manifest block. Changed arm, recipe, base config,
training archive and `--eval_only` on a trained directory are separately refused
by passing tests.

**Proposed patch:** hash the actual checkpoint contents. Checking a cached digest
without validating the bytes would preserve this defect.

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -82,7 +82,7 @@
 _base = os.path.join(ROOT, "artifacts", "wmt14")
 manifest = {"config": OmegaConf.to_container(cfg, resolve=True), "data_sha256": digest(a.data),
             "base_config_sha256": digest(os.path.join(_base, "config.yaml")),
-            "base_bytes": {n: os.path.getsize(os.path.join(_base, n)) for n in ("tokenizer.pt", "generator.pt")},
+            "base_sha256": {n: digest(os.path.join(_base, n)) for n in ("tokenizer.pt", "generator.pt")},
             "device": dev}
 if os.path.exists(ckpt_path) and not a.calib_steps:
     if not os.path.exists(manifest_path) or json.load(open(manifest_path)) != manifest:
```

### A4-3 — P1: dropping degenerate sources breaks the reported FLOP match

**Locations:** `scripts/train_wmt14_L.py:115`, `:172-185`, `:271-274`, `:318`.

The teacher correctly executes all 12 passes over 16 sources, even when stopped.
However, each dropped source also removes four gradient-bearing consistency rows.
`fl_step` and `generator_flops_proxy_total` still charge all 64 consistency rows.
With d degenerate sources in a step, the executed registered proxy is
`3 * (64 + 4 * (16-d)) + 12 * 16 = 576 - 12d` forward rows. It is 564 for one
degenerate source and 384 for an all-degenerate step, while the report says 576.
The latter is one-third less work. Even a run below J1b's 1% degenerate-source
threshold can violate the exact compute match. This is conditional evidence;
the unlaunched arms' degenerate rate is unknown.

**Test:** `test_realized_generator_rows_match_registered_proxy` instruments
actual teacher and consistency forward row counts for d = 0, 1, 16. The d = 0
case passes; d = 1 and d = 16 are strict xfails. This counts the registered
forward/backward proxy, not a profiler measurement of physical FLOPs.

**Proposed patch:** retain the registered 2000-step schedule and execute zero-loss
padding rows for dropped sources. They contribute no consistency labels or loss,
but keep a full batch through the student graph, including the all-degenerate
case. Slice valid rows for CE; use a finite vocabulary column for the all-dropped
zero so padded `-inf` vocabulary logits cannot make `0 * inf` NaN. This is a
proposal for maintaining the registered proxy, not an unregistered resampling
policy. The degenerate-source count and the J1b 1% reporting rule still apply.

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -174,15 +174,22 @@
     traj_stats["states_before_fp"] += float(first_fp.sum())
     traj_stats["n"] += g
     traj_stats["degenerate"] += int((~keep).sum())
+    # Execute zero-loss padding rows for dropped sources to keep the registered student work.
+    pad_z = z[~keep].repeat_interleave(m_states, 0)
+    pad_pm = pm[~keep].repeat_interleave(m_states, 0)
+    pad_inp = torch.zeros(len(pad_z), S, dtype=torch.long, device=dev)
     if not bool(keep.any()):
-        return None, 0
+        lg = parallel_logits(net, pad_z, pad_inp, pad_pm)
+        return lg[..., 0].float().sum() * 0.0, 0
     hist, y, z, pm, first_fp = hist[keep], y[keep], z[keep], pm[keep], first_fp[keep]
     gk = len(y)
     t = (torch.rand(gk, m_states, device=dev) * first_fp[:, None]).long()   # uniform over states strictly before it
     inp = hist[torch.arange(gk, device=dev)[:, None], t].reshape(-1, S)
     tgt = y.repeat_interleave(m_states, 0)
     zz, pmm = z.repeat_interleave(m_states, 0), pm.repeat_interleave(m_states, 0)
-    return ce(parallel_logits(net, zz, inp, pmm), tgt), len(tgt)
+    lg = parallel_logits(net, torch.cat([zz, pad_z], 0), torch.cat([inp, pad_inp], 0),
+                         torch.cat([pmm, pad_pm], 0))
+    return ce(lg[:len(tgt)], tgt), len(tgt)
 
 
 def detok(ids):
```

### A4-4 — P2: an initially fixed WMT source is charged two detection passes

**Locations:** `src/ncmr/jacobi_train.py:46`;
`scripts/train_wmt14_L.py:164`, `:211`.

`fixed` is gated by `t > 1`, even though WMT always starts from known all-zero
codes and its prefix is static. If the first output is also all zero, the first
pass has already detected convergence. The code reports NFE 2 rather than 1;
at cap 1 it instead labels that row a cap hit. Output codes are correct. This
affects degenerate sources, which the teacher drops, and could overstate inference
NFE for the corresponding evaluation sentences.

**Test:** `test_initial_fixed_point_needs_one_detection_pass` executes the real
WMT `consist_loss` and decoder on two all-zero fixed points. The teacher still
performs 12 passes; its summed detection NFE should be 2 and is 4.

**Proposed patch:** opt into first-pass detection for WMT's static prefix. Keep
the shared helper's default for callers whose prefix changes after the first
pass; changing the default unconditionally would be unsafe for feedback decoding.

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -161,7 +161,7 @@
     net.eval()
     # the teacher is the fp32 greedy map that exact decoding evaluates, not the bf16 training map (codex audit2 A2-4)
     with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
-        out = jacobi_decode(net, lambda c, h: z.float(), g, S, dev, K + 1, pad_mask=pm, keep_history=True, full_budget=True)
+        out = jacobi_decode(net, lambda c, h: z.float(), g, S, dev, K + 1, pad_mask=pm, keep_history=True, full_budget=True, initial_fixed_ok=True)
     net.train(was)
     if not bool(out["stop"].eq(1).all()):
         raise RuntimeError("consistency teacher did not reach a fixed point in K + 1 passes")
@@ -215,7 +215,7 @@
             ids = [(pt.encode(t)[:P] + [PAD] * P)[:P] for t in srcs[s:s + bs_]]
             x = torch.tensor(ids, device=dev)
             z, pm = prefix(x)
-            out = jacobi_decode(net, lambda c, h: z, len(x), S, dev, K + 1, pad_mask=pm, keep_history=True)
+            out = jacobi_decode(net, lambda c, h: z, len(x), S, dev, K + 1, pad_mask=pm, keep_history=True, initial_fixed_ok=True)
             if sequential:   # disagreements are recorded per sentence (reported, not silently averaged)
                 seq = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=EOS)
                 same += (seq == out["codes"]).all(1).tolist()
--- a/src/ncmr/jacobi_train.py
+++ b/src/ncmr/jacobi_train.py
@@ -26,14 +26,15 @@
 
 @torch.no_grad()
 def jacobi_decode(net, prefix_fn, n: int, S: int, device, max_passes: int, pad_mask=None, check_fn=None,
-                  keep_history: bool = False, full_budget: bool = False):
+                  keep_history: bool = False, full_budget: bool = False, initial_fixed_ok: bool = False):
     """Greedy Jacobi decoding with per-sample stopping.
 
     prefix_fn(codes, has_prev) -> z_prefix: lets the prefix depend on the previous pass (board feedback);
     has_prev is False on pass 1. check_fn(codes) -> bool tensor (n,): optional task check (stop when True).
     Returns dict(codes=(n, S) final codes, nfe=(n,) passes used, stop=(n,) 0 = cap, 1 = fixed point, 2 = check,
     history=list of (n, S) codes per pass if keep_history). full_budget: always run max_passes passes (stopped
-    samples keep their codes), so the executed compute is fixed (training teacher, codex audit2 A2-7)."""
+    samples keep their codes), so the executed compute is fixed (training teacher, codex audit2 A2-7).
+    initial_fixed_ok: allow pass-1 convergence detection when prefix_fn is static (WMT14)."""
     codes = torch.zeros(n, S, dtype=torch.long, device=device)
     done = torch.zeros(n, dtype=torch.bool, device=device)
     nfe = torch.full((n,), max_passes, dtype=torch.long, device=device)
@@ -43,7 +44,7 @@
         logits = parallel_logits(net, prefix_fn(codes, t > 1), codes, pad_mask).float()
         new = logits.argmax(-1)
         new = torch.where(done[:, None], codes, new)
-        fixed = (new == codes).all(1) & ~done & (t > 1)
+        fixed = (new == codes).all(1) & ~done & (initial_fixed_ok or t > 1)
         chk = check_fn(new) & ~done & ~fixed if check_fn is not None else torch.zeros_like(done)
         for m, code in ((fixed, 1), (chk, 2)):
             nfe[m], stop[m] = t, code
```

### A4-5 — P2: intermediate evaluations discard the registered per-sentence data

**Location:** `scripts/train_wmt14_L.py:287-291`;
collection list in `notebooks/make_notebooks.py:300`.

The curve evaluates all caps but assigns both hypothesis outputs to `_`, then
removes per-sentence NFE. Only aggregate metrics survive. J1b says every evaluation
keeps ordered exact/capped hypotheses on val and test. The final dumps meet that
requirement; the step-500/1000/etc. val curves do not. Intermediate weights are
overwritten, so their paired comparisons cannot be reconstructed later. Those
evaluations also request `sequential=False`, so their claimed exact outputs have
no sentence-level equality check. This does not invalidate the saved final J1a
comparison, hence P2.

**Test:** `test_curve_evaluation_persists_ordered_comparator_inputs` executes
the actual curve block with ordered stub outputs and checks its persisted JSON
and log records. No record contains the hypotheses and sentence NFE together.

**Proposed patch:** save step-specific validation records and collect them; check
sequential equality there too. Regenerate the two WMT notebooks from the amended
generator during implementation. Other trainer/notebook changes are out of this
audit's scope.

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -292,8 +292,10 @@
             traj_stats.update(passes=0.0, n=0, states_before_fp=0.0, degenerate=0)
         print(json.dumps(trainer.log({"loss": round(loss.item(), 4), "gnorm": round(gn, 3), **logs})), flush=True)
     if trainer.step % int(tc.eval_every) == 0 or trainer.step == int(tc.steps):
-        ev, _, _ = evaluate(val_text["src"][:n_curve], val_text["ref"][:n_curve], sequential=False)
-        ev.pop("nfe_per_sentence")
+        ev, curve_hyp, curve_cap = evaluate(val_text["src"][:n_curve], val_text["ref"][:n_curve], sequential=True)
+        json.dump({"hyp_val": curve_hyp, "hyp_val_capped": curve_cap,
+                   "nfe_val": ev.pop("nfe_per_sentence")},
+                  open(os.path.join(a.run_dir, f"hyp_val_step{trainer.step}.json"), "w"))
         ev["val_nll"] = val_nll()
         print(json.dumps(trainer.log({"eval": ev})), flush=True)
     if trainer.step % int(tc.ckpt_every) == 0:
--- a/notebooks/make_notebooks.py
+++ b/notebooks/make_notebooks.py
@@ -297,7 +297,7 @@
         rd = c.split('--run_dir ')[1].split()[0].format(R=R, S=seed)
         name = os.path.basename(rd)
         sessions[name] = history(rd)
-        for f in ['result.json', 'metrics.jsonl', 'config.yaml', 'calib.json', 'hyp_test.json', 'hyp_val.json', 'run_manifest.json', 'eval_final.json', 'eval_6000.json', 'eval_20500.json', 'samples_final.json', 'provenance.jsonl']:
+        for f in ['result.json', 'metrics.jsonl', 'config.yaml', 'calib.json', 'hyp_test.json', 'hyp_val.json', 'run_manifest.json', 'eval_final.json', 'eval_6000.json', 'eval_20500.json', 'samples_final.json', 'provenance.jsonl'] + sorted(n for n in os.listdir(rd) if n.startswith('hyp_val_step') and n.endswith('.json')):
             if os.path.exists(f'{rd}/{f}'):
                 shutil.copy(f'{rd}/{f}', f'{out}/{name}__{f}')
 for f in os.listdir('results'):   # diagnostics write straight into the repo's results/
```

## Answers to the audit questions

1. **J1b state selection:** correct apart from the degenerate compute and NFE
   findings. The code constructs `[s0, s1, ..., s12]`, with s0 all-zero and each
   s(t+1) the output of a pass on st. `first_fp` finds the first complete state
   equal to the final code vector. Sampling `floor(U * first_fp)` with independent
   uniform U excludes that state and all later ones, with replacement as registered.
   The known trajectory `F(s) = [source_bit, s[:-1]]` reaches its 11-code fixed
   point after 11 updates; a controlled draw covers all 11 pre-fixed states in
   the right order. A 40,000-draw check covers the uniform support. All-zero fixed
   starts are dropped and counted. The production loss block draws exactly one
   common 128-row batch and uses TF rows 1-64 and sources 65, 69, ..., 125.

2. **Precision and gradients:** no no-grad **generator** forward runs with autocast
   enabled in the J2 loss region. The teacher has `no_grad` plus autocast disabled;
   canonical encoding and VQ-to-input conversion also disable autocast. Frozen
   prefix embedding lookups are not a no-grad autocast matrix forward through
   trainable generator weights. Hooks observe TF `(grad=True, amp=True)`, 12 teacher
   calls `(False, False)`, and student consistency `(True, True)`. All **41/41
   trainable tensors**, containing **16,235 parameters**, in the small upstream
   WMT-style model have finite gradient tensors with nonzero norm. This counts
   reached parameter tensors, not nonzero individual gradient entries. Thus Torch 2.6 does not reproduce audit
   3's cache detachment here. On affected older Torch versions the dangerous
   no-grad weight cast requires enabled autocast; these teacher calls cannot
   create that cached cast. Fresh `Trainer.ctx` objects also avoid context reuse
   leaks. `val_nll` is a separate precision failure, A4-1; its fresh outer context
   exits before the next training step, so this is not evidence of missing
   training gradients. Full released-WMT gradient coverage on CUDA remains unmeasured.

3. **RNG/resume:** the state sampler intentionally shares the global device Torch
   generator with dropout. `Trainer.state` saves CPU Torch RNG, all CUDA RNGs,
   the NumPy data RNG, model, optimizer, step and scaler in one atomically replaced
   checkpoint. `maybe_resume` restores them after model construction; no intervening
   data/random draw occurs before the next training step. A checkpoint after one
   J2 step followed by two further steps is bitwise equal to a fresh process-style
   reconstruction plus restore, an extra validation-NLL call and those same two steps: losses, weights, AdamW
   state, data RNG and global Torch RNG. No additional sampling generator is
   missing. This is same-runtime CPU evidence, not a guarantee across Torch,
   device or kernel changes; the manifest does not lock those numerical details.
   This equivalence check covers stochastic training state, not wall-clock or
   interval logging buffers.

4. **Identifiability:** for fixed parameters and the visible, truncated/padded
   source prefix p, the eval-mode fp32 map Fp is deterministic. Its level k reads
   only lower-level reconstructed codes and p. Induction on levels therefore gives
   one sequential greedy code vector G(p), reached from every initial code vector
   in at most K passes; the extra pass detects convergence. Every selected input
   state from that source is labelled with the same G(p). The source prefix is
   visible at every level (level 0 uses the source representation itself), so
   omitting the pass index does not create two labels for the same visible input.
   EOS and PAD are included in the fixed-length code target, not removed from
   the training label or independently randomized; even their codes are functions
   of p. A repeated-prefix CPU test includes EOS and source padding and verifies
   equality at every target position plus fixed-point replay. Degenerate sources
   contribute no labels. This avoids the specific wave-1 ambiguity from reference
   information absent from the input. It does **not** establish that the shortcut
   is easy to learn, that ignoring intermediate codes is undesirable, or that
   self-distillation cannot collapse; the registered quality/control screens remain
   necessary.

5. **Evaluation/data:** final evaluation runs under autocast disabled; CUDA startup
   turns TF32 off for matmul and cuDNN. Greedy sequential and Jacobi codes are
   compared per sentence before detokenization, and disagreement indices are saved.
   The small upstream WMT-style model passes with padded sources, and the shared
   primitive also matches sequential decoding on the hash-pinned fixture. Numerical
   argmax ties on the released WMT GPU model remain a measured gate, not a theorem
   of bitwise equality across different matrix shapes. Final JSON retains exact
   hypotheses, caps 1-12 and ordered exact NFE for both splits; capped NFE is
   derivable as `min(exact_nfe, cap)`. BLEU is lowercased corpus sacreBLEU and length
   ratio is its `sys_len/ref_len`, with EOS/PAD removed as in w0. A two-batch test
   checks order, metrics, caps and both real dump statements. Curve record loss
   is A4-5. Prep uses the WMT validation parquet (newstest2013) for val, a full
   seeded permutation until 400k eligible examples, targets `English + EOS + PAD`,
   and source truncation/right padding. It verifies saved selection metadata and
   exact archive size before reuse. The training loop reads only val; released
   newstest2014 is first loaded after training, then scored once per completion
   invocation. Reinvoking an already completed trainer repeats final test evaluation;
   this is not an automatic hyperparameter search, but operators must preserve
   the registered single final read. `--eval_only` rejects a training checkpoint;
   config and archive identity checks pass the refusal tests. Base identity is A4-2.

6. **Compute/time:** without degenerates, J1 is `3000 * (3 * 128)` and J2 is
   `2000 * (3 * 128 + 12 * 16)`, both **1,152,000 forward rows**. The 12-pass
   teacher actually executes its full batch budget. Loss weighting after dropping
   a source is consistent with dropping its labels, but its claimed work is A4-3.
   The field is explicitly named `generator_flops_proxy_total`: 3x backward scaling,
   frozen tokenizer work and the attention approximation are not physical execution
   measurements. J2's fp32 teacher runs after `Trainer.t_start`, so it is included
   in both logged and final `wall_s`; it is not omitted. Final `wall_s` also includes
   evaluation, and loader/data-prep time precedes that timer. The notebook command
   timer includes the entire trainer invocation. No per-sentence batch-size-1
   inference timing on 256 sentences or attention-inclusive operating-point cost
   is produced here; J1a requires those additional measurements before a criterion
   (d) claim. R028 raised the J2 estimate to ~5 CU, while the notebook metadata
   still says 4; neither value is a measured cost.

## Historical number recomputation

`tests/test_audit4_numbers.py --write` generates
`docs/reviews/codex_audit4_numbers.json`. It records subject Git-blob hashes,
normalized-LF result-text hashes, each calculation's method and whether it is an
independent recomputation or an aggregate readback. It inspects all **62**
`results/**/*.json` in the subject checkout. No historical smoke evidence is
invented from current tests.

| R020 WMT quantity | Result | Evidence |
|---|---:|---|
| Sentence count | 3003 | Sum of exact-pass histogram |
| Exact total passes / mean | 33,233 / 11.066600066600067 | Weighted histogram, rounds to **11.07** |
| Exact NFE saving vs 11 | -0.6054551509097061% | Recomputed from histogram mean |
| Exact BLEU | 22.625980088896835 | Saved aggregate, rounds to **22.63** |
| Stable-output mean passes | 10.588411588411589 | Saved aggregate, rounds to **10.59** |
| Stable-output BLEU | 22.639255560616412 | Saved aggregate, rounds to **22.64** |
| Fixed point equals sequential | 100% | Saved aggregate only |
| First final decoded output | 9.58075258075258 passes | Saved aggregate only |
| Registered target | 8.25 passes / 21.63 BLEU | 0.75 x 11; rounded baseline minus 1 |

**Disagreements: none among the checkable R020 WMT arithmetic or rounded
aggregate values.** The 21.63 registered floor is based on the rounded 22.63
baseline; the unrounded baseline minus one is 21.625980088896835. The old n6
files are byte-identical stale copies and report sequential NFE 10; they are not
used for the full-set result. The JSON does not save the WMT sentence hypotheses,
reference strings, stable-pass histogram or sequential code comparisons, so
BLEU, the stable mean and the 100% equality cannot be independently rederived.
Matching their saved scalars is not rescoring BLEU.

**R028 WMT smoke claims are unverifiable from `results/**.json`.** The only
WMT JSONs are the full-set w0 aggregate and the two stale n6 copies. None of the
other JSON contents mentions WMT. No saved WMT smoke contains the stated zero
degenerate count, cap-1..12 records, sentence-level equality or 20k prep-selection
comparison; no saved test transcript supports the historical 59/25/14 counts.
This is missing evidence, not a numerical disagreement or a failed current smoke
test. The present CPU tests corroborate particular current code paths but cannot
authenticate those historical runs or their CU estimates.

R028's earlier **up to 28%** compute shortfall is consistent with the registered
proxy algebra: a two-pass teacher used `384 + 2 * 16 = 416` rows rather than
576, a **27.77777777777778%** shortfall at the same 2000 steps. This recomputes
the possible bound; it is not a measurement recovered from a historical smoke
run. The JSON labels that distinction explicitly.

## Summary

| ID | Priority | File | One-line issue |
|---|---|---|---|
| A4-1 | P1 | `scripts/train_wmt14_L.py:251` | Validation NLL enables training autocast instead of fp32 scoring. |
| A4-2 | P1 | `scripts/train_wmt14_L.py:85` | Same-size different base weights pass run identity. |
| A4-3 | P1 | `scripts/train_wmt14_L.py:185` | Dropped consistency rows remain charged at the full registered proxy. |
| A4-4 | P2 | `src/ncmr/jacobi_train.py:46` | Static-prefix initial fixed points take two detection passes. |
| A4-5 | P2 | `scripts/train_wmt14_L.py:288` | Intermediate evaluations discard ordered comparator inputs. |
