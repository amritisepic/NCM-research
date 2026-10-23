# Wave-1 audit

Audit branch: `codex/audit`, based on `f2222ca`. The requested R018–R021 entries and
`ncmr.jacobi_train` exist on the already available analysis ref, commit
`1c7185c0e0de23d364c8bad33b2606c852bbd00d`, rather than the starting checkout.
Those objects are read directly through Git. `scripts/train_gen.py`,
`src/ncmr/corruption.py`, the six W1 configurations, and W1's registration agree
between the two refs. The pre-existing working-tree edit to `src/ncmr/models.py`
is outside this audit. No implementation patches below have been applied.

Severity: P1 blocks a trustworthy W1 comparison or reproducible continuation;
P2 affects reproducibility, interpretation, or a supported non-default execution.

## 1. Hostile implementation review

### F1 — P1: checkpoints omit the corruption RNG

**Locations:** `scripts/train_gen.py:106`, `:224`, `:253–256`, `:264`;
`src/ncmr/train.py:62–68`, `:75–92`.

**Failure:** resume at step 250 constructs a new generator seeded with
`seed * 7919 + 250`, rather than continuing the generator after 250 batches of
masks, replacements, and level-0 samples. Restoring the global Torch and NumPy
states does not restore this separate generator. The next data batch agrees but
the corrupted hierarchy does not, so interrupted and uninterrupted runs differ.
This affects every trained W1 arm, with different random draw counts per arm.

**Proposed patch:** use Trainer's existing `extra` payload at every save site.

```diff
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@ -103,7 +103,7 @@
                   ckpt_every=int(tc.ckpt_every), device=device, amp=bool(tc.get("amp", True)),
                   min_lr_frac=float(tc.get("min_lr_frac", 0.1)))
 modules = {"model": net}
-trainer.maybe_resume(modules, data_rng)
+resume_extra = trainer.maybe_resume(modules, data_rng)
 fnet = torch.compile(net) if (device == "cuda" and bool(tc.get("compile", False))) else net
 print(f"[gen] init={cfg.init} params {n_params(net) / 1e6:.2f}M | corruption {corr} policy_temp {policy_temp} | "
       f"{fl_step / 1e12:.2f} TFLOP/step, {fl_step * int(tc.steps) / 1e15:.1f} PFLOP total | device {device}", flush=True)
@@ -221,7 +221,13 @@
 # ------------------------------------------------------------------------------------------------ loop
 net.train()
 deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
-gen = torch.Generator(device=device).manual_seed(seed * 7919 + trainer.step)
+gen = torch.Generator(device=device).manual_seed(seed * 7919)
+if trainer.step:
+    if "corruption_rng" not in resume_extra:
+        raise RuntimeError("checkpoint lacks corruption RNG; exact resume unavailable")
+    gen.set_state(resume_extra["corruption_rng"].cpu())
+def save_checkpoint():
+    trainer.save(modules, data_rng, extra={"corruption_rng": gen.get_state()})
 accum = int(tc.get("grad_accum", 1))
 calib = Calib(a.run_dir, a.calib_steps, fl_step) if a.calib_steps else None
 every_s = int(sc.get("every", 0))
@@ -251,9 +257,9 @@
             ev.update({k: v for k, v in sm.items() if not k.endswith(("per_group", "per_seed"))})
         print(json.dumps(trainer.log({"eval": ev})), flush=True)
     if trainer.step % int(tc.ckpt_every) == 0:
-        trainer.save(modules, data_rng)
+        save_checkpoint()
     if deadline and time.time() > deadline:
-        trainer.save(modules, data_rng)
+        save_checkpoint()
         print("[stop] time budget reached; resume by re-running the same command", flush=True)
         sys.exit(0)
 
@@ -261,7 +267,7 @@
     calib.report(trainer.step, {"init": cfg.init, "corruption": corr.mode, "batch_size": int(tc.batch_size), "grad_accum": accum})
     sys.exit(0)
 if not a.eval_only:
-    trainer.save(modules, data_rng)
+    save_checkpoint()
     torch.save({"model": net.state_dict(), "init": cfg.init, "model_cfg": OmegaConf.to_container(cfg.model) if "model" in cfg else {},
                 "level0_unigram": L0.cpu()}, os.path.join(a.run_dir, "final.pt"))
 res_path = os.path.join(a.run_dir, "result.json")
```

### F2 — P1: the same autocast context is entered recursively in on-policy arms

**Locations:** `scripts/train_gen.py:130–131`, `:233–234`;
`src/ncmr/train.py:57`.

**Failure:** `compute_loss` already runs inside `trainer.ctx`. Its policy forward
enters that *same context object* again, overwriting its stored previous autocast
state. On exit, autocast can remain enabled outside the intended block. Uniform
arms do not take the nested path. In particular, Gen-PPL's scorer forward in
`src/ncmr/ncm_ref/evaluation/gen_ppl.py:74` has no autocast guard: later evaluations
can therefore score on-policy arms in bf16/fp16 while scoring uniform arms in
fp32. Merely calling
`.float()` on logits afterwards cannot undo reduced-precision computation.

**Proposed patch:** inherit the caller's active context for the policy forward;
retain the existing explicitly disabled tokenizer blocks.

```diff
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@ -127,8 +127,8 @@
     for _ in range(policy_passes):
         with torch.autocast(device_type=device, enabled=False):
             z_in = tok.vq.idx_to_ncm_input(codes)
-        with trainer.ctx:
-            logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
+        # compute_loss's caller owns autocast; do not re-enter its context object.
+        logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
         samp = torch.distributions.Categorical(logits=logits / policy_temp).sample()
         codes = torch.cat([idx[:, :l0], samp], 1)
     net.train(was)
```

### F3 — P1: two-pass policy codes never feed the sampled level 0 into pass 2

**Locations:** `scripts/train_gen.py:125–136`.

**Failure:** A5 claims pass 2 samples from pass-1 codes. Line 133 inserts the
*canonical* level 0 after every pass; the inference unigram draw occurs only after
the final pass. For an example whose unigram code differs from its canonical
code, pass 2 therefore sees a different hierarchy from pass 1's returned hierarchy.
The purported compounding intervention excludes level-0 errors at every depth.
One-step samples conditional on clean coarser levels are intentional; they are
not themselves full ancestral samples.

**Proposed patch:** preserve clean conditioning on pass 1, but put the same
sampled inference level 0 into the hierarchy consumed by every later pass.

```diff
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@ -124,16 +124,17 @@
     with torch.autocast(device_type=device, enabled=False):
         idx = tok.vq.z_to_idx(tok._encode_pre_quant(x))
     codes = idx
+    lv0 = torch.multinomial(L0, x.shape[0] * l0, replacement=True,
+                            generator=gen).view(x.shape[0], l0)
     for _ in range(policy_passes):
         with torch.autocast(device_type=device, enabled=False):
             z_in = tok.vq.idx_to_ncm_input(codes)
         with trainer.ctx:
             logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
         samp = torch.distributions.Categorical(logits=logits / policy_temp).sample()
-        codes = torch.cat([idx[:, :l0], samp], 1)
+        codes = torch.cat([lv0, samp], 1)
     net.train(was)
-    lv0 = torch.multinomial(L0, x.shape[0] * l0, replacement=True, generator=gen).view(x.shape[0], l0)
-    return torch.cat([lv0, samp], 1)
+    return codes
 
 
 def compute_loss(x, gen):
```

This also changes RNG draw order; re-run every affected arm with a new manifest.
For partial replacement rates, these are proposals conditioned on a full previous
policy hierarchy, not on the later mask-mixed training hierarchy. That distinction
must remain explicit when interpreting A5.

### F4 — P1: equal reported generator FLOPs do not establish equal training FLOPs

**Locations:** `scripts/train_gen.py:98–99`, `:125–129`, `:141`, `:278`;
`src/ncmr/corruption.py:74–76`, `:96–99`;
`src/ncmr/flops.py:20–28`.

**Failure:** the 4000/3000/2400 step ratio equalizes the proxy
`steps * (3 + policy_passes) * generator_forward`. It excludes all tokenizer and
hierarchy construction. Retargeting adds a second distance computation per level;
policy arms also encode the clean hierarchy and rebuild policy inputs every pass.
The hierarchy always reconstructs both canonical and used paths; soft labels add
top-k/softmax work. These costs differ between arms. The forward formula also
prices attention as if masked-out blocks were skipped, although the released
configuration uses ordinary dense masked SDPA (`use_flex_attention: false`). Thus
349 PFLOP is an analytic generator proxy, not a verified end-to-end budget.
The proxy recomputes to **349.032778 PFLOP** for all three step schedules;
retargeting's extra distance dot products alone add approximately **2.0164 GFLOP
per window** before considering the other omissions.

**Proposed patch:** label the proxy honestly and fail closed until per-arm full
training-step costs and stopping steps have been calibrated with the same
counting convention. Count tokenizer, policy, hierarchy, forward, backward and
optimizer consistently; evaluation is a separate budget. Reconcile changed step
counts and learning-rate schedules with the registration before running W1.

```diff
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@ -96,7 +96,10 @@
 policy_temp = float(cc.get("policy_temp", 1.0))
 policy_passes = int(cc.get("policy_passes", 1))   # 1 = one-step errors; t = errors after t parallel passes
 fwd = FL.ncm_gen_fwd_per_seq(n_embd, n_layer, levels, codebook=tok.vq.vqvae_vocab_sizes[0], vq_dim=tok.embed_dim)
-fl_step = (3 + (policy_passes if corr.mode == "onpolicy" else 0)) * fwd * int(tc.batch_size)
+generator_fl_step = (3 + (policy_passes if corr.mode == "onpolicy" else 0)) * fwd * int(tc.batch_size)
+if not a.calib_steps and not a.eval_only and "full_training_flops_per_step" not in tc:
+    raise ValueError("full per-arm FLOP calibration required for a matched W1 run")
+fl_step = float(tc.get("full_training_flops_per_step", generator_fl_step))
 params = [p for p in net.parameters() if p.requires_grad]
 trainer = Trainer(a.run_dir, params, max_lr=float(tc.lr), total_steps=int(tc.steps), warmup=int(tc.warmup),
                   weight_decay=float(tc.weight_decay), betas=(0.9, float(tc.get("beta2", 0.95))),
@@ -275,7 +278,9 @@
               open(os.path.join(a.run_dir, "samples_final.json"), "w"))
 wall = trainer.wall_prev + time.time() - trainer.t_start
 res = {"run": os.path.basename(a.run_dir.rstrip("/")), "config": OmegaConf.to_container(cfg),
-       "params_total": n_params(net), "steps": trainer.step, "flops_est_total": fl_step * trainer.step,
+       "params_total": n_params(net), "steps": trainer.step,
+       "generator_flops_proxy_total": generator_fl_step * trainer.step,
+       "training_flops_est_total": fl_step * trainer.step,
        "wall_s": wall, "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch": torch.__version__, "final_eval": final}
 json.dump(res, open(res_path, "w"), indent=1)
```

Calibration must include a cost-model identity in the run manifest; a throughput
measurement alone is not a FLOP measurement. This is a fail-closed proposal, not
a claim that multiplying wall time by nominal device TFLOPS fixes the accounting.

### F5 — P1: automatic resume can silently change the arm or contaminate A0

**Locations:** `scripts/train_gen.py:65–73`, `:92–106`;
`src/ncmr/train.py:75–92`.

**Failure:** start control in a run directory, then invoke a re-quantized config
or `--eval_only` in that same directory. `config.yaml` is overwritten before
checking the checkpoint; Trainer then loads its weights, optimizer, data RNG and
step unconditionally. The result is labelled by the new config. In eval-only
mode, A0 can consequently evaluate already fine-tuned weights. Changes to seed,
precision, microbatch partition or LR schedule also pass without validation.

**Proposed patch:** validate before overwriting metadata or loading state. Include
base checkpoint digests and precision policy in a production manifest.

```diff
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@ -70,7 +70,15 @@
 stream = TrainStream(a.data_dir) if a.data_dir else TrainStream()
 val_x = torch.from_numpy(examples("val", int(tc.get("eval_n", 1000))))
 os.makedirs(a.run_dir, exist_ok=True)
-OmegaConf.save(cfg, os.path.join(a.run_dir, "config.yaml"))
+cfg_path = os.path.join(a.run_dir, "config.yaml")
+ckpt_path = os.path.join(a.run_dir, "ckpt.pt")
+if a.eval_only and os.path.exists(ckpt_path):
+    raise ValueError("A0 requires a directory without a training checkpoint")
+if os.path.exists(ckpt_path):
+    if not os.path.exists(cfg_path) or OmegaConf.to_container(
+            OmegaConf.load(cfg_path), resolve=True) != OmegaConf.to_container(cfg, resolve=True):
+        raise ValueError("resume configuration differs from the saved run")
+OmegaConf.save(cfg, cfg_path)
 if device == "cuda":
     torch.backends.cuda.matmul.allow_tf32 = True
     torch.backends.cudnn.allow_tf32 = True
```

### F6 — P2: sample evaluation mutates the training RNG

**Locations:** `scripts/train_gen.py:179–180`, `:197–218`, `:249–250`.

**Failure:** each curve evaluation calls `torch.manual_seed(seed + 10000)`.
Subsequent dropout and policy `Categorical.sample()` use this reset global RNG.
Adding/removing an evaluation or changing its batch size changes later training;
repeated curve evaluations also reset to the same stream. Scorer initialization
can consume that stream on the first evaluation. Data order stays unchanged
because `TrainStream.batch` uses a separate NumPy generator.

**Proposed patch:** isolate the *whole* sample metric call, including scorer
initialization, in a fork of the CPU and relevant CUDA RNGs.

```diff
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@ -194,7 +194,7 @@
     return out, calls / max(1, -(-n // bs)), time.time() - t0
 
 
-def sample_metrics(n, seeds, mauve=False):
+def _sample_metrics(n, seeds, mauve=False):
     from ncmr.eval_text import evaluate_samples, mauve_features
     ids, groups, nfe, wall = [], [], 0.0, 0.0
     for sd in seeds:
@@ -216,6 +216,11 @@
     res = evaluate_samples(reenc, bpe.decode, groups, scorers=scorers, ref_feats=ref, device=device)
     res.update({"nfe_per_sample": nfe, "gen_wall_s": wall})
     return res, ids, groups
+
+def sample_metrics(n, seeds, mauve=False):
+    devices = [torch.cuda.current_device()] if device == "cuda" else []
+    with torch.random.fork_rng(devices=devices):
+        return _sample_metrics(n, seeds, mauve)
 
 
 # ------------------------------------------------------------------------------------------------ loop
```

### F7 — P2: B7 labels depend on unrelated examples in the microbatch

**Locations:** `src/ncmr/corruption.py:81–83`;
`scripts/train_gen.py:63–64`, `:232`.

**Failure:** soft-label temperature uses the mean first/second-nearest distance
gap over the current microbatch. Changing `--micro_batch` to fit memory changes
the target distribution for an otherwise identical window. A window with a tiny
gap becomes much softer when batched with a large-gap window. Therefore B7's
objective and reported ~77% canonical mass are not invariant to accumulation.

**Proposed patch:** define a local gap per code position, or register and freeze
a calibrated gap per level. The local option below changes B7's effective
temperature and needs a declared protocol amendment rather than a silent patch.

```diff
--- a/src/ncmr/corruption.py
+++ b/src/ncmr/corruption.py
@@ -79,7 +79,7 @@
             targets.append(c_lab)
             if cfg.soft_tau > 0:
                 top_d, top_i = d_lab.topk(cfg.soft_topm, dim=-1, largest=False)
-                gap = (top_d[..., 1] - top_d[..., 0]).mean().clamp_min(1e-6)
+                gap = (top_d[..., 1:2] - top_d[..., :1]).clamp_min(1e-6)
                 s_w.append(torch.softmax(-(top_d - top_d[..., :1]) / (cfg.soft_tau * gap), -1))
                 s_idx.append(top_i)
             used = c_lab.clone()
```

### F8 — P2: final sample precision is implicit and includes tokenizer operations

**Locations:** `scripts/train_gen.py:185–189`, `:74–76`;
upstream `models/next_concept.py:generate` and
`models/quant.py:get_next_autoregressive_input` (read only).

**Failure:** the `trainer.ctx` around the entire sequential `generate` call also
autocasts the tokenizer's residual convolutions used between levels. In contrast,
training and policy-input construction explicitly disable autocast for those
operations. A0's earlier CPU fp32 samples and the W1 CUDA samples therefore need
not share code trajectories, even at the same seed. Switching from an fp16 device
to a bf16 device also changes the protocol without an explicit config field.
R020's reference sanity check is reassuring but does not prove precision equivalence.

**Proposed patch:** adopt explicit fp32 evaluation for the whole generation and
decoder path, record the policy, and rerun A0 with exactly that policy. Alternatively
implement generator-only autocast with fp32 tokenizer boundaries in a wrapper;
do not modify the vendored upstream files.

```diff
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@ -72,8 +72,8 @@
 os.makedirs(a.run_dir, exist_ok=True)
 OmegaConf.save(cfg, os.path.join(a.run_dir, "config.yaml"))
 if device == "cuda":
-    torch.backends.cuda.matmul.allow_tf32 = True
-    torch.backends.cudnn.allow_tf32 = True
+    torch.backends.cuda.matmul.allow_tf32 = False
+    torch.backends.cudnn.allow_tf32 = False
 
 # ------------------------------------------------------------------------------------------------ build
 BUNDLE = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"), device=device)
@@ -182,7 +182,7 @@
     for s in range(0, n, bs):
         b = min(bs, n - s)
         lv0 = torch.multinomial(L0, b * l0, replacement=True, generator=g).view(b, l0)
-        with trainer.ctx:
+        with torch.autocast(device_type=device, enabled=False):
             codes = net.generate(lv0, torch.full((b, 1), EOS, device=device), temperature=1.0, top_k=0, top_p=1.0,
                                  cfg_scale=0.0, cfg_null_eos=EOS)
         with torch.autocast(device_type=device, enabled=False):
@@ -277,6 +277,6 @@
 res = {"run": os.path.basename(a.run_dir.rstrip("/")), "config": OmegaConf.to_container(cfg),
        "params_total": n_params(net), "steps": trainer.step, "flops_est_total": fl_step * trainer.step,
        "wall_s": wall, "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
-       "torch": torch.__version__, "final_eval": final}
+       "torch": torch.__version__, "eval_precision": "fp32_tf32_off", "final_eval": final}
 json.dump(res, open(res_path, "w"), indent=1)
 print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1, default=str))
```

This disables TF32 globally; if retaining it during training, save and restore
those flags around evaluation instead. Cross-device bitwise identity still needs
testing; recording precision is essential to interpreting differences.

### Checks that did not produce a blocking finding

- Data order: line 69's independent NumPy RNG feeds line 230; on-policy draws and
  evaluator reseeding do not advance it. All arms share the same prefix of windows
  for a seed. Different final exposures (4000/3000/2400 batches) are explicit in W1.
- Re-quantized targets use the residual after *used* coarser codes; canonical
  targets use the clean reconstruction. Updating unmasked used codes to repair
  labels in retargeted arms is consistent with that recursive hierarchy definition.
- Quantization in `build_hierarchy` is under disabled autocast and no-grad, and
  the tokenizer is frozen/eval. The tokenizer is kept outside the generator's
  module tree, so `net.train()` does not re-enable tokenizer dropout.
- Final W1 seed groups, sample counts, temperature, 11 NFE, both scorers, entropy,
  repetition/distinctness, word duplicates, pooled 1024-reference MAUVE and its
  five quantization seeds are wired through the expected evaluator. Curves use
  256 samples and only the primary scorer, consistent with the registered curve
  count. Validation NLL excludes level 0, matching the R008 ~870 convention.
- The script exports metrics but does not enforce advancement/kill decisions;
  those registered decisions must be applied during analysis. The A1 damage
  rule must be checked before interpreting the treatment arms.

## 2. R018–R021 numeric reconstruction

Run `python -B tests/recompute_r018_r021.py`. It uses only the standard library,
reads numeric inputs from `results/**/*.json` at the immutable analysis commit,
and writes [codex_audit_numbers.json](codex_audit_numbers.json). The output contains
the source hashes, per-claim expected values, reconstructed values, tolerances,
and evidence limitations. Exit 1 deliberately signals the mismatches below.
It imports no model or analysis code and launches no training or evaluation.

**231 checks:** 135 agree by reconstruction/arithmetic, 82 agree by reading a
stored aggregate, 6 disagree, and 8 are explicitly unverifiable from JSON.
Readback is weaker than reproducing a measurement: the JSON generally does not
contain scorer token losses, BLEU hypotheses/references, or per-board outcomes.
The arithmetic tolerance is half the last displayed decimal unit. Approximate
phrases are identified below so that small rounding discrepancies are not
confused with changes to the research conclusion.

| Entry / claim | Logged | Recomputed | Assessment |
| --- | ---: | ---: | --- |
| R018: TC rises monotonically over all 12 evals | true | false | First two evaluations are **-142.1450 -> -148.9330**; subsequent values increase to **-57.3170**. Last-three-eval trend does increase. |
| R020: quality loss, levels 7–11 | 5% | **7.27391%** | Arithmetic mismatch; use 7% at whole-percent precision. |
| R020: quality loss, token-straddling levels 8–10 | 3% | **3.96374%** | Arithmetic mismatch; use 4%. |
| R020: quality loss, levels 1–5 combined | 87% | **86.34093%** | Summing separately rounded 48% and 39% gives 87%; recomputing before rounding gives 86%. |
| R020: token-straddling PPL cost | ~0.2 | **0.29257387** | Should read ~0.3 at one decimal; already inside the later stated ~0.2–0.3 range. |
| R020: upper end of MAUVE range for j >= 1 | 0.97 | **0.96471842** | Minor approximate-range/rounding discrepancy; conventional two-decimal rounding gives 0.96. |

Band shares use the unrounded GPT-2 curve and the same denominator throughout:
`PPL(j=-1) - PPL(j=11) = 7.3812543732`. Their numerators are
`PPL(j=6)-PPL(j=11)` for levels 7–11, `PPL(j=7)-PPL(j=10)` for
levels 8–10, and `PPL(j=0)-PPL(j=5)` for levels 1–5. The band shares
0 / 1–3 / 4–5 / 6 / 7–11 sum to 100% before rounding. The strict majority
trigger for levels 1–3 still does not fire (**47.80832%**); the schedule demotion
remains qualitatively consistent with a small finest-level upside.

**R018:** final totals, gaps, TC by level, -5.4% share, tokenizer accuracies,
relative sanity check, parameter counts, training steps and reported FLOP proxies
agree to their displayed precision. Data-matched AR NLL 212.432791 at step 1500
is present only in the derived decomposition JSON; reconstructing that checkpoint
selection independently needs the JSONL trajectory. The claim that all four curves
still fall, including NCM's -11 nats in the last 500 steps, likewise cannot be
reconstructed from final-only JSON. The norm measurements 7–15 and future ~1 CU
estimate have no saved JSON measurement. Width/init and 128-token length are
design properties. The literal registered TC < 10% falsification condition holds;
R018 explicitly overrides that rule on instrument-failure grounds. Arithmetic
agreement does not independently validate that post-result interpretation.

**R019:** all displayed rates, depth-bin counts, weighted totals and differences
agree. Test successes reconstructed from bin aggregates are **60/2000 vs
21/2000**, with **3/441 vs 0/441** at depth >= 9. Differences are **-1.7 val,
-1.95 test and -0.680272 deep points**; 20,000 x 64 = **1.28M** exposures.
The final training CEs **0.159/0.127** are in JSONL, absent from the JSON results.
Paper exposure/architecture comparisons (51.2M, 25.5M, 5x) are external or
architecture provenance, not independently recoverable from these result JSONs.
The registered < +1-point falsification applies to the saved val rates.

**R020:** reference Gen-PPL and MAUVE means/sample SDs recompute from their saved
groups/seeds; the displayed reference NLL, entropy, repetition, both scorers,
13-point conditional PPL curve, and Jacobi pass means agree. Exact-pass means
are independently recomputed from histograms: **6.406**, **11.939453125** and
**11.0666000666** for Sudoku, TinyStories and WMT14. Sudoku NFE saving is
**46.6166667%** and its stored cost ratio recomputes to **1.871059406**.
That last quantity is a **token-position proxy**, not a measurement of transformer
FLOPs; `scripts/jacobi_levels.py:155–167` ignores attention's quadratic work.
The solve rates/BLEU scores can only be read back, because per-example trajectories
and WMT14 hypotheses are not saved in these JSON files. The text target computes
to **8.25 passes** and **21.62598009 BLEU**. The stale 500-puzzle Sudoku and
6-example WMT14 copies in the upload match the older files exactly and were
excluded from full-set conclusions.

The diagnostic has one seed group, so **SE ~0.4** is an extrapolation rather than
a measurable SE from its JSON; device capacity, **5.4 CU/h** and **~2.5 CU** are
provenance/estimates. Several command JSONs do contain `wall_s`; the statement
that command wall times were not recorded is too broad, though complete setup,
calibration and billed duration cannot be recovered from them. The three
calibration JSON files now on the starting branch were added after the audited
analysis ref; R020's missing-calibration statement describes its earlier upload.
The statement that an intermediate solution occurs with a wrong fixed point in
~4% of puzzles is supported only as a **net +4.15-point** difference of aggregates;
individual transitions cannot be reclassified without trajectories.

**R021:** the 250-puzzle baseline, all nine oracle rates, 30 initial failures,
6.67/20/63.33% failure repairs at levels 1/3/6, and the plateau all agree. The
98.78437-second saved wall time is **1.646406 minutes**, consistent with 1.6 at
one decimal; process CPU time was not stored. The 16-code finest codebook is an
architecture property, verified from the released configuration. Oracle rates
cannot be independently measured again from aggregate JSON, but their arithmetic
is internally consistent. No R021 numerical mismatch was found.

## 3. CPU verification

Required checks: **30 passed in 24.91 seconds**, four Torch CPU threads, fp32
generation, no training. The first run emitted a CUDA-initialization warning from
the installed Torch build; all model tensors and both decoders ran on CPU.

- [test_audit_hierarchy.py](../../tests/test_audit_hierarchy.py): 4 clean/zero-rate
  variants on the actual released Sudoku tokenizer, plus 6 small upstream text
  tokenizer cases with W1's irregular level schedule and learned L0 on/off.
  Asserts exact code, target, input and reconstruction equality with upstream
  `encode`, `idx_to_ncm_input` and `get_ncm_inputs`, including an enclosing bf16
  context to test the hierarchy's fp32 boundary.
- [test_audit_jacobi.py](../../tests/test_audit_jacobi.py): first **20** released
  Sudoku-hard puzzles in released order. Asserts code-for-code and decoded-output
  equality with upstream `net.generate` greedy, a fixed-point stop rather than
  the pass cap, consistent history/NFE, and stability under one independent
  additional parallel update. It runs the exact `ncmr.jacobi_train` blob from
  the analysis commit in memory when the module is absent in the checkout;
  [audit_support.py](../../tests/audit_support.py) documents that fallback.
- [test_audit_regressions.py](../../tests/test_audit_regressions.py): **3 strict
  expected failures**, confirming F2, F3 and F7 on CPU (1.84 seconds). Policy tests
  extract the actual function AST without executing the training script. Only
  `AssertionError` counts as an expected defect; unrelated exceptions fail the
  suite. An eventual fix becomes strict XPASS and requires removing its xfail.

After adding asset-hash enforcement and tightening expected-failure exception
types, the affected hierarchy/regression subset was rerun: **10 passed, 3 xfailed
in 4.04 seconds**. The Jacobi implementation and its 20 already passing cases
were unchanged.

Assets were downloaded into ignored `tests/_assets/`; pytest and its small
dependencies into ignored `tests/_deps/`. The asset manifest pins SHA-256 hashes
of the official config, checkpoint pair and released puzzle file. Checkpoints
are loaded with `weights_only=True`, and data with the existing restricted
unpickler. The small text tokenizer is randomly initialized upstream code,
not a claim of parity on the full released TinyStories checkpoint.

Reproduce in a Python environment with Torch, NumPy, OmegaConf, hydra-core,
the repository's tokenizer dependencies, and pytest available:

```sh
python -B tests/fetch_audit_assets.py
python -B tests/recompute_r018_r021.py
python -B -m pytest -p no:cacheprovider tests/test_audit_hierarchy.py tests/test_audit_jacobi.py tests/test_audit_regressions.py -q
```

If pytest was installed locally under `tests/_deps`, set `PYTHONPATH=tests/_deps`
and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` before the pytest command. The numeric
script's exit 1 is expected while the immutable source log contains the six
reported discrepancies. No downloaded assets or dependencies belong in a commit.
Only `docs/reviews/` and `tests/` were written; no protected source, config,
registration or continuity file was edited, and no commit was created.
