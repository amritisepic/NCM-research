# Launch audit and P1c synthetic calibration

Audit branch: `codex/audit2`, based on `f9f90ff`; the requested launch
files and S1/S1a/S1b, J1/J1a and P1b-withdrawal text are absent there. This audit
therefore reads **`b3976566a998f2362b82c665780f7573f2606b0c`** through Git,
without checking its files out. The audit ran before the public-history author
rewrite (R026); the base and subject hashes here use their published replacements.
Their trees, including all four subject files and the registration, are identical
to the objects used during the audit. No old author history is republished.
All file:line references below refer to that object. The tests execute its exact
function/class ASTs or complete Jacobi module in memory. Production code is
unchanged. The existing `models.py` Windows portability edit and older audit
artifacts were preserved. No production patch or training launch was made.

P1 blocks a trustworthy registered comparison or reproducible continuation;
P2 is an edge case, incomplete required reporting, or data-preparation robustness
issue. **Resolve P1 items before launching.** These are review proposals, not
applied fixes. The diffs are cumulative in the order printed, against the pinned
subject (later hunks include earlier proposals). They require normal integration
and CUDA validation before use. The CPU tests intentionally test the original
subject; strict xfail marks a demonstrated defect, not a passing implementation.

## Task A — findings

### A2-1 — P1: repair-pool checkpoints do not represent one resumable state

**Locations:** `scripts/train_sudoku_L.py:101`, `:139–144`, `:162–166`,
`:291–307`; `src/ncmr/train.py:80–109` (shared checkpoint payload).

**Failure:** a restart restores the data RNG, constructs a new pool (consuming
another augmented batch), then overwrites the pool with `pool.pt`. Thus even a
perfectly paired save changes the next teacher-forcing batch. `ckpt.pt` is atomic
but `pool.pt` is a second, non-atomic write: interruption can combine new weights
with old pool states. A missing pool silently starts fresh. The final save never
saves the pool, so a non-checkpoint-aligned final step also leaves it stale.
These are distinct failures even though a smoke test can successfully resume.

Put pool and RNG in Trainer's existing atomic `extra` payload and construct a
restored pool without drawing data. Reject legacy repair checkpoints with no pool.
CPU regressions reproduce the extra data draw and missing final pool; another
passing test verifies state advance, solved resets, age-16 resets and untouched
non-reset rows.

**Proposed patch (not applied):**

```diff
--- a/scripts/train_sudoku_L.py
+++ b/scripts/train_sudoku_L.py
@@ -98,7 +98,7 @@
                   ckpt_every=int(tc.ckpt_every), device=dev, amp=bool(tc.get("amp", True)),
                   min_lr_frac=float(tc.get("min_lr_frac", 0.1)), extra_groups=extra)
 modules = {"model": net, **({"feedback": fb} if fb is not None else {})}
-trainer.maybe_resume(modules, rng)
+resume_extra = trainer.maybe_resume(modules, rng)
 gen = torch.Generator(device=dev).manual_seed(seed * 7919 + trainer.step)
 print(f"[sudoku-L] arm={arm} params {n_params(net) / 1e6:.2f}M (+{n_params(fb) if fb else 0} feedback) | "
       f"{fl_step / 1e12:.2f} TFLOP/step, {fl_step * int(tc.steps) / 1e15:.1f} PFLOP total | device {dev}", flush=True)
@@ -136,7 +136,10 @@
 class Pool:
     """In-progress Jacobi decodes of training puzzles (repair arm)."""
 
-    def __init__(self, n):
+    def __init__(self, n, state=None):
+        if state is not None:
+            self.load_state_dict(state)
+            return
         self.prompt, self.sol = batch(n)
         self.codes = torch.zeros(n, S, dtype=torch.long, device=dev)
         self.guess = torch.zeros(n, 89, dtype=torch.long, device=dev)
@@ -159,11 +162,15 @@
             setattr(self, k, v.to(dev))
 
 
-pool = Pool(int(tc.batch_size) // 2) if arm == "repair" else None
-if pool is not None:
-    pst = os.path.join(a.run_dir, "pool.pt")
-    if trainer.step > 0 and os.path.exists(pst):
-        pool.load_state_dict(torch.load(pst, map_location="cpu", weights_only=False))
+pool_state = resume_extra.get("pool")
+if arm == "repair" and trainer.step and pool_state is None:
+    raise RuntimeError("checkpoint lacks synchronized repair pool; exact resume unavailable")
+pool = Pool(int(tc.batch_size) // 2, state=pool_state) if arm == "repair" else None
+
+
+def save_checkpoint():
+    extra = {"pool": pool.state_dict()} if pool is not None else {}
+    trainer.save(modules, rng, extra=extra)
 pool_stats = {"solved": 0, "aged": 0, "advanced": 0}
 
 
@@ -289,13 +296,9 @@
         ev = evaluate(val_p, val_s, caps=(12, 24))
         print(json.dumps(trainer.log({"eval": ev})), flush=True)
     if trainer.step % int(tc.ckpt_every) == 0:
-        trainer.save(modules, rng)
-        if pool is not None:
-            torch.save(pool.state_dict(), os.path.join(a.run_dir, "pool.pt"))
+        save_checkpoint()
     if deadline and time.time() > deadline:
-        trainer.save(modules, rng)
-        if pool is not None:
-            torch.save(pool.state_dict(), os.path.join(a.run_dir, "pool.pt"))
+        save_checkpoint()
         print("[stop] time budget reached; resume by re-running the same command", flush=True)
         sys.exit(0)
 
@@ -303,7 +306,7 @@
     calib.report(trainer.step, {"arm": arm, "batch_size": bs})
     sys.exit(0)
 if not a.eval_only:
-    trainer.save(modules, rng)
+    save_checkpoint()
     torch.save({k: m.state_dict() for k, m in modules.items()}, os.path.join(a.run_dir, "final.pt"))
 caps = tuple(int(c) for c in ec.get("caps", [4, 8, 12, 16, 24]))
 final = {"val": evaluate(val_p, val_s, caps=caps)}
```

### A2-2 — P1: Sudoku corruption RNG restarts instead of continuing

**Locations:** `scripts/train_sudoku_L.py:102`, `:184`, `:292`, `:296`, `:306`.

**Failure:** uniform and on-policy masks use a separate Torch `gen`, but Trainer
only checkpoints the global Torch RNG. Resume seeds `gen` with `seed*7919+step`;
that is not its state after previous masks/replacements. Identical data and
weights therefore produce different corruption immediately after interruption.
The earlier W1 fix was not carried into this trainer. The CPU counterexample
executes the actual generator-initialization AST and compares the next draws.

**Proposed patch (not applied):**

```diff
--- a/scripts/train_sudoku_L.py
+++ b/scripts/train_sudoku_L.py
@@ -99,7 +99,11 @@
                   min_lr_frac=float(tc.get("min_lr_frac", 0.1)), extra_groups=extra)
 modules = {"model": net, **({"feedback": fb} if fb is not None else {})}
 resume_extra = trainer.maybe_resume(modules, rng)
-gen = torch.Generator(device=dev).manual_seed(seed * 7919 + trainer.step)
+gen = torch.Generator(device=dev).manual_seed(seed * 7919)
+if trainer.step and corr.mode != "none":
+    if "corruption_rng" not in resume_extra:
+        raise RuntimeError("checkpoint lacks corruption RNG; exact resume unavailable")
+    gen.set_state(resume_extra["corruption_rng"].cpu())
 print(f"[sudoku-L] arm={arm} params {n_params(net) / 1e6:.2f}M (+{n_params(fb) if fb else 0} feedback) | "
       f"{fl_step / 1e12:.2f} TFLOP/step, {fl_step * int(tc.steps) / 1e15:.1f} PFLOP total | device {dev}", flush=True)
 
@@ -170,6 +174,7 @@
 
 def save_checkpoint():
     extra = {"pool": pool.state_dict()} if pool is not None else {}
+    extra["corruption_rng"] = gen.get_state()
     trainer.save(modules, rng, extra=extra)
 pool_stats = {"solved": 0, "aged": 0, "advanced": 0}
 
```

### A2-3 — P1: auto-resume can relabel an arm or contaminate A0/J0

**Locations:** `scripts/train_sudoku_L.py:66–73`, `:100–101`;
`scripts/train_wmt14_L.py:61–67`, `:93–94`.

**Failure:** use a control run directory for the uniform/on-policy arm, change
seed/data/recipe in the same directory, or run `--eval_only` there. Both scripts
overwrite `config.yaml` and unconditionally load the checkpoint. A0/J0 can then
be fine-tuned weights labelled as released, and a continuation can change its
intervention or data without a warning. Compatible state-dict shapes do not
validate scientific run identity. Repair extra groups can instead fail on load;
that does not protect the compatible-arm cases. Both eval-only paths have CPU
counterexamples. W1's manifest fix is absent here.

Validate before saving metadata. The proposal includes data and base-model
digests and precision; existing checkpoints without a manifest fail closed.

**Proposed patch (not applied):**

```diff
--- a/scripts/train_sudoku_L.py
+++ b/scripts/train_sudoku_L.py
@@ -70,6 +70,26 @@
 rng = np.random.default_rng(seed)
 data = SudokuTrain("hard", a.data)
 os.makedirs(a.run_dir, exist_ok=True)
+# Validate the immutable run identity before overwriting metadata or resuming.
+import hashlib
+def digest(path):
+    with open(path, "rb") as stream:
+        return hashlib.file_digest(stream, "sha256").hexdigest()
+
+ckpt_path = os.path.join(a.run_dir, "ckpt.pt")
+manifest_path = os.path.join(a.run_dir, "run_manifest.json")
+if a.eval_only and os.path.exists(ckpt_path):
+    raise ValueError("released-model evaluation requires a directory without a checkpoint")
+base_dir = os.path.join(ROOT, "artifacts", "sudoku", "hard")
+amp = bool(tc.get("amp", True))
+amp_dtype = ("bf16" if dev == "cpu" or torch.cuda.get_device_capability()[0] >= 8 else "fp16") if amp else "fp32"
+manifest = {"config": OmegaConf.to_container(cfg, resolve=True), "data_sha256": digest(a.data),
+            "base": {name: digest(os.path.join(base_dir, name)) for name in ("config.yaml", "tokenizer.pt", "generator.pt")},
+            "precision": {"device": dev, "amp_dtype": amp_dtype, "torch": torch.__version__}}
+if os.path.exists(ckpt_path):
+    if not os.path.exists(manifest_path) or json.load(open(manifest_path)) != manifest:
+        raise ValueError("resume manifest differs or is absent; use a fresh run directory")
+json.dump(manifest, open(manifest_path, "w"), indent=2)
 OmegaConf.save(cfg, os.path.join(a.run_dir, "config.yaml"))
 
 B_ = load_ncm(os.path.join(ROOT, "artifacts", "sudoku", "hard"), device=dev)
```

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -64,6 +64,26 @@
 torch.manual_seed(seed)
 rng = np.random.default_rng(seed)
 os.makedirs(a.run_dir, exist_ok=True)
+# Validate the immutable run identity before overwriting metadata or resuming.
+import hashlib
+def digest(path):
+    with open(path, "rb") as stream:
+        return hashlib.file_digest(stream, "sha256").hexdigest()
+
+ckpt_path = os.path.join(a.run_dir, "ckpt.pt")
+manifest_path = os.path.join(a.run_dir, "run_manifest.json")
+if a.eval_only and os.path.exists(ckpt_path):
+    raise ValueError("released-model evaluation requires a directory without a checkpoint")
+base_dir = os.path.join(ROOT, "artifacts", "wmt14")
+amp = bool(tc.get("amp", True))
+amp_dtype = ("bf16" if dev == "cpu" or torch.cuda.get_device_capability()[0] >= 8 else "fp16") if amp else "fp32"
+manifest = {"config": OmegaConf.to_container(cfg, resolve=True), "data_sha256": digest(a.data),
+            "base": {name: digest(os.path.join(base_dir, name)) for name in ("config.yaml", "tokenizer.pt", "generator.pt")},
+            "precision": {"device": dev, "amp_dtype": amp_dtype, "torch": torch.__version__}}
+if os.path.exists(ckpt_path):
+    if not os.path.exists(manifest_path) or json.load(open(manifest_path)) != manifest:
+        raise ValueError("resume manifest differs or is absent; use a fresh run directory")
+json.dump(manifest, open(manifest_path, "w"), indent=2)
 OmegaConf.save(cfg, os.path.join(a.run_dir, "config.yaml"))
 
 B_ = load_ncm(os.path.join(ROOT, "artifacts", "wmt14"), device=dev)
```

### A2-4 — P1: J2 targets the autocast map, not the registered fp32 greedy map

**Locations:** `scripts/train_wmt14_L.py:127–148`, `:223–227`;
`src/ncmr/jacobi_train.py:21–24`, `:42`.

**Failure:** `consist_loss` is called inside `trainer.ctx`. `net.eval()` turns
off dropout but leaves autocast enabled. `parallel_logits` only disables autocast
for VQ input reconstruction, then runs the generator under the caller's bf16/fp16
context. Casting its logits to float does not recover fp32 argmax decisions.
Consequently the sampled trajectory and its label can be a different fixed point
from the fp32 sequential decoder whose NFE J1 optimizes. On a T4 it can even be an
fp16 map. This is not the old nested-context leak; the latest Trainer fixes that
leak, but inheriting the outer context here is enough.

Run the entire no-gradient teacher trajectory in explicitly disabled autocast;
keep gradient-bearing training passes in the intended AMP context. Require a
fixed-point stop before accepting the target. The CPU test uses a deterministic
precision-sensitive external network and the actual trajectory/sampling code.
Real CUDA argmax divergence rates are not measured by that test. Make the TF32
policy explicit and use the same policy for teacher, exact and sequential eval;
the diff below adopts strict fp32 on CUDA.

**Proposed patch (not applied):**

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -61,6 +61,9 @@
 if a.eval_only:
     tc.steps = 0
 dev = pick_device()
+if dev == "cuda":
+    torch.backends.cuda.matmul.allow_tf32 = False
+    torch.backends.cudnn.allow_tf32 = False
 torch.manual_seed(seed)
 rng = np.random.default_rng(seed)
 os.makedirs(a.run_dir, exist_ok=True)
@@ -151,8 +154,10 @@
     z, pm = prefix(src)
     was = net.training
     net.eval()
-    with torch.no_grad():
-        out = jacobi_decode(net, lambda c, h: z, g, S, dev, K + 1, pad_mask=pm, keep_history=True)
+    with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
+        out = jacobi_decode(net, lambda c, h: z.float(), g, S, dev, K + 1, pad_mask=pm, keep_history=True)
+        if not bool(out["stop"].eq(1).all()):
+            raise RuntimeError("consistency teacher failed to reach a fixed point")
     net.train(was)
     y = out["codes"]
     states = [torch.zeros_like(y)] + out["history"]            # input of pass t+1 = output of pass t
```

### A2-5 — P2: an initially fixed zero state is sampled as pre-fixed-point

**Locations:** `scripts/train_wmt14_L.py:138–143`; `src/ncmr/jacobi_train.py:45`.

**Failure:** if the model's true fixed point is the all-zero initialization,
`first_fp` is zero. `rand * 0` selects state zero, which equals the target.
There are no strictly pre-fixed-point states for this source. The normal,
nondegenerate path is correct: it excludes the first state equal to `y`, not
just the later pass detecting equality. Sampling uses replacement, so four draws
from fewer than four states are intentional. Both cases have CPU coverage.

This edge is not evidence that the released WMT14 model emits all-zero codes.
The conservative patch fails explicitly, rather than silently changing the
registered sampling distribution or training on a forbidden state. If such
sources occur, define exclusion/replacement and its compute accounting before
continuing; no silent fallback to fixed-point states is valid.

**Proposed patch (not applied):**

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -163,6 +163,8 @@
     states = [torch.zeros_like(y)] + out["history"]            # input of pass t+1 = output of pass t
     hist = torch.stack(states, 1)                               # (g, T+1, S)
     first_fp = (hist == y[:, None]).all(-1).float().argmax(1)   # first state equal to the fixed point (>= 1)
+    if not bool((first_fp > 0).all()):
+        raise RuntimeError("no strictly pre-fixed-point state: define a degenerate-trajectory policy before continuing")
     t = (torch.rand(g, m_states, device=dev) * first_fp[:, None]).long()   # uniform over states before it
     inp = hist[torch.arange(g, device=dev)[:, None], t].reshape(-1, S)[:n]
     tgt = y.repeat_interleave(m_states, 0)[:n]
```

### A2-6 — P1: same seed does not give the registered common data order

**Locations:** `scripts/train_sudoku_L.py:70`, `:131–152`, `:162`, `:267–275`;
`scripts/train_wmt14_L.py:113–115`, `:129–130`, `:224–227`.

**Failure:** Sudoku control/uniform/onpolicy do share the NumPy stream, but
repair consumes it for initial pool creation and for model-dependent solved/aged
resets, then draws only 128 fresh teacher-forcing examples per step. Reset
behavior therefore changes later supervised examples and augmentation. WMT14
control draws 128 rows/step; consistency draws 64 TF rows plus 16 trajectory
sources (80 total), so even the next step's common supervised positions differ.
These differences are independent of Torch dropout or trajectory-sampling RNG.
Both mismatches have executable CPU counterexamples.

Reserve a full, common batch per optimizer step and use its fixed subsets for
the different objectives. Use a separately checkpointed pool-data RNG for
stateful repair replacement. The repair pool necessarily has its own exposure
history; record it as such rather than claiming identical exposures. This changes
draw order: old runs/checkpoints cannot be mixed with corrected runs.

**Proposed patch (not applied):**

```diff
--- a/scripts/train_sudoku_L.py
+++ b/scripts/train_sudoku_L.py
@@ -68,6 +68,7 @@
 dev = pick_device()
 torch.manual_seed(seed)
 rng = np.random.default_rng(seed)
+pool_rng = np.random.default_rng(np.random.SeedSequence([seed, 2201]))
 data = SudokuTrain("hard", a.data)
 os.makedirs(a.run_dir, exist_ok=True)
 # Validate the immutable run identity before overwriting metadata or resuming.
@@ -152,8 +153,8 @@
         return tok.encode(sol)
 
 
-def batch(n):
-    p, s = data.batch(rng, n)
+def batch(n, data_rng=None):
+    p, s = data.batch(rng if data_rng is None else data_rng, n)
     return torch.from_numpy(p).to(dev), torch.from_numpy(s).to(dev)
 
 
@@ -164,7 +165,7 @@
         if state is not None:
             self.load_state_dict(state)
             return
-        self.prompt, self.sol = batch(n)
+        self.prompt, self.sol = batch(n, data_rng=pool_rng)
         self.codes = torch.zeros(n, S, dtype=torch.long, device=dev)
         self.guess = torch.zeros(n, 89, dtype=torch.long, device=dev)
         self.active = torch.zeros(n, dtype=torch.bool, device=dev)
@@ -174,7 +175,7 @@
         k = int(m.sum())
         if k == 0:
             return
-        p, s = batch(k)
+        p, s = batch(k, data_rng=pool_rng)
         self.prompt[m], self.sol[m] = p, s
         self.codes[m], self.guess[m], self.active[m], self.age[m] = 0, 0, False, 0
 
@@ -186,6 +187,10 @@
             setattr(self, k, v.to(dev))
 
 
+if trainer.step and arm == "repair":
+    if "pool_data_rng" not in resume_extra:
+        raise RuntimeError("checkpoint lacks pool data RNG")
+    pool_rng.bit_generator.state = resume_extra["pool_data_rng"]
 pool_state = resume_extra.get("pool")
 if arm == "repair" and trainer.step and pool_state is None:
     raise RuntimeError("checkpoint lacks synchronized repair pool; exact resume unavailable")
@@ -195,6 +200,7 @@
 def save_checkpoint():
     extra = {"pool": pool.state_dict()} if pool is not None else {}
     extra["corruption_rng"] = gen.get_state()
+    extra["pool_data_rng"] = pool_rng.bit_generator.state
     trainer.save(modules, rng, extra=extra)
 pool_stats = {"solved": 0, "aged": 0, "advanced": 0}
 
@@ -297,8 +303,8 @@
 while trainer.step < int(tc.steps):
     with trainer.ctx:
         if arm == "repair":
-            prompt, sol = batch(bs - bs // 2)
-            loss_tf = tf_loss(prompt, sol)
+            prompt, sol = batch(bs)
+            loss_tf = tf_loss(prompt[:bs - bs // 2], sol[:bs - bs // 2])
             loss_pool = pool_loss()
             loss = 0.5 * (loss_tf + loss_pool)
             logs = {"loss_tf": round(loss_tf.item(), 4), "loss_pool": round(loss_pool.item(), 4)}
```

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -106,6 +106,8 @@
                              codebook=vq.vqvae_vocab_sizes[0], vq_dim=tok.embed_dim, query_level0=True)
 bs = int(tc.batch_size)
 n_c = bs // 2 if arm == "consist" else 0
+if m_states < 1 or (n_c and (bs % 2 or n_c % m_states)):
+    raise ValueError("consistency requires an even batch and a half-batch divisible by states_per_traj")
 # per step: (bs - n_c) TF + n_c consistency samples (3 units each) + trajectories for n_c / m sources (<= K + 1 passes)
 fl_step = fwd * (3 * bs + (K + 1) * (n_c // max(m_states, 1)))
 params = [p for p in net.parameters() if p.requires_grad]
@@ -147,10 +149,10 @@
 traj_stats = {"passes": 0.0, "n": 0, "states_before_fp": 0.0}
 
 
-def consist_loss(n):
+def consist_loss(n, sources):
     """Self-consistency on the current model's own greedy Jacobi trajectories."""
     g = max(1, n // m_states)
-    src, _ = batch(g)
+    src = sources[:g]
     z, pm = prefix(src)
     was = net.training
     net.eval()
@@ -248,10 +250,10 @@
 n_curve = int(ec.get("n_curve", 500))
 while trainer.step < int(tc.steps):
     with trainer.ctx:
-        src, tgt = batch(bs - n_c)
-        loss_tf = tf_loss(src, tgt)
+        src, tgt = batch(bs)
+        loss_tf = tf_loss(src[:bs - n_c], tgt[:bs - n_c])
         if n_c:
-            loss_c = consist_loss(n_c)
+            loss_c = consist_loss(n_c, src[bs - n_c:][::m_states])
             loss = ((bs - n_c) * loss_tf + n_c * loss_c) / bs
             logs = {"loss_tf": round(loss_tf.item(), 4), "loss_consist": round(loss_c.item(), 4)}
         else:
```

### A2-7 — P1: WMT14 matches upper bounds, not necessarily executed training compute

**Locations:** `scripts/train_wmt14_L.py:82–96`, `:135`, `:145–147`, `:271`;
`src/ncmr/jacobi_train.py:41–54`; `configs/scale/L_wmt_{ctrl,consist}.yaml`.

**Failure:** with K=11, B=128 and m=4, control costs 384 forward-equivalent
rows/step; consistency's upper bound is 384+12*16=576. Thus 3000*384 and 2000*576
both equal **1,152,000**. There is no arithmetic error in this upper bound.
However the actual teacher breaks when its slowest source finishes. A batch
converging in two passes costs 384+2*16=416 rows, giving **832,000** units over
2000 steps, **27.78% below** control. For an 11-pass slowest source the deficit
is 2.78%. The log's mean per-sentence `nfe` cannot reconstruct actual work:
stopped rows remain in every full-batch forward until all rows stop. Count
`g * len(history)`, not the sum of per-sentence NFE. The CPU test executes the
decoder and reproduces the two-pass counterexample while confirming matched
upper bounds. This does not assert a measured deficit on unreleased training runs.

One concrete way to preserve the registered 2000-step schedule is to execute
all K+1 teacher forwards even after all sources stop, holding stopped codes
fixed. The diff makes that budget explicit; this spends compute without adding
training examples. Alternatively checkpoint actual forward-row counts and stop
on a matched budget, with a declared change to the schedule. Merely relabelling
an upper bound does not establish equal executed compute.

Sudoku's 2500*3 and 1875*4 schedules match the common generator proxy exactly.
Repair uses half TF + half repair, so no extra generator forward is missing in
that proxy. Decoder/verifier/pool work and feedback optimization are additional.
Neither trainer establishes end-to-end FLOP equality. The shared analytic model
also prices masked attention as sparse although these models use dense SDPA;
WMT14 prices level 0 as a query although it is predicted from the prefix. These
common-scale limitations do not by themselves create a new arm-ratio error.
Report the generator proxy, precision and wall time explicitly; use measured,
attention-inclusive costs for S1b/J1a criterion-(d) claims.

**Proposed patch (not applied):**

```diff
--- a/src/ncmr/jacobi_train.py
+++ b/src/ncmr/jacobi_train.py
@@ -26,7 +26,7 @@
 
 @torch.no_grad()
 def jacobi_decode(net, prefix_fn, n: int, S: int, device, max_passes: int, pad_mask=None, check_fn=None,
-                  keep_history: bool = False):
+                  keep_history: bool = False, full_budget: bool = False):
     """Greedy Jacobi decoding with per-sample stopping.
 
     prefix_fn(codes, has_prev) -> z_prefix: lets the prefix depend on the previous pass (board feedback);
@@ -50,7 +50,7 @@
         codes = new
         if keep_history:
             hist.append(codes.clone())
-        if bool(done.all()):
+        if bool(done.all()) and not full_budget:
             break
     return {"codes": codes, "nfe": nfe, "stop": stop, "history": hist}
 
```

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -157,7 +157,7 @@
     was = net.training
     net.eval()
     with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
-        out = jacobi_decode(net, lambda c, h: z.float(), g, S, dev, K + 1, pad_mask=pm, keep_history=True)
+        out = jacobi_decode(net, lambda c, h: z.float(), g, S, dev, K + 1, pad_mask=pm, keep_history=True, full_budget=True)
         if not bool(out["stop"].eq(1).all()):
             raise RuntimeError("consistency teacher failed to reach a fixed point")
     net.train(was)
@@ -297,7 +297,7 @@
 json.dump({"hyp_test": hyp, "nfe_test": final["test"].pop("nfe_per_sentence")}, open(os.path.join(a.run_dir, "hyp_test.json"), "w"))
 final["val"].pop("nfe_per_sentence", None)
 res = {"run": os.path.basename(a.run_dir.rstrip("/")), "arm": arm, "config": OmegaConf.to_container(cfg),
-       "params_total": n_params(net), "steps": trainer.step, "flops_est_total_upper": fl_step * trainer.step,
+       "params_total": n_params(net), "steps": trainer.step, "generator_flops_proxy_total": fl_step * trainer.step,
        "wall_s": trainer.wall_prev + time.time() - trainer.t_start, "device": dev,
        "gpu": torch.cuda.get_device_name() if dev == "cuda" else None, "torch": torch.__version__, "final_eval": final}
 json.dump(res, open(os.path.join(a.run_dir, "result.json"), "w"), indent=1)
```

### A2-8 — P2: WMT preparation can silently return too few or stale pairs

**Locations:** `scripts/prep_wmt14.py:38–40`, `:79–99`.

**Failure:** preparation inspects only `int(n*1.15)+1000` candidates. Filtering
English targets above 63 tokens can leave fewer than n, despite enough eligible
pairs in the remaining permutation; the script saves the shorter archive without
an assertion. Existing files also cause unconditional success, even if produced
with a different `--n` or `--shards`. This is a supported-input failure, not a
measurement of the actual default corpus's acceptance rate. A CPU synthetic
corpus retains 10% of candidates and reproduces the undershoot.

Walk the complete uniform permutation until n eligible rows are collected,
then assert n. This samples uniformly from the eligible (length-filtered)
population, not unconditionally from all training pairs. Save and validate the
requested selection settings before reusing an archive. Pin/cache the dataset
revision and tokenizer revision/hashes for production provenance as well; the
original mutable `resolve/main` URL alone is not an immutable dataset identity.

**Proposed patch (not applied):**

```diff
--- a/scripts/prep_wmt14.py
+++ b/scripts/prep_wmt14.py
@@ -35,7 +35,14 @@
 
 from omegaconf import OmegaConf  # noqa: E402
 
+selection = {"n": a.n, "shards": a.shards, "seed": 0}
+selection_path = a.out.replace(".npz", "_selection.json")
 if os.path.exists(a.out) and os.path.exists(a.out.replace(".npz", "_val_text.json")):
+    if not os.path.exists(selection_path) or json.load(open(selection_path)) != selection:
+        raise ValueError("existing archive has different or missing selection metadata")
+    with np.load(a.out) as previous:
+        if len(previous["src"]) != a.n or len(previous["tgt"]) != a.n:
+            raise ValueError("existing archive has the wrong number of training pairs")
     print("exists:", a.out)
     sys.exit(0)
 BASE = "https://huggingface.co/datasets/wmt/wmt14/resolve/main/de-en/"
@@ -81,7 +88,7 @@
 for s in a.shards:
     train += pairs(fetch(f"train-0000{s}-of-00003.parquet"))
 print(f"{len(train)} training pairs in shards {a.shards}", flush=True)
-sel = rng.permutation(len(train))[: int(a.n * 1.15) + 1000]
+sel = rng.permutation(len(train))
 Ss, Ts = [], []
 for i in range(0, len(sel), 20000):
     s_, t_, _ = encode([train[j] for j in sel[i:i + 20000]])
@@ -90,9 +97,12 @@
     if sum(len(x) for x in Ss) >= a.n:
         break
 src, tgt = np.concatenate(Ss)[: a.n], np.concatenate(Ts)[: a.n]
+if len(src) != a.n:
+    raise ValueError(f"only {len(src)} eligible pairs, requested {a.n}")
 val = pairs(fetch("validation-00000-of-00001.parquet"))
 vs, vt, keep = encode(val)
 np.savez_compressed(a.out, src=src, tgt=tgt, val_src=vs, val_tgt=vt, pad=PAD, eos=EOS)
+json.dump(selection, open(selection_path, "w"), indent=2)
 json.dump({"src": [d for d, _ in val], "ref": [e for _, e in val]},   # all 3000, for BLEU
           open(a.out.replace(".npz", "_val_text.json"), "w"))
 print(json.dumps({"train_pairs": int(len(src)), "val_pairs": int(len(vs)), "val_total": len(val),
```

### A2-9 — P2: S1b tail statistics are missing and c* is not recorded

**Locations:** `scripts/train_sudoku_L.py:238–254`, `:308–317`.

**Failure:** each cap reports solve rate and mean NFE, but not cap-hit count,
p90 or maximum NFE as S1b requires. `stop_counts` describes the maximum-cap run,
not each cap. Test per-puzzle data retains only the normal check mode, losing
the other modes' stop/NFE traces. The file also never records the cap selected
on validation, making later test-cap selection easy to confuse with the
registered rule. These omissions do not invalidate the current solve-rate
arithmetic; they prevent a complete S1b report. A CPU evaluator counterexample
checks the missing fields. Largest-cap selection must use validation only;
verify the resulting test NFE bound separately before giving H10 credit.

The diff adds per-cap tails (cap-hit = not stopped at or before c), keeps every
mode's per-puzzle records, and records c* from the ordinary check validation
curve. A0's saved sequential booleans can then be aligned with each arm's test
booleans to report wrong→right/right→wrong transitions; those comparisons and
256-puzzle batch-size-1 timing are still required before a criterion-(d) claim.

**Proposed patch (not applied):**

```diff
--- a/scripts/train_sudoku_L.py
+++ b/scripts/train_sudoku_L.py
@@ -275,7 +275,10 @@
     res["seq_solved_pct"] = 100 * float(np.mean(seq_ok))
     for mode, rows in d2.items():
         res[f"d2_{mode}"] = {f"cap{c}": {"solved_pct": 100 * float(np.mean([r["ok_at"][i] for r in rows])),
-                                        "mean_nfe": float(np.mean([min(r["nfe"], c) for r in rows]))}
+                                        "mean_nfe": float(np.mean([min(r["nfe"], c) for r in rows])),
+                                        "p90_nfe": float(np.percentile([min(r["nfe"], c) for r in rows], 90)),
+                                        "max_nfe": int(max(min(r["nfe"], c) for r in rows)),
+                                        "cap_hit_count": sum(r["nfe"] > c or (r["nfe"] == c and r["stop"] == 0) for r in rows)}
                              for i, c in enumerate(caps)}
         res[f"d2_{mode}"]["stop_counts"] = np.bincount([r["stop"] for r in rows], minlength=3).tolist()
     if final:
@@ -289,7 +292,8 @@
             res["by_depth"][name] = {"n": len(m), "seq_pct": 100 * float(np.mean([seq_ok[i] for i in m])) if m else None,
                                      "d2_check_capmax_pct": 100 * float(np.mean([d2["check"][i]["ok_at"][-1] for i in m])) if m else None}
         res["per_puzzle"] = {"seq": [bool(v) for v in seq_ok], "d2_check_nfe": [r["nfe"] for r in d2["check"]],
-                             "d2_check_ok_cap": [[bool(v) for v in r["ok_at"]] for r in d2["check"]]}
+                             "d2_check_ok_cap": [[bool(v) for v in r["ok_at"]] for r in d2["check"]],
+                             "d2_all_modes": d2}
     net.train()
     return res
 
@@ -341,12 +345,13 @@
     torch.save({k: m.state_dict() for k, m in modules.items()}, os.path.join(a.run_dir, "final.pt"))
 caps = tuple(int(c) for c in ec.get("caps", [4, 8, 12, 16, 24]))
 final = {"val": evaluate(val_p, val_s, caps=caps)}
+c_star = max((c for c in caps if final["val"]["d2_check"][f"cap{c}"]["mean_nfe"] <= 12), default=None)
 tp = np.array(safe_pickle_load(os.path.join(ROOT, "data", "sudoku", "hard", "puzzles.pickle"))[: a.n_test])
 ts = np.array(safe_pickle_load(os.path.join(ROOT, "data", "sudoku", "hard", "solutions.pickle"))[: a.n_test])
 final["test_released"] = evaluate(tp, ts, caps=caps, final=True)
 res = {"run": os.path.basename(a.run_dir.rstrip("/")), "arm": arm, "config": OmegaConf.to_container(cfg),
        "params_total": n_params(net) + (n_params(fb) if fb else 0), "steps": trainer.step,
-       "flops_est_total": fl_step * trainer.step, "wall_s": trainer.wall_prev + time.time() - trainer.t_start,
+       "operating_cap_from_val": c_star, "generator_flops_proxy_total": fl_step * trainer.step, "wall_s": trainer.wall_prev + time.time() - trainer.t_start,
        "device": dev, "gpu": torch.cuda.get_device_name() if dev == "cuda" else None, "torch": torch.__version__,
        "final_eval": final}
 json.dump(res, open(os.path.join(a.run_dir, "result.json"), "w"), indent=1)
```

### A2-10 — P1 for a J1a claim: required comparator hypotheses are discarded

**Locations:** `scripts/train_wmt14_L.py:159`, `:179–198`, `:244`, `:263–269`.

**Failure:** capped hypotheses are computed but discarded. Only exact test
hypotheses and passes are saved, so the registered 1000-resample paired BLEU
comparison of J2 exact against J0/J1 capped cannot be reconstructed. Default
caps {6,8,9} also omit `ceil(J2 exact mean)` whenever it is another integer
(including 7 or 10); meeting the original <=8.25 threshold does not ensure the
needed comparator cap is present. Logging corpus BLEU alone cannot yield a
paired bootstrap interval. The exact-vs-sequential check is only a percentage,
so an unequal sentence does not stop a purported exact result.

Preserve exact and capped hypotheses in sentence order, covering every integer
cap through K+1, and reject sequential/code disagreement. A CPU regression
checks the discarded capped-hypothesis payload. The proposal collects the data;
the registered paired bootstrap across J0/J1/J2, length-ratio constraint and
batch-size-1 attention-inclusive timing must still be performed before a claim.
Do not select models or operating points from repeated test evaluations.

**Proposed patch (not applied):**

```diff
--- a/scripts/train_wmt14_L.py
+++ b/scripts/train_wmt14_L.py
@@ -185,10 +185,11 @@
 
 
 @torch.no_grad()
-def evaluate(srcs, refs, bs_=128, sequential=True, caps=(6, 8, 9)):
+def evaluate(srcs, refs, bs_=128, sequential=True, caps=None):
     """fp32 eval: exact Jacobi passes + BLEU of its output (= sequential output); optionally verifies that.
     caps: also the output after at most c passes (fixed-cap Jacobi, J1a), with BLEU, mean passes and length ratio."""
     import sacrebleu
+    caps = tuple(range(1, K + 2)) if caps is None else tuple(caps)
     net.eval()
     hyp, nfe, same = [], [], []
     hyp_cap = {c: [] for c in caps}
@@ -200,7 +201,10 @@
             out = jacobi_decode(net, lambda c, h: z, len(x), S, dev, K + 1, pad_mask=pm, keep_history=True)
             if sequential:
                 seq = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=EOS)
-                same += (seq == out["codes"]).all(1).tolist()
+                equality = (seq == out["codes"]).all(1)
+                if not bool(equality.all()):
+                    raise RuntimeError("exact Jacobi output differs from sequential greedy")
+                same += equality.tolist()
             with torch.autocast(device_type=dev, enabled=False):
                 toks = tok.decoder(vq.idx_to_z_q(out["codes"])).argmax(-1).tolist()
             hyp += [detok(r) for r in toks]
@@ -224,7 +228,7 @@
     res["nfe_per_sentence"] = nfe
     if sequential:
         res["fixed_point_equals_sequential_pct"] = 100 * float(np.mean(same))
-    return res, hyp
+    return res, hyp, hyp_cap
 
 
 @torch.no_grad()
@@ -270,7 +274,7 @@
             traj_stats.update(passes=0.0, n=0, states_before_fp=0.0)
         print(json.dumps(trainer.log({"loss": round(loss.item(), 4), "gnorm": round(gn, 3), **logs})), flush=True)
     if trainer.step % int(tc.eval_every) == 0 or trainer.step == int(tc.steps):
-        ev, _ = evaluate(val_text["src"][:n_curve], val_text["ref"][:n_curve], sequential=False)
+        ev, _, _ = evaluate(val_text["src"][:n_curve], val_text["ref"][:n_curve], sequential=False)
         ev.pop("nfe_per_sentence")
         ev["val_nll"] = val_nll()
         print(json.dumps(trainer.log({"eval": ev})), flush=True)
@@ -289,13 +293,14 @@
     torch.save({"model": net.state_dict()}, os.path.join(a.run_dir, "final.pt"))
 nv = a.n_val or len(val_text["src"])
 final = {"val_nll": val_nll()}
-final["val"], _ = evaluate(val_text["src"][:nv], val_text["ref"][:nv])
+final["val"], hyp_val, hyp_val_cap = evaluate(val_text["src"][:nv], val_text["ref"][:nv])
 srcs = safe_pickle_load(os.path.join(ROOT, "data", "wmt14", "sources.pickle"))
 refs = safe_pickle_load(os.path.join(ROOT, "data", "wmt14", "reference_translations.pickle"))
 nt = a.n_test or len(srcs)
-final["test"], hyp = evaluate(srcs[:nt], refs[:nt])
-json.dump({"hyp_test": hyp, "nfe_test": final["test"].pop("nfe_per_sentence")}, open(os.path.join(a.run_dir, "hyp_test.json"), "w"))
-final["val"].pop("nfe_per_sentence", None)
+final["test"], hyp, hyp_cap = evaluate(srcs[:nt], refs[:nt])
+json.dump({"hyp_test": hyp, "hyp_test_capped": hyp_cap, "nfe_test": final["test"].pop("nfe_per_sentence")}, open(os.path.join(a.run_dir, "hyp_test.json"), "w"))
+json.dump({"hyp_val": hyp_val, "hyp_val_capped": hyp_val_cap,
+           "nfe_val": final["val"].pop("nfe_per_sentence")}, open(os.path.join(a.run_dir, "hyp_val.json"), "w"))
 res = {"run": os.path.basename(a.run_dir.rstrip("/")), "arm": arm, "config": OmegaConf.to_container(cfg),
        "params_total": n_params(net), "steps": trainer.step, "generator_flops_proxy_total": fl_step * trainer.step,
        "wall_s": trainer.wall_prev + time.time() - trainer.t_start, "device": dev,
```

## Checks without a new blocking finding

- **Repair update/reset:** the current pass's argmax advances codes, its decoded
  board becomes the next feedback, `active` switches on, and age increments once.
  Solved states and age>=16 reset prompt/solution/codes/board/active/age together.
  No fixed-point reset is registered for this training pool. It intentionally
  uses the gradient-bearing training pass (including dropout); it is not an
  eval-mode frozen-model trajectory. The CPU state/reset test passes.
- **Jacobi and caps:** NFE includes the detecting pass; the `t>1` guard means an
  initially fixed initialization is detected on pass 2. Absorbing stopped rows
  make both trainers' history indexing correct, including batches that stop
  before a requested cap. Independent capped reruns agree in the CPU stub test.
  Four real released hard puzzles match sequential greedy code-for-code and are
  stable under an independent further pass. Feedback breaks the triangular
  no-feedback convergence bound; cap stops must be reported, not called exact.
- **Verifier:** rules plus puzzle clues, without access to the target solution.
  Correct grid, broken grid and valid-but-clue-inconsistent grid tests pass.
  Dataset prompts are assumed well formed. Scoring against the supplied unique
  solution is consistent with verifier correctness for these puzzles.
- **Precision boundaries:** canonical encoding, VQ reconstruction and token
  decoding explicitly disable autocast. WMT final decoding is outside training
  autocast and explicitly disables it; Sudoku final decoding is called outside
  autocast and its decoder helper disables it. Latest Trainer creates fresh
  contexts, so the earlier W1 nested-context leak is not present here. These
  observations do not fix J2's teacher precision (A2-4) or prove CUDA/CPU parity.
  Sudoku on-policy proposals also inherit training AMP; S1 does not explicitly
  require a fp32 policy teacher. Record that policy rather than assuming those
  proposals equal fp32 inference. Sudoku's TF32 settings remain implicit.
- **WMT targets (R022):** `encode` uses unadorned text token ids, one EOS, then
  PAD through length 64, filters empty and >63-token targets and right-pads or
  truncates source to the 96-slot prefix. Boundary/empty/long-target CPU tests
  pass. BLEU evaluates all validation text; val NLL uses the eligible subset.
  No test sentences are read in preparation. End-to-end Qwen-tokenizer parity
  and released WMT14 checkpoint behavior were not run locally: those assets
  are absent. The R022 48-test-pair format diagnostic is prior test-set protocol
  development, so the full history is not a pristine never-inspected test set.
- **Splits:** preparation reads train shards and `validation-...`, while final
  test reads the released `sources.pickle` and reference file. HF's
  [primary dataset card](https://huggingface.co/datasets/wmt/wmt14/blob/main/README.md)
  identifies separate de-en training/validation/test files and 3000/3003
  validation/test rows. The official historical
  [builder](https://huggingface.co/datasets/wmt/wmt14/blame/fd7a7091c00eddd88654550900dbb00369d00be5/wmt14.py)
  and [subset definitions](https://huggingface.co/datasets/wmt/wmt14/blob/fd7a7091c00eddd88654550900dbb00369d00be5/wmt_utils.py)
  map German-English validation to newstest2013 and test to newstest2014;
  the builder's additional newsdev2014 subset applies only to Hindi-English.
  Byte-level equality of current HF/released files was not established without
  those local assets. No control-flow path uses final test for training
  or curve selection. Bind file hashes and save source indices in launch manifests
  before treating naming alone as dataset provenance.
- **Sudoku separation:** actual `assets/sudoku_train/hard.npz` has 100,000 rows;
  `SudokuTrain` holds out its last 2,000, leaving 98,000. Validation uses its
  first 1,000; test reads separate released puzzles/solutions only at the end.
  Exact raw generated train/val and generated/released puzzle overlaps were
  checked separately in `codex_audit2_data.json`. This is not a certificate
  against augmentation-equivalent puzzles or repeated solution grids: the
  released solutions file and an orbit-level comparison were not available.
  `gen_sudoku.py` only excludes exact released solution grids when those files
  exist at generation time; it does not enforce augmentation-orbit exclusion.
- **S1b feedback controls:** normal, zero and shuffled feedback modes exist.
  Shuffling rolls within an evaluation batch; a singleton batch rolls to itself.
  Default n=1000/2000 and bs=250 avoid that edge. A one-puzzle smoke test cannot
  validate the other-puzzle ablation. Keep batch partition fixed and use an
  external donor for singleton use before interpreting that supported edge case.

## CPU verification and reproducibility

New combined suite: **12 passed, 13 strict xfailed in 37.56 seconds**.
No runtime/fixture exception is accepted as an expected defect. The released
check uses hash-verified existing audit assets.
An installed CUDA Torch build emits a CUDA-initialization warning even during
these CPU operations; no tensor or model was moved to CUDA.

```sh
# Activate the existing PyTorch environment; if needed, PYTHONPATH=tests/_deps
# and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 for the audit's locally installed pytest.
python -B tests/fetch_audit_assets.py
python -B -m pytest -p no:cacheprovider tests/test_audit2_launch.py tests/test_p1c_synthetic.py -q
python -B scripts/p1c_synthetic.py --out docs/reviews/p1c_synthetic.json
```

`audit2_support.py` pins the audited Git object. Regression tests will remain
expected failures against that historical object after production fixes; repin
explicitly, update expectations for the chosen degenerate-trajectory policy and
remove the corresponding xfail when validating integrated fixes.
This audit does not claim to have run a full training step on the 120M WMT model,
measured launch GPU memory/throughput, or run three complete fine-tunes.

## Task B — P1c step 1

`scripts/p1c_synthetic.py` uses the actual working-tree `TwoStreamLevelModel`
fact/ar implementation, one frozen tiny upstream tokenizer, the actual upstream
transformer and tied heads. The two models start from identical common weights;
AR keeps its existing normal previous-code embeddings and its different mask.
The default constructor has the same function as at the audit ref; the newer
ref's optional `prev_init` argument is not used. No source or model change, no
P1b rerun and no P1c step-2 side channel is included.

There is one modeled four-code binary level, constant coarser conditioning,
width 32, one layer, two heads, VQ dim 8, no dropout, CPU fp32, 800 optimizer
steps/model, seeds 0/1/2. Exact population probabilities over all 16 strings are
used for both fitting and evaluation. This is an instrument calibration, with
no empirical train/test sampling noise: evaluation is the exact population
objective, not a fresh finite dataset or a claim about generalization. Natural
logarithms give nats per modeled level. The true entropies are checked against
independent enumeration in `test_p1c_synthetic.py`; target/future leakage checks
also pass. The original mask difference persists with zero previous-code
embeddings, matching the P1b withdrawal rationale.

- Independent fair bits: joint entropy = 4 ln 2; TC = 0.
- Four copies of one fair bit: joint entropy = ln 2; TC = 3 ln 2.
- Stationary symmetric binary Markov chain with flip probability q:
  joint entropy = ln 2 + 3 h(q); TC = 3(ln 2 - h(q)). Both q=0.10 and q=0.25
  are calibrated, so the test does not merely distinguish zero from duplication.

The fixed acceptance threshold is absolute error <=0.05 nats in **each seed**,
specified in the script before the run. JSON includes fact/AR population NLLs,
initial/final NLLs, a midpoint/final curve, per-seed TC, mean/SD, true entropy,
parameter counts, source SHA-256 and configuration. No clipping of negative
estimates, selection of best checkpoint or tuning to the true TC is performed.
The measured numeric table appears below. Passing this small
constant-context calibration supports basic instrument behavior; it does not
validate high-dimensional conditional TC on learned hierarchical codes or
separate real-data probe approximation/optimization error from true TC.

### P1c measured results

| Case | True TC | Estimated TC, mean (SD) | Max absolute error |
| --- | ---: | ---: | ---: |
| independent | 0.000000000 | -0.000000010 (8.6e-09) | 1.49e-08 |
| duplicated | 2.079441542 | 2.079418560 (6.73e-06) | 3.06e-05 |
| markov_0.10 | 1.104192622 | 1.104192545 (1.24e-07) | 1.56e-07 |
| markov_0.25 | 0.392436108 | 0.392436027 (8.55e-08) | 1.5e-07 |

All 12 case/seed calibrations passed the pre-set 0.05-nat tolerance. Total CPU fitting wall time: 258.0 seconds. Full fact and AR NLLs and curves: [p1c_synthetic.json](p1c_synthetic.json).

All four cumulative proposed Python sources compile in memory. This is a syntax check of the review diffs, not an execution of patched production trainers.
