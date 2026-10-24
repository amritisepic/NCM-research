# Deep audit 3: W1 on-policy corruption and S1 onpolicy

Audit date: 2026-10-23. Subject: `d5013334808ab4b946874e06671a6ef3f2c79ef6`,
the fetched head of `claude/ncm-research`. The shared checkout later moved to
an analysis branch; the subject files remained byte-identical. This audit's
publication contains only this report, `codex_audit3_numbers.json`, and
`tests/test_audit3_*.py`. No production patch below is applied. Per the explicit
file restriction, this report is also the run record; `log.md` is untouched.

## Verdict

**There are reproducibility and precision defects to fix before rerunning, but
no demonstrated alignment defect or established causal explanation for the
W1 generation collapse. Do not declare H6/H8 falsified from this audit.** The
canonical-policy identity test passes bit for bit on the released Sudoku
tokenizer, including re-quantized targets. A small text model with W1's irregular
level schedule also matches forced sequential inference logits at every level.

The old autocast *state leak* is fixed. Text policy proposals and clean validation
NLL nevertheless still run under training AMP; this is distinct from that leak.
Canonical targets and the clean hierarchy are the same scoring path for every
arm. A2/A4's lower canonical NLL can coexist with bad unconditional generation:
teacher forcing measures conditional likelihood on real coarser contexts, whereas
generation visits the model's own contexts. There is no mathematical requirement
that these metrics improve together. The saved W1 checkpoints are absent, so this
audit cannot independently rescore their fp32 NLL or identify their exact weights.

The saved aggregate arithmetic agrees with the user's table. **W1b nominally
advances A4 and A5 against A3**, although both are far worse than A1; A2 misses
the 0.35 threshold. This is what the registered disjunctive comparator rule says,
not a recommendation to launch those seeds before resolving the audit.

## Findings

Priorities below are P1: each blocks the specified protocol or reproducible
continuation. None is promoted to a proven explanation of the large PPL effect.
Strict xfails catch only `AssertionError`, so missing assets or broken imports
cannot masquerade as a reproduced defect.

### A3-1 — P1: policy and canonical-NLL forwards violate the fp32 protocol

**Locations:** `scripts/train_gen.py:90`, `scripts/train_gen.py:169`,
`scripts/train_gen.py:173`, `scripts/train_gen.py:208`,
`scripts/train_gen.py:300`; related `scripts/train_sudoku_L.py:223`.

`compute_loss` is entered inside `trainer.ctx`. Only encoding and VQ reconstruction
disable autocast inside `policy_codes`; the generator forward does not. Calling
`.float()` on its output cannot recover a fp32 forward. The script also globally
enables TF32. `val_nll` explicitly enters `trainer.ctx` and does not turn off TF32.
Consequently the recorded clean NLL is an AMP-forward score on A100, including A0,
even though the final result's broad precision field says fp32 sampling/scoring.
Final generated-sample metrics *do* use the fp32, TF32-off wrapper. This is not
evidence that GPT-2 scored A3 differently from A1.

**Tests:** `test_text_policy_forward_is_fp32_with_tf32_off` and
`test_val_nll_is_fp32_scoring` observe the actual forward output dtype and backend
flags while deliberately entering AMP. Both strict-xfail. They also check that
the caller's autocast state is restored, which succeeds. `Trainer.ctx` creates
a fresh context on each access; there is no R024-style object re-entry.

**Sudoku:** the greedy teacher inherits AMP too, but **S1c explicitly registers
this behavior**. It is not an unreported deviation from S1c. Changing that teacher
to fp32 requires documenting the protocol change before an S1 rerun. Sudoku
already disables TF32 globally, has a learned level-0 query, and predicts every
level from the puzzle; replacing its level 0 with a text unigram would be wrong.

**Proposed patch (not applied):** add a precision/mode boundary shared by both
text forwards. Restore each submodule's state even after exceptions. This also
addresses the mode-restoration part of A3-2. Apply the same boundary around the
Sudoku teacher only with the corresponding S1 amendment. Training's outer AMP
region can remain; an explicitly disabled nested context is a fresh object.

```diff
--- a/src/ncmr/train.py
+++ b/src/ncmr/train.py
@@ -12,10 +12,27 @@
 import math
 import os
 import time
-from contextlib import nullcontext
+from contextlib import contextmanager, nullcontext
 
 import numpy as np
 import torch
+
+
+
+@contextmanager
+def fp32_eval(net, device):
+    modes = [(m, m.training) for m in net.modules()]
+    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
+    try:
+        net.eval()
+        torch.backends.cuda.matmul.allow_tf32 = False
+        torch.backends.cudnn.allow_tf32 = False
+        with torch.autocast(device_type=device, enabled=False):
+            yield
+    finally:
+        for module, training in modes:
+            module.training = training
+        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
 
 
 def pick_device() -> str:
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@ -38,7 +38,7 @@
 from ncmr.data import EOS, PAD, TrainStream, cut_at_eos, examples  # noqa: E402
 from ncmr.models import build_ncm_generator, n_params  # noqa: E402
 from ncmr.ncm_ref_loader import ForwardCounter, load_ncm  # noqa: E402
-from ncmr.train import Calib, Trainer, pick_device  # noqa: E402
+from ncmr.train import Calib, Trainer, pick_device, fp32_eval  # noqa: E402
 
 ap = argparse.ArgumentParser()
 ap.add_argument("--config", required=True)
@@ -155,25 +155,16 @@
 # ------------------------------------------------------------------------------------------------ steps
 @torch.no_grad()
 def policy_codes(x, gen):
-    """The generator's own samples at every level (level 0: unigram, as at inference). Pass 1 samples each level
-    given the clean coarser levels (one-step errors); pass t samples given pass t-1's codes (t-step errors:
-    t = K-1 would be full sequential sampling given the true level 0)."""
-    was = net.training
-    net.eval()
-    with torch.autocast(device_type=device, enabled=False):
-        idx = tok.vq.z_to_idx(tok._encode_pre_quant(x))
-    codes = idx
-    # inference level 0 (unigram), drawn once: pass 1 sees the clean levels, pass t > 1 sees pass t-1's hierarchy
-    # including this level 0 (codex audit F3)
-    lv0 = torch.multinomial(L0, x.shape[0] * l0, replacement=True, generator=gen).view(x.shape[0], l0)
-    for _ in range(policy_passes):
-        with torch.autocast(device_type=device, enabled=False):
+    """Pass 1 uses clean contexts; later passes use the full prior policy hierarchy."""
+    with fp32_eval(net, device):
+        codes = tok.vq.z_to_idx(tok._encode_pre_quant(x))
+        lv0 = torch.multinomial(L0, x.shape[0] * l0, replacement=True,
+                                generator=gen).view(x.shape[0], l0)
+        for _ in range(policy_passes):
             z_in = tok.vq.idx_to_ncm_input(codes)
-        # runs inside compute_loss's autocast; entering another context here leaked autocast (codex audit F2)
-        logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
-        samp = torch.distributions.Categorical(logits=logits / policy_temp).sample()
-        codes = torch.cat([lv0, samp], 1)
-    net.train(was)
+            logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
+            samp = torch.distributions.Categorical(logits=logits / policy_temp).sample()
+            codes = torch.cat([lv0, samp], 1)
     return codes
 
 
@@ -198,21 +189,18 @@
 
 @torch.no_grad()
 def val_nll():
-    """Teacher-forced NLL of the canonical codes given the clean hierarchy (nats/window, per level)."""
-    net.eval()
-    tot, per = 0.0, torch.zeros(K - 1)
-    for s in range(0, len(val_x), 100):
-        x = val_x[s:s + 100].to(device)
-        with torch.autocast(device_type=device, enabled=False):
+    """Canonical teacher-forced NLL, fp32, TF32 off, with caller modes restored."""
+    with fp32_eval(net, device):
+        tot, per = 0.0, torch.zeros(K - 1)
+        for s in range(0, len(val_x), 100):
+            x = val_x[s:s + 100].to(device)
             z_in, tgt = tok.get_ncm_inputs(x)
-        with trainer.ctx:
             logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
-        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
-        tot += nll.sum().item()
-        per += torch.stack([c.sum(1) for c in torch.split(nll, levels[1:], dim=1)], 1).sum(0).cpu()
-    net.train()
-    n = len(val_x)
-    return {"val_nll": tot / n, "val_nll_per_level": (per / n).round(decimals=2).tolist()}
+            nll = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(),
+                                  reduction="none").view(tgt.shape)
+            tot += nll.sum().item()
+            per += torch.stack([c.sum(1) for c in torch.split(nll, levels[1:], dim=1)], 1).sum(0).cpu()
+    return {"val_nll": tot / len(val_x), "val_nll_per_level": (per / len(val_x)).round(decimals=2).tolist()}
 
 
 @torch.no_grad()
```

Validate the implementation with the forward-hook tests and a CUDA replay before
spending a W1 budget. Combining this diff with A3-2 changes the categorical draw
inside the new scope; preserve its nesting.

### A3-2 — P1: policy sampling consumes dropout's RNG and incompletely restores mode

**Locations:** `scripts/train_gen.py:161`, `scripts/train_gen.py:174`,
`scripts/train_gen.py:176`; shared mode issue `scripts/train_sudoku_L.py:225`.

Level 0 uses the supplied generator, but `Categorical.sample()` uses the global
Torch generator. A policy call therefore changes the subsequent training dropout
stream, contrary to the explicit isolation requirement. `net.train(was)` restores
the root flag but overwrites any intentionally mixed child modes; exceptions
leave the network in eval mode. In the present W1 configuration all generator
children normally share one mode, so the latter is a state-contract defect, not
evidence of persistent dropout being disabled in these logs.

**Tests:** `test_policy_preserves_global_training_rng` and
`test_policy_preserves_each_module_mode` strict-xfail. The one/two-pass tests show
that normal calls restore the outer autocast state. Greedy Sudoku proposals do
not draw from the global RNG, so they do **not** share the categorical-sampling
problem. They share the root-only, non-exception-safe mode restoration.

`Trainer.state` already checkpoints the global CPU/CUDA RNG, so this finding does
not by itself prove that a same-code resume loses policy draws. Indeed the exact
CPU replay probe passes. Separate a policy RNG from corruption and dropout, and
checkpoint all three. Ambient training RNGs should be preserved; the explicitly
owned sampling RNG must advance and be checkpointed, otherwise every call would
repeat its random numbers.

**Proposed patch (not applied; combine with A3-1 and A3-3):**

```diff
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@ -163,13 +163,15 @@
         for _ in range(policy_passes):
             z_in = tok.vq.idx_to_ncm_input(codes)
             logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
-            samp = torch.distributions.Categorical(logits=logits / policy_temp).sample()
+            probabilities = torch.softmax(logits / policy_temp, dim=-1)
+            samp = torch.multinomial(probabilities.flatten(0, 1), 1,
+                                     generator=gen).view(logits.shape[:-1])
             codes = torch.cat([lv0, samp], 1)
     return codes
 
 
 def compute_loss(x, gen):
-    pol = policy_codes(x, gen) if corr.mode == "onpolicy" else None
+    pol = policy_codes(x, policy_gen) if corr.mode == "onpolicy" else None
     h = build_hierarchy(tok, x, corr, policy_codes=pol, generator=gen)
     logits = fnet(torch.cat([prefix_emb(x.shape[0]), h["inputs"]], 1))
     tgt = h["targets"][:, l0:]
@@ -265,6 +267,11 @@
 net.train()
 deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
 gen = torch.Generator(device=device).manual_seed(seed * 7919)
+policy_gen = torch.Generator(device=device).manual_seed(seed * 7919 + 104729)
+if trainer.step and corr.mode == "onpolicy":
+    if "policy_rng" not in resume_extra:
+        raise RuntimeError("checkpoint predates isolated policy RNG; start a new run")
+    policy_gen.set_state(resume_extra["policy_rng"].cpu())
 if trainer.step:
     if "corruption_rng" in resume_extra:   # codex audit F1
         gen.set_state(resume_extra["corruption_rng"].cpu())
@@ -274,7 +281,7 @@
 
 
 def save_ckpt():
-    trainer.save(modules, data_rng, extra={"corruption_rng": gen.get_state()})
+    trainer.save(modules, data_rng, extra={"corruption_rng": gen.get_state(), "policy_rng": policy_gen.get_state()})
 
 
 accum = int(tc.get("grad_accum", 1))
```

This deliberately changes draw order. Use a new manifest/protocol version and a
fresh run directory; do not silently resume an old W1 checkpoint under the change.

### A3-3 — P1: resume identity and missing-state checks still fail open

**Locations:** `scripts/train_gen.py:79`, `scripts/train_gen.py:86`,
`scripts/train_gen.py:95`, `scripts/train_gen.py:280`,
`scripts/train_gen.py:299`; `scripts/train_sudoku_L.py:90`.

Text excludes `grad_accum` from its run key, but changing the microbatch partition
changes dropout draws, mask/RNG ordering, and floating-point reductions. A direct
hierarchy test confirms different used codes for the same batch and generator
seed split into two microbatches. A checkpoint without `config.yaml` bypasses
validation entirely. A checkpoint without corruption RNG is explicitly reseeded
and continued. Text has no binding to tokenizer, base weights, data bytes or
runtime/source identity. Replacing the frozen tokenizer changes training after
the model checkpoint is restored, even if all dimensions match.

Sudoku correctly refuses missing manifests and corruption RNG, and includes its
full config and data hash. Its base checkpoint identity is only **file sizes**,
however: same-size tokenizer/weight replacements pass. The CPU test mutates equal
length bytes and demonstrates identical manifests. This is not a claim that any
asset was actually replaced during W1/S1.

**Tests:** strict xfails `test_resume_key_rejects_changed_microbatch_partition`,
`test_resume_without_config_fails_closed`,
`test_resume_without_corruption_rng_fails_closed`, and
`test_sudoku_manifest_rejects_same_size_base_change`.
`test_microbatch_partition_changes_masks` passes as the behavioral witness.
`test_identical_config_cpu_resume_is_exact` passes with model, AdamW, global
Torch RNG, NumPy data RNG, and explicit corruption RNG restored atomically.

**Proposed patch (not applied):** reject incompatible state before resuming;
bind immutable content identities in both the manifest and atomic checkpoint.
Runtime equality does not guarantee deterministic CUDA kernels: additionally
enforce deterministic algorithms and a compatible deterministic CUDA setup,
then demonstrate replay on the launch GPU. Record unsupported-kernel failures
instead of claiming bitwise replay.

```diff
--- a/scripts/train_gen.py
+++ b/scripts/train_gen.py
@@
-    t = {k: v for k, v in c.get("train", {}).items() if k not in ("grad_accum", "eval_every", "ckpt_every", "log_every")}
+    t = {k: v for k, v in c.get("train", {}).items() if k not in ("eval_every", "ckpt_every", "log_every")}
@@
-if os.path.exists(ckpt_path) and os.path.exists(cfg_path) and not a.calib_steps:
+if os.path.exists(ckpt_path) and not a.calib_steps:
+    if not os.path.exists(cfg_path):
+        raise RuntimeError("checkpoint lacks run identity")
     if _run_key(OmegaConf.load(cfg_path)) != _run_key(cfg):
@@
-    else:   # checkpoints written before that fix
-        gen.manual_seed(seed * 7919 + trainer.step)
-        print("[resume] checkpoint has no corruption RNG state: reseeded (not bit-identical to an uninterrupted run)")
+    else:
+        raise RuntimeError("checkpoint lacks corruption RNG; exact resume unavailable")
--- a/scripts/train_sudoku_L.py
+++ b/scripts/train_sudoku_L.py
@@
-            "base_bytes": {n: os.path.getsize(os.path.join(_base, n)) for n in ("tokenizer.pt", "generator.pt")},
+            "base_sha256": {n: digest(os.path.join(_base, n)) for n in ("tokenizer.pt", "generator.pt")},
```

The above minimal diff closes demonstrated guards. Complete the text manifest
by hashing base config, tokenizer, generator, BPE assets, train token stream,
offsets and validation IDs; record resolved config including microbatch size,
source hash, Torch/CUDA/cuDNN versions, attention backend and deterministic flags.
Persist that identity inside `extra`, compare it before loading optimizer/model,
and reject old or mismatched identities. These additions are required, not a
claim that the minimal guard diff alone provides full provenance.

### A3-4 — P1: notebook result bundles cannot identify code or checkpoint lineage

**Locations:** `notebooks/make_notebooks.py:183`,
`notebooks/make_notebooks.py:185`, `notebooks/make_notebooks.py:225`,
`notebooks/make_notebooks.py:253`, `notebooks/make_notebooks.py:261`;
`scripts/train_gen.py:352`.

Setup resets the checkout to the moving branch, while collection records only
run name, GPU and an unzoned collection time. It does not preserve the executed
source revision, dirty state, command start/end, attempt ID, restored checkpoint
hash/step or final model hash. `finished - wall_s` cannot recover these: wall time
accumulates only the surviving checkpoint lineage, attempts can be lost, and
collection happens after all commands. The control notebook runs A0 *before* A1,
so subtracting A0's duration from the notebook finish does not date A0's launch.

**Test:** `test_notebook_collection_carries_source_identity` executes the actual
metadata-dump statement with I/O intercepted and strict-xfails on absent code
identity. Metadata is not retroactively recoverable from this test.

**Proposed patch (not applied; illustrative notebook changes):**

```diff
--- a/notebooks/make_notebooks.py
+++ b/notebooks/make_notebooks.py
@@
 RUN = r'''
+from datetime import datetime, timezone
+CODE_SHA = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
+if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
+    raise RuntimeError('tracked source is dirty; record a reviewed immutable revision first')
+ATTEMPTS = []
 CMDS_RAW = list(CMDS)
@@
         t = time.time()
+        started_utc = datetime.now(timezone.utc).isoformat()
         p = subprocess.run(c, shell=True)
+        ATTEMPTS.append({'git_sha': CODE_SHA, 'command_index': i, 'seed': seed,
+                         'started_utc': started_utc,
+                         'finished_utc': datetime.now(timezone.utc).isoformat(),
+                         'exit_code': p.returncode})
@@
-json.dump({'run': '__NAME__', 'gpu': gpu, 'finished': time.ctime()}, open(f'{out}/_meta.json', 'w'))
+json.dump({'run': '__NAME__', 'gpu': gpu, 'git_sha': CODE_SHA,
+           'finished_utc': datetime.now(timezone.utc).isoformat(),
+           'attempts': ATTEMPTS}, open(f'{out}/_meta.json', 'w'), indent=2)
```

Persist attempts **at command start and completion**, not just in notebook RAM,
so interruptions retain lineage; use command indices plus sanitized resolved
arguments rather than publishing Drive/local paths. Include the A3-3 manifests,
environment lock/package versions, source/config hashes, model/checkpoint digests,
resume step, and cumulative versus per-attempt time. Verify/pin the revision
before each process, rather than allowing a setup-cell rerun to change it silently.
Record actual GPU UUID/type and numeric backend flags without account identifiers.
Never publish credentials or local/session paths. The sketch above alone is not
an interruption-safe attempt ledger.

## Checks without an alignment finding

- `build_hierarchy` appends the cumulative reconstruction before adding level k;
  its input slots therefore predict k from levels <k. Targets drop only text's
  level 0. Prefix is the same EOS embedding as text generation. The actual W1
  config uses temperature 1, no top-k/top-p truncation, no CFG, `compile: false`,
  and `fnet is net`. A compiled-mode failure cannot explain these recorded runs.
- Pass 1 receives canonical coarser codes, including canonical level 0. A single
  independent unigram level-0 draw is prepended to the sampled hierarchy. Pass 2
  consumes the entire previous policy hierarchy, including that sampled level 0.
  Only afterward does the corruption mask mix proposals and clean/re-quantized
  codes. This implements W1b H8; it is not sequential rollout training.
- Positive-rate canonical-policy identity passes bit for bit, for both canonical
  and re-quantized labels. Arbitrary used codes produce the same conditioning
  as upstream `idx_to_ncm_input`. A separate forced-sequential text test confirms
  output-level offsets/logits, not merely equality between two helper calls.
- A2/A4 set `d_lab = nearest(z-z_used)`, use its argmin for targets and initialize
  `used = c_lab`; only masked slots take replacements. Thus unmasked finer inputs
  are re-quantized as W1b requires. A direct perturbed-level-0 test confirms changed
  finer targets and used-code/input consistency. A3/A5 keep canonical targets.
- Sudoku's prompt, learned level-0 query and all-level target positions pass an
  actual `tf_loss` identity test. Its ordinary greedy policy draws no RNG.
- Same-seed data order is controlled by the independent NumPy stream, with one
  full batch drawn per optimizer step. Evaluation does not draw from that stream;
  text sample evaluation forks/restores global Torch RNG and TF32 flags. No test
  data are used by the text validation path; `examples('val')` selects even-index
  validation items. This audit adds no test-set tuning.
- Generator FLOP totals are the registered **proxy**, not measured hardware FLOPs:
  `(3 + policy_passes) * analytic_forward * batch * steps`. Every trained W1 arm
  records 157.0647501176832 PFLOP by this definition. There is no early-stopped
  teacher hidden inside the one/two fixed policy passes. Runtime and discarded
  attempts are not equalized by that proxy.

## CPU diagnostic and runtime checks

The opt-in experiment is in `tests/test_audit3_cpu.py`. It uses the hash-verified
released Sudoku-hard generator/tokenizer from `tests/conftest.py`, generated
training puzzles, and the first 32 generated held-out puzzles. It never trains
on the released evaluation puzzles. The preset is seed 0, 200 paired steps,
batch 2, one CPU thread, fp32, original S1 optimizer/LR/warmup and per-arm schedule,
uniform versus the actual extracted greedy `tf_loss`. Dropout seeds and augmented
training batches match per step. A CPU-time cap stops both arms at the same step.
This is a small-batch diagnostic at equal steps, not an equal-FLOP S1 replication.

CPU_RESULT_PENDING

**Important negative control:** the initially available Torch 2.6 runtime showed
the known no-grad/autocast cache issue: on a released Sudoku forward/backward,
uniform gave gradients to all 25,457,600 trainable parameters, while on-policy
gave gradients to only 24,832 (25,432,768 missing). W1 records **2.11.0+cu130**.
The [Torch 2.11 cached-cast implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/autocast_mode.cpp#L112-L138)
preserves autograd when caching inside no-grad. Accordingly the audit uses
2.11.0+cpu, and its actual `compute_loss` test gives gradients to every trainable
parameter. **The 2.6 failure is not evidence for the W1 collapse.** Avoid drawing
that attractive but unsupported causal conclusion from an older CPU environment.

## W1 arithmetic and registered screen

All group means/SDs and MAUVE means/SDs below were recomputed from stored arrays;
SD uses n-1. NLL and entropy are saved readbacks, not independently rescored
checkpoints. Sixteen groups contain 256 samples each. Array group order is sorted
seed order **0..14,42**, not generation order **42,0..14**. MAUVE pools 4096 samples
against 4096 validation windows and averages five quantization seeds.

| Arm | Steps | GPT-2 L mean (SD) | MAUVE mean (SD) | Clean NLL | Entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 0 | 16.164383 (0.448430) | 0.827189 (0.005595) | 861.838523 | 3.970222 |
| A1 | 1800 | 16.604238 (0.311428) | 0.881099 (0.007218) | 843.992656 | 3.962499 |
| A2 | 1800 | 16.289748 (0.368720) | 0.914693 (0.007488) | 749.253773 | 3.962645 |
| A3 | 1350 | 41.436881 (1.718856) | 0.495774 (0.021550) | 885.667102 | 4.026335 |
| A4 | 1350 | 36.805781 (1.020728) | 0.673064 (0.015129) | 707.404344 | 4.037413 |
| A5 | 1080 | 32.168815 (0.711781) | 0.543029 (0.031405) | 913.889391 | 3.989512 |
| A6 | 1800 | 17.455435 (0.408024) | 0.832429 (0.010188) | 888.888031 | 3.958134 |

Gain is comparator minus arm PPL; positive is better. Entropy and MAUVE gaps are
arm minus comparator. Every condition must hold for the same comparator.

| Arm / comparator | PPL gain >=0.35 | Entropy gap, abs <=0.05 | MAUVE gap >=-0.01 | All pass |
| --- | ---: | ---: | ---: | --- |
| A2 / A1 | 0.314491 | 0.000146 | 0.033594 | No |
| A3 / A1 | -24.832643 | 0.063836 | -0.385326 | No |
| A4 / A1 | -20.201543 | 0.074913 | -0.208036 | No |
| A4 / A3 | 4.631100 | 0.011077 | 0.177290 | Yes |
| A5 / A1 | -15.564576 | 0.027013 | -0.338071 | No |
| A5 / A3 | 9.268067 | -0.036823 | 0.047255 | Yes |
| A6 / A1 | -0.851197 | -0.004365 | -0.048670 | No |

A1 minus A0 is +0.439855 PPL, below the +1.0 recipe-damage gate. A1 always
advances under the registration; A1m and A6s are not triggered. No three-seed
claim, training-seed SD, entropy-matched claim temperature or matched-damage
control can be inferred from these single-seed files.

The same 256-sample seed-10000 A1 curve is 16.486431 at 1080, 16.705433 at
1350, and 16.299354 at 1800. Fewer-step gains are **-0.187077 and -0.406079**;
neither reaches +0.5. Corresponding final-arm curve PPLs are A5 30.158330,
A3 37.150225 and A4 36.388172, giving gains against step-matched A1 of
-13.671899, -20.444792 and -19.682739. Do not compare these single-group curve
values directly with the 16-group endpoint means to trigger the confound.

The W1b sampling-only approximation also recomputes: SD assumption 0.59,
difference SE 0.2085965, probabilities of passing the PPL threshold 4.6685%,
76.3958%, 95.3315% for true gains 0, 0.5, 0.7. These exclude training-seed
variance, entropy/MAUVE gates and multiple-comparator selection.

The JSON retains both scorers' 16 means, sample SDs, all comparator differences,
sorted-seed difference arrays, entropy/MAUVE gaps, five MAUVE values, every eval
curve occurrence, duplicate training payloads and source hashes. No discrepancy
was found in the stored aggregate means/SDs. Entropy cannot be independently
recomputed from saved raw IDs without the original BPE decode/re-encode assets;
MAUVE features and per-example scorer losses are also not present.

## Resume evidence

Repeated train rows are different from a normal train row followed by an eval
row at the same step. Line numbers below refer to each `__metrics.jsonl`.

| Arm | Evidence | Inference |
| --- | --- | --- |
| A1 | line 12: 250 -> 1; train steps 1..250 repeated | Restart from initialization; not proof of checkpoint resume |
| A3 | line 3: 25 -> 1; line 24: 475 -> 275 | Early restart; later rollback consistent with checkpoint 250 |
| A4 | repeated train 275; line 33: 700 -> 525; line 54: 975 -> 775 | At least two rollbacks consistent with checkpoints 500/750; repeated 275 also consistent with restarting from 250 |
| A2, A5, A6 | no repeated train or eval steps | No visible replay; does not exclude restart immediately after a checkpoint |

A3 step-300 NLL is 912.502546875 before rollback and 913.666515625 after,
a difference of **1.16396875**. Its repeated step-275 loss is 2.1809 versus
2.1806 and per-level corruption fractions differ. All 12 duplicated semantic
step/type payloads differ (including the initialization restart). A4's 20
duplicates differ; step-600 NLL is 731.4809140625 versus 731.530453125,
and step-900 is 716.5040078125 versus 716.27778125. A1 has 11 duplicated
train steps, 10 with differing semantic payloads. Logs retain only rounded
metrics and no checkpoint hashes, so they demonstrate **non-identical traces**,
not the exact cause or full tensor trajectory.

A3 therefore very likely resumed; **it did not replay the logged evaluation
exactly**. The source saves global Torch CPU/CUDA state, data RNG, optimizer,
step, scaler and corruption RNG in one atomic replacement. CPU replay passes.
CUDA determinism, code/runtime/asset changes, actual restored checkpoint contents
and changed microbatch settings cannot be separated retrospectively. Do not
assign the replay difference specifically to a missing RNG without that evidence.
The first poor A3 evaluation already precedes its visible checkpoint rollback,
so that rollback cannot explain the onset by itself.

## Code provenance: inference limits

All runs contain R024-era fields: 16 groups, `eval_precision`, and the new
per-level corruption logs (training arms). This supports **post-R024 code or an
equivalent local patch**, not one unique SHA. `b397656` is R024 (16:25:56 UTC).
`a049712` (17:14:35 UTC) adds a train-derived level-0 branch that these released
configs never enter. `0f9f10f` (17:47:30 UTC) changes Sudoku/WMT trainers but not
W1's relevant path. Later review-gate commits likewise leave that W1 path alone.

Assuming the unzoned collection clock was UTC and immediate, the following
arithmetic is useful for narrowing guesses, **not a recovered launch ledger**:

| Arm | Collected 2026-10-23 | Surviving wall s | Finish minus own wall | Latest prior commit by timestamp (heuristic) |
| --- | --- | ---: | --- | --- |
| A0 | 18:48:12 | 204.439 | 18:44:47.561 | Invalid A0 launch estimate: A0 precedes A1 |
| A1 | 18:48:12 | 4069.163 | 17:40:22.837 | a049712 |
| A4 | 19:11:14 | 4379.024 | 17:58:14.976 | 0f9f10f |
| A3 | 20:24:59 | 4241.806 | 19:14:17.194 | 0f9f10f |
| A5 | 21:49:53 | 4240.058 | 20:39:12.942 | 0f9f10f |
| A2 | 22:53:42 | 4121.779 | 21:45:00.221 | 5691e88 |
| A6 | 23:25:30 | 4131.754 | 22:16:38.246 | 5691e88 |

Timezone, setup/collection delay, notebook command order and discarded work make
these guesses non-unique. Commit timestamps do not establish when a revision
became reachable from the notebook branch. A1/A3/A4 explicitly lost attempts.
Even A0's simple no-retry command-order estimate would be earlier than A1, not
18:44. **Exact commit for each run: unknown.** The JSON marks SHA as null and
lists time-compatible candidates under the UTC assumption; no SHA is fabricated.
The runtime listed in every final result is Torch 2.11.0+cu130, with grad_accum=4
and compilation disabled. It does not establish the runtime of discarded attempts.

## R014 and R024 numerical audit

`results/b1_repair_capacity.json` re-derives R014's 256 windows and 665.550474 s
(11.0925 min). Single coarse-level uniform replacement keeps 99.182–99.973%
of tokens. Replacing all but the finest keeps 42.560% (uniform) or 97.992%
(one-step policy). The recipes keep 92.935 -> 96.899% (uniform) and
99.396 -> 99.844% (on-policy) with re-quantization. These support the rounded
survival claims, but they are **decoder ceilings with the canonical finest level
retained**, not full on-policy inference or a prediction of fine-tuning behavior.

Disagreements/corrections:

1. R014's sampled medians **1055/1118/454 are L0/L1/L2**, not L1/L2/L3.
   Actual L1/L2/L3 medians are **1118/454/37**.
2. The logged 58% changed codes comes from `corrupted_frac` saved only during
   `retarget=True`; it compares used codes with the *re-quantized labels*, not
   canonical codes. Weighting stored sampled disagreements by level size gives
   approximately **41.6419% of all code slots**, or **52.0320% of the replaced
   coarse slots**, differing from canonical when all-but-finest is replaced.
   Inputs were rounded to four decimals, so these weighted numbers are approximate.
   The analogous logged recipe rates 31%/21% use that same re-quantized denominator.
3. Recipe label changes are **57.533% uniform / 45.475% on-policy**; “46–58%”
   rounds the lower end upward. L8–L10 sampled top-16 shares are
   **88.29/80.75/80.49%**, rather than a strict 81–88% range.

The JSON includes every R014 pattern's percentages and complete disagreement/rank
arrays, so all rounded level claims can be checked without rerunning the model.
R024's 34-test count, 20-puzzle test equality, soft-gap discrepancy 0.05 to <=3e-6,
~1% omitted hierarchy cost, and ~0.5 CPU-hour count have no raw measurement ledger
in `results/**.json`. They are explicitly marked **not independently recomputable**,
not silently accepted or called numerical disagreements. The prior
`codex_audit_numbers.json` audits R018–R021, not those R024 execution totals.
Current source/tests verify the autocast-state and two-pass-level-0 fixes; current
tests cannot prove the historical timing, execution count or launch environment.

## Verification and reproducibility

VERIFICATION_PENDING

Use the existing dependency environment with Torch 2.11 CPU and the hash-pinned
audit assets. The optional local pytest installation can be exposed through
`PYTHONPATH=tests/_deps:tests:src` (use the platform's path separator). No automatic
asset download is performed by the tests.

```sh
python -B -m pytest -p no:cacheprovider tests/test_audit3_protocol.py tests/test_audit3_numbers.py -q -rx
python -B tests/test_audit3_numbers.py --write
python -B tests/test_audit3_cpu.py --run
```

The number writer preserves the CPU experiment section. The CPU driver writes
only the permitted audit JSON. Default pytest does not start training. The tests
read current working-tree functions using AST extraction, so fixed code can turn
strict xfails into XPASS; remove the corresponding marker only after reviewing
the fix. No test is pinned to an old defective implementation by default.

| ID | Priority | File | One-line issue |
| --- | --- | --- | --- |
| A3-1 | P1 | `scripts/train_gen.py:173` | Text policy and clean-NLL forwards inherit training AMP/TF32 |
| A3-2 | P1 | `scripts/train_gen.py:174` | Policy consumes global dropout RNG; mode restoration is incomplete |
| A3-3 | P1 | `scripts/train_gen.py:82` | Resume permits changed partition/missing state; base identity is incomplete |
| A3-4 | P1 | `notebooks/make_notebooks.py:261` | Result bundles omit source and checkpoint attempt lineage |
