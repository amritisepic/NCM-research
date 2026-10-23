"""Tier L Sudoku: fine-tune the released Sudoku-hard generator (25.5M; tokenizer frozen). Resumable; CUDA or CPU.

Arms (config `arm`; pre-registered as S1 in docs/preregistration.md):
  ctrl     plain teacher forcing on clean codes (the released config records no corruption)
  uniform  NCM's conditioning corruption (uniform codes, clean labels; the TinyStories recipe)
  onpolicy conditioning codes replaced by the model's own one-step greedy predictions, clean labels
           (one extra no-grad forward per step -> 3/4 of the steps at equal FLOPs)
  repair   ctrl on half the batch + board-feedback repair passes on the other half: a pool of in-progress
           Jacobi decodes; each step trains one parallel pass from every pool state (input: the puzzle plus an
           embedding of the board decoded from the previous pass, and the previous pass's codes; label: the
           canonical codes of the solution) and advances the state with that pass's argmax. Solved or old
           states are replaced by fresh puzzles. Same forward/backward cost per step as ctrl.
Data: our generated hard puzzles (assets/sudoku_train/hard.npz, augmented); val = their held-out tail;
the released test set is used only by the final evaluation.
Decoding at evaluation (every arm): D1 = sequential greedy (12 NFE, the paper's protocol); D2 = greedy
Jacobi passes, stop at the first board that satisfies the rules and the clues ("check"), or at a fixed point,
or at the cap (board feedback only for the repair arm); D2-nocheck = the same without the verifier.

  python scripts/train_sudoku_L.py --config configs/scale/L_sdk_ctrl.yaml --run_dir runs/L_sdk_ctrl_s0 --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr import flops as FL  # noqa: E402
from ncmr.corruption import CorruptionCfg, build_hierarchy  # noqa: E402
from ncmr.jacobi_train import PrefixFeedback, jacobi_decode, parallel_logits  # noqa: E402
from ncmr.models import n_params  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm, safe_pickle_load  # noqa: E402
from ncmr.sudoku import cells, is_solution, propagation_rounds  # noqa: E402
from ncmr.sudoku_data import SudokuTrain  # noqa: E402
from ncmr.train import Calib, Trainer, pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--run_dir", required=True)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--data", default=os.path.join(ROOT, "assets", "sudoku_train", "hard.npz"))
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--max_minutes", type=float, default=0)
ap.add_argument("--calib_steps", type=int, default=0)
ap.add_argument("--eval_only", action="store_true", help="no training: final eval of the released model (A0)")
ap.add_argument("--n_test", type=int, default=2000)
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
cfg = OmegaConf.load(a.config)
tc, ec, rc = cfg.train, cfg.get("eval", {}), cfg.get("repair", {})
seed = int(a.seed if a.seed is not None else tc.get("seed", 0))
cfg.train.seed = seed
arm = str(cfg.arm)
if a.calib_steps:
    tc.steps, tc.eval_every, tc.ckpt_every = a.calib_steps, 10 ** 9, 10 ** 9
if a.eval_only:
    tc.steps = 0
dev = pick_device()
if dev == "cuda":   # explicit precision policy: fp32 evaluation without TF32, as wave 0 decoded (codex audit2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
rng = np.random.default_rng(seed)
pool_rng = np.random.default_rng(np.random.SeedSequence([seed, 2201]))   # repair-pool refills only (codex audit2 A2-6)
data = SudokuTrain("hard", a.data)
os.makedirs(a.run_dir, exist_ok=True)


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# run identity, validated before anything in run_dir is overwritten (codex audit2 A2-3)
ckpt_path, manifest_path = os.path.join(a.run_dir, "ckpt.pt"), os.path.join(a.run_dir, "run_manifest.json")
if a.eval_only and os.path.exists(ckpt_path):
    sys.exit(f"--eval_only evaluates the released model, but {a.run_dir} holds a training checkpoint")
_base = os.path.join(ROOT, "artifacts", "sudoku", "hard")
manifest = {"config": OmegaConf.to_container(cfg, resolve=True), "data_sha256": digest(a.data),
            "base_config_sha256": digest(os.path.join(_base, "config.yaml")),
            "base_bytes": {n: os.path.getsize(os.path.join(_base, n)) for n in ("tokenizer.pt", "generator.pt")},
            "device": pick_device()}
if os.path.exists(ckpt_path) and not a.calib_steps:
    if not os.path.exists(manifest_path) or json.load(open(manifest_path)) != manifest:
        sys.exit(f"{a.run_dir} holds a checkpoint of a different (or unrecorded) run: use a fresh run directory")
if not a.calib_steps:
    json.dump(manifest, open(manifest_path, "w"), indent=1)
OmegaConf.save(cfg, os.path.join(a.run_dir, "config.yaml"))

B_ = load_ncm(os.path.join(ROOT, "artifacts", "sudoku", "hard"), device=dev)
net, tok = B_.net, B_.vqvae.eval()
for p in tok.parameters():
    p.requires_grad_(False)
vq = tok.vq
levels = [int(l) for l in vq.levels]
K, S = len(levels), sum(levels)
P = int(net.prefix_len)
if "dropout" in cfg.get("model", {}):
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = float(cfg.model.dropout)
fb = PrefixFeedback(12, tok.embed_dim, offset=1, length=89).to(dev) if arm == "repair" else None
cc = cfg.get("corruption", {})
corr = CorruptionCfg(mode={"uniform": "uniform", "onpolicy": "onpolicy"}.get(arm, "none"),
                     mean_rate=float(cc.get("mean_rate", 0.5)), spread=float(cc.get("spread", 0.5)))
fwd = FL.ncm_gen_fwd_per_seq(int(B_.cfg.model.n_embd), int(B_.cfg.model.n_layer), levels, prefix=P,
                             codebook=16, vq_dim=tok.embed_dim, query_level0=True)
fl_step = (3 + (1 if arm == "onpolicy" else 0)) * fwd * int(tc.batch_size)
params = [p for p in net.parameters() if p.requires_grad]
extra = [(list(fb.parameters()), float(rc.get("new_lr_mult", 20.0)))] if fb is not None else ()
trainer = Trainer(a.run_dir, params, max_lr=float(tc.lr), total_steps=int(tc.steps), warmup=int(tc.warmup),
                  weight_decay=float(tc.weight_decay), betas=(0.9, float(tc.get("beta2", 0.99))),
                  ckpt_every=int(tc.ckpt_every), device=dev, amp=bool(tc.get("amp", True)),
                  min_lr_frac=float(tc.get("min_lr_frac", 0.1)), extra_groups=extra)
modules = {"model": net, **({"feedback": fb} if fb is not None else {})}
resume_extra = trainer.maybe_resume(modules, rng)
gen = torch.Generator(device=dev).manual_seed(seed * 7919)
if trainer.step and corr.mode != "none":   # codex audit2 A2-2: continue the corruption stream, do not reseed
    if "corruption_rng" not in resume_extra:
        raise RuntimeError("checkpoint lacks the corruption RNG state; exact resume unavailable")
    gen.set_state(resume_extra["corruption_rng"].cpu())
print(f"[sudoku-L] arm={arm} params {n_params(net) / 1e6:.2f}M (+{n_params(fb) if fb else 0} feedback) | "
      f"{fl_step / 1e12:.2f} TFLOP/step, {fl_step * int(tc.steps) / 1e15:.1f} PFLOP total | device {dev}", flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def wte(prompt):
    return tok.encoder.wte(prompt)


def decode_ids(codes):
    with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
        return tok.decoder(vq.idx_to_z_q(codes)).argmax(-1)


def checks(ids, prompts):
    """(n,) bool: decoded board satisfies the rules and the clues (no access to the solution)."""
    out = []
    for r, p in zip(ids.tolist(), prompts.tolist()):
        b = cells(r)
        out.append(b is not None and is_solution(b, cells(p)))
    return torch.tensor(out, device=ids.device)


def canon(sol):
    with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
        return tok.encode(sol)


def batch(n, data_rng=None):
    p, s = data.batch(rng if data_rng is None else data_rng, n)
    return torch.from_numpy(p).to(dev), torch.from_numpy(s).to(dev)


class Pool:
    """In-progress Jacobi decodes of training puzzles (repair arm)."""

    def __init__(self, n, state=None):
        if state is not None:   # restored from the checkpoint: draws no data (codex audit2 A2-1)
            self.load_state_dict(state)
            return
        self.prompt, self.sol = batch(n, data_rng=pool_rng)
        self.codes = torch.zeros(n, S, dtype=torch.long, device=dev)
        self.guess = torch.zeros(n, 89, dtype=torch.long, device=dev)
        self.active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.age = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, m):
        k = int(m.sum())
        if k == 0:
            return
        p, s = batch(k, data_rng=pool_rng)
        self.prompt[m], self.sol[m] = p, s
        self.codes[m], self.guess[m], self.active[m], self.age[m] = 0, 0, False, 0

    def state_dict(self):
        return {k: getattr(self, k).cpu() for k in ("prompt", "sol", "codes", "guess", "active", "age")}

    def load_state_dict(self, st):
        for k, v in st.items():
            setattr(self, k, v.to(dev))


pool = None
if arm == "repair":
    if trainer.step:
        if "pool" not in resume_extra or "pool_data_rng" not in resume_extra:
            raise RuntimeError("checkpoint lacks the repair pool / pool RNG; exact resume unavailable")
        pool_rng.bit_generator.state = resume_extra["pool_data_rng"]
    pool = Pool(int(tc.batch_size) // 2, state=resume_extra.get("pool") if trainer.step else None)


def save_checkpoint():
    """Model, optimizer, data RNG, corruption RNG and repair pool in one atomic file (codex audit2 A2-1/A2-2)."""
    extra = {"corruption_rng": gen.get_state()}
    if pool is not None:
        extra.update(pool=pool.state_dict(), pool_data_rng=pool_rng.bit_generator.state)
    trainer.save(modules, rng, extra=extra)


pool_stats = {"solved": 0, "aged": 0, "advanced": 0}


def tf_loss(prompt, sol):
    """Teacher forcing (ctrl / uniform / onpolicy hierarchies), no board feedback."""
    z_prefix = wte(prompt)
    if corr.mode == "none":
        idx = canon(sol)
        logits = parallel_logits(net, z_prefix, idx)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), idx.reshape(-1))
    pol = None
    if corr.mode == "onpolicy":
        # policy proposals inherit the training autocast (bf16 on A100/L4); inference decodes in fp32 (S1 records this)
        was = net.training
        net.eval()
        with torch.no_grad():
            pol = parallel_logits(net, z_prefix, canon(sol)).float().argmax(-1)   # greedy, as at inference
        net.train(was)
    h = build_hierarchy(tok, sol, corr, policy_codes=pol, generator=gen)
    q0 = [net.build_l0_query_input(sol.shape[0], h["inputs"].dtype)]
    logits = net(torch.cat([z_prefix, *q0, h["inputs"]], 1))
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), h["targets"].reshape(-1))


def pool_loss():
    z_prefix = fb(wte(pool.prompt), pool.guess, pool.active)
    logits = parallel_logits(net, z_prefix, pool.codes)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), canon(pool.sol).reshape(-1))
    with torch.no_grad():
        new = logits.float().argmax(-1)
        ids = decode_ids(new)
        ok = checks(ids, pool.prompt)
        pool.codes, pool.guess, pool.active = new, ids, torch.ones_like(pool.active)
        pool.age += 1
        old = pool.age >= int(rc.get("max_age", 16))
        pool_stats["solved"] += int(ok.sum())
        pool_stats["aged"] += int((old & ~ok).sum())
        pool_stats["advanced"] += len(ok)
        pool.reset(ok | old)
    return loss


# ------------------------------------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(prompts, sols, caps=(4, 8, 12, 16, 24), bs=250, final=False):
    net.eval()
    res = {"n": len(prompts)}
    # S1b: for the repair arm, also board feedback zeroed (fb0) and taken from another puzzle's board (fbshuf)
    d2 = {m: [] for m in ("check", "nocheck") + (("check_fb0", "check_fbshuf") if fb is not None else ())}
    seq_ok = []
    for s in range(0, len(prompts), bs):
        p = torch.as_tensor(prompts[s:s + bs], device=dev)
        y = [cells(r) for r in np.asarray(sols[s:s + bs]).tolist()]
        g = net.generate(None, p, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=None)
        seq_ok += [cells(r) == t for r, t in zip(decode_ids(g).tolist(), y)]
        for mode in d2:
            def prefix_fn(codes, has_prev):
                z = wte(p)
                if fb is None or not has_prev or mode == "check_fb0":
                    return z
                guess = decode_ids(codes)
                return fb(z, guess.roll(1, 0) if mode == "check_fbshuf" else guess)

            out = jacobi_decode(net, prefix_fn, len(p), S, dev, max(caps), keep_history=True,
                                check_fn=(lambda c: checks(decode_ids(c), p)) if mode.startswith("check") else None)
            boards = [[cells(r) for r in decode_ids(h).tolist()] for h in out["history"]]
            nfe = out["nfe"].tolist()
            for j in range(len(p)):
                # at cap c a decode that has not stopped outputs its pass-c board
                d2[mode].append({"nfe": nfe[j], "ok_at": [boards[min(nfe[j], c) - 1][j] == y[j] for c in caps],
                                 "stop": int(out["stop"][j])})
    res["seq_solved_pct"] = 100 * float(np.mean(seq_ok))
    for mode, rows in d2.items():
        # S1b tails (codex audit2 A2-9): cap hit = not stopped by the verifier or a fixed point at or before c
        res[f"d2_{mode}"] = {f"cap{c}": {"solved_pct": 100 * float(np.mean([r["ok_at"][i] for r in rows])),
                                        "mean_nfe": float(np.mean([min(r["nfe"], c) for r in rows])),
                                        "p90_nfe": float(np.percentile([min(r["nfe"], c) for r in rows], 90)),
                                        "max_nfe": int(max(min(r["nfe"], c) for r in rows)),
                                        "cap_hit_count": int(sum(r["nfe"] > c or (r["nfe"] == c and r["stop"] == 0) for r in rows))}
                             for i, c in enumerate(caps)}
        res[f"d2_{mode}"]["stop_counts"] = np.bincount([r["stop"] for r in rows], minlength=3).tolist()
    if final:
        rounds = [propagation_rounds(cells(list(p))) for p in prompts]
        bins = [("le6", lambda r, ok: ok and r <= 6), ("7-8", lambda r, ok: ok and 7 <= r <= 8),
                ("9-10", lambda r, ok: ok and 9 <= r <= 10), ("ge11", lambda r, ok: ok and r >= 11),
                ("search", lambda r, ok: not ok)]
        res["by_depth"] = {}
        for name, sel in bins:
            m = [i for i, (r, ok) in enumerate(rounds) if sel(r, ok)]
            res["by_depth"][name] = {"n": len(m), "seq_pct": 100 * float(np.mean([seq_ok[i] for i in m])) if m else None,
                                     "d2_check_capmax_pct": 100 * float(np.mean([d2["check"][i]["ok_at"][-1] for i in m])) if m else None}
        res["per_puzzle"] = {"seq": [bool(v) for v in seq_ok], "caps": list(caps),
                             **{f"d2_{m}": {"nfe": [r["nfe"] for r in rows], "stop": [r["stop"] for r in rows],
                                            "ok_at_cap": [[bool(v) for v in r["ok_at"]] for r in rows]}
                                for m, rows in d2.items()}}
    net.train()
    return res


# ------------------------------------------------------------------------------------------------ loop
val_p, val_s = data.val(int(ec.get("n_val", 1000)))
net.train()
deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
calib = Calib(a.run_dir, a.calib_steps, fl_step) if a.calib_steps else None
bs = int(tc.batch_size)
while trainer.step < int(tc.steps):
    with trainer.ctx:
        if arm == "repair":
            prompt, sol = batch(bs)   # same rows per step as every other arm; the repair arm uses the first half (A2-6)
            loss_tf = tf_loss(prompt[:bs - bs // 2], sol[:bs - bs // 2])
            loss_pool = pool_loss()
            loss = 0.5 * (loss_tf + loss_pool)
            logs = {"loss_tf": round(loss_tf.item(), 4), "loss_pool": round(loss_pool.item(), 4)}
        else:
            prompt, sol = batch(bs)
            loss = tf_loss(prompt, sol)
            logs = {}
    trainer.backward(loss)
    gn = trainer.step_opt(params + (list(fb.parameters()) if fb is not None else []))
    if calib:
        calib.tick(trainer.step)
        continue
    if trainer.step % int(tc.get("log_every", 50)) == 0 or trainer.step == 1:
        if pool is not None and pool_stats["advanced"]:
            logs.update({"pool_solved_frac": round(pool_stats["solved"] / pool_stats["advanced"], 4),
                         "pool_aged_frac": round(pool_stats["aged"] / pool_stats["advanced"], 4)})
            pool_stats.update(solved=0, aged=0, advanced=0)
        print(json.dumps(trainer.log({"loss": round(loss.item(), 4), "gnorm": round(gn, 3), **logs})), flush=True)
    if trainer.step % int(tc.eval_every) == 0 or trainer.step == int(tc.steps):
        ev = evaluate(val_p, val_s, caps=(12, 24))
        print(json.dumps(trainer.log({"eval": ev})), flush=True)
    if trainer.step % int(tc.ckpt_every) == 0:
        save_checkpoint()
    if deadline and time.time() > deadline:
        save_checkpoint()
        print("[stop] time budget reached; resume by re-running the same command", flush=True)
        sys.exit(0)

if calib:
    calib.report(trainer.step, {"arm": arm, "batch_size": bs})
    sys.exit(0)
if not a.eval_only:
    save_checkpoint()
    torch.save({k: m.state_dict() for k, m in modules.items()}, os.path.join(a.run_dir, "final.pt"))
caps = tuple(int(c) for c in ec.get("caps", [4, 8, 12, 16, 24]))
final = {"val": evaluate(val_p, val_s, caps=caps)}
# S1 operating point, chosen on val only: the largest cap whose mean NFE (verifier stop) is <= 12 (A2-9)
c_star = max((c for c in caps if final["val"]["d2_check"][f"cap{c}"]["mean_nfe"] <= 12), default=None)
tp = np.array(safe_pickle_load(os.path.join(ROOT, "data", "sudoku", "hard", "puzzles.pickle"))[: a.n_test])
ts = np.array(safe_pickle_load(os.path.join(ROOT, "data", "sudoku", "hard", "solutions.pickle"))[: a.n_test])
final["test_released"] = evaluate(tp, ts, caps=caps, final=True)
res = {"run": os.path.basename(a.run_dir.rstrip("/")), "arm": arm, "config": OmegaConf.to_container(cfg),
       "params_total": n_params(net) + (n_params(fb) if fb else 0), "steps": trainer.step,
       "operating_cap_from_val": c_star, "generator_flops_proxy_total": fl_step * trainer.step, "wall_s": trainer.wall_prev + time.time() - trainer.t_start,
       "device": dev, "gpu": torch.cuda.get_device_name() if dev == "cuda" else None, "torch": torch.__version__,
       "final_eval": final}
json.dump(res, open(os.path.join(a.run_dir, "result.json"), "w"), indent=1)
summ = {k: {kk: vv for kk, vv in v.items() if kk != "per_puzzle"} for k, v in final.items()}
print(json.dumps(summ, indent=1))
