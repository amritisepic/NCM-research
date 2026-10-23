"""Train or fine-tune an NCM generator on the released TinyStories tokenizer (frozen). Resumable; CUDA or CPU.

Training hierarchies (src/ncmr/corruption.py): clean (upstream), NCM's conditioning corruption (uniform codes,
clean labels), B1 (on-policy codes and/or labels re-quantized against the true residual), B7 (soft labels).

  python scripts/train_gen.py --config configs/scale/L_ncm.yaml --run_dir runs/L_ncm_s0 --seed 0

Config:
  init: released | scratch     # released = the paper's 88.6M TinyStories generator (fine-tuning, tier L)
  model: {n_layer, n_head, n_embd, dropout}                  # scratch only (tier M)
  corruption: {mode, mean_rate, spread, retarget, soft_tau, soft_topm, policy_temp}
  train: {steps, batch_size, grad_accum, lr, warmup, min_lr_frac, weight_decay, beta2, ckpt_every, eval_every,
          eval_n, log_every, compile}
  sample_eval: {every, n, final_seeds, final_n, mauve}
On-policy sampling costs one extra no-grad forward per step; it is counted in `flops_est_total`, so arms are
compared at equal FLOPs by giving them steps in proportion (see the configs).
Writes run_dir/{ckpt.pt, final.pt, metrics.jsonl, result.json, samples_final.json}.
"""
from __future__ import annotations

import argparse
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
from ncmr.corruption import CorruptionCfg, build_hierarchy, soft_ce  # noqa: E402
from ncmr.data import EOS, PAD, TrainStream, cut_at_eos, examples  # noqa: E402
from ncmr.models import build_ncm_generator, n_params  # noqa: E402
from ncmr.ncm_ref_loader import ForwardCounter, load_ncm  # noqa: E402
from ncmr.train import Calib, Trainer, pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--run_dir", required=True)
ap.add_argument("--seed", type=int, default=None, help="overrides train.seed")
ap.add_argument("--data_dir", default="")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--max_minutes", type=float, default=0, help="stop (after checkpointing) after this long")
ap.add_argument("--skip_final_eval", action="store_true")
ap.add_argument("--calib_steps", type=int, default=0, help="throughput calibration only: run this many steps, no eval/ckpt")
ap.add_argument("--micro_batch", type=int, default=0, help="overrides train.grad_accum = batch_size / micro_batch")
ap.add_argument("--eval_only", action="store_true", help="no training: final eval of the initial model (reference arm)")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
cfg = OmegaConf.load(a.config)
tc, sc = cfg.train, cfg.get("sample_eval", {})
seed = int(a.seed if a.seed is not None else tc.get("seed", 0))
cfg.train.seed = seed
if a.calib_steps:
    tc.steps, tc.eval_every, tc.ckpt_every = a.calib_steps, 10 ** 9, 10 ** 9
    sc = {}
if a.micro_batch:
    tc.grad_accum = max(1, int(tc.batch_size) // a.micro_batch)
if a.eval_only:
    tc.steps = 0
device = pick_device()
torch.manual_seed(seed)
data_rng = np.random.default_rng(seed)
stream = TrainStream(a.data_dir) if a.data_dir else TrainStream()
val_x = torch.from_numpy(examples("val", int(tc.get("eval_n", 1000))))
os.makedirs(a.run_dir, exist_ok=True)
OmegaConf.save(cfg, os.path.join(a.run_dir, "config.yaml"))
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ------------------------------------------------------------------------------------------------ build
BUNDLE = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"), device=device)
tok, bpe = BUNDLE.vqvae.eval(), BUNDLE.pre_tokenizer
for p in tok.parameters():
    p.requires_grad_(False)
if cfg.init == "released":
    net = BUNDLE.net
else:
    net = build_ncm_generator(tok, **OmegaConf.to_container(cfg.model, resolve=True)).to(device)
L0 = BUNDLE.ckpt_extra["inference"]["level0_unigram"].float().to(device)
levels = [int(l) for l in tok.vq.levels]
K, S, l0 = len(levels), sum(levels), levels[0]
n_embd = int(cfg.model.n_embd) if cfg.init != "released" else int(BUNDLE.cfg.model.n_embd)
n_layer = int(cfg.model.n_layer) if cfg.init != "released" else int(BUNDLE.cfg.model.n_layer)
cc = cfg.get("corruption", {})
corr = CorruptionCfg(mode=str(cc.get("mode", "none")), mean_rate=float(cc.get("mean_rate", 0.5)),
                     spread=float(cc.get("spread", 0.5)), retarget=bool(cc.get("retarget", False)),
                     soft_tau=float(cc.get("soft_tau", 0.0)), soft_topm=int(cc.get("soft_topm", 16)))
policy_temp = float(cc.get("policy_temp", 1.0))
policy_passes = int(cc.get("policy_passes", 1))   # 1 = one-step errors; t = errors after t parallel passes
fwd = FL.ncm_gen_fwd_per_seq(n_embd, n_layer, levels, codebook=tok.vq.vqvae_vocab_sizes[0], vq_dim=tok.embed_dim)
fl_step = (3 + (policy_passes if corr.mode == "onpolicy" else 0)) * fwd * int(tc.batch_size)
params = [p for p in net.parameters() if p.requires_grad]
trainer = Trainer(a.run_dir, params, max_lr=float(tc.lr), total_steps=int(tc.steps), warmup=int(tc.warmup),
                  weight_decay=float(tc.weight_decay), betas=(0.9, float(tc.get("beta2", 0.95))),
                  ckpt_every=int(tc.ckpt_every), device=device, amp=bool(tc.get("amp", True)),
                  min_lr_frac=float(tc.get("min_lr_frac", 0.1)))
modules = {"model": net}
trainer.maybe_resume(modules, data_rng)
fnet = torch.compile(net) if (device == "cuda" and bool(tc.get("compile", False))) else net
print(f"[gen] init={cfg.init} params {n_params(net) / 1e6:.2f}M | corruption {corr} policy_temp {policy_temp} | "
      f"{fl_step / 1e12:.2f} TFLOP/step, {fl_step * int(tc.steps) / 1e15:.1f} PFLOP total | device {device}", flush=True)


def prefix_emb(b):
    return tok.encoder.wte(torch.full((b, 1), EOS, dtype=torch.long, device=device))


# ------------------------------------------------------------------------------------------------ steps
@torch.no_grad()
def policy_codes(x, gen):
    """The generator's own samples at every level (level 0: unigram, as at inference). Pass 1 samples each level
    given the clean coarser levels (one-step errors); pass t samples given pass t-1's codes (t-step errors:
    t = K-1 would be full sequential sampling given the true level 0)."""
    was = net.training
    net.eval()
    with torch.autocast(device_type=device, enabled=False):
        idx = tok.vq.z_to_idx(tok._encode_pre_quant(x))
    codes = idx
    for _ in range(policy_passes):
        with torch.autocast(device_type=device, enabled=False):
            z_in = tok.vq.idx_to_ncm_input(codes)
        with trainer.ctx:
            logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
        samp = torch.distributions.Categorical(logits=logits / policy_temp).sample()
        codes = torch.cat([idx[:, :l0], samp], 1)
    net.train(was)
    lv0 = torch.multinomial(L0, x.shape[0] * l0, replacement=True, generator=gen).view(x.shape[0], l0)
    return torch.cat([lv0, samp], 1)


def compute_loss(x, gen):
    pol = policy_codes(x, gen) if corr.mode == "onpolicy" else None
    h = build_hierarchy(tok, x, corr, policy_codes=pol, generator=gen)
    logits = fnet(torch.cat([prefix_emb(x.shape[0]), h["inputs"]], 1))
    tgt = h["targets"][:, l0:]
    V = logits.shape[-1]
    if corr.soft_tau > 0:
        loss = soft_ce(logits.reshape(-1, V), h["soft_idx"][:, l0:].reshape(-1, corr.soft_topm),
                       h["soft_w"][:, l0:].reshape(-1, corr.soft_topm))
    else:
        loss = F.cross_entropy(logits.float().reshape(-1, V), tgt.reshape(-1))
    with torch.no_grad():
        agree = (logits.argmax(-1) == h["canonical"][:, l0:]).float().mean().item()
    return loss, {"corrupted_frac": round(h["corrupted_frac"], 4), "argmax_eq_canonical": round(agree, 4),
                  "labels_changed": round((h["targets"] != h["canonical"]).float().mean().item(), 4)}


@torch.no_grad()
def val_nll():
    """Teacher-forced NLL of the canonical codes given the clean hierarchy (nats/window, per level)."""
    net.eval()
    tot, per = 0.0, torch.zeros(K - 1)
    for s in range(0, len(val_x), 100):
        x = val_x[s:s + 100].to(device)
        with torch.autocast(device_type=device, enabled=False):
            z_in, tgt = tok.get_ncm_inputs(x)
        with trainer.ctx:
            logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
        tot += nll.sum().item()
        per += torch.stack([c.sum(1) for c in torch.split(nll, levels[1:], dim=1)], 1).sum(0).cpu()
    net.train()
    n = len(val_x)
    return {"val_nll": tot / n, "val_nll_per_level": (per / n).round(decimals=2).tolist()}


@torch.no_grad()
def draw_samples(n, seed_, bs=256):
    """Unconditional samples with the paper protocol (level 0 ~ unigram, temperature 1, eager, 11 NFE)."""
    net.eval()
    g = torch.Generator(device=device).manual_seed(seed_)
    torch.manual_seed(seed_)
    out, nfe, t0 = [], ForwardCounter(net), time.time()
    for s in range(0, n, bs):
        b = min(bs, n - s)
        lv0 = torch.multinomial(L0, b * l0, replacement=True, generator=g).view(b, l0)
        with trainer.ctx:
            codes = net.generate(lv0, torch.full((b, 1), EOS, device=device), temperature=1.0, top_k=0, top_p=1.0,
                                 cfg_scale=0.0, cfg_null_eos=EOS)
        with torch.autocast(device_type=device, enabled=False):
            ids = tok.decoder(tok.vq.idx_to_z_q(codes)).argmax(-1)
        out += [[t for t in cut_at_eos(r) if t != PAD] for r in ids.tolist()]
    calls = nfe.n
    nfe.close()
    net.train()
    return out, calls / max(1, -(-n // bs)), time.time() - t0


def sample_metrics(n, seeds, mauve=False):
    from ncmr.eval_text import evaluate_samples, mauve_features
    ids, groups, nfe, wall = [], [], 0.0, 0.0
    for sd in seeds:
        o, nfe, w = draw_samples(n, sd)
        ids += o
        groups += [sd] * len(o)
        wall += w
    reenc = [bpe.encode(bpe.decode(s)) for s in ids]    # token statistics on the re-encoding, as upstream
    ref = None
    if mauve:
        cache = os.path.join(ROOT, "runs", "cache", f"mauve_ref_val_{len(ids)}.npy")
        if os.path.exists(cache):
            ref = np.load(cache)
        else:
            ref = mauve_features([bpe.decode(cut_at_eos(r)) for r in examples("val", len(ids))], device)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, ref)
    scorers = ("gpt2-large", "roneneldan/TinyStories-33M") if mauve else ("gpt2-large",)
    res = evaluate_samples(reenc, bpe.decode, groups, scorers=scorers, ref_feats=ref, device=device)
    res.update({"nfe_per_sample": nfe, "gen_wall_s": wall})
    return res, ids, groups


# ------------------------------------------------------------------------------------------------ loop
net.train()
deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
gen = torch.Generator(device=device).manual_seed(seed * 7919 + trainer.step)
accum = int(tc.get("grad_accum", 1))
calib = Calib(a.run_dir, a.calib_steps, fl_step) if a.calib_steps else None
every_s = int(sc.get("every", 0))
t_last, s_last = time.time(), trainer.step
while trainer.step < int(tc.steps):
    xb_full = torch.from_numpy(stream.batch(data_rng, int(tc.batch_size))).to(device)
    loss_sum, logs = 0.0, {}
    for xb in torch.chunk(xb_full, accum):
        with trainer.ctx:
            loss, logs = compute_loss(xb, gen)
        trainer.backward(loss / accum)
        loss_sum += loss.item() / accum
    gn = trainer.step_opt(params)
    if calib:
        calib.tick(trainer.step)
        continue
    if trainer.step % int(tc.get("log_every", 50)) == 0 or trainer.step == 1:
        sps = (trainer.step - s_last) / max(time.time() - t_last, 1e-9)
        t_last, s_last = time.time(), trainer.step
        rec = trainer.log({"loss": round(loss_sum, 4), "gnorm": round(gn, 3), "steps_per_s": round(sps, 3),
                           "tflops_eff": round(sps * fl_step / 1e12, 1), **logs})
        print(json.dumps(rec), flush=True)
    if trainer.step % int(tc.eval_every) == 0 or trainer.step == int(tc.steps):
        ev = val_nll()
        if every_s and (trainer.step % every_s == 0 or trainer.step == int(tc.steps)):
            sm, _, _ = sample_metrics(int(sc.get("n", 256)), [seed + 10_000])
            ev.update({k: v for k, v in sm.items() if not k.endswith(("per_group", "per_seed"))})
        print(json.dumps(trainer.log({"eval": ev})), flush=True)
    if trainer.step % int(tc.ckpt_every) == 0:
        trainer.save(modules, data_rng)
    if deadline and time.time() > deadline:
        trainer.save(modules, data_rng)
        print("[stop] time budget reached; resume by re-running the same command", flush=True)
        sys.exit(0)

if calib:
    calib.report(trainer.step, {"init": cfg.init, "corruption": corr.mode, "batch_size": int(tc.batch_size), "grad_accum": accum})
    sys.exit(0)
if not a.eval_only:
    trainer.save(modules, data_rng)
    torch.save({"model": net.state_dict(), "init": cfg.init, "model_cfg": OmegaConf.to_container(cfg.model) if "model" in cfg else {},
                "level0_unigram": L0.cpu()}, os.path.join(a.run_dir, "final.pt"))
res_path = os.path.join(a.run_dir, "result.json")
final = {}
if not a.skip_final_eval:
    final = val_nll()
    seeds = [int(s) for s in sc.get("final_seeds", [42, 0, 1, 2])]
    sm, ids, groups = sample_metrics(int(sc.get("final_n", 256)), seeds, mauve=bool(sc.get("mauve", True)))
    final.update(sm)
    json.dump({"ids": ids, "groups": groups, "nfe_per_sample": sm["nfe_per_sample"], "run_dir": a.run_dir},
              open(os.path.join(a.run_dir, "samples_final.json"), "w"))
wall = trainer.wall_prev + time.time() - trainer.t_start
res = {"run": os.path.basename(a.run_dir.rstrip("/")), "config": OmegaConf.to_container(cfg),
       "params_total": n_params(net), "steps": trainer.step, "flops_est_total": fl_step * trainer.step,
       "wall_s": wall, "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
       "torch": torch.__version__, "final_eval": final}
json.dump(res, open(res_path, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1, default=str))
