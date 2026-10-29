"""Train or fine-tune an NCM generator on the released TinyStories tokenizer (frozen). Resumable; CUDA or CPU.

Training hierarchies (src/ncmr/corruption.py): clean (upstream), NCM's conditioning corruption (uniform codes,
clean labels), B1 (on-policy codes and/or labels re-quantized against the true residual), B7 (soft labels).

  python scripts/train_gen.py --config configs/scale/L_ncm.yaml --run_dir runs/L_ncm_s0 --seed 0

Config:
  init: released | scratch     # released = the paper's 88.6M TinyStories generator (fine-tuning, tier L)
  model: {n_layer, n_head, n_embd, dropout}                  # scratch only (tier M)
  corruption: {mode, mean_rate, spread, retarget, soft_tau, soft_topm, policy_temp}
              mode: mech = the W2M arms M0-M7 (src/ncmr/w2m.py): {arm, clean_row_frac, d_target (M7)}; arm m2r = M2 plus
              the L1R target reassignment (src/ncmr/l1r.py, config key l1r: {coverage_min}; replayed and gated before training)
  train: {steps, batch_size, grad_accum, lr, warmup, min_lr_frac, weight_decay, beta2, ckpt_every, eval_every,
          eval_n, log_every, compile}
  sample_eval: {every, at, n, final_seeds, final_n, mauve}   # at: explicit steps for the curve samples (W2M: 100/300/600)
On-policy sampling costs one extra no-grad forward per step; it is counted in `flops_est_total`, so arms are
compared at equal FLOPs by giving them steps in proportion (see the configs).
Writes run_dir/{ckpt.pt, final.pt, metrics.jsonl, result.json, samples_final.json}.
"""
from __future__ import annotations

import argparse
import dataclasses
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
from ncmr import l1r as L1R  # noqa: E402
from ncmr import w2m as W2M  # noqa: E402
from ncmr.corruption import CorruptionCfg, build_hierarchy, soft_ce  # noqa: E402
from ncmr.data import DATA_DIR, EOS, PAD, TrainStream, cut_at_eos, examples  # noqa: E402
from ncmr.models import build_ncm_generator, n_params  # noqa: E402
from ncmr.ncm_ref_loader import ForwardCounter, load_ncm  # noqa: E402
from ncmr.train import Calib, Trainer, fp32_eval, pick_device  # noqa: E402

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
ckpt_path, cfg_path = os.path.join(a.run_dir, "ckpt.pt"), os.path.join(a.run_dir, "config.yaml")
if a.eval_only and os.path.exists(ckpt_path):
    sys.exit(f"--eval_only evaluates the initial model, but {a.run_dir} holds a training checkpoint")


def _run_key(c):
    """What defines a training run. Eval / log / checkpoint cadence do not change it; the micro-batch partition does
    (dropout, corruption and policy draws and float reductions follow it; codex audit 3, A3-3)."""
    c = OmegaConf.to_container(c, resolve=True)
    t = {k: v for k, v in c.get("train", {}).items() if k not in ("eval_every", "ckpt_every", "log_every")}
    return [c.get("init"), c.get("model"), c.get("corruption"), c.get("level0"), t]


# a recorded run is identified even before its first checkpoint (codex audit 6, A6-1: a run interrupted before step
# ckpt_every leaves its manifest and metrics, and must not be continued under another identity)
_recorded = os.path.exists(ckpt_path) or os.path.exists(os.path.join(a.run_dir, "run_manifest.json"))
if _recorded and not a.calib_steps:
    if not os.path.exists(cfg_path):   # codex audit 3, A3-3: a checkpoint whose run cannot be identified fails closed
        sys.exit(f"{a.run_dir} holds a recorded run without its config.yaml: the run cannot be verified; use a fresh run directory")
    if _run_key(OmegaConf.load(cfg_path)) != _run_key(cfg):   # codex audit F5
        sys.exit(f"{a.run_dir} holds a checkpoint of a different run (init / model / corruption / train differ)")


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# run identity beyond the config (codex audit 3, A3-3): released bundle, BPE, data bytes and runtime. A checkpoint is
# resumed only with an identical manifest; checked before anything in run_dir is overwritten.
_art, _data = os.path.join(ROOT, "artifacts", "tinystories"), a.data_dir or DATA_DIR
manifest = {"base_sha256": {n: _digest(os.path.join(_art, n)) for n in ("config.yaml", "tokenizer.pt", "generator.pt",
                                                                        "bpe_vocab_8192.pickle", "bpe_merges_8192.pickle")},
            "data_sha256": {n: _digest(os.path.join(d, n)) for d, n in ((_data, "train_bpe_8192.bin"),
                                                                       (_data, "train_bpe_8192_offsets.npy"),
                                                                       (DATA_DIR, "valid_bpe_8192.pickle"))},
            "torch": torch.__version__, "cuda": torch.version.cuda}
manifest_path = os.path.join(a.run_dir, "run_manifest.json")
if _recorded and not a.calib_steps:
    if not os.path.exists(manifest_path) or json.load(open(manifest_path)) != manifest:
        sys.exit(f"{a.run_dir} holds a run whose base checkpoint, data or torch runtime differs from now (or was "
                 "not recorded): use a fresh run directory")
OmegaConf.save(cfg, cfg_path)
if not a.calib_steps:
    json.dump(manifest, open(manifest_path, "w"), indent=1)
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
if str(cfg.get("level0", "released")) == "train":
    # M1: from-scratch runs estimate the level-0 prior from training windows (own RNG, so the data order is unchanged)
    # instead of borrowing the released generator's stored unigram (tier-M review, 2026-10-23)
    _rng0 = np.random.default_rng(seed + 12345)
    _cnt = torch.zeros(tok.vq.vqvae_vocab_sizes[0], device=device)
    with torch.no_grad(), torch.autocast(device_type=device, enabled=False):
        for _ in range(int(cfg.get("level0_windows", 65536)) // 512):
            _x = torch.from_numpy(stream.batch(_rng0, 512)).to(device)
            _cnt += torch.bincount(tok.vq.z_to_idx(tok._encode_pre_quant(_x))[:, :l0].reshape(-1), minlength=_cnt.numel()).float()
    _rel = L0
    L0 = _cnt / _cnt.sum()
    print(f"[gen] level-0 prior from {int(_cnt.sum())} training codes: {int((L0 > 0).sum())} codes used, "
          f"total-variation distance to the released prior {0.5 * float((L0 - _rel).abs().sum()):.4f}", flush=True)
n_embd = int(cfg.model.n_embd) if cfg.init != "released" else int(BUNDLE.cfg.model.n_embd)
n_layer = int(cfg.model.n_layer) if cfg.init != "released" else int(BUNDLE.cfg.model.n_layer)
cc = cfg.get("corruption", {})
corr = CorruptionCfg(mode=str(cc.get("mode", "none")), mean_rate=float(cc.get("mean_rate", 0.5)),
                     spread=float(cc.get("spread", 0.5)), retarget=bool(cc.get("retarget", False)),
                     soft_tau=float(cc.get("soft_tau", 0.0)), soft_topm=int(cc.get("soft_topm", 16)))
if corr.mode == "mech":
    # W2M: canonical labels, one-step proposals, released generator (the frozen copy and the flag vector assume it)
    assert cfg.init == "released" and not corr.retarget and corr.soft_tau <= 0 and int(cc.get("policy_passes", 1)) == 1
    assert str(cc.get("arm")) in W2M.ALL_ARMS, f"corruption.arm must be one of {sorted(W2M.ALL_ARMS)}"
    if cc.get("d_target") is not None:   # M7: uniform replacement at A3's realized per-level disagreement
        corr = dataclasses.replace(corr, level_rates=W2M.m7_level_rates([float(v) for v in cc.d_target],
                                                                        tok.vq.vqvae_vocab_sizes))
if corr.soft_tau > 0:
    # W1b: freeze the per-level soft-label gap (mean nearest / second-nearest distance gap over 512 val windows), so
    # the targets do not depend on the micro-batch (codex audit F7)
    _gaps = [build_hierarchy(tok, val_x[s_:s_ + 128].to(device), corr,
                             generator=torch.Generator(device=device).manual_seed(0))["soft_gap_per_level"]
             for s_ in range(0, 512, 128)]
    corr = dataclasses.replace(corr, soft_gap=tuple(float(np.mean(g)) for g in zip(*_gaps)))
    print(f"[gen] frozen soft-label gap per level: {[round(g, 4) for g in corr.soft_gap]}", flush=True)
policy_temp = float(cc.get("policy_temp", 1.0))
policy_passes = int(cc.get("policy_passes", 1))   # 1 = one-step errors; t = errors after t parallel passes
fwd = FL.ncm_gen_fwd_per_seq(n_embd, n_layer, levels, codebook=tok.vq.vqvae_vocab_sizes[0], vq_dim=tok.embed_dim)
# FLOP-matching unit (W1): generator-only proxy = (fwd + 2x bwd + policy passes) x analytic forward per window. It omits
# tokenizer / hierarchy work and prices masked attention as block-sparse (the same in every arm; codex audit F4).
mech, flagmod = None, None
if corr.mode == "mech":
    # the frozen proposer is a copy of the RELEASED generator: made here, before any checkpoint restore
    mech = W2M.MechArm(str(cc.arm), tok, net, L0,
                       prefix_emb=lambda b: tok.encoder.wte(torch.full((b, 1), EOS, dtype=torch.long, device=device)),
                       device=device, seed=seed, rates=corr.rates(K), clean_frac=float(cc.get("clean_row_frac", 0.25)),
                       policy_temp=float(cc.get("policy_temp", 1.0)),
                       frozen=W2M.MechArm.frozen_copy(net) if W2M.ALL_ARMS[str(cc.arm)][0] == "frozen_wrong" else None)
    if mech.flag_mode:
        flagmod = W2M.FlagVector(tok.embed_dim).to(device)
fl_step = (3 + (policy_passes if corr.mode == "onpolicy" else 0) + (mech.proposal_forwards if mech else 0)) * fwd * int(tc.batch_size)
params = [p for p in net.parameters() if p.requires_grad] + (list(flagmod.parameters()) if flagmod else [])
trainer = Trainer(a.run_dir, params, max_lr=float(tc.lr), total_steps=int(tc.steps), warmup=int(tc.warmup),
                  weight_decay=float(tc.weight_decay), betas=(0.9, float(tc.get("beta2", 0.95))),
                  ckpt_every=int(tc.ckpt_every), device=device, amp=bool(tc.get("amp", True)),
                  min_lr_frac=float(tc.get("min_lr_frac", 0.1)))
modules = {"model": net, **({"flag": flagmod} if flagmod else {})}
resume_extra = trainer.maybe_resume(modules, data_rng)
if mech is not None and mech.arm == "m2r" and not (a.calib_steps or a.eval_only):
    # L1R: replay this run's own data / mask / replacement streams (no network forward, nothing from the global torch RNG or
    # data_rng), select the balanced set, gate on coverage. Plan files are written only by a run that starts at step 0.
    _bsz, _t0 = int(tc.batch_size), time.time()
    _rng = np.random.default_rng(seed)
    _dry = W2M.MechArm(mech.arm, tok, net, L0, mech.prefix_emb, device, seed, corr.rates(K), dry=True,
                       clean_frac=float(cc.get("clean_row_frac", 0.25)), policy_temp=float(cc.get("policy_temp", 1.0)))
    _rec = L1R.replay(lambda: torch.from_numpy(stream.batch(_rng, _bsz)).to(device), _dry, int(tc.steps),
                      int(tc.get("grad_accum", 1)), out_vocab=int(net.max_vocab))
    mech.l1r = L1R.build_plan(_rec, seed)
    print(f"[l1r] replay {int(tc.steps)} steps in {time.time() - _t0:.0f}s: {json.dumps(mech.l1r.summary())}", flush=True)
    try:
        L1R.gate(mech.l1r, float(cfg.get("l1r", {}).get("coverage_min", 0.80)), a.run_dir, write=not trainer.step)
    except L1R.L1RCoverageError as e:
        sys.exit(str(e))
fnet = torch.compile(net) if (device == "cuda" and bool(tc.get("compile", False))) else net
if mech:
    mech.fwd = fnet
print(f"[gen] init={cfg.init} params {n_params(net) / 1e6:.2f}M | corruption {corr} policy_temp {policy_temp} | "
      f"{fl_step / 1e12:.2f} TFLOP/step, {fl_step * int(tc.steps) / 1e15:.1f} PFLOP total | device {device}", flush=True)


def prefix_emb(b):
    return tok.encoder.wte(torch.full((b, 1), EOS, dtype=torch.long, device=device))


# ------------------------------------------------------------------------------------------------ steps
@torch.no_grad()
def policy_codes(x, gen):
    """The generator's own samples at every level (level 0: unigram, as at inference). Pass 1 samples each level
    given the clean coarser levels (one-step errors); pass t samples given pass t-1's codes (t-step errors:
    t = K-1 would be full sequential sampling given the true level 0). fp32, TF32 off, every module in eval mode
    and all draws from `gen`, the policy's own checkpointed RNG (codex audit 3, A3-1 / A3-2)."""
    with fp32_eval(net, device):
        codes = tok.vq.z_to_idx(tok._encode_pre_quant(x))
        # inference level 0 (unigram), drawn once: pass 1 sees the clean levels, pass t > 1 sees pass t-1's hierarchy
        # including this level 0 (codex audit F3)
        lv0 = torch.multinomial(L0, x.shape[0] * l0, replacement=True, generator=gen).view(x.shape[0], l0)
        for _ in range(policy_passes):
            z_in = tok.vq.idx_to_ncm_input(codes)
            logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
            probs = torch.softmax(logits / policy_temp, -1)
            samp = torch.multinomial(probs.flatten(0, 1), 1, generator=gen).view(logits.shape[:-1])
            codes = torch.cat([lv0, samp], 1)
    return codes


def compute_loss(x, gen):
    pol = policy_codes(x, policy_gen) if corr.mode == "onpolicy" else None
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
    return loss, {"corrupted_frac": round(h["corrupted_frac"], 4),
                  "corrupted_frac_per_level": [round(f, 4) for f in h["corrupted_frac_per_level"]],
                  "argmax_eq_canonical": round(agree, 4),
                  "labels_changed": round((h["targets"] != h["canonical"]).float().mean().item(), 4)}


def compute_loss_mech(x, plan, prepared=None):
    """W2M arms: canonical labels; replaced slots from the plan's mask stream, codes from the arm's proposer; flags (M4 /
    M5) enter as f * w on the code inputs. Evaluation never calls this: inference and evals use zero flags."""
    canon, repl = prepared if prepared is not None else (None, None)
    if canon is None:
        canon = mech.canonical(x)
        repl = mech.proposals(canon, plan)
    h = build_hierarchy(tok, x, corr, policy_codes=repl, masks=plan.masks, canonical=canon)
    mech.record(h["used"], h["targets"], plan)
    tgt, extra_logs = h["targets"], {}
    if mech.l1r is not None:   # M2R: roots checked against the replay, then the receivers' L1 targets reassigned
        tgt, n_re = mech.l1r.reassign(trainer.step, plan, canon, repl, tgt, l0, mech.levels[1])
        extra_logs = {"l1r_reassigned_frac": round(n_re / len(x), 4)}
    inputs = h["inputs"]
    if flagmod is not None:
        inputs = flagmod(inputs, mech.flags(plan))
    logits = fnet(torch.cat([prefix_emb(x.shape[0]), inputs], 1))
    V = logits.shape[-1]
    loss = F.cross_entropy(logits.float().reshape(-1, V), tgt[:, l0:].reshape(-1))
    with torch.no_grad():
        agree = (logits.argmax(-1) == h["canonical"][:, l0:]).float().mean().item()
    return loss, {"corrupted_frac": round(h["corrupted_frac"], 4),
                  "corrupted_frac_per_level": [round(f, 4) for f in h["corrupted_frac_per_level"]],
                  "argmax_eq_canonical": round(agree, 4),
                  "labels_changed": round((tgt != h["canonical"]).float().mean().item(), 4), **extra_logs}


@torch.no_grad()
def val_nll():
    """Teacher-forced NLL of the canonical codes given the clean hierarchy (nats/window, per level); fp32, TF32 off,
    caller's module modes restored (codex audit 3, A3-1; wave 1 scored this under the training autocast)."""
    tot, per = 0.0, torch.zeros(K - 1)
    with fp32_eval(net, device):
        for s in range(0, len(val_x), 100):
            x = val_x[s:s + 100].to(device)
            z_in, tgt = tok.get_ncm_inputs(x)
            logits = fnet(torch.cat([prefix_emb(x.shape[0]), z_in], 1)).float()
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
            tot += nll.sum().item()
            per += torch.stack([c.sum(1) for c in torch.split(nll, levels[1:], dim=1)], 1).sum(0).cpu()
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
        with torch.autocast(device_type=device, enabled=False):   # fp32 sampling incl. tokenizer (codex audit F8)
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
    """Isolated from the training RNG streams (codex audit F6), with TF32 off (F8)."""
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.random.fork_rng(devices=devices):
            return _sample_metrics(n, seeds, mauve)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


def _sample_metrics(n, seeds, mauve=False):
    from ncmr.eval_text import evaluate_samples, mauve_features
    ids, groups, nfe, wall = [], [], 0.0, 0.0
    for sd in seeds:
        o, nfe, w = draw_samples(n, sd)
        ids += o
        groups += [sd] * len(o)
        wall += w
    reenc = [bpe.encode(bpe.decode(s)) for s in ids]    # token statistics on the re-encoding, as upstream
    ref = None
    fp32_mauve = corr.mode == "mech"   # W2M registers MAUVE in fp32; W1's arms keep their bf16 featurization (audit 8, A8-1)
    if mauve:
        tag = "fp32_" if fp32_mauve else ""   # a separate reference cache: never mix bf16 human features with fp32 samples
        cache = os.path.join(ROOT, "runs", "cache", f"mauve_ref_val_{tag}{len(ids)}.npy")
        if os.path.exists(cache):
            ref = np.load(cache)
        else:
            ref = mauve_features([bpe.decode(cut_at_eos(r)) for r in examples("val", len(ids))], device,
                                 amp=not fp32_mauve)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, ref)
    scorers = ("gpt2-large", "roneneldan/TinyStories-33M") if mauve else ("gpt2-large",)
    res = evaluate_samples(reenc, bpe.decode, groups, scorers=scorers, ref_feats=ref, device=device,
                           mauve_amp=not fp32_mauve)
    res.update({"nfe_per_sample": nfe, "gen_wall_s": wall})
    return res, ids, groups


# ------------------------------------------------------------------------------------------------ loop
net.train()
deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
gen = torch.Generator(device=device).manual_seed(seed * 7919)
policy_gen = torch.Generator(device=device).manual_seed(seed * 7919 + 104729)   # on-policy draws only (audit 3, A3-2)
if trainer.step:
    # an exact continuation needs every RNG the run draws from (codex audit F1; audit 3, A3-3): fail closed
    if "corruption_rng" not in resume_extra:
        raise RuntimeError("checkpoint lacks the corruption RNG state; exact resume unavailable: use a fresh run directory")
    gen.set_state(resume_extra["corruption_rng"].cpu())
    if corr.mode == "onpolicy":
        if "policy_rng" not in resume_extra:
            raise RuntimeError("checkpoint lacks the policy RNG state (written before audit 3's fix): use a fresh run directory")
        policy_gen.set_state(resume_extra["policy_rng"].cpu())


if trainer.step and mech is not None:
    if "mech" not in resume_extra:
        raise RuntimeError("checkpoint lacks the W2M mask / replacement / flag RNG states; exact resume unavailable: "
                           "use a fresh run directory")
    mech.load_state_dict(resume_extra["mech"])
    if mech.l1r is not None and resume_extra.get("l1r_plan_sha256") != mech.l1r.sha256:   # the plan is part of the run's identity
        raise RuntimeError("the recomputed L1R plan differs from the checkpoint's (or the checkpoint has none): exact resume "
                           "unavailable; use a fresh run directory")


def save_ckpt():
    trainer.save(modules, data_rng, extra={"corruption_rng": gen.get_state(), "policy_rng": policy_gen.get_state(),
                                           **({"mech": mech.state_dict()} if mech else {}),
                                           **({"l1r_plan_sha256": mech.l1r.sha256} if mech and mech.l1r is not None else {})})


accum = int(tc.get("grad_accum", 1))
calib = Calib(a.run_dir, a.calib_steps, fl_step) if a.calib_steps else None
every_s = int(sc.get("every", 0))
at_s = {int(v) for v in sc.get("at", [])}
t_last, s_last = time.time(), trainer.step
while trainer.step < int(tc.steps):
    xb_full = torch.from_numpy(stream.batch(data_rng, int(tc.batch_size))).to(device)
    loss_sum, logs = 0.0, {}
    plan = mech.draw_plan(len(xb_full)) if mech else None   # one mask stream, drawn for the whole batch
    prepared = mech.prepare_batch(xb_full, plan, accum) if mech and mech.arm == "pd" else None
    row0 = 0
    for xb in torch.chunk(xb_full, accum):
        with trainer.ctx:
            ready = tuple(t[row0:row0 + len(xb)] for t in prepared) if prepared is not None else None
            loss, logs = compute_loss_mech(xb, plan.rows(row0, row0 + len(xb)), ready) if mech else compute_loss(xb, gen)
        row0 += len(xb)
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
                           "tflops_eff": round(sps * fl_step / 1e12, 1), **logs,
                           **({"realized": mech.realized()} if mech else {})})
        print(json.dumps(rec), flush=True)
    if trainer.step % int(tc.eval_every) == 0 or trainer.step == int(tc.steps):
        ev = val_nll()
        if (trainer.step in at_s) if at_s else (every_s and (trainer.step % every_s == 0 or trainer.step == int(tc.steps))):
            sm, _, _ = sample_metrics(int(sc.get("n", 256)), [seed + 10_000])
            ev.update({k: v for k, v in sm.items() if not k.endswith(("per_group", "per_seed"))})
        print(json.dumps(trainer.log({"eval": ev})), flush=True)
    if trainer.step % int(tc.ckpt_every) == 0:
        save_ckpt()
    if deadline and time.time() > deadline:
        save_ckpt()
        print("[stop] time budget reached; resume by re-running the same command", flush=True)
        sys.exit(0)

if calib:
    calib.report(trainer.step, {"init": cfg.init, "corruption": corr.mode, "batch_size": int(tc.batch_size), "grad_accum": accum})
    sys.exit(0)
if not a.eval_only:
    save_ckpt()
    torch.save({"model": net.state_dict(), "init": cfg.init, "model_cfg": OmegaConf.to_container(cfg.model) if "model" in cfg else {},
                "level0_unigram": L0.cpu(), **({"flag": flagmod.state_dict()} if flagmod else {})},
               os.path.join(a.run_dir, "final.pt"))
res_path = os.path.join(a.run_dir, "result.json")
final = {}
if not a.skip_final_eval:
    final = val_nll()
    seeds = [int(s) for s in sc.get("final_seeds", [42, 0, 1, 2])]
    sm, ids, groups = sample_metrics(int(sc.get("final_n", 256)), seeds, mauve=bool(sc.get("mauve", True)))
    final.update(sm)
    json.dump({"ids": ids, "groups": groups, "nfe_per_sample": sm["nfe_per_sample"], "run": os.path.basename(os.path.normpath(a.run_dir))},   # no absolute path in published results
              open(os.path.join(a.run_dir, "samples_final.json"), "w"))
wall = trainer.wall_prev + time.time() - trainer.t_start
res = {"run": os.path.basename(a.run_dir.rstrip("/")), "config": OmegaConf.to_container(cfg),
       "params_total": n_params(net), "steps": trainer.step, "flops_est_total": fl_step * trainer.step,
       "wall_s": wall, "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
       "torch": torch.__version__, "eval_precision": "fp32 sampling and scoring, TF32 off", "final_eval": final}
if mech:
    res.update({"w2m_arm": mech.arm, "proposal_forwards_per_step": mech.proposal_forwards,
                "realized_disagreement": mech.realized(), "flag_vector_norm": float(flagmod.w.norm()) if flagmod else None,
                **({"pd": mech.realized()["pd_coverage"]} if mech.arm == "pd" else {}),
                **({"l1r": mech.l1r.summary()} if mech.l1r is not None else {})})
json.dump(res, open(res_path, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1, default=str))
