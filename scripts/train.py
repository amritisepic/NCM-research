"""Train one of our models on TinyStories story openings (128 BPE tokens). Resumable; CPU or CUDA.

  python scripts/train.py tokenizer --config configs/tok_lookup_s.yaml --run_dir runs/tok_lookup_s
  python scripts/train.py ncm  --config configs/ncm_s.yaml --tokenizer runs/tok_lookup_s/final.pt --run_dir runs/ncm_s
  python scripts/train.py ts_fact|ts_ar --config configs/ncm_s.yaml --tokenizer runs/tok_lookup_s/final.pt --run_dir runs/ts_ar_s
  python scripts/train.py ar   --config configs/ar_s.yaml --run_dir runs/ar_s

Config keys (YAML): model{...}, train{steps, batch_size, lr, warmup, weight_decay, ckpt_every, eval_every,
eval_n, seed}. Writes run_dir/{ckpt.pt, final.pt, metrics.jsonl, result.json}.
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
from ncmr.data import EOS, TrainStream, examples  # noqa: E402
from ncmr.models import (ARTransformer, TwoStreamLevelModel, build_ncm_generator, build_tokenizer,  # noqa: E402
                         n_params, ncm_batch, tokenizer_loss)
from ncmr.train import Trainer, pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["tokenizer", "ncm", "ts_fact", "ts_ar", "ar"])
ap.add_argument("--config", required=True)
ap.add_argument("--run_dir", required=True)
ap.add_argument("--tokenizer", default="")
ap.add_argument("--data_dir", default="")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--max_minutes", type=float, default=0, help="stop (after checkpointing) after this long")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
cfg = OmegaConf.load(a.config)
tc = cfg.train
device = pick_device()
torch.manual_seed(int(tc.seed))
data_rng = np.random.default_rng(int(tc.seed))
stream = TrainStream(a.data_dir) if a.data_dir else TrainStream()
val_x = torch.from_numpy(examples("val", int(tc.get("eval_n", 1000))))
os.makedirs(a.run_dir, exist_ok=True)
OmegaConf.save(cfg, os.path.join(a.run_dir, "config.yaml"))


def load_tokenizer(path):
    st = torch.load(path, map_location="cpu", weights_only=False)  # our own file
    tok = build_tokenizer(**st["tok_kwargs"])
    tok.load_state_dict(st["model"])
    return tok.to(device).eval(), st


def prefix_of(B):
    return torch.full((B, 1), EOS, dtype=torch.long, device=device)


# ------------------------------------------------------------------------------------------------ build
extra_save = {}
if a.kind == "tokenizer":
    tok_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_tokenizer(**tok_kwargs).to(device)
    modules = {"model": model}
    extra_save["tok_kwargs"] = tok_kwargs
    fl_step = 3 * FL.tokenizer_fwd_per_seq(model.embed_dim, model.vq.levels, model.vq.vqvae_vocab_sizes[0],
                                           enc_layers=tok_kwargs.get("enc_layers", 0)) * tc.batch_size
else:
    if a.kind in ("ncm", "ts_fact", "ts_ar"):
        tok, tok_state = load_tokenizer(a.tokenizer)
        for p in tok.parameters():
            p.requires_grad_(False)
        gen = build_ncm_generator(tok, **OmegaConf.to_container(cfg.model, resolve=True)).to(device)
        model = TwoStreamLevelModel(gen, a.kind[3:]).to(device) if a.kind.startswith("ts_") else gen
        extra_save["tokenizer_path"] = os.path.abspath(a.tokenizer)
        extra_save["gen_kwargs"] = OmegaConf.to_container(cfg.model, resolve=True)
        levels = [int(l) for l in tok.vq.levels]
        fl_step = 3 * FL.ncm_gen_fwd_per_seq(cfg.model.n_embd, cfg.model.n_layer, levels,
                                             codebook=tok.vq.vqvae_vocab_sizes[0], vq_dim=tok.embed_dim) * tc.batch_size
        if a.kind.startswith("ts_"):
            fl_step *= 2  # two streams: ~2x positions
    else:
        model = ARTransformer(**OmegaConf.to_container(cfg.model, resolve=True)).to(device)
        extra_save["ar_kwargs"] = OmegaConf.to_container(cfg.model, resolve=True)
        fl_step = 3 * FL.ar_fwd_per_seq(cfg.model.n_embd, cfg.model.n_layer) * tc.batch_size
    modules = {"model": model}

params = [p for p in model.parameters() if p.requires_grad]
trainer = Trainer(a.run_dir, params, max_lr=float(tc.lr), total_steps=int(tc.steps), warmup=int(tc.warmup),
                  weight_decay=float(tc.weight_decay), ckpt_every=int(tc.ckpt_every), device=device,
                  amp=bool(tc.get("amp", True)))
trainer.maybe_resume(modules, data_rng)
print(f"[{a.kind}] params total {n_params(model) / 1e6:.2f}M trainable {n_params(model, True) / 1e6:.2f}M | "
      f"est. {fl_step / 1e12:.3f} TFLOP/step, {fl_step * tc.steps / 1e15:.2f} PFLOP total | device {device}", flush=True)


# ------------------------------------------------------------------------------------------------ steps
def compute_loss(xb: torch.Tensor, train: bool):
    if a.kind == "tokenizer":
        return tokenizer_loss(model, xb)
    if a.kind == "ar":
        x = torch.cat([prefix_of(xb.shape[0]), xb], 1)
        return model.loss(x), {}
    z_all, targets = ncm_batch(tok, xb, prefix_of(xb.shape[0]))
    logits = model(z_all, targets) if a.kind.startswith("ts_") else model(z_all)
    nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return nll, {}


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    if a.kind == "tokenizer":
        accs, ces = [], []
        for s in range(0, len(val_x), 250):
            xb = val_x[s:s + 250].to(device)
            idx = model.vq.z_to_idx(model._encode_pre_quant(xb))
            from ncmr.ncm_tools import zhat_snapshots
            snaps = zhat_snapshots(model.vq, idx)
            accs.append([(model.decoder(z).argmax(-1) == xb).float().mean().item() for z in snaps])
        out["val_acc_per_level"] = np.mean(accs, 0).round(4).tolist()
        out["val_acc_full"] = out["val_acc_per_level"][-1]
    else:
        tot, n, per_level = 0.0, 0, None
        for s in range(0, len(val_x), 100):
            xb = val_x[s:s + 100].to(device)
            if a.kind == "ar":
                x = torch.cat([prefix_of(xb.shape[0]), xb], 1)
                logits = model(x[:, :-1]).float()
                nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
                tot += nll.sum().item()
                n += xb.shape[0]
            else:
                z_all, targets = ncm_batch(tok, xb, prefix_of(xb.shape[0]))
                logits = (model(z_all, targets) if a.kind.startswith("ts_") else model(z_all)).float()
                nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                      reduction="none").view(targets.shape)
                tot += nll.sum().item()
                n += xb.shape[0]
                lv = tok.vq.levels[1:]
                pl = torch.stack([c.sum(1) for c in torch.split(nll, list(lv), dim=1)], 1).sum(0)
                per_level = pl if per_level is None else per_level + pl
        out["val_nll_nats_per_seq"] = tot / n
        if per_level is not None:
            out["val_nll_per_level_nats_per_seq"] = (per_level / n).round(decimals=3).tolist()
    model.train()
    return out


t0 = time.time()
model.train()
deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
last = {}
accum = int(tc.get("grad_accum", 1))  # micro-batching for small GPUs; same data per optimizer step
while trainer.step < int(tc.steps):
    xb_full = torch.from_numpy(stream.batch(data_rng, int(tc.batch_size))).to(device)
    loss_sum = 0.0
    for xb in torch.chunk(xb_full, accum):
        with trainer.ctx:
            loss, logs = compute_loss(xb, True)
        trainer.backward(loss / accum)
        loss_sum += loss.item() / accum
    gn = trainer.step_opt(params)
    if trainer.step % int(tc.get("log_every", 50)) == 0 or trainer.step == 1:
        rec = trainer.log({"loss": round(loss_sum, 4), "gnorm": round(gn, 3), **logs})
        print(json.dumps(rec), flush=True)
    if trainer.step % int(tc.eval_every) == 0 or trainer.step == int(tc.steps):
        last = evaluate()
        print(json.dumps(trainer.log({"eval": last})), flush=True)
    if trainer.step % int(tc.ckpt_every) == 0:
        trainer.save(modules, data_rng, extra_save)
    if deadline and time.time() > deadline:
        trainer.save(modules, data_rng, extra_save)
        print("[stop] time budget reached; resume by re-running the same command", flush=True)
        sys.exit(0)

if not last:  # re-run of an already finished run: evaluate once so result.json is complete
    last = evaluate()
trainer.save(modules, data_rng, extra_save)
final = {"model": model.state_dict(), **extra_save}
if a.kind in ("ncm", "ts_fact", "ts_ar"):
    # level-0 unigram over training codes (NCM samples level 0 from it)
    counts = torch.zeros(tok.vq.vqvae_vocab_sizes[0])
    with torch.no_grad():
        for _ in range(20):
            xb = torch.from_numpy(stream.batch(data_rng, 256)).to(device)
            l0 = tok.encode(xb)[:, 0]
            counts += torch.bincount(l0.cpu(), minlength=counts.numel()).float()
    final["level0_unigram"] = counts / counts.sum()
    # NLL of val level-0 codes under that unigram (NCM's level-0 "model"), add-one smoothed, nats/window
    with torch.no_grad():
        l0v = torch.cat([tok.encode(val_x[s:s + 250].to(device))[:, 0].cpu() for s in range(0, len(val_x), 250)])
    p0 = (counts + 1) / (counts.sum() + counts.numel())
    last["val_nll_level0_unigram"] = float(-torch.log(p0[l0v]).mean())
torch.save(final, os.path.join(a.run_dir, "final.pt"))
wall = trainer.wall_prev + time.time() - trainer.t_start
res = {"run": os.path.basename(a.run_dir.rstrip("/")), "kind": a.kind, "config": OmegaConf.to_container(cfg),
       "params_total": n_params(model), "params_trainable": n_params(model, True),
       "steps": trainer.step, "flops_est_total": fl_step * trainer.step, "wall_s": wall, "device": device,
       "torch": torch.__version__, "final_eval": last}
with open(os.path.join(a.run_dir, "result.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1))
