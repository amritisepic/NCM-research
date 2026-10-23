"""B1 pre-check (no training): can later levels repair errors in earlier levels, if their targets are
re-quantized against the true residual? Released TinyStories tokenizer + generator, CPU.

For each corruption pattern we build the hierarchy twice and decode the codes that are fed forward:
  no-repair : later levels keep their canonical codes (what NCM's clean labels teach the generator to emit);
  repair    : later levels are re-quantized against z - z_used (B1's targets).
Token accuracy of the decoded text vs the true window is the ceiling a perfect generator reaches under
each training target, given those coarse errors. Patterns: one level fully replaced (uniform random codes,
or the released generator's own one-step samples given clean coarser levels), and the full recipes
(NCM's uniform mean-0.5 schedule; the same schedule with on-policy codes).
Also reports the generator's one-step disagreement with the canonical codes per level (sampled, greedy).

Usage: python scripts/repair_capacity.py --n 256
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.corruption import CorruptionCfg, build_hierarchy  # noqa: E402
from ncmr.data import EOS, examples  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
t0 = time.time()

B = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"))
vqvae, net, vq = B.vqvae, B.net, B.vqvae.vq
levels = [int(l) for l in vq.levels]
K, S = len(levels), sum(levels)
L0 = B.ckpt_extra["inference"]["level0_unigram"].float()
X = torch.from_numpy(examples("val", a.n))


@torch.no_grad()
def policy(x, g):
    """One-step samples (temperature 1) and greedy codes of the generator given the clean hierarchy."""
    idx = vq.z_to_idx(vqvae._encode_pre_quant(x))
    z_in = vq.idx_to_ncm_input(idx)
    logits = net(torch.cat([vqvae.encoder.wte(torch.full((x.shape[0], 1), EOS)), z_in], 1)).float()
    samp = torch.distributions.Categorical(logits=logits).sample()
    l0 = torch.multinomial(L0, x.shape[0] * levels[0], replacement=True, generator=g).view(x.shape[0], levels[0])
    return idx, torch.cat([l0, samp], 1), torch.cat([idx[:, :levels[0]], logits.argmax(-1)], 1)


def acc_of(x, h):
    with torch.no_grad():
        return (vqvae.decoder(h["z_used"]).argmax(-1) == x).float().mean().item()


patterns = [("none", CorruptionCfg(mode="none"))]
for j in range(K - 1):
    for mode in ("uniform", "onpolicy"):
        rates = [0.0] * K
        rates[j] = 1.0
        patterns.append((f"L{j}_{mode}_r1", CorruptionCfg(mode=mode, level_rates=tuple(rates))))
patterns += [("recipe_uniform_mean0.5", CorruptionCfg(mode="uniform", mean_rate=0.5)),
             ("recipe_onpolicy_mean0.5", CorruptionCfg(mode="onpolicy", mean_rate=0.5))]
for mode in ("uniform", "onpolicy"):   # every level but the finest replaced: can the finest level alone carry the text?
    patterns.append((f"all_but_finest_{mode}_r1", CorruptionCfg(mode=mode, level_rates=tuple([1.0] * (K - 1) + [0.0]))))


@torch.no_grad()
def code_ranks(x, codes):
    """Rank (0 = nearest) of `codes` in each level's distance ordering to the clean residual."""
    z = vqvae._encode_pre_quant(x).float()
    zc, out, acc = torch.zeros_like(z), [], 0
    for k, lk in enumerate(levels):
        resid = z - zc
        u = vq._down_for_quant(resid, k, lk) if k != K - 1 else resid
        d = vq._distances(u.reshape(-1, z.shape[-1]), vq.embeddings[k].weight).view(x.shape[0], lk, -1)
        c = codes[:, acc:acc + lk]
        out.append((d < d.gather(-1, c[..., None])).sum(-1))
        canon = d.argmin(-1)
        q = vq.embeddings[k](canon)
        q = vq._up_to_full(q, k, z.shape[1]) if k != K - 1 else q
        zc += vq._apply_quant_resi(q, k)
        acc += lk
    return out

sums = {name: {"no_repair": [], "repair": [], "corrupted_frac": [], "labels_changed": []} for name, _ in patterns}
dis_s, dis_g = np.zeros(K), np.zeros(K)
ranks = {"sampled": [[] for _ in range(K)], "greedy": [[] for _ in range(K)]}
g = torch.Generator().manual_seed(a.seed)
for s in range(0, a.n, a.bs):
    x = X[s:s + a.bs]
    idx, pol, greedy = policy(x, g)
    off = np.cumsum([0] + levels)
    for k in range(K):
        dis_s[k] += (pol[:, off[k]:off[k + 1]] != idx[:, off[k]:off[k + 1]]).float().sum().item()
        dis_g[k] += (greedy[:, off[k]:off[k + 1]] != idx[:, off[k]:off[k + 1]]).float().sum().item()
    for nm, cc in (("sampled", pol), ("greedy", greedy)):
        for k, r in enumerate(code_ranks(x, cc)):
            ranks[nm][k].append(r.flatten())
    for name, cfg in patterns:
        for arm, rt in (("no_repair", False), ("repair", True)):
            cfg.retarget = rt
            gg = torch.Generator().manual_seed(a.seed * 1000 + s)   # same corruption mask for both arms
            h = build_hierarchy(vqvae, x, cfg, policy_codes=pol, generator=gg)
            sums[name][arm].append(acc_of(x, h) * x.shape[0])
            if rt:
                sums[name]["corrupted_frac"].append(h["corrupted_frac"] * x.shape[0])
                sums[name]["labels_changed"].append((h["targets"] != h["canonical"]).float().mean().item() * x.shape[0])
    print(f"  {s + x.shape[0]}/{a.n}  {time.time() - t0:.0f}s", flush=True)

n = a.n
out = {"run": "b1_repair_capacity", "n_windows": n, "seed": a.seed,
       "generator_one_step_disagreement": {"sampled": (dis_s / (n * np.array(levels))).round(4).tolist(),
                                           "greedy": (dis_g / (n * np.array(levels))).round(4).tolist()},
       "token_accuracy": {}, "generator_code_rank_in_encoder_distance_order": {}}
for nm, rr in ranks.items():   # how "near" the generator's choices are to the encoder's: soft-target (B7) relevance
    rr = [torch.cat(r).float() for r in rr]
    out["generator_code_rank_in_encoder_distance_order"][nm] = {
        "median": [float(r.median()) for r in rr],
        "frac_top1": [round(float((r < 1).float().mean()), 4) for r in rr],
        "frac_top16": [round(float((r < 16).float().mean()), 4) for r in rr],
        "frac_top256": [round(float((r < 256).float().mean()), 4) for r in rr]}
for name, d in sums.items():
    out["token_accuracy"][name] = {k: round(float(np.sum(v) / n), 5) for k, v in d.items()}
out["wall_s"] = time.time() - t0
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "b1_repair_capacity.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
