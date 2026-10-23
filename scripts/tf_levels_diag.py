"""Where does NCM lose sample quality? Released TinyStories model, no training.

For a real validation window, keep its canonical codes at levels 0..j ("plan" from real text) and sample
levels j+1..K-1 with the generator (temperature 1, sequential, the paper's protocol). j = -1 is ordinary
unconditional sampling (level 0 from the unigram); j = K-1 is the tokenizer's reconstruction of real text.
The quality curve over j (Gen-PPL with entropy, duplicates, MAUVE) shows at which levels quality is lost:
a big drop between j and j+1 means sampling level j+1 (given a real coarser plan) is where it goes wrong.

Usage: python scripts/tf_levels_diag.py --n 512 --js -1 0 1 2 3 4 5 6 7 8 9 10 11
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

from ncmr.data import EOS, PAD, cut_at_eos, examples  # noqa: E402
from ncmr.eval_text import evaluate_samples, mauve_features  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--js", type=int, nargs="*", default=list(range(-1, 12)))
ap.add_argument("--no_mauve", action="store_true")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.device
B = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"), device=dev)
tok, net, bpe = B.vqvae, B.net, B.pre_tokenizer
vq = tok.vq
levels = [int(l) for l in vq.levels]
K, S = len(levels), sum(levels)
off = np.cumsum([0] + levels)
L0 = B.ckpt_extra["inference"]["level0_unigram"].float().to(dev)
X = torch.from_numpy(examples("val", a.n))


@torch.no_grad()
def sample_from(x, j, g):
    b = x.shape[0]
    codes = torch.zeros(b, S, dtype=torch.long, device=dev)
    if j >= 0:
        codes[:, :off[j + 1]] = vq.z_to_idx(tok._encode_pre_quant(x))[:, :off[j + 1]]
    else:
        codes[:, :levels[0]] = torch.multinomial(L0, b * levels[0], replacement=True, generator=g).view(b, levels[0])
    pre = tok.encoder.wte(torch.full((b, 1), EOS, dtype=torch.long, device=dev))
    for k in range(max(j + 1, 1), K):
        # full block-diagonal pass; level k's logits depend only on levels < k, which are final
        logits = net(torch.cat([pre, vq.idx_to_ncm_input(codes)], 1)).float()
        lk = logits[:, off[k] - levels[0]:off[k + 1] - levels[0]]
        u = torch.rand(lk.shape, device=dev, generator=g).clamp_min(1e-20)
        codes[:, off[k]:off[k + 1]] = (lk - torch.log(-torch.log(u))).argmax(-1)   # Gumbel-max = sampling at T=1
    ids = tok.decoder(vq.idx_to_z_q(codes)).argmax(-1)
    return [[t for t in cut_at_eos(r) if t != PAD] for r in ids.tolist()]


t0 = time.time()
ref = None
if not a.no_mauve:
    cache = os.path.join(ROOT, "runs", "cache", f"mauve_ref_test_{a.n}.npy")   # human references: test split
    if os.path.exists(cache):
        ref = np.load(cache)
    else:
        ref = mauve_features([bpe.decode(cut_at_eos(r)) for r in examples("test", a.n)], dev)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, ref)
out = {"run": "diag_tf_levels_released_ts", "n": a.n, "seed": a.seed, "levels": levels, "by_j": {}}
for j in a.js:
    g = torch.Generator(device=dev).manual_seed(a.seed * 100 + j + 1)
    ids = []
    for s in range(0, a.n, a.bs):
        ids += sample_from(X[s:s + a.bs].to(dev), j, g)
    reenc = [bpe.encode(bpe.decode(s)) for s in ids]
    r = evaluate_samples(reenc, bpe.decode, [0] * len(reenc), ref_feats=ref, device=dev)
    out["by_j"][str(j)] = {k: v for k, v in r.items() if not k.endswith(("per_group", "per_seed"))}
    print(f"j={j:2d} " + " ".join(f"{k} {v:.3f}" for k, v in out["by_j"][str(j)].items() if isinstance(v, float)),
          f"| {time.time() - t0:.0f}s", flush=True)
out["wall_s"] = time.time() - t0
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "diag_tf_levels_released_ts.json"), "w"), indent=1)
