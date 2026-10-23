"""Released NCM TinyStories generator: (1) teacher-forced per-level NLL on val codes; (2) unconditional
samples with the paper's protocol (5 seeds x 256, temperature 1, level 0 from the stored unigram), saved
for the evaluation suite.

Usage: python scripts/released_generator_eval.py nll --n 1000
       python scripts/released_generator_eval.py sample --seeds 42 0 1 2 3 --n 256
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.data import EOS, PAD, examples  # noqa: E402
from ncmr.ncm_ref_loader import ForwardCounter, load_ncm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["nll", "sample"])
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--seeds", type=int, nargs="*", default=[42, 0, 1, 2, 3])
a = ap.parse_args()

B = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"))
net, vqvae, tok = B.net, B.vqvae, B.pre_tokenizer
levels = [int(l) for l in vqvae.vq.levels]

if a.mode == "nll":
    X = torch.from_numpy(examples("val", a.n))
    tot = torch.zeros(len(levels) - 1, dtype=torch.float64)
    ent = torch.zeros(len(levels) - 1, dtype=torch.float64)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(X), a.bs):
            xb = X[s:s + a.bs]
            z_in, targets = vqvae.get_ncm_inputs(xb)
            z_prefix = vqvae.encoder.wte(torch.full((xb.shape[0], 1), EOS))
            logits = net(torch.cat([z_prefix, z_in], 1)).float()
            lp = F.log_softmax(logits, -1)
            nll = -lp.gather(-1, targets[..., None])[..., 0]
            h = -(lp.exp() * lp).sum(-1)
            for k, c in enumerate(torch.split(nll, levels[1:], 1)):
                tot[k] += c.sum().item()
            for k, c in enumerate(torch.split(h, levels[1:], 1)):
                ent[k] += c.sum().item()
            print(f"  {s + xb.shape[0]}/{len(X)} {time.time() - t0:.0f}s", flush=True)
    per_level = (tot / len(X)).tolist()
    res = {"run": "phase0_released_generator_nll", "n": len(X), "split": "val", "levels": levels,
           "nll_nats_per_seq_per_level": per_level, "nll_nats_per_seq_total_levels1plus": float(sum(per_level)),
           "pred_entropy_nats_per_seq_per_level": (ent / len(X)).tolist(),
           "nll_nats_per_code_per_level": [v / l for v, l in zip(per_level, levels[1:])],
           "note": "level 0 (1 code) is sampled from a unigram at generation time; not included here",
           "wallclock_s": time.time() - t0}
    out = os.path.join(ROOT, "results", "phase0_released_generator_nll.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in res.items()}, indent=1))
else:
    L0 = B.ckpt_extra["inference"]["level0_unigram"].float()
    counter = ForwardCounter(net)
    rows, groups, t_gen = [], [], 0.0
    for seed in a.seeds:
        torch.manual_seed(seed)
        for s in range(0, a.n, a.bs):
            b = min(a.bs, a.n - s)
            l0 = torch.multinomial(L0, b * levels[0], replacement=True).view(b, levels[0])
            t0 = time.time()
            with torch.no_grad():
                g = net.generate(l0, torch.full((b, 1), EOS), temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0,
                                 cfg_null_eos=EOS)
                _, zl = vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                ids = vqvae.decoder(zl[-1]).argmax(-1)
            t_gen += time.time() - t0
            for r in ids.tolist():
                r = r[:r.index(EOS)] if EOS in r else r
                rows.append([t for t in r if t not in (EOS, PAD)])
                groups.append(seed)
        print(f"  seed {seed}: done ({t_gen:.0f}s gen)", flush=True)
    nfe = counter.n / (len(a.seeds) * ((a.n + a.bs - 1) // a.bs))
    out = os.path.join(ROOT, "runs", "samples", "released_ncm_tinystories.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"ids": rows, "groups": groups, "nfe_per_sample": nfe, "gen_wall_s": t_gen,
               "protocol": "temperature 1, top_k 0, top_p 1, level0 ~ stored unigram, argmax VQ decode, cut at EOS"},
              open(out, "w"))
    print(f"saved {len(rows)} samples, NFE/sample {nfe}, gen wall {t_gen:.0f}s -> {out}")
