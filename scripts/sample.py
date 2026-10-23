"""Draw unconditional samples from one of our trained models with the NCM paper protocol.

  python scripts/sample.py --run_dir runs/ncm_s --out runs/samples/ncm_s.json [--seeds 42 0 1 2 3 --n 256]

NCM: level 0 ~ stored unigram, levels 1..K-1 sampled at temperature 1 (upstream `generate`, eager),
final VQ decode by argmax, cut at EOS. AR: ancestral sampling at temperature 1 for 128 tokens, cut at EOS.
Records NFE (generator forward passes per sample) and generation wall-clock.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.data import EOS, PAD, cut_at_eos  # noqa: E402
from ncmr.models import ARTransformer, build_ncm_generator, build_tokenizer  # noqa: E402
from ncmr.ncm_ref_loader import ForwardCounter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, nargs="*", default=[42, 0, 1, 2, 3])
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--ckpt", default="final.pt", help="file in run_dir (e.g. snap_15000.pt for an AR run)")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
dev = "cuda" if torch.cuda.is_available() else "cpu"
st = torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu", weights_only=False)  # our own file

rows, groups, t_gen, nfe_calls, n_calls = [], [], 0.0, 0, 0
if "gen_kwargs" in st:
    tst = torch.load(st["tokenizer_path"], map_location="cpu", weights_only=False)
    tok = build_tokenizer(**tst["tok_kwargs"])
    tok.load_state_dict(tst["model"])
    tok = tok.to(dev).eval()
    net = build_ncm_generator(tok, **st["gen_kwargs"]).to(dev)
    net.load_state_dict(st["model"])
    net.eval()
    L0 = st["level0_unigram"].float()
    counter = ForwardCounter(net)
    l0len = tok.vq.levels[0]
    for seed in a.seeds:
        torch.manual_seed(seed)
        for s in range(0, a.n, a.bs):
            b = min(a.bs, a.n - s)
            l0 = torch.multinomial(L0, b * l0len, replacement=True).view(b, l0len).to(dev)
            c0 = counter.n
            t0 = time.time()
            with torch.no_grad():
                g = net.generate(l0, torch.full((b, 1), EOS, device=dev), temperature=a.temperature, top_k=0,
                                 top_p=1.0, cfg_scale=0.0, cfg_null_eos=EOS)
                ids = tok.decoder(tok.vq.idx_to_z_q(g)).argmax(-1)
            t_gen += time.time() - t0
            nfe_calls += counter.n - c0
            n_calls += 1
            for r in ids.tolist():
                rows.append([t for t in cut_at_eos(r) if t != PAD])
                groups.append(seed)
    kind = "ncm"
else:
    net = ARTransformer(**st["ar_kwargs"]).to(dev)
    net.load_state_dict(st["model"])
    net.eval()
    counter = ForwardCounter(type("N", (), {"transformer": type("T", (), {"blocks": net.blocks})})())
    for seed in a.seeds:
        gen = torch.Generator(device=dev).manual_seed(seed)
        for s in range(0, a.n, a.bs):
            b = min(a.bs, a.n - s)
            c0 = counter.n
            t0 = time.time()
            ids = net.generate(b, temperature=a.temperature, device=dev, generator=gen)
            t_gen += time.time() - t0
            nfe_calls += counter.n - c0
            n_calls += 1
            for r in ids.tolist():
                rows.append([t for t in cut_at_eos(r) if t != PAD])
                groups.append(seed)
    kind = "ar"
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({"ids": rows, "groups": groups, "kind": kind, "run_dir": a.run_dir, "nfe_per_sample": nfe_calls / n_calls,
           "gen_wall_s": t_gen, "device": dev, "temperature": a.temperature}, open(a.out, "w"))
print(f"{kind}: {len(rows)} samples, NFE/sample {nfe_calls / n_calls:.1f}, gen wall {t_gen:.0f}s -> {a.out}")
