"""Pre-check for full-resolution repair levels on the released Sudoku generator (B6/C5, tier L). CPU, no training.

A repair level appends one more residual level at full resolution (89 positions) after the 12-level hierarchy,
reusing the finest level's codebook and quant-resi map. This script measures the *representational* ceiling:
starting from the released model's greedy hierarchy, an oracle picks each repair level's codes as the nearest
codes to the true residual (z_true - z_used, exactly how the tokenizer encodes a residual), and we decode after
every repair level. If the oracle cannot fix failed boards in a few levels, no trained repair pass can.
Also reports the released model's solve rate on our generated val puzzles (distribution check for fine-tuning).

Usage: python scripts/sudoku_repair_ceiling.py --n 1000 [--repairs 8] [--set val|test]
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

from ncmr.ncm_ref_loader import load_ncm, safe_pickle_load  # noqa: E402
from ncmr.sudoku import cells  # noqa: E402
from ncmr.sudoku_data import SudokuTrain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--repairs", type=int, default=8)
ap.add_argument("--bs", type=int, default=250)
ap.add_argument("--set", default="val", choices=["val", "test"])
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
torch.manual_seed(0)

B = load_ncm(os.path.join(ROOT, "artifacts", "sudoku", "hard"))
net, tok = B.net, B.vqvae
vq = tok.vq
K = len(vq.levels)
E = vq.embeddings[K - 1].weight
if a.set == "val":
    prompts, sols = SudokuTrain("hard", os.path.join(ROOT, "assets", "sudoku_train", "hard.npz")).val(a.n)
else:
    prompts = np.array(safe_pickle_load(os.path.join(ROOT, "data", "sudoku", "hard", "puzzles.pickle"))[: a.n])
    sols = np.array(safe_pickle_load(os.path.join(ROOT, "data", "sudoku", "hard", "solutions.pickle"))[: a.n])


def board(z):
    return [cells(r) for r in tok.decoder(z).argmax(-1).tolist()]


t0 = time.time()
solved = np.zeros((len(prompts), a.repairs + 1), dtype=bool)
canon_dist, used_dist = [], []
for s in range(0, len(prompts), a.bs):
    p = torch.as_tensor(prompts[s:s + a.bs])
    y = torch.as_tensor(sols[s:s + a.bs])
    truth = [cells(r) for r in y.tolist()]
    with torch.no_grad():
        g = net.generate(None, p, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=None)
        z_true = tok._encode_pre_quant(y).float()
        z_used = vq.idx_to_z_q(g)
        for t in range(a.repairs + 1):
            if t > 0:
                r = z_true - z_used
                c = vq._distances(r.reshape(-1, r.shape[-1]), E).argmin(-1).view(r.shape[0], r.shape[1])
                z_used = z_used + vq._apply_quant_resi(E[c], K - 1)
            solved[s:s + len(p), t] = [b == tt for b, tt in zip(board(z_used), truth)]
        canon_dist.append((z_true - vq.idx_to_z_q(vq.z_to_idx(z_true))).norm(dim=-1).mean().item())
        used_dist.append((z_true - vq.idx_to_z_q(g)).norm(dim=-1).mean().item())
    print(f"  {s + len(p)}/{len(prompts)}  solved {100 * solved[:s + len(p), 0].mean():.1f}%  {time.time() - t0:.0f}s", flush=True)

fail = ~solved[:, 0]
res = {"run": f"sudoku_repair_ceiling_{a.set}", "n": len(prompts), "set": a.set,
       "released_solved_pct": 100 * float(solved[:, 0].mean()),
       "oracle_solved_pct_after_t_repairs": (100 * solved.mean(0)).round(2).tolist(),
       "oracle_fixed_pct_of_failures_after_t": (100 * solved[fail].mean(0)).round(2).tolist() if fail.any() else None,
       "n_failures": int(fail.sum()),
       "residual_norm_canonical": float(np.mean(canon_dist)), "residual_norm_generated": float(np.mean(used_dist)),
       "wall_s": time.time() - t0}
json.dump(res, open(os.path.join(ROOT, "results", f"{res['run']}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
