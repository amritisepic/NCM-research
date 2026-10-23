"""B2: fixed-point (Jacobi) decoding across NCM levels on the released Sudoku checkpoints (greedy).

Sequential NCM decoding computes level k from the codes of levels < k, one level per pass. Jacobi decoding
instead updates *all* levels in one parallel pass (the training-style block-diagonal forward) from the
current guess of every level, and repeats. Because level k depends only on levels < k, the sequential
greedy solution is the unique fixed point and level k is exact after k+1 passes, so the iteration
converges in at most K passes. Stopping at the first pass that changes nothing returns exactly the
sequential output (zero quality change). NFE = passes until no change (the last pass only verifies).

Reports, per puzzle: passes to the fixed point (exact stopping rule, includes the verifying pass), the
first pass whose decoded board already equals the final board (oracle), and FLOP cost relative to
sequential decoding. Init: all codes 0 ("empty" hierarchy) or the previous puzzle's... (only 'zeros').

Usage: python scripts/jacobi_levels.py --level hard --n 500
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

ap = argparse.ArgumentParser()
ap.add_argument("--level", default="hard")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--bs", type=int, default=100)
a = ap.parse_args()

B = load_ncm(os.path.join(ROOT, "artifacts", "sudoku", a.level))
net, vq = B.net, B.vqvae
P = net.prefix_len
levels = [int(l) for l in vq.vq.levels]
K, S = len(levels), sum(levels)
puzzles = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", a.level, "puzzles.pickle"))[: a.n]
solutions = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", a.level, "solutions.pickle"))[: a.n]


@torch.no_grad()
def jacobi(x):
    b = x.shape[0]
    z_prefix = vq.encoder.wte(x)
    q0 = net.build_l0_query_input(b)
    codes = torch.zeros(b, S, dtype=torch.long)
    hist = []
    for t in range(K + 1):
        z_in = vq.vq.idx_to_ncm_input(codes)
        logits = net(torch.cat([z_prefix, q0, z_in], 1))
        new = logits.argmax(-1)
        hist.append(new)
        if torch.equal(new, codes):
            break
        codes = new
    return hist


t0 = time.time()
passes, oracle, solved, seq_equal = [], [], [], []
for s in range(0, len(puzzles), a.bs):
    x = torch.tensor([list(p) for p in puzzles[s:s + a.bs]])
    with torch.no_grad():
        seq = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=None)
    hist = jacobi(x)
    final = hist[-1]
    boards = [vq.decoder(vq.vq.idx_to_z_q(h)).argmax(-1) for h in hist]
    for j in range(x.shape[0]):
        # passes until this puzzle's codes stop changing (+1 verifying pass), per puzzle
        conv = next(t for t in range(len(hist)) if torch.equal(hist[t][j], final[j]))
        passes.append(min(conv + 2, len(hist)))  # pass index conv produced the final codes; next pass verifies
        ob = next(t for t in range(len(boards)) if torch.equal(boards[t][j], boards[-1][j]))
        oracle.append(ob + 1)
        seq_equal.append(bool(torch.equal(final[j], seq[j])))
        solved.append(cells(boards[-1][j].tolist()) == cells(list(solutions[s + j])))
    print(f"  {s + x.shape[0]}/{len(puzzles)}  mean passes {np.mean(passes):.2f}  {time.time() - t0:.0f}s", flush=True)

pos_seq = K * P + S                     # positions processed by sequential decoding (eager, prefix per level)
pos_jac = np.mean(passes) * (P + S)     # positions processed by Jacobi
res = {"run": f"b2_jacobi_levels_{a.level}", "n": len(puzzles), "K": K,
       "fixed_point_equals_sequential_pct": 100 * float(np.mean(seq_equal)),
       "solved_pct": 100 * float(np.mean(solved)),
       "passes_exact_stop_mean": float(np.mean(passes)), "passes_exact_stop_hist": np.bincount(passes).tolist(),
       "passes_oracle_board_mean": float(np.mean(oracle)),
       "nfe_saving_vs_sequential_pct": 100 * (1 - float(np.mean(passes)) / K),
       "flops_ratio_vs_sequential": float(pos_jac / pos_seq), "wall_s": time.time() - t0}
json.dump(res, open(os.path.join(ROOT, "results", f"{res['run']}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
