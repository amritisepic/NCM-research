"""Phase 0: reproduce the released NCM Sudoku solve rates on CPU, and log per-level trajectories.

Mirrors upstream scripts/eval_shard.py + eval_merge.py:exact() for the sudoku configs (greedy,
eager path, fp32), and additionally records, for every puzzle and every level k, the board decoded
from the truncated hierarchy (levels 0..k), the decoder's per-cell confidence, and the generator's
per-level code entropy. Because decoding is greedy and levels are generated in order, the board
decoded at level k is exactly what an early exit after level k would output, so one full run gives
the whole adaptive-compute curve.

Usage: python scripts/repro_sudoku.py --level hard [--n 2000] [--bs 128]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.ncm_ref_loader import ForwardCounter, load_ncm, safe_pickle_load  # noqa: E402
from ncmr.ncm_tools import zhat_snapshots  # noqa: E402
from ncmr.sudoku import cells, is_solution, is_valid_grid, propagation_rounds  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--level", required=True, choices=["easy", "medium", "hard"])
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
torch.manual_seed(0)

ck = os.path.join(ROOT, "artifacts", "sudoku", a.level)
dd = os.path.join(ROOT, "data", "sudoku", a.level)
B = load_ncm(ck, device="cpu")
net, vq = B.net, B.vqvae
G = B.cfg.generation
assert float(G.temperature) == 0.0 and float(G.cfg_scale) == 0.0
P = int(B.cfg.model.prefix_len)
n_levels = len(vq.vq.levels)

puzzles = safe_pickle_load(os.path.join(dd, "puzzles.pickle"))[: a.n]
solutions = safe_pickle_load(os.path.join(dd, "solutions.pickle"))[: a.n]

# Record the generator's code distributions per level (entropy, max prob) without touching upstream code.
_records = []
_orig_sample = net._sample_level_raw


def _recording_sample(logits, temperature, top_k, top_p):
    lp = F.log_softmax(logits.float(), dim=-1)
    ent = -(lp.exp() * lp).sum(-1)  # (B, l_k)
    _records.append((ent.mean(-1), ent.max(-1).values, lp.max(-1).values.exp().min(-1).values))
    return _orig_sample(logits, temperature, top_k, top_p)


net._sample_level_raw = _recording_sample

counter = ForwardCounter(net)
per_level_boards = np.zeros((len(puzzles), n_levels, 81), dtype=np.uint8)
per_level_conf_min = np.zeros((len(puzzles), n_levels), dtype=np.float32)
per_level_conf_mean = np.zeros((len(puzzles), n_levels), dtype=np.float32)
code_ent_mean = np.zeros((len(puzzles), n_levels), dtype=np.float32)
code_ent_max = np.zeros((len(puzzles), n_levels), dtype=np.float32)
code_pmax_min = np.zeros((len(puzzles), n_levels), dtype=np.float32)
gen_time = 0.0
nfe_per_call = []

t_all = time.time()
for s in range(0, len(puzzles), a.bs):
    batch = puzzles[s:s + a.bs]
    x = torch.tensor([list(p)[:P] for p in batch], dtype=torch.long)
    _records.clear()
    c0 = counter.n
    t0 = time.time()
    with torch.no_grad():
        g = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=None)
        _, zl = vq.vq.idx_to_zhats(g, return_zlevels=True)
        logits_final = vq.decoder(zl[-1])
    gen_time += time.time() - t0  # generation + final VQ decode, as in the paper's timing protocol
    nfe_per_call.append(counter.n - c0)
    with torch.no_grad():
        snaps = zhat_snapshots(vq.vq, g)  # full-resolution truncated latents (what early exit would decode)
        assert torch.equal(snaps[-1], zl[-1])
        for k in range(n_levels):
            lg = logits_final if k == n_levels - 1 else vq.decoder(snaps[k])
            pr = lg.float().softmax(-1)
            ids = pr.argmax(-1)  # (b, 89)
            conf = pr.max(-1).values
            cell_mask = torch.tensor([(i + 1) % 10 != 0 for i in range(ids.shape[1])])
            for j in range(ids.shape[0]):
                c = cells(ids[j].tolist())
                per_level_boards[s + j, k] = np.array(c if c is not None else [0] * 81, dtype=np.uint8)
            per_level_conf_min[s:s + len(batch), k] = conf[:, cell_mask].min(-1).values.numpy()
            per_level_conf_mean[s:s + len(batch), k] = conf[:, cell_mask].mean(-1).numpy()
    assert len(_records) == n_levels, len(_records)
    for k, (em, ex, pm) in enumerate(_records):
        code_ent_mean[s:s + len(batch), k] = em.numpy()
        code_ent_max[s:s + len(batch), k] = ex.numpy()
        code_pmax_min[s:s + len(batch), k] = pm.numpy()
    print(f"  {s + len(batch)}/{len(puzzles)}  gen {gen_time:.1f}s", flush=True)
counter.close()

# Metrics, matching upstream eval_merge.exact(): exact string match of decoded 89-token outputs.
P_cells = [cells(p) for p in puzzles]
S_cells = [cells(sol) for sol in solutions]
final = per_level_boards[:, -1]
solved = np.array([list(final[i]) == S_cells[i] for i in range(len(puzzles))])
cell_acc = np.array([np.mean(final[i] == np.array(S_cells[i])) for i in range(len(puzzles))])
valid = np.array([is_valid_grid(list(final[i])) for i in range(len(puzzles))])
solved_by_level = np.array([[list(per_level_boards[i, k]) == S_cells[i] for k in range(n_levels)]
                            for i in range(len(puzzles))])
verifiable_by_level = np.array([[is_solution(list(per_level_boards[i, k]), P_cells[i]) for k in range(n_levels)]
                                for i in range(len(puzzles))])
prop = [propagation_rounds(p) for p in P_cells]

summary = {
    "run": f"phase0_sudoku_repro_{a.level}",
    "task": f"sudoku-{a.level}",
    "n": len(puzzles),
    "solved_pct": 100 * solved.mean(),
    "cell_pct": 100 * cell_acc.mean(),
    "valid_pct": 100 * valid.mean(),
    "paper_solved_pct": {"easy": 99.5, "medium": 96.5, "hard": 86.1}[a.level],
    "independent_repro_solved_pct_velaia": {"easy": 99.4, "medium": 96.4, "hard": 86.3}[a.level],
    "nfe_per_sample": float(np.mean(nfe_per_call)),
    "wallclock_gen_s": gen_time,
    "wallclock_total_s": time.time() - t_all,
    "batch_size": a.bs,
    "precision": "fp32 eager (no KV cache)",
    "device": f"cpu ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)",
    "torch": torch.__version__,
    "solved_by_level_pct": (100 * solved_by_level.mean(0)).tolist(),
    "verifiable_by_level_pct": (100 * verifiable_by_level.mean(0)).tolist(),
    "cell_acc_by_level_pct": [100 * float(np.mean([np.mean(per_level_boards[i, k] == np.array(S_cells[i]))
                                                   for i in range(len(puzzles))])) for k in range(n_levels)],
    "solved_vs_propagation": {
        "prop_solvable_n": int(sum(ok for _, ok in prop)),
        "solved_and_prop_solvable": int(sum(bool(solved[i]) and prop[i][1] for i in range(len(puzzles)))),
        "solved_not_prop_solvable": int(sum(bool(solved[i]) and not prop[i][1] for i in range(len(puzzles)))),
        "failed_and_prop_solvable": int(sum((not solved[i]) and prop[i][1] for i in range(len(puzzles)))),
    },
}
out = a.out or os.path.join(ROOT, "results", f"phase0_sudoku_repro_{a.level}.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=1)
np.savez_compressed(out.replace(".json", "_trajectories.npz"),
                    boards=per_level_boards, conf_min=per_level_conf_min, conf_mean=per_level_conf_mean,
                    code_ent_mean=code_ent_mean, code_ent_max=code_ent_max, code_pmax_min=code_pmax_min,
                    solved=solved, prop_rounds=np.array([r for r, _ in prop]),
                    prop_solvable=np.array([ok for _, ok in prop]))
print(json.dumps({k: v for k, v in summary.items() if not k.endswith("by_level_pct")}, indent=1))
print("solved by level:", [round(v, 1) for v in summary["solved_by_level_pct"]])
