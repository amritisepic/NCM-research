"""C10 / H1: does constraint-propagation depth predict NCM's per-puzzle success? Also per-level decodes.

Reads results/phase0_sudoku_repro_<level>_trajectories.npz (from repro_sudoku.py) and writes
results/phase0_sudoku_depth_analysis.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


out = {}
for lvl in ["easy", "medium", "hard"]:
    f = os.path.join(ROOT, "results", f"phase0_sudoku_repro_{lvl}_trajectories.npz")
    if not os.path.exists(f):
        continue
    T = np.load(f)
    solved, rounds, ok = T["solved"].astype(bool), T["prop_rounds"], T["prop_solvable"].astype(bool)
    rows = []
    for lo, hi in [(0, 4), (5, 6), (7, 8), (9, 10), (11, 99)]:
        m = ok & (rounds >= lo) & (rounds <= hi)
        rows.append({"rounds": f"{lo}-{hi}", "n": int(m.sum()), "solved_pct": round(100 * solved[m].mean(), 1) if m.any() else None,
                     "ci95": wilson(int(solved[m].sum()), int(m.sum()))})
    m = ~ok
    rows.append({"rounds": "needs search", "n": int(m.sum()), "solved_pct": round(100 * solved[m].mean(), 1) if m.any() else None,
                 "ci95": wilson(int(solved[m].sum()), int(m.sum()))})
    # logistic-free summary: Spearman between rounds and failure among propagation-solvable puzzles
    from scipy.stats import spearmanr, mannwhitneyu
    rho = spearmanr(rounds[ok], (~solved[ok]).astype(float))
    # does the generator's uncertainty flag failures? (per-puzzle max code entropy at the final level)
    ent_final = T["code_ent_max"][:, -1]
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(~solved, ent_final))
        auc_conf = float(roc_auc_score(~solved, -T["conf_min"][:, -1]))
    except Exception:
        auc_conf = None
    out[lvl] = {"n": int(len(solved)), "solved_pct": round(100 * solved.mean(), 2),
                "by_propagation_depth": rows,
                "spearman_rounds_vs_failure_prop_solvable": [float(rho.correlation), float(rho.pvalue)],
                "auc_failure_from_final_level_max_code_entropy": auc,
                "auc_failure_from_decoder_min_cell_confidence": auc_conf}
    print(lvl, json.dumps(out[lvl], indent=None))
json.dump(out, open(os.path.join(ROOT, "results", "phase0_sudoku_depth_analysis.json"), "w"), indent=1)
