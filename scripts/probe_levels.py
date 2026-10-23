"""Linear probes on NCM level codes (released TinyStories tokenizer): what does each level encode?

Features for level k (per window): mean of the level-k codebook vectors of the window's codes, and (for
the cumulative view) the full-resolution truncated latent z_hat_k mean-pooled. Probes (logistic
regression, 5-fold CV accuracy, train on val windows only):
  main_char  : most frequent character name in the window (top-8 names; windows without one excluded)
  dialogue   : window contains a quotation mark
  content    : 8 separate binary probes, "word w occurs in the window" (w in dog, cat, ball, park, tree,
               bird, toy, cake); reported as mean balanced accuracy
  tok@64     : identity of the token at position 64 among the 20 most frequent tokens there (local info)
  order      : is the window's first half before its second half? (the window vs. its halves swapped;
               a bag-of-words summary of the whole window cannot tell, a positional one can)
Chance levels are reported (majority class / 0.5 balanced).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.data import EOS, cut_at_eos, examples  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm  # noqa: E402
from ncmr.ncm_tools import level_slices, zhat_snapshots  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tokenizer", default="released", help="released, or path to our final.pt")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "phase0_level_probes.json"))
a = ap.parse_args()

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ncmr.models import load_text_tokenizer  # noqa: E402
vqvae, tok = load_text_tokenizer(a.tokenizer)
vq = vqvae.vq
levels = [int(l) for l in vq.levels]
sl = level_slices(levels)
X = examples("val", a.n)
# order probe: second copy of each window with halves swapped
Xswap = np.concatenate([X[:, 64:], X[:, :64]], 1)


def feats(xs):
    code_feats, cum_feats = [[] for _ in levels], [[] for _ in levels]
    with torch.no_grad():
        for s in range(0, len(xs), 250):
            xb = torch.from_numpy(xs[s:s + 250])
            idx = vq.z_to_idx(vqvae._encode_pre_quant(xb))
            snaps = zhat_snapshots(vq, idx)
            for k in range(len(levels)):
                code_feats[k].append(vq.embeddings[k](idx[:, sl[k]]).mean(1).numpy())
                cum_feats[k].append(snaps[k].mean(1).numpy())
    return [np.concatenate(f) for f in code_feats], [np.concatenate(f) for f in cum_feats]


codeF, cumF = feats(X)
_, cumF_swap = feats(Xswap)
texts = [tok.decode([t for t in cut_at_eos(r)]) for r in X]
words = [Counter(w.strip('.,!?"\'').lower() for w in t.split()) for t in texts]

NAMES = ["lily", "tom", "ben", "sue", "timmy", "max", "mia", "anna", "sara", "jack", "lucy", "sam", "tim", "bob"]
name_counts = Counter()
for t in texts:
    for w in t.split():
        if w.strip('.,!?"\'').lower() in NAMES:
            name_counts[w.strip('.,!?"\'').lower()] += 1
top_names = [n for n, _ in name_counts.most_common(8)]
main_char = []
for wc in words:
    c = [(wc[n], n) for n in top_names if wc[n] > 0]
    main_char.append(max(c)[1] if c else None)
dialogue = np.array(['"' in t for t in texts])
CONTENT = ["dog", "cat", "ball", "park", "tree", "bird", "toy", "cake"]
tok64 = X[:, 64]
top64 = [t for t, _ in Counter(tok64.tolist()).most_common(20)]


def cv(F, y, balanced=False):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0,
                                                             class_weight="balanced" if balanced else None))
    return float(cross_val_score(clf, F, y, cv=5, scoring="balanced_accuracy" if balanced else "accuracy").mean())


res = {"run": "phase0_level_probes", "n": int(a.n), "levels": levels, "top_names": top_names, "probes": {}}
mc_idx = np.array([i for i, m in enumerate(main_char) if m is not None])
mc_y = np.array([top_names.index(main_char[i]) for i in mc_idx])
t64_idx = np.array([i for i, t in enumerate(tok64) if t in top64])
t64_y = np.array([top64.index(tok64[i]) for i in t64_idx])
res["chance"] = {"main_char": float(np.bincount(mc_y).max() / len(mc_y)), "dialogue_balanced": 0.5,
                 "content_balanced": 0.5, "tok@64": float(np.bincount(t64_y).max() / len(t64_y)), "order_balanced": 0.5}
order_y = np.concatenate([np.zeros(len(X)), np.ones(len(X))])
for k in range(len(levels)):
    r = {}
    for name, F in [("code", codeF[k]), ("cum", cumF[k])]:
        r[f"main_char_{name}"] = cv(F[mc_idx], mc_y)
        r[f"dialogue_{name}"] = cv(F, dialogue, balanced=True)
        r[f"content_{name}"] = float(np.mean([cv(F, np.array([wc[w] > 0 for wc in words]), balanced=True) for w in CONTENT]))
        r[f"tok64_{name}"] = cv(F[t64_idx], t64_y)
    r["order_cum"] = cv(np.concatenate([cumF[k], cumF_swap[k]]), order_y, balanced=True)
    res["probes"][f"L{k}"] = r
    print(f"L{k:2d} len {levels[k]:3d} | " + " ".join(f"{kk} {v:.3f}" for kk, v in r.items()), flush=True)
print("chance:", res["chance"])
json.dump(res, open(a.out, "w"), indent=1)
