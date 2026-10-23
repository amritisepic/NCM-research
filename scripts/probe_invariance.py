"""H2 test: are NCM's coarse "concept" codes bags of words?

Released TinyStories tokenizer, N val story openings. For each perturbation of the 128 input tokens we
re-encode and report, per level, the fraction of codes that stay identical:
  shuffle_w16  : permute tokens inside each aligned 16-token window (destroys word order / meaning
                 inside the window; keeps the window's bag of tokens). Level 1 has exactly these windows.
  shuffle_w8   : same with 8-token windows (level 2's windows).
  swap_across  : swap two random tokens that sit in different 16-token windows (changes 2 windows' bags).
  synonym      : replace one word by a near-synonym (meaning-preserving; list below); changes 1-2 tokens.
  name_swap    : replace a character name by another common name (meaning-preserving up to identity).
  random_tok   : replace one random non-special token by a random frequent token (meaning-damaging).
A semantic hierarchy would keep coarse codes under synonym/name swaps and change them under shuffles.
A bag-of-words hierarchy does the opposite at the level whose windows are shuffled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.data import EOS, PAD, examples  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm  # noqa: E402
from ncmr.ncm_tools import level_slices, z_to_idx_forced  # noqa: E402

SYNONYMS = [(" big", " large"), (" little", " small"), (" happy", " glad"), (" scared", " afraid"),
            (" mom", " mother"), (" dad", " father"), (" quickly", " fast"), (" said", " replied"),
            (" pretty", " lovely"), (" sad", " unhappy"), (" yelled", " shouted"), (" looked", " glanced"),
            (" toy", " plaything"), (" jumped", " leaped"), (" walked", " strolled")]
NAMES = [" Lily", " Tom", " Ben", " Sue", " Timmy", " Max", " Mia", " Anna", " Sara", " Jack", " Lucy", " Sam"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tokenizer", default="released", help="released, or path to our final.pt")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "phase0_invariance_probe.json"))
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

from ncmr.models import load_text_tokenizer  # noqa: E402
vqvae, tok = load_text_tokenizer(a.tokenizer)
vq = vqvae.vq
levels = [int(l) for l in vq.levels]
sl = level_slices(levels)
X = examples("val", a.n)


def enc(xs: np.ndarray, forced: np.ndarray | None = None, n_forced: int = 0) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(xs), 250):
            z = vqvae._encode_pre_quant(torch.from_numpy(xs[s:s + 250]))
            if n_forced:
                out.append(z_to_idx_forced(vq, z, torch.from_numpy(forced[s:s + 250]), n_forced).numpy())
            else:
                out.append(vq.z_to_idx(z).numpy())
    return np.concatenate(out)


def single_token_id(word: str):
    ids = tok.encode(word)
    return ids[0] if len(ids) == 1 else None


syn_pairs = [(single_token_id(u), single_token_id(v)) for u, v in SYNONYMS]
syn_map = {}
for u, v in syn_pairs:
    if u is not None and v is not None:
        syn_map[u] = v
        syn_map[v] = u
name_ids = [i for i in (single_token_id(n) for n in NAMES) if i is not None]
flat = X[(X != PAD) & (X != EOS)]
freq_ids = np.bincount(flat, minlength=8192).argsort()[::-1][:500]


def real_positions(row):
    return np.where((row != PAD) & (row != EOS))[0]


def perturb(kind: str, row: np.ndarray):
    r = row.copy()
    pos = real_positions(r)
    if kind.startswith("shuffle_w"):
        w = int(kind[len("shuffle_w"):])
        for s in range(0, 128, w):
            win = [p for p in range(s, s + w) if p in set(pos)]
            if len(win) > 1:
                r[win] = r[rng.permutation(win)]
        return r, int((r != row).sum())
    if kind == "swap_across":
        for _ in range(50):
            i, j = rng.choice(pos, 2, replace=False)
            if i // 16 != j // 16 and r[i] != r[j]:
                r[i], r[j] = r[j], r[i]
                return r, 2
        return None, 0
    if kind == "synonym":
        cand = [p for p in pos if r[p] in syn_map]
        if not cand:
            return None, 0
        p = rng.choice(cand)
        r[p] = syn_map[r[p]]
        EDIT_POS.append(int(p))
        return r, 1
    if kind == "name_swap":
        cand = [p for p in pos if r[p] in name_ids]
        if not cand:
            return None, 0
        old = r[cand[0]]
        new = rng.choice([n for n in name_ids if n != old])
        r[r == old] = new  # rename the character consistently
        return r, int((r != row).sum())
    if kind == "random_tok":
        p = rng.choice(pos)
        r[p] = rng.choice([t for t in freq_ids[:200] if t != r[p]])
        EDIT_POS.append(int(p))
        return r, 1
    raise ValueError(kind)


base_codes = enc(X)
res = {"run": "phase0_invariance_probe", "tokenizer": a.tokenizer, "n": int(a.n), "seed": a.seed, "levels": levels,
       "kinds": {}, "conditional": {}}
for kind in ["shuffle_w16", "shuffle_w8", "swap_across", "synonym", "name_swap", "random_tok"]:
    EDIT_POS = []
    rows, keep, nchg = [], [], []
    for i, row in enumerate(X):
        r, c = perturb(kind, row)
        if r is not None and c > 0:
            rows.append(r)
            keep.append(i)
            nchg.append(c)
    if not rows:
        continue
    codes = enc(np.stack(rows))
    same = codes == base_codes[keep]
    per_level = [float(same[:, s].mean()) for s in sl]
    # Window-local view for level 1/2: fraction of untouched codes among windows whose bag changed vs not
    res["kinds"][kind] = {"n_examples": len(rows), "mean_tokens_changed": float(np.mean(nchg)),
                          "frac_codes_unchanged_per_level": per_level}
    print(f"{kind:12s} n={len(rows):4d} tok_changed={np.mean(nchg):5.1f} | " +
          " ".join(f"L{k}:{100 * v:5.1f}" for k, v in enumerate(per_level)), flush=True)
    # Conditional view: re-encode with all coarser levels forced to the original codes, so level k's
    # change is caused by the input edit alone. For single-token edits also report the affected window only.
    cond = {}
    base_k = base_codes[keep]
    for k in (1, 2, 3, 4):
        ck = enc(np.stack(rows), base_k, n_forced=k)[:, sl[k]]
        same_k = ck == base_k[:, sl[k]]
        entry = {"frac_unchanged": float(same_k.mean())}
        if EDIT_POS and len(EDIT_POS) == len(rows):
            w = [min(levels[k] - 1, p * levels[k] // 128) for p in EDIT_POS]
            entry["affected_window_unchanged"] = float(np.mean([same_k[i, w[i]] for i in range(len(rows))]))
        cond[f"L{k}"] = entry
    res["conditional"][kind] = cond
    print("    conditional (coarser levels forced): " + " ".join(
        f"L{k}: {100 * v['frac_unchanged']:5.1f}" + (f" (affected window {100 * v['affected_window_unchanged']:5.1f})"
                                                     if "affected_window_unchanged" in v else "")
        for k, v in ((int(kk[1:]), vv) for kk, vv in cond.items())), flush=True)

with open(a.out, "w") as f:
    json.dump(res, f, indent=1)
