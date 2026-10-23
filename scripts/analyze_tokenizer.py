"""Phase 0 harness: what does each level of the released NCM TinyStories tokenizer carry?

For N validation story openings (our fixed `val` split), report per level k:
  - token accuracy of decoding the full-resolution truncated latent (levels 0..k), overall and per
    token class (function words, nouns, names, numerals, verbs, modifiers, punctuation, rare words);
  - code usage (distinct codes, code-distribution entropy);
  - re-encoding consistency: encode(decode(codes)) == codes, per level (a fixed-point check).
Also prints a few native-resolution level decodes (the "coarse-to-fine" view), for random examples
with the seed stated.

Usage: python scripts/analyze_tokenizer.py [--n 2000] [--out results/phase0_tokenizer_analysis.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.data import TrainStream, examples  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm  # noqa: E402
from ncmr.ncm_tools import level_slices, zhat_snapshots  # noqa: E402
from ncmr.token_classes import CLASSES, classify, load_spacy, per_class_accuracy, rare_vocab  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=250)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "phase0_tokenizer_analysis.json"))
a = ap.parse_args()
torch.manual_seed(a.seed)

t0 = time.time()
B = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"))
vqvae, vq, tok = B.vqvae, B.vqvae.vq, B.pre_tokenizer
levels = [int(l) for l in vq.levels]
K = len(levels)
sl = level_slices(levels)
X = examples("val", a.n)

codes, preds, reenc = [], [[] for _ in range(K)], []
with torch.no_grad():
    for s in range(0, len(X), a.bs):
        x = torch.from_numpy(X[s:s + a.bs])
        idx = vq.z_to_idx(vqvae._encode_pre_quant(x))
        codes.append(idx)
        snaps = zhat_snapshots(vq, idx)
        for k in range(K):
            preds[k].append(vqvae.decoder(snaps[k]).argmax(-1))
        # fixed point: re-encode the full reconstruction
        reenc.append(vq.z_to_idx(vqvae._encode_pre_quant(preds[-1][-1])))
codes = torch.cat(codes).numpy()
preds = [torch.cat(p).numpy() for p in preds]
reenc = torch.cat(reenc).numpy()
print(f"encoded/decoded {len(X)} examples in {time.time() - t0:.1f}s", flush=True)

cache = os.path.join(ROOT, "runs", "cache", f"val{len(X)}_token_classes.npz")
if os.path.exists(cache):
    _c = np.load(cache)
    cls, rare = _c["cls"], _c["rare"]
else:
    nlp = load_spacy()
    common = rare_vocab(tok, TrainStream(), cache=os.path.join(ROOT, "results", "tinystories_common_words.json"))
    cls, rare = classify(X, tok, nlp, common)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, cls=cls, rare=rare)
print(f"classified tokens in {time.time() - t0:.1f}s; class counts:",
      {c: int((cls == i).sum()) for i, c in enumerate(CLASSES)}, "rare:", int(rare.sum()), flush=True)

per_level = []
for k in range(K):
    c = codes[:, sl[k]]
    counts = np.bincount(c.ravel(), minlength=vq.vqvae_vocab_sizes[k])
    p = counts[counts > 0] / counts.sum()
    per_level.append({
        "level": k, "length": levels[k], "codes_cum": int(sum(levels[:k + 1])),
        "acc": per_class_accuracy(preds[k], X, cls, rare),
        "distinct_codes": int((counts > 0).sum()),
        "code_entropy_bits": float(-(p * np.log2(p)).sum()),
        "reencode_code_match": float((reenc[:, sl[k]] == c).mean()),
        "acc_special_positions": float((preds[k] == X)[cls < 0].mean()),
        "error_rate_ppm": {cn: 1e6 * (1 - per_class_accuracy(preds[k], X, cls, rare).get(cn, 1.0))
                           for cn in ["all", "name", "numeral", "rare", "noun", "function"]},
    })
    acc = per_level[-1]["acc"]
    print(f"L{k:2d} len {levels[k]:3d}: acc {100 * acc['all']:5.1f} | " + " ".join(
        f"{cn[:5]} {100 * acc[cn]:5.1f}" for cn in CLASSES + ["rare"] if cn in acc)
        + f" | codes {per_level[-1]['distinct_codes']:4d} H {per_level[-1]['code_entropy_bits']:.2f}b"
        + f" | reenc {100 * per_level[-1]['reencode_code_match']:.1f}", flush=True)

final_exact = float((preds[-1] == X).all(1).mean())
ex_rng = np.random.default_rng(a.seed)
show = ex_rng.choice(len(X), size=3, replace=False)
samples = []
with torch.no_grad():
    _, zl = vq.idx_to_zhats(torch.from_numpy(codes[show]), return_zlevels=True)  # z_levels are correct (issue #11 affects z_hats only)
    for k in [0, 1, 2, 4, 7, 11]:
        out = vqvae.decoder(zl[k]).argmax(-1)
        for j, i in enumerate(show):
            samples.append({"example": int(i), "level": k, "native_len": levels[k],
                            "text": tok.decode(out[j].tolist())})
    for j, i in enumerate(show):
        samples.append({"example": int(i), "level": "input", "text": tok.decode(X[i].tolist())})

res = {
    "run": "phase0_tokenizer_analysis", "checkpoint": "hazyresearch/ncm-checkpoints tinystories/tokenizer.pt",
    "split": "val", "n": len(X), "seed": a.seed,
    "tokenizer_params_trainable": int(sum(p.numel() for p in vqvae.parameters() if p.requires_grad)),
    "tokenizer_params_total": int(sum(p.numel() for p in vqvae.parameters())),
    "levels": levels, "final_sequence_exact_match": final_exact,
    "per_level": per_level, "samples_native_resolution": samples,
    "wallclock_s": time.time() - t0,
}
with open(a.out, "w") as f:
    json.dump(res, f, indent=1)
print("final-level sequence exact match: %.3f" % final_exact)
for s_ in samples:
    if s_["level"] in (0, 2, 11, "input"):
        print(f"[ex {s_['example']} L{s_['level']}] {s_['text'][:300]!r}")
