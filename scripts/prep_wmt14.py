"""WMT14 De-En training/validation pairs in the released NCM's format (for scripts/train_wmt14_L.py).

Downloads the HF `wmt/wmt14` de-en parquet files (cached under data/wmt14/hf/), samples N training pairs
uniformly from the chosen shards (seed 0), and tokenizes with the released model's Qwen3 tokenizer exactly as
upstream decoding does: source = first 96 ids, right-padded with PAD; target = English ids + EOS, right-padded
with PAD to 64. The released generator's teacher-forced NLL identifies this target format (392 nats/seq vs 460
for PAD-only and 2706 for EOS padding, 48 test pairs; R022). Pairs whose English side needs > 63 ids are
dropped, not truncated (training and val NLL only; val BLEU uses all 3000 newstest2013 sentences). Validation = newstest2013 (model selection); newstest2014 stays the test set
(data/wmt14/sources.pickle, as in w0).

Usage: python scripts/prep_wmt14.py --n 400000 [--shards 0 1 2] [--out data/wmt14/train_pairs.npz]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=400000)
ap.add_argument("--shards", type=int, nargs="*", default=[0, 1, 2])
ap.add_argument("--out", default=os.path.join(ROOT, "data", "wmt14", "train_pairs.npz"))
ap.add_argument("--cache", default=os.path.join(ROOT, "data", "wmt14", "hf"))
a = ap.parse_args()

import pyarrow.parquet as pq  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402

selection = {"n": a.n, "shards": a.shards, "seed": 0, "max_target_ids": 63}
selection_path = a.out.replace(".npz", "_selection.json")
if os.path.exists(a.out) and os.path.exists(a.out.replace(".npz", "_val_text.json")):
    # reuse only an archive built with the same selection (codex audit2 A2-8)
    if not os.path.exists(selection_path) or json.load(open(selection_path)) != selection:
        sys.exit(f"{a.out} exists with different or missing selection metadata; delete it or use another --out")
    with np.load(a.out) as prev:
        if len(prev["src"]) != a.n:
            sys.exit(f"{a.out} holds {len(prev['src'])} pairs, not {a.n}")
    print("exists:", a.out)
    sys.exit(0)
BASE = "https://huggingface.co/datasets/wmt/wmt14/resolve/main/de-en/"
os.makedirs(a.cache, exist_ok=True)


def fetch(name):
    p = os.path.join(a.cache, name)
    if not os.path.exists(p):
        print("downloading", name, flush=True)
        urllib.request.urlretrieve(BASE + name, p + ".tmp")
        os.replace(p + ".tmp", p)
    return p


def pairs(path):
    col = pq.read_table(path, columns=["translation"]).column("translation").to_pylist()
    return [(t["de"], t["en"]) for t in col]


cfg = OmegaConf.load(os.path.join(ROOT, "artifacts", "wmt14", "config.yaml"))
tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id))
EOS = tok.convert_tokens_to_ids(str(cfg.tokenizer.pre_tokenizer.special_tokens.eos_token))
PAD = tok.convert_tokens_to_ids(str(cfg.tokenizer.pre_tokenizer.special_tokens.pad_token))
P, L = int(cfg.model.prefix_len), int(cfg.tokenizer.vqvae.context_length)


def encode(batch):
    src = tok([d for d, _ in batch], add_special_tokens=False)["input_ids"]
    tgt = tok([e for _, e in batch], add_special_tokens=False)["input_ids"]
    S, T, keep = [], [], []
    for i, (s, t) in enumerate(zip(src, tgt)):
        if len(t) > L - 1 or len(s) == 0 or len(t) == 0:
            continue
        S.append((s[:P] + [PAD] * P)[:P])
        T.append(t + [EOS] + [PAD] * (L - 1 - len(t)))
        keep.append(i)
    return np.array(S, dtype=np.int32).reshape(-1, P), np.array(T, dtype=np.int32).reshape(-1, L), keep


rng = np.random.default_rng(0)
train = []
for s in a.shards:
    train += pairs(fetch(f"train-0000{s}-of-00003.parquet"))
print(f"{len(train)} training pairs in shards {a.shards}", flush=True)
sel = rng.permutation(len(train))   # walked until n eligible pairs are found (uniform over eligible pairs)
Ss, Ts = [], []
for i in range(0, len(sel), 20000):
    s_, t_, _ = encode([train[j] for j in sel[i:i + 20000]])
    Ss.append(s_)
    Ts.append(t_)
    if sum(len(x) for x in Ss) >= a.n:
        break
src, tgt = np.concatenate(Ss)[: a.n], np.concatenate(Ts)[: a.n]
if len(src) != a.n:
    sys.exit(f"only {len(src)} eligible pairs in shards {a.shards}, requested {a.n}")
val = pairs(fetch("validation-00000-of-00001.parquet"))
vs, vt, keep = encode(val)
np.savez_compressed(a.out, src=src, tgt=tgt, val_src=vs, val_tgt=vt, pad=PAD, eos=EOS)
json.dump(selection, open(selection_path, "w"), indent=1)
json.dump({"src": [d for d, _ in val], "ref": [e for _, e in val]},   # all 3000, for BLEU
          open(a.out.replace(".npz", "_val_text.json"), "w"))
print(json.dumps({"train_pairs": int(len(src)), "val_pairs": int(len(vs)), "val_total": len(val),
                  "train_tgt_len_mean": float((tgt != PAD).sum(1).mean()), "out": os.path.relpath(a.out, ROOT)}))
