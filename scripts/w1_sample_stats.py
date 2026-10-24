"""Exploratory (not registered): token-level character of the wave-1 samples, from the saved ids only.

  python scripts/w1_sample_stats.py [--root results/colab] [--train_tokens 20000000] [--read 30 --read_seed 2026]

For every arm and for 4096 human validation windows (the MAUVE reference set):
  - unigram fit: mean log p_train(token) (bag-of-words content; add-one unigram from the first `train_tokens` training
    tokens);
  - local order: mean log p_train(token | previous token) from an interpolated bigram model, and its gain over the
    unigram (how much the sequence order beats a shuffled bag of the same tokens);
  - both by position in the 128-token window, plus the adjacent-duplicate rate by position;
  - pooled unigram Jensen-Shannon divergence to the human windows.
Writes `results/w1_sample_stats.json` and, with --read, `results/w1_samples_read_s<seed>.md` (random samples per arm).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.data import EOS, PAD, cut_at_eos, examples  # noqa: E402
from ncmr.ncm_ref_loader import safe_pickle_load  # noqa: E402

ARMS = {"A0": ("w1_L_ts_ctrl", "L_ts_released16"), "A1": ("w1_L_ts_ctrl", "L_ts_ctrl_s0"),
        "A2": ("w1_L_ts_requant", "L_ts_requant_s0"), "A3": ("w1_L_ts_onpol", "L_ts_onpol_s0"),
        "A4": ("w1_L_ts_b1", "L_ts_b1_s0"), "A5": ("w1_L_ts_onpol2", "L_ts_onpol2_s0"),
        "A6": ("w1_L_ts_soft", "L_ts_soft_s0")}
V, W, LAM = 8192, 128, 0.1


def bigram_model(n_tokens, data_dir):
    tok = np.memmap(os.path.join(data_dir, "train_bpe_8192.bin"), dtype=np.uint16, mode="r")[:n_tokens].astype(np.int64)
    uni = np.bincount(tok, minlength=V).astype(np.float64)
    keys, cnt = np.unique(tok[:-1] * V + tok[1:], return_counts=True)
    return (uni + 1) / (uni.sum() + V), uni, keys, cnt


def scores(seqs, p_uni, uni, keys, cnt):
    """Per-position arrays (n, W) of unigram and bigram log-probs (NaN where absent) and adjacent duplicates."""
    n = len(seqs)
    lu, lb, dup = (np.full((n, W), np.nan) for _ in range(3))
    for i, s in enumerate(seqs):
        s = np.asarray(s[:W], dtype=np.int64)
        if len(s) == 0:
            continue
        lu[i, :len(s)] = np.log(p_uni[s])
        if len(s) > 1:
            a, b = s[:-1], s[1:]
            j = np.searchsorted(keys, a * V + b)
            hit = (j < len(keys)) & (keys[np.minimum(j, len(keys) - 1)] == a * V + b)
            c_ab = np.where(hit, cnt[np.minimum(j, len(keys) - 1)], 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                cond = np.where(uni[a] > 0, c_ab / np.maximum(uni[a], 1), 0.0)
            lb[i, 1:len(s)] = np.log((1 - LAM) * cond + LAM * p_uni[b])
            dup[i, 1:len(s)] = (a == b).astype(float)
    return lu, lb, dup


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda x, y: float(np.sum(np.where(x > 0, x * np.log(np.where(x > 0, x, 1) / y), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "colab"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "tinystories"))
    ap.add_argument("--train_tokens", type=int, default=20_000_000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "w1_sample_stats.json"))
    ap.add_argument("--read", type=int, default=0)
    ap.add_argument("--read_seed", type=int, default=2026)
    a = ap.parse_args()
    p_uni, uni, keys, cnt = bigram_model(a.train_tokens, a.data_dir)
    sets = {"human_val": [[t for t in cut_at_eos(r) if t != PAD] for r in examples("val", 4096).tolist()]}
    for arm, (d, prefix) in ARMS.items():
        sets[arm] = json.load(open(os.path.join(a.root, d, f"{prefix}__samples_final.json")))["ids"]
    ref_counts = np.bincount(np.concatenate([np.asarray(s, dtype=np.int64) for s in sets["human_val"]]), minlength=V)
    out = {"train_tokens": a.train_tokens, "bigram_interpolation": LAM, "sets": {}}
    for name, seqs in sets.items():
        lu, lb, dup = scores(seqs, p_uni, uni, keys, cnt)
        pooled = np.bincount(np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs if len(s)]), minlength=V)
        gain = lb - lu
        bands = [(1, 16), (16, 32), (32, 64), (64, 96), (96, 128)]
        out["sets"][name] = {
            "n": len(seqs), "mean_len": float(np.mean([len(s) for s in seqs])),
            "unigram_logp": float(np.nanmean(lu)), "bigram_logp": float(np.nanmean(lb)),
            "bigram_gain": float(np.nanmean(gain)), "adjacent_dup_per_1k": 1000 * float(np.nanmean(dup)),
            "js_unigram_to_human": js(pooled / pooled.sum(), ref_counts / ref_counts.sum()),
            "by_position": {f"{lo}-{hi - 1}": {"unigram_logp": float(np.nanmean(lu[:, lo:hi])),
                                               "bigram_gain": float(np.nanmean(gain[:, lo:hi])),
                                               "adjacent_dup_per_1k": 1000 * float(np.nanmean(dup[:, lo:hi]))}
                            for lo, hi in bands},
            "per_sample_bigram_gain_quantiles": np.nanquantile(np.nanmean(gain, 1), [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        }
        r = out["sets"][name]
        print(f"{name:9} uni {r['unigram_logp']:.3f}  bi {r['bigram_logp']:.3f}  gain {r['bigram_gain']:.3f}  "
              f"dup/1k {r['adjacent_dup_per_1k']:.1f}  JS {r['js_unigram_to_human']:.4f}  gain by pos " +
              " ".join(f"{v['bigram_gain']:.2f}" for v in r["by_position"].values()), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    if a.read:
        vocab = safe_pickle_load(os.path.join(a.data_dir, "bpe_vocab_8192.pickle"))
        dec = lambda ids: b"".join(vocab[int(t)] for t in ids if int(t) not in (EOS, PAD)).decode("utf-8", "replace")
        rng = np.random.default_rng(a.read_seed)
        path = os.path.join(os.path.dirname(a.out), f"w1_samples_read_s{a.read_seed}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Wave-1 samples, {a.read} per arm, drawn with numpy default_rng({a.read_seed}) from each arm's "
                    "4096 final samples (same indices for every arm)\n\n")
            idx = rng.choice(4096, a.read, replace=False)
            for arm in ("A0", "A1", "A2", "A3", "A4", "A5"):
                f.write(f"## {arm}\n\n")
                for i in idx:
                    f.write(f"- [{int(i)}] {' '.join(dec(sets[arm][i]).split())}\n")
                f.write("\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
