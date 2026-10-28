"""Exploratory (not registered): token-level character of the wave-1 samples, from the saved ids only.

  python scripts/w1_sample_stats.py [--root results/colab] [--train_tokens 20000000] [--read 30 --read_seed 2026]
  python scripts/w1_sample_stats.py --set w2m       (W2M's registered bigram gain / adjacent duplicates: A0 + M0-M7,
                                                     writes results/w2m_sample_stats.json; audit 8, question 8; an arm whose
                                                     samples are absent, e.g. M5, is skipped with a printed note)
  python scripts/w1_sample_stats.py --set w2c1      (W2C stage 1: A0 + M2 / M3 at seeds 1-3, from results/colab/w2c1_L_ts_m{2,3})
  python scripts/w1_sample_stats.py --set w2p       (M3-0: A0, W2M's seed-0 M3 and the M3-0 pilot, results/colab/w2p_L_ts_m3_c0)
  python scripts/w1_sample_stats.py --set l1r       (L1R: A0, W2M's seed-0 M2 and M3 and M2R, results/colab/l1r_L_ts_m2r)

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
W2M_ARMS = {"A0": ("w1_L_ts_ctrl", "L_ts_released16"), "M0": ("w2m_L_ts_m0", "L_ts_m0_uniform_s0"),
            "M1": ("w2m_L_ts_m1", "L_ts_m1_policy_s0"), "M2": ("w2m_L_ts_m2", "L_ts_m2_frozen_pw_s0"),
            "M3": ("w2m_L_ts_m3", "L_ts_m3_uniform_wrong_s0"), "M4": ("w2m_L_ts_m4", "L_ts_m4_frozen_pw_flags_s0"),
            "M5": ("w2m_L_ts_m5", "L_ts_m5_frozen_pw_shuffled_s0"), "M6": ("w2m_L_ts_m6", "L_ts_m6_policy_usedparent_s0"),
            "M7": ("w2m_L_ts_m7", "L_ts_m7_uniform_a3rates_s0")}
_A0 = ("w1_L_ts_ctrl", "L_ts_released16")
_M2, _M3 = ("w2m_L_ts_m2", "L_ts_m2_frozen_pw_s0"), ("w2m_L_ts_m3", "L_ts_m3_uniform_wrong_s0")
# R056 package: result directories as the notebooks name them under results/colab/<notebook name>/
W2C1_ARMS = {"A0": _A0, **{f"M{m}_s{s}": (f"w2c1_L_ts_m{m}", f"{cfg}_s{s}")
                           for m, cfg in ((2, "L_ts_m2_frozen_pw"), (3, "L_ts_m3_uniform_wrong")) for s in (1, 2, 3)}}
W2P_ARMS = {"A0": _A0, "M3": _M3, "M3-0": ("w2p_L_ts_m3_c0", "L_ts_m3_uniform_wrong_c0_s0")}
L1R_ARMS = {"A0": _A0, "M2": _M2, "M3": _M3, "M2R": ("l1r_L_ts_m2r", "L_ts_m2r_frozen_pw_l1reassign_s0")}
SETS = {"w1": ARMS, "w2m": W2M_ARMS, "w2c1": W2C1_ARMS, "w2p": W2P_ARMS, "l1r": L1R_ARMS}
SKIP_ABSENT = {"w2m", "w2c1", "w2p", "l1r"}     # W1 stays strict: all its arms exist, and a missing one is an error
V, W, LAM = 8192, 128, 0.1


def shown(path):
    """`path` for printing: repo-relative inside the repository, else only its name (AGENTS.md: no local paths)."""
    try:
        rel = os.path.relpath(os.path.abspath(path), ROOT)
    except ValueError:  # another drive on Windows
        return os.path.basename(path)
    return os.path.basename(path) if rel.split(os.sep)[0] == os.pardir else rel.replace(os.sep, "/")


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
    ap.add_argument("--set", default="w1", choices=sorted(SETS))
    ap.add_argument("--out", default="", help="default results/<set>_sample_stats.json")
    ap.add_argument("--read", type=int, default=0)
    ap.add_argument("--read_seed", type=int, default=2026)
    a = ap.parse_args()
    a.out = a.out or os.path.join(ROOT, "results", f"{a.set}_sample_stats.json")
    p_uni, uni, keys, cnt = bigram_model(a.train_tokens, a.data_dir)
    sets = {"human_val": [[t for t in cut_at_eos(r) if t != PAD] for r in examples("val", 4096).tolist()]}
    for arm, (d, prefix) in SETS[a.set].items():
        path = os.path.join(a.root, d, f"{prefix}__samples_final.json")
        if a.set in SKIP_ABSENT and not os.path.exists(path):
            print(f"{arm:9} skipped: no samples at {d}/{prefix}__samples_final.json", flush=True)
            continue
        sets[arm] = json.load(open(path))["ids"]
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
    print("wrote", shown(a.out))
    if a.read:
        vocab = safe_pickle_load(os.path.join(a.data_dir, "bpe_vocab_8192.pickle"))
        dec = lambda ids: b"".join(vocab[int(t)] for t in ids if int(t) not in (EOS, PAD)).decode("utf-8", "replace")
        rng = np.random.default_rng(a.read_seed)
        path = os.path.join(os.path.dirname(a.out), f"{a.set}_samples_read_s{a.read_seed}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {a.set} samples, {a.read} per arm, drawn with numpy default_rng({a.read_seed}) from each arm's "
                    "4096 final samples (same indices for every arm)\n\n")
            idx = rng.choice(4096, a.read, replace=False)
            for arm in (("A0", "A1", "A2", "A3", "A4", "A5") if a.set == "w1" else tuple(SETS[a.set])):
                if arm not in sets:
                    continue
                f.write(f"## {arm}\n\n")
                for i in idx:
                    f.write(f"- [{int(i)}] {' '.join(dec(sets[arm][i]).split())}\n")
                f.write("\n")
        print("wrote", shown(path))


if __name__ == "__main__":
    main()
