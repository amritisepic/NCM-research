"""E1 (docs/preregistration.md): how much of NCM's code NLL is representation choice. For each item (window x, canonical
codes c*, reconstruction x^ = D(c*)) and cut k:
  N_k = -log Q(c*_{>=k} | c*_{<k})       teacher-forced, fp32
  A_k = -log Q(c*_{<k})                   (TinyStories level 0: the released unigram)
  rho_k = P(D(c*_{<k}, C_{>=k}) = x^)     C_{>=k} ancestral samples from the model at T = 1 (levels k, k+1, .. each
                                          given the coarser sampled ones), Monte Carlo: rounds of --round samples, up
                                          to --max_samples, stopping once --hits hits are reached
  T_k = -log rho_k,  R_k = N_k - T_k (representation term),  U_k = A_k + T_k (upper bound on the text NLL)
Equality ("hit"): TinyStories every token of the 128-token window; WMT14 the tokens up to the first EOS, pads dropped
(the text the evaluation reads); Sudoku the board. Gumbel noise is seeded per (item, cut), the same for every model,
so model differences are paired. fp32, TF32 off; sampling uses src/ncmr/cached_decode.py's level calls, which equal
the teacher-forced logits (tests/test_d1_bench.py, tests/test_e1_equivalence.py).

  python scripts/e1_equivalence.py --domain tinystories --models A0,A1,A2,A4 --run_root runs --run_dir runs/e1_ts
  python scripts/e1_equivalence.py --domain wmt14 --models J0,J1,J2F --run_root runs --data runs/../data/wmt14_train_pairs.npz --run_dir runs/e1_wmt
  python scripts/e1_equivalence.py --domain sudoku --models S0 --run_dir runs/e1_sdk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr import cached_decode as CD  # noqa: E402
from ncmr.jacobi_train import parallel_logits  # noqa: E402

CKPTS = {"A1": "L_ts_ctrl_s0", "A2": "L_ts_requant_s0", "A3": "L_ts_onpol_s0", "A4": "L_ts_b1_s0",
         "J1": "L_wmt_ctrl_s0", "J2F": "L_wmt_consist_frozen_s0"}
ART = {"tinystories": "tinystories", "wmt14": "wmt14", "sudoku": "sudoku/hard"}


# ------------------------------------------------------------------------------------------------ per-domain pieces
class Domain:
    """prefix ids, canonical codes, per-level teacher-forced NLL and the hit test for one domain."""

    def __init__(self, name, net, tok, extra):
        self.name, self.net, self.tok, self.extra = name, net, tok, extra
        self.levels = [int(l) for l in tok.vq.levels]

    def canonical(self, x):
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            return self.tok.vq.z_to_idx(self.tok._encode_pre_quant(x))

    def decode_ids(self, codes):
        with torch.no_grad():
            return self.tok.decoder(self.tok.vq.idx_to_z_q(codes)).argmax(-1)

    def key(self, ids_row):
        """What must be equal for a hit."""
        ids = [int(t) for t in ids_row]
        if self.name == "wmt14":
            eos, pad = self.extra["eos"], self.extra["pad"]
            if eos in ids:
                ids = ids[:ids.index(eos)]
            return tuple(t for t in ids if t != pad)
        if self.name == "sudoku":
            from ncmr.sudoku import cells
            b = cells(ids)
            return None if b is None else tuple(b)
        return tuple(ids)

    @torch.no_grad()
    def level_nll(self, prefix_ids, canon):
        """(K,) per-level NLL of the canonical codes, teacher-forced (level 0: unigram for TinyStories)."""
        net, lv = self.net, self.levels
        z, pm = CD.prefix_inputs(net, prefix_ids)
        lg = parallel_logits(net, z, canon, pm).float()
        lp = F.log_softmax(lg, -1)
        tgt = canon if lg.shape[1] == canon.shape[1] else canon[:, lv[0]:]
        nll = -lp.gather(-1, tgt[..., None])[..., 0][0]
        out, off = [], 0
        if lg.shape[1] != canon.shape[1]:                        # level 0 from the unigram
            out.append(float(-torch.log(self.extra["unigram"][canon[0, :lv[0]]]).sum()))
            levels = lv[1:]
        else:
            levels = lv
        for l in levels:
            out.append(float(nll[off:off + l].sum()))
            off += l
        return out


@torch.no_grad()
def mc_hits(dom, prefix_ids, canon, k, target, seed, max_samples, round_, hits_needed, temperature=1.0):
    """Monte Carlo of rho_k for one item: (hits, samples)."""
    net = dom.net
    cache = CD.PrefixCache(net, *CD.prefix_inputs(net, prefix_ids))
    g = torch.Generator(device=canon.device).manual_seed(int(seed))
    hits = n = 0
    if target is None:              # x^ itself is not a valid output (Sudoku board that does not parse): no hit counts
        return 0, 0
    while n < max_samples and hits < hits_needed:
        m = min(round_, max_samples - n)
        codes = CD.sample_from_cut(net, cache, canon, k, m, g, temperature)
        hits += sum(dom.key(r) == target for r in dom.decode_ids(codes).tolist())
        n += m
    return hits, n


def measure(dom, items, cuts, seed, max_samples=2048, round_=256, hits_needed=20, log_every=16, full_cut_min=8,
            coarse_n=16):
    """items: [(prefix_ids (1, P), x (1, L))]. Per item: per-level NLL, reconstruction exactness, and per cut
    (hits, samples). Cuts below `full_cut_min` (descriptive) run on the first `coarse_n` items only: their samples
    rarely hit and cost most of the budget (prototype, R046)."""
    rows, t0 = [], time.time()
    for i, (pre, x) in enumerate(items):
        canon = dom.canonical(x)
        xhat = dom.decode_ids(canon)
        target = dom.key(xhat[0].tolist())
        r = {"level_nll": dom.level_nll(pre, canon), "recon_exact": dom.key(x[0].tolist()) == target, "mc": {}}
        for k in cuts:
            if k < full_cut_min and i >= coarse_n:
                continue
            h, n = mc_hits(dom, pre, canon, k, target, seed * 1_000_003 + i * 101 + k, max_samples, round_, hits_needed)
            r["mc"][str(k)] = [h, n]
        rows.append(r)
        if (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(items)} items, {time.time() - t0:.0f}s, last {r['mc']}", flush=True)
    return rows


# ------------------------------------------------------------------------------------------------ summaries
def summarise(rows_all, cuts, K):
    """Per cut: hit share, N, A, and T / R / U (mean ± SE over items when every item has a hit; otherwise the median
    and bounds on mean R with zero-hit items at T = N (R_lo) and T = log(n/3) (R_hi))."""
    out = {}
    for k in cuts:
        rows = [r for r in rows_all if str(k) in r["mc"] and r["mc"][str(k)][1] > 0]   # x^-invalid items excluded
        if not rows:
            continue
        N = np.array([sum(r["level_nll"][k:]) for r in rows])
        A = np.array([sum(r["level_nll"][:k]) for r in rows])
        h = np.array([r["mc"][str(k)][0] for r in rows])
        n = np.array([r["mc"][str(k)][1] for r in rows])
        T = np.where(h > 0, -np.log(np.maximum(h, 1) / n), np.inf)
        se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        o = {"items": len(rows), "hit_share": float((h > 0).mean()), "samples_mean": float(n.mean()),
             "N_mean": float(N.mean()), "N_se": se(N), "A_mean": float(A.mean()), "A_se": se(A),
             "T_median": float(np.median(T)) if (h > 0).mean() > 0.5 else None}
        Tlo = np.where(h > 0, T, np.minimum(np.log(n / 3.0), N))      # rule of three: 95% bound after 0 hits
        R_lo, R_hi = np.where(h > 0, N - T, 0.0), N - Tlo
        o.update(R_lo_mean=float(R_lo.mean()), R_hi_mean=float(R_hi.mean()),
                 R_lo_share_of_N=float(R_lo.mean() / N.mean()) if N.mean() > 0 else None,
                 R_hi_share_of_N=float(R_hi.mean() / N.mean()) if N.mean() > 0 else None)
        if (h > 0).all():
            R, U = N - T, A + T
            o.update(T_mean=float(T.mean()), T_se=se(T), R_mean=float(R.mean()), R_se=se(R), U_mean=float(U.mean()),
                     U_se=se(U), R_share_of_N=float(R.mean() / N.mean()) if N.mean() > 0 else None)
        out[str(k)] = o
    return out


def paired(rows_a, rows_b, k):
    """Model b minus model a at cut k over the same items: dN, and dT where both have a hit on every item."""
    keep = [i for i, (a, b) in enumerate(zip(rows_a, rows_b)) if a["mc"][str(k)][1] > 0 and b["mc"][str(k)][1] > 0]
    rows_a, rows_b = [rows_a[i] for i in keep], [rows_b[i] for i in keep]
    dN = np.array([sum(b["level_nll"][k:]) - sum(a["level_nll"][k:]) for a, b in zip(rows_a, rows_b)])
    o = {"dN_mean": float(dN.mean()), "dN_se": float(dN.std(ddof=1) / np.sqrt(len(dN)))}
    ha = np.array([a["mc"][str(k)] for a in rows_a], float)
    hb = np.array([b["mc"][str(k)] for b in rows_b], float)
    if (ha[:, 0] > 0).all() and (hb[:, 0] > 0).all():
        dT = -np.log(hb[:, 0] / hb[:, 1]) + np.log(ha[:, 0] / ha[:, 1])
        o.update(dT_mean=float(dT.mean()), dT_se=float(dT.std(ddof=1) / np.sqrt(len(dT))),
                 dT_share_of_dN=float(abs(dT.mean()) / abs(dN.mean())) if dN.mean() != 0 else None)
    return o


# ------------------------------------------------------------------------------------------------ main
def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_items(domain, B, n, device, data):
    from ncmr.ncm_ref_loader import safe_pickle_load
    if domain == "tinystories":
        from ncmr.data import EOS, examples
        x = torch.from_numpy(examples("val", n)).to(device)
        pre = torch.full((1, 1), EOS, dtype=torch.long, device=device)
        return [(pre, x[i:i + 1]) for i in range(n)]
    if domain == "wmt14":
        D = np.load(data)
        src, tgt = D["val_src"][:n].astype(np.int64), D["val_tgt"][:n].astype(np.int64)
        return [(torch.from_numpy(src[i:i + 1]).to(device), torch.from_numpy(tgt[i:i + 1]).to(device)) for i in range(n)]
    P = int(B.net.prefix_len)
    pz = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", "hard", "puzzles.pickle"))[:n]
    sol = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", "hard", "solutions.pickle"))[:n]
    return [(torch.tensor([list(p)[:P]], device=device), torch.tensor([list(s)], device=device)) for p, s in zip(pz, sol)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(ART))
    ap.add_argument("--models", required=True, help="A0,A1,A2,A4 | J0,J1,J2F | S0 (A0 / J0 / S0 = released)")
    ap.add_argument("--run_root", default="", help="directory holding the fine-tuned run directories (final.pt)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data", default="", help="WMT14: prep_wmt14.py's train_pairs.npz (val_src / val_tgt)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--cut_min", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=2048)
    ap.add_argument("--round", type=int, default=64)
    ap.add_argument("--hits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=16)
    ap.add_argument("--full_cut_min", type=int, default=8, help="registered cuts: >= this on every item")
    ap.add_argument("--coarse_n", type=int, default=8, help="descriptive cuts below full_cut_min: first n items only")
    a = ap.parse_args()
    from ncmr.ncm_ref_loader import load_ncm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    os.makedirs(a.run_dir, exist_ok=True)
    B = load_ncm(os.path.join(ROOT, "artifacts", ART[a.domain]), device=device)
    net, tok = B.net.eval(), B.vqvae.eval()
    for p in list(net.parameters()) + list(tok.parameters()):
        p.requires_grad_(False)
    extra = {}
    if a.domain == "tinystories":
        extra["unigram"] = B.ckpt_extra["inference"]["level0_unigram"].float().to(device)
    if a.domain == "wmt14":
        extra.update(eos=B.pre_tokenizer.eos_token_id, pad=B.pre_tokenizer.pad_token_id)
    items = load_items(a.domain, B, a.n, device, a.data)
    K = len(tok.vq.levels)
    cuts = list(range(a.cut_min, K))
    released = {k: v.clone() for k, v in net.state_dict().items()}
    res = {"domain": a.domain, "n": len(items), "cuts": cuts, "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
           "torch": torch.__version__,
           "protocol": {"temperature": 1.0, "max_samples": a.max_samples, "round": a.round, "hits": a.hits, "seed": a.seed,
                        "full_cut_min": a.full_cut_min, "coarse_n": a.coarse_n,
                        "R_hi_zero_hit_T": "min(log(n/3), N_k): T <= N_k always (c* itself decodes to x^)",
                        "x_hat_invalid": "items whose x^ is not a valid output get (0 hits, 0 samples) and are excluded",
                        "precision": "fp32, TF32 off", "items": {"tinystories": "first n even-index val windows",
                                                                 "wmt14": "first n newstest2013 (val) pairs, target given source",
                                                                 "sudoku": "first n released hard puzzles"}[a.domain],
                        "hit": {"tinystories": "all 128 tokens equal x^", "wmt14": "tokens up to the first EOS (pads dropped) equal x^'s",
                                "sudoku": "board equals x^'s"}[a.domain]},
           "models": {}, "summary": {}, "rows": {}}
    for name in a.models.split(","):
        if name in CKPTS:
            path = os.path.join(a.run_root, CKPTS[name], "final.pt")
            if not os.path.exists(path):
                sys.exit(f"missing checkpoint for {name}: {CKPTS[name]}/final.pt")
            st = torch.load(path, map_location=device, weights_only=False)
            net.load_state_dict(st["model"])
            res["models"][name] = {"ckpt": f"{CKPTS[name]}/final.pt", "sha256": digest(path)}
        else:
            net.load_state_dict(released)
            res["models"][name] = {"ckpt": "released"}
        t = time.time()
        rows = measure(Domain(a.domain, net, tok, extra), items, cuts, a.seed, a.max_samples, a.round, a.hits, a.log_every,
                       a.full_cut_min, a.coarse_n)
        res["models"][name]["wall_s"] = time.time() - t
        res["rows"][name] = rows
        res["summary"][name] = summarise(rows, cuts, K)
        print(name, json.dumps(res["summary"][name].get("8", {})), flush=True)
        with open(os.path.join(a.run_dir, "result.json"), "w") as f:      # after every model: a partial result survives
            json.dump(res, f)
    names = a.models.split(",")
    base = {"tinystories": "A1", "wmt14": "J0"}.get(a.domain)
    if base in names:
        res["paired_vs_" + base] = {m: {str(k): paired(res["rows"][base], res["rows"][m], k)
                                        for k in cuts if k >= a.full_cut_min}
                                    for m in names if m != base}
    net.load_state_dict(released)
    with open(os.path.join(a.run_dir, "result.json"), "w") as f:
        json.dump(res, f)
    print("wrote result.json")


if __name__ == "__main__":
    main()
