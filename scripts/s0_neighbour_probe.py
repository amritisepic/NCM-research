"""S0 (docs/preregistration.md, "S0"): within-level left-neighbour dependence of the released TinyStories generator.

  python scripts/s0_neighbour_probe.py [--out results/s0_neighbour_gain.json]

The released generator and tokenizer are frozen (eval mode, fp32, TF32 off, no autocast). For each level k = 1..6 a small
trainable term is added to the level-k head's query: at position p >= 1, with h the frozen hidden state (after the final
layer norm) and n the frozen level-k codebook vector of the canonical left neighbour c*_{k,p-1},
    z' = proj(h) + U[(A h) * (B n)] + U0 (B0 n)            (rank 64; U, U0 start at zero)
and the logits keep the head's own distance form and temperature. Position 0 is unchanged. 3584 training windows,
512 development windows (early stopping only), 1024 val windows for the report.
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

LEVELS_PROBED = (1, 2, 3, 4, 5, 6)
GATE_LEVELS = (1, 2, 3, 4, 5)
GATE_MIN_NATS = 10.0
GATE_MIN_FRAC = 0.05
RANK = 64
LR, WD, BS, MAX_EPOCHS, PATIENCE = 1e-3, 0.01, 64, 30, 3
N_TRAIN, N_DEV, N_VAL = 3584, 512, 1024


class NeighbourTerm(nn.Module):
    """delta(h, n) = U[(A h) * (B n)] + U0 (B0 n); A, B, B0 ~ N(0, 1/fan_in) from a torch generator, U, U0 zero."""

    def __init__(self, n_embd: int, d: int, rank: int = RANK, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(torch.randn(rank, n_embd, generator=g) / math.sqrt(n_embd))
        self.B = nn.Parameter(torch.randn(rank, d, generator=g) / math.sqrt(d))
        self.B0 = nn.Parameter(torch.randn(rank, d, generator=g) / math.sqrt(d))
        self.U = nn.Parameter(torch.zeros(d, rank))
        self.U0 = nn.Parameter(torch.zeros(d, rank))

    def forward(self, h, n):
        return ((h @ self.A.t()) * (n @ self.B.t())) @ self.U.t() + (n @ self.B0.t()) @ self.U0.t()


def head_logits_from_z(head, z):
    """The tied head's own distance form and temperature (src/ncmr/ncm_ref/models/next_concept.py::_TiedCodebookHead)."""
    assert head.bias is None and not head.trainable
    return (2.0 * (z @ head.E.t()) - head.E_sq) / head.d_scale * torch.exp(head.log_inv_T)


def level_logits(head, term, h, codes_k, emb_weight):
    """h: (b, l, n_embd) hidden states at the level-k positions; codes_k: (b, l) canonical codes of that level;
    emb_weight: the level-k codebook (V, d). Position p >= 1 sees the codebook vector of codes_k[:, p-1]; position 0 and
    `term=None` give the released head exactly. The code at p itself never enters position p's query."""
    z = head.proj(h)
    if term is not None and h.shape[1] > 1:
        n = emb_weight[codes_k[:, :-1]]
        z = torch.cat([z[:, :1], z[:, 1:] + term(h[:, 1:], n)], 1)
    return head_logits_from_z(head, z)


def level_nll(head, term, h, codes_k, emb_weight):
    """Per-window NLL (nats), summed over ALL positions of the level (position 0 included)."""
    lg = level_logits(head, term, h, codes_k, emb_weight)
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), codes_k.reshape(-1), reduction="none").view(codes_k.shape).sum(1)


@torch.no_grad()
def eval_nll(head, term, h, codes_k, emb_weight, device, bs=128):
    out = []
    for s in range(0, h.shape[0], bs):
        out.append(level_nll(head, term, h[s:s + bs].to(device), codes_k[s:s + bs].to(device), emb_weight).cpu())
    return torch.cat(out)


def early_stop_loop(train_epoch, dev_nll, get_state, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Run epochs 1..max_epochs; after each, record the development NLL. Stop after `patience` epochs without a lower
    development NLL than the best so far. Returns (best_epoch, best_state, dev_history, epochs_run)."""
    best, best_epoch, best_state, hist, since = float("inf"), 0, None, [], 0
    for ep in range(1, max_epochs + 1):
        train_epoch(ep)
        d = float(dev_nll())
        hist.append(d)
        if d < best:
            best, best_epoch, best_state, since = d, ep, get_state(), 0
        else:
            since += 1
            if since >= patience:
                break
    return best_epoch, best_state, hist, len(hist)


def fit_level(head, emb_weight, h_tr, c_tr, h_dev, c_dev, device, seed=0, max_epochs=MAX_EPOCHS, patience=PATIENCE,
              lr=LR, wd=WD, bs=BS, rank=RANK):
    """Train one level's term. Returns (term with the best-development-epoch parameters, info)."""
    term = NeighbourTerm(head.proj.in_features, head.proj.out_features, rank=rank, seed=seed).to(device)
    opt = torch.optim.AdamW(term.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    n = h_tr.shape[0]

    def train_epoch(_ep):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = torch.as_tensor(perm[s:s + bs], device=h_tr.device)
            loss = level_nll(head, term, h_tr[idx].to(device), c_tr[idx].to(device), emb_weight).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def dev_nll():
        return eval_nll(head, term, h_dev, c_dev, emb_weight, device).mean()

    def get_state():
        return {k: v.detach().clone() for k, v in term.state_dict().items()}

    best_epoch, best_state, hist, run = early_stop_loop(train_epoch, dev_nll, get_state, max_epochs, patience)
    term.load_state_dict(best_state)
    return term, {"epochs_run": run, "best_epoch": best_epoch, "dev_nll_per_epoch": hist}


def mean_se(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return float(x.mean()), float(x.std(unbiased=True) / math.sqrt(x.numel()))


def gate_part1(gain_1to5: float, base_1to5: float):
    frac = gain_1to5 / base_1to5 if base_1to5 > 0 else float("nan")
    ok = bool(gain_1to5 >= GATE_MIN_NATS and gain_1to5 >= GATE_MIN_FRAC * base_1to5)
    return {"rule": f"gain summed over levels 1-5 >= {GATE_MIN_NATS} nats/window AND >= {GATE_MIN_FRAC * 100:g}% of their summed "
                    "canonical NLL", "gain_1to5": gain_1to5, "base_1to5": base_1to5, "gain_pct_of_base": 100.0 * frac,
            "verdict": "pass" if ok else "fail"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "s0_neighbour_gain.json"))
    a = ap.parse_args()
    t0 = time.time()

    from ncmr.data import EOS, TrainStream, examples
    from ncmr.ncm_ref_loader import load_ncm

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"), device=dev)
    tok, net = B.vqvae.eval(), B.net.eval()
    for m in (tok, net):
        for p in m.parameters():
            p.requires_grad_(False)
    levels = [int(l) for l in tok.vq.levels]
    off_code = [sum(levels[:k]) for k in range(len(levels))]            # start of level k in the code sequence
    off_hid = {k: sum(levels[1:k]) for k in LEVELS_PROBED}               # start of level k in the net's outputs
    emb = {k: tok.vq.embeddings[k].weight.detach() for k in LEVELS_PROBED}
    heads = {k: net.lm_heads[k - 1].eval() for k in LEVELS_PROBED}
    for k in LEVELS_PROBED:
        assert torch.equal(heads[k].E, emb[k]), f"head of level {k} is not tied to the level-{k} codebook"

    train_all = TrainStream().batch(np.random.default_rng(0), N_TRAIN + N_DEV)
    val = examples("val", N_VAL)
    windows = np.concatenate([train_all, val])                           # train | dev | val
    N = windows.shape[0]
    cap = {}
    hook = net.transformer.ln_f.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    H = {k: torch.empty(N, levels[k], net.config.n_embd) for k in LEVELS_PROBED}
    C = {k: torch.empty(N, levels[k], dtype=torch.long) for k in LEVELS_PROBED}
    max_diff = 0.0
    with torch.no_grad():
        for s in range(0, N, 32):
            x = torch.from_numpy(windows[s:s + 32]).to(dev)
            codes = tok.vq.z_to_idx(tok._encode_pre_quant(x))
            prefix = tok.encoder.wte(torch.full((x.shape[0], 1), EOS, dtype=torch.long, device=dev))
            logits = net(torch.cat([prefix, tok.vq.idx_to_ncm_input(codes)], 1))
            hid = cap["h"][:, net.prefix_len:].float()
            for k in LEVELS_PROBED:
                hk = hid[:, off_hid[k]:off_hid[k] + levels[k]]
                ref = logits[:, off_hid[k]:off_hid[k] + levels[k], :heads[k].out_features]
                max_diff = max(max_diff, float((head_logits_from_z(heads[k], heads[k].proj(hk)) - ref).abs().max()))
                H[k][s:s + 32] = hk.cpu()
                C[k][s:s + 32] = codes[:, off_code[k]:off_code[k] + levels[k]].cpu()
    hook.remove()
    assert max_diff < 1e-4, f"head(h) does not reproduce the model's logits (max abs diff {max_diff})"
    print(f"cached {N} windows; head(h) vs model logits max abs diff {max_diff:.2e}; {time.time() - t0:.0f}s", flush=True)

    tr, dv, va = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_DEV), slice(N_TRAIN + N_DEV, N)
    per_level, base_v, gain_v = {}, {}, {}
    for k in LEVELS_PROBED:
        head, E = heads[k], emb[k]
        for p in head.parameters():
            p.requires_grad_(False)
        h_tr = H[k][tr].to(dev)
        h_dv = H[k][dv].to(dev)
        term, info = fit_level(head, E, h_tr, C[k][tr].to(dev), h_dv, C[k][dv].to(dev), dev, seed=0)
        del h_tr, h_dv
        hv, cv = H[k][va], C[k][va]
        base = eval_nll(head, None, hv, cv, E, dev)
        with_t = eval_nll(head, term, hv, cv, E, dev)
        gain = base - with_t
        (bm, bse), (gm, gse), (wm, wse) = mean_se(base), mean_se(gain), mean_se(with_t)
        base_v[k], gain_v[k] = base, gain
        per_level[str(k)] = {
            "positions": levels[k], "val_windows": int(base.numel()),
            "base_nll_mean": bm, "base_nll_se": bse, "base_nll_per_code": bm / levels[k], "base_nll_per_code_se": bse / levels[k],
            "with_term_nll_mean": wm, "with_term_nll_se": wse,
            "gain_mean": gm, "gain_se": gse, "gain_per_code": gm / levels[k], "gain_per_code_se": gse / levels[k],
            "gain_pct_of_base": 100.0 * gm / bm, **info}
        print(f"level {k}: base {bm:.3f}±{bse:.3f}  gain {gm:+.3f}±{gse:.3f} nats/window  ({100 * gm / bm:+.2f}%)  "
              f"epochs {info['epochs_run']} best {info['best_epoch']}  {time.time() - t0:.0f}s", flush=True)
        del term

    b15 = torch.stack([base_v[k] for k in GATE_LEVELS]).sum(0)
    g15 = torch.stack([gain_v[k] for k in GATE_LEVELS]).sum(0)
    (bm, bse), (gm, gse) = mean_se(b15), mean_se(g15)
    gate = gate_part1(gm, bm)
    res = {
        "protocol": {
            "experiment": "S0 (docs/preregistration.md)", "model": "released TinyStories NCM, frozen, eval mode",
            "precision": "fp32, TF32 off, no autocast", "device": dev,
            "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "torch": torch.__version__,
            "levels": levels, "levels_probed": list(LEVELS_PROBED),
            "train_windows": N_TRAIN, "dev_windows": N_DEV, "val_windows": N_VAL,
            "train_source": "TrainStream().batch(np.random.default_rng(0), 4096); first 3584 train, last 512 development (early stopping only)",
            "val_source": "examples('val', 1024)", "term": "z' = proj(h) + U[(A h) * (B n)] + U0 (B0 n); position 0 unchanged",
            "rank": RANK, "init": "A, B, B0 ~ N(0, 1/fan_in) (torch generator seed 0, drawn in the order A, B, B0); U, U0 zero",
            "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WD}, "batch_windows": BS, "max_epochs": MAX_EPOCHS,
            "patience_epochs": PATIENCE, "shuffle_seed": 0, "loss": "mean over the batch's windows of the level NLL (summed over positions)",
            "nll_unit": "nats per window, summed over all positions of the level (position 0 included)",
            "se": "std(ddof=1)/sqrt(n) over val windows; gain SE paired over windows",
            "head_reproduction_max_abs_diff": max_diff, "wall_time_s": round(time.time() - t0, 1)},
        "levels": per_level,
        "sum_levels_1to5": {"base_nll_mean": bm, "base_nll_se": bse, "gain_mean": gm, "gain_se": gse,
                            "gain_pct_of_base": 100.0 * gm / bm},
        "gate": {"part1": gate, "part2_E2": "pending (E2 not run)"},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"levels 1-5: base {bm:.2f} gain {gm:+.3f}±{gse:.3f} ({100 * gm / bm:+.2f}%) -> gate part 1: {gate['verdict']}; "
          f"part 2: pending (E2 not run); wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
