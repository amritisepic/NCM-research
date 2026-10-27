"""W2M eval-only mechanism measures (docs/preregistration.md, "W2M"): the released model and the eight final checkpoints
of configs/scale/L_ts_m{0..7}_*.yaml, on the same windows and seeds. fp32, TF32 off; one JSON.

  python scripts/mech_eval_w2m.py --run_root <runs dir> --run_dir <out dir>        (writes <out dir>/result.json)

  (a) coarse-context dependence: 512 clean val windows, one code at level j (j = 1..6) changed to a uniformly random
      different code at a fixed random position, finer inputs rebuilt; mean KL(clean || perturbed) of the predictive
      distributions of the levels > j, per j.
  (b) replacement detectability (amendment W2Ma, R043: the pooled position-level probe is confounded by level identity and
      is stored as out["probe_pooled"], status "not read"). Row-level probe per level k = 1..6: contexts at W1's rates with
      50% clean windows mixed in, zero flags, replacements plausible-wrong (as M2: frozen released generator, canonical
      excluded) and uniform-wrong (as M3; the positive control), the SAME clean rows and masked slots for both kinds;
      label = window corrupted; feature = mean over level k's positions of the last hidden states (after ln_f); logistic
      probe trained on windows 0-383, AUC on windows 384-511. "too_weak_to_read" iff the uniform-wrong AUC < 0.75 at every k.
  (c) flag use (M4, M5): clean-label NLL per window on the corrupted held-out windows (plausible-wrong), true flags minus
      zero flags.
The corrupted windows, perturbations and probe split are built once from the released generator and reused for every model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr import w2m as W  # noqa: E402
from ncmr.corruption import CorruptionCfg, build_hierarchy  # noqa: E402
from ncmr.data import EOS, examples  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm  # noqa: E402
from ncmr.train import pick_device  # noqa: E402

ARM_DIRS = {"m0": "L_ts_m0_uniform", "m1": "L_ts_m1_policy", "m2": "L_ts_m2_frozen_pw", "m3": "L_ts_m3_uniform_wrong",
            "m4": "L_ts_m4_frozen_pw_flags", "m5": "L_ts_m5_frozen_pw_shuffled", "m6": "L_ts_m6_policy_usedparent",
            "m7": "L_ts_m7_uniform_a3rates"}
JS = (1, 2, 3, 4, 5, 6)
ROW_CLEAN_FRAC = 0.5                 # W2Ma: clean windows mixed into the row-level probe's contexts
ROW_AUC_FLOOR = 0.75                 # W2Ma: uniform-wrong AUC below this at every level = probe too weak to read
POOLED_STATUS = "not read (W2Ma: confounded by level identity, R043)"


@torch.no_grad()
def row_contexts(kind, tok, net, L0, prefix_emb, x, device, seed, rates, clean_frac=ROW_CLEAN_FRAC):
    """W2Ma row-level contexts: `W.corrupted_contexts` with `clean_frac` of the windows left clean (replicated here because
    that function fixes clean_frac = 0). `net` must be the released generator. label `corrupted` = ~plan.clean."""
    arm = {"plausible_wrong": "m2", "uniform_wrong": "m3"}[kind]
    m = W.MechArm(arm, tok, net, L0, prefix_emb, device, seed, rates, clean_frac=clean_frac, policy_temp=1.0,
                  frozen=W.MechArm.frozen_copy(net) if arm == "m2" else None)
    plan = m.draw_plan(x.shape[0])
    canon = m.canonical(x)
    repl = m.proposals(canon, plan)
    h = build_hierarchy(tok, x, CorruptionCfg(mode="mech"), policy_codes=repl, masks=plan.masks, canonical=canon)
    flags = W.mask_flags(tok.vq, plan.masks)
    n_masked = sum(mk.sum(1) for mk in plan.masks)
    corrupted = ~plan.clean
    assert bool((n_masked[corrupted] >= 1).all()), "a corrupted row without a masked slot"
    assert bool((n_masked[plan.clean] == 0).all()) and float(flags[plan.clean].abs().sum()) == 0.0, "a clean row with a mask or flag"
    return {"inputs": h["inputs"], "flags": flags, "targets": canon[:, m.l0:], "corrupted": corrupted, "clean": plan.clean,
            "masks": plan.masks, "canonical": canon}


def row_contexts_pair(tok, net, L0, prefix_emb, x, device, seed, rates, clean_frac=ROW_CLEAN_FRAC):
    """Plausible-wrong and uniform-wrong row contexts from the SAME seed: identical clean rows and masked slots (paired)."""
    pw, uw = (row_contexts(k, tok, net, L0, prefix_emb, x, device, seed, rates, clean_frac) for k in ("plausible_wrong", "uniform_wrong"))
    assert torch.equal(pw["clean"], uw["clean"]) and all(torch.equal(a, b) for a, b in zip(pw["masks"], uw["masks"])), \
        "plausible-wrong and uniform-wrong contexts must share clean rows and masks"
    return {"plausible_wrong": pw, "uniform_wrong": uw}


def level_span(levels, k):
    """[lo, hi) of level k's inputs in the generator's target layout (levels 1..K-1 in order)."""
    lo = sum(levels[1:k])
    return lo, lo + levels[k]


def pool_level(hid, levels, k):
    """(windows, positions, d) hidden states -> (windows, d): mean over the positions of level k's inputs."""
    lo, hi = level_span(levels, k)
    return hid[:, lo:hi].mean(1)


def row_probe_aucs(hid, corrupted, levels, js, n_train):
    """{k: held-out AUC} of the row-level probe: label = window corrupted, feature = level-k mean hidden state."""
    out = {}
    for k in js:
        f = pool_level(hid, levels, k)
        out[str(k)] = W.fit_probe_auc(f[:n_train, None], corrupted[:n_train, None], f[n_train:, None], corrupted[n_train:, None])
    return out


def too_weak_to_read(uniform_aucs, floor=ROW_AUC_FLOOR):
    """True iff the positive control (uniform-wrong) is below `floor` at every level (a missing AUC counts as below)."""
    return not any(a is not None and a >= floor for a in uniform_aucs.values())


def evaluate_model(net, flag, tok, prefix_emb, canon, perturb, ctxs, n_train, device, seed, rctxs=None):
    out = {"context_dependence_kl": W.context_dependence_kl(net, tok, prefix_emb, canon, perturb, device)}
    probe = {}
    for kind, ctx in ctxs.items():
        _, hid = W.forward_logits(net, prefix_emb, ctx["inputs"], device, hidden=True, keep_logits=False)
        y = ctx["contaminated"]
        probe[kind] = W.fit_probe_auc(hid[:n_train], y[:n_train], hid[n_train:], y[n_train:])
        c0 = ~ctx["masks"][0].any(1)            # exploratory: held-out windows whose level-0 code is NOT replaced
        te0 = c0[n_train:]
        probe[kind + "_l0_clean_windows_only"] = (W.fit_probe_auc(hid[:n_train][c0[:n_train]], y[:n_train][c0[:n_train]],
                                                                  hid[n_train:][te0], y[n_train:][te0])
                                                  if bool(c0[:n_train].any()) and bool(te0.any()) else None)
    out["probe_pooled"] = {"status": POOLED_STATUS, **probe}
    if rctxs is not None:
        levels = [int(l) for l in tok.vq.levels]
        rl = {}
        for kind, ctx in rctxs.items():
            _, hid = W.forward_logits(net, prefix_emb, ctx["inputs"], device, hidden=True, keep_logits=False)
            rl[kind] = row_probe_aucs(hid, ctx["corrupted"], levels, JS, n_train)
        out["probe_row_level"] = {**rl, "too_weak_to_read": too_weak_to_read(rl["uniform_wrong"])}
    if flag is not None:
        te = {k: (v[n_train:] if torch.is_tensor(v) else v) for k, v in ctxs["plausible_wrong"].items() if k in ("inputs", "flags", "targets")}
        out["flag_use"] = W.flag_use(net, flag, prefix_emb, te, device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True, help="directory holding the eight run directories (final.pt)")
    ap.add_argument("--run_dir", required=True, help="output directory; writes result.json")
    ap.add_argument("--seed", type=int, default=0, help="run seed of the arms (directory suffix _s<seed>)")
    ap.add_argument("--n_windows", type=int, default=512)
    ap.add_argument("--n_train", type=int, default=384)
    ap.add_argument("--eval_seed", type=int, default=2026)
    ap.add_argument("--only", default="", help="comma-separated subset of: released,m0..m7")
    a = ap.parse_args()
    device = pick_device()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    os.makedirs(a.run_dir, exist_ok=True)
    B = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"), device=device)
    tok, net = B.vqvae.eval(), B.net
    for p in tok.parameters():
        p.requires_grad_(False)
    L0 = B.ckpt_extra["inference"]["level0_unigram"].float().to(device)
    prefix_emb = lambda b: tok.encoder.wte(torch.full((b, 1), EOS, dtype=torch.long, device=device))
    released = {k: v.clone() for k, v in net.state_dict().items()}
    names = [n for n in (["released"] + sorted(ARM_DIRS)) if not a.only or n in a.only.split(",")]
    ckpts = {}
    for n in names:
        if n != "released":
            p = os.path.join(a.run_root, f"{ARM_DIRS[n]}_s{a.seed}", "final.pt")
            if not os.path.exists(p):
                sys.exit(f"missing final checkpoint for {n}: {ARM_DIRS[n]}_s{a.seed}/final.pt")
            ckpts[n] = p
    x = torch.from_numpy(examples("val", a.n_windows)).to(device)
    K = len(tok.vq.levels)
    rates = CorruptionCfg(mode="mech", mean_rate=0.5, spread=0.5).rates(K)
    levels, vocab = [int(l) for l in tok.vq.levels], [int(v) for v in tok.vq.vqvae_vocab_sizes]
    with torch.no_grad(), torch.autocast(device_type=device, enabled=False):
        canon = tok.vq.z_to_idx(tok._encode_pre_quant(x))
    perturb = W.make_perturbations(canon, levels, vocab, JS, a.eval_seed)
    # the corrupted windows come from the RELEASED generator (frozen proposer), whatever model is scored afterwards
    ctxs = {"plausible_wrong": W.corrupted_contexts("plausible_wrong", tok, net, L0, prefix_emb, x, device, a.eval_seed, rates),
            "uniform_wrong": W.corrupted_contexts("uniform_wrong", tok, net, L0, prefix_emb, x, device, a.eval_seed + 1, rates)}
    rctxs = row_contexts_pair(tok, net, L0, prefix_emb, x, device, a.eval_seed + 2, rates)    # W2Ma, same seed for both kinds
    res = {"protocol": {"n_windows": a.n_windows, "n_probe_train": a.n_train, "levels_j": list(JS), "eval_seed": a.eval_seed,
                        "rates": [round(r, 4) for r in rates], "probe": "logistic, L2 1e-2 on standardised features, L-BFGS 100",
                        "contaminated_label": "flag > 0 (the input position's cumulative reconstruction contains a replaced ancestor)",
                        "probe_row_level": {"clean_frac": ROW_CLEAN_FRAC, "seed": a.eval_seed + 2, "paired_masks": True,
                                            "levels_k": list(JS), "features": "mean of the last hidden states (after ln_f) over level k's positions",
                                            "label": "window corrupted (not a clean row)", "too_weak_below_auc": ROW_AUC_FLOOR,
                                            "amendment": "W2Ma"},
                        "precision": "fp32, TF32 off", "torch": torch.__version__},
           "runs": {n: (f"{ARM_DIRS[n]}_s{a.seed}" if n != "released" else "released") for n in names}, "models": {}}
    for n in names:
        flag = None
        if n == "released":
            net.load_state_dict(released)
        else:
            ck = torch.load(ckpts[n], map_location=device, weights_only=False)
            net.load_state_dict(ck["model"])
            if "flag" in ck:
                flag = W.FlagVector(tok.embed_dim).to(device)
                flag.load_state_dict(ck["flag"])
        res["models"][n] = evaluate_model(net, flag, tok, prefix_emb, canon, perturb, ctxs, a.n_train, device, a.eval_seed, rctxs=rctxs)
        print(n, json.dumps(res["models"][n]), flush=True)
    net.load_state_dict(released)
    with open(os.path.join(a.run_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote result.json")


if __name__ == "__main__":
    main()
