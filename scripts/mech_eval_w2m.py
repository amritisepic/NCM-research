"""W2M eval-only mechanism measures (docs/preregistration.md, "W2M"): the released model and the eight final checkpoints
of configs/scale/L_ts_m{0..7}_*.yaml, on the same windows and seeds. fp32, TF32 off; one JSON.

  python scripts/mech_eval_w2m.py --run_root <runs dir> --run_dir <out dir>        (writes <out dir>/result.json)

  (a) coarse-context dependence: 512 clean val windows, one code at level j (j = 1..6) changed to a uniformly random
      different code at a fixed random position, finer inputs rebuilt; mean KL(clean || perturbed) of the predictive
      distributions of the levels > j, per j.
  (b) replacement detectability: logistic probe on the last hidden states separating contaminated from clean input
      positions; inputs corrupted plausible-wrong (as M2: frozen released generator, canonical excluded) and
      uniform-wrong (as M3) at W1's rates, zero flags; trained on windows 0-383, AUC on windows 384-511.
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
from ncmr.corruption import CorruptionCfg  # noqa: E402
from ncmr.data import EOS, examples  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm  # noqa: E402
from ncmr.train import pick_device  # noqa: E402

ARM_DIRS = {"m0": "L_ts_m0_uniform", "m1": "L_ts_m1_policy", "m2": "L_ts_m2_frozen_pw", "m3": "L_ts_m3_uniform_wrong",
            "m4": "L_ts_m4_frozen_pw_flags", "m5": "L_ts_m5_frozen_pw_shuffled", "m6": "L_ts_m6_policy_usedparent",
            "m7": "L_ts_m7_uniform_a3rates"}
JS = (1, 2, 3, 4, 5, 6)


def evaluate_model(net, flag, tok, prefix_emb, canon, perturb, ctxs, n_train, device, seed):
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
    out["probe_auc"] = probe
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
    res = {"protocol": {"n_windows": a.n_windows, "n_probe_train": a.n_train, "levels_j": list(JS), "eval_seed": a.eval_seed,
                        "rates": [round(r, 4) for r in rates], "probe": "logistic, L2 1e-2 on standardised features, L-BFGS 100",
                        "contaminated_label": "flag > 0 (the input position's cumulative reconstruction contains a replaced ancestor)",
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
        res["models"][n] = evaluate_model(net, flag, tok, prefix_emb, canon, perturb, ctxs, a.n_train, device, a.eval_seed)
        print(n, json.dumps(res["models"][n]), flush=True)
    net.load_state_dict(released)
    with open(os.path.join(a.run_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote result.json")


if __name__ == "__main__":
    main()
