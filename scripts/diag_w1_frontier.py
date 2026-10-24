"""Eval-only diagnostic of wave 1's re-quantized-label arm (W1d in docs/preregistration.md; exploratory): does A2's
gain over the control hold across sampling temperatures, and at which levels does it arise? For one generator (the
released model A0, or a wave-1 `final.pt`):

  (1) temperature frontier: unconditional samples at each --temps value, one group of --n per --seeds entry, with W1b's
      sampler (level 0 from the unigram, `net.generate`, fp32, TF32 off, the same seeds as W1b's groups): Gen-PPL (GPT-2
      large, TinyStories-33M), unigram entropy, repetition, MAUVE against as many human val windows (5 quantization
      seeds). At T = 1 each group must reproduce the wave-1 result file's group for the same seed (--expect).
  (2) oracle-intervention curve (R020): the canonical codes of levels 0..j of the first --n_oracle val windows are
      supplied and the generator samples levels j+1.. (Gumbel-max with one seed per j, shared by every model);
      Gen-PPL (GPT-2 large) and entropy per j, in groups of --n windows; j = -1 is unconditional sampling.
Windows and seeds are identical for every model, so A1 - A0 and A2 - A1 differences are paired.

  python scripts/diag_w1_frontier.py --name A2 --ckpt runs/L_ts_requant_s0/final.pt --run_dir runs/w1d_A2 \
      --expect results/colab/w1_L_ts_requant/L_ts_requant_s0__result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.data import EOS, PAD, cut_at_eos, examples  # noqa: E402


@torch.no_grad()
def draw(net, tok, L0, n, seed, temp, dev, bs=256):
    """Unconditional samples exactly as train_gen.py's `draw_samples` draws them (seeded per group, fp32)."""
    l0 = int(tok.vq.levels[0])
    g = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    out = []
    for s in range(0, n, bs):
        b = min(bs, n - s)
        lv0 = torch.multinomial(L0, b * l0, replacement=True, generator=g).view(b, l0)
        with torch.autocast(device_type=dev, enabled=False):
            codes = net.generate(lv0, torch.full((b, 1), EOS, device=dev), temperature=temp, top_k=0, top_p=1.0,
                                 cfg_scale=0.0, cfg_null_eos=EOS)
            ids = tok.decoder(tok.vq.idx_to_z_q(codes)).argmax(-1)
        out += [[t for t in cut_at_eos(r) if t != PAD] for r in ids.tolist()]
    return out


@torch.no_grad()
def oracle_sample(net, tok, L0, x, j, g, dev, temp=1.0, return_codes=False):
    """Canonical codes of levels 0..j of the windows x, the generator's samples for levels j+1.. (R020's
    `tf_levels_diag.sample_from`, fp32). j = -1: level 0 from the unigram, everything else sampled."""
    vq, levels = tok.vq, [int(l) for l in tok.vq.levels]
    K, S, off = len(levels), sum(levels), np.cumsum([0] + levels)
    b = x.shape[0]
    with torch.autocast(device_type=dev, enabled=False):
        codes = torch.zeros(b, S, dtype=torch.long, device=dev)
        if j >= 0:
            codes[:, :off[j + 1]] = vq.z_to_idx(tok._encode_pre_quant(x))[:, :off[j + 1]]
        else:
            codes[:, :levels[0]] = torch.multinomial(L0, b * levels[0], replacement=True, generator=g).view(b, levels[0])
        pre = tok.encoder.wte(torch.full((b, 1), EOS, dtype=torch.long, device=dev))
        for k in range(max(j + 1, 1), K):
            # one block-diagonal pass; level k's logits depend only on levels < k, which are final
            logits = net(torch.cat([pre, vq.idx_to_ncm_input(codes)], 1)).float()
            lk = logits[:, off[k] - levels[0]:off[k + 1] - levels[0]]
            u = torch.rand(lk.shape, device=dev, generator=g).clamp_min(1e-20)
            codes[:, off[k]:off[k + 1]] = (lk / temp - torch.log(-torch.log(u))).argmax(-1)   # Gumbel-max
        ids = tok.decoder(vq.idx_to_z_q(codes)).argmax(-1)
    out = [[t for t in cut_at_eos(r) if t != PAD] for r in ids.tolist()]
    return (out, codes) if return_codes else out


def reproduction(per_group, seeds, expect, tol=0.01):
    """Compare this run's T = 1 groups with a wave-1 result's groups for the same seeds. Both lists are in sorted-seed
    order (evaluate_samples sorts its groups). ok only if every requested seed has a wave-1 group, every value is finite
    and no group differs by more than tol (W1d: a wrong checkpoint or sampler must not pass; Codex PR 7)."""
    fe = expect["final_eval"]
    theirs = dict(zip(sorted(int(s) for s in expect["config"]["sample_eval"]["final_seeds"]),
                      fe["genppl_gpt2-large_per_group"]))
    ours = dict(zip(sorted(seeds), per_group))
    pairs = {s: [ours[s], theirs[s]] for s in sorted(ours) if s in theirs}
    missing = sorted(s for s in ours if s not in theirs)
    finite = all(np.isfinite(v) for pair in pairs.values() for v in pair)
    diff = max((abs(u - v) for u, v in pairs.values()), default=float("inf")) if finite else float("inf")
    return {"pairs": pairs, "missing_seeds": missing, "max_abs_diff": diff, "tol": tol,
            "ok": bool(pairs) and not missing and finite and diff <= tol}


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="", help="a wave-1 final.pt; empty = the released model (A0)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.8, 0.9, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--n", type=int, default=256, help="samples per group")
    ap.add_argument("--js", type=int, nargs="*", default=[-1, 3, 5, 7])
    ap.add_argument("--n_oracle", type=int, default=1024)
    ap.add_argument("--oracle_seed", type=int, default=0)
    ap.add_argument("--expect", default="", help="wave-1 result.json the T = 1 groups must reproduce")
    ap.add_argument("--tol", type=float, default=0.01, help="largest per-group Gen-PPL difference that counts as reproduced")
    return ap


def main():
    a = build_parser().parse_args()
    if a.expect and 1.0 not in a.temps:
        sys.exit("--expect needs T = 1.0 in --temps (the reproduction check runs at T = 1)")
    from ncmr.eval_text import evaluate_samples, mauve_features
    from ncmr.ncm_ref_loader import load_ncm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False   # W1b: fp32 sampling and scoring
    B = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"), device=dev)
    tok, net, bpe = B.vqvae.eval(), B.net, B.pre_tokenizer
    L0 = B.ckpt_extra["inference"]["level0_unigram"].float().to(dev)
    ident = {"model": "released"}
    if a.ckpt:
        st = torch.load(a.ckpt, map_location=dev, weights_only=False)   # our own train_gen.py final.pt
        if st.get("init") != "released":
            sys.exit("wave-1 checkpoints fine-tune the released model; this one has init=" + str(st.get("init")))
        net.load_state_dict(st["model"])                                 # strict: every tensor must match
        L0 = st["level0_unigram"].float().to(dev)
        ident = {"model": os.path.join(os.path.basename(os.path.dirname(a.ckpt)), os.path.basename(a.ckpt)),
                 "sha256": digest(a.ckpt)}
    net.eval()
    os.makedirs(a.run_dir, exist_ok=True)
    t0 = time.time()
    out = {"name": a.name, "checkpoint": ident, "device": dev,
           "gpu": torch.cuda.get_device_name() if dev == "cuda" else None, "torch": torch.__version__,
           "eval_precision": "fp32 sampling, scoring and MAUVE featurization, TF32 off",
           "args": {k: v for k, v in vars(a).items() if k not in ("ckpt", "run_dir", "expect")},
           "frontier": {}, "oracle": {}}
    n_f = a.n * len(a.seeds)
    # fp32 featurization (W1b's helper default is bf16 autocast), so its own reference cache (Codex PR 7)
    cache = os.path.join(ROOT, "runs", "cache", f"mauve_ref_val_{n_f}_fp32.npy")
    if os.path.exists(cache):
        ref = np.load(cache)
    else:
        ref = mauve_features([bpe.decode(cut_at_eos(r)) for r in examples("val", n_f)], dev, amp=False)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, ref)
    samples = {}
    # T = 1 first: with --expect, a run that does not reproduce wave 1 stops before spending anything else
    for temp in sorted(a.temps, key=lambda t: abs(t - 1.0) > 1e-9):
        ids, groups = [], []
        for sd in a.seeds:
            o = draw(net, tok, L0, a.n, sd, temp, dev)
            ids += o
            groups += [sd] * len(o)
        reenc = [bpe.encode(bpe.decode(s)) for s in ids]   # token statistics on the re-encoding, as train_gen.py
        res = evaluate_samples(reenc, bpe.decode, groups, scorers=("gpt2-large", "roneneldan/TinyStories-33M"),
                               ref_feats=ref, device=dev, mauve_amp=False)
        if a.expect and abs(temp - 1.0) < 1e-9:
            res["reproduction"] = rep = reproduction(res["genppl_gpt2-large_per_group"], a.seeds,
                                                     json.load(open(a.expect)), a.tol)
            if not rep["ok"]:
                out["frontier"][str(temp)] = res
                out["reproduction_failed"] = True
                json.dump(out, open(os.path.join(a.run_dir, "result.json"), "w"), indent=1)
                sys.exit(f"T = 1 does not reproduce wave 1 (missing seeds {rep['missing_seeds']}, max |diff| "
                         f"{rep['max_abs_diff']:.4f} > {a.tol}): wrong checkpoint, sampler or GPU type; nothing else run")
        out["frontier"][str(temp)] = res
        samples[str(temp)] = {"ids": ids, "groups": groups}
        print(f"T={temp}: Gen-PPL {res['genppl_gpt2-large']:.2f} H {res['entropy']:.3f} MAUVE {res['mauve']:.3f}"
              + (f" | reproduced, max |diff| {res['reproduction']['max_abs_diff']:.4f}" if "reproduction" in res else "")
              + f" | {time.time() - t0:.0f}s", flush=True)
    X = torch.from_numpy(examples("val", a.n_oracle))
    for j in a.js:
        g = torch.Generator(device=dev).manual_seed(a.oracle_seed * 100 + j + 1)   # R020's per-j seed
        ids = []
        for s in range(0, a.n_oracle, 128):
            ids += oracle_sample(net, tok, L0, X[s:s + 128].to(dev), j, g, dev)
        groups = [i // a.n for i in range(len(ids))]
        reenc = [bpe.encode(bpe.decode(s)) for s in ids]
        res = evaluate_samples(reenc, bpe.decode, groups, scorers=("gpt2-large",), ref_feats=None, device=dev)
        out["oracle"][str(j)] = res
        print(f"j={j}: Gen-PPL {res['genppl_gpt2-large']:.2f} H {res['entropy']:.3f} | {time.time() - t0:.0f}s", flush=True)
    out["wall_s"] = time.time() - t0
    json.dump(out, open(os.path.join(a.run_dir, "result.json"), "w"), indent=1)
    json.dump(samples, open(os.path.join(a.run_dir, "samples_final.json"), "w"))
    print("wrote", os.path.join(a.run_dir, "result.json"))


if __name__ == "__main__":
    main()
