"""B2: fixed-point (Jacobi) decoding across NCM levels, on released checkpoints. No training.

Sequential NCM decoding computes level k from the codes of levels < k, one level per pass. Jacobi decoding
updates *all* levels in one parallel pass (the training-style block-diagonal forward) from the current guess
of every level, and repeats. Sampling uses Gumbel-max with noise fixed per (level, position, code), so the
sequential sample with the same noise is the unique fixed point; level k is exact after k+1 passes, hence
convergence in at most K passes. Stopping rules (per sample):
  exact  : stop at the first pass that changes no code (output identical to sequential decoding);
  stable : stop when the decoded output (board / tokens) is unchanged for two consecutive passes (not exact);
  check  : Sudoku only — stop at the first pass whose decoded board satisfies the rules and the clues.
NFE = passes run (including the pass that detects convergence). FLOPs ratio vs sequential eager decoding.
Text note: the fixed point is the sequential Gumbel-max sample computed with the parallel pass's logits,
which match the per-level pass to ~1e-4, so it is an exact sample from the model up to float error.

Usage: python scripts/jacobi_levels.py --task sudoku --level hard --n 2000
       python scripts/jacobi_levels.py --task tinystories --n 256 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.data import EOS, PAD, cut_at_eos  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm, safe_pickle_load  # noqa: E402
from ncmr.sudoku import cells, is_solution  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="sudoku", choices=["sudoku", "tinystories", "wmt14"])
ap.add_argument("--level", default="hard")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--bs", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
torch.manual_seed(a.seed)

ck = {"sudoku": os.path.join(ROOT, "artifacts", "sudoku", a.level), "tinystories": os.path.join(ROOT, "artifacts", "tinystories"),
      "wmt14": os.path.join(ROOT, "artifacts", "wmt14")}[a.task]
DEV = a.device
B = load_ncm(ck, device=DEV)
net, vq = B.net, B.vqvae
P = net.prefix_len
levels = [int(l) for l in vq.vq.levels]
K, S = len(levels), sum(levels)
first = net._lvl_first                 # 0 if level 0 is predicted (Sudoku), 1 if drawn from a unigram (text)
greedy = a.task in ("sudoku", "wmt14")   # released configs decode greedily (WMT14: top_k=1)
PAD_ID = getattr(net, "prefix_pad_id", None)


@torch.no_grad()
def jacobi(prefix_ids, l0=None):
    """Returns the list of code tensors produced by each pass (pass 1 starts from all-zero codes)."""
    b = prefix_ids.shape[0]
    z_prefix = vq.encoder.wte(prefix_ids)
    q0 = [net.build_l0_query_input(b)] if net.query_level0 else []
    V = vq.vq.vqvae_vocab_sizes[0]
    gumbel = None if greedy else -torch.log(-torch.log(torch.rand(b, S - (0 if first == 0 else levels[0]), V, device=DEV).clamp_min(1e-20)))
    codes = torch.zeros(b, S, dtype=torch.long, device=DEV)
    if l0 is not None:
        codes[:, :levels[0]] = l0
    pm = (prefix_ids[:, :P] == PAD_ID) if (PAD_ID is not None and a.task == "wmt14") else None
    hist = []
    for _ in range(K + 1):
        z_in = vq.vq.idx_to_ncm_input(codes)
        logits = net(torch.cat([z_prefix, *q0, z_in], 1), prefix_pad_mask=pm).float()
        new_pred = logits.argmax(-1) if greedy else (logits + gumbel[:, -logits.shape[1]:]).argmax(-1)
        new = codes.clone()
        new[:, S - new_pred.shape[1]:] = new_pred   # predict_level0 / query_level0 models also emit level 0
        hist.append(new)
        if torch.equal(new, codes):
            break
        codes = new
    return hist


def cut_at_eos_id(ids, eos):
    return ids[:ids.index(eos)] if eos in ids else ids


def decode_ids(codes):
    with torch.no_grad():
        return vq.decoder(vq.vq.idx_to_z_q(codes)).argmax(-1)


t0 = time.time()
rows = []
if a.task == "sudoku":
    puzzles = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", a.level, "puzzles.pickle"))[: a.n]
    solutions = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", a.level, "solutions.pickle"))[: a.n]
    batches = [(s, torch.tensor([list(p) for p in puzzles[s:s + a.bs]]), None) for s in range(0, len(puzzles), a.bs)]
elif a.task == "wmt14":
    from transformers import AutoTokenizer
    srcs = safe_pickle_load(os.path.join(ROOT, "data", "wmt14", "sources.pickle"))[: a.n]
    refs = safe_pickle_load(os.path.join(ROOT, "data", "wmt14", "reference_translations.pickle"))[: a.n]
    src_tok = AutoTokenizer.from_pretrained(str(B.cfg.tokenizer.pre_tokenizer.model_id))
    PADP = B.pre_tokenizer.pad_token_id
    pre = [(src_tok(t)["input_ids"][:P] + [PADP] * P)[:P] for t in srcs]   # as upstream eval_shard.prefix()
    batches = [(s, torch.tensor(pre[s:s + a.bs]), None) for s in range(0, len(pre), a.bs)]
else:
    L0 = B.ckpt_extra["inference"]["level0_unigram"].float()
    batches = []
    for s in range(0, a.n, a.bs):
        b = min(a.bs, a.n - s)
        l0 = torch.multinomial(L0, b * levels[0], replacement=True).view(b, levels[0])
        batches.append((s, torch.full((b, 1), EOS), l0))

for s, x, l0 in batches:
    x = x.to(DEV)
    l0 = l0.to(DEV) if l0 is not None else None
    if greedy:
        with torch.no_grad():
            seq = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0,
                               cfg_null_eos=getattr(B.pre_tokenizer, "eos_token_id", None))
    hist = jacobi(x, l0)
    final = hist[-1]
    outs = [decode_ids(h) for h in hist]
    for j in range(x.shape[0]):
        conv = next(t for t in range(len(hist)) if torch.equal(hist[t][j], final[j]))
        exact = min(conv + 2, len(hist))
        stable = next((t + 1 for t in range(1, len(outs)) if torch.equal(outs[t][j], outs[t - 1][j])), len(outs))
        r = {"exact": exact, "stable": stable, "stable_matches_final": bool(torch.equal(outs[stable - 1][j], outs[-1][j])),
             "first_final_output": next(t + 1 for t in range(len(outs)) if torch.equal(outs[t][j], outs[-1][j]))}
        if greedy:
            r["fixed_point_equals_sequential"] = bool(torch.equal(final[j], seq[j]))
        if a.task == "wmt14":
            eos = B.pre_tokenizer.eos_token_id
            dec = lambda ids: B.pre_tokenizer.decode([t for t in cut_at_eos_id(ids, eos) if t not in (eos, PADP)], skip_special_tokens=True)
            r["hyp_exact"] = dec(outs[-1][j].tolist())
            r["hyp_stable"] = dec(outs[stable - 1][j].tolist())
            r["ref"] = refs[s + j] if isinstance(refs[s + j], str) else dec(list(refs[s + j]))
        elif greedy:
            pc, sc = cells(list(puzzles[s + j])), cells(list(solutions[s + j]))
            boards = [cells(o[j].tolist()) for o in outs]
            r["solved"] = boards[-1] == sc
            chk = next((t + 1 for t, bd in enumerate(boards) if bd is not None and is_solution(bd, pc)), len(boards))
            r["check"] = chk
            r["check_solved"] = boards[chk - 1] == sc
            r["stable_solved"] = boards[stable - 1] == sc
        if a.task == "tinystories":
            r["text_final"] = [t for t in cut_at_eos(outs[-1][j].tolist()) if t != PAD]
            r["text_stable"] = [t for t in cut_at_eos(outs[stable - 1][j].tolist()) if t != PAD]
        rows.append(r)
    print(f"  {s + x.shape[0]}  mean exact passes {np.mean([r['exact'] for r in rows]):.2f}  {time.time() - t0:.0f}s", flush=True)

K_eff = K - first + (1 if getattr(net, "predict_level0", False) else 0)   # sequential NFE (+1: level 0 from the prefix pass)
seq_positions = K_eff * P + (S - (0 if first == 0 else levels[0]))   # eager sequential decoding
jac_positions = P + (S - (0 if first == 0 else levels[0]))
res = {"run": f"b2_jacobi_{a.task}" + {"sudoku": f"_{a.level}", "tinystories": f"_seed{a.seed}", "wmt14": f"_n{a.n}"}[a.task],
       "n": len(rows), "sequential_nfe": K_eff,
       "exact_passes_mean": float(np.mean([r["exact"] for r in rows])),
       "exact_passes_hist": np.bincount([r["exact"] for r in rows]).tolist(),
       "stable_passes_mean": float(np.mean([r["stable"] for r in rows])),
       "stable_output_equals_exact_pct": 100 * float(np.mean([r["stable_matches_final"] for r in rows])),
       "first_final_output_pass_mean": float(np.mean([r["first_final_output"] for r in rows])),
       "flops_per_pass_ratio_vs_sequential_total": jac_positions / seq_positions,
       "wall_s": time.time() - t0, "device": DEV}
res["exact_nfe_saving_pct"] = 100 * (1 - res["exact_passes_mean"] / K_eff)
res["exact_flops_ratio"] = res["exact_passes_mean"] * res["flops_per_pass_ratio_vs_sequential_total"]
if a.task == "wmt14":
    import sacrebleu
    hy_e, hy_s, rf = [r["hyp_exact"] for r in rows], [r["hyp_stable"] for r in rows], [r["ref"] for r in rows]
    res.update({"fixed_point_equals_sequential_pct": 100 * float(np.mean([r["fixed_point_equals_sequential"] for r in rows])),
                "bleu_lc_exact": sacrebleu.corpus_bleu(hy_e, [rf], lowercase=True).score,
                "bleu_lc_stable_rule": sacrebleu.corpus_bleu(hy_s, [rf], lowercase=True).score,
                "stable_rule_identical_output_pct": 100 * float(np.mean([e == t for e, t in zip(hy_e, hy_s)]))})
elif greedy:
    res.update({"fixed_point_equals_sequential_pct": 100 * float(np.mean([r["fixed_point_equals_sequential"] for r in rows])),
                "solved_pct_exact": 100 * float(np.mean([r["solved"] for r in rows])),
                "solved_pct_stable_rule": 100 * float(np.mean([r["stable_solved"] for r in rows])),
                "check_passes_mean": float(np.mean([r["check"] for r in rows])),
                "solved_pct_check_rule": 100 * float(np.mean([r["check_solved"] for r in rows]))})
else:
    os.makedirs(os.path.join(ROOT, "runs", "samples"), exist_ok=True)
    json.dump({"ids": [r["text_final"] for r in rows], "groups": [a.seed] * len(rows), "nfe_per_sample": res["exact_passes_mean"]},
              open(os.path.join(ROOT, "runs", "samples", f"jacobi_exact_seed{a.seed}.json"), "w"))
    json.dump({"ids": [r["text_stable"] for r in rows], "groups": [a.seed] * len(rows), "nfe_per_sample": res["stable_passes_mean"]},
              open(os.path.join(ROOT, "runs", "samples", f"jacobi_stable_seed{a.seed}.json"), "w"))
json.dump(res, open(os.path.join(ROOT, "results", f"{res['run']}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
