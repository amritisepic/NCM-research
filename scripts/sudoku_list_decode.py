"""C9: successive-cancellation-list (beam) decoding across NCM levels on the released Sudoku checkpoints.

Mirrors upstream `generate` for the Sudoku configs (query_level0, eager, greedy) but keeps L hypotheses
per puzzle. At each level every hypothesis is expanded into its argmax configuration plus the M-1
single-position alternatives with the smallest log-prob loss (the exact next-best configurations of a
factorized distribution, up to one flip), scored by the generator's own cumulative log-prob, and the best
L are kept. NFE is unchanged (one batched forward per level over all hypotheses); FLOPs grow ~L x.
Selection at the end: (a) highest log-prob ("list-LP"); (b) the highest-log-prob hypothesis whose decoded
board satisfies the Sudoku rules and the clues ("list-check", the CRC analogue; falls back to (a)).

Usage: python scripts/sudoku_list_decode.py --level hard --n 500 --L 4 --M 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.ncm_ref_loader import load_ncm, safe_pickle_load  # noqa: E402
from ncmr.sudoku import cells, is_solution  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--level", default="hard")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--L", type=int, default=4)
ap.add_argument("--M", type=int, default=4)
a = ap.parse_args()

B = load_ncm(os.path.join(ROOT, "artifacts", "sudoku", a.level))
net, vq = B.net, B.vqvae
assert net.query_level0 and not net.l0_ar
P = net.prefix_len
levels = list(vq.vq.levels)
puzzles = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", a.level, "puzzles.pickle"))[: a.n]
solutions = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", a.level, "solutions.pickle"))[: a.n]


@torch.no_grad()
def level_logprobs(prefix_emb, x_level, k):
    """Log-probs (N, l_k, V) for level k given its input embedding (already wte'd + level embedding)."""
    l_k = levels[k]
    h = torch.cat([prefix_emb, x_level], 1)
    m = torch.zeros(1, 1, P + l_k, P + l_k)
    m[:, :, :P, P:] = float("-inf")
    rope = (getattr(net, f"rope_cos_gen_{k}"), getattr(net, f"rope_sin_gen_{k}"))
    for blk in net.transformer.blocks:
        h = blk(h, attn_mask=m, rope_cos_sin=rope)
    h = net.transformer.ln_f(h)
    return F.log_softmax(net.lm_heads[k](h[:, P:]).float(), -1)


def expand(lp, M):
    """Top-M configurations of a factorized distribution, restricted to argmax + single flips.
    lp: (N, l, V). Returns codes (N, M, l) and their log-probs (N, M)."""
    N, l, V = lp.shape
    best_lp, best = lp.max(-1)                       # (N, l)
    second_lp, second = lp.topk(2, -1).values[..., 1], lp.topk(2, -1).indices[..., 1]
    gap = best_lp - second_lp                          # cost of flipping each position to its runner-up
    order = gap.argsort(-1)[:, : M - 1]                # cheapest flips
    codes = best[:, None, :].repeat(1, M, 1)
    scores = best_lp.sum(-1, keepdim=True).repeat(1, M)
    for j in range(1, M):
        pos = order[:, j - 1]
        codes[torch.arange(N), j, pos] = second[torch.arange(N), pos]
        scores[:, j] = scores[:, j] - gap[torch.arange(N), pos]
    return codes, scores


@torch.no_grad()
def list_decode(x_prefix_ids, L, M):
    Bsz = x_prefix_ids.shape[0]
    z_prefix = vq.encoder.wte(x_prefix_ids)
    pe = net.transformer.wte(z_prefix)                 # (B, P, d)
    lvl_emb = net.transformer.wle(net.lvl_1L)          # (1, sum(levels), d)
    # level 0 from learned queries
    q = net.transformer.wte(net.build_l0_query_input(Bsz, pe.dtype)) + lvl_emb[:, : levels[0]]
    lp = level_logprobs(pe, q, 0)
    codes, sc = expand(lp, max(M, L))
    keep = sc.topk(L, -1).indices                      # (B, L)
    hyp_codes = [torch.gather(codes, 1, keep[..., None].expand(-1, -1, levels[0]))]   # list of (B, L, l_k)
    hyp_score = torch.gather(sc, 1, keep)
    acc = levels[0]
    for k in range(len(levels) - 1):
        # rebuild z_hat for each hypothesis from its codes so far (cheap: tokenizer-side ops only)
        flat = torch.cat(hyp_codes, -1).reshape(Bsz * L, -1)
        z = torch.zeros(Bsz * L, levels[-1], vq.embed_dim)
        off = 0
        for j in range(k + 1):
            e = vq.vq.embeddings[j](flat[:, off:off + levels[j]])
            qj = vq.vq._up_to_full(e, j, levels[-1]) if j != len(levels) - 1 else e
            z = z + vq.vq._apply_quant_resi(qj, j)
            off += levels[j]
        x_in = F.interpolate(z.transpose(1, 2), size=(levels[k + 1],), mode="area").transpose(1, 2)
        x = net.transformer.wte(x_in) + lvl_emb[:, acc:acc + levels[k + 1]]
        lp = level_logprobs(pe.repeat_interleave(L, 0), x, k + 1)       # (B*L, l, V)
        c, s = expand(lp, M)                                             # (B*L, M, l), (B*L, M)
        s = (hyp_score.reshape(Bsz * L, 1) + s).reshape(Bsz, L * M)
        c = c.reshape(Bsz, L * M, levels[k + 1])
        keep = s.topk(L, -1).indices
        parent = keep // M
        hyp_codes = [torch.gather(h, 1, parent[..., None].expand(-1, -1, h.shape[-1])) for h in hyp_codes]
        hyp_codes.append(torch.gather(c, 1, keep[..., None].expand(-1, -1, levels[k + 1])))
        hyp_score = torch.gather(s, 1, keep)
        acc += levels[k + 1]
    full = torch.cat(hyp_codes, -1)                    # (B, L, sum(levels))
    boards = vq.decoder(vq.vq.idx_to_z_q(full.reshape(Bsz * L, -1))).argmax(-1).reshape(Bsz, L, -1)
    return boards, hyp_score


t0 = time.time()
res = {"greedy_L1": 0, "list_lp": 0, "list_check": 0, "oracle_any": 0}
for s in range(0, len(puzzles), a.bs):
    batch = puzzles[s:s + a.bs]
    x = torch.tensor([list(p)[:P] for p in batch])
    boards1, _ = list_decode(x, 1, 1)
    boardsL, scores = list_decode(x, a.L, a.M)
    for j, p in enumerate(batch):
        pc, sc_ = cells(p), cells(solutions[s + j])
        g = cells(boards1[j, 0].tolist())
        res["greedy_L1"] += int(g == sc_)
        hyps = [cells(boardsL[j, i].tolist()) for i in range(a.L)]
        order = scores[j].argsort(descending=True).tolist()
        res["list_lp"] += int(hyps[order[0]] == sc_)
        chk = next((hyps[i] for i in order if hyps[i] is not None and is_solution(hyps[i], pc)), hyps[order[0]])
        res["list_check"] += int(chk == sc_)
        res["oracle_any"] += int(any(h == sc_ for h in hyps))
    print(f"  {s + len(batch)}/{len(puzzles)} {time.time() - t0:.0f}s {res}", flush=True)
out = {"run": f"c9_sudoku_list_decode_{a.level}_L{a.L}_M{a.M}", "n": len(puzzles), "L": a.L, "M": a.M,
       **{k: 100 * v / len(puzzles) for k, v in res.items()}, "nfe_per_sample": len(levels),
       "flops_multiplier_vs_greedy": a.L, "wall_s": time.time() - t0}
json.dump(out, open(os.path.join(ROOT, "results", f"{out['run']}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
