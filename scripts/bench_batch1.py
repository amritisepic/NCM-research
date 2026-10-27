"""D1 (docs/preregistration.md): batch-1 wall-clock gate for criterion (d). Per-item latency of sequential and Jacobi
decoding of the released WMT14 De-En NCM (J0), J2-F, and the released Sudoku-hard NCM, CUDA-synchronised per item,
warm-up excluded, fp32 with TF32 off (primary: the precision of every quality number) and bf16 autocast (secondary).

Configurations (each decodes one item from its token prefix to its output text / board, all inside the timed region):
  seq            greedy sequential decoding, one forward call per level (WMT14: level 0 from the source prefix, then
                 levels 1-10; Sudoku: level-0 queries, then levels 1-11)
  jacobi exact   greedy Jacobi passes over every level until no code changes (output = the sequential output)
  jacobi stable1 exact, or stop at the first pass t >= 2 whose decoded text equals pass t-1's (R042's "stable x1";
                 the per-pass tokenizer decode and detokenisation are inside the timed region)
  jacobi verifier (Sudoku) exact, or stop at the first pass whose decoded board satisfies the rules and the clues
Every configuration runs in two implementations: `eager` (upstream: the prefix is re-run in every forward call; for
seq that is `generate`, which re-encodes the 96-token source / 91-token puzzle in each of its 11 / 12 calls) and
`cached` (src/ncmr/cached_decode.py: the prefix once per item, with its pad mask; later calls run the target rows
only). The gate compares the faster implementation of each side. Per item, every configuration runs `--repeats` times
in a seeded random interleaved order; an item's latency is the median of its repeats.
FLOPs per item: executed FLOPs counted by torch's FlopCounterMode with math attention (matmuls, attention scores and
mixing, convolutions, heads, tokenizer decodes) from the per-call counts and the measured passes; plus the generator
blocks alone in src/ncmr/flops.py's convention (24 d^2 + 4 d * keys per row and layer, keys = every key the call's
dense attention computes).

  python scripts/bench_batch1.py --task wmt14 --run_dir runs/d1_wmt --j2f_ckpt runs/L_wmt_consist_frozen_s0/final.pt
  python scripts/bench_batch1.py --task sudoku --run_dir runs/d1_sudoku
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr import cached_decode as CD  # noqa: E402
from ncmr import flops as FL  # noqa: E402

GATE = 0.75          # registered: Jacobi + stop rule <= 0.75 x sequential mean latency (fp32)
MAX_MISMATCH = 2     # registered: at most 2 of 256 items may differ from the saved outputs (float-order near-ties)


# ------------------------------------------------------------------------------------------------ configurations
class Item:
    def __init__(self, prefix_ids, extra=None):
        self.x, self.extra = prefix_ids, extra or {}


def memo(f):
    """f(codes) with the last result kept: the output after a stable / verifier / fixed-point stop is the decode the
    rule already made (a fixed point repeats the previous pass's codes), so it is not decoded twice."""
    last = {}

    def g(codes):
        if "c" in last and torch.equal(codes, last["c"]):
            return last["v"]
        last["c"], last["v"] = codes, f(codes)
        return last["v"]
    return g


def make_configs(task, models, tok, detok, max_passes):
    """[(name, side, fn(item) -> (output, passes, stop))]. side: 'seq' or the Jacobi rule; models: {'J0': net, ...}.
    Sequential 'passes' = its forward calls (K)."""
    vq = tok.vq
    K, S = len(vq.levels), sum(int(l) for l in vq.levels)

    def ids_of(codes):
        return tok.decoder(vq.idx_to_z_q(codes)).argmax(-1)[0].tolist()

    out = []
    if task == "wmt14":
        text = lambda codes: detok(ids_of(codes))
        for impl in ("eager", "cached"):
            out.append((f"J0 seq {impl}", "seq",
                        lambda it, c=impl == "cached": (text(CD.seq_greedy(models["J0"], it.x, c)), K, "seq")))
        for name, rule in (("J0", "exact"), ("J0", "stable1"), ("J2-F", "stable1")):
            if name not in models:
                continue
            for impl in ("eager", "cached"):
                def fn(it, net=models[name], rule=rule, c=impl == "cached"):
                    dec = memo(text)
                    codes, t, stop = CD.jacobi_stop(CD.make_pass(net, it.x, c), S, it.x.device, max_passes, rule,
                                                    decode=dec, initial_fixed_ok=True)   # static prefix (audit 4, A4-4)
                    return dec(codes), t, stop
                out.append((f"{name} jacobi {rule} {impl}", rule, fn))
    else:
        from ncmr.sudoku import cells, is_solution
        board = lambda codes: cells(ids_of(codes))
        for impl in ("eager", "cached"):
            out.append((f"seq {impl}", "seq",
                        lambda it, c=impl == "cached": (board(CD.seq_greedy(models["S"], it.x, c)), K, "seq")))
        for rule in ("exact", "verifier"):
            for impl in ("eager", "cached"):
                def fn(it, rule=rule, c=impl == "cached"):
                    dec = memo(board)
                    chk = lambda codes: (lambda b: b is not None and is_solution(b, it.extra["clues"]))(dec(codes))
                    codes, t, stop = CD.jacobi_stop(CD.make_pass(models["S"], it.x, c), S, it.x.device, max_passes,
                                                    rule, check=chk)
                    return dec(codes), t, stop
                out.append((f"jacobi {rule} {impl}", rule, fn))
    return out


def precision_ctx(prec, device):
    if prec == "fp32":
        return torch.autocast(device_type=device, enabled=False)
    if prec == "bf16":
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    raise ValueError(prec)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_timing(configs, items, device, prec, repeats=3, warmup=16, seed=0):
    """Warm-up: every configuration on the first `warmup` items, untimed. Then, per item, every configuration
    `repeats` times in a seeded random interleaved order. Returns {name: {"lat": [per-item median s], "out": [...],
    "passes": [...], "stop": [...], "nondeterministic": [item indices whose repeats disagreed]}}."""
    res = {n: {"lat": [], "out": [], "passes": [], "stop": [], "nondeterministic": []} for n, _, _ in configs}
    with torch.no_grad(), precision_ctx(prec, device):
        for it in items[:warmup]:
            for _, _, fn in configs:
                fn(it)
        sync(device)
        for i, it in enumerate(items):
            times = {n: [] for n, _, _ in configs}
            outs = {n: [] for n, _, _ in configs}
            for r in range(repeats):
                order = np.random.default_rng([seed, i, r]).permutation(len(configs))
                for j in order:
                    n, _, fn = configs[j]
                    sync(device)
                    t0 = time.perf_counter()
                    o = fn(it)
                    sync(device)
                    times[n].append(time.perf_counter() - t0)
                    outs[n].append(o)
            for n in times:
                d = res[n]
                d["lat"].append(float(np.median(times[n])))
                d["out"].append(outs[n][0][0])
                d["passes"].append(int(outs[n][0][1]))
                d["stop"].append(outs[n][0][2])
                if any(o[:2] != outs[n][0][:2] for o in outs[n][1:]):
                    d["nondeterministic"].append(i)
    return res


def summarise(lat, passes):
    a = np.asarray(lat) * 1e3
    return {"mean_ms": float(a.mean()), "se_ms": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
            "median_ms": float(np.median(a)), "p10_ms": float(np.percentile(a, 10)), "p90_ms": float(np.percentile(a, 90)),
            "passes_mean": float(np.mean(passes)), "passes_hist": np.bincount(passes).tolist()}


def ratio_ci(num, den, n_boot=1000, seed=0):
    """Ratio of mean latencies with a paired bootstrap 95% interval over items."""
    num, den = np.asarray(num), np.asarray(den)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return {"ratio": float(num.mean() / den.mean()), "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}


# ------------------------------------------------------------------------------------------------ FLOPs
def counted_flops(fn):
    """Executed FLOPs of fn() by torch's FlopCounterMode, with math attention so that the attention matmuls are
    counted as run (every key of the dense mask)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=False), fc:
        fn()
    return int(fc.get_total_flops())


def generator_block_flops(net):
    """Generator blocks only, flops.py convention, per call structure: {"prefix", "pass", "pass_cached", "seq_eager",
    "seq_cached"} (FLOPs) and the positions (rows) each processes."""
    d, L, P = int(net.config.n_embd), int(net.config.n_layer), int(net.prefix_len)
    lv = [int(l) for l in net.levels]
    tgt = lv[net._lvl_first:]
    T = sum(tgt)
    bf = lambda q, k: FL.block_fwd(d, L, q, k)
    calls_eager, calls_cached = [], []
    if net.predict_level0:                       # WMT14: level 0 from a prefix-only call
        calls_eager.append((P, P))
        calls_cached.append((P, P))
    else:
        calls_cached.append((P, P))              # the prefix once
    for lk in tgt:                               # Sudoku: tgt starts with the level-0 queries
        calls_eager.append((P + lk, P + lk))
        calls_cached.append((lk, P + lk))
    return {"pass": bf(P + T, P + T), "pass_rows": P + T, "prefix": bf(P, P), "prefix_rows": P,
            "pass_cached": bf(T, P + T), "pass_cached_rows": T,
            "seq_eager": sum(bf(q, k) for q, k in calls_eager), "seq_eager_rows": sum(q for q, _ in calls_eager),
            "seq_cached": sum(bf(q, k) for q, k in calls_cached), "seq_cached_rows": sum(q for q, _ in calls_cached)}


def flops_per_item(name, side, passes, unit):
    """(executed counted FLOPs, generator-block FLOPs, generator rows) for one item of configuration `name`."""
    cached = name.endswith("cached")
    dec = passes if side in ("stable1", "verifier") else 1      # per-pass decodes, or the final output decode
    if side == "seq":
        k = "seq_cached" if cached else "seq_eager"
        return unit[f"count_{k}"] + unit["count_decode"], unit[k], unit[f"{k}_rows"]
    p = "pass_cached" if cached else "pass"
    pre = (unit["count_prefix"], unit["prefix"], unit["prefix_rows"]) if cached else (0, 0, 0)
    return (pre[0] + passes * unit[f"count_{p}"] + dec * unit["count_decode"],
            pre[1] + passes * unit[p], pre[2] + passes * unit[f"{p}_rows"])


def flop_units(net, tok, item):
    """Per-call FLOP counts for one item (shapes are fixed: the prefix is padded to P, levels are fixed)."""
    vq = tok.vq
    S = sum(int(l) for l in vq.levels)
    codes = torch.zeros(1, S, dtype=torch.long, device=item.x.device)
    z, pm = CD.prefix_inputs(net, item.x)
    u = generator_block_flops(net)
    cache = CD.PrefixCache(net, z, pm)
    u.update({"count_seq_eager": counted_flops(lambda: CD.seq_greedy(net, item.x, False)),
              "count_seq_cached": counted_flops(lambda: CD.seq_greedy(net, item.x, True)),
              "count_pass": counted_flops(lambda: CD.make_pass(net, item.x, False)(codes)),
              "count_prefix": counted_flops(lambda: CD.PrefixCache(net, z, pm)),
              "count_pass_cached": counted_flops(lambda: CD.cached_parallel_logits(net, cache, codes)),
              "count_decode": counted_flops(lambda: tok.decoder(vq.idx_to_z_q(codes)).argmax(-1))})
    return u


# ------------------------------------------------------------------------------------------------ integrity
def mismatches(a, b):
    return [i for i, (u, v) in enumerate(zip(a, b)) if u != v]


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rule_on_saved(h, split, n):
    """R042's stable x1 on an arm's saved capped hypotheses: [(passes, text)] for the first n items."""
    nfe, cap = h[f"nfe_{split}"], h[f"hyp_{split}_capped"]
    return [CD.stable_stop([cap[str(t)][i] for t in range(1, nfe[i] + 1)], nfe[i], 1) for i in range(n)]


def rule_reproduction(val_text=None):
    """Gate: the timed rule, applied to the four arms' saved val outputs, reproduces results/j2f_stop_rule_val_s0.json
    (passes exactly; BLEU too if the newstest2013 references are given)."""
    rec = json.load(open(os.path.join(ROOT, "results", "j2f_stop_rule_val_s0.json"), encoding="utf-8"))["arms"]
    files = {"J0": "w2c_L_wmt_ctrl/L_wmt_released", "J1": "w2c_L_wmt_ctrl/L_wmt_ctrl_s0",
             "J2": "w2c_L_wmt_consist/L_wmt_consist_s0", "J2-F": "w2c_L_wmt_consist_frozen/L_wmt_consist_frozen_s0"}
    refs = json.load(open(val_text, encoding="utf-8"))["ref"] if val_text and os.path.exists(val_text) else None
    out = {}
    for arm, f in files.items():
        h = json.load(open(os.path.join(ROOT, "results", "colab", f + "__hyp_val.json"), encoding="utf-8"))
        res = rule_on_saved(h, "val", len(h["nfe_val"]))
        p = float(np.mean([t for t, _ in res]))
        o = {"passes": p, "passes_recorded": rec[arm]["stable x1"]["passes"],
             "passes_ok": abs(p - rec[arm]["stable x1"]["passes"]) < 1e-9}
        if refs is not None:
            import sacrebleu
            b = sacrebleu.corpus_bleu([t for _, t in res], [list(refs)], lowercase=True).score
            o.update(bleu=b, bleu_recorded=rec[arm]["stable x1"]["bleu"], bleu_ok=abs(b - rec[arm]["stable x1"]["bleu"]) < 1e-9)
        out[arm] = o
    out["ok"] = all(v["passes_ok"] and v.get("bleu_ok", True) for v in out.values() if isinstance(v, dict))
    out["bleu_checked"] = refs is not None
    return out


def integrity_wmt(res, expect_j0, expect_j2f, n):
    g = {}
    seq_e, seq_c = res["J0 seq eager"]["out"], res["J0 seq cached"]["out"]
    g["seq_cached_vs_eager_text_mismatch"] = mismatches(seq_c, seq_e)
    j0 = expect_j0
    for impl in ("eager", "cached"):
        r = res[f"J0 jacobi exact {impl}"]
        g[f"J0_exact_{impl}_vs_saved_text"] = mismatches(r["out"], j0["hyp_test"][:n])
        g[f"J0_exact_{impl}_vs_saved_passes"] = mismatches(r["passes"], j0["nfe_test"][:n])
        g[f"J0_exact_{impl}_vs_seq_text"] = mismatches(r["out"], seq_e)
        want = rule_on_saved(j0, "test", n)
        r = res[f"J0 jacobi stable1 {impl}"]
        g[f"J0_stable1_{impl}_vs_saved_rule"] = mismatches(list(zip(r["passes"], r["out"])), want)
        if expect_j2f is not None and f"J2-F jacobi stable1 {impl}" in res:
            r = res[f"J2-F jacobi stable1 {impl}"]
            g[f"J2F_stable1_{impl}_vs_saved_rule"] = mismatches(list(zip(r["passes"], r["out"])), rule_on_saved(expect_j2f, "test", n))
    return g


def integrity_sudoku(res):
    g = {"seq_cached_vs_eager_board_mismatch": mismatches(res["seq cached"]["out"], res["seq eager"]["out"])}
    for impl in ("eager", "cached"):
        g[f"exact_{impl}_vs_seq_board"] = mismatches(res[f"jacobi exact {impl}"]["out"], res["seq eager"]["out"])
    for rule in ("exact", "verifier"):
        a, b = res[f"jacobi {rule} eager"], res[f"jacobi {rule} cached"]
        g[f"{rule}_cached_vs_eager"] = mismatches(list(zip(a["passes"], a["out"])), list(zip(b["passes"], b["out"])))
    return g


def gate_ok(g):
    return all(len(v) <= MAX_MISMATCH for k, v in g.items() if isinstance(v, list))


# ------------------------------------------------------------------------------------------------ main
def load_items(task, tok_or_pt, net, n, device):
    if task == "wmt14":
        from ncmr.ncm_ref_loader import safe_pickle_load
        srcs = safe_pickle_load(os.path.join(ROOT, "data", "wmt14", "sources.pickle"))[:n]
        P, PAD = int(net.prefix_len), tok_or_pt.pad_token_id
        ids = [(tok_or_pt.encode(t)[:P] + [PAD] * P)[:P] for t in srcs]          # as train_wmt14_L.evaluate
        return [Item(torch.tensor([r], device=device)) for r in ids]
    from ncmr.ncm_ref_loader import safe_pickle_load
    from ncmr.sudoku import cells
    pz = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", "hard", "puzzles.pickle"))[:n]
    sol = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", "hard", "solutions.pickle"))[:n]
    P = int(net.prefix_len)
    return [Item(torch.tensor([list(p)[:P]], device=device), {"clues": cells(list(p)), "solution": cells(list(s))})
            for p, s in zip(pz, sol)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["wmt14", "sudoku"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--precisions", default="fp32,bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--j2f_ckpt", default="", help="J2-F's final.pt (w2c_L_wmt_consist_frozen, seed 0)")
    ap.add_argument("--expect_j0", default=os.path.join(ROOT, "results/colab/w2c_L_wmt_ctrl/L_wmt_released__hyp_test.json"))
    ap.add_argument("--expect_j2f", default=os.path.join(ROOT, "results/colab/w2c_L_wmt_consist_frozen/L_wmt_consist_frozen_s0__hyp_test.json"))
    ap.add_argument("--val_text", default="", help="newstest2013 val text json (prep_wmt14.py) for the rule's BLEU check")
    ap.add_argument("--allow_cpu", action="store_true", help="smoke runs only: the gate is read on an A100")
    a = ap.parse_args()
    from ncmr.ncm_ref_loader import load_ncm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not a.allow_cpu:
        sys.exit("D1 times a GPU; pass --allow_cpu for a smoke run")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    os.makedirs(a.run_dir, exist_ok=True)
    B = load_ncm(os.path.join(ROOT, "artifacts", "wmt14" if a.task == "wmt14" else "sudoku/hard"), device=device)
    net, tok, pt = B.net.eval(), B.vqvae.eval(), B.pre_tokenizer
    for p in list(net.parameters()) + list(tok.parameters()):
        p.requires_grad_(False)
    models, ident = {"J0" if a.task == "wmt14" else "S": net}, {}
    detok = None
    if a.task == "wmt14":
        EOS, PAD = pt.eos_token_id, pt.pad_token_id
        assert PAD == net.prefix_pad_id, "the prefix pad mask must be the evaluation's"

        def detok(ids):                                   # train_wmt14_L.detok
            ids = list(ids)
            if EOS in ids:
                ids = ids[:ids.index(EOS)]
            return pt.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)

        if a.j2f_ckpt:
            if not os.path.exists(a.j2f_ckpt):
                sys.exit(f"J2-F checkpoint not found: {a.j2f_ckpt}")
            import copy
            j2f = copy.deepcopy(net)
            j2f.load_state_dict(torch.load(a.j2f_ckpt, map_location=device, weights_only=True)["model"])
            models["J2-F"] = j2f.eval()
            ident["J2-F"] = {"file": os.path.basename(os.path.dirname(a.j2f_ckpt)) + "/final.pt", "sha256": digest(a.j2f_ckpt)}
    items = load_items(a.task, pt if a.task == "wmt14" else tok, net, a.n, device)
    K = len(tok.vq.levels)
    configs = make_configs(a.task, models, tok, detok, K + 1)
    res = {"task": a.task, "n": len(items), "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
           "torch": torch.__version__, "models": ident,
           "protocol": {"batch": 1, "warmup_items": a.warmup, "repeats": a.repeats, "item_latency": "median of repeats",
                        "order": "configurations interleaved per item, seeded random order per repeat", "seed": a.seed,
                        "timed_region": "token prefix on device -> output text (WMT14) / board (Sudoku), CUDA-synchronised",
                        "max_passes": K + 1, "precisions": a.precisions, "tf32": False, "gate": GATE,
                        "max_mismatch": MAX_MISMATCH, "first_items": f"first {len(items)} of the released test set"}}
    if a.task == "wmt14":
        res["rule_reproduction_val"] = rule_reproduction(a.val_text)
        print(json.dumps({"rule_reproduction_val": res["rule_reproduction_val"]}), flush=True)
    res["flops_units"] = {m: flop_units(net_, tok, items[0]) for m, net_ in models.items()}
    for prec in a.precisions.split(","):
        r = run_timing(configs, items, device, prec, a.repeats, a.warmup, a.seed)
        out = {"configs": {}}
        for name, side, _ in configs:
            d = r[name]
            unit = res["flops_units"][name.split()[0] if a.task == "wmt14" else "S"]
            fl = [flops_per_item(name, side, p, unit) for p in d["passes"]]
            out["configs"][name] = {**summarise(d["lat"], d["passes"]), "side": side,
                                    "stops": {s: d["stop"].count(s) for s in sorted(set(d["stop"]))},
                                    "executed_flops_per_item": float(np.mean([f[0] for f in fl])),
                                    "generator_block_flops_per_item": float(np.mean([f[1] for f in fl])),
                                    "generator_rows_per_item": float(np.mean([f[2] for f in fl])),
                                    "nondeterministic_items": d["nondeterministic"], "latency_s": d["lat"],
                                    "passes": d["passes"]}
        if a.task == "wmt14":
            g = integrity_wmt(r, json.load(open(a.expect_j0, encoding="utf-8")),
                              json.load(open(a.expect_j2f, encoding="utf-8")) if "J2-F" in models else None, len(items))
            g["rule_reproduction_val_ok"] = bool(res["rule_reproduction_val"]["ok"])
        else:
            g = integrity_sudoku(r)
            for name in r:
                out["configs"][name]["solved_pct"] = 100 * float(np.mean([o == it.extra["solution"] for o, it in zip(r[name]["out"], items)]))
        g["nondeterministic"] = sorted({i for n in r for i in r[n]["nondeterministic"]})
        out["integrity"] = {k: v for k, v in g.items()}
        out["integrity_ok"] = gate_ok(g) and g.get("rule_reproduction_val_ok", True)
        lat = {n: np.asarray(out["configs"][n]["latency_s"]) for n in out["configs"]}
        best = lambda prefix: min((n for n in lat if n.startswith(prefix)), key=lambda n: lat[n].mean())
        if a.task == "wmt14":
            seq = best("J0 seq ")
            out["ratios_vs_fastest_seq"] = {"sequential": seq, **{k: ratio_ci(lat[best(k + " ")], lat[seq]) | {"config": best(k + " ")}
                                                                   for k in ("J0 jacobi exact", "J0 jacobi stable1", "J2-F jacobi stable1") if any(n.startswith(k) for n in lat)}}
            cand = "J2-F jacobi stable1"
        else:
            seq = best("seq ")
            out["ratios_vs_fastest_seq"] = {"sequential": seq, **{k: ratio_ci(lat[best(k + " ")], lat[seq]) | {"config": best(k + " ")}
                                                                   for k in ("jacobi exact", "jacobi verifier")}}
            cand = "jacobi verifier"
        if cand in out["ratios_vs_fastest_seq"]:
            rr = out["ratios_vs_fastest_seq"][cand]["ratio"]
            out["gate"] = {"candidate": cand, "ratio": rr, "threshold": GATE, "precision": prec,
                           "verdict": ("not read (integrity)" if not out["integrity_ok"] else
                                       "pass" if rr <= GATE else "fail") + ("" if prec == "fp32" else " (secondary precision; the gate reads fp32)")}
        res[prec] = out
        print(json.dumps({prec: {k: v for k, v in out.items() if k not in ("configs",)}}, indent=1), flush=True)
        print(json.dumps({prec: {n: {k: c[k] for k in ("mean_ms", "se_ms", "passes_mean")} for n, c in out["configs"].items()}}, indent=1), flush=True)
    with open(os.path.join(a.run_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", os.path.join(os.path.basename(a.run_dir.rstrip("/")), "result.json"))


if __name__ == "__main__":
    main()
