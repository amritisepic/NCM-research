"""E1b (docs/preregistration.md, "E1b - continued E1 sampling on censored and wide items"): continue E1's Monte Carlo of
rho_8 on the released WMT14 (J0) and TinyStories (A0) items whose E1 reading at cut 8 is censored (0 hits) or whose E1a
interval for T_8 is wider than 2 nats (after the exp(-N_8) intersection, delta = 0.05 / (items x registered cuts in the
saved file), analyze_e1.delta_for). The lists are recomputed from the saved E1 results; Sudoku (no such item) is not
continued.

Per selected item, E1's own generator (seed 0: 1_000_003 * 0 + 101 i + 8, one CUDA generator per (item, cut)):
  1. replay in rounds of 64 up to E1's sample count n_E1 (exactly `e1_equivalence.mc_hits`'s loop, without its hit stop:
     it never fires before n_E1) and record whether the replayed hit count equals E1's;
  2. then rounds of 512 on the same generator until the E1a interval is no wider than 2 nats or 2^15 samples in total
     (the last round is shortened so the total never exceeds the cap).
The item's E1b counts (from sample 1, replay included) are what is read, whether or not the replay matched.
Order: J0's items by index, then A0's. fp32, TF32 off. The time cap is cumulative over sessions and checked between
items; items not reached keep E1's counts (recorded in `not_reached`). `result.json` is rewritten after every item and
a rerun resumes, refusing a run directory whose protocol or selection differs.

  python scripts/e1b_continue.py --run_dir runs/e1b --data data/wmt14/wmt14_train_pairs.npz [--max_minutes 150]
         [--only J0,A0] [--max_items N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E1 = _load("e1_equivalence")
AE = _load("analyze_e1")
from ncmr import cached_decode as CD  # noqa: E402

CUT = 8                              # the only continued cut (registered cuts of E1 start at 8)
REPLAY_ROUND = 64                    # E1's round
ROUND = 512
CAP = 2 ** 15                        # total samples per item, replay included
MAX_WIDTH = 2.0                      # nats of T_8
SEED = 0
E1_DIR = os.path.join(ROOT, "results", "colab", "e1_equivalence")
DOMAINS = [("J0", "wmt14", "e1_wmt__result.json"), ("A0", "tinystories", "e1_ts__result.json")]   # run order
PROTOCOL = {"cut": CUT, "seed": SEED, "replay_round": REPLAY_ROUND, "round": ROUND, "cap_samples": CAP,
            "max_T_width_nats": MAX_WIDTH, "temperature": 1.0, "precision": "fp32, TF32 off",
            "generator": "torch.Generator(device).manual_seed(1_000_003 * seed + 101 * item + cut), one per item",
            "delta": "analyze_e1.delta_for(items in the model's rows, registered cuts k >= 8 in the saved E1 file)",
            "order": "J0 items by index, then A0",
            "selection": "E1 count 0 hits, or E1a T_8 interval (after the exp(-N_8) intersection) wider than 2 nats",
            "stop": "width (interval <= 2 nats, checked before each round and after the replay) | cap",
            "read": "the final E1b counts from sample 1, whether or not the replay matched E1's count"}


# ------------------------------------------------------------------------------------------------ selection
def interval_width(hits, n, N, delta):
    """Width in nats of the E1a interval for T = -log rho after the exp(-N) intersection (analyze_e1's functions)."""
    lo, hi = AE.cs_log_interval(int(hits), int(n), delta)
    lo, hi, _ = AE.intersect_floor(lo, hi, N)
    return hi - lo


def e1_delta(data):
    """E1a's delta for a saved E1 file: ALPHA / (items in the rows x registered cuts in the file)."""
    reg = [k for k in data["cuts"] if k >= AE.REG_CUT_MIN]
    return AE.delta_for(len(next(iter(data["rows"].values()))), len(reg))


def select_items(rows, delta, cut=CUT):
    """Indices of the items continued: (censored, wide). Items without samples at the cut (x^ invalid) are excluded."""
    cens, wide = [], []
    for i, r in enumerate(rows):
        h, n = r["mc"][str(cut)]
        if n == 0:
            continue
        if h == 0:
            cens.append(i)
        elif interval_width(h, n, sum(r["level_nll"][cut:]), delta) > MAX_WIDTH:
            wide.append(i)
    return cens, wide


def build_plan(e1_dir=E1_DIR, only=("J0", "A0")):
    """[{domain, model, item, e1: [hits, n], kind}] in run order, plus the per-domain saved data."""
    plan, sel = [], {}
    for model, dom, fn in DOMAINS:
        if model not in only:
            continue
        with open(os.path.join(e1_dir, fn), encoding="utf-8") as f:
            data = json.load(f)
        delta = e1_delta(data)
        rows = data["rows"][model]
        cens, wide = select_items(rows, delta)
        sel[model] = {"domain": dom, "delta": delta, "n_items": len(rows), "censored": cens, "wide": wide, "e1_file": fn}
        for i in sorted(cens + wide):
            plan.append({"domain": dom, "model": model, "item": i, "e1": list(rows[i]["mc"][str(CUT)]),
                         "kind": "censored" if i in cens else "wide"})
    return plan, sel


# ------------------------------------------------------------------------------------------------ one item
@torch.no_grad()
def continue_item(dom, prefix_ids, canon, target, seed, e1_hits, e1_n, N, delta, cut=CUT, replay_round=REPLAY_ROUND,
                  round_=ROUND, cap=CAP, max_width=MAX_WIDTH):
    """E1b for one item: replay E1's generator in rounds of `replay_round` up to e1_n, then rounds of `round_` on the
    same generator until the interval is no wider than `max_width` or `cap` samples in total. Returns
    {replay_hits, replay_n, replay_match, final: [hits, n], stop}."""
    net = dom.net
    cache = CD.PrefixCache(net, *CD.prefix_inputs(net, prefix_ids))
    g = torch.Generator(device=canon.device).manual_seed(int(seed))
    hits = n = 0

    def draw(m):
        codes = CD.sample_from_cut(net, cache, canon, cut, m, g, 1.0)
        return sum(dom.key(r) == target for r in dom.decode_ids(codes).tolist())

    while n < e1_n:                                    # mc_hits's loop
        m = min(replay_round, e1_n - n)
        hits += draw(m)
        n += m
    out = {"replay_hits": hits, "replay_n": n, "replay_match": hits == e1_hits and n == e1_n}
    while True:
        if interval_width(hits, n, N, delta) <= max_width:
            stop = "width"
            break
        if n >= cap:
            stop = "cap"
            break
        m = min(round_, cap - n)
        hits += draw(m)
        n += m
    out.update(final=[hits, n], stop=stop)
    return out


# ------------------------------------------------------------------------------------------------ resumable driver
def _write(path, res):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f)
    os.replace(tmp, path)


def run(plan, sel, runner_for, run_dir, max_minutes=150.0, max_items=None, clock=time.time, extra=None):
    """Process `plan` in order, writing `<run_dir>/result.json` after every item. Finished items (same domain, model,
    item) are skipped on a rerun; the run directory must match the saved protocol and selection. The cumulative minutes
    (all sessions) are checked between items: at or above `max_minutes` the run stops cleanly. `runner_for(domain)` ->
    callable(entry) -> dict of the item's fields. Returns the result dict."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "result.json")
    manifest = {"protocol": PROTOCOL, "selection": sel}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        for key in ("protocol", "selection"):
            if res.get(key) != json.loads(json.dumps(manifest[key])):
                raise SystemExit(f"{path}: {key} differs from this run's; refusing to resume")
    else:
        res = {"protocol": manifest["protocol"], "selection": sel, "items": [], "minutes": 0.0}
    res.update(extra or {})
    done = {(r["domain"], r["model"], r["item"]) for r in res["items"]}
    prior, t0, processed, runners = res["minutes"], clock(), 0, {}

    def cumulative():
        return prior + (clock() - t0) / 60.0

    for entry in plan:
        key = (entry["domain"], entry["model"], entry["item"])
        if key in done:
            continue
        if cumulative() >= max_minutes or (max_items is not None and processed >= max_items):
            break
        if entry["domain"] not in runners:
            runners[entry["domain"]] = runner_for(entry["domain"])
        t = clock()
        rec = {"domain": entry["domain"], "model": entry["model"], "item": entry["item"], "kind": entry["kind"],
               "e1": entry["e1"]}
        rec.update(runners[entry["domain"]](entry))
        rec["wall_s"] = clock() - t
        res["items"].append(rec)
        done.add(key)
        processed += 1
        res["minutes"] = cumulative()
        _write(path, res)
    res["minutes"] = cumulative()
    res["not_reached"] = [[e["model"], e["item"]] for e in plan if (e["domain"], e["model"], e["item"]) not in done]
    res["complete"] = not res["not_reached"]
    res["max_minutes"] = max_minutes
    _write(path, res)
    return res


# ------------------------------------------------------------------------------------------------ models
def make_runner_factory(data_path, e1_dir, device):
    """runner_for(domain): loads the released model and items once per domain (the previous domain is released)."""
    from ncmr.ncm_ref_loader import load_ncm
    state = {}

    def runner_for(domain):
        state.clear()
        if device == "cuda":
            torch.cuda.empty_cache()
        model = {d: m for m, d, _ in DOMAINS}[domain]
        with open(os.path.join(e1_dir, {m: f for m, _, f in DOMAINS}[model]), encoding="utf-8") as f:
            saved = json.load(f)
        rows = saved["rows"][model]
        B = load_ncm(os.path.join(ROOT, "artifacts", E1.ART[domain]), device=device)
        net, tok = B.net.eval(), B.vqvae.eval()
        for p in list(net.parameters()) + list(tok.parameters()):
            p.requires_grad_(False)
        extra = {}
        if domain == "tinystories":
            extra["unigram"] = B.ckpt_extra["inference"]["level0_unigram"].float().to(device)
        if domain == "wmt14":
            extra.update(eos=B.pre_tokenizer.eos_token_id, pad=B.pre_tokenizer.pad_token_id)
        items = E1.load_items(domain, B, saved["n"], device, data_path)
        dom = E1.Domain(domain, net, tok, extra)
        delta = e1_delta(saved)
        state.update(dom=dom, items=items)

        def one(entry):
            i = entry["item"]
            pre, x = items[i]
            canon = dom.canonical(x)
            target = dom.key(dom.decode_ids(canon)[0].tolist())
            nll = dom.level_nll(pre, canon)
            N = float(sum(rows[i]["level_nll"][CUT:]))       # E1's saved N_8 (what the analysis uses), not the recomputed
            out = continue_item(dom, pre, canon, target, SEED * 1_000_003 + i * 101 + CUT, entry["e1"][0], entry["e1"][1],
                                N, delta, cut=CUT, replay_round=REPLAY_ROUND, round_=ROUND, cap=CAP, max_width=MAX_WIDTH)
            out["nll_max_abs_diff_vs_e1"] = float(np.max(np.abs(np.array(nll) - np.array(rows[i]["level_nll"]))))
            return out
        return one
    return runner_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data", default="", help="WMT14: prep_wmt14.py's train_pairs.npz (val_src / val_tgt)")
    ap.add_argument("--e1_dir", default=E1_DIR, help="the saved E1 results (e1_{ts,wmt}__result.json)")
    ap.add_argument("--max_minutes", type=float, default=150.0, help="cumulative over sessions, checked between items")
    ap.add_argument("--only", default="J0,A0", help="comma list of J0, A0 (run order is always J0 then A0)")
    ap.add_argument("--max_items", type=int, default=None, help="process at most this many items in this session")
    a = ap.parse_args()
    only = tuple(a.only.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    plan, sel = build_plan(a.e1_dir, only)
    print(f"E1b: {len(plan)} items " + str({m: (len(s['censored']), len(s['wide'])) for m, s in sel.items()}), flush=True)
    res = run(plan, sel, make_runner_factory(a.data, a.e1_dir, device), a.run_dir, a.max_minutes, a.max_items,
              extra={"gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu", "torch": torch.__version__})
    mism = [(r["model"], r["item"]) for r in res["items"] if not r["replay_match"]]
    print(f"done {len(res['items'])}/{len(plan)} items, {res['minutes']:.1f} min, replay mismatches {mism}, "
          f"not reached {res['not_reached']}")


if __name__ == "__main__":
    main()
