"""WMT14 pair registered analysis (J1 + J1a + J1b + J1c, seed 0): recompute the screen, kill and sanity quantities
from the saved results, plus labelled exploratory diagnostics of where J2's pass saving comes from.

  python scripts/analyze_j1.py [--root results/colab] [--out results/j1_analysis_s0.json]
                               [--val_refs FILE --test_refs FILE]

Reads each arm's `*__result.json` and its sentence-ordered `*__hyp_{val,test}.json` (exact and capped hypotheses,
per-sentence passes). Thresholds are the registered ones (`docs/preregistration.md`, J1 / J1a / J1b / J1c); nothing
is fitted. The J1a paired bootstrap needs the reference translations (one per line, sentence-ordered; not in the
repository): without `--val_refs` / `--test_refs` it is reported as not computed.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os

import numpy as np

ARMS = {   # arm -> (run directory, result prefix, steps)
    "J0": ("w2c_L_wmt_ctrl", "L_wmt_released", 0),
    "J1": ("w2c_L_wmt_ctrl", "L_wmt_ctrl_s0", 3000),
    "J2": ("w2c_L_wmt_consist", "L_wmt_consist_s0", 2000),
}
FROZEN_ARM = ("w2c_L_wmt_consist_frozen", "L_wmt_consist_frozen_s0", 2000)
SEQ = 11                          # sequential NFE (11 levels)
H12_PASSES, BLEU_LOSS = 8.25, 1.0   # J1 H12: <= 8.25 mean exact passes on test, BLEU >= J0 - 1
SCREEN_PASSES = 9.0               # J1 screen (val): passes <= 9.0 and BLEU >= J0 val - 1
J1_MARGIN = 1.0                   # J1 kill: J2's passes must be below J1's by >= 1
LEN_TOL = 0.02                    # J1a: J2's length ratio within 0.02 of J0's
DEGEN_LIMIT = 1.0                 # J1b: > 1% degenerate trajectory sources blocks reading H12
J0_REF = (11.07, 22.63)           # R020 / w0: J0 test passes and BLEU (pass-1 detection off, J1c)
SHARED_MIN = 5                    # exploratory: a hypothesis string emitted for >= this many sentences


def load_arm(root, arm):
    d, prefix, steps = FROZEN_ARM if arm == "J2-F" else ARMS[arm]
    r = json.load(open(os.path.join(root, d, f"{prefix}__result.json"), encoding="utf-8"))
    assert r["steps"] == steps, f"{arm}: {r['steps']} steps, registered {steps}"
    out = {"run": r["run"], "steps": r["steps"], "gpu": r.get("gpu"), "torch": r.get("torch"),
           "generator_flops_proxy_total": r["generator_flops_proxy_total"], "val_nll": r["final_eval"]["val_nll"],
           "trajectory_sources": r.get("trajectory_sources", 0), "degenerate_sources": r.get("degenerate_sources", 0),
           "degenerate_pct": r.get("degenerate_pct", 0.0)}
    for split in ("val", "test"):
        e = r["final_eval"][split]
        h = json.load(open(os.path.join(root, d, f"{prefix}__hyp_{split}.json"), encoding="utf-8"))
        hyp, nfe = h[f"hyp_{split}"], np.array(h[f"nfe_{split}"], dtype=int)
        assert len(hyp) == len(nfe) == e["n"], f"{arm} {split}: record lengths differ from n"
        assert abs(nfe.mean() - e["exact_passes_mean"]) < 1e-9, f"{arm} {split}: passes are not the record mean"
        out[split] = {"n": e["n"], "passes": float(nfe.mean()), "bleu": e["bleu_lc"], "len_ratio": e["len_ratio"],
                      "hist": np.bincount(nfe, minlength=13).tolist(),
                      "capped": {int(c): {"bleu": v["bleu_lc"], "passes": v["passes_mean"],
                                          "len_ratio": v["len_ratio"], "identical_pct": v["identical_to_exact_pct"]}
                                 for c, v in e["capped"].items()},
                      "exact_eq_sequential_pct": e["fixed_point_equals_sequential_pct"],
                      "exact_ne_sequential_idx": e.get("fixed_point_differs_from_sequential_idx", []),
                      "_hyp": hyp, "_cap": h[f"hyp_{split}_capped"], "_nfe": nfe}
    return out


def shared_outputs(hyp, k=SHARED_MIN):
    """Sentences whose hypothesis string is emitted for >= k sentences (input-independent outputs)."""
    c = collections.Counter(hyp)
    mask = np.array([c[s] >= k for s in hyp])
    return mask, [(s, n) for s, n in c.most_common(5) if n >= k]


def dup_per_1k(hyp):
    words = sum(len(s.split()) for s in hyp)
    dups = sum(sum(a == b for a, b in zip(s.split(), s.split()[1:])) for s in hyp)
    return 1000.0 * dups / max(1, words)


def paired_bootstrap(hyp_a, hyp_b, refs, n=1000, seed=0):
    """BLEU(a) - BLEU(b) with a 95% paired bootstrap interval over sentences (lowercased sacreBLEU, as evaluated)."""
    import sacrebleu
    rng = np.random.default_rng(seed)
    m = len(refs)
    bleu = lambda h, r: sacrebleu.corpus_bleu(h, [r], lowercase=True).score
    diffs = []
    for _ in range(n):
        ix = rng.integers(0, m, m)
        r = [refs[i] for i in ix]
        diffs.append(bleu([hyp_a[i] for i in ix], r) - bleu([hyp_b[i] for i in ix], r))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": bleu(hyp_a, refs) - bleu(hyp_b, refs), "ci95": [float(lo), float(hi)], "n_resamples": n}


def analyse(root, refs=None, candidate="J2"):
    if candidate not in ("J2", "J2-F"):
        raise ValueError(f"unknown candidate {candidate!r}")
    A = {a: load_arm(root, a) for a in ARMS}
    if candidate == "J2-F":
        A[candidate] = load_arm(root, candidate)
    J0, J1, J2 = A["J0"], A["J1"], A[candidate]
    res = {"arms": {a: {k: v for k, v in x.items() if k not in ("val", "test")} |
                    {s: {k: v for k, v in x[s].items() if not k.startswith("_")} for s in ("val", "test")}
                    for a, x in A.items()}}
    if candidate != "J2":
        res["candidate"] = candidate

    # registered: J1 sanity checks
    res["sanity"] = {
        "J0_test_vs_R020": {"passes": J0["test"]["passes"], "bleu": J0["test"]["bleu"], "ref": J0_REF,
                            "pass1_share_pct": 100.0 * J0["test"]["hist"][1] / J0["test"]["n"]},
        "J1_bleu_loss_vs_J0": {s: J0[s]["bleu"] - J1[s]["bleu"] for s in ("val", "test")},
        "J1_recipe_ok": all(J0[s]["bleu"] - J1[s]["bleu"] <= BLEU_LOSS for s in ("val", "test")),
        f"degenerate_pct_{candidate}": J2["degenerate_pct"], "degenerate_ok": J2["degenerate_pct"] <= DEGEN_LIMIT,
        "flop_proxy_equal": J1["generator_flops_proxy_total"] == J2["generator_flops_proxy_total"],
        "exact_ne_sequential": {a: {s: A[a][s]["exact_ne_sequential_idx"] for s in ("val", "test")} for a in A},
    }
    # registered: J1 screen (seed 0, val)
    v = J2["val"]
    res["screen"] = {"val_passes": v["passes"], "passes_ok": v["passes"] <= SCREEN_PASSES,
                     "val_bleu": v["bleu"], "bleu_floor": J0["val"]["bleu"] - BLEU_LOSS,
                     "bleu_ok": v["bleu"] >= J0["val"]["bleu"] - BLEU_LOSS}
    res["screen"]["advance"] = res["screen"]["passes_ok"] and res["screen"]["bleu_ok"]
    # registered: H12 on test (seed 0 reading; the claim needs 3 seeds) and its kill conditions
    t = J2["test"]
    saving = 100.0 * (1 - t["passes"] / SEQ)
    res["h12_test"] = {"passes": t["passes"], "passes_ok": t["passes"] <= H12_PASSES, "nfe_saving_pct": saving,
                       "bleu": t["bleu"], "bleu_loss_vs_J0": J0["test"]["bleu"] - t["bleu"],
                       "bleu_ok": t["bleu"] >= J0["test"]["bleu"] - BLEU_LOSS,
                       "passes_below_J1": J1["test"]["passes"] - t["passes"],
                       "below_J1_ok": J1["test"]["passes"] - t["passes"] >= J1_MARGIN}
    res["h12_test"]["kill"] = not (saving >= 25.0 and res["h12_test"]["bleu_ok"]) or not res["h12_test"]["below_J1_ok"]
    # registered: J1a comparators at the cap ceil(J2's mean exact passes)
    cap = math.ceil(t["passes"])
    res["j1a"] = {"cap": cap, f"{candidate}_bleu": t["bleu"],
                  "J0_at_cap": J0["test"]["capped"][cap], "J1_at_cap": J1["test"]["capped"][cap],
                  f"{candidate}_minus_J0_at_cap": t["bleu"] - J0["test"]["capped"][cap]["bleu"],
                  f"{candidate}_minus_J1_at_cap": t["bleu"] - J1["test"]["capped"][cap]["bleu"],
                  "len_ratio_diff_vs_J0": t["len_ratio"] - J0["test"]["len_ratio"],
                  "len_ok": abs(t["len_ratio"] - J0["test"]["len_ratio"]) <= LEN_TOL,
                  "J0_cap8_bleu_loss": J0["test"]["bleu"] - J0["test"]["capped"][8]["bleu"],
                  "J0_cap8_within_1": J0["test"]["bleu"] - J0["test"]["capped"][8]["bleu"] <= BLEU_LOSS}
    if refs and refs.get("test"):
        hc = {a: A[a]["test"]["_cap"][str(cap)] for a in ("J0", "J1")}
        res["j1a"]["bootstrap"] = {f"{candidate}_vs_{a}": paired_bootstrap(t["_hyp"], hc[a], refs["test"]) for a in hc}
    else:
        res["j1a"]["bootstrap"] = "not computed: reference translations not supplied"

    # exploratory: input-independent outputs, length, duplication, passes on the remaining sentences
    ex = {}
    for s in ("val", "test"):
        mask, top = shared_outputs(A["J2"][s]["_hyp"])
        words = {a: np.array([len(h.split()) for h in A[a][s]["_hyp"]]) for a in A}
        ex[s] = {"shared_output_pct": {a: 100.0 * shared_outputs(A[a][s]["_hyp"])[0].mean() for a in A},
                 "J2_top_shared": top,
                 "J2_passes_on_shared": float(A["J2"][s]["_nfe"][mask].mean()) if mask.any() else None,
                 "passes_on_rest": {a: float(A[a][s]["_nfe"][~mask].mean()) for a in A},
                 "words_mean": {a: float(w.mean()) for a, w in words.items()},
                 "words_ratio_vs_J0_on_rest": {a: float(words[a][~mask].sum() / words["J0"][~mask].sum()) for a in A},
                 "dup_adjacent_per_1k_words": {a: dup_per_1k(A[a][s]["_hyp"]) for a in A},
                 "identical_to_J0_pct": {a: 100.0 * np.mean([x == y for x, y in zip(A[a][s]["_hyp"], J0[s]["_hyp"])])
                                         for a in (("J1", "J2", "J2-F") if candidate == "J2-F" else ("J1", "J2"))}}
        if candidate == "J2-F":
            frozen = A[candidate][s]
            frozen_mask, frozen_top = shared_outputs(frozen["_hyp"])
            ex[s][f"{candidate}_top_shared"] = frozen_top
            ex[s][f"{candidate}_passes_on_shared"] = (
                float(frozen["_nfe"][frozen_mask].mean()) if frozen_mask.any() else None)
            ex[s]["passes_on_rest"][candidate] = (
                float(frozen["_nfe"][~frozen_mask].mean()) if (~frozen_mask).any() else None)
            denominator = words["J0"][~frozen_mask].sum()
            ex[s]["words_ratio_vs_J0_on_rest"][candidate] = (
                float(words[candidate][~frozen_mask].sum() / denominator) if denominator else None)
        ex[s]["saving_on_rest_pct"] = {a: 100.0 * (1 - p / SEQ) if p is not None else None for a, p in ex[s]["passes_on_rest"].items()}
    # exploratory: does the 500-sentence curve subset of J2's last curve point equal the final eval's first 500?
    d, prefix, steps = ARMS["J2"]
    cv = json.load(open(os.path.join(root, d, f"{prefix}__hyp_val_step{steps}.json"), encoding="utf-8"))
    n = len(cv["hyp_val"])
    ex["curve_subset_matches_final"] = (cv["hyp_val"] == A["J2"]["val"]["_hyp"][:n]
                                        and list(cv["nfe_val"]) == A["J2"]["val"]["_nfe"][:n].tolist())
    res["exploratory"] = ex
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/colab")
    ap.add_argument("--out", default="results/j1_analysis_s0.json")
    ap.add_argument("--candidate", choices=("J2", "J2-F"), default="J2")
    ap.add_argument("--val_refs")
    ap.add_argument("--test_refs")
    a = ap.parse_args()
    rd = lambda f: [l.rstrip("\n") for l in open(f, encoding="utf-8")] if f else None
    res = analyse(a.root, {"val": rd(a.val_refs), "test": rd(a.test_refs)}, candidate=a.candidate)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    sc, h, j = res["screen"], res["h12_test"], res["j1a"]
    print(f"screen (val): passes {sc['val_passes']:.3f} (<= {SCREEN_PASSES}: {sc['passes_ok']}), "
          f"BLEU {sc['val_bleu']:.2f} (>= {sc['bleu_floor']:.2f}: {sc['bleu_ok']}) -> advance {sc['advance']}")
    print(f"H12 (test): passes {h['passes']:.3f}, saving {h['nfe_saving_pct']:.1f}%, BLEU loss {h['bleu_loss_vs_J0']:.2f}, "
          f"below J1 by {h['passes_below_J1']:.2f} -> kill {h['kill']}")
    print(f"J1a cap {j['cap']}: {a.candidate} - J0 {j[f'{a.candidate}_minus_J0_at_cap']:.2f}, {a.candidate} - J1 {j[f'{a.candidate}_minus_J1_at_cap']:.2f}, "
          f"len diff {j['len_ratio_diff_vs_J0']:.3f}")
    ex = res["exploratory"]["test"]
    print(f"exploratory (test): shared outputs {ex['shared_output_pct']}, passes on rest {ex['passes_on_rest']}")
    print(f"wrote {os.path.basename(a.out)}")


if __name__ == "__main__":
    main()
