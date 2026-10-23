"""P1 analysis: decompose the NCM-vs-AR NLL gap into within-level TC and a representation term.

Inputs: run dirs with metrics.jsonl (from train.py) for tok_s, ncm_s, ts_fact_s, ts_ar_s, ar_s.
All quantities in nats per 128-token window (sequence):
  NLL_AR        : AR-token model, val NLL per window (FLOP-matched run; also the data-matched point =
                  the eval at the same number of sequences as the NCM runs).
  NLL_NCM       : -log p_unigram(level-0 code) + sum_k NLL_fact,k  (single-stream NCM generator)
  TC_k          : NLL_ts_fact,k - NLL_ts_ar,k (two-stream pair; identical except within-level code access)
  representation term = (NLL_L0 + sum_k NLL_ts_ar,k) - NLL_AR
Usage: python scripts/analyze_p1.py --runs <dir containing the run folders> [--l0_nats X]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--l0_nats", type=float, default=None, help="entropy of the level-0 unigram (nats); "
                "default: estimated from tok_s level-0 code usage if available")
ap.add_argument("--out", default="results/p1_tc_decomposition.json")
a = ap.parse_args()


def evals(run):
    f = os.path.join(a.runs, run, "metrics.jsonl")
    if not os.path.exists(f):
        f = os.path.join(a.runs, f"{run}__metrics.jsonl")  # flattened Colab results folder
    rows = [json.loads(l) for l in open(f)]
    return [(r["step"], r["eval"]) for r in rows if "eval" in r]


ncm = evals("ncm_s")
fact = evals("ts_fact_s")
ar_codes = evals("ts_ar_s")
ar = evals("ar_s")
out = {"by_step": []}
def result(run):
    for f in (os.path.join(a.runs, run, "result.json"), os.path.join(a.runs, f"{run}__result.json")):
        if os.path.exists(f):
            return json.load(open(f))
    return {}


l0 = a.l0_nats if a.l0_nats is not None else result("ncm_s").get("final_eval", {}).get("val_nll_level0_unigram", float("nan"))
for (s1, e_f), (s2, e_a) in zip(fact, ar_codes):
    assert s1 == s2
    f_l, a_l = np.array(e_f["val_nll_per_level_nats_per_seq"]), np.array(e_a["val_nll_per_level_nats_per_seq"])
    out["by_step"].append({"step": s1, "tc_per_level": (f_l - a_l).round(2).tolist(), "tc_total": float((f_l - a_l).sum()),
                           "nll_fact_total": float(f_l.sum()), "nll_ar_within_total": float(a_l.sum())})
last = out["by_step"][-1]
ncm_last = ncm[-1][1]["val_nll_nats_per_seq"]
n_seq_ncm = ncm[-1][0] * 32
ar_final = ar[-1][1]["val_nll_nats_per_seq"]
ar_data_matched = min(ar, key=lambda r: abs(r[0] * 128 - n_seq_ncm))
out.update({
    "level0_unigram_nats": l0,
    "nll_ncm_total": l0 + ncm_last, "nll_ts_fact_total": l0 + last["nll_fact_total"],
    "nll_ts_ar_total": l0 + last["nll_ar_within_total"],
    "nll_ar_flop_matched": ar_final, "nll_ar_data_matched": ar_data_matched[1]["val_nll_nats_per_seq"],
    "ar_data_matched_step": ar_data_matched[0],
    "gap_ncm_minus_ar_flop_matched": l0 + ncm_last - ar_final,
    "tc_total": last["tc_total"], "tc_per_level": last["tc_per_level"],
    "representation_term_flop_matched": l0 + last["nll_ar_within_total"] - ar_final,
})
out["tc_share_of_gap"] = out["tc_total"] / out["gap_ncm_minus_ar_flop_matched"]
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "by_step"}, indent=1))
print("TC trend over training:", [(r["step"], round(r["tc_total"], 1)) for r in out["by_step"]])
