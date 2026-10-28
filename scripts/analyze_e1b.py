"""E1b readings (docs/preregistration.md, "E1b - continued E1 sampling on censored and wide items"): prediction 1 re-read
for the three released models (TinyStories A0 on all 64 and the 61 untouched windows, WMT14 J0, Sudoku S0) after
substituting each continued item's E1b final counts at cut 8 for its E1 counts. E1's rule (lower bound of mean R_8 / mean
N_8 >= 0.8) and E1a's added rule are applied unchanged, with E1's delta (ALPHA / (items x registered cuts of the saved
file), `analyze_e1.delta_for`), via analyze_e1's functions. The saved E1 files are read, never edited: the substitution
works on deep copies. Items not reached keep E1's counts (listed). Prediction 2 is not re-read.

  python scripts/analyze_e1b.py --e1b runs/e1b/result.json [--e1_dir results/colab/e1_equivalence]
                                [--out results/e1b_analysis.json]

Do not run analyze_e1's saved-summary integrity check on substituted data (it compares the saved summaries with a
recomputation and would flag the substituted rows); it is run on the originals by analyze_e1 itself.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AE = _load("analyze_e1")
B = _load("e1b_continue")
E1_DIR = os.path.join(ROOT, "results", "colab", "e1_equivalence")
CUT = B.CUT


# ------------------------------------------------------------------------------------------------ readings
def readings(domain, data, rows=None):
    """prediction 1 of the released model of `domain` at cut 8 from `data` (a saved E1 file's dict), with `rows`
    (default the file's) as the released model's rows. delta is the file's, whatever the substituted counts."""
    model = AE.RELEASED[domain]
    rows = data["rows"][model] if rows is None else rows
    delta = B.e1_delta(data)
    K = len(rows[0]["level_nll"])
    iv = AE.build_intervals(rows, [CUT], delta)
    summ = {model: AE.model_summaries(domain, rows, iv, [CUT], K)}
    return AE.prediction1(domain, model, summ), delta


def substitute(rows, items, model):
    """A copy of `rows` with the continued items' final E1b counts at cut 8. `items`: the e1b result's records."""
    out = copy.deepcopy(rows)
    for r in items:
        if r["model"] != model:
            continue
        cur = out[r["item"]]["mc"][str(CUT)]
        if list(cur) != list(r["e1"]):
            raise ValueError(f"{model} item {r['item']}: e1b's E1 counts {r['e1']} differ from the saved E1 file's {cur}")
        out[r["item"]]["mc"][str(CUT)] = [int(r["final"][0]), int(r["final"][1])]
    return out


def e1_rule_holds(p1):
    """E1's rule holds on every item set of the reading (A0: all 64 and the 61 untouched)."""
    return bool(p1) and all(v["e1_rule"]["verdict"] == "holds" for v in p1["sets"].values())


def check_selection(e1b, files, digests=None):
    """The e1b run's saved selection must equal the one recomputed from the saved E1 files, and each source file must be
    the one the run pinned (`e1_sha256`, Codex PR 16 round 3): equal counts and indices do not pin the NLLs."""
    for model, dom, fn in B.DOMAINS:
        if model not in e1b["selection"]:
            continue
        pinned = e1b["selection"][model].get("e1_sha256")
        if digests is not None and pinned != digests[dom]:
            raise ValueError(f"{model}: the saved E1 file {fn} is not the one the e1b run pinned (sha256 {pinned!r} vs "
                             f"{digests[dom]!r}): refusing to combine them")
        data = files[dom]
        cens, wide = B.select_items(data["rows"][model], B.e1_delta(data))
        got = e1b["selection"][model]
        if got["censored"] != cens or got["wide"] != wide:
            raise ValueError(f"{model}: the e1b run's selection differs from the recomputation from the saved E1 file")


def analyse(e1b, e1_dir=E1_DIR):
    files, digests = {}, {}
    for dom, fn in AE.FILES.items():
        with open(os.path.join(e1_dir, fn), "rb") as f:
            raw = f.read()
        files[dom], digests[dom] = json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()
    check_selection(e1b, files, digests)
    items = e1b["items"]
    out = {"protocol": {"alpha": AE.ALPHA, "registered_cut_min": AE.REG_CUT_MIN, "untouched_from_item": AE.UNTOUCHED_FROM,
                        "delta": "analyze_e1.delta_for per model, unchanged from E1a",
                        "substitution": "an item's E1b final counts at cut 8 replace its E1 counts; every other item, "
                                        "cut and model keeps E1's",
                        "bar": "E1's (c) replication bar is met iff E1's rule holds in all three released models "
                               "(A0 on all 64 and on the 61 untouched windows)",
                        "e1b": e1b["protocol"]},
           "models": {}}
    for dom, data in files.items():
        model = AE.RELEASED[dom]
        e1_p1, delta = readings(dom, data)
        mine = [r for r in items if r["model"] == model]
        rows_b = substitute(data["rows"][model], mine, model)
        b_p1, _ = readings(dom, data, rows_b)
        cont = {"continued": len(mine), "selected": (len(e1b["selection"][model]["censored"]) +
                                                      len(e1b["selection"][model]["wide"])) if model in e1b["selection"] else 0,
                "stop_width": sum(r["stop"] == "width" for r in mine), "stop_cap": sum(r["stop"] == "cap" for r in mine),
                "replay_matched": sum(bool(r["replay_match"]) for r in mine),
                "replay_mismatched": [r["item"] for r in mine if not r["replay_match"]],
                "samples_total": sum(r["final"][1] for r in mine),
                "not_reached": [i for m, i in e1b.get("not_reached", []) if m == model],
                "per_item": [{"item": r["item"], "kind": r.get("kind"), "e1": r["e1"], "replay_hits": r["replay_hits"],
                              "replay_match": bool(r["replay_match"]), "final": r["final"], "stop": r["stop"]}
                             for r in mine]}
        out["models"][model] = {"domain": dom, "delta": delta, "e1": e1_p1, "e1b": b_p1, "continuation": cont,
                                "e1_rule_holds_e1": e1_rule_holds(e1_p1), "e1_rule_holds_e1b": e1_rule_holds(b_p1)}
    holds = [m["e1_rule_holds_e1b"] for m in out["models"].values()]
    not_reached = {m: v["continuation"]["not_reached"] for m, v in out["models"].items() if v["continuation"]["not_reached"]}
    out["verdict"] = {"e1_replication_bar_met": len(holds) == len(AE.FILES) and all(holds),
                      "e1_rule_by_model": {m: v["e1_rule_holds_e1b"] for m, v in out["models"].items()},
                      "e1a_rule_by_model": {m: {s: x["e1a_rule"]["verdict"] for s, x in v["e1b"]["sets"].items()}
                                            for m, v in out["models"].items()},
                      "run_complete": bool(e1b.get("complete", not not_reached)), "items_not_reached": not_reached,
                      "note": "items not reached keep E1's counts" if not_reached else "every selected item was continued"}
    return out


def print_table(res):
    print("E1b cut-8 readings (E1 -> E1b)")
    for m, v in res["models"].items():
        for s, x in v["e1b"]["sets"].items():
            e = v["e1"]["sets"][s]
            print(f"  P1 {m} [{s:9s}] cens {e['censored']}->{x['censored']} "
                  f"Rshare [{_f(e['R_share_lo'])},{_f(e['R_share_hi'])}] -> [{_f(x['R_share_lo'])},{_f(x['R_share_hi'])}] "
                  f"Twidth<=2 {e['T_width_le2_share']:.2f}->{x['T_width_le2_share']:.2f}  "
                  f"E1 {e['e1_rule']['verdict']}->{x['e1_rule']['verdict']}  "
                  f"E1a {e['e1a_rule']['verdict']}->{x['e1a_rule']['verdict']}")
        c = v["continuation"]
        print(f"     continued {c['continued']}/{c['selected']} (width {c['stop_width']}, cap {c['stop_cap']}), "
              f"replay mismatches {c['replay_mismatched']}, not reached {c['not_reached']}")
    print("  verdict:", res["verdict"])


def _f(x):
    return "None" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1b", required=True, help="result.json of scripts/e1b_continue.py")
    ap.add_argument("--e1_dir", default=E1_DIR)
    ap.add_argument("--out", default=os.path.join("results", "e1b_analysis.json"))
    a = ap.parse_args()
    with open(a.e1b, encoding="utf-8") as f:
        e1b = json.load(f)
    res = AE._clean(analyse(e1b, a.e1_dir))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(AE._dump(res) + "\n")
    print_table(res)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
