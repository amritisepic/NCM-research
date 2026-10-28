"""W2M-G direction-1 run gate (docs/preregistration.md, "W2M-G", "L1R"): reads scripts/w2mg_eval.py's result.json.

  python scripts/w2mg_gate.py --result <result.json>

Passes if pi_2 >= 0.25 under GPT-2 large log loss AND pi_2 > 0 under TinyStories-33M, recomputed here from the stored
per-group log losses of the four cells it needs (the stored shares are not trusted). Exit 0: passes; 2: fails;
3: the result is missing, unreadable or incomplete (the run has not finished, or a needed cell is absent).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr import w2mg as G  # noqa: E402

CELLS = ("m2_m2", "m3_m3", "m2to3_c2", "m3to2_c2")


def gate_numbers(res):
    """{scorer: pi_2} from the stored per-group log losses; None if the result lacks a needed cell or is not finished."""
    if not isinstance(res, dict) or res.get("complete") is not True:
        return None
    cells = res.get("cells", {})
    try:
        out = {}
        for sc in G.SCORERS:
            ll = [cells[k][f"genppl_{sc}"]["log_loss_per_group"] for k in CELLS]
            if len({len(v) for v in ll}) != 1 or len(ll[0]) < 2 or not all(math.isfinite(x) for v in ll for x in v):
                return None
            out[sc] = G.shares(*ll)["pi"]
    except (KeyError, TypeError):
        return None
    return out if all(math.isfinite(v) for v in out.values()) else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    a = ap.parse_args(argv)
    try:
        with open(a.result) as f:
            res = json.load(f)
    except (OSError, ValueError):
        print("gate: result missing or unreadable")
        return 3
    pi = gate_numbers(res)
    if pi is None:
        print("gate: result incomplete (run not finished or a cell of cut 2 missing)")
        return 3
    ok = G.gate_ii(pi["gpt2-large"], pi["TinyStories-33M"])
    print(f"pi_2 (GPT-2 large log loss) = {pi['gpt2-large']:.4f}  (needs >= 0.25)")
    print(f"pi_2 (TinyStories-33M log loss) = {pi['TinyStories-33M']:.4f}  (needs > 0)")
    print("direction-1 gate: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
