"""W2M-GX / W2M-GXa: admit registered replay and GR references, recompute all ratios and readings.

Input: results/colab/w2mgx_eval/w2mgx_eval__result.json and W2M-GR's three saved results.
Pure cells are re-sampled; their GPT-2 large Gen-PPL must agree with GR within three new-cell SEs.
Missing, malformed, mismatched or incomplete inputs yield 'not available' without a reading.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ncmr import w2mg as G
import w2mg_eval as E
import analyze_w2mgr as GR

NB = "w2mgx_eval"
DEFAULT_ROOT = os.path.join(ROOT, "results", "colab")
DEFAULT_OUT = os.path.join(ROOT, "results", "w2mgx_analysis.json")
SHA = re.compile(r"[0-9a-f]{64}")
REGISTERED = {"experiment": "W2M-GX", "amendment": "W2M-GXa", "run_seed": 0,
              "model_seeds": [1, 2, 3], "arms": list(G.GX_ARMS), "group_n": 256,
              "replay_seeds": list(range(8)), "root_seed": 2027, "cuts": [3, 5, 6],
              "precision": "fp32, TF32 off, eval mode",
              "scorers": {"gpt2-large": "gpt2-large", "TinyStories-33M": "roneneldan/TinyStories-33M"},
              "sampler": "ancestral, T=1, Gumbel-max, one generator per group seeded with the group seed (common random numbers)",
              "ci": "paired delete-one-group jackknife, 8 groups",
              "pure_source": "kappa pure means resampled, sanity within 3 SE of W2M-GR; delta and all other baselines from W2M-GR",
              "cells": {"new_cells": 66, "pure_cells": 6, "total": 72, "groups_per_cell": 8,
                        "samples_per_group": 256, "crossed_cuts": [3, 5, 6],
                        "cut_meaning": "prefix model samples levels 1..c-1, suffix model c..11",
                        "assignment_meaning": "assignment[j-1] samples level j, j=1..11"}}


def result_path(root):
    return os.path.join(root, NB, f"{NB}__result.json")


def expected_runs():
    return {"released": "released", **{f"{a}_s{s}": f"{E.ARM_DIRS[a]}_s{s}/final.pt"
                                       for s in G.GX_SEEDS for a in ("m2", "m3")}}


def check_result(r):
    """Every protocol field, checkpoint-to-level mapping and paired score vector is checked before reading."""
    if not isinstance(r, dict):
        return ["not a JSON object"]
    bad = []
    if r.get("complete") is not True:
        bad.append("not complete")
    pr = r.get("protocol")
    if not isinstance(pr, dict):
        return bad + ["no protocol"]
    for key, value in REGISTERED.items():
        if pr.get(key) != value:
            bad.append(f"protocol {key} differs from W2M-GX / W2M-GXa")
    if pr.get("runs") != expected_runs():
        bad.append("checkpoint mapping differs from W2C final.pt at seeds 1-3")
    hashes = pr.get("ckpt_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(G.GX_ARMS) or not all(
            isinstance(h, str) and SHA.fullmatch(h) for h in (hashes or {}).values()):
        bad.append("need exactly six checkpoint SHA256 digests")
    identity = pr.get("identity")
    gpu = identity.get("gpu") if isinstance(identity, dict) else None
    if not (isinstance(gpu, str) and re.search(r"\bL4\b", gpu)):
        bad.append("GPU must be an L4")
    ref_hashes = identity.get("w2mgr_reference_sha256") if isinstance(identity, dict) else None
    if not isinstance(ref_hashes, dict) or set(ref_hashes) != {"1", "2", "3"} or not all(
            isinstance(h, str) and SHA.fullmatch(h) for h in (ref_hashes or {}).values()):
        bad.append("need canonical SHA256 identities of all three W2M-GR reference results")
    cells = r.get("cells")
    if not isinstance(cells, dict):
        return bad + ["no cells"]
    specs = {s["key"]: s for s in E.cell_specs(G.GXProtocol(), arms=G.GX_ARMS)}
    if set(cells) != set(specs):
        bad.append("need exactly 66 new cells and six resampled pure cells")
    for key, spec in specs.items():
        cell = cells.get(key)
        if not isinstance(cell, dict):
            bad.append(f"{key}: missing cell")
            continue
        if cell.get("spec") != spec:
            bad.append(f"{key}: spec differs (levels, checkpoints, cut, seeds or group size)")
        if cell.get("n") != 2048 or cell.get("n_groups") != 8 or cell.get("group_seeds") != list(range(8)):
            bad.append(f"{key}: need eight groups of 256, in replay seed order 0-7")
        bad.extend(f"{key}: {reason}" for reason in GR.check_cell_scores(cell))
    return bad


def load_references(root, hashes, reference_hashes=None):
    """W2M-GR's admitted runs must use the exact checkpoint hashes of this GX run."""
    refs, bad = {}, []
    for seed in G.GX_SEEDS:
        r, why = GR.load_json(GR.seed_path(root, seed))
        if why:
            bad.append(f"W2M-GR seed {seed}: {why}")
            continue
        try:
            reasons = GR.check_result(r, seed)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError, ZeroDivisionError) as exc:
            reasons = [f"malformed reference ({type(exc).__name__})"]
        if reasons:
            bad.extend(f"W2M-GR seed {seed}: {v}" for v in reasons)
            continue
        if reference_hashes is not None and E.sha256_json(r) != reference_hashes.get(str(seed)):
            bad.append(f"W2M-GR seed {seed}: reference content digest differs")
        for arm in ("m2", "m3"):
            if r["protocol"]["ckpt_sha256"][arm] != hashes.get(f"{arm}_s{seed}"):
                bad.append(f"W2M-GR seed {seed}: {arm} checkpoint digest differs")
        refs[seed] = r
    return refs, bad


def summarize(cells, refs):
    """Read the primary scorer only; retain TinyStories-33M quantities beside it."""
    quantities = G.gx_quantities(cells, refs)
    primary = quantities["gpt2-large"]
    sanity = {}
    for seed in G.GX_SEEDS:
        for arm in ("m2", "m3"):
            m = cells[f"pure_{arm}_s{seed}"]["genppl_gpt2-large"]
            ref = refs[seed]["cells"][f"{arm}_{arm}"]["genppl_gpt2-large"]["mean"]
            sanity[f"{arm}_s{seed}"] = {"mean": m["mean"], "se": m["se"], "gr_mean": ref,
                                          "within_3se": abs(m["mean"] - ref) <= 3 * m["se"]}
    return {"quantities": quantities, "sanity": sanity,
            "readings": {"compatibility": G.read_gx_compatibility(primary["compatibility"]),
                         "cross_seed_localization": G.read_gx_localization(primary["cross_seed"]),
                         "concentrated_levels": G.read_gx_concentration(primary["single_level"])}}


def analyze_w2mgx(root=DEFAULT_ROOT):
    path = result_path(root)
    r, why = GR.load_json(path)
    bad = [why] if why else check_result(r)
    refs = {}
    if not bad:
        refs, bad = load_references(root, r["protocol"]["ckpt_sha256"],
                                    r["protocol"]["identity"]["w2mgr_reference_sha256"])
    out = {"available": False, "file": f"{NB}/{NB}__result.json", "problems": bad}
    if bad:
        return out
    try:
        summary = summarize(r["cells"], refs)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, AssertionError) as exc:
        out["problems"] = [f"undefined registered quantity ({type(exc).__name__}: {exc})"]
        return out
    # A failed sanity check blocks a trustworthy reading. Ratios remain descriptive.
    if not all(v["within_3se"] for v in summary["sanity"].values()):
        out.update({"problems": ["a resampled pure cell differs from W2M-GR by more than 3 SE"],
                    "sanity": summary["sanity"], "quantities": summary["quantities"]})
        return out
    out.update({"available": True, "primary": "gpt2-large", "checkpoint_sha256": r["protocol"]["ckpt_sha256"], **summary,
                "note": "single-level effects and cuts 3/5 are descriptive; supports the scoped W2C + W2M-GR result"})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    out = analyze_w2mgx(args.root)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    E.write_json(args.out, out)
    print("W2M-GX:", out["readings"] if out["available"] else "not available: " + "; ".join(out["problems"]))


if __name__ == "__main__":
    main()
