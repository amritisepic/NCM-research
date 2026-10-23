"""Reconstruct R018--R021 without importing training code or loading models.

Run from any directory: python -B tests/recompute_r018_r021.py
The default immutable Git commit contains the requested entries. Use --ref
WORKTREE to inspect a checkout containing those entries, or --ref COMMIT.
Only results/**/*.json are numeric inputs. No training, downloads, or writes
outside docs/reviews are performed. Exit 1 means a logged claim mismatches;
exit 2 means required evidence is missing or malformed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

AUDIT_COMMIT = "1c7185c0e0de23d364c8bad33b2606c852bbd00d"
ROOT = Path(__file__).resolve().parents[1]


class Evidence:
    def __init__(self, ref=AUDIT_COMMIT):
        self.ref = ref
        if ref != "WORKTREE":
            self.ref = subprocess.check_output(
                ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
                cwd=ROOT, text=True,
            ).strip()
        self.hashes = {}

    def read(self, path):
        if self.ref == "WORKTREE":
            raw = (ROOT / path).read_bytes()
        else:
            raw = subprocess.check_output(["git", "show", f"{self.ref}:{path}"], cwd=ROOT)
        self.hashes[path] = hashlib.sha256(raw).hexdigest()
        return raw.decode("utf-8-sig")

    def data(self, path):
        if not path.startswith("results/") or not path.endswith(".json"):
            raise ValueError("numeric evidence must be results/**/*.json")
        return json.loads(self.read(path))


class Audit:
    def __init__(self):
        self.rows = []

    def check(self, entry, claim, logged, actual, digits, source, *, readback=False, note=""):
        # A displayed decimal denotes rounding to that decimal place, not exact equality.
        tolerance = 0 if digits is None else 0.5 * 10 ** -digits + 1e-10
        if isinstance(logged, bool):
            ok = actual is logged
        else:
            ok = math.isfinite(actual) and abs(actual - logged) <= tolerance
        self.rows.append(dict(entry=entry, claim=claim, logged=logged, actual=actual,
                              status=("readback" if readback else "agrees") if ok else "mismatch",
                              tolerance=tolerance, source=source, note=note))

    def missing(self, entry, claim, source, note):
        self.rows.append(dict(entry=entry, claim=claim, status="unverifiable", source=source, note=note))


P1 = "results/colab/p1_tc_pilot/"
C4 = "results/colab/c4_sudoku_pilot/"
W0 = "results/colab/w0_calib_diag/"


def reconstruct(e):
    a = Audit()
    log = e.read("log.md")
    entries = {n: re.search(rf"(?ms)^## {n}\b.*?(?=^## R|\Z)", log).group() for n in
               ("R018", "R019", "R020", "R021")}
    p = {n: e.data(P1 + n + "__result.json") for n in
         ("tok_s", "ncm_s", "ts_fact_s", "ts_ar_s", "ar_s")}
    d = e.data("results/p1_tc_decomposition.json")
    l0 = p["ncm_s"]["final_eval"]["val_nll_level0_unigram"]
    ncm = p["ncm_s"]["final_eval"]["val_nll_nats_per_seq"] + l0
    fact_lv = p["ts_fact_s"]["final_eval"]["val_nll_per_level_nats_per_seq"]
    ar_lv = p["ts_ar_s"]["final_eval"]["val_nll_per_level_nats_per_seq"]
    fact, ar = sum(fact_lv) + l0, sum(ar_lv) + l0
    ar_token = p["ar_s"]["final_eval"]["val_nll_nats_per_seq"]
    gap, tc = ncm - ar_token, sum(f - r for f, r in zip(fact_lv, ar_lv))
    src = P1 + "{ncm_s,ts_fact_s,ts_ar_s,ar_s}__result.json#/final_eval"
    for label, expected, actual, digits in [
        ("tokenizer parameters (million)", 8.4, p["tok_s"]["params_total"] / 1e6, 1),
        ("NCM parameters (million)", 5.5, p["ncm_s"]["params_total"] / 1e6, 1),
        ("seed", 0, p["ncm_s"]["config"]["train"]["seed"], None),
        ("validation windows", 1000, p["ncm_s"]["config"]["train"]["eval_n"], None),
        ("final steps", 6000, p["ncm_s"]["steps"], None),
        ("level-0 NLL", 4.51, l0, 2), ("NCM NLL", 1232.7, ncm, 1),
        ("factorized NLL", 1232.8, fact, 1), ("within-level AR NLL", 1290.1, ar, 1),
        ("AR FLOP-matched NLL", 173.6, ar_token, 1),
        ("AR data-matched NLL", 212.4, d["nll_ar_data_matched"], 1),
        ("AR data-matched step", 1500, d["ar_data_matched_step"], None),
        ("AR batch size", 128, p["ar_s"]["config"]["train"]["batch_size"], None),
        ("AR FLOPs (PFLOP proxy)", 4.1, p["ar_s"]["flops_est_total"] / 1e15, 1),
        ("NCM FLOPs (PFLOP proxy)", 4.0, p["ncm_s"]["flops_est_total"] / 1e15, 1),
        ("FLOP-matched gap", 1059.1, gap, 1),
        ("data-matched gap", 1020.3, ncm - d["nll_ar_data_matched"], 1),
        ("TC", -57.3, tc, 1), ("TC share (%)", -5.4, 100 * tc / gap, 1),
        ("factorized relative difference (%)", 0.01, 100 * abs(fact - ncm) / ncm, 2),
        ("tokenizer full accuracy (%)", 99.17, 100 * p["tok_s"]["final_eval"]["val_acc_full"], 2),
        ("truncated accuracy first (%)", 7.6, 100 * p["tok_s"]["final_eval"]["val_acc_per_level"][0], 1),
        ("truncated accuracy last (%)", 99.2, 100 * p["tok_s"]["final_eval"]["val_acc_per_level"][-1], 1),
        ("representation/excess term", 1116.5, ar - ar_token, 1),
        ("NCM excess lower bound (rounded nats)", 1059, gap, 0),
        ("AR-code excess lower bound (conservative integer nats)", 1116, math.floor(ar - ar_token), None),
        ("NCM excess fraction (%)", 86, 100 * gap / ncm, 0),
    ]:
        a.check("R018", label, expected, actual, digits, src,
                readback=label in {"AR data-matched NLL", "AR data-matched step"})
    for k, expected in enumerate([1.1, 1.9, 1.0, 0.3, -2.0, -2.7, -2.6, -21.0, -14.7, -17.3, -1.2], 1):
        a.check("R018", f"TC level {k}", expected, fact_lv[k - 1] - ar_lv[k - 1], 1, src)
    trend = [r["nll_fact_total"] - r["nll_ar_within_total"] for r in d["by_step"]]
    for label, expected, actual, digits in [
        ("TC evaluations", 12, len(trend), None), ("TC first evaluation", -142, trend[0], 0),
        ("TC last evaluation", -57, trend[-1], 0),
        ("TC rises monotonically over all evaluations", True, all(y >= x for x, y in zip(trend, trend[1:])), None),
        ("TC negative at end", True, trend[-1] < 0, None),
        ("truncated accuracy monotonic", True, all(y >= x for x, y in zip(
            p["tok_s"]["final_eval"]["val_acc_per_level"], p["tok_s"]["final_eval"]["val_acc_per_level"][1:])), None),
    ]:
        a.check("R018", label, expected, actual, digits, "results/p1_tc_decomposition.json#/by_step")
    # Recheck every decomposition row against its component totals, including rows not quoted in the log.
    for row in d["by_step"]:
        a.check("R018", f"stored TC arithmetic at step {row['step']}", row["tc_total"],
                row["nll_fact_total"] - row["nll_ar_within_total"], 8,
                "results/p1_tc_decomposition.json#/by_step")
    a.missing("R018", "NCM -11 nats over last 500 steps; all four curves falling",
              P1 + "*__result.json", "JSON stores final values only for NCM/token AR; trajectories are in JSONL.")
    a.missing("R018", "embedding norms 7--15; default norm ~16 at width 256; P1b ~1 CU",
              "results/**/*.json", "Norm measurement not saved in JSON; init theory and future compute estimate are not run results.")
    a.missing("R018", "128-token window length", P1 + "*__result.json",
              "The JSON has no sequence-length field; length is a data-pipeline/design constant.")

    c = {n: e.data(C4 + n + "__result.json") for n in ("sdk_ncm_s_hard", "sdk_rncm_s_hard", "sdk_tok_hard")}
    cn, cr, ct = c["sdk_ncm_s_hard"], c["sdk_rncm_s_hard"], c["sdk_tok_hard"]
    src = C4 + "{sdk_ncm_s_hard,sdk_rncm_s_hard,sdk_tok_hard}__result.json"
    for key in ("val_token_acc", "val_board_exact"):
        a.check("R019", "tokenizer " + key + " (%)", 100, 100 * ct["final_eval_val"][key], 0, src, readback=True)
    for model, r, val, test in (("NCM", cn, 2.1, 3.00), ("RNCM", cr, 0.4, 1.05)):
        for label, expected, actual, digits in [
            ("params (million)", 5.66, r["params_total"] / 1e6, 2),
            ("steps", 20000, r["steps"], None), ("batch", 64, r["config"]["train"]["batch_size"], None),
            ("NFE", 12, r["test_released"]["nfe_per_sample"], None),
            ("val solve (%)", val, 100 * r["final_eval_val"]["val_solved"], 1),
            ("test solve (%)", test, r["test_released"]["solved_pct"], 2),
            ("val puzzles", 1000, r["config"]["train"]["eval_n"], None),
        ]:
            a.check("R019", model + " " + label, expected, actual, digits, src, readback=True)
    buckets = ["le6", "7-8", "9-10", "ge11", "search"]
    counts, solves = {}, {}
    for model, r, percentages in (("NCM", cn, [6.7, 2.4, 0.6, 0.8, 0.7]),
                                  ("RNCM", cr, [3.4, 0.3, 0.0, 0.0, 0.0])):
        depth = r["test_released"]["by_propagation_depth"]
        counts[model] = [depth[b]["n"] for b in buckets]
        solves[model] = [round(depth[b]["n"] * depth[b]["solved_pct"] / 100) for b in buckets]
        for b, expected_n, expected_pct, n, solved in zip(buckets, [563, 705, 322, 119, 291], percentages,
                                                        counts[model], solves[model]):
            a.check("R019", model + " " + b + " count", expected_n, n, None, src, readback=True)
            a.check("R019", model + " " + b + " solve (%)", expected_pct, 100 * solved / n, 1, src)
        a.check("R019", model + " test puzzles", 2000, sum(counts[model]), None, src)
        a.check("R019", model + " weighted overall test (%)", r["test_released"]["solved_pct"],
                100 * sum(solves[model]) / sum(counts[model]), 9, src)
    deep_n = sum(counts["NCM"][2:4])
    a.check("R019", "deep puzzles", 441, deep_n, None, src)
    for model, expected in (("NCM", 3), ("RNCM", 0)):
        a.check("R019", model + " deep solved", expected, sum(solves[model][2:4]), None, src)
    for label, expected, actual, digits in [
        ("val difference (points)", -1.7, 100 * (cr["final_eval_val"]["val_solved"] - cn["final_eval_val"]["val_solved"]), 1),
        ("test difference (points)", -1.95, cr["test_released"]["solved_pct"] - cn["test_released"]["solved_pct"], 2),
        ("deep difference (points)", -0.68, 100 * (sum(solves["RNCM"][2:4]) - sum(solves["NCM"][2:4])) / deep_n, 2),
        ("puzzle exposures (million)", 1.28, cn["steps"] * cn["config"]["train"]["batch_size"] / 1e6, 2),
        ("registered +1-point falsification fires", True,
         100 * (cr["final_eval_val"]["val_solved"] - cn["final_eval_val"]["val_solved"]) < 1, None),
    ]:
        a.check("R019", label, expected, actual, digits, src)
    a.missing("R019", "final train CE 0.159 / 0.127", src, "Not present in JSON; training records are JSONL.")
    a.missing("R019", "paper 51.2M exposures; released 25.5M parameters; 5x smaller body",
              "results/**/*.json", "External/model-architecture provenance; not reconstructible from the saved result JSON.")

    ref = e.data(W0 + "L_ts_released__result.json")
    ev = ref["final_eval"]
    src = W0 + "L_ts_released__result.json#/final_eval"
    for label, expected, actual, digits in [
        ("reference val NLL", 861.8, ev["val_nll"], 1),
        ("GPT2 Gen-PPL", 16.35, statistics.mean(ev["genppl_gpt2-large_per_group"]), 2),
        ("GPT2 group SD", 0.51, statistics.stdev(ev["genppl_gpt2-large_per_group"]), 2),
        ("TinyStories Gen-PPL", 12.01, statistics.mean(ev["genppl_TinyStories-33M_per_group"]), 2),
        ("entropy", 3.976, ev["entropy"], 3),
        ("MAUVE", 0.888, statistics.mean(ev["mauve_per_seed"]), 3),
        ("MAUVE SD", 0.021, statistics.stdev(ev["mauve_per_seed"]), 3),
        ("rep-2", 0.100, ev["rep_2"], 3), ("NFE", 11, ev["nfe_per_sample"], None),
        ("groups", 4, len(ev["genppl_gpt2-large_per_group"]), None),
        ("samples per group", 256, ev["n"] / len(ev["genppl_gpt2-large_per_group"]), None),
    ]:
        a.check("R020", label, expected, actual, digits, src,
                readback=label in {"reference val NLL", "entropy", "rep-2", "NFE"})
    # Validate stored aggregate statistics independently of the prose rounding.
    for name, values in (("genppl_gpt2-large", ev["genppl_gpt2-large_per_group"]),
                         ("genppl_TinyStories-33M", ev["genppl_TinyStories-33M_per_group"]),
                         ("mauve", ev["mauve_per_seed"])):
        for suffix, fn in (("", statistics.mean), ("_sd", statistics.stdev)):
            a.check("R020", "stored " + name + suffix, ev[name + suffix], fn(values), 10, src)
    old = e.data("results/phase0_released_ncm_tinystories_eval.json")
    old_ev = old.get("metrics", old.get("final_eval", old))
    # Older result uses the same metric names but can wrap them in 'eval'.
    if "genppl_gpt2-large" not in old_ev:
        old_ev = old["eval"]
    for key, val in (("genppl_gpt2-large", 16.10), ("genppl_gpt2-large_sd", 0.59)):
        a.check("R020", "R010 reference " + key, val, old_ev[key], 2,
                "results/phase0_released_ncm_tinystories_eval.json", readback=True)
    nll0 = e.data("results/phase0_released_generator_nll.json")
    # Field names are recorded in the evidence manifest; support the original format.
    total0 = sum(nll0["nll_nats_per_seq_per_level"])
    a.check("R020", "R008 reference NLL", 870.5, total0, 1,
            "results/phase0_released_generator_nll.json", readback=True)
    a.check("R020", "R008 token-straddling NLL share (%)", 43,
            100 * sum(nll0["nll_nats_per_seq_per_level"][7:10]) / total0, 0,
            "results/phase0_released_generator_nll.json")
    a.check("R020", "reference NLL change (%)", -1.0, 100 * (ev["val_nll"] / total0 - 1), 1, src)
    diag = e.data(W0 + "diag_tf_levels_released_ts.json")
    rows = diag["by_j"]
    src = W0 + "diag_tf_levels_released_ts.json#/by_j"
    for key, expected in (("n", 512), ("seed", 0)):
        a.check("R020", "diagnostic " + key, expected, diag[key], None, src, readback=True)
    for level, expected in ((1, 8), (3, 24), (4, 32), (5, 40)):
        a.check("R020", f"level {level} code positions", expected, diag["levels"][level], None, src, readback=True)
    ppls = [rows[str(j)]["genppl_gpt2-large"] for j in range(-1, 12)]
    logged_ppls = [16.00, 15.88, 15.39, 14.47, 12.36, 10.60, 9.51, 9.15, 8.96, 8.81, 8.75, 8.67, 8.62]
    for j, expected, actual in zip(range(-1, 12), logged_ppls, ppls):
        a.check("R020", f"diagnostic GPT2 PPL j={j}", expected, actual, 2, src, readback=True)
    for j, expected in ((-1, 11.71), (11, 3.49)):
        a.check("R020", f"diagnostic TinyStories PPL j={j}", expected,
                rows[str(j)]["genppl_TinyStories-33M"], 2, src, readback=True)
    gap_ppl = ppls[0] - ppls[-1]
    a.check("R020", "diagnostic PPL gap", 7.38, gap_ppl, 2, src)
    bands = [("level 0", -1, 0, 2), ("levels 1--3", 0, 3, 48),
             ("levels 4--5", 3, 5, 39), ("level 6", 5, 6, 5),
             ("levels 7--11", 6, 11, 5), ("levels 8--10", 7, 10, 3),
             ("levels 1--5", 0, 5, 87)]
    for label, left, right, expected in bands:
        a.check("R020", "quality loss share " + label + " (%)", expected,
                100 * (rows[str(left)]["genppl_gpt2-large"] - rows[str(right)]["genppl_gpt2-large"]) / gap_ppl, 0, src)
    a.check("R020", "straddling PPL cost (~0.2)", 0.2,
            rows["7"]["genppl_gpt2-large"] - rows["10"]["genppl_gpt2-large"], 1, src)
    for label, expected, actual in [
        ("entropy minimum", 3.94, min(r["entropy"] for r in rows.values())),
        ("entropy maximum", 3.97, max(r["entropy"] for r in rows.values())),
        ("MAUVE unconditional", 0.82, rows["-1"]["mauve"]),
        ("MAUVE real level 0", 0.87, rows["0"]["mauve"]),
        ("MAUVE j>=1 minimum", 0.91, min(rows[str(j)]["mauve"] for j in range(1, 12))),
        ("MAUVE j>=1 maximum", 0.97, max(rows[str(j)]["mauve"] for j in range(1, 12))),
    ]:
        a.check("R020", label, expected, actual, 2, src)
    a.check("R020", "levels 1--3 strict majority trigger", False,
            (rows["0"]["genppl_gpt2-large"] - rows["3"]["genppl_gpt2-large"]) / gap_ppl > 0.5, None, src)
    # This is a declared scaling estimate, not a measured standard error.
    a.missing("R020", "diagnostic SE ~0.4; ~0.2--0.3 schedule upside; A100 40GB, 5.4 CU/h, ~2.5 CU",
              src, "Single diagnostic group cannot estimate SE. Upside/compute are estimates, not saved measurements.")
    for task, name, expected_n, expected_passes, digits in [
        ("Sudoku", "b2_jacobi_sudoku_hard.json", 2000, 6.41, 2),
        ("TinyStories", "b2_jacobi_tinystories_seed0.json", 512, 11.94, 2),
        ("WMT14", "b2_jacobi_wmt14_n3003.json", 3003, 11.07, 2),
    ]:
        r = e.data(W0 + name)
        src = W0 + name
        hist = r["exact_passes_hist"]
        mean = sum(i * count for i, count in enumerate(hist)) / sum(hist)
        a.check("R020", task + " count", expected_n, sum(hist), None, src)
        a.check("R020", task + " histogram count matches n", r["n"], sum(hist), None, src)
        a.check("R020", task + " passes", expected_passes, mean, digits, src)
        a.check("R020", task + " stored passes from histogram", r["exact_passes_mean"], mean, 10, src)
        a.check("R020", task + " stored NFE saving (%)", r["exact_nfe_saving_pct"],
                100 * (1 - mean / r["sequential_nfe"]), 10, src)
        a.check("R020", task + " stored FLOP proxy ratio", r["exact_flops_ratio"],
                mean * r["flops_per_pass_ratio_vs_sequential_total"], 10, src,
                note="Position-count proxy, not a measured FLOP ratio.")
        a.check("R020", task + " sequential NFE", 12 if task == "Sudoku" else 11, r["sequential_nfe"], None, src)
        if task == "Sudoku":
            for label, expected, actual, nd in [
                ("NFE saving (%)", 46.6, 100 * (1 - mean / 12), 1),
                ("FLOP proxy ratio", 1.87, mean * r["flops_per_pass_ratio_vs_sequential_total"], 2),
                ("fixed point equals sequential (%)", 100, r["fixed_point_equals_sequential_pct"], None),
                ("exact solve (%)", 86.15, r["solved_pct_exact"], 2),
                ("verifier passes", 2.37, r["check_passes_mean"], 2),
                ("verifier solve (%)", 90.3, r["solved_pct_check_rule"], 1),
                ("verifier gain (points)", 4.15, r["solved_pct_check_rule"] - r["solved_pct_exact"], 2),
                ("stable passes", 2.92, r["stable_passes_mean"], 2),
                ("stable solve (%)", 88.2, r["solved_pct_stable_rule"], 1),
            ]:
                a.check("R020", task + " " + label, expected, actual, nd, src,
                        readback="solve" in label or "fixed point" in label or "passes" in label)
        if task == "WMT14":
            for label, expected, actual, nd in [
                ("fixed point equals sequential (%)", 100, r["fixed_point_equals_sequential_pct"], None),
                ("exact BLEU", 22.63, r["bleu_lc_exact"], 2),
                ("stable passes", 10.59, r["stable_passes_mean"], 2),
                ("stable BLEU", 22.64, r["bleu_lc_stable_rule"], 2),
                ("criterion d passes", 8.25, 0.75 * r["sequential_nfe"], 2),
                ("criterion d BLEU", 21.63, r["bleu_lc_exact"] - 1, 2),
            ]:
                a.check("R020", task + " " + label, expected, actual, nd, src,
                        readback=not label.startswith("criterion"))
    for stale, original in (("b2_jacobi_levels_hard.json", "results/b2_jacobi_levels_hard.json"),
                             ("b2_jacobi_wmt14_n6.json", "results/b2_jacobi_wmt14_n6.json")):
        a.check("R020", stale + " is a stale copy", True, e.data(W0 + stale) == e.data(original), None, original)
    old_sudoku = e.data("results/phase0_sudoku_repro_hard.json")
    # The old result is a readback of a different run, not a recomputed model score.
    solved_old = old_sudoku.get("solved_pct", old_sudoku.get("solve_rate_pct"))
    if solved_old is None:
        solved_old = old_sudoku["final"]["solved_pct"]
    a.check("R019", "released reference test solve (%)", 86.3, solved_old, 1,
            "results/phase0_sudoku_repro_hard.json", readback=True)

    r = e.data("results/sudoku_repair_ceiling_val.json")
    src = "results/sudoku_repair_ceiling_val.json"
    a.check("R021", "puzzles", 250, r["n"], None, src, readback=True)
    a.check("R021", "CPU minutes (log: 1.6)", 1.6, r["wall_s"] / 60, 1, src,
            note="JSON records wall time, not process CPU time.")
    a.check("R021", "baseline solve (%)", 88.0, r["released_solved_pct"], 1, src, readback=True)
    a.check("R021", "failure count", r["n_failures"], round(r["n"] * (1 - r["released_solved_pct"] / 100)), None, src)
    for t, expected in enumerate([88.0, 88.8, 88.8, 90.4, 93.2, 95.2, 95.6, 95.6, 95.6]):
        a.check("R021", f"oracle solved after {t} levels (%)", expected,
                r["oracle_solved_pct_after_t_repairs"][t], 1, src, readback=True)
    for t, expected in ((1, 7), (3, 20), (6, 63)):
        # Net aggregate improvement equals initial failures repaired only if earlier successes survive.
        net_gain = (r["oracle_solved_pct_after_t_repairs"][t] - r["released_solved_pct"]) / (100 - r["released_solved_pct"]) * 100
        a.check("R021", f"initial failures fixed after {t} levels (%)", expected,
                r["oracle_fixed_pct_of_failures_after_t"][t], 0, src, readback=True)
        a.check("R021", f"net gain agrees with failure-fix aggregate after {t}",
                r["oracle_fixed_pct_of_failures_after_t"][t], net_gain, 2, src,
                note="No per-board trajectories are stored to independently reclassify repairs.")
    a.check("R021", "plateau starts after six levels", True,
            len(set(r["oracle_solved_pct_after_t_repairs"][6:])) == 1, None, src)
    a.check("R021", "released test comparison (%)", 86.3, solved_old, 1,
            "results/phase0_sudoku_repro_hard.json", readback=True)
    a.missing("R021", "finest codebook has 16 codes", src,
              "Architecture property: present in the released config, not this result JSON.")
    a.missing("R018--R021", "registration thresholds, level labels, dates and identifiers",
              "log.md; docs/preregistration.md", "Design/provenance constants are not measured outcomes; no numerical recomputation can validate them.")
    return {"ref": e.ref, "log_entry_sha256": {n: hashlib.sha256(v.encode()).hexdigest() for n, v in entries.items()},
            "source_sha256": dict(sorted(e.hashes.items())), "counts": dict(Counter(r["status"] for r in a.rows)),
            "checks": a.rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", default=AUDIT_COMMIT)
    p.add_argument("--output", default="docs/reviews/codex_audit_numbers.json")
    args = p.parse_args(argv)
    dest = Path(args.output)
    if dest.is_absolute() or dest.parts[:2] != ("docs", "reviews") or ".." in dest.parts:
        p.error("output must be a relative path under docs/reviews")
    try:
        report = reconstruct(Evidence(args.ref))
    except (KeyError, OSError, ValueError, AttributeError, subprocess.CalledProcessError) as exc:
        print("Required evidence missing or malformed:", type(exc).__name__, str(exc)[:180])
        return 2
    (ROOT / dest).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / dest).write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))
    for row in report["checks"]:
        if row["status"] == "mismatch":
            print(f"{row['entry']} {row['claim']}: logged={row['logged']}, actual={row['actual']}")
    return int(bool(report["counts"].get("mismatch")))


if __name__ == "__main__":
    raise SystemExit(main())
