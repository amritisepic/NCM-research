"""Recompute saved W1 aggregates: python -B tests/test_audit3_numbers.py --write.

No neural scorer is rerun. NLL is a readback; group and MAUVE aggregates,
comparisons, screens, rollback traces and log arithmetic are recomputed.
Use --sample-statistics to independently check entropy and other sample metrics.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/reviews/codex_audit3_numbers.json"
ARMS = {"A0": ("ctrl", "released16"), "A1": ("ctrl", "ctrl_s0"),
        "A2": ("requant", "requant_s0"), "A3": ("onpol", "onpol_s0"),
        "A4": ("b1", "b1_s0"), "A5": ("onpol2", "onpol2_s0"), "A6": ("soft", "soft_s0")}
COMPARATORS = {"A2": ["A1"], "A3": ["A1"], "A4": ["A1", "A3"],
               "A5": ["A1", "A3"], "A6": ["A1"]}


def recompute_sample_statistics(samples, hashes):
    """Use the actual BPE and pure metric definitions, without loading neural models."""
    import math
    import pickle
    import re
    from audit2_support import definitions

    sys.path.insert(0, str(ROOT / "src/ncmr/ncm_ref"))
    from tokenization.bpe_tokenizer import BPETokenizer

    loader = definitions("src/ncmr/ncm_ref_loader.py",
                         ["_NoCodeUnpickler", "safe_pickle_load"], {"pickle": pickle})
    paths = {name: ROOT / "data/tinystories" / name for name in
             ["bpe_vocab_8192.pickle", "bpe_merges_8192.pickle"]}
    for path in paths.values():
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    bpe = BPETokenizer(loader["safe_pickle_load"](paths["bpe_vocab_8192.pickle"]),
                       loader["safe_pickle_load"](paths["bpe_merges_8192.pickle"]),
                       {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"})
    ns = definitions("src/ncmr/eval_text.py", ["unigram_entropy", "rep_n", "distinct_n", "word_dups"],
                     {"math": math, "Counter": Counter, "_WORD": re.compile(r"[a-z']+")})
    ids = [bpe.encode(bpe.decode(row)) for row in samples["ids"]]
    entropy = [ns["unigram_entropy"](row) for row in ids if row]
    values = {"entropy": st.mean(entropy), "len_tokens": st.mean(map(len, ids)),
              **ns["word_dups"]([bpe.decode(row) for row in ids])}
    for n in [2, 3, 4]:
        values["rep_" + str(n)] = st.mean(ns["rep_n"](row, n) for row in ids)
    for n in [2, 3]:
        values["distinct_" + str(n)] = ns["distinct_n"](ids, n)
    return values


def compute(sample_statistics=False):
    hashes, discrepancies, arms, traces, provenance = {}, [], {}, {}, {}
    subject_paths = ["scripts/train_gen.py", "scripts/train_sudoku_L.py", "src/ncmr/train.py",
                     "src/ncmr/corruption.py", "notebooks/make_notebooks.py", "docs/preregistration.md"]
    baseline_hashes = {}
    for path in subject_paths:
        hashes[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        baseline_hashes[path] = hashlib.sha256(subprocess.check_output(
            ["git", "show", "d501333:" + path], cwd=ROOT)).hexdigest()
    history = subprocess.check_output(["git", "log", "--format=%h %cI", "d501333"], cwd=ROOT, text=True)
    revisions = [(line.split()[0], datetime.fromisoformat(line.split()[1])) for line in history.splitlines()]
    def read(path):
        path = ROOT / path
        raw = path.read_bytes()
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    def check(label, got, saved, tolerance=1e-10):
        if abs(got-saved) > tolerance:
            discrepancies.append(dict(quantity=label, recomputed=got, saved=saved))
    for arm, (folder, stem) in ARMS.items():
        directory = Path("results/colab") / ("w1_L_ts_" + folder)
        source = directory / ("L_ts_" + stem + "__result.json")
        result = read(source)
        ev = result["final_eval"]
        samples = read(directory / ("L_ts_" + stem + "__samples_final.json"))
        counts = Counter(samples["groups"])
        assert sorted(counts) == list(range(15)) + [42]
        assert set(counts.values()) == {256}
        assert len(samples["ids"]) == 4096
        summary = {"source": source.as_posix(), "steps": result["steps"],
                   "wall_s": result["wall_s"], "generator_flops_proxy": result["flops_est_total"],
                   "group_seed_order": sorted(counts), "n_per_group": 256,
                   "clean_val_nll_readback": ev["val_nll"],
                   "clean_val_nll_per_level_rounded_readback": ev["val_nll_per_level"],
                   "entropy_readback": ev["entropy"], "nfe_per_sample": ev["nfe_per_sample"]}
        if sample_statistics:
            values = recompute_sample_statistics(samples, hashes)
            summary["sample_statistics_recomputed"] = values
            for key, value in values.items():
                check(arm + "/recomputed_" + key, value, ev[key])
            print("recomputed sample statistics", arm, flush=True)
        for metric, suffix in [("genppl_gpt2-large", "per_group"),
                               ("genppl_TinyStories-33M", "per_group"), ("mauve", "per_seed")]:
            values = ev[metric + "_" + suffix]
            avg, sd = st.mean(values), st.stdev(values)
            check(arm+"/"+metric, avg, ev[metric])
            check(arm+"/"+metric+"_sd", sd, ev[metric+"_sd"])
            summary[metric] = {"values": values, "mean": avg, "sd_ddof1": sd,
                               "se": sd/math.sqrt(len(values))}
        for field in ["rep_2", "rep_3", "rep_4", "distinct_2", "distinct_3",
                      "len_tokens", "dup_adj1_per_1k", "dup_adj2_per_1k", "mauve_n"]:
            summary[field+"_readback"] = ev[field]
        arms[arm] = summary
        meta = read(directory / "_meta.json")
        finished = datetime.strptime(meta["finished"], "%a %b %d %H:%M:%S %Y").replace(tzinfo=timezone.utc)
        nominal = finished - timedelta(seconds=result["wall_s"])
        provenance[arm] = dict(finished_unzoned=meta["finished"],
            nominal_start_if_finish_is_utc_and_immediate=nominal.isoformat(),
            torch=result["torch"], grad_accum=result["config"]["train"]["grad_accum"],
            compile=result["config"]["train"]["compile"], exact_sha=None,
            evidence="16 groups, eval_precision, per-level corruption logs imply R024 or equivalent changes; branch pulls, restarts, collection delay and unzoned times prevent exact SHA inference.")
        revision_date = dict(revisions)["b397656"]
        candidates = [(sha, when) for sha,when in revisions if revision_date <= when <= finished]
        candidates.sort(key=lambda row: row[1])
        provenance[arm]["post_R024_candidates_by_commit_time_if_UTC"] = [sha for sha,_ in candidates]
        prior = [(sha,when) for sha,when in candidates if when <= nominal]
        provenance[arm]["latest_commit_before_nominal_start_heuristic_only"] = prior[-1][0] if prior else None
        metric_path = ROOT / directory / ("L_ts_" + stem + "__metrics.jsonl")
        if metric_path.exists():
            raw = metric_path.read_bytes()
            hashes[metric_path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
            rows = [json.loads(line) for line in raw.splitlines()]
            rollbacks, repeats, curve = [], defaultdict(list), []
            for line, row in enumerate(rows, 1):
                if line > 1 and row["step"] < rows[line-2]["step"]:
                    rollbacks.append(dict(line=line, from_step=rows[line-2]["step"], to_step=row["step"],
                                          inferred_checkpoint=(row["step"]//250)*250))
                kind = "eval" if "eval" in row else "train"
                # Preserve semantic payloads; exclude timings and throughput from replay comparison.
                payload = {k:v for k,v in row.items() if k not in
                           ("wall_s", "steps_per_s", "tflops_eff")}
                if "eval" in payload:
                    payload["eval"] = {k:v for k,v in payload["eval"].items() if k != "gen_wall_s"}
                    curve.append(dict(line=line, **payload))
                repeats[(row["step"],kind)].append(dict(line=line, wall_s=row["wall_s"], payload=payload))
            duplicate = [dict(step=k[0], kind=k[1], occurrences=v,
                              identical_semantic_payload=all(z["payload"]==v[0]["payload"] for z in v[1:]))
                         for k,v in repeats.items() if len(v)>1]
            traces[arm] = dict(rollbacks=rollbacks, duplicated_steps=duplicate, eval_curve=curve,
                              interpretation="repeated training steps imply restart/replay; a training and eval row at the same step is normal; no checkpoint files are present")
    comparisons = []
    for arm, controls in COMPARATORS.items():
        for control in controls:
            a, b = arms[arm], arms[control]
            delta = a["genppl_gpt2-large"]["mean"] - b["genppl_gpt2-large"]["mean"]
            entropy = a["entropy_readback"] - b["entropy_readback"]
            mauve = a["mauve"]["mean"] - b["mauve"]["mean"]
            groupdiff = [x-y for x,y in zip(a["genppl_gpt2-large"]["values"], b["genppl_gpt2-large"]["values"])]
            record = dict(arm=arm, comparator=control, genppl_delta_arm_minus_comparator=delta,
                          genppl_gain=-delta, group_differences_in_sorted_seed_order=groupdiff,
                          group_difference_sd=st.stdev(groupdiff),
                          unpaired_sampling_se=math.sqrt((a["genppl_gpt2-large"]["sd_ddof1"]**2+b["genppl_gpt2-large"]["sd_ddof1"]**2)/16),
                          entropy_gap=entropy, mauve_gap=mauve,
                          mauve_quantization_seed_differences=[x-y for x,y in zip(a["mauve"]["values"],b["mauve"]["values"])],
                          clean_nll_gap=a["clean_val_nll_readback"]-b["clean_val_nll_readback"],
                          passes_ppl=(-delta >= .35), passes_entropy=(abs(entropy) <= .05),
                          passes_mauve=(mauve >= -.01))
            record["passes_all"] = all(record[k] for k in ["passes_ppl", "passes_entropy", "passes_mauve"])
            comparisons.append(record)
    curve = {r["step"]:r["eval"] for r in traces["A1"]["eval_curve"]}
    matched = {}
    for step in [1080,1350]:
        gain = curve[1800]["genppl_gpt2-large"]-curve[step]["genppl_gpt2-large"]
        matched[str(step)] = dict(a1_curve=curve[step], a1_final_curve=curve[1800],
                                 fewer_steps_gain_same_256_protocol=gain, confound_triggered=gain>=.5,
                                 onpolicy_curve_comparisons={})
        for arm in (["A5"] if step==1080 else ["A3","A4"]):
            other = next(r["eval"] for r in reversed(traces[arm]["eval_curve"]) if r["step"]==step)
            matched[str(step)]["onpolicy_curve_comparisons"][arm] = dict(
                final_curve_genppl=other["genppl_gpt2-large"],
                gain_vs_matched_a1=curve[step]["genppl_gpt2-large"]-other["genppl_gpt2-large"],
                entropy_gap=other["entropy"]-curve[step]["entropy"])
    r014 = read("results/b1_repair_capacity.json")
    acc = r014["token_accuracy"]
    single = [100*v["no_repair"] for k,v in acc.items() if k.startswith("L") and "uniform" in k]
    levels = [1,8,16,24,32,40,52,64,76,92,108,128]
    dis = r014["generator_one_step_disagreement"]
    r014_summary = dict(source="results/b1_repair_capacity.json", n=r014["n_windows"],
        cpu_minutes=r014["wall_s"]/60, single_uniform_token_survival_range_pct=[min(single),max(single)],
        sampled_disagreement_pct=[100*x for x in dis["sampled"]],
        greedy_disagreement_pct=[100*x for x in dis["greedy"]],
        all_but_finest_sampled_changed_codes_all_slots_pct=100*sum(x*l for x,l in zip(dis["sampled"][:-1],levels[:-1]))/sum(levels),
        all_but_finest_sampled_changed_codes_coarse_slots_pct=100*sum(x*l for x,l in zip(dis["sampled"][:-1],levels[:-1]))/sum(levels[:-1]),
        token_accuracy_percent={k:{m:100*v for m,v in a.items()} for k,a in acc.items()},
        rank_table=r014["generator_code_rank_in_encoder_distance_order"],
        disagreements=["R014 calls 1055/1118/454 the L1-L3 sampled medians; the JSON indexes them L0-L2. Actual L1-L3 are 1118/454/37.",
                       "R014's corrupted_frac (including 58% all-but-finest on-policy) was recorded in the retarget=True branch, versus re-quantized labels. It is not direct disagreement with canonical codes; see separately weighted canonical disagreement.",
                       "The on-policy recipe changes 45.475% of labels, below the stated 46-58% range (uniform 57.533%).",
                       "L8-L10 sampled top-16 fractions range from 80.49% to 88.29%, rather than the stated 81-88%."])
    se = .59*math.sqrt(2/16)
    normal = st.NormalDist()
    screen_noise = dict(assumed_group_sd=.59, groups=16, unpaired_difference_se=se,
                        probability_pass={str(g):1-normal.cdf((.35-g)/se) for g in [0,.5,.7]},
                        interpretation="Sampling-only normal approximation; no training-seed uncertainty.")
    summary = dict(audit_ref="d501333",
        original_subject_git_blob_sha256=baseline_hashes,
        recomputation_checkout_ref=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        method="stdlib means/sample SDs from saved group scores, five-seed MAUVE means/SDs; not a neural scorer rerun",
        limitations=[("Sample entropy, length, repetitions, distinct n-grams and word duplicates are independently recomputed after BPE decode/re-encode."
                      if sample_statistics else "Sample statistics are saved readbacks; use --sample-statistics with the BPE assets to independently recompute them."),
                     "NLL is read back, not rescored: W1 weights/checkpoints are absent. Per-level entries are rounded to .01.",
                     "MAUVE SD is quantization-seed variability, not uncertainty across independent datasets.",
                     "The group's sorted seed order differs from generation order [42,0,...,14]."],
        source_sha256=hashes, arms=arms, comparisons=comparisons, step_matched=matched,
        sanity_a1_minus_a0=arms["A1"]["genppl_gpt2-large"]["mean"]-arms["A0"]["genppl_gpt2-large"]["mean"],
        nominal_advancers=sorted({r["arm"] for r in comparisons if r["passes_all"]}),
        conditional_controls=dict(A1m_triggered=any(r["arm"]=="A3" and r["passes_all"] for r in comparisons),
                                  A6s_triggered=any(r["arm"]=="A6" and r["passes_all"] for r in comparisons)),
        screen_sampling_noise=screen_noise, traces=traces, provenance=provenance,
        R014=r014_summary,
        R024=dict(numeric_claims_not_recomputable_from_results_json=["34 passing tests", "20-puzzle equality: fixture size, no historical per-puzzle result", "soft-gap discrepancy 0.05 -> <=3e-6", "~1% omitted work", "~0.5 CPU h"],
                 artifact_reference="docs/reviews/codex_audit_numbers.json audits R018-R021, not a raw measurement ledger for R024",
                 current_reverification="See audit report and protocol tests; historical test counts/timing cannot be recovered from result files."),
        aggregate_discrepancies=discrepancies)
    return summary


def test_saved_w1_aggregates_recompute():
    data = compute()
    assert not data["aggregate_discrepancies"]
    assert all(a["n_per_group"] == 256 for a in data["arms"].values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--sample-statistics", action="store_true")
    args = ap.parse_args()
    data = compute(sample_statistics=args.sample_statistics)
    if args.write:
        old = json.loads(OUT.read_text()) if OUT.exists() else {}
        for key in ["cpu_experiment", "verification"]:
            if key in old:
                data[key] = old[key]
        OUT.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"arms":{k:{m:v for m,v in a.items() if m in
                      ("steps", "clean_val_nll_readback", "entropy_readback", "wall_s")}
                             for k,a in data["arms"].items()},
                      "comparisons":data["comparisons"], "step_matched":data["step_matched"],
                      "nominal_advancers":data["nominal_advancers"],
                      "discrepancies":data["aggregate_discrepancies"]}, indent=2))
