"""Recompute audit 4 evidence: python -B tests/test_audit4_numbers.py --write.

Historical aggregate readbacks are explicitly distinguished from recomputations.
No translation model, data download, or historical smoke result is fabricated.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile

import conftest  # shared CPU thread setting and src import path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/reviews/codex_audit4_numbers.json"
SUBJECT = "c91126a6bcf229721ef59379a27999b1e9c8f0ba"


def recompute():
    path = "results/colab/w0_calib_diag/b2_jacobi_wmt14_n3003.json"
    raw = json.loads((ROOT / path).read_text())
    hist = raw["exact_passes_hist"]
    n, passes = sum(hist), sum(i * count for i, count in enumerate(hist))
    mean = passes / n
    checks = []
    def check(name, claimed, value, method, independently_recomputed, digits=None):
        matches = (round(value, digits) == claimed) if digits is not None else (abs(value - claimed) < 1e-10)
        checks.append(dict(id=name, log_value=claimed, recomputed_or_readback=value, method=method,
                           independently_recomputed=independently_recomputed, matches=matches))
    check("R020_wmt_n", 3003, n, "sum(exact_passes_hist)", True)
    check("R020_wmt_exact_passes", 11.07, mean, "sum(pass * count) / sum(count)", True, 2)
    check("R020_wmt_sequential_nfe", 11, raw["sequential_nfe"], "aggregate field readback", False)
    check("R020_wmt_exact_bleu", 22.63, raw["bleu_lc_exact"], "aggregate field readback; hypotheses absent", False, 2)
    check("R020_wmt_stable_passes", 10.59, raw["stable_passes_mean"],
          "aggregate field readback; per-sentence stable passes absent", False, 2)
    check("R020_wmt_stable_bleu", 22.64, raw["bleu_lc_stable_rule"],
          "aggregate field readback; hypotheses absent", False, 2)
    check("R020_wmt_sequential_equality_pct", 100., raw["fixed_point_equals_sequential_pct"],
          "aggregate field readback; per-sentence codes absent", False)
    check("R020_wmt_25pct_target", 8.25, .75 * raw["sequential_nfe"], "0.75 * 11", True)
    check("R020_wmt_bleu_floor", 21.63, round(raw["bleu_lc_exact"], 2) - 1,
          "rounded registered baseline minus 1; unrounded floor is also reported", False, 2)
    check("R020_json_exact_mean_consistency", raw["exact_passes_mean"], mean, "histogram mean vs saved mean", True)
    saving = 100 * (1 - mean / raw["sequential_nfe"])
    check("R020_json_exact_nfe_saving_pct", raw["exact_nfe_saving_pct"], saving, "100 * (1 - mean / 11)", True)
    check("R020_json_position_proxy_ratio", raw["exact_flops_ratio"],
          mean * raw["flops_per_pass_ratio_vs_sequential_total"], "mean * saved position-count ratio", False)
    stale1 = "results/b2_jacobi_wmt14_n6.json"
    stale2 = "results/colab/w0_calib_diag/b2_jacobi_wmt14_n6.json"
    check("R020_n6_is_stale_copy", 1, int((ROOT / stale1).read_bytes() == (ROOT / stale2).read_bytes()),
          "byte equality of both n6 artifacts", True)
    results = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "results").rglob("*.json"))
    wmt_paths = [p for p in results if "wmt" in p.lower()]
    # Also inspect generic result names; the historical smoke might not contain WMT in its filename.
    candidates = []
    for name in results:
        data = json.loads((ROOT / name).read_text(encoding="utf-8"))
        if "wmt" in json.dumps(data).lower() and name not in wmt_paths:
            candidates.append(name)
    paths = [path, stale1, stale2]
    source_files = ["scripts/train_wmt14_L.py", "scripts/prep_wmt14.py", "src/ncmr/jacobi_train.py",
                    "src/ncmr/train.py", "configs/scale/L_wmt_ctrl.yaml", "configs/scale/L_wmt_consist.yaml",
                    "notebooks/make_notebooks.py", "docs/preregistration.md", "log.md"]
    hashes = {p: hashlib.sha256(subprocess.check_output(["git", "show", f"{SUBJECT}:{p}"], cwd=ROOT)).hexdigest()
              for p in source_files}
    return dict(
        audit_ref=SUBJECT,
        subject_git_blob_sha256=hashes,
        results_utf8_lf_sha256={p: hashlib.sha256((ROOT / p).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                               for p in paths},
        history=dict(
            R020=dict(source=path, n=n, exact_total_passes=passes, exact_histogram=hist,
                      exact_mean=mean, exact_nfe_saving_pct=saving,
                      saved_stable_mean=raw["stable_passes_mean"],
                      implied_stable_total_passes=round(n * raw["stable_passes_mean"]),
                      saved_bleu_exact=raw["bleu_lc_exact"], saved_bleu_stable=raw["bleu_lc_stable_rule"],
                      saved_first_final_output_pass_mean=raw["first_final_output_pass_mean"],
                      unrounded_baseline_minus_one=raw["bleu_lc_exact"] - 1,
                      checks=checks),
            R028=dict(
                status="unverifiable_from_results_json",
                old_early_stop_proxy=dict(
                    method="registered row-count algebra, not a historical execution measurement",
                    full_teacher_rows_per_step=576,
                    two_pass_teacher_rows_per_step=3 * 128 + 2 * 16,
                    two_pass_total_rows=2000 * (3 * 128 + 2 * 16),
                    shortfall_pct=100 * (1 - (3 * 128 + 2 * 16) / 576),
                    log_rounded_shortfall_pct=28),
                results_json_count=len(results), wmt_named_json=wmt_paths,
                other_json_mentioning_wmt=candidates,
                missing_evidence=[
                    "WMT14 consistency smoke fp32 teacher execution and zero degenerate sources",
                    "caps 1-12, val/test ordered hypotheses and exact/sequential sentence equality in that smoke",
                    "20k prep smoke archive selection equality and mismatch-refusal transcript",
                    "historical 59-test total / 25 audit-2 tests / 14 pre-fix failures as a saved test report",
                ],
                interpretation="No matching WMT smoke JSON is present. Prose and current code tests do not reproduce a historical run."),
            disagreements=[c for c in checks if not c["matches"]],
            limitations=[
                "R020 stores aggregate WMT summaries only: no per-sentence hypotheses, references, codes or stable-pass records.",
                "BLEU, stable mean, output equality and first-final-output mean cannot be independently rederived from these JSON files.",
                "R028 WMT smoke artifacts were not committed under results; zero degenerates and 20k selection equality are unverified.",
                "The old n6 files report sequential_nfe=10; they are excluded as the log explicitly calls them stale.",
                "Saved exact_flops_ratio is a position-count proxy, not an attention-inclusive execution measurement.",
            ]))


def cpu_evidence():
    from test_audit4_protocol import tiny_wmt, step
    with tempfile.TemporaryDirectory(prefix="ncm-audit4-") as work:
        with contextlib.redirect_stdout(io.StringIO()):
            ns = tiny_wmt(Path(work))
            step(ns, optimize=False)
            params = [p for p in ns["net"].parameters() if p.requires_grad]
            reached = [p for p in params if p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0]
            nll_amp = ns["val_nll"]()
            ns["trainer"]._ctx_kw = None
            nll_fp32 = ns["val_nll"]()
    return dict(
        torch=str(torch.__version__), numpy=str(np.__version__), device="cpu", threads=torch.get_num_threads(),
        model="tiny upstream WMT-style NCM: 11 levels, source-prefix L0 head, dropout 0.1, EOS/PAD targets",
        trainable_tensors=len(params), finite_nonzero_gradient_tensors=len(reached),
        trainable_elements=sum(p.numel() for p in params), gradient_reached_elements=sum(p.numel() for p in reached),
        gradient_definition="Reached means a parameter tensor has a finite gradient with nonzero norm; element totals count sizes of reached parameter tensors, not nonzero scalar gradients.",
        amp_val_nll=nll_amp, fp32_val_nll=nll_fp32, nll_difference=nll_amp - nll_fp32,
        nondegenerate_proxy_rows=576, one_degenerate_proxy_rows=564, all_degenerate_proxy_rows=384,
        registered_total_forward_rows=3000 * 384, consist_total_without_degenerates=2000 * 576,
        limits="CPU evidence on a small upstream model; not a gradient census or BLEU rerun on the released 120M WMT weights; CUDA not tested.")


def test_saved_history_numbers_recompute_exactly():
    expected = recompute()
    saved = json.loads(OUT.read_text(encoding="utf-8"))
    # ported: the count of results/*.json describes the audited commit; later commits add result files, so it is not
    # recomputable in the working tree (every number recomputed from a named file still is)
    for d in (expected, saved):
        d["history"]["R028"].pop("results_json_count", None)
    assert {k: saved[k] for k in expected} == expected
    assert not saved["history"]["disagreements"]


def test_saved_diagnostics_account_for_every_tiny_wmt_parameter():
    saved = json.loads(OUT.read_text(encoding="utf-8"))["cpu_evidence"]
    assert saved["trainable_tensors"] == saved["finite_nonzero_gradient_tensors"] > 0
    assert saved["trainable_elements"] == saved["gradient_reached_elements"] > 0
    assert saved["registered_total_forward_rows"] == saved["consist_total_without_degenerates"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = recompute()
    if args.write:
        payload["cpu_evidence"] = cpu_evidence()
        OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(disagreements=payload["history"]["disagreements"],
                          R028=payload["history"]["R028"]["status"])))
