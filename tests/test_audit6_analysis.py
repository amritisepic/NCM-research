"""Audit 6: the registered frozen arm must be selectable without replacing J2.

This probe executes the production analysis CLI on four distinct synthetic runs.
The proposed repair adds a backwards-compatible ``--candidate J2-F`` selector.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_j1.py"


def _analysis_module():
    spec = importlib.util.spec_from_file_location("audit6_analyze_j1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(root, directory, prefix, steps, passes, bleu, shared=False):
    directory = root / directory
    directory.mkdir(exist_ok=True)
    hyp = ["shared output"] * 5 + [f"unique {i}" for i in range(5)] if shared else [f"unique {i}" for i in range(10)]
    fe = {"val_nll": 1.0}
    for split in ("val", "test"):
        caps = {str(c): {"bleu_lc": bleu, "passes_mean": min(passes, c),
                         "len_ratio": 0.93, "identical_to_exact_pct": 100.0} for c in range(1, 13)}
        fe[split] = {"n": len(hyp), "exact_passes_mean": passes, "bleu_lc": bleu, "len_ratio": 0.93,
                     "capped": caps, "fixed_point_equals_sequential_pct": 100.0}
        payload = {f"hyp_{split}": hyp, f"nfe_{split}": [passes] * len(hyp),
                   f"hyp_{split}_capped": {str(c): hyp for c in range(1, 13)}}
        (directory / f"{prefix}__hyp_{split}.json").write_text(json.dumps(payload), encoding="utf-8")
    result = {"run": prefix, "steps": steps, "generator_flops_proxy_total": 0 if steps == 0 else 7,
              "trajectory_sources": 32000 if steps == 2000 else 0, "degenerate_sources": 0,
              "degenerate_pct": 0.0, "final_eval": fe}
    (directory / f"{prefix}__result.json").write_text(json.dumps(result), encoding="utf-8")
    if steps:
        (directory / f"{prefix}__hyp_val_step{steps}.json").write_text(
            json.dumps({"hyp_val": hyp, "nfe_val": [passes] * len(hyp)}), encoding="utf-8")


# fixed (was a strict xfail: A6-2: analysis CLI cannot select J2-F or retain all four registered diagnostics)
def test_registered_frozen_analysis_keeps_the_original_three_arms(tmp_path):
    aj = _analysis_module()
    _write_run(tmp_path, "w2c_L_wmt_ctrl", "L_wmt_released", 0, 11, 23.0)
    _write_run(tmp_path, "w2c_L_wmt_ctrl", "L_wmt_ctrl_s0", 3000, 12, 22.5)
    _write_run(tmp_path, "w2c_L_wmt_consist", "L_wmt_consist_s0", 2000, 10, 18.0, shared=True)
    _write_run(tmp_path, "w2c_L_wmt_consist_frozen", "L_wmt_consist_frozen_s0", 2000, 8, 22.75)
    original = json.loads(json.dumps(aj.analyse(str(tmp_path))))
    output = tmp_path / "analysis.json"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--root", str(tmp_path), "--out", str(output),
                           "--candidate", "J2-F"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, "production analysis rejects the registered frozen arm: " + proc.stderr
    frozen = json.loads(output.read_text(encoding="utf-8"))
    for arm in ("J0", "J1", "J2"):
        assert frozen["arms"][arm] == original["arms"][arm]
    assert frozen["arms"]["J2-F"]["test"]["passes"] == 8.0
    assert frozen["screen"]["advance"] and frozen["screen"]["val_bleu"] == 22.75
    assert frozen["h12_test"]["passes"] == 8.0 and not frozen["h12_test"]["kill"]
    assert frozen["j1a"]["cap"] == 8
    for split in ("val", "test"):
        share = frozen["exploratory"][split]["shared_output_pct"]
        assert set(share) == {"J0", "J1", "J2", "J2-F"}
        assert share["J2"] == 50.0 and share["J2-F"] == 0.0
        for key in ("passes_on_rest", "words_ratio_vs_J0_on_rest", "saving_on_rest_pct"):
            for arm in ("J0", "J1", "J2"):
                assert frozen["exploratory"][split][key][arm] == original["exploratory"][split][key][arm]
        assert frozen["exploratory"][split]["passes_on_rest"]["J2-F"] == 8.0
        assert frozen["exploratory"][split]["words_ratio_vs_J0_on_rest"]["J2-F"] == 1.0
