"""scripts/analyze_j1.py: registered J1 screen / H12 / J1a readings and the shared-output diagnostic, on synthetic runs."""
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("analyze_j1", os.path.join(ROOT, "scripts", "analyze_j1.py"))
aj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aj)


def _write_arm(root, arm, hyp, nfe, bleu, len_ratio, cap_bleu, degenerate_pct=0.0):
    d, prefix, steps = aj.FROZEN_ARM if arm == "J2-F" else aj.ARMS[arm]
    os.makedirs(os.path.join(root, d), exist_ok=True)
    fe = {"val_nll": 1.0}
    for split in ("val", "test"):
        n = len(hyp)
        capped = {str(c): {"bleu_lc": cap_bleu.get(c, bleu), "passes_mean": float(min(c, max(nfe))),
                           "len_ratio": len_ratio, "identical_to_exact_pct": 100.0} for c in range(1, 13)}
        fe[split] = {"n": n, "exact_passes_mean": sum(nfe) / n, "bleu_lc": bleu, "len_ratio": len_ratio,
                     "capped": capped, "fixed_point_equals_sequential_pct": 100.0}
        h = {f"hyp_{split}": hyp, f"nfe_{split}": nfe, f"hyp_{split}_capped": {str(c): hyp for c in range(1, 13)}}
        json.dump(h, open(os.path.join(root, d, f"{prefix}__hyp_{split}.json"), "w"))
    if arm == "J2":
        json.dump({"hyp_val": hyp[:4], "nfe_val": nfe[:4]},
                  open(os.path.join(root, d, f"{prefix}__hyp_val_step{steps}.json"), "w"))
    r = {"run": prefix, "steps": steps, "generator_flops_proxy_total": 0 if arm == "J0" else 7,
         "final_eval": fe, "degenerate_pct": degenerate_pct}
    json.dump(r, open(os.path.join(root, d, f"{prefix}__result.json"), "w"))


def _runs(tmp_path, j2_nfe, j2_bleu, j2_hyp=None, j2_len=0.93, j1_nfe=12):
    hyp = [f"sentence {i}" for i in range(10)]
    _write_arm(tmp_path, "J0", hyp, [11] * 10, 22.6, 0.93, {9: 22.5})
    _write_arm(tmp_path, "J1", hyp, [j1_nfe] * 10, 21.8, 0.93, {9: 21.5, 8: 20.0})
    _write_arm(tmp_path, "J2", j2_hyp or hyp, j2_nfe, j2_bleu, j2_len, {})
    return aj.analyse(str(tmp_path))


def test_passing_j2_advances_and_is_not_killed(tmp_path):
    r = _runs(tmp_path, [8] * 10, 22.0)
    assert r["screen"]["advance"] and not r["h12_test"]["kill"]
    assert r["j1a"]["cap"] == 8 and r["j1a"]["len_ok"]


@pytest.mark.parametrize("nfe,bleu", [([10] * 10, 22.0), ([8] * 10, 21.0)])
def test_screen_fails_on_passes_or_bleu(tmp_path, nfe, bleu):
    r = _runs(tmp_path, nfe, bleu)
    assert not r["screen"]["advance"] and r["h12_test"]["kill"]


def test_kill_when_not_below_j1_by_one_pass(tmp_path):
    r = _runs(tmp_path, [8] * 10, 22.0, j1_nfe=8)
    assert r["screen"]["advance"] and not r["h12_test"]["below_J1_ok"] and r["h12_test"]["kill"]


def test_shared_outputs_are_counted_and_excluded(tmp_path):
    hyp = ["X"] * 5 + [f"s {i}" for i in range(5)]
    nfe = [2] * 5 + [10] * 5
    r = _runs(tmp_path, nfe, 18.0, j2_hyp=hyp, j2_len=0.75)
    ex = r["exploratory"]["test"]
    assert ex["shared_output_pct"]["J2"] == pytest.approx(50.0) and ex["shared_output_pct"]["J0"] == 0.0
    assert ex["J2_passes_on_shared"] == 2.0 and ex["passes_on_rest"]["J2"] == 10.0
    assert not r["j1a"]["len_ok"]


# J2-F candidate (audit 6, A6-2): the original arms keep J2's shared-output mask; J2-F's own rest uses its own mask
def test_candidate_j2f_keeps_j2_mask_for_original_arms(tmp_path):
    hyp = [f"sentence {i}" for i in range(10)]
    _write_arm(tmp_path, "J0", hyp, [11] * 10, 22.6, 0.93, {9: 22.5})
    _write_arm(tmp_path, "J1", hyp, [12] * 10, 21.8, 0.93, {9: 21.5, 8: 20.0})
    _write_arm(tmp_path, "J2", ["X"] * 5 + hyp[5:], [2] * 5 + [10] * 5, 18.0, 0.75, {})
    _write_arm(tmp_path, "J2-F", hyp, [9, 9, 9, 9, 9, 7, 7, 7, 7, 7], 22.0, 0.93, {})
    base = aj.analyse(str(tmp_path))
    r = aj.analyse(str(tmp_path), candidate="J2-F")
    for split in ("val", "test"):
        ex, ex0 = r["exploratory"][split], base["exploratory"][split]
        for arm in ("J0", "J1", "J2"):
            assert ex["passes_on_rest"][arm] == ex0["passes_on_rest"][arm]     # J2's mask: sentences 5-9
        assert ex["passes_on_rest"]["J0"] == 11.0 and ex["passes_on_rest"]["J2"] == 10.0
        assert ex["passes_on_rest"]["J2-F"] == 8.0                             # its own (empty) mask: all ten
        assert ex["shared_output_pct"]["J2-F"] == 0.0
    assert r["candidate"] == "J2-F" and r["h12_test"]["passes"] == 8.0
