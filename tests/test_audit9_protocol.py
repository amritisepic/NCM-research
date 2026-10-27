"""Audit 9: timing boundaries, main-program integrity gate and notebook wiring."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_d1_bench import BB, tiny_wmt_like
from test_audit9_numbers import ARMS, independent_rule

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_timer_excludes_warmup_and_brackets_every_interleaved_decode(monkeypatch, precision):
    events = []
    clock = iter(range(1000))
    monkeypatch.setattr(BB, "sync", lambda device: events.append("sync"))
    monkeypatch.setattr(BB.time, "perf_counter", lambda: (events.append("clock"), next(clock))[1])
    items = [BB.Item(torch.tensor([[i]])) for i in range(3)]
    observations = []

    def make_fn(name):
        def fn(item):
            observations.append((name, int(item.x[0, 0]), torch.is_autocast_enabled("cpu"), torch.is_grad_enabled()))
            events.append((name, int(item.x[0, 0])))
            return "output", 4, "fixed"
        return fn

    configs = [(name, "exact", make_fn(name)) for name in ("a", "b", "c")]
    result = BB.run_timing(configs, items, "cpu", precision, repeats=3, warmup=1, seed=17)
    assert events[:4] == [("a", 0), ("b", 0), ("c", 0), "sync"]
    expected = []
    for i in range(3):
        for repeat in range(3):
            for j in np.random.default_rng([17, i, repeat]).permutation(3):
                expected.extend(["sync", "clock", (configs[j][0], i), "sync", "clock"])
    assert events[4:] == expected
    assert all(amp == (precision == "bf16") and not grad for _, _, amp, grad in observations)
    assert not torch.is_autocast_enabled("cpu")
    for value in result.values():
        assert value["lat"] == [1.0] * 3 and value["passes"] == [4] * 3
        assert value["nondeterministic"] == []


@pytest.mark.parametrize("failure", [None, "wrong_j2f", "val_rule", "repeat"])
def test_real_main_selects_fastest_sides_and_refuses_integrity_failures(tmp_path, monkeypatch, failure):
    import ncmr.ncm_ref_loader as loader
    tok, net, _ = tiny_wmt_like()
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"audit checkpoint identity")
    bundle = SimpleNamespace(net=net, vqvae=tok,
                             pre_tokenizer=SimpleNamespace(eos_token_id=0, pad_token_id=39, decode=lambda *a, **k: "unused"))
    monkeypatch.setattr(loader, "load_ncm", lambda *a, **k: bundle)
    monkeypatch.setattr(BB.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(BB.torch, "load", lambda *a, **k: {"model": net.state_dict()})
    monkeypatch.setattr(BB, "load_items", lambda *a: [BB.Item(torch.zeros(1, 12, dtype=torch.long)) for _ in range(256)])
    monkeypatch.setattr(BB, "flop_units", lambda *a: {})
    monkeypatch.setattr(BB, "flops_per_item", lambda *a: (100, 80, 10))
    monkeypatch.setattr(BB, "rule_reproduction", lambda *a: {"ok": failure != "val_rule", "bleu_checked": False})
    j0 = json.loads((ROOT / "results/colab" / (ARMS["J0"] + "__hyp_test.json")).read_text())
    j2 = json.loads((ROOT / "results/colab" / (ARMS["J2-F"] + "__hyp_test.json")).read_text())
    stable0, stable2 = independent_rule(j0, "test", 1)[:256], independent_rule(j2, "test", 1)[:256]

    def timing(configs, *args):
        result = {}
        assert not any(p.requires_grad for p in net.parameters())
        for name, side, _ in configs:
            cached = name.endswith("cached")
            if side == "seq":
                outputs, passes, latency = j0["hyp_test"][:256], [11] * 256, 1.0 if cached else 1.2
            elif side == "exact":
                outputs, passes, latency = j0["hyp_test"][:256], j0["nfe_test"][:256], 2.0 if cached else 2.5
            else:
                expected = stable2 if name.startswith("J2-F") else stable0
                if failure == "wrong_j2f" and name.startswith("J2-F"):
                    expected = stable0  # A concrete wrong checkpoint: released J0 in place of J2-F.
                outputs, passes = [o for _, o in expected], [p for p, _ in expected]
                latency = (0.5 if cached else 0.6) if name.startswith("J2-F") else 1.5
            result[name] = {"lat": [latency] * 256, "out": list(outputs), "passes": list(passes),
                            "stop": ["fixed"] * 256,
                            "nondeterministic": [0] if failure == "repeat" and name == "J2-F jacobi stable1 cached" else []}
        return result

    monkeypatch.setattr(BB, "run_timing", timing)
    run_dir = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["bench_batch1", "--task", "wmt14", "--allow_cpu",
                                      "--run_dir", str(run_dir), "--j2f_ckpt", str(checkpoint), "--precisions", "fp32"])
    BB.main()
    r = json.loads((run_dir / "result.json").read_text())["fp32"]
    assert r["ratios_vs_fastest_seq"]["sequential"] == "J0 seq cached"
    assert r["ratios_vs_fastest_seq"]["J2-F jacobi stable1"]["config"] == "J2-F jacobi stable1 cached"
    assert r["gate"]["ratio"] == 0.5
    assert r["integrity_ok"] == (failure is None)
    assert r["gate"]["verdict"].startswith("pass" if failure is None else "not read (integrity)")
    if failure == "wrong_j2f":
        assert len(r["integrity"]["J2F_stable1_cached_vs_saved_rule"]) == 253


def test_notebooks_equal_generator_and_keep_registered_commands():
    spec = importlib.util.spec_from_file_location("audit9_notebooks", ROOT / "notebooks/make_notebooks.py")
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    for name in ("d1_batch1_wallclock", "e1_equivalence", "w2m_mech_eval"):
        saved = json.loads((ROOT / "notebooks" / (name + ".ipynb")).read_text())
        assert saved == nb.make(name, nb.RUNS[name])
        source = "\n".join("".join(c["source"]) for c in saved["cells"] if c["cell_type"] == "code")
        assert "'result.json'" in source and "provenance.jsonl" in source
        assert "'hyp_test.json'" in source and "'hyp_val.json'" in source
        assert nb.RUNS[name]["seeds"] == [0]
    d1 = nb.RUNS["d1_batch1_wallclock"]["cmds"]
    assert len(d1) == 2 and "L_wmt_consist_frozen_s0/final.pt" in d1[0]
    assert "--val_text {R}/../data/wmt14_train_pairs_val_text.json" in d1[0]
    assert "--task sudoku" in d1[1]
    e1 = nb.RUNS["e1_equivalence"]["cmds"]
    for cmd, n in zip(e1, (64, 128, 128)):
        assert f"--n {n}" in cmd and "--round 64" in cmd and "--coarse_n 8" in cmd
    assert "--models A0,A1,A2,A4" in e1[0]
    assert "--models J0,J1,J2F" in e1[1]
    assert "wmt14_train_pairs.npz" in e1[1]
    assert "--models S0" in e1[2]
    # Defaults used by these exact commands are part of the notebook protocol.
    tree = ast.parse((ROOT / "scripts/e1_equivalence.py").read_text(encoding="utf-8"))
    defaults = {}
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument" and call.args:
            for kw in call.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[ast.literal_eval(call.args[0])] = kw.value.value
    assert {k: defaults[k] for k in ("--cut_min", "--full_cut_min", "--max_samples", "--hits", "--seed")} == {
        "--cut_min": 5, "--full_cut_min": 8, "--max_samples": 2048, "--hits": 20, "--seed": 0}
