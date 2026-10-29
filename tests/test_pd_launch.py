"""CPU prelaunch gate, generated notebooks and cumulative pilot budget constraints."""
import importlib.util
import json

import pytest

from w2m_support import ROOT


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_coverage_cpu_gate_inclusive_boundary():
    coverage = module("pd_coverage", "scripts/pd_coverage.py")
    report = {"steps": 20, "seed": 0, "device": "cpu", "coverage": {"overall": 0.80}}
    assert coverage.check_coverage(report) == 0.80
    report["coverage"]["overall"] = 0.799999
    with pytest.raises(ValueError, match="below 0.80"):
        coverage.check_coverage(report)
    for key, value in (("steps", 19), ("seed", 1), ("device", "cuda")):
        broken = dict(report, **{key: value})
        with pytest.raises(ValueError, match="twenty seed-0"):
            coverage.check_coverage(broken)


def test_cpu_coverage_forward_subbatch_preserves_eval_logits_and_sampling_draw_shapes():
    import torch
    from w2m_support import make_arm, tiny_case

    coverage = module("pd_coverage_split", "scripts/pd_coverage.py")
    tok, net = tiny_case()
    original = make_arm("pd", tok, net, seed=0)
    split = make_arm("pd", tok, net, seed=0)
    coverage.chunk_frozen_forward(split, 2)
    x = torch.randint(0, 32, (8, 128), generator=torch.Generator().manual_seed(123))
    canon = original.canonical(x)
    plan = original.draw_plan(8)
    with torch.no_grad():
        full_logits = original._logits(original.frozen, original.frozen, canon)
        split_logits = split._logits(split.frozen, split.frozen, canon)
    torch.testing.assert_close(split_logits, full_logits, rtol=1e-6, atol=1e-6)
    shapes = []
    draw = split.draw_fine

    def record(probs):
        shapes.append(tuple(probs.shape))
        return draw(probs)

    split.draw_fine = record
    p, q = original.proposals(canon, plan), split.proposals(canon, plan)
    assert shapes == [(8, original.S - original.l0, int(net.max_vocab))]
    assert torch.equal(p, q)
    assert torch.equal(original.repl_gen.get_state(), split.repl_gen.get_state())


def test_pd_budget_includes_resumes_and_outstanding_concurrent_reservations():
    budget = module("pilot_budget", "scripts/run_pilot_budget.py")
    ledger = {"sessions": []}
    assert budget.available(ledger, "pd", "p") == 25 * 60
    ledger["sessions"].append({"arm": "p", "reserved_s": 1500, "elapsed_s": 900})
    assert budget.available(ledger, "pd", "p") == 10 * 60
    ledger["sessions"].append({"arm": "d", "reserved_s": 1500})
    assert budget.available(ledger, "pd", "u") == 15 * 60
    ledger["sessions"].append({"arm": "p", "reserved_s": 600})
    assert budget.available(ledger, "pd", "u") == 10 * 60
    ledger["sessions"].append({"arm": "u", "reserved_s": 600})
    assert budget.available(ledger, "pd", "u") == 0


def test_pd_generated_notebooks_have_guard_gate_budget_provenance_release():
    maker = module("pd_notebooks", "notebooks/make_notebooks.py")
    for arm in "pdu":
        name = f"pd_L_ts_{arm}"
        notebook = maker.make(name, maker.RUNS[name])
        assert notebook == json.loads((ROOT / f"notebooks/{name}.ipynb").read_text())
        source = "\n".join("".join(c["source"]) for c in notebook["cells"])
        assert "This run needs an A100" in source
        assert "scripts/pd_coverage.py --verify" in source
        assert f"--experiment pd --arm {arm}" in source
        assert "--micro_batch 64" in source
        assert "'event': 'start'" in source and "'event': 'end'" in source
        assert "runtime.unassign()" in source
        assert maker.RUNS[name]["seeds"] == [0]


def test_pd_cap_path_collects_and_releases_before_raising(monkeypatch):
    import ast
    import sys
    from types import ModuleType, SimpleNamespace

    maker = module("pd_cap_notebooks", "notebooks/make_notebooks.py")
    for arm in "pdu":
        name = f"pd_L_ts_{arm}"
        notebook = maker.make(name, maker.RUNS[name])
        source = next("".join(c["source"]) for c in notebook["cells"] if "CMDS_RAW = list(CMDS)" in "".join(c["source"]))
        tree = ast.parse(source)
        cap = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "p.returncode == 124")
        error = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "p.returncode != 0")
        order = []
        colab = ModuleType("google.colab")
        colab.runtime = SimpleNamespace(unassign=lambda: order.append("release"))
        monkeypatch.setitem(sys.modules, "google.colab", colab)

        def collect(code):
            assert code == maker.COLLECT.replace("__NAME__", name)
            order.append("collect")

        ns = {"p": SimpleNamespace(returncode=124), "exec": collect}
        with pytest.raises(SystemExit, match="command failed"):
            exec(compile(ast.Module(body=[cap, error], type_ignores=[]), "cap_path", "exec"), ns)
        assert order == ["collect", "release"]
