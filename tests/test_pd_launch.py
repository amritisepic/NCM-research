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
