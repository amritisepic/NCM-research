"""WMT14 J2: run-level degenerate-source totals (J1b's 1% rule) survive interval logging and restarts.

Codex audit 4's draft noted that the totals lived only in an interval logging buffer, so a time-budget stop could lose
them; the fix keeps run-level counters in `traj_stats` that logging never resets, saves them in every checkpoint and
reports them in result.json."""
import ast
from pathlib import Path

import torch

from test_audit2_launch import consistency_fixture

ROOT = Path(__file__).resolve().parents[1]
WMT = (ROOT / "scripts/train_wmt14_L.py").read_text(encoding="utf-8")


def test_run_totals_accumulate_and_survive_the_logging_reset():
    ns, _, _, _ = consistency_fixture("zero")             # the all-zero start is the fixed point: degenerate
    ns["traj_stats"].update(run_sources=0, run_degenerate=0)
    for _ in range(2):
        ns["consist_loss"](4, torch.zeros(1, 1, dtype=torch.long))
    ns["traj_stats"].update(passes=0.0, n=0, states_before_fp=0.0, degenerate=0)   # what the loop does when it logs
    assert (ns["traj_stats"]["run_sources"], ns["traj_stats"]["run_degenerate"]) == (2, 2)


def test_every_checkpoint_saves_and_every_resume_restores_the_totals():
    tree = ast.parse(WMT)
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "trainer.save"]
    assert saves and all("extra={'run_stats': run_stats()}" in ast.unparse(s) for s in saves)
    assert "resume_extra = trainer.maybe_resume(modules, rng)" in WMT
    assert '_rs = resume_extra.get("run_stats", {})' in WMT and '"degenerate_pct"' in WMT
