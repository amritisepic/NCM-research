"""Audit 6 probes for the frozen-teacher run's production identity gate.

Only demonstrated defects are strict xfails; no CLI training or downloads.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from test_audit4_protocol import ROOT, WMT, source


def manifest_program():
    """Execute the trainer's actual manifest construction and validation block."""
    tree = ast.parse(source(WMT))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.If) and
                 ast.unparse(n.test) == "a.eval_only and os.path.exists(ckpt_path)")
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Expr) and
               isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "OmegaConf.save")
    return compile(ast.Module(body=tree.body[start:end], type_ignores=[]), WMT, "exec")


@pytest.mark.parametrize("first,second", [("consist", "consist_frozen"), ("consist_frozen", "consist")])
@pytest.mark.parametrize("changed", ["arm", "recipe", "base_generator"])
# fixed (was a strict xfail: A6-1: the manifest is ignored when no training checkpoint exists)
def test_recorded_run_without_checkpoint_refuses_changed_identity(tmp_path, first, second, changed):
    base, run = tmp_path / "artifacts/wmt14", tmp_path / "run"
    base.mkdir(parents=True)
    run.mkdir()
    data = tmp_path / "data.npz"
    data.write_bytes(b"training archive")
    for name in ("config.yaml", "tokenizer.pt", "generator.pt"):
        (base / name).write_bytes(b"base file")
    cfg = OmegaConf.load(ROOT / "configs/scale/L_wmt_consist_frozen.yaml")
    cfg.arm = first
    ns = dict(ROOT=str(tmp_path), os=os, json=json, sys=sys, OmegaConf=OmegaConf,
              a=SimpleNamespace(run_dir=str(run), data=str(data), eval_only=False, calib_steps=0),
              cfg=cfg, dev="cpu", ckpt_path=str(run / "ckpt.pt"),
              manifest_path=str(run / "run_manifest.json"),
              digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    program = manifest_program()
    exec(program, ns)
    original_manifest = (run / "run_manifest.json").read_bytes()
    # A preemption before the first checkpoint leaves a real manifest and logged steps.
    (run / "metrics.jsonl").write_text('{"step": 1, "loss": 1.0}\n', encoding="utf-8")
    assert not (run / "ckpt.pt").exists()
    if changed == "arm":
        cfg.arm = second
    elif changed == "recipe":
        cfg.train.lr *= 2
    else:
        (base / "generator.pt").write_bytes(b"other gen")  # Same size, different digest.
    rejected = False
    try:
        exec(program, ns)
    except (SystemExit, ValueError):
        rejected = True
    assert rejected, "a recorded run accepted a different identity before its first checkpoint"
    assert (run / "run_manifest.json").read_bytes() == original_manifest
