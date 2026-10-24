"""The exploratory scripts print no local paths (Codex PR 6 P1 4367731857; AGENTS.md: the repository is public)."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stats(tmp_path, monkeypatch):
    """w1_sample_stats with a stand-in checkout under tmp_path: tiny bigram model, human windows, samples, vocab."""
    mod = load_script("w1_sample_stats")
    repo = tmp_path / "repo"
    monkeypatch.setattr(mod, "ROOT", str(repo))
    uni = np.ones(mod.V)
    monkeypatch.setattr(mod, "bigram_model", lambda n, d: ((uni + 1) / (uni.sum() + mod.V), uni,
                                                           np.array([5 * mod.V + 6]), np.array([3])))
    monkeypatch.setattr(mod, "examples", lambda split, n: np.full((4, 8), 5, dtype=np.int64))
    for d, prefix in mod.ARMS.values():
        (repo / "results/colab" / d).mkdir(parents=True, exist_ok=True)
        (repo / "results/colab" / d / f"{prefix}__samples_final.json").write_text(json.dumps({"ids": [[5, 6, 7]] * 4096}))
    (repo / "data/tinystories").mkdir(parents=True)
    with open(repo / "data/tinystories/bpe_vocab_8192.pickle", "wb") as f:
        pickle.dump({i: b"x" for i in range(mod.V)}, f)
    return mod, repo


def run(mod, monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["w1_sample_stats.py", *args])
    mod.main()
    return capsys.readouterr().out


def no_local_path(text, path):
    return all(form not in text for form in {str(path), Path(path).as_posix()})


def test_default_invocation_prints_repo_relative_paths(stats, monkeypatch, capsys, tmp_path):
    mod, repo = stats
    out = run(mod, monkeypatch, capsys, "--read", "2")
    assert (repo / "results/w1_sample_stats.json").is_file()
    assert "wrote results/w1_sample_stats.json" in out
    assert "wrote results/w1_samples_read_s2026.md" in out
    assert no_local_path(out, tmp_path)


def test_absolute_out_outside_the_repo_prints_only_the_name(stats, monkeypatch, capsys, tmp_path):
    mod, _ = stats
    target = tmp_path / "elsewhere" / "stats.json"
    out = run(mod, monkeypatch, capsys, "--out", str(target), "--read", "2")
    assert target.is_file()
    assert "wrote stats.json" in out and "wrote w1_samples_read_s2026.md" in out
    assert no_local_path(out, tmp_path)
    assert no_local_path(target.read_text(encoding="utf-8"), tmp_path)


def test_shown_paths(stats):
    mod, repo = stats
    assert mod.shown(str(repo / "results" / "x.json")) == "results/x.json"
    assert mod.shown(str(repo.parent / "outside" / "x.json")) == "x.json"
    if os.name == "nt":   # another drive: relpath raises ValueError
        other = "Z:" if not str(repo).upper().startswith("Z:") else "Y:"
        assert mod.shown(os.path.join(other + os.sep, "elsewhere", "x.json")) == "x.json"


def output_path_prints(source):
    """print(...) calls that use a.out / a.save other than through shown(...)."""
    bad = []

    def leaks(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "shown":
            return False
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "a"
                and node.attr in ("out", "save")):
            return True
        return any(leaks(child) for child in ast.iter_child_nodes(node))

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            if any(leaks(arg) for arg in [*node.args, *(k.value for k in node.keywords)]):
                bad.append(node.lineno)
    return bad


def test_onpolicy_repro_prints_output_paths_through_shown():
    source = (ROOT / "scripts/onpolicy_repro.py").read_text(encoding="utf-8")
    assert output_path_prints(source) == []
    assert "shown(a.out)" in source
    tree = ast.parse(source)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "shown"]
    assert len(fn) == 1
    ns = {"os": os, "ROOT": str(ROOT)}
    exec(compile(ast.Module(body=fn, type_ignores=[]), "scripts/onpolicy_repro.py", "exec"), ns)
    assert ns["shown"](str(ROOT / "results" / "onpolicy_repro" / "x.json")) == "results/onpolicy_repro/x.json"
    assert ns["shown"](str(ROOT.parent / "elsewhere" / "x.json")) == "x.json"
    assert output_path_prints('print("wrote", a.out)') == [1]   # the checker itself sees a bare path
