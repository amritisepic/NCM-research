"""R056 package build: runtime-release cell, the six new notebooks and the L1R GPU guard, the M3-0 config, w1_sample_stats'
skip-absent behaviour and every reading of scripts/analyze_w2c.py at its registered thresholds (docs/preregistration.md:
W2C, M3-0, L1R). CPU only; the notebook cells are executed from `notebooks/make_notebooks.py` itself with stub subprocess / torch.
"""
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


MK = load("make_notebooks_r056", "notebooks/make_notebooks.py")
AN = load("analyze_w2c_r056", "scripts/analyze_w2c.py")
NEW = ["w2c1_L_ts_m2", "w2c1_L_ts_m3", "w2p_L_ts_m3_c0", "w2mg_eval", "e1b_continue", "l1r_L_ts_m2r"]


def src(c):
    return "".join(c["source"])


def nl(text):
    return text.strip(chr(10))


# ------------------------------------------------------------------------------------------------ AC1: release cell
@pytest.mark.parametrize("name", sorted(MK.RUNS))
def test_every_notebook_ends_with_release_cell(name):
    for nb in (MK.make(name, MK.RUNS[name]), json.load(open(ROOT / "notebooks" / f"{name}.ipynb"))):   # built and on disk
        last, before = src(nb["cells"][-1]), src(nb["cells"][-2])
        assert nb["cells"][-1]["cell_type"] == "code"
        assert "RELEASE_RUNTIME = True" in last and "from google.colab import runtime" in last
        assert last.rstrip().splitlines()[-1].lstrip().startswith("runtime.unassign()")
        assert "Results copied to" in before          # the release comes after the results are on Drive
        earlier = nb["cells"][:-1]
        if name == "w2mgx_eval":
            # GX's executed cap tests cover this single exception: failure, exhausted ledger, copy, then release.
            run_cells = [c for c in earlier if src(c) == nl(MK.GX_RUN)]
            assert len(run_cells) == 1
            assert "unassign" not in "".join(src(c) for c in earlier if c is not run_cells[0])
        else:
            assert "unassign" not in "".join(src(c) for c in earlier)


def test_release_cell_unassigns_and_respects_switch(monkeypatch):
    calls = []
    runtime = types.ModuleType("google.colab.runtime")
    runtime.unassign = lambda: calls.append(1)
    colab = types.ModuleType("google.colab")
    colab.runtime = runtime
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.runtime", runtime)
    exec(MK.RELEASE, {})
    assert calls == [1]
    calls.clear()
    exec(MK.RELEASE.replace("RELEASE_RUNTIME = True", "RELEASE_RUNTIME = False"), {})
    assert calls == []


def test_notebooks_are_the_old_five_cells_plus_the_release_cell():
    """Every notebook (the L1R one also has its A100 guard, the W2M-GR one its GPU guard) is markdown, setup, data, run, collect +
    the release cell. GX alone uses the cap-aware run template; every other run uses the common template."""
    for name, spec in MK.RUNS.items():
        cells = MK.make(name, spec)["cells"]
        guard = 1 if spec.get("require_a100") or spec.get("require_gpu") else 0
        assert len(cells) == 6 + guard, name
        run = MK.GX_RUN if name == "w2mgx_eval" else MK.RUN
        assert src(cells[2 + guard]) == nl(MK.DATA) and src(cells[3 + guard]) == nl(run)
        assert src(cells[4 + guard]) == nl(MK.COLLECT.replace("__NAME__", name))
        assert src(cells[-1]) == nl(MK.RELEASE)


# ------------------------------------------------------------------------------------------------ AC2: new notebooks
def spec_cmds(name):
    return MK.RUNS[name]["cmds"]


def test_new_entries_exist_with_exact_commands():
    assert all(n in MK.RUNS and (ROOT / "notebooks" / f"{n}.ipynb").exists() for n in NEW)
    assert spec_cmds("w2c1_L_ts_m2") == ["python scripts/train_gen.py --config configs/scale/L_ts_m2_frozen_pw.yaml "
                                         "--run_dir {R}/L_ts_m2_frozen_pw_s{S} --seed {S}"]
    assert spec_cmds("w2c1_L_ts_m3") == ["python scripts/train_gen.py --config configs/scale/L_ts_m3_uniform_wrong.yaml "
                                         "--run_dir {R}/L_ts_m3_uniform_wrong_s{S} --seed {S}"]
    assert spec_cmds("w2p_L_ts_m3_c0") == ["python scripts/train_gen.py --config configs/scale/L_ts_m3_uniform_wrong_c0.yaml "
                                           "--run_dir {R}/L_ts_m3_uniform_wrong_c0_s{S} --seed {S}"]
    assert spec_cmds("w2mg_eval") == ["python scripts/w2mg_eval.py --run_root {R} --run_dir {R}/w2mg_eval --seed 0"]
    assert spec_cmds("e1b_continue") == ["python scripts/e1b_continue.py --run_dir {R}/e1b "
                                         "--data {R}/../data/wmt14_train_pairs.npz --max_minutes 150"]
    assert spec_cmds("l1r_L_ts_m2r") == [
        "python scripts/w2mg_gate.py --result {R}/w2mg_eval/result.json",
        "python scripts/train_gen.py --config configs/scale/L_ts_m2r_frozen_pw_l1reassign.yaml "
        "--run_dir {R}/L_ts_m2r_frozen_pw_l1reassign_s{S} --seed {S}"]


def test_new_entries_seeds_gpus_and_needs():
    for n in ("w2c1_L_ts_m2", "w2c1_L_ts_m3"):
        assert MK.RUNS[n]["seeds"] == [1, 2, 3] and MK.RUNS[n]["gpu"] == "A100-40GB"
        nb = json.load(open(ROOT / "notebooks" / f"{n}.ipynb"))
        assert "SEEDS = [1, 2, 3]" in src(nb["cells"][1])
        assert "~51" in MK.RUNS["w2c1_L_ts_m2"]["desc"] and "~30" in MK.RUNS["w2c1_L_ts_m3"]["desc"]
        assert "SEEDS" in MK.RUNS[n]["desc"] and "one seed" in MK.RUNS[n]["desc"]
    assert MK.RUNS["w2p_L_ts_m3_c0"]["seeds"] == [0] and MK.RUNS["w2p_L_ts_m3_c0"]["gpu"] == "A100-40GB"
    assert MK.RUNS["w2mg_eval"]["gpu"] == "L4" and MK.RUNS["e1b_continue"]["gpu"] == "L4"
    assert {"ckpt", "wmt14", "ts"} <= set(MK.RUNS["e1b_continue"]["needs"])
    assert MK.RUNS["l1r_L_ts_m2r"]["gpu"].startswith("A100-40GB")
    assert "gate" in MK.RUNS["l1r_L_ts_m2r"]["desc"] and "approved" in MK.RUNS["l1r_L_ts_m2r"]["desc"]
    caps = {"w2c1_L_ts_m2": "70", "w2c1_L_ts_m3": "45", "w2p_L_ts_m3_c0": "45", "w2mg_eval": "120", "e1b_continue": "150",
            "l1r_L_ts_m2r": "75"}
    for n, cap in caps.items():
        assert f"cap {cap}" in MK.RUNS[n]["desc"] or f"cap: {cap}" in src(json.load(open(ROOT / "notebooks" / f"{n}.ipynb"))["cells"][0])


# files owned by the other implementers of the package may not be merged yet when this runs on a partial tree
PENDING = {"scripts/w2mg_eval.py", "scripts/w2mg_gate.py", "scripts/e1b_continue.py", "configs/scale/L_ts_m2r_frozen_pw_l1reassign.yaml"}


@pytest.mark.parametrize("name", NEW)
def test_commands_point_at_existing_files(name):
    for c in spec_cmds(name):
        toks = c.split()
        for t in [toks[1]] + [toks[i + 1] for i, x in enumerate(toks) if x == "--config"]:
            if not (ROOT / t).exists() and t in PENDING:
                pytest.skip(f"{t} comes from another part of the package")
            assert (ROOT / t).exists(), t


# ------------------------------------------------------------------------------------------------ the L1R GPU guard
def run_guard(device_name, mem_bytes, cuda=True):
    props = SimpleNamespace(total_memory=mem_bytes)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda i: device_name,
                                                 get_device_properties=lambda i: props))
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch
    try:
        exec(MK.GUARD, {})
    finally:
        sys.modules["torch"] = saved
    return True


def test_guard_cell_only_in_l1r_notebook():
    for n in MK.RUNS:
        nb = MK.make(n, MK.RUNS[n])
        has = any("GPU guard passed" in src(c) for c in nb["cells"])
        assert has == (n == "l1r_L_ts_m2r"), n
        if has:
            order = [src(c) for c in nb["cells"]]
            i = next(k for k, t in enumerate(order) if "GPU guard passed" in t)
            assert i < next(k for k, t in enumerate(order) if "ALLOW_CODE_CHANGE" in t)     # before the run cell
            assert i < next(k for k, t in enumerate(order) if "urllib.request" in t)        # before any download


def test_guard_accepts_a100_40gb_and_80gb():
    assert run_guard("NVIDIA A100-SXM4-40GB", 42.4e9)
    assert run_guard("NVIDIA A100-SXM4-80GB", 85.0e9)


@pytest.mark.parametrize("name,mem,cuda", [("NVIDIA L4", 24e9, True), ("NVIDIA H100 80GB HBM3", 85e9, True),
                                           ("Tesla T4", 15e9, True), ("NVIDIA A100 slice", 20e9, True), ("none", 0, False)])
def test_guard_refuses_other_gpus(name, mem, cuda):
    with pytest.raises(SystemExit, match="A100"):
        run_guard(name, mem, cuda)


def run_cell(spec, rc_by_script, mem=42e9, tmp=None, seeds=None):
    """Execute the notebook's helpers and run cell with a stub subprocess; returns the commands that would have run."""
    helpers = MK.SETUP[MK.SETUP.index("utc = lambda"):]
    commands = []

    class Sub:
        def run(self, cmd, shell=False, **kw):
            if isinstance(cmd, list):                  # git diff --quiet
                return SimpleNamespace(returncode=0)
            commands.append(cmd)
            return SimpleNamespace(returncode=next((rc for k, rc in rc_by_script.items() if k in cmd), 0))

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "A100",
                                                 get_device_properties=lambda i: SimpleNamespace(total_memory=mem)))
    import os, time
    ns = {"os": os, "json": json, "time": time, "subprocess": Sub(), "torch": torch, "CMDS": spec["cmds"],
          "SEEDS": seeds or spec["seeds"], "R": str(tmp), "D": str(tmp), "GIT_SHA": "a" * 40}
    exec(helpers, ns)
    try:
        exec(MK.RUN, ns)
        return commands, None
    except SystemExit as e:
        return commands, e


def test_l1r_runs_gate_first_then_training(tmp_path):
    cmds, err = run_cell(MK.RUNS["l1r_L_ts_m2r"], {}, tmp=tmp_path)
    assert err is None and len(cmds) == 2
    assert "scripts/w2mg_gate.py --result" in cmds[0] and "w2mg_eval/result.json" in cmds[0]
    assert "scripts/train_gen.py" in cmds[1] and "--seed 0" in cmds[1] and "--micro_batch 64" in cmds[1]
    assert "--micro_batch" not in cmds[0] and "--data_dir" not in cmds[0]


@pytest.mark.parametrize("rc", [2, 3])
def test_l1r_stops_when_gate_fails_or_is_missing(tmp_path, rc):
    cmds, err = run_cell(MK.RUNS["l1r_L_ts_m2r"], {"w2mg_gate.py": rc}, tmp=tmp_path)
    assert isinstance(err, SystemExit)
    assert len(cmds) == 1 and "w2mg_gate.py" in cmds[0]          # training never started
    assert not any("train_gen.py" in c for c in cmds)


def test_w2c1_runs_one_command_per_seed_with_micro_batch(tmp_path):
    cmds, err = run_cell(MK.RUNS["w2c1_L_ts_m2"], {}, tmp=tmp_path)
    assert err is None and len(cmds) == 3
    for s, c in zip((1, 2, 3), cmds):
        assert f"L_ts_m2_frozen_pw_s{s} --seed {s}" in c and "--micro_batch 64" in c
    cmds, _ = run_cell(MK.RUNS["w2c1_L_ts_m2"], {}, tmp=tmp_path / "b", seeds=[2])      # SEEDS edited: one seed per session
    assert len(cmds) == 1 and "--seed 2" in cmds[0]


# ------------------------------------------------------------------------------------------------ AC3: M3-0 config
def run_key(cfg):
    """train_gen._run_key on plain dicts: what identifies a run."""
    t = {k: v for k, v in cfg.get("train", {}).items() if k not in ("eval_every", "ckpt_every", "log_every")}
    return [cfg.get("init"), cfg.get("model"), cfg.get("corruption"), cfg.get("level0"), t]


def test_m3_0_config_differs_from_m3_only_in_clean_row_frac():
    m3 = yaml.safe_load(open(ROOT / "configs/scale/L_ts_m3_uniform_wrong.yaml"))
    c0 = yaml.safe_load(open(ROOT / "configs/scale/L_ts_m3_uniform_wrong_c0.yaml"))
    assert m3["corruption"]["clean_row_frac"] == 0.25 and c0["corruption"]["clean_row_frac"] == 0.0
    c0_as_m3 = json.loads(json.dumps(c0))
    c0_as_m3["corruption"]["clean_row_frac"] = 0.25
    assert c0_as_m3 == m3                                          # every other key (train, sample_eval, arm, rates, ...) equal
    assert run_key(c0) != run_key(m3)                              # a c0 run directory cannot resume an M3 checkpoint
    assert c0["train"]["seed"] == 0 and c0["train"]["steps"] == 600 and c0["corruption"]["arm"] == "m3"
    assert "M3-0" in open(ROOT / "configs/scale/L_ts_m3_uniform_wrong_c0.yaml").read()


# ------------------------------------------------------------------------------------------------ AC4: w1_sample_stats
@pytest.fixture
def stats_env(tmp_path, monkeypatch):
    import ncmr.data as D
    st = load("w1_sample_stats_r056", "scripts/w1_sample_stats.py")
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(2, 8192, 5000).astype(np.uint16).tofile(data / "train_bpe_8192.bin")
    monkeypatch.setattr(st, "examples", lambda split, n: np.array([[5, 6, 7, 8, D.EOS, D.PAD, D.PAD, D.PAD]] * 4))
    return st, tmp_path, data


def write_samples(root, d, prefix, ids):
    (root / d).mkdir(parents=True, exist_ok=True)
    json.dump({"ids": ids}, open(root / d / f"{prefix}__samples_final.json", "w"))


def run_stats(st, monkeypatch, tmp_path, data, setname, capsys):
    out = tmp_path / f"{setname}.json"
    monkeypatch.setattr(sys, "argv", ["w1_sample_stats.py", "--set", setname, "--root", str(tmp_path / "colab"),
                                      "--data_dir", str(data), "--train_tokens", "5000", "--out", str(out)])
    st.main()
    return json.load(open(out)), capsys.readouterr().out


def test_w2m_set_skips_absent_arms_with_a_note(stats_env, monkeypatch, capsys):
    st, tmp_path, data = stats_env
    root = tmp_path / "colab"
    write_samples(root, "w1_L_ts_ctrl", "L_ts_released16", [[5, 5, 6, 7]] * 8)             # A0: 1 adjacent duplicate in 3 pairs
    write_samples(root, "w2m_L_ts_m2", "L_ts_m2_frozen_pw_s0", [[5, 6, 7, 8]] * 8)         # M2: none
    out, text = run_stats(st, monkeypatch, tmp_path, data, "w2m", capsys)
    assert list(out["sets"]) == ["human_val", "A0", "M2"]
    assert out["sets"]["A0"]["adjacent_dup_per_1k"] == pytest.approx(1000 / 3)
    assert out["sets"]["M2"]["adjacent_dup_per_1k"] == 0.0
    assert "M5" in text and "skipped" in text and "M0" in text                              # a note per absent arm


def test_w1_set_stays_strict(stats_env, monkeypatch, capsys):
    st, tmp_path, data = stats_env
    write_samples(tmp_path / "colab", "w1_L_ts_ctrl", "L_ts_released16", [[5, 5, 6, 7]] * 8)
    with pytest.raises(FileNotFoundError):
        run_stats(st, monkeypatch, tmp_path, data, "w1", capsys)


@pytest.mark.parametrize("setname,present,absent", [
    ("w2c1", {"M2_s1": ("w2c1_L_ts_m2", "L_ts_m2_frozen_pw_s1"), "M3_s3": ("w2c1_L_ts_m3", "L_ts_m3_uniform_wrong_s3")},
     ["M2_s2", "M3_s1"]),
    ("w2p", {"M3": ("w2m_L_ts_m3", "L_ts_m3_uniform_wrong_s0"), "M3-0": ("w2p_L_ts_m3_c0", "L_ts_m3_uniform_wrong_c0_s0")}, []),
    ("l1r", {"M2": ("w2m_L_ts_m2", "L_ts_m2_frozen_pw_s0"), "M2R": ("l1r_L_ts_m2r", "L_ts_m2r_frozen_pw_l1reassign_s0")},
     ["M3"])])
def test_new_sets_read_the_notebook_directories(stats_env, monkeypatch, capsys, setname, present, absent):
    st, tmp_path, data = stats_env
    for i, (arm, (d, prefix)) in enumerate(present.items()):
        write_samples(tmp_path / "colab", d, prefix, [[5, 6, 7, 8]] * 4 if i else [[5, 5, 5, 7]] * 4)
    out, text = run_stats(st, monkeypatch, tmp_path, data, setname, capsys)
    assert set(out["sets"]) == {"human_val", *present}
    assert all(f"{a}" in text and "skipped" in text for a in absent)
    first = next(iter(present))
    assert out["sets"][first]["adjacent_dup_per_1k"] == pytest.approx(1000 * 2 / 3)


# ------------------------------------------------------------------------------------------------ AC5: readings
def test_paired_diff_and_pooled_sd():
    m, se = AN.paired_diff([3.0, 5.0, 10.0, 14.0], [1.0, 1.0, 2.0, 2.0])      # differences 2, 4, 8, 12
    assert m == pytest.approx(6.5) and se == pytest.approx(np.std([2, 4, 8, 12], ddof=1) / 2)
    assert AN.pooled_seed_sd([10, 20, 30], [1, 2, 3]) == pytest.approx(math.sqrt((100 + 1) / 2))


def damage(d=(12.0, 12.0, 12.0), d_ts=(1.0, 1.0, 1.0), dup=(1.0, 1.0, 1.0), sp=5.0, mean_ts=1.0, mauve=-0.4):
    return AN.read_damage(list(d), list(d_ts), list(dup), sp, mean_ts, mauve)


def test_damage_replicates_and_each_condition_alone_breaks_it():
    assert damage()["reading"] == "replicates"
    assert damage(d=(10.0, 12.0, 14.0), sp=5.9)["reading"] == "replicates"           # D_s = 10 inclusive; mean 12 > 2 s_p = 11.8
    assert damage(d=(9.999, 12.0, 14.0))["reading"] == "mixed"                       # one seed just under 10
    assert damage(d=(12.0, 12.0, 12.0), sp=6.0)["reading"] == "mixed"                # mean D = 2 s_p exactly: needs >
    assert damage(d=(12.0, 12.0, 12.0), sp=5.999)["reading"] == "replicates"
    assert damage(d_ts=(1.0, 0.0, 1.0))["reading"] == "mixed"                        # TS-33M D_s = 0 is not > 0
    assert damage(dup=(1.0, 1.0, 0.0))["reading"] == "mixed"                         # dDup_s = 0 is not > 0
    assert damage(dup=(1.0, 1.0, -0.1))["reading"] == "mixed"


def test_damage_overturned_thresholds():
    assert damage(d=(5.0, 5.0, 5.0))["reading"] == "mixed"                           # mean D = 5 is not < 5
    assert damage(d=(5.0, 5.0, 4.999))["reading"] == "overturned"
    assert damage(d=(-3.0, 2.0, 4.0))["reading"] == "overturned"
    assert damage(mean_ts=0.0)["reading"] == "replicates"                            # TS mean 0: not opposite
    assert damage(mean_ts=-0.001)["reading"] == "overturned"
    assert damage(mauve=0.0)["reading"] == "replicates"                              # equal MAUVE: not opposite
    assert damage(mauve=1e-6)["reading"] == "overturned"                             # M2 better under MAUVE
    r = damage(d=(12.0, 12.0, 12.0), mauve=0.3)
    assert r["overturned_conditions"]["mauve_reversed"] and r["replicates_conditions"]["every_D_ge_10"]
    assert r["mean_D"] == 12.0


@pytest.mark.parametrize("x,expected", [(0.0, "holds"), (10.0, "holds"), (-10.0, "holds"), (10.001, "not established"),
                                        (-10.001, "not established"), (20.0, "not established"), (20.001, "removed"),
                                        (-25.0, "not established"), (100.0, "removed")])
def test_dissociation_thresholds(x, expected):
    assert AN.read_dissociation(x) == expected


@pytest.mark.parametrize("i,expected", [(3.0, "prediction"), (2.999, "inconclusive"), (1.0, "inconclusive"),
                                        (0.999, "weakened"), (-2.0, "weakened"), (7.0, "prediction")])
def test_m3_0_thresholds(i, expected):
    assert AN.read_m30(i) == expected


def test_rescue_formula():
    assert AN.rescue(40.0, 35.0, 20.0) == 0.25
    assert AN.rescue(40.0, 40.0, 20.0) == 0.0 and AN.rescue(40.0, 20.0, 20.0) == 1.0
    assert AN.rescue(20.0, 10.0, 20.0) is None and AN.rescue(19.0, 10.0, 20.0) is None


def l1r(r_g=0.3, r_t=0.3, ent=3.99, ent_m3=3.99, mv=0.5, mv_m2=0.412, dup=10.0, dup_m2=20.0, cov=0.9):
    return AN.read_l1r(r_g, r_t, ent, ent_m3, mv, mv_m2, dup, dup_m2, cov)


def test_l1r_met_and_boundaries():
    assert l1r()["reading"] == "met"
    assert l1r(r_g=0.25, r_t=0.25)["reading"] == "met"                               # >= 25% inclusive
    assert l1r(r_g=0.2499)["reading"] == "between"
    assert l1r(r_t=0.2499)["reading"] == "between"                                   # both scorers required
    assert l1r(r_t=None)["reading"] == "between"
    assert l1r(mv=0.412)["reading"] == "met" and l1r(mv=0.4119)["reading"] == "between"
    assert l1r(dup=20.0)["reading"] == "met" and l1r(dup=20.001)["reading"] == "between"
    assert l1r(dup=None)["reading"] == "incomplete"                                  # all else met, duplicates not measured
    assert l1r(ent=4.0 - 0.05, ent_m3=4.0)["reading"] == "met"                       # entropy floor inclusive
    assert abs(l1r(ent_m3=3.9918)["entropy_floor"] - 3.9418) < 1e-12                 # the registered ">= 3.94"


def test_l1r_kill_conditions():
    assert l1r(r_g=0.10, r_t=0.5)["reading"] == "killed"                             # <= 0.10 inclusive
    assert l1r(r_g=0.1001, r_t=0.5)["reading"] == "between"
    assert l1r(r_g=-0.2)["reading"] == "killed"
    assert l1r(ent=3.94, ent_m3=4.0)["reading"] == "killed"                          # a gain with entropy below M3's - 0.05
    assert l1r(ent=3.95, ent_m3=4.0, r_g=0.15, r_t=0.0)["reading"] == "between"      # at the floor: not killed
    assert l1r(cov=0.79)["reading"] == "killed" and l1r(cov=0.80)["reading"] == "met"
    assert l1r(cov=None)["reading"] == "met"                                         # a finished run passed the run-time gate
    assert l1r(r_g=None)["reading"] == "not available"


# ------------------------------------------------------------------------------------------------ analysis end to end
def write_result(root, nb, run, g, t, nll=750.0, ent=4.0, mauve=0.5, extra=None):
    d = root / nb
    d.mkdir(parents=True, exist_ok=True)
    r = {"run": run, "steps": 600, "final_eval": {"genppl_gpt2-large": float(np.mean(g)), "genppl_gpt2-large_per_group": list(map(float, g)),
                                                  "genppl_TinyStories-33M": float(np.mean(t)),
                                                  "genppl_TinyStories-33M_per_group": list(map(float, t)),
                                                  "val_nll": nll, "entropy": ent, "mauve": mauve, "dup_adj1_per_1k": 9.0}}
    r.update(extra or {})
    json.dump(r, open(d / f"{run}__result.json", "w"))


def write_dups(stats_dir, setname, dups):
    stats_dir.mkdir(exist_ok=True)
    json.dump({"sets": {k: {"adjacent_dup_per_1k": v} for k, v in dups.items()}}, open(stats_dir / f"{setname}_sample_stats.json", "w"))


K = np.arange(16.0)


def build_w2c(root, stats, gaps=(15.0, 12.0, 18.0), dnll=(1.0, 2.0, 3.0), dup_gap=4.0):
    dups = {}
    for s, gap, dn in zip((1, 2, 3), gaps, dnll):
        write_result(root, "w2c1_L_ts_m2", f"L_ts_m2_frozen_pw_s{s}", 20 + 3 * s + gap + 0.1 * K, 30 + 1.0 * K + gap, nll=750 + dn, mauve=0.4)
        write_result(root, "w2c1_L_ts_m3", f"L_ts_m3_uniform_wrong_s{s}", 20 + 3 * s + 0.0 * K, 30 + 1.0 * K, nll=750.0, mauve=0.8)
        dups.update({f"M2_s{s}": 10.0 + dup_gap, f"M3_s{s}": 10.0})
    write_dups(stats, "w2c1", dups)


def test_w2c_stage1_end_to_end(tmp_path):
    root, stats = tmp_path / "colab", tmp_path / "results"
    build_w2c(root, stats)
    out = AN.analyze_w2c(str(root), str(stats))
    assert out["available"]
    d1 = out["per_seed"][1]
    diff = (20 + 3 + 15 + 0.1 * K) - (20 + 3 + 0.0 * K)
    assert d1["D_gpt2"] == pytest.approx(15.0 + 0.1 * 7.5) and d1["D_gpt2_se_paired"] == pytest.approx(np.std(diff, ddof=1) / 4)
    assert d1["D_ts33m"] == pytest.approx(15.0) and d1["dNLL"] == pytest.approx(1.0) and d1["dDup"] == pytest.approx(4.0)
    g2, g3 = [20 + 3 * s + g + 0.75 for s, g in zip((1, 2, 3), (15.0, 12.0, 18.0))], [20 + 3 * s for s in (1, 2, 3)]
    assert out["pooled_seed_sd"] == pytest.approx(math.sqrt((np.var(g2, ddof=1) + np.var(g3, ddof=1)) / 2))
    assert out["mean_dNLL"] == pytest.approx(2.0) and out["dissociation"] == "holds"
    # TinyStories-33M and duplicates move the same way, but MAUVE says M2 is worse: the damage reading follows the data
    assert out["damage"]["mean_D"] == pytest.approx(np.mean([15.75, 12.75, 18.75]))
    assert out["damage"]["reading"] == "replicates" and out["criterion_c_candidate_claim"] is True
    assert "/" not in out["per_seed"][1]["m2"]["file"][:1] and ":" not in out["per_seed"][1]["m2"]["file"]   # no local path


def test_w2c_stage1_reversal_and_partial_inputs(tmp_path):
    root, stats = tmp_path / "colab", tmp_path / "results"
    build_w2c(root, stats, gaps=(4.0, 2.0, 3.0))                       # mean D = 3.75 < 5
    out = AN.analyze_w2c(str(root), str(stats))
    assert out["damage"]["reading"] == "overturned" and out["criterion_c_candidate_claim"] is False
    # missing a seed: per-seed values reported, no reading
    (root / "w2c1_L_ts_m3" / "L_ts_m3_uniform_wrong_s3__result.json").unlink()
    out = AN.analyze_w2c(str(root), str(stats))
    assert out["available"] is False and out["seeds_present"] == [1, 2] and "damage" not in out and out["missing"]
    # missing the duplicate statistics: no reading either
    build_w2c(root, stats)
    (stats / "w2c1_sample_stats.json").unlink()
    out = AN.analyze_w2c(str(root), str(stats))
    assert out["available"] is False and any("w2c1_sample_stats" in m for m in out["missing"])


def test_m3_0_end_to_end(tmp_path):
    root, stats = tmp_path / "colab", tmp_path / "results"
    write_result(root, "w2m_L_ts_m3", "L_ts_m3_uniform_wrong_s0", 21.69 + 0.1 * K, 30.0 + 0 * K, ent=3.99)
    write_result(root, "w2p_L_ts_m3_c0", "L_ts_m3_uniform_wrong_c0_s0", 18.69 + 0.2 * K - 0.1 * K, 28.0 + 0 * K, ent=4.01)
    out = AN.analyze_m30(str(root), str(stats))
    assert out["available"] and out["I_gpt2"] == pytest.approx(3.0) and out["reading"] == "prediction"
    assert out["I_gpt2_se_paired"] == pytest.approx(0.0, abs=1e-9) and out["I_ts33m"] == pytest.approx(2.0) and out["entropy"]["m3_0"] == 4.01
    write_result(root, "w2p_L_ts_m3_c0", "L_ts_m3_uniform_wrong_c0_s0", 21.0 + 0.1 * K, 30.0 + 0 * K)
    assert AN.analyze_m30(str(root), str(stats))["reading"] == "weakened"           # I = 0.69
    (root / "w2p_L_ts_m3_c0" / "L_ts_m3_uniform_wrong_c0_s0__result.json").unlink()
    assert AN.analyze_m30(str(root), str(stats))["available"] is False


def test_l1r_end_to_end(tmp_path):
    root, stats = tmp_path / "colab", tmp_path / "results"
    write_result(root, "w2m_L_ts_m2", "L_ts_m2_frozen_pw_s0", 40 + 0 * K, 50 + 0 * K, ent=4.02, mauve=0.412)
    write_result(root, "w2m_L_ts_m3", "L_ts_m3_uniform_wrong_s0", 20 + 0 * K, 30 + 0 * K, ent=3.99, mauve=0.865)
    write_result(root, "l1r_L_ts_m2r", "L_ts_m2r_frozen_pw_l1reassign_s0", 34 + 0 * K, 43 + 0 * K, ent=4.0, mauve=0.45,
                 extra={"l1r": {"coverage": 0.91}})
    write_dups(stats, "l1r", {"M2": 21.0, "M2R": 15.0})
    out = AN.analyze_l1r(str(root), str(stats))
    assert out["rescue_gpt2"] == pytest.approx(0.3) and out["rescue_ts33m"] == pytest.approx(0.35)
    assert out["coverage"] == 0.91 and out["reading"] == "met"
    assert AN.analyze_l1r(str(root), str(stats), coverage_override=0.5)["reading"] == "killed"
    write_dups(stats, "l1r", {"M2": 21.0, "M2R": 25.0})
    assert AN.analyze_l1r(str(root), str(stats))["reading"] == "between"
    (stats / "l1r_sample_stats.json").unlink()
    assert AN.analyze_l1r(str(root), str(stats))["reading"] == "incomplete"
    (root / "l1r_L_ts_m2r" / "L_ts_m2r_frozen_pw_l1reassign_s0__result.json").unlink()
    assert AN.analyze_l1r(str(root), str(stats))["available"] is False


def test_script_runs_with_no_inputs(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out" / "w2c_analysis.json"
    monkeypatch.setattr(sys, "argv", ["analyze_w2c.py", "--root", str(tmp_path / "none"), "--stats_dir", str(tmp_path / "none"),
                                      "--out", str(out)])
    AN.main()
    res = json.load(open(out))
    assert set(res) == {"w2c_stage1", "m3_0", "l1r"} and not any(r["available"] for r in res.values())
    assert "not available" in capsys.readouterr().out
