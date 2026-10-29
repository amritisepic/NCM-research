"""W2M-GR (docs/preregistration.md, "W2M-GR"): `scripts/w2mg_eval.py --arms m2,m3`, the saved-W2C sanity reference for run
seeds >= 1, `scripts/analyze_w2mgr.py` (the registered readings and their thresholds) and the `w2mgr_eval` notebook. CPU only:
the eval-script tests run the real code on the tiny models and the text-dependent Gen-PPL stand-in of tests/test_w2mg.py."""
import importlib.util
import json
import math
import os
import shlex
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_w2mg import S, G, SMALL, make_env, _quiet, world  # noqa: F401  (world: the module-scoped fixture)
from test_r056_build import run_cell

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


AN = load("analyze_w2mgr_t", "scripts/analyze_w2mgr.py")
MK = load("make_notebooks_w2mgr", "notebooks/make_notebooks.py")
REDUCED = ("m2", "m3")
FULL = ("m0", "m2", "m3", "m4")


def fake_gen_ppl(texts, scorer, device=None, batch_size=16):
    h = sum(int(wd[1:]) * (i % 7 + 1) for t in texts for i, wd in enumerate(t.split()))
    return 5.0 + (h % 1000) / 10 + (3.0 if "Tiny" in scorer else 0.0)


@pytest.fixture
def fake_scorers(monkeypatch):
    from ncmr import eval_text
    monkeypatch.setattr(eval_text, "gen_ppl", fake_gen_ppl)


def reduced_env(w):
    """make_env's environment restricted to what `--arms m2,m3` loads: the released model, M2 and M3, no flag."""
    env = make_env(w)
    keep = ("released", "m2", "m3")
    env.nets = {k: v for k, v in env.nets.items() if k in keep}
    env.ckpt_sha256 = {k: v for k, v in env.ckpt_sha256.items() if k in REDUCED}          # as load_env: hashes of the arms' checkpoints
    env.runs = {k: v for k, v in env.runs.items() if k in keep}
    env.flag = None
    return env


def read(path):
    with open(path) as f:
        return json.load(f)


# ================================================================================================ the eval script, reduced arms
@pytest.fixture(scope="module")
def paired(world, tmp_path_factory):
    """The same tiny models run once with the full arm set and once with `--arms m2,m3` (seed 0: W2M's sanity reference)."""
    from ncmr import eval_text
    mp = pytest.MonkeyPatch()
    mp.setattr(eval_text, "gen_ppl", fake_gen_ppl)
    base = tmp_path_factory.mktemp("paired")
    try:
        S.run(make_env(world), str(base / "full"), SMALL, 0, _quiet)
        S.run(reduced_env(world), str(base / "red"), SMALL, 0, _quiet, arms=REDUCED)
    finally:
        mp.undo()
    return SimpleNamespace(full=read(base / "full" / "result.json"), red=read(base / "red" / "result.json"),
                           fdir=base / "full", rdir=base / "red")


def test_reduced_run_gives_the_full_runs_i_ii_iii_and_nothing_else(paired):
    full, red = paired.full, paired.red
    assert red["complete"] is True and full["complete"] is True
    cells = {"m2_m2", "m3_m3", *[f"{p}_c{c}" for c in (2, 4, 6) for p in ("m2to3", "m3to2")]}
    assert set(red["cells"]) == cells and set(full["cells"]) == cells | {"m4_f0", "m4_f0.5", "m4_f1"}
    for k in cells:                                                    # spec, metrics, per-group values: identical
        assert red["cells"][k] == full["cells"][k], k
    assert red["ii"] == full["ii"]                                     # shares, readings, sanity, gate: identical
    assert red["direction1_gate"] == full["direction1_gate"]
    assert red["i"] == full["i"]                                       # the ratios, CIs, per-level J and the reading
    for k in ("n_roots", "root_seed", "unit", "m3_minus_m2", "reading"):
        assert red["iii"][k] == full["iii"][k], k
    assert set(red["iii"]["rho"]) == {"released", "m2", "m3"} and set(full["iii"]["rho"]) == {"released", "m0", "m2", "m3", "m4"}
    for n in ("released", "m2", "m3"):
        assert red["iii"]["rho"][n] == full["iii"]["rho"][n], n
    # a non-vacuous comparison: the readings exist and the three quantities are not constants
    assert set(red["ii"]["shares"]["gpt2-large"]) == {"2", "4", "6"} and red["i"]["reading"] in ("holds", "overturned", "mixed")
    assert len({red["iii"]["rho"][n]["mean"] for n in ("released", "m2", "m3")}) == 3


def test_reduced_run_has_no_iv_no_m4_no_m0_and_no_corrupted_bank(paired):
    red, full = paired.red, paired.full
    assert "iv" not in red and "iv" in full
    assert not any(k.startswith("m4") for k in red["cells"]) and not any("m0" in k for k in red["cells"])
    assert red["protocol"]["arms"] == ["m2", "m3"] and full["protocol"]["arms"] == ["m0", "m2", "m3", "m4"]
    assert set(red["protocol"]["runs"]) == {"released", "m2", "m3"} and set(red["protocol"]["ckpt_sha256"]) == {"m2", "m3"}
    assert red["protocol"]["corrupted_bank"] is None and full["protocol"]["corrupted_bank"]
    files = {p.name for p in (paired.rdir / "cells").iterdir()}
    assert files == ({f"{k}.json" for k in red["cells"]} | {"bank_released.npy", "bank_m2.npy", "bank_m3.npy", "js_clean.npy",
                                                            "js_released.npy", "js_m2.npy", "js_m3.npy", "rho_released.npy",
                                                            "rho_m2.npy", "rho_m3.npy"})
    full_files = {p.name for p in (paired.fdir / "cells").iterdir()}
    assert {"bank_m4f0.npy", "an_iv_diag.json", "rho_m0.npy", "rho_m4.npy"} <= full_files
    for f in files - {f"{k}.json" for k in red["cells"]}:              # banks, JS and rho arrays are the full run's, byte for byte
        assert np.array_equal(np.load(paired.rdir / "cells" / f), np.load(paired.fdir / "cells" / f)), f


def test_reduced_run_never_calls_the_m4_or_corrupted_bank_code(world, fake_scorers, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("W2M-GR must not build the corrupted bank or the (iv) diagnostics")
    monkeypatch.setattr(S, "build_corrupted_bank", boom)
    monkeypatch.setattr(S, "analysis_iv_diag", boom)
    monkeypatch.setattr(S, "analysis_iv", boom)
    res = S.run(reduced_env(world), str(tmp_path), SMALL, 0, _quiet, arms=REDUCED)
    assert res["complete"] is True and "iv" not in res


def test_default_arms_are_the_full_set_and_the_protocol_defaults_are_unchanged(world, fake_scorers, tmp_path):
    assert S.ALL_ARMS == FULL and S.parse_arms(",".join(FULL)) == FULL
    res = S.run(make_env(world), str(tmp_path), SMALL, 0, _quiet)          # no arms argument: W2M-G
    assert res["protocol"]["arms"] == list(FULL) and "iv" in res
    man = read(tmp_path / "manifest.json")
    assert man["arms"] == list(FULL) and man["protocol"] == SMALL.to_json()
    assert G.Protocol().to_json()["f_sweep"] == [0.0, 0.5, 1.0] and "arms" not in G.Protocol().to_json()


def test_env_must_hold_exactly_the_released_model_and_the_arms(world, fake_scorers, tmp_path):
    with pytest.raises(SystemExit, match="holds models"):
        S.run(make_env(world), str(tmp_path / "a"), SMALL, 0, _quiet, arms=REDUCED)           # M0 / M4 loaded but not requested
    with pytest.raises(SystemExit, match="holds models"):
        S.run(reduced_env(world), str(tmp_path / "b"), SMALL, 0, _quiet)                      # M0 / M4 requested but not loaded
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


# ------------------------------------------------------------------------------------------------ the arm set in the manifest
def test_manifest_records_the_arm_set_and_refuses_another(world, fake_scorers, tmp_path):
    out = str(tmp_path / "m")
    S.run(reduced_env(world), out, SMALL, 0, _quiet, arms=REDUCED)
    assert read(Path(out) / "manifest.json")["arms"] == ["m2", "m3"]
    S.run(reduced_env(world), out, SMALL, 0, _quiet, arms=REDUCED)                           # the same arm set resumes
    path = Path(out) / "manifest.json"
    man = read(path)
    man["arms"] = list(FULL)                                          # ONLY the arm set differs; the checkpoints etc. are the same
    path.write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="arm set"):
        S.run(reduced_env(world), out, SMALL, 0, _quiet, arms=REDUCED)


def test_a_full_run_directory_refuses_the_reduced_arm_set_and_vice_versa(world, fake_scorers, tmp_path):
    full_dir, red_dir = str(tmp_path / "f"), str(tmp_path / "r")
    S.run(reduced_env(world), red_dir, SMALL, 0, _quiet, arms=REDUCED)
    with pytest.raises(SystemExit, match="manifest"):
        S.run(make_env(world), red_dir, SMALL, 0, _quiet)
    S.run(make_env(world), full_dir, SMALL, 0, _quiet)
    with pytest.raises(SystemExit, match="manifest"):
        S.run(reduced_env(world), full_dir, SMALL, 0, _quiet, arms=REDUCED)


@pytest.mark.parametrize("arms", ["m2", "m3", "m0,m2", "m3,m4", "m0,m4", "", "m0", "m4"])
def test_arms_without_m2_or_m3_are_refused(arms):
    with pytest.raises(SystemExit, match="m2 and m3"):
        S.parse_arms(arms)


@pytest.mark.parametrize("arms", ["m2,m3,m4", "m0,m2,m3", "m2,m3,m5", "m2,m2,m3", "m2,m3,released"])
def test_other_arm_sets_are_refused(arms):
    with pytest.raises(SystemExit):
        S.parse_arms(arms)


def test_parse_arms_accepts_the_two_registered_sets_in_any_order():
    assert S.parse_arms("m2,m3") == REDUCED and S.parse_arms("m3, m2") == REDUCED and S.parse_arms(["m3", "m2"]) == REDUCED
    assert S.parse_arms("m4,m3,m2,m0") == FULL


def test_load_env_with_reduced_arms_loads_only_released_m2_m3(tmp_path, monkeypatch, world):
    w = world
    ck = {}
    for a in ("m2", "m3"):
        d = tmp_path / f"{S.ARM_DIRS[a]}_s2"
        d.mkdir()
        (d / "final.pt").write_bytes(f"placeholder {a}".encode())
        ck[d.name] = {"model": w.nets[a].state_dict()}
    # no M0 / M4 directories exist: a full-arm load would exit, the reduced one must not look for them
    from ncmr import ncm_ref_loader
    fake = SimpleNamespace(vqvae=w.tok, net=w.nets["released"], pre_tokenizer=None,
                           ckpt_extra={"inference": {"level0_unigram": w.L0}})
    monkeypatch.setattr(ncm_ref_loader, "load_ncm", lambda *a, **k: fake)
    real_load = torch.load
    monkeypatch.setattr(S.torch, "load", lambda p, **kw: ck[Path(p).parent.name] if Path(p).name == "final.pt" else real_load(p, **kw))
    monkeypatch.setattr(S, "examples", lambda split, n: np.zeros((n, 128), dtype=np.int64))
    monkeypatch.setattr(S, "run_identity", lambda device: {"device": str(device)})
    env = S.load_env(str(tmp_path), 2, "cpu", G.Protocol(n_clean=2), arms=REDUCED)
    assert set(env.nets) == {"released", "m2", "m3"} and env.flag is None
    assert set(env.ckpt_sha256) == {"m2", "m3"} and env.runs["m2"] == "L_ts_m2_frozen_pw_s2/final.pt"
    assert env.runs["m3"] == "L_ts_m3_uniform_wrong_s2/final.pt" and set(env.runs) == {"released", "m2", "m3"}
    with pytest.raises(SystemExit, match="m0"):                                 # the default arm set still wants M0's checkpoint
        S.load_env(str(tmp_path), 2, "cpu", G.Protocol(n_clean=2))
    with pytest.raises(SystemExit, match="m2 and m3"):
        S.load_env(str(tmp_path), 2, "cpu", G.Protocol(n_clean=2), arms=("m2",))


def test_cli_passes_the_arm_set_to_load_env_and_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(S, "pick_device", lambda: "cpu")
    monkeypatch.setattr(S, "load_env", lambda root, seed, device, P=None, arms=None: (seen.__setitem__("load", (root, seed, arms)), "ENV")[1])
    monkeypatch.setattr(S, "run", lambda env, out, P, seed, log=print, arms=None: seen.__setitem__("run", (env, out, seed, arms)))
    monkeypatch.setattr(sys, "argv", ["w2mg_eval.py", "--run_root", "R", "--run_dir", "O", "--seed", "3", "--arms", "m3,m2"])
    S.main()
    assert seen["load"] == ("R", 3, REDUCED) and seen["run"] == ("ENV", "O", 3, REDUCED)
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["w2mg_eval.py", "--run_root", "R", "--run_dir", "O"])
    S.main()
    assert seen["load"] == ("R", 0, FULL) and seen["run"][3] == FULL                          # the default invocation: W2M-G
    monkeypatch.setattr(sys, "argv", ["w2mg_eval.py", "--run_root", "R", "--run_dir", "O", "--arms", "m3,m4"])
    with pytest.raises(SystemExit):
        S.main()


# ------------------------------------------------------------------------------------------------ the sanity reference
def write_w2c(root, seed, g_m2, g_m3):
    for nb, run, g in (("w2c1_L_ts_m2", "L_ts_m2_frozen_pw", g_m2), ("w2c1_L_ts_m3", "L_ts_m3_uniform_wrong", g_m3)):
        d = Path(root) / "results" / "colab" / nb
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{run}_s{seed}__result.json").write_text(json.dumps({"final_eval": {"genppl_gpt2-large": g, "val_nll": 750.0}}))


def test_sanity_reference_for_seed_ge_1_is_w2cs_final_eval(world, fake_scorers, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ROOT", str(tmp_path / "repo"))
    write_w2c(tmp_path / "repo", 2, 12.25, 7.75)
    write_w2c(tmp_path / "repo", 1, 99.0, 98.0)                                        # another seed's values must not be used
    res = S.run(reduced_env(world), str(tmp_path / "out"), SMALL, 2, _quiet, arms=REDUCED)
    san = res["ii"]["sanity_vs_w2c_final_eval"]
    assert "sanity_vs_w2m" not in res["ii"]
    assert san["ref"] == {"m2": 12.25, "m3": 7.75}
    assert san["source"] == {"m2": "results/colab/w2c1_L_ts_m2/L_ts_m2_frozen_pw_s2__result.json",
                             "m3": "results/colab/w2c1_L_ts_m3/L_ts_m3_uniform_wrong_s2__result.json"}
    c = res["cells"]
    g22, se22 = c["m2_m2"]["genppl_gpt2-large"]["mean"], c["m2_m2"]["genppl_gpt2-large"]["se"]
    g33, se33 = c["m3_m3"]["genppl_gpt2-large"]["mean"], c["m3_m3"]["genppl_gpt2-large"]["se"]
    assert san["G_m2m2"] == g22 and san["G_m3m3"] == g33
    assert san["z_m2m2"] == pytest.approx((g22 - 12.25) / se22) and san["z_m3m3"] == pytest.approx((g33 - 7.75) / se33)
    assert san["within_3se"] is bool(abs(g22 - 12.25) <= 3 * se22 and abs(g33 - 7.75) <= 3 * se33)
    assert read(tmp_path / "out" / "result.json")["ii"]["sanity_vs_w2c_final_eval"] == san


def test_sanity_reference_follows_the_files_and_a_miss_is_reported_not_fatal(world, fake_scorers, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ROOT", str(tmp_path / "repo"))
    write_w2c(tmp_path / "repo", 3, 1000.0, 1000.0)                                    # far from anything the tiny models give
    res = S.run(reduced_env(world), str(tmp_path / "out"), SMALL, 3, _quiet, arms=REDUCED)
    san = res["ii"]["sanity_vs_w2c_final_eval"]
    assert san["within_3se"] is False and san["ref"] == {"m2": 1000.0, "m3": 1000.0}
    assert res["complete"] is True and "shares" in res["ii"]                          # the shares are still computed


def test_seed_zero_keeps_w2ms_reference(paired):
    san = paired.red["ii"]["sanity_vs_w2m"]
    assert san["ref"] == {"m2": 42.62, "m3": 21.69} and "source" not in san and "sanity_vs_w2c_final_eval" not in paired.red["ii"]


@pytest.mark.parametrize("missing", ["m2", "m3", "both"])
def test_a_missing_w2c_result_exits_before_anything_is_written(world, fake_scorers, tmp_path, monkeypatch, missing):
    monkeypatch.setattr(S, "ROOT", str(tmp_path / "repo"))
    write_w2c(tmp_path / "repo", 1, 30.0, 20.0)
    d = tmp_path / "repo" / "results" / "colab"
    if missing in ("m2", "both"):
        (d / "w2c1_L_ts_m2" / "L_ts_m2_frozen_pw_s1__result.json").unlink()
    if missing in ("m3", "both"):
        (d / "w2c1_L_ts_m3" / "L_ts_m3_uniform_wrong_s1__result.json").unlink()
    with pytest.raises(SystemExit, match=r"results/colab/w2c1_L_ts_m[23]/L_ts_m[23]_.*_s1__result.json"):
        S.run(reduced_env(world), str(tmp_path / "out"), SMALL, 1, _quiet, arms=REDUCED)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("body", [{}, {"final_eval": {}}, {"final_eval": {"genppl_gpt2-large": "x"}},
                                  {"final_eval": {"genppl_gpt2-large": -1.0}}, {"final_eval": {"genppl_gpt2-large": float("nan")}}])
def test_a_w2c_result_without_a_usable_genppl_exits(tmp_path, monkeypatch, body):
    monkeypatch.setattr(S, "ROOT", str(tmp_path))
    write_w2c(tmp_path, 1, 30.0, 20.0)
    (tmp_path / "results" / "colab" / "w2c1_L_ts_m3" / "L_ts_m3_uniform_wrong_s1__result.json").write_text(json.dumps(body))
    with pytest.raises(SystemExit, match="w2c1_L_ts_m3"):
        S.w2c_reference(1)


def test_the_repository_holds_the_w2c_references_for_seeds_1_to_3():
    for s in (1, 2, 3):
        ref, src = S.w2c_reference(s)
        for arm, rel in src.items():
            assert not os.path.isabs(rel) and (ROOT / rel).exists() and ":" not in rel
            assert ref[arm] == read(ROOT / rel)["final_eval"]["genppl_gpt2-large"] > 0
        assert ref["m2"] > ref["m3"]


def test_sanity_ii_parameterised_reference_and_default_unchanged():
    a = G.sanity_ii(42.62 + 3.0, 1.0, 21.69 - 3.0, 1.0)
    assert a["within_3se"] and a["ref"] == {"m2": 42.62, "m3": 21.69} and "source" not in a
    b = G.sanity_ii(50.0 + 3.0, 1.0, 30.0 - 3.0, 1.0, {"m2": 50.0, "m3": 30.0}, {"m2": "a", "m3": "b"})
    assert b["within_3se"] and b["ref"] == {"m2": 50.0, "m3": 30.0} and b["source"] == {"m2": "a", "m3": "b"}
    assert b["z_m2m2"] == pytest.approx(3.0) and b["z_m3m3"] == pytest.approx(-3.0)
    assert not G.sanity_ii(50.0 + 3.01, 1.0, 30.0, 1.0, {"m2": 50.0, "m3": 30.0})["within_3se"]
    assert not G.sanity_ii(50.0, 1.0, 30.0 - 3.01, 1.0, {"m2": 50.0, "m3": 30.0})["within_3se"]
    assert not G.sanity_ii(50.0, 1.0, 30.0, 1.0)["within_3se"]                          # 50 is nowhere near W2M's 42.62


# ================================================================================================ analyze_w2mgr
GOOD = {2: (0.009, 0.868), 4: (0.25, 0.50), 6: (0.511, 0.061)}          # (pi_c, sigma_c); interaction = 1 - pi - sigma
GOOD_RATIOS = {b: {"r": 4.0, "lo": 2.0, "hi": 6.0, "per_level": []} for b in AN.BANKS}


# the specs scripts/w2mg_eval.py writes for the reduced run's cells, as they appear in result.json
EVAL_SPECS = {sp["key"]: json.loads(json.dumps(sp)) for sp in S.cell_specs(G.Protocol(), arms=REDUCED)}


def cells_for(shares_gpt2, shares_ts, delta=0.6):
    """Eight registered cells whose per-group log losses give exactly the wanted shares (group jitter cancels in the means)."""
    cells = {}
    for sc, key in (("gpt2-large", "gpt2-large"), ("TinyStories-33M", "TinyStories-33M")):
        sh = shares_gpt2 if sc == "gpt2-large" else shares_ts
        base = 3.0 if sc == "gpt2-large" else 3.4
        jit = np.array([0.01 * (-1) ** g for g in range(8)])
        ll = {"m3_m3": base + jit, "m2_m2": base + delta + jit}
        for c in AN.CUTS:
            ll[f"m2to3_c{c}"] = base + sh[c][0] * delta + jit
            ll[f"m3to2_c{c}"] = base + sh[c][1] * delta + jit
        for k, v in ll.items():
            pg = np.exp(v)
            m = cells.setdefault(k, {"spec": json.loads(json.dumps(EVAL_SPECS[k])), "n": 2048, "entropy": 4.0, "len_tokens": 100.0, "dup_adj1_per_1k": 10.0,
                                     "dup_adj2_per_1k": 1.0, "n_groups": 8})
            sd = float(np.std(pg, ddof=1))
            m[f"genppl_{key}"] = {"mean": float(pg.mean()), "sd": sd, "se": sd / math.sqrt(8), "per_group": list(map(float, pg)),
                                  "log_loss_per_group": list(map(float, v))}
    return cells


def share_block(sh):
    return {str(c): {"delta": 0.6, "pi": sh[c][0], "pi_se": 0.01, "sigma": sh[c][1], "sigma_se": 0.01,
                     "interaction": 1.0 - sh[c][0] - sh[c][1], "interaction_se": 0.01} for c in AN.CUTS}


def make_result(seed, gpt2=GOOD, ts=GOOD, ratios=GOOD_RATIOS, rho=(0.8, 1.0, 1.2), gpu="NVIDIA L4", shift=0.0):
    cells = cells_for(gpt2, ts)
    g = lambda k: cells[k]["genppl_gpt2-large"]
    ref = {"m2": g("m2_m2")["mean"] + shift * g("m2_m2")["se"], "m3": g("m3_m3")["mean"]}
    san = AN.sanity(g("m2_m2")["mean"], g("m2_m2")["se"], g("m3_m3")["mean"], g("m3_m3")["se"], ref)
    san["source"] = {a: f"results/colab/{AN.W2C_NB[a]}/{AN.RUN_DIRS[a]}_s{seed}__result.json" for a in AN.RUN_DIRS}
    proto = {**json.loads(json.dumps(AN.REGISTERED)), "run_seed": seed, "arms": ["m2", "m3"],
             "runs": {"released": "released", **{a: f"{d}_s{seed}/final.pt" for a, d in AN.RUN_DIRS.items()}},
             "ckpt_sha256": {"m2": "a" * 64, "m3": "b" * 64}, "corrupted_bank": None, "identity": {"gpu": gpu, "device": "cuda"}}
    return {"protocol": proto, "complete": True, "cells": cells,
            "ii": {"shares": {"gpt2-large": share_block(gpt2), "TinyStories-33M": share_block(ts)},
                   "sanity_vs_w2c_final_eval": san},
            "i": {"ratios": json.loads(json.dumps(ratios)), "reading": AN.read_i(ratios)},
            "iii": {"m3_minus_m2": {"mean": rho[1], "se": 0.1, "lo": rho[0], "hi": rho[2]}, "reading": AN.read_iii(rho[0], rho[2])}}


def put(root, seed, r):
    p = Path(AN.seed_path(str(root), seed))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r))
    return p


def put_all(root, **kw):
    for s in AN.SEEDS:
        put(root, s, make_result(s, **kw))


def rec(pi2=0.0, pi4=0.3, pi6=0.5, sigma6=0.1, i4=0.3, i6=0.4):
    return {"pi2": pi2, "pi4": pi4, "pi6": pi6, "sigma6": sigma6, "i4": i4, "i6": i6}


# ------------------------------------------------------------------------------------------------ constants
def test_registered_constants_and_protocol_match_the_registration_and_the_eval_script():
    assert (AN.PI2_MAX, AN.PI6_MIN, AN.SIGMA6_MAX, AN.INTERACTION_MIN) == (0.10, 0.30, 0.20, 0.20)
    assert (AN.OVERTURN_PI2, AN.OVERTURN_SIGMA6, AN.OVERTURN_INTERACTION) == (0.25, 0.50, 0.05)
    assert (AN.I_RATIO_MIN, AN.I_LOWER_MIN, AN.EARLY_PI4, AN.EARLY_PI6, AN.COND_SIGMA6, AN.SANITY_SE) == (3.0, 1.2, 0.5, 0.75, 0.8, 3.0)
    assert AN.SEEDS == (1, 2, 3) and AN.CUTS == (2, 4, 6) and AN.EPS <= 1e-9
    p = G.Protocol().to_json()
    for k, v in AN.REGISTERED.items():
        if k in p:
            assert p[k] == v, k
    assert AN.REGISTERED["precision"] == "fp32, TF32 off, eval mode" and AN.REGISTERED["scorers"] == S.SCORER_IDS
    assert AN.RUN_DIRS == {a: S.ARM_DIRS[a] for a in ("m2", "m3")}
    assert {a: (nb, f"{AN.RUN_DIRS[a]}_s{{s}}__result.json") for a, nb in AN.W2C_NB.items()} == S.W2C_RESULTS
    keys = {sp["key"] for sp in S.cell_specs(G.Protocol(), arms=REDUCED)}
    assert keys == set(AN.CELL_KEYS) and AN.CELL_GROUPS == len(G.Protocol().replay_seeds) and AN.CELL_N == 8 * G.Protocol().group_n
    assert AN.DEFAULT_OUT == str(ROOT / "results" / "w2mgr_analysis.json").replace("/", os.sep)


def test_the_restated_w2mg_rules_equal_w2mgs_own():
    for r in (1.0, 2.99, 3.0, 10.0):
        for lo in (1.0, 1.2, 1.2000001, 2.0):
            for hi in (1.0, 1.2, 1.2000001, 20.0):
                ratios = {"a": {"r": r, "lo": lo, "hi": hi}, "b": {"r": 5.0, "lo": 2.0, "hi": 9.0}}
                assert AN.read_i(ratios) == G.read_i(ratios)
                ratios = {"a": {"r": r, "lo": lo, "hi": hi}, "b": {"r": 0.5, "lo": 0.1, "hi": 1.2}}
                assert AN.read_i(ratios) == G.read_i(ratios)
    grid = [-0.2, 0.0, 0.2, 0.49, 0.5, 0.74, 0.75, 0.79, 0.8, 1.0]
    for a in grid:
        for b in grid:
            for c in grid:
                assert AN.read_ii(a, b, c) == G.read_ii(a, b, c)
    for lo, hi in [(0.1, 2.0), (-1.0, 1.0), (-2.0, -0.1), (0.0, 1.0), (-1.0, 0.0)]:
        assert AN.read_iii(lo, hi) == G.read_iii(lo, hi)


# ------------------------------------------------------------------------------------------------ localization thresholds
@pytest.mark.parametrize("key,ok,bad", [("pi2", 0.10, 0.10 + 1e-9), ("pi6", 0.30, 0.30 - 1e-9), ("sigma6", 0.20, 0.20 + 1e-9),
                                        ("i4", 0.20, 0.20 - 1e-9), ("i6", 0.20, 0.20 - 1e-9)])
def test_each_condition_a_to_d_holds_at_its_threshold_and_fails_just_beyond(key, ok, bad):
    base = rec()
    assert conditions_all(base)
    assert conditions_all({**base, key: ok}) and not conditions_all({**base, key: bad})
    only = AN.conditions({**base, key: bad})
    assert sum(1 for k, v in only.items() if k != "all" and not v) == 1                    # exactly the one condition


def conditions_all(r):
    return AN.conditions(r)["all"]


def test_conditions_are_named_a_to_d():
    c = AN.conditions(rec(pi2=0.2))
    assert c["a_pi2_le_0.10"] is False and c["b_pi6_ge_0.30"] and c["c_sigma6_le_0.20"] and c["d_interaction_ge_0.20_at_cuts_4_and_6"]
    c = AN.conditions(rec(i4=0.1))
    assert c["d_interaction_ge_0.20_at_cuts_4_and_6"] is False and c["a_pi2_le_0.10"]


def seeds3(**override):
    return [rec(**{k: v[i] if isinstance(v, (list, tuple)) else v for k, v in override.items()}) for i in range(3)]


def test_localization_replicates_when_every_seed_and_the_ts_mean_meet_a_to_d():
    r = AN.read_localization(seeds3(), seeds3())
    assert r["reading"] == "replicates" and r["gpt2_every_seed_a_to_d"] and r["ts33m_seed_mean_conditions"]["all"]
    assert not r["overturn_conditions"]["any"]
    # exactly at every threshold, in every seed
    edge = dict(pi2=0.10, pi6=0.30, sigma6=0.20, i4=0.20, i6=0.20)
    assert AN.read_localization(seeds3(**edge), seeds3(**edge))["reading"] == "replicates"


@pytest.mark.parametrize("key,bad", [("pi2", 0.1001), ("pi6", 0.2999), ("sigma6", 0.2001), ("i4", 0.1999), ("i6", 0.1999)])
@pytest.mark.parametrize("seed_idx", [0, 1, 2])
def test_one_gpt2_seed_failing_one_condition_makes_it_mixed_not_replicates(key, bad, seed_idx):
    g = seeds3()
    g[seed_idx][key] = bad                           # the seed MEAN still meets the condition: the rule is per seed
    r = AN.read_localization(g, seeds3())
    assert r["reading"] == "mixed" and r["gpt2_per_seed_conditions"][seed_idx]["all"] is False


@pytest.mark.parametrize("key,bad", [("pi2", 0.1001), ("pi6", 0.2999), ("sigma6", 0.2001), ("i4", 0.1999), ("i6", 0.1999)])
def test_ts33m_conditions_are_on_the_seed_means(key, bad):
    ts = seeds3(**{key: [bad, bad, bad]})
    assert AN.read_localization(seeds3(), ts)["reading"] == "mixed"
    # one TS seed beyond the threshold but the mean on it (or inside): still replicates
    spread = {"pi2": [0.2, 0.05, 0.05], "pi6": [0.2, 0.4, 0.5], "sigma6": [0.3, 0.1, 0.2], "i4": [0.1, 0.3, 0.2], "i6": [0.1, 0.3, 0.2]}[key]
    ts = seeds3(**{key: spread})
    assert not all(AN.conditions(r)["all"] for r in ts)                                   # a TS seed alone fails ...
    assert AN.read_localization(seeds3(), ts)["reading"] == "replicates"                  # ... the seed mean does not


def test_a_ts_seed_mean_that_lands_on_a_threshold_by_float_noise_counts():
    # the mean of 0.0, 0.1, 0.2 is 0.10000000000000002 in floating point; the registered threshold is 0.10
    assert np.mean([0.0, 0.1, 0.2]) > 0.1
    assert AN.read_localization(seeds3(), seeds3(pi2=[0.0, 0.1, 0.2]))["reading"] == "replicates"
    assert AN.read_localization(seeds3(), seeds3(pi2=[0.0, 0.1, 0.2001]))["reading"] == "mixed"


@pytest.mark.parametrize("override,expected", [
    (dict(pi2=[0.5, 0.25, 0.0]), "overturned"),                       # mean 0.25 exactly: pi_2 >= 0.25
    (dict(pi2=[0.5, 0.25, -0.0001]), "mixed"),                        # mean just under 0.25 (but per-seed (a) fails)
    (dict(sigma6=[0.5, 0.5, 0.5]), "overturned"),                     # sigma_6 >= 0.50
    (dict(sigma6=[0.5, 0.5, 0.4999]), "mixed"),
    (dict(i4=0.0499, i6=0.0499), "overturned"),                       # interaction < 0.05 at both cuts
    (dict(i4=0.04, i6=0.05), "mixed"),                                # 0.05 is not < 0.05
    (dict(i4=0.05, i6=0.04), "mixed"),
    (dict(i4=0.0, i6=0.30), "mixed"),                                 # only one cut additive
    (dict(i4=[0.01, 0.1, 0.03], i6=[0.0, 0.0, 0.14]), "overturned"),  # seed means 0.0467 at both cuts (a seed itself is above 0.05)
])
def test_overturn_conditions_on_the_gpt2_seed_means(override, expected):
    assert AN.read_localization(seeds3(**override), seeds3())["reading"] == expected


def test_overturn_looks_at_gpt2_means_only_and_each_condition_alone_suffices():
    bad_ts = seeds3(pi2=0.5, sigma6=0.9, i4=0.0, i6=0.0)               # TinyStories-33M looks overturned: not an overturn
    assert AN.read_localization(seeds3(), bad_ts)["reading"] == "mixed"
    over = AN.overturn_conditions
    assert over(rec(pi2=0.25))["pi2_ge_0.25_L1_carries_it"] and not over(rec(pi2=0.2499))["pi2_ge_0.25_L1_carries_it"]
    assert over(rec(sigma6=0.5))["sigma6_ge_0.50_late_levels_carry_it"] and not over(rec(sigma6=0.4999))["any"]
    assert over(rec(i4=0.0499, i6=0.0499))["interaction_lt_0.05_at_cuts_4_and_6_additive"]
    assert not over(rec(i4=0.0499, i6=0.05))["any"] and not over(rec(i4=0.05, i6=0.05))["any"]


def test_seed_zeros_pattern_replicates_and_the_means_are_arithmetic_means():
    s0 = rec(pi2=0.009, pi4=0.25, pi6=0.511, sigma6=0.061, i4=0.25, i6=0.469)
    assert AN.read_localization([s0] * 3, [s0] * 3)["reading"] == "replicates"
    m = AN.mean_record([rec(pi2=0.0), rec(pi2=0.3), rec(pi2=0.3)])
    assert m["pi2"] == pytest.approx(0.2)


# ------------------------------------------------------------------------------------------------ (i), (iii), (ii) readings, stats
def test_i_rule_boundaries():
    ok = lambda **o: {"a": {"r": 3.0, "lo": 1.2001, "hi": 9.0, **o}, "b": {"r": 7.0, "lo": 3.0, "hi": 9.0}}
    assert AN.read_i(ok()) == "holds"
    assert AN.read_i(ok(r=2.999)) == "mixed" and AN.read_i(ok(lo=1.2)) == "mixed"        # lower bound must be > 1.2
    assert AN.read_i({k: {"r": 1.0, "lo": 0.9, "hi": 1.2} for k in "abc"}) == "overturned"
    assert AN.read_i({k: {"r": 1.0, "lo": 0.9, "hi": 1.2001} for k in "abc"}) == "mixed"


def test_iii_rule_boundaries():
    assert AN.read_iii(1e-9, 2.0) == "holds" and AN.read_iii(0.0, 2.0) == "neither"
    assert AN.read_iii(-2.0, -1e-9) == "overturned" and AN.read_iii(-2.0, 0.0) == "neither"


def test_replicates_all_needs_every_seed_and_at_least_one():
    assert AN.replicates_all([True, True, True]) and not AN.replicates_all([True, False, True]) and not AN.replicates_all([])


def test_spread_is_mean_and_sample_sd_ddof1():
    s = AN.spread([0.1, 0.4, 0.7])
    assert s["mean"] == pytest.approx(0.4) and s["sd"] == pytest.approx(0.3) and s["per_seed"] == [0.1, 0.4, 0.7]
    assert AN.spread([1.0, 3.0])["sd"] == pytest.approx(math.sqrt(2.0))


# ------------------------------------------------------------------------------------------------ result checks
def test_a_valid_result_passes_the_checks():
    assert AN.check_result(make_result(2), 2) == []


MUTATIONS = {
    "incomplete": lambda r: r.update(complete=False),
    "arms_full": lambda r: r["protocol"].update(arms=["m0", "m2", "m3", "m4"]),
    "arms_m2_only": lambda r: r["protocol"].update(arms=["m2"]),
    "arms_missing": lambda r: r["protocol"].pop("arms"),
    "arms_order": lambda r: r["protocol"].update(arms=["m3", "m2"]),
    "wrong_seed": lambda r: r["protocol"].update(run_seed=3),
    "runs_other_seed": lambda r: r["protocol"]["runs"].update(m2="L_ts_m2_frozen_pw_s1/final.pt"),
    "runs_with_m4": lambda r: r["protocol"]["runs"].update(m4="L_ts_m4_frozen_pw_flags_s2/final.pt"),
    "ckpt_with_m0": lambda r: r["protocol"]["ckpt_sha256"].update(m0="c" * 64),
    "n_clean": lambda r: r["protocol"].update(n_clean=256),
    "group_n": lambda r: r["protocol"].update(group_n=128),
    "bank_seeds": lambda r: r["protocol"].update(bank_seeds=[900, 902]),
    "replay_seeds": lambda r: r["protocol"].update(replay_seeds=list(range(7))),
    "cuts": lambda r: r["protocol"].update(cuts=[2, 4]),
    "js_levels": lambda r: r["protocol"].update(js_levels=[2, 3, 4]),
    "n_boot": lambda r: r["protocol"].update(n_boot=1000),
    "boot_seed": lambda r: r["protocol"].update(boot_seed=1),
    "root_seed": lambda r: r["protocol"].update(root_seed=2026),
    "n_roots": lambda r: r["protocol"].update(n_roots=4),
    "precision": lambda r: r["protocol"].update(precision="bf16"),
    "scorers": lambda r: r["protocol"].update(scorers={"gpt2-large": "gpt2"}),
    "corrupted_bank": lambda r: r["protocol"].update(corrupted_bank="plausible_wrong, seed 2026, W1 rates"),
    "gpu_a100": lambda r: r["protocol"]["identity"].update(gpu="NVIDIA A100-SXM4-40GB"),
    "gpu_l40s": lambda r: r["protocol"]["identity"].update(gpu="NVIDIA L40S"),
    "gpu_none": lambda r: r["protocol"]["identity"].update(gpu=None),
    "extra_cell": lambda r: r["cells"].update(m4_f0=dict(r["cells"]["m2_m2"])),
    "missing_cell": lambda r: r["cells"].pop("m3to2_c6"),
    "cell_groups": lambda r: r["cells"]["m2_m2"].update(n_groups=7),
    "cell_n": lambda r: r["cells"]["m2_m2"].update(n=1024),
    "iv_present": lambda r: r.update(iv={}),
    "no_i": lambda r: r.pop("i"),
    "no_iii": lambda r: r.pop("iii"),
    "no_ii": lambda r: r.pop("ii"),
    "shares_not_cells": lambda r: r["ii"]["shares"]["gpt2-large"]["4"].update(pi=0.9),
    "ts_shares_not_cells": lambda r: r["ii"]["shares"]["TinyStories-33M"]["6"].update(sigma=0.9),
    "share_nan": lambda r: r["ii"]["shares"]["gpt2-large"]["2"].update(pi=float("nan")),
    "ratio_missing": lambda r: r["i"]["ratios"].pop("m3"),
    "sanity_w2m": lambda r: r["ii"].update(sanity_vs_w2m=r["ii"].pop("sanity_vs_w2c_final_eval")),
    "sanity_source": lambda r: r["ii"]["sanity_vs_w2c_final_eval"].update(source={"m2": "x", "m3": "y"}),
    "sanity_flag": lambda r: r["ii"]["sanity_vs_w2c_final_eval"].update(within_3se=False),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_a_result_that_breaks_the_registered_form_is_rejected(name):
    r = make_result(2)
    MUTATIONS[name](r)
    assert AN.check_result(r, 2), name


def test_gpu_token_must_be_an_l4():
    assert AN.check_result(make_result(1, gpu="NVIDIA L4"), 1) == [] and AN.check_result(make_result(1, gpu="L4"), 1) == []
    assert AN.check_result(make_result(1, gpu="Tesla T4"), 1)


# ------------------------------------------------------------------------------------------------ end to end
def test_end_to_end_replicates_with_seed_means_sds_and_seed0_beside(tmp_path):
    seeds = {1: {2: (0.00, 0.80), 4: (0.20, 0.50), 6: (0.45, 0.10)},
             2: {2: (0.05, 0.85), 4: (0.30, 0.45), 6: (0.55, 0.05)},
             3: {2: (0.02, 0.90), 4: (0.25, 0.40), 6: (0.50, 0.15)}}
    ts = {s: {c: (a * 0.9, b * 0.9) for c, (a, b) in sh.items()} for s, sh in seeds.items()}
    for s in AN.SEEDS:
        put(tmp_path, s, make_result(s, gpt2=seeds[s], ts=ts[s]))
    s0 = make_result(1, gpt2={2: (0.9, 0.05), 4: (0.9, 0.05), 6: (0.95, 0.0)})          # a wild seed 0: must not enter anything
    s0["ii"]["sanity_vs_w2m"] = s0["ii"].pop("sanity_vs_w2c_final_eval")
    s0["protocol"]["arms"] = ["m0", "m2", "m3", "m4"]
    p0 = Path(AN.seed0_path(str(tmp_path)))
    p0.parent.mkdir(parents=True)
    p0.write_text(json.dumps(s0))
    out = AN.analyze_w2mgr(str(tmp_path))
    assert out["available"] and out["missing"] == [] and out["rejected"] == {} and out["seeds_present"] == [1, 2, 3]
    loc = out["localization"]
    assert loc["reading"] == "replicates" and out["localization_replicates"] is True
    assert out["i_replicates"] is True and out["iii_replicates"] is True
    for sc, f in (("gpt2-large", 1.0), ("TinyStories-33M", 1.0)):
        for c in (2, 4, 6):
            for q, idx in (("pi", 0), ("sigma", 1)):
                vals = [seeds[s][c][idx] * (1.0 if sc == "gpt2-large" else 0.9) for s in AN.SEEDS]
                st = out["seed_stats"][sc][str(c)][q]
                assert st["mean"] == pytest.approx(np.mean(vals)) and st["sd"] == pytest.approx(np.std(vals, ddof=1))
                assert st["per_seed"] == pytest.approx(vals)
            inter = [1 - seeds[s][c][0] - seeds[s][c][1] for s in AN.SEEDS]
            if sc == "gpt2-large":
                assert out["seed_stats"][sc][str(c)]["interaction"]["sd"] == pytest.approx(np.std(inter, ddof=1))
    assert loc["gpt2_seed_mean"]["pi2"] == pytest.approx(np.mean([0.0, 0.05, 0.02]))
    assert out["ii_reading_per_seed"] == {"1": {"gpt2-large": "distributed", "TinyStories-33M": "distributed"},
                                          "2": {"gpt2-large": "distributed", "TinyStories-33M": "distributed"},
                                          "3": {"gpt2-large": "distributed", "TinyStories-33M": "distributed"}}
    assert out["sanity_within_3se_per_seed"] == {"1": True, "2": True, "3": True}
    b = AN.analyze_seed0(str(tmp_path))
    assert b["available"] and b["pooled"] is False and b["shares"]["gpt2-large"]["2"]["pi"] == pytest.approx(0.9, abs=1e-12)
    assert b["ii_reading"]["gpt2-large"] == "early-state selection"
    assert out["per_seed"]["1"]["file"].endswith("w2mgr_eval_s1__result.json") and "tmp" not in out["per_seed"]["1"]["file"].lower()


def test_end_to_end_overturned_and_mixed(tmp_path):
    l1 = {2: (0.4, 0.4), 4: (0.4, 0.3), 6: (0.6, 0.1)}                    # pi_2 = 0.4: L1 carries it
    put_all(tmp_path, gpt2=l1, ts=l1)
    out = AN.analyze_w2mgr(str(tmp_path))
    assert out["localization"]["reading"] == "overturned" and out["localization_replicates"] is False
    late = {2: (0.0, 0.9), 4: (0.1, 0.9), 6: (0.2, 0.6)}                  # sigma_6 = 0.6: late levels carry it
    put_all(tmp_path, gpt2=late, ts=late)
    assert AN.analyze_w2mgr(str(tmp_path))["localization"]["reading"] == "overturned"
    additive = {2: (0.0, 0.5), 4: (0.4, 0.58), 6: (0.6, 0.38)}            # interaction 0.02 at cuts 4 and 6
    put_all(tmp_path, gpt2=additive, ts=additive)
    assert AN.analyze_w2mgr(str(tmp_path))["localization"]["reading"] == "overturned"
    inside = {2: (0.15, 0.6), 4: (0.25, 0.5), 6: (0.5, 0.1)}              # pi_2 = 0.15: neither replicates nor overturned
    put_all(tmp_path, gpt2=inside, ts=inside)
    out = AN.analyze_w2mgr(str(tmp_path))
    assert out["localization"]["reading"] == "mixed" and out["available"]
    put_all(tmp_path, gpt2=GOOD, ts=inside)                               # GPT-2 large fine, TinyStories-33M means fail (a)
    assert AN.analyze_w2mgr(str(tmp_path))["localization"]["reading"] == "mixed"


def test_end_to_end_i_and_iii_need_every_seed(tmp_path):
    put_all(tmp_path)
    low = {b: dict(v) for b, v in GOOD_RATIOS.items()}
    low["m3"] = {"r": 2.9, "lo": 2.0, "hi": 4.0, "per_level": []}
    put(tmp_path, 2, make_result(2, ratios=low, rho=(-0.1, 0.2, 0.5)))
    out = AN.analyze_w2mgr(str(tmp_path))
    assert out["i_replicates"] is False and out["iii_replicates"] is False and out["localization_replicates"] is True
    assert out["per_seed"]["2"]["i"]["rule"] == "mixed" and out["per_seed"]["2"]["iii"]["reading"] == "neither"
    assert out["per_seed"]["1"]["i"]["rule"] == "holds" and out["per_seed"]["1"]["iii"]["reading"] == "holds"


def test_a_sanity_miss_is_reported_and_the_shares_are_still_read(tmp_path):
    put_all(tmp_path)
    put(tmp_path, 3, make_result(3, shift=5.0))                           # G(M2_3, M2_3) is 5 SE from W2C's final eval
    out = AN.analyze_w2mgr(str(tmp_path))
    assert out["available"] and out["sanity_within_3se_per_seed"] == {"1": True, "2": True, "3": False}
    assert out["per_seed"]["3"]["sanity"]["z_m2m2"] == pytest.approx(-5.0) and out["localization"]["reading"] == "replicates"


def test_missing_inputs_are_reported_without_a_reading_or_a_crash(tmp_path):
    out = AN.analyze_w2mgr(str(tmp_path / "none"))
    assert out["available"] is False and len(out["missing"]) == 3 and out["per_seed"] == {} and "localization" not in out
    assert all(m.endswith(f"w2mgr_eval_s{s}__result.json") for m, s in zip(out["missing"], AN.SEEDS))
    assert AN.analyze_seed0(str(tmp_path / "none"))["available"] is False
    put(tmp_path, 1, make_result(1))
    put(tmp_path, 2, make_result(2))
    out = AN.analyze_w2mgr(str(tmp_path))
    assert out["available"] is False and out["seeds_present"] == [1, 2] and len(out["missing"]) == 1
    assert out["missing"][0].endswith("w2mgr_eval_s3__result.json")
    assert "localization" not in out and "i_replicates" not in out and set(out["per_seed"]) == {"1", "2"}


@pytest.mark.parametrize("name", ["incomplete", "arms_full", "wrong_seed", "iv_present", "gpu_a100"])
def test_a_rejected_seed_blocks_the_reading_and_says_why(tmp_path, name):
    put_all(tmp_path)
    r = make_result(2)
    MUTATIONS[name](r)
    put(tmp_path, 2, r)
    out = AN.analyze_w2mgr(str(tmp_path))
    assert out["available"] is False and "localization" not in out and set(out["rejected"]) == {"2"} and out["rejected"]["2"]
    assert set(out["per_seed"]) == {"1", "3"} and out["missing"] == []


def test_an_unreadable_result_is_rejected_not_a_crash(tmp_path):
    put_all(tmp_path)
    Path(AN.seed_path(str(tmp_path), 1)).write_text("{not json")
    out = AN.analyze_w2mgr(str(tmp_path))
    assert out["available"] is False and "unreadable" in out["rejected"]["1"][0]
    Path(AN.seed_path(str(tmp_path), 1)).write_text("[]")
    assert AN.analyze_w2mgr(str(tmp_path))["rejected"]["1"] == ["not a JSON object"]


def test_summarise_seed_agrees_with_the_stored_readings_of_the_real_seed0_result():
    path = ROOT / "results" / "colab" / "w2mg_eval" / "w2mg_eval__result.json"
    r = read(path)
    out = AN.analyze_seed0(str(ROOT / "results" / "colab"))
    assert out["available"] and out["pooled"] is False and out["complete"] is True and out["file"] == path.relative_to(ROOT).as_posix()
    assert out["ii_reading"] == r["ii"]["reading"] and out["i"]["rule"] == r["i"]["reading"] == out["i"]["stored_reading"]
    assert out["iii"]["reading"] == r["iii"]["reading"] and out["sanity"] == r["ii"]["sanity_vs_w2m"]
    sh = r["ii"]["shares"]["gpt2-large"]
    assert AN.compact(sh)["pi2"] == sh["2"]["pi"] == pytest.approx(0.009, abs=5e-4) and AN.compact(sh)["sigma6"] == pytest.approx(0.061, abs=5e-4)
    assert out["conditions_gpt2"]["a_pi2_le_0.10"] and out["conditions_gpt2"]["all"] is True        # the pattern the thresholds were set from
    # the real seed-0 shares also satisfy the stored-vs-recomputed check, so the check is not stricter than the real data
    for sc in AN.SCORERS:
        for c in AN.CUTS:
            ll = {k: r["cells"][k][f"genppl_{sc}"]["log_loss_per_group"] for k in AN.CELL_KEYS}
            got = AN.point_shares(ll["m2_m2"], ll["m3_m3"], ll[f"m2to3_c{c}"], ll[f"m3to2_c{c}"])
            assert got == pytest.approx([r["ii"]["shares"][sc][str(c)][q] for q in ("pi", "sigma", "interaction")], abs=1e-9)


def test_script_runs_with_no_inputs_and_writes_the_json(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out" / "w2mgr_analysis.json"
    monkeypatch.setattr(sys, "argv", ["analyze_w2mgr.py", "--root", str(tmp_path / "none"), "--out", str(out)])
    AN.main()
    res = json.load(open(out))
    assert set(res) == {"w2mgr", "seed0_beside"} and res["w2mgr"]["available"] is False and res["seed0_beside"]["available"] is False
    text = capsys.readouterr().out
    assert "not available" in text and "w2mgr_eval_s1__result.json" in text and str(tmp_path) not in text
    assert str(tmp_path) not in out.read_text()                              # no local path in the published JSON


def test_script_end_to_end_prints_the_readings(tmp_path, monkeypatch, capsys):
    put_all(tmp_path)
    out = tmp_path / "o" / "a.json"
    monkeypatch.setattr(sys, "argv", ["analyze_w2mgr.py", "--root", str(tmp_path), "--out", str(out)])
    AN.main()
    res = json.load(open(out))
    assert res["w2mgr"]["localization"]["reading"] == "replicates" and res["w2mgr"]["i_replicates"] is True
    text = capsys.readouterr().out
    assert "localization replicates" in text and "gpt2-large cut 2" in text and "TinyStories-33M cut 6" in text
    assert str(tmp_path) not in out.read_text()


# ================================================================================================ the notebook
def src(c):
    return "".join(c["source"])


def run_gpu_guard(nb, name, cuda=True):
    cell = next(src(c) for c in nb["cells"] if "GPU requirement passed" in src(c))
    torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda i: name))
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch_stub
    try:
        exec(cell, {})
    finally:
        sys.modules["torch"] = saved


def test_notebook_command_seeds_gpu_and_cost():
    spec = MK.RUNS["w2mgr_eval"]
    assert spec["cmds"] == ["python scripts/w2mg_eval.py --run_root {R} --run_dir {R}/w2mgr_eval_s{S} --seed {S} --arms m2,m3"]
    assert spec["seeds"] == [1, 2, 3] and spec["gpu"].startswith("L4") and spec["require_gpu"] == "L4"
    assert spec["units"] == 2.5 and spec["cap"] == "40 L4-minutes per seed" and set(spec["needs"]) == {"ts", "ckpt"}
    for word in ("W2M-GR", "eval-only", "same Drive", "w2c1", "fp32", "TF32 off", "20-25 L4-minutes per seed", "cap 40 L4-minutes per seed"):
        assert word in spec["desc"], word
    nb = json.load(open(ROOT / "notebooks" / "w2mgr_eval.ipynb"))
    assert nb == json.loads(json.dumps(MK.make("w2mgr_eval", spec)))                     # the committed notebook is the generated one
    assert "SEEDS = [1, 2, 3]" in src(nb["cells"][1]) and "--arms m2,m3" in src(nb["cells"][1])
    assert "cap: 40 L4-minutes per seed" in src(nb["cells"][0]) and "2.5 compute units" in src(nb["cells"][0])


def test_notebook_command_parses_in_the_eval_script(monkeypatch):
    seen = {}
    monkeypatch.setattr(S, "pick_device", lambda: "cpu")
    monkeypatch.setattr(S, "load_env", lambda root, seed, device, P=None, arms=None: (seen.update(load=(root, seed, arms)), "ENV")[1])
    monkeypatch.setattr(S, "run", lambda env, out, P, seed, log=print, arms=None: seen.update(run=(out, seed, arms)))
    cmd = MK.RUNS["w2mgr_eval"]["cmds"][0].format(R="/drive/runs", D="/d", S=2)
    monkeypatch.setattr(sys, "argv", shlex.split(cmd)[1:])
    S.main()
    assert seen["load"] == ("/drive/runs", 2, REDUCED) and seen["run"] == ("/drive/runs/w2mgr_eval_s2", 2, REDUCED)


def test_notebook_runs_one_command_per_seed_and_copies_results_under_the_analysis_names(tmp_path):
    cmds, err = run_cell(MK.RUNS["w2mgr_eval"], {}, tmp=tmp_path)
    assert err is None and len(cmds) == 3
    for s, c in zip((1, 2, 3), cmds):
        assert c.endswith(f"/w2mgr_eval_s{s} --seed {s} --arms m2,m3") and c.startswith("python scripts/w2mg_eval.py --run_root ")
        assert "--micro_batch" not in c and "--data_dir" not in c
    cmds, _ = run_cell(MK.RUNS["w2mgr_eval"], {}, tmp=tmp_path / "b", seeds=[3])
    assert len(cmds) == 1 and "--seed 3" in cmds[0]
    # COLLECT copies <run_dir>/result.json to <out>/<basename>__result.json: the file names scripts/analyze_w2mgr.py reads
    rd = MK.RUNS["w2mgr_eval"]["cmds"][0].split("--run_dir ")[1].split()[0].format(R="R", S=2)
    assert f"{os.path.basename(rd)}__result.json" == os.path.basename(AN.seed_path("x", 2)) == "w2mgr_eval_s2__result.json"
    assert "__NAME__" in MK.COLLECT and "{out}/{name}__{f}" in MK.COLLECT
    assert os.path.basename(os.path.dirname(AN.seed_path("x", 2))) == "w2mgr_eval"


@pytest.mark.parametrize("name,ok", [("NVIDIA L4", True), ("NVIDIA A100-SXM4-40GB", False), ("NVIDIA L40S", False), ("NVIDIA L40", False),
                                     ("Tesla T4", False), ("NVIDIA H100 80GB HBM3", False), ("NVIDIA RTX 6000 Ada Generation", False)])
def test_notebook_refuses_any_gpu_but_an_l4(name, ok):
    nb = MK.make("w2mgr_eval", MK.RUNS["w2mgr_eval"])
    if ok:
        run_gpu_guard(nb, name)
    else:
        with pytest.raises(SystemExit, match="L4"):
            run_gpu_guard(nb, name)


def test_notebook_refuses_when_there_is_no_gpu():
    with pytest.raises(SystemExit, match="L4"):
        run_gpu_guard(MK.make("w2mgr_eval", MK.RUNS["w2mgr_eval"]), "none", cuda=False)


def test_the_gpu_guard_runs_before_any_download_and_before_the_run_cell():
    order = [src(c) for c in MK.make("w2mgr_eval", MK.RUNS["w2mgr_eval"])["cells"]]
    i = next(k for k, t in enumerate(order) if "GPU requirement passed" in t)
    assert i < next(k for k, t in enumerate(order) if "urllib.request" in t) and i < next(k for k, t in enumerate(order) if "ALLOW_CODE_CHANGE" in t)
    assert order[-1] == MK.RELEASE.strip("\n")


def test_no_other_notebook_has_the_new_guard_and_the_a100_guard_is_unchanged():
    for n, spec in MK.RUNS.items():
        cells = [src(c) for c in MK.make(n, spec)["cells"]]
        assert any("GPU requirement passed" in t for t in cells) == (n in ("w2mgr_eval", "w2mgx_eval")), n
        assert any("GPU guard passed" in t for t in cells) == bool(spec.get("require_a100")), n
        if n not in ("w2mgr_eval", "w2mgx_eval"):
            assert "require_gpu" not in spec
    assert MK.RUNS["w2mg_eval"]["cmds"] == ["python scripts/w2mg_eval.py --run_root {R} --run_dir {R}/w2mg_eval --seed 0"]   # no --arms: W2M-G
    assert MK.RUNS["w2mg_eval"]["seeds"] == [0]


# ------------------------------------------------------------------------------------------------ audit 13 (A13-1)
def test_admission_fields_equal_what_the_eval_script_writes():
    """A13-1's fix compares the protocol block's descriptions and each cell's spec: they must equal what a registered
    reduced run of scripts/w2mg_eval.py writes, or every real result would be rejected."""
    from types import SimpleNamespace
    env = SimpleNamespace(runs={}, ckpt_sha256={}, identity={})
    pb = json.loads(json.dumps(S.protocol_block(env, G.Protocol(), 1, REDUCED)))
    for k in ("clean_bank", "cells", "sampler", "ci", "precision", "scorers"):
        assert pb[k] == AN.REGISTERED[k], k
    assert pb["corrupted_bank"] is None
    assert set(EVAL_SPECS) == set(AN.CELL_KEYS)
    for k in AN.CELL_KEYS:
        assert AN.expected_cell_spec(k) == EVAL_SPECS[k], k


@pytest.mark.parametrize("name", ["spec_cut", "spec_seeds", "spec_group_n", "spec_prefix", "seven_groups", "sd_mismatch",
                                  "log_loss_mismatch", "nonpositive", "clean_bank", "sampler"])
def test_admission_rejects_declared_departures(name):
    r = make_result(2)
    assert AN.check_result(r, 2) == []
    c = r["cells"]["m2to3_c4"]
    m = c["genppl_gpt2-large"]
    if name == "spec_cut":
        c["spec"]["cut"] = 6
    elif name == "spec_seeds":
        c["spec"]["seeds"] = list(range(8, 16))
    elif name == "spec_group_n":
        c["spec"]["group_n"] = 128
    elif name == "spec_prefix":
        c["spec"]["prefix"] = "m3"
    elif name == "seven_groups":
        m["per_group"], m["log_loss_per_group"] = m["per_group"][:7], m["log_loss_per_group"][:7]
    elif name == "sd_mismatch":
        m["sd"] *= 1.01
    elif name == "log_loss_mismatch":
        m["log_loss_per_group"][0] += 1e-6
    elif name == "nonpositive":
        m["per_group"][0] = 0.0
    elif name == "clean_bank":
        r["protocol"]["clean_bank"] = 'examples("test", 512)'
    elif name == "sampler":
        r["protocol"]["sampler"] = "ancestral, T=0.9"
    assert AN.check_result(r, 2), name


def test_readings_use_shares_recomputed_from_the_cells():
    """Stored share points within admission's 1e-9 tolerance cannot move a reading: the points are recomputed."""
    r = make_result(1)
    sh = r["ii"]["shares"]["gpt2-large"]["2"]
    true_pi = sh["pi"]
    sh["pi"] = true_pi + 5e-10
    assert AN.check_result(r, 1) == []
    out = AN.summarise_seed(r)
    assert out["shares"]["gpt2-large"]["2"]["pi"] == pytest.approx(true_pi, abs=1e-14)


def test_admission_layers_that_other_checks_do_not_cover():
    """Departures invisible to the share and summary-statistic checks: (a) every log loss of one scorer shifted by one
    constant (shares unchanged), caught only by log loss = log Gen-PPL; (b) every cell cut to 7 groups with consistent
    statistics (shares unchanged, the jitter cancels across cells), caught only by the group count."""
    r = make_result(2)
    for c in r["cells"].values():
        c["genppl_TinyStories-33M"]["log_loss_per_group"] = [v + 0.5 for v in c["genppl_TinyStories-33M"]["log_loss_per_group"]]
    assert AN.check_result(r, 2) and all("log losses differ" in w for w in AN.check_result(r, 2))
    r = make_result(2)
    for c in r["cells"].values():
        for sc in AN.SCORERS:
            m = c["genppl_" + sc]
            m["per_group"], m["log_loss_per_group"] = m["per_group"][:7], m["log_loss_per_group"][:7]
            sd = float(np.std(m["per_group"], ddof=1))
            m["mean"], m["sd"], m["se"] = float(np.mean(m["per_group"])), sd, sd / math.sqrt(8)
    why = AN.check_result(r, 2)
    assert why and all("finite group" in w or "se differs" in w for w in why)

