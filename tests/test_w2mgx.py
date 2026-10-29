"""Registered W2M-GX / W2M-GXa on CPU: boundaries, nontrivial paired ratios, admission and replay equivalence."""
import copy
import json
import math
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_w2mg import G, S, world, member
from test_w2mgr import load, make_result, put, run_gpu_guard

ROOT = Path(__file__).resolve().parents[1]
A = load("analyze_w2mgx", "scripts/analyze_w2mgx.py")
MK = load("make_notebooks_gx", "notebooks/make_notebooks.py")


def metric(ll):
    ll = np.asarray(ll, dtype=float)
    pg = np.exp(ll)
    sd = float(pg.std(ddof=1))
    return {"mean": float(pg.mean()), "sd": sd, "se": sd / math.sqrt(len(pg)),
            "per_group": pg.tolist(), "log_loss_per_group": ll.tolist()}


def synthetic():
    refs = {s: make_result(s) for s in G.GX_SEEDS}
    env = SimpleNamespace(runs=A.expected_runs(), ckpt_sha256={f"{a}_s{s}": ("a" if a == "m2" else "b") * 64
                          for s in G.GX_SEEDS for a in ("m2", "m3")},
                          identity={"gpu": "NVIDIA L4", "device": "cuda",
                                    "w2mgr_reference_sha256": {str(s): S.sha256_json(refs[s]) for s in G.GX_SEEDS}})
    cells = {}
    for spec in S.cell_specs(G.GXProtocol(), arms=G.GX_ARMS):
        cells[spec["key"]] = {"spec": copy.deepcopy(spec), "n": 2048, "n_groups": 8, "group_seeds": list(range(8))}
    for sc in G.SCORERS:
        pure = lambda a, s: np.asarray(refs[s]["cells"][f"{a}_{a}"][f"genppl_{sc}"]["log_loss_per_group"])
        for key, cell in cells.items():
            spec = cell["spec"]
            if spec["kind"] == "pure":
                a, s = spec["assignment"][0].split("_s")
                ll = pure(a, int(s))
            elif spec["kind"] == "cross_seed":
                a, s = spec["assignment"][0].split("_s")
                b, t = spec["assignment"][-1].split("_s")
                s, t = int(s), int(t)
                if a == b:
                    ll = (pure(a, s) + pure(a, t)) / 2 + 0.06  # kappa = .10, delta = .6
                else:
                    ll = pure("m3", t) + 0.6 * (0.4 if a == "m2" else 0.1)
            elif spec["kind"] == "single_level":
                direction, seed, level = key.split("_")[1:]
                seed, level = int(seed[1:]), int(level[1:])
                effect = 0.5 if level == 2 else 0.125
                ll = pure("m3", seed) + 0.6 * effect if direction == "up" else pure("m2", seed) - 0.6 * effect
            else:
                seed = int(key.split("_s")[1].split("_")[0])
                ll = pure("m3", seed) + 0.6 * (0.4 if "m2to3" in key else 0.1)
            cell[f"genppl_{sc}"] = metric(ll)
    return {"protocol": S.protocol_block(env, G.GXProtocol(), 0, G.GX_ARMS), "complete": True, "cells": cells}, refs


def write_inputs(root, result, refs):
    for seed, ref in refs.items():
        put(root, seed, ref)
    path = Path(A.result_path(str(root)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")


def test_specs_cover_all_registered_cells_and_explicit_assignments():
    specs = S.cell_specs(G.GXProtocol(), arms=G.GX_ARMS)
    assert len(specs) == len({s["key"] for s in specs}) == 72
    assert {k: sum(s["kind"] == k for s in specs) for k in ("pure", "cross_seed", "single_level", "crossed")} == {
        "pure": 6, "cross_seed": 24, "single_level": 30, "crossed": 12}
    assert all(len(s["assignment"]) == 11 and s["seeds"] == list(range(8)) and s["group_n"] == 256 for s in specs)
    assert S.parse_arms(",".join(reversed(G.GX_ARMS))) == G.GX_ARMS
    for sp in specs:
        if sp["kind"] in ("cross_seed", "crossed"):
            assert sp["assignment"] == [sp["assignment"][0]] * (sp["cut"] - 1) + [sp["assignment"][-1]] * (12 - sp["cut"])
    r, _ = synthetic()
    assert all(r["protocol"][k] == v for k, v in A.REGISTERED.items())
    for seed in G.GX_SEEDS:
        for level in range(1, 6):
            for direction, base, swap in (("up", "m3", "m2"), ("down", "m2", "m3")):
                sp = next(s for s in specs if s["key"] == f"S_{direction}_s{seed}_l{level}")
                assert sp["assignment"] == [f"{swap if j == level else base}_s{seed}" for j in range(1, 12)]


def test_assignment_sampler_is_bitwise_identical_to_old_two_model_cut_path(world):
    m2, m3 = member(world, "m2"), member(world, "m3")
    for cut in (2, 3, 4, 5, 6):
        old_trace, new_trace = [], []
        old = G.sample_codes(G.crossed(m2, m3, cut), world.tok, world.L0, 2, 7, "cpu", trace=old_trace)
        fn = G.assigned([m2] * (cut - 1) + [m3] * (12 - cut))
        new = G.sample_codes(fn, world.tok, world.L0, 2, 7, "cpu", trace=new_trace)
        assert torch.equal(old, new) and all(torch.equal(a, b) for a, b in zip(old_trace, new_trace))
    one = G.sample_codes(G.assigned([m2] * 11), world.tok, world.L0, 2, 7, "cpu")
    assert torch.equal(one, G.sample_codes(G.pure(m2), world.tok, world.L0, 2, 7, "cpu"))


def test_known_ratios_jackknife_and_registered_gr_baselines():
    r, refs = synthetic()
    q = G.gx_quantities(r["cells"], refs)
    for sc in G.SCORERS:
        assert q[sc]["compatibility"]["m2"]["1_2"]["value"] == pytest.approx(0.1)
        assert q[sc]["cross_seed"]["1_2"]["pi"] == pytest.approx(0.4)
        assert q[sc]["cross_seed"]["1_2"]["sigma"] == pytest.approx(0.1)
        assert q[sc]["cross_seed"]["1_2"]["interaction"] == pytest.approx(0.5)
        assert q[sc]["single_level"]["1"]["2"]["e_up"] == pytest.approx(0.5)
        assert q[sc]["single_level"]["1"]["2"]["e_down"] == pytest.approx(0.5)
        assert q[sc]["cuts"]["1"]["3"]["pi"] == pytest.approx(0.4)
    # A new pure-cell perturbation changes kappa, but never GR delta or cross/single/cut baselines.
    for sc in G.SCORERS:
        r["cells"]["pure_m2_s1"][f"genppl_{sc}"] = metric(np.array([3.7] * 8))
    changed = G.gx_quantities(r["cells"], refs)["gpt2-large"]
    assert changed["cross_seed"] == q["gpt2-large"]["cross_seed"]
    assert changed["single_level"] == q["gpt2-large"]["single_level"]
    assert changed["cuts"] == q["gpt2-large"]["cuts"]
    assert changed["compatibility"] != q["gpt2-large"]["compatibility"]
    a, b = np.arange(1, 9), np.arange(2, 10) ** 1.1
    out = G.jackknife_ratio(a, b)
    loo = np.array([np.delete(a, i).mean() / np.delete(b, i).mean() for i in range(8)])
    assert out == pytest.approx({"value": a.mean() / b.mean(), "se": np.sqrt(7 / 8 * ((loo - loo.mean()) ** 2).sum())})


def test_gxa_sigma_uses_reverse_seed_pair_with_same_baselines():
    r, refs = synthetic()
    # Distinct directional cells ensure either superseded candidate formula is exposed.
    sc = "genppl_gpt2-large"
    r["cells"]["X_m3_s2_to_m2_s1_c6"][sc] = metric([3.12] * 8)
    r["cells"]["X_m3_s1_to_m2_s2_c6"][sc] = metric([3.30] * 8)
    q = G.gx_quantities(r["cells"], refs)["gpt2-large"]["cross_seed"]
    assert q["1_2"]["sigma"] == pytest.approx(0.2)
    assert q["2_1"]["sigma"] == pytest.approx(0.5)
    assert q["1_2"]["interaction"] == pytest.approx(0.4)
    # Changing M3_t's baseline changes BOTH shares' denominator to that of the SAME pair.
    refs[2]["cells"]["m3_m3"][sc] = metric([3.1] * 8)
    q = G.gx_quantities(r["cells"], refs)["gpt2-large"]["cross_seed"]
    assert q["1_2"]["sigma"] == pytest.approx((3.12 - 3.1) / (3.6 - 3.1))


def comp(value=0.0):
    return {a: {f"{s}_{t}": {"value": value, "se": 0.0} for s in G.GX_SEEDS for t in G.GX_SEEDS if s != t}
            for a in ("m2", "m3")}


def test_reading_compatibility_minor_boundary_every_arm_pair():
    c = comp(0.1)
    c["m3"]["1_2"]["value"] = -0.1
    assert G.read_gx_compatibility(c) == "minor"
    for arm in ("m2", "m3"):
        for key in c[arm]:
            bad = copy.deepcopy(c)
            bad[arm][key]["value"] = -0.10000001
            assert G.read_gx_compatibility(bad) == "mixed"


def test_reading_compatibility_material_mean_boundary():
    c = comp(0.25)
    c["m3"]["1_2"]["value"] = -10
    assert G.read_gx_compatibility(c) == "material"
    c["m2"]["1_2"]["value"] -= 1e-6
    assert G.read_gx_compatibility(c) == "mixed"


def cross(pi=0.3, interaction=0.2):
    return {f"{s}_{t}": {"pi": pi, "sigma": 1 - pi - interaction, "interaction": interaction}
            for s in G.GX_SEEDS for t in G.GX_SEEDS if s != t}


def test_reading_localization_holds_boundaries_every_pair():
    c = cross()
    assert G.read_gx_localization(c) == "holds"
    for key in c:
        for quantity in ("pi", "interaction"):
            bad = copy.deepcopy(c)
            bad[key][quantity] -= 1e-8
            assert G.read_gx_localization(bad) == "mixed"


def test_reading_localization_fails_strict_mean_boundary():
    assert G.read_gx_localization(cross(pi=0.15)) == "mixed"
    assert G.read_gx_localization(cross(pi=0.15 - 1e-8)) == "fails"


def test_reading_concentration_half_summed_effect_both_directions_every_seed():
    r, refs = synthetic()
    single = G.gx_quantities(r["cells"], refs)["gpt2-large"]["single_level"]
    assert G.read_gx_concentration(single) == [2]
    for seed in G.GX_SEEDS:
        for direction in ("e_up", "e_down"):
            bad = copy.deepcopy(single)
            bad[str(seed)]["2"][direction] -= 1e-7
            assert G.read_gx_concentration(bad) == []


@pytest.mark.parametrize("mutation", ["gpu", "replay", "groups", "scores", "cut", "mapping", "sha", "assignment", "incomplete"])
def test_admission_refuses_protocol_and_mapping_departures(mutation):
    r, _ = synthetic()
    assert A.check_result(r) == []
    cell = r["cells"]["X_m2_s1_to_m3_s2_c6"]
    if mutation == "gpu": r["protocol"]["identity"]["gpu"] = "NVIDIA L40S"
    if mutation == "replay": r["protocol"]["replay_seeds"] = list(range(8, 16))
    if mutation == "groups": cell["group_seeds"] = list(reversed(range(8)))
    if mutation == "scores": cell["genppl_gpt2-large"]["log_loss_per_group"][0] += 0.01
    if mutation == "cut": cell["spec"]["cut"] = 5
    if mutation == "mapping": r["protocol"]["runs"]["m2_s1"] = "L_ts_m2_frozen_pw_s2/final.pt"
    if mutation == "sha": r["protocol"]["ckpt_sha256"]["m2_s1"] = "not-a-hash"
    if mutation == "assignment": cell["spec"]["assignment"][4] = "m3_s2"
    if mutation == "incomplete": r["complete"] = False
    assert A.check_result(r)


def test_analysis_reads_only_admitted_inputs_and_recomputes(tmp_path):
    r, refs = synthetic()
    write_inputs(tmp_path, r, refs)
    r["readings"] = {"compatibility": "fabricated"}
    write_inputs(tmp_path, r, refs)
    got = A.analyze_w2mgx(str(tmp_path))
    assert got["available"] and got["readings"] == {"compatibility": "minor", "cross_seed_localization": "holds", "concentrated_levels": [2]}
    refs[2]["protocol"]["ckpt_sha256"]["m2"] = "f" * 64
    write_inputs(tmp_path, r, refs)
    got = A.analyze_w2mgx(str(tmp_path))
    assert not got["available"] and "readings" not in got and "digest" in ";".join(got["problems"])


def test_reading_sanity_exact_three_se_and_beyond():
    r, refs = synthetic()
    m = r["cells"]["pure_m2_s1"]["genppl_gpt2-large"]
    ref = refs[1]["cells"]["m2_m2"]["genppl_gpt2-large"]["mean"]
    m["mean"], m["se"] = ref + 3, 1.0
    assert A.summarize(r["cells"], refs)["sanity"]["m2_s1"]["within_3se"]
    m["mean"] += 1e-8
    assert not A.summarize(r["cells"], refs)["sanity"]["m2_s1"]["within_3se"]


def test_missing_malformed_and_sanity_miss_have_no_reading(tmp_path):
    assert A.analyze_w2mgx(str(tmp_path))["available"] is False
    r, refs = synthetic()
    for sc in G.SCORERS:
        r["cells"]["pure_m2_s1"][f"genppl_{sc}"] = metric([4.0 + 0.01 * (-1) ** g for g in range(8)])
    write_inputs(tmp_path, r, refs)
    out = A.analyze_w2mgx(str(tmp_path))
    assert not out["available"] and "readings" not in out and out["sanity"]["m2_s1"]["within_3se"] is False
    Path(A.result_path(str(tmp_path))).write_text("[]")
    assert A.analyze_w2mgx(str(tmp_path))["problems"] == ["not a JSON object"]


def test_notebook_generated_command_guard_cap_and_collector(tmp_path):
    spec = MK.RUNS["w2mgx_eval"]
    assert spec["seeds"] == [0] and spec["require_gpu"] == "L4" and spec["cap"].startswith("3.5 L4-hours")
    assert spec["cmds"][0].endswith("--arms " + ",".join(G.GX_ARMS))
    nb = json.loads((ROOT / "notebooks/w2mgx_eval.ipynb").read_text())
    assert nb == MK.make("w2mgx_eval", spec)
    for gpu in ("NVIDIA L40S", "Tesla T4", "A100"):
        with pytest.raises(SystemExit): run_gpu_guard(nb, gpu)
    run_gpu_guard(nb, "NVIDIA L4")
    assert "runtime-release" in spec["desc"] and "one run" in spec["desc"].lower()
    assert "3.5" in spec["cap"] and "same Drive" in spec["desc"]


def test_collector_copies_per_window_js_and_rho_arrays(tmp_path):
    rd = tmp_path / "runs" / "w2mgr_eval_s1"
    (rd / "cells").mkdir(parents=True)
    for name in ("js_clean.npy", "js_m2.npy", "rho_m2.npy", "rho_m3.npy"):
        np.save(rd / "cells" / name, np.arange(4))
    import shutil
    ns = {"DRIVE": str(tmp_path), "R": str(tmp_path / "runs"), "SEEDS": [1],
          "CMDS": ["python eval.py --run_dir {R}/w2mgr_eval_s{S}"], "os": os, "shutil": shutil,
          "history": lambda rd: [], "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
          "utc": lambda: "time", "GIT_SHA": "a" * 40, "json": json, "T_START": 0}
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        exec(MK.COLLECT.replace("__NAME__", "w2mgr_eval"), ns)
    finally:
        os.chdir(old)
    for name in ("js_clean.npy", "js_m2.npy", "rho_m2.npy", "rho_m3.npy"):
        assert np.array_equal(np.load(tmp_path / "results/w2mgr_eval/w2mgr_eval_s1__cells" / name), np.arange(4))


def test_runtime_cap_accumulates_across_resume_and_refuses_exhaustion(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(S.time, "time", lambda: now[0])
    if hasattr(signal, "SIGALRM"):
        monkeypatch.setattr(S.signal, "setitimer", lambda *a: None)
    with S.gx_budget(str(tmp_path), limit_seconds=10): now[0] = 106.0
    with S.gx_budget(str(tmp_path), limit_seconds=10): now[0] = 111.0
    assert json.loads((tmp_path / "runtime_budget.json").read_text())["spent_seconds"] == 11
    with pytest.raises(SystemExit, match="cap reached"):
        with S.gx_budget(str(tmp_path), limit_seconds=10): pass


def test_runtime_cap_charges_unclosed_session_before_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(S.time, "time", lambda: 111.0)
    path = tmp_path / "runtime_budget.json"
    path.write_text(json.dumps({"spent_seconds": 9, "active_since": 100, "limit_seconds": 10}))
    with pytest.raises(SystemExit, match="cap reached"):
        with S.gx_budget(str(tmp_path), limit_seconds=10):
            pytest.fail("an exhausted interrupted session must not resume")
    ledger = json.loads(path.read_text())
    assert ledger == {"spent_seconds": 20, "active_since": None, "limit_seconds": 10}


@pytest.mark.parametrize("returncode,spent,releases", [(1, 12601, True), (1, 12599, False), (0, 12601, False)])
def test_notebook_cap_path_collects_partial_and_releases_before_raising(tmp_path, monkeypatch, returncode, spent, releases):
    import sys
    import types
    import shutil
    rd = tmp_path / "runs" / "w2mgx_eval"
    rd.mkdir(parents=True)
    (rd / "runtime_budget.json").write_text(json.dumps({"spent_seconds": spent, "limit_seconds": 12600}))
    (rd / "result.json").write_text(json.dumps({"complete": False, "cells": {"cached": {}}}))
    released, history = [], []
    colab = types.ModuleType("google.colab")
    saved_path = tmp_path / "results/w2mgx_eval/w2mgx_eval__result.json"
    meta_path = tmp_path / "results/w2mgx_eval/_meta.json"
    def unassign():
        # The executed order matters: a release must never lose partial results or provenance.
        assert json.loads(saved_path.read_text())["complete"] is False
        assert json.loads(meta_path.read_text())["cap_reached"] is True
        assert history[-1]["event"] == "end"
        released.append(True)
    colab.runtime = SimpleNamespace(unassign=unassign)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    ns = {"os": os, "json": json, "shutil": shutil, "time": SimpleNamespace(time=lambda: 12601),
          "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)), "SEEDS": [0],
          "CMDS": MK.RUNS["w2mgx_eval"]["cmds"], "R": str(tmp_path / "runs"), "D": "data",
          "DRIVE": str(tmp_path), "GIT_SHA": "a" * 40, "utc": lambda: "time", "last_step": lambda rd: None,
          "history": lambda rd: history, "provenance": lambda rd, event: history.append(event),
          "subprocess": SimpleNamespace(run=lambda *a, **k: SimpleNamespace(returncode=returncode))}
    if returncode:
        with pytest.raises(SystemExit, match="command failed"):
            exec(MK.GX_RUN, ns)
    else:
        exec(MK.GX_RUN, ns)
    if not releases:
        assert released == [] and not saved_path.exists() and not meta_path.exists()
        return
    assert released == [True]
    saved = json.loads(saved_path.read_text())
    assert saved["complete"] is False and "cap" in saved["stop_reason"] and saved["cells"] == {"cached": {}}
    assert json.loads(meta_path.read_text())["cap_reached"] is True


def test_manifest_refuses_changed_checkpoint_assignment_or_protocol(tmp_path):
    r, _ = synthetic()
    env = SimpleNamespace(ckpt_sha256=r["protocol"]["ckpt_sha256"], identity=r["protocol"]["identity"])
    S.check_manifest(str(tmp_path), env, G.GXProtocol(), 0, G.GX_ARMS)
    env.ckpt_sha256 = {**env.ckpt_sha256, "m2_s1": "f" * 64}
    with pytest.raises(SystemExit, match="manifest"):
        S.check_manifest(str(tmp_path), env, G.GXProtocol(), 0, G.GX_ARMS)


def test_changed_gr_reference_content_is_refused_without_platform_line_ending_dependency(tmp_path):
    r, refs = synthetic()
    write_inputs(tmp_path, r, refs)
    refs[1]["i"]["reading"] = "changed reference"
    write_inputs(tmp_path, r, refs)
    out = A.analyze_w2mgx(str(tmp_path))
    assert not out["available"] and "reference content digest" in ";".join(out["problems"])
    assert S.sha256_json(refs[1]) == S.sha256_json(json.loads(json.dumps(refs[1], indent=1).replace("\n", "\r\n")))


def test_load_env_pins_all_six_checkpoint_digests(tmp_path, monkeypatch, world):
    from ncmr import ncm_ref_loader
    import hashlib
    ck = {}
    hashes = {}
    for s in G.GX_SEEDS:
        for arm in ("m2", "m3"):
            d = tmp_path / f"{S.ARM_DIRS[arm]}_s{s}"
            d.mkdir()
            blob = f"checkpoint {arm} seed {s}".encode()
            (d / "final.pt").write_bytes(blob)
            hashes[f"{arm}_s{s}"] = hashlib.sha256(blob).hexdigest()
            ck[d.name] = {"model": world.nets[arm].state_dict()}
    fake = SimpleNamespace(vqvae=world.tok, net=world.nets["released"], pre_tokenizer=None,
                           ckpt_extra={"inference": {"level0_unigram": world.L0}})
    monkeypatch.setattr(ncm_ref_loader, "load_ncm", lambda *a, **k: fake)
    monkeypatch.setattr(S.torch, "load", lambda p, **kw: ck[Path(p).parent.name])
    monkeypatch.setattr(S, "examples", lambda split, n: np.zeros((n, 128), dtype=np.int64))
    monkeypatch.setattr(S, "run_identity", lambda device: {"device": str(device)})
    env = S.load_env(str(tmp_path), 0, "cpu", G.GXProtocol(n_clean=2), arms=G.GX_ARMS)
    assert env.ckpt_sha256 == hashes and env.runs == A.expected_runs()
    assert set(env.nets) == {"released", *G.GX_ARMS}


def test_resumable_gx_driver_writes_every_cell_once(tmp_path, monkeypatch):
    r, refs = synthetic()
    env = SimpleNamespace(nets={n: None for n in ("released", *G.GX_ARMS)},
                          runs=r["protocol"]["runs"], ckpt_sha256=r["protocol"]["ckpt_sha256"],
                          identity=r["protocol"]["identity"])
    monkeypatch.setattr(A, "load_references", lambda *a: (refs, []))
    calls = []
    def fake_cell(env, P, spec):
        calls.append(spec["key"])
        cell = r["cells"][spec["key"]]
        return {"spec": spec, "metrics": {k: v for k, v in cell.items() if k not in ("spec", "n_groups", "group_seeds")},
                "groups": [s for s in range(8) for _ in range(256)], "sample_ids": []}
    monkeypatch.setattr(S, "run_cell", fake_cell)
    # get_cell logs these two descriptive metrics, unrelated to ratio analysis.
    for cell in r["cells"].values():
        cell.update(entropy=4, dup_adj1_per_1k=10)
    first = S.run(env, str(tmp_path), G.GXProtocol(), 0, lambda *a: None, arms=G.GX_ARMS)
    assert first["complete"] and len(calls) == 72 and A.check_result(first) == []
    calls.clear()
    second = S.run(env, str(tmp_path), G.GXProtocol(), 0, lambda *a: None, arms=G.GX_ARMS)
    assert calls == [] and first == second
