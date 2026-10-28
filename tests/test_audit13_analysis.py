"""Audit 13: reject declared departures from W2M-GR before reading them.

These are CPU schema probes, not new W2M-GR measurements. The numerical
control comes from the hash-pinned exploratory seed-0 result; its metadata
is adapted into the reduced schema solely to exercise result validation.
Pytest loads the repository's existing tests/conftest.py unchanged.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ncmr import w2mg


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "audit13_analyze_w2mgr", ROOT / "scripts/analyze_w2mgr.py"
)
AN = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(AN)

SEED0 = "results/colab/w2mg_eval/w2mg_eval__result.json"
SEED0_SHA256 = "ef78b288a4d702582e858a2cf05753a45f636bd7e7351d70fc44bbf6c8fce393"
REFERENCE_SHA256 = {
    (1, "m2"): "d079dc520e2cbc1b93530c3701aafc7e90b99f3b03ec78ad50a9b3423109c989",
    (1, "m3"): "bd6898e17fc1c3a63363a2ba2bfd97201f296b47da4640b971525a69c644d959",
    (2, "m2"): "a837c0dd61bdee8413d693700cf844b8f85f9b711a9b7367690051eb3b85829e",
    (2, "m3"): "f7761c8391762747f18df6848c7a18c081b45d0ad1043409175830d9fbcca15d",
    (3, "m2"): "33b9c036e6ad57f51469ef5760f840c4d9d03cefcdf8a8672707c600c0fd2a9e",
    (3, "m3"): "98cf9e5d95c24e07cf88a4869b57e3f888ed6371510d7b73ffb02a0dda54f66c",
}


def read_pinned(relative, digest):
    raw = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")  # git's stored bytes on any checkout
    assert hashlib.sha256(raw).hexdigest() == digest, relative
    return json.loads(raw)


def refresh_sanity(result, seed):
    refs, sources = {}, {}
    for arm in ("m2", "m3"):
        relative = (
            f"results/colab/{AN.W2C_NB[arm]}/"
            f"{AN.RUN_DIRS[arm]}_s{seed}__result.json"
        )
        source = read_pinned(relative, REFERENCE_SHA256[seed, arm])
        refs[arm] = source["final_eval"]["genppl_gpt2-large"]
        sources[arm] = relative
    g22 = result["cells"]["m2_m2"]["genppl_gpt2-large"]
    g33 = result["cells"]["m3_m3"]["genppl_gpt2-large"]
    result["ii"]["sanity_vs_w2c_final_eval"] = {
        **AN.sanity(g22["mean"], g22["se"], g33["mean"], g33["se"], refs),
        "source": sources,
    }


def control_result(seed):
    result = read_pinned(SEED0, SEED0_SHA256)
    protocol = result["protocol"]
    protocol.update(run_seed=seed, arms=["m2", "m3"], corrupted_bank=None)
    protocol["runs"] = {
        "released": "released",
        **{a: f"{d}_s{seed}/final.pt" for a, d in AN.RUN_DIRS.items()},
    }
    protocol["ckpt_sha256"] = {
        a: protocol["ckpt_sha256"][a] for a in ("m2", "m3")
    }
    result["cells"] = {k: result["cells"][k] for k in AN.CELL_KEYS}
    result["iii"]["rho"] = {
        a: result["iii"]["rho"][a] for a in ("released", "m2", "m3")
    }
    result.pop("iv", None)
    result["ii"].pop("sanity_vs_w2m", None)
    refresh_sanity(result, seed)
    return result


def retain_four_groups(result, seed):
    """An internally consistent four-group calculation with stale 8-group counts."""
    for cell in result["cells"].values():
        for scorer in AN.SCORERS:
            metric = cell[f"genppl_{scorer}"]
            per_group = np.asarray(metric["per_group"][:4], dtype=float)
            sd = float(per_group.std(ddof=1))
            metric.update(
                per_group=per_group.tolist(),
                log_loss_per_group=np.log(per_group).tolist(),
                mean=float(per_group.mean()),
                sd=sd,
                se=sd / math.sqrt(4),
            )
    for scorer in AN.SCORERS:
        ll = {
            k: cell[f"genppl_{scorer}"]["log_loss_per_group"]
            for k, cell in result["cells"].items()
        }
        result["ii"]["shares"][scorer] = {
            str(c): w2mg.shares(
                ll["m2_m2"], ll["m3_m3"],
                ll[f"m2to3_c{c}"], ll[f"m3to2_c{c}"],
            )
            for c in AN.CUTS
        }
    refresh_sanity(result, seed)


def test_pinned_controls_have_the_registered_reduced_form():
    for seed in AN.SEEDS:
        assert AN.check_result(control_result(seed), seed) == []


@pytest.mark.parametrize(
    "departure", ["clean_bank_test", "crossed_cut", "replay_seeds", "four_groups"]
)
def test_declared_nonregistered_replay_is_rejected_before_any_reading(tmp_path, departure):
    controls = {seed: control_result(seed) for seed in AN.SEEDS}
    for seed, result in controls.items():
        assert AN.check_result(result, seed) == [], "the unmodified control must be accepted"

    result = copy.deepcopy(controls[2])
    if departure == "clean_bank_test":
        result["protocol"]["clean_bank"] = 'examples("test", 512)'
    elif departure == "crossed_cut":
        result["cells"]["m2to3_c2"]["spec"]["cut"] = 4
    elif departure == "replay_seeds":
        result["cells"]["m2to3_c2"]["spec"]["seeds"] = list(range(8, 16))
    else:
        retain_four_groups(result, 2)
    controls[2] = result
    for seed, body in controls.items():
        path = Path(AN.seed_path(str(tmp_path), seed))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body), encoding="utf-8")

    analysis = AN.analyze_w2mgr(str(tmp_path))
    assert analysis["available"] is False
    assert "2" in analysis["rejected"]
    assert "localization" not in analysis
