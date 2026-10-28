"""E1b (docs/preregistration.md, "E1b - continued E1 sampling on censored and wide items"): scripts/e1b_continue.py
(replay of E1's generator, stopping, selection, resume / time cap) and scripts/analyze_e1b.py (substitution and the
replication verdict). CPU only. The generator replay is checked on the released TinyStories model (cut 11 and 10, small
budgets: cut 8 costs ~20 s per 64 samples on CPU); the stopping rules on a fake sampler that draws from the generator."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("e1b_continue")
AB = _load("analyze_e1b")
AE = B.AE
E1 = B.E1
REAL = ROOT / "results" / "colab" / "e1_equivalence"
E1A_JSON = ROOT / "results" / "e1_analysis.json"

J0_CENSORED = [11, 38, 49, 64, 70, 71, 86, 95, 103, 110, 111, 115, 117, 118, 119, 121, 122, 123]
J0_WIDE = [2, 18, 19, 24, 25, 27, 29, 30, 34, 36, 37, 41, 42, 47, 48, 52, 54, 58, 74, 76, 77, 83, 85, 87, 88, 89, 91, 93,
           94, 96, 97, 98, 99, 104, 105, 108, 109, 114, 116, 127]
A0_CENSORED = [7, 8, 11, 34, 35, 43]
A0_WIDE = [0, 13, 16, 18, 23, 53, 55, 62]


# ------------------------------------------------------------------------------------------------ fake sampler rig
class FakeDom:
    """A 'sample' is one uniform draw from the generator; it hits iff it is below p (rho = p)."""
    net = None

    def __init__(self, p):
        self.p = p

    def decode_ids(self, codes):
        return codes

    def key(self, row):
        return row[0] < self.p


@pytest.fixture
def fake(monkeypatch):
    calls = []

    def sample(net, cache, canon, k, n, gen, temperature):
        calls.append(n)
        return torch.rand(n, 1, generator=gen)

    monkeypatch.setattr(B.CD, "sample_from_cut", sample)
    monkeypatch.setattr(B.CD, "PrefixCache", lambda *a: None)
    monkeypatch.setattr(B.CD, "prefix_inputs", lambda net, pre: ())
    return calls


CANON = torch.zeros(1, 1, dtype=torch.long)
DELTA = 0.05 / 384
N8 = 300.0                               # large: the exp(-N) floor plays no role


def run_fake(p, seed, e1_hits, e1_n, N=N8, **kw):
    return B.continue_item(FakeDom(p), None, CANON, True, seed, e1_hits, e1_n, N, DELTA, **kw)


def reference_counts(p, seed, sizes):
    """Independent re-draw with the recorded call sizes: [(cumulative hits, cumulative n)] after each call."""
    g = torch.Generator().manual_seed(seed)
    h = n = 0
    out = []
    for m in sizes:
        h += int((torch.rand(m, 1, generator=g)[:, 0] < p).sum())
        n += m
        out.append((h, n))
    return out


# ------------------------------------------------------------------------------------------------ AC1: replay = mc_hits
@pytest.fixture(scope="module")
def ts_model():
    from ncmr.ncm_ref_loader import load_ncm
    bundle = load_ncm(str(ROOT / "artifacts" / "tinystories"), device="cpu")
    net, tok = bundle.net.eval(), bundle.vqvae.eval()
    for p in list(net.parameters()) + list(tok.parameters()):
        p.requires_grad_(False)
    dom = E1.Domain("tinystories", net, tok, {"unigram": bundle.ckpt_extra["inference"]["level0_unigram"].float()})
    items = E1.load_items("tinystories", bundle, 3, "cpu", "")
    return dom, items


def _item(ts_model, i):
    dom, items = ts_model
    pre, x = items[i]
    canon = dom.canonical(x)
    return dom, pre, canon, dom.key(dom.decode_ids(canon)[0].tolist())


def _manual_stream(dom, pre, canon, k, seed, sizes, target):
    """The generator stream drawn by hand: rounds of the given sizes on ONE generator; cumulative (hits, n)."""
    cache = B.CD.PrefixCache(dom.net, *B.CD.prefix_inputs(dom.net, pre))
    g = torch.Generator(device=canon.device).manual_seed(seed)
    h = n = 0
    out = []
    for m in sizes:
        codes = B.CD.sample_from_cut(dom.net, cache, canon, k, m, g, 1.0)
        h += sum(dom.key(r) == target for r in dom.decode_ids(codes).tolist())
        n += m
        out.append((h, n))
    return out


@pytest.mark.parametrize("item,cut,seed", [(0, 11, 11), (2, 11, 2 * 101 + 11)])
def test_replay_equals_mc_hits_and_the_stream_continues(ts_model, item, cut, seed):
    dom, pre, canon, target = _item(ts_model, item)
    n_e1 = 128
    ref = E1.mc_hits(dom, pre, canon, cut, target, seed, n_e1, 64, 10 ** 9)      # E1's loop, no hit stop
    assert ref[1] == n_e1
    res = B.continue_item(dom, pre, canon, target, seed, ref[0], n_e1, 593.0, DELTA, cut=cut, round_=64, cap=n_e1 + 128,
                          max_width=-1.0)                                       # never narrow: runs to the cap
    assert (res["replay_hits"], res["replay_n"]) == ref and res["replay_match"] is True
    # the continuation extends the same stream: rounds 64, 64 (replay), then 64, 64 on the one generator
    manual = _manual_stream(dom, pre, canon, cut, seed, [64, 64, 64, 64], target)
    assert manual[1] == ref
    assert res["final"] == list(manual[3]) and res["stop"] == "cap"
    # discriminating power: a generator re-seeded for the extension would add the first two rounds again, a different count
    reseeded = ref[0] + manual[1][0]
    assert reseeded != res["final"][0]


def test_replay_matches_mc_hits_with_its_own_hit_stop(ts_model):
    """E1 stopped at 20 hits (here after the first round at cut 11); replaying up to that n reproduces its count."""
    dom, pre, canon, target = _item(ts_model, 0)
    e1 = E1.mc_hits(dom, pre, canon, 11, target, 11, 2048, 64, 20)
    assert e1[1] == 64 and e1[0] >= 20
    res = B.continue_item(dom, pre, canon, target, 11, e1[0], e1[1], 593.0, DELTA, cut=11, round_=64, cap=128,
                          max_width=-1.0)
    assert res["replay_match"] is True and (res["replay_hits"], res["replay_n"]) == tuple(e1)
    assert res["final"] == list(_manual_stream(dom, pre, canon, 11, 11, [64, 64], target)[1])


# ------------------------------------------------------------------------------------------------ AC2: stopping
def _width(h, n, N=N8):
    return B.interval_width(h, n, N, DELTA)


@pytest.mark.parametrize("p,seed", [(0.004, 2), (0.006, 1), (0.01, 3)])
def test_rounds_of_512_stop_exactly_when_the_width_rule_is_first_met(fake, p, seed):
    res = run_fake(p, seed, 0, 2048)
    sizes = list(fake)
    assert sizes[:32] == [64] * 32 and set(sizes[32:]) == {512}                    # replay 64s, then 512s
    ref = reference_counts(p, seed, sizes)
    assert ref[-1] == tuple(res["final"]) and ref[31] == (res["replay_hits"], 2048)
    assert res["stop"] == "width" and res["final"][1] < B.CAP
    assert _width(*ref[-1]) <= 2.0
    assert all(_width(*c) > 2.0 for c in ref[31:-1]), "stopped later than the first round that met the rule"


def test_no_hits_run_to_the_cap_and_never_exceed_it(fake):
    res = run_fake(0.0, 1, 0, 2048)
    assert res["stop"] == "cap" and res["final"] == [0, B.CAP] and sum(fake) == B.CAP
    assert max(fake[32:]) == 512 and len(fake[32:]) == (B.CAP - 2048) // 512


def test_last_round_is_shortened_when_the_replay_leaves_a_remainder(fake):
    res = run_fake(0.0, 1, 0, 1984)                      # 32768 - 1984 = 60 x 512 + 64
    assert res["final"][1] == B.CAP and res["stop"] == "cap"
    assert fake[-2:] == [512, 64] and sum(fake) == B.CAP and max(fake) <= 512


def test_width_already_met_after_the_replay_stops_without_extension(fake):
    res = run_fake(0.3, 1, 600, 2048)                    # 0.3 x 2048 hits: an interval far inside 2 nats
    assert res["stop"] == "width" and res["final"][1] == 2048 and sum(fake) == 2048


def test_the_floor_intersection_enters_the_width(fake):
    """N small: the exp(-N) floor caps the interval below, so a 0-hit item stops once hi + N <= 2 (here at once)."""
    res = run_fake(0.0, 1, 0, 2048, N=1.0)
    assert res["stop"] == "width" and res["final"] == [0, 2048]
    assert _width(0, 2048, 1.0) <= 2.0 < _width(0, 2048, 300.0)


# ------------------------------------------------------------------------------------------------ AC3: selection
def test_selection_from_the_saved_e1_files_is_58_j0_and_14_a0_items():
    plan, sel = B.build_plan(str(REAL))
    assert sel["J0"]["censored"] == J0_CENSORED and sel["J0"]["wide"] == J0_WIDE
    assert sel["A0"]["censored"] == A0_CENSORED and sel["A0"]["wide"] == A0_WIDE
    assert len(J0_CENSORED + J0_WIDE) == 58 and len(A0_CENSORED + A0_WIDE) == 14
    # order: J0's items by index, then A0's; each record carries E1's saved counts
    assert [(e["model"], e["item"]) for e in plan] == ([("J0", i) for i in sorted(J0_CENSORED + J0_WIDE)] +
                                                       [("A0", i) for i in sorted(A0_CENSORED + A0_WIDE)])
    saved = json.load(open(REAL / "e1_wmt__result.json", encoding="utf-8"))["rows"]["J0"]
    assert all(e["e1"] == saved[e["item"]]["mc"]["8"] for e in plan if e["model"] == "J0")
    assert all(e["e1"][0] == 0 for e in plan if e["kind"] == "censored")
    assert sel["J0"]["delta"] == pytest.approx(0.05 / (128 * 3)) and sel["A0"]["delta"] == pytest.approx(0.05 / (64 * 4))
    assert B.build_plan(str(REAL), only=("A0",))[0] == [e for e in plan if e["model"] == "A0"]


def test_selection_rule_on_synthetic_rows():
    nll = [1.0] * 8 + [10.0] * 4                          # N_8 = 40
    row = lambda h, n: {"level_nll": nll, "mc": {"8": [h, n]}}
    rows = [row(0, 2048), row(3, 64), row(1000, 2048), row(0, 0)]
    cens, wide = B.select_items(rows, DELTA)
    assert cens == [0] and wide == [1]                    # (0, 0) = x^ invalid: excluded; 1000/2048 is narrow


# ------------------------------------------------------------------------------------------------ AC4: mismatch
def test_a_replay_that_does_not_reproduce_e1_is_flagged_and_read_on_e1b_counts(fake):
    p, seed = 0.006, 1
    true_hits = reference_counts(p, seed, [64] * 32)[-1][0]
    ok = run_fake(p, seed, true_hits, 2048)
    bad = run_fake(p, seed, true_hits + 1, 2048)
    assert ok["replay_match"] is True and bad["replay_match"] is False
    assert bad["replay_hits"] == true_hits                                   # the replayed count is reported
    assert bad["final"] == ok["final"]                                       # read on E1b's counts from sample 1
    n_wrong = run_fake(p, seed, true_hits, 2048 - 64)                        # E1 stopped earlier than the replay did
    assert n_wrong["replay_n"] == 2048 - 64


def test_mismatch_flag_reaches_the_result_file_and_the_analysis(fake, tmp_path):
    plan = [{"domain": "wmt14", "model": "J0", "item": 11, "e1": [0, 2048], "kind": "censored"}]
    sel = {"J0": {"domain": "wmt14", "delta": DELTA, "n_items": 128, "censored": [11], "wide": [], "e1_file": "x"}}

    def runner_for(domain):
        return lambda e: run_fake(0.006, 1, e["e1"][0], e["e1"][1])
    res = B.run(plan, sel, runner_for, str(tmp_path))
    rec = res["items"][0]
    assert rec["replay_match"] is False and rec["e1"] == [0, 2048] and rec["final"][1] > 2048


# ------------------------------------------------------------------------------------------------ AC5: resume, time cap
def _plan(n=5):
    plan = [{"domain": "wmt14", "model": "J0", "item": i, "e1": [0, 2048], "kind": "censored"} for i in range(n)]
    sel = {"J0": {"domain": "wmt14", "delta": DELTA, "n_items": 128, "censored": list(range(n)), "wide": [],
                  "e1_file": "x"}}
    return plan, sel


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _runner(clock, seen, minutes_per_item=1.0):
    def runner_for(domain):
        def one(entry):
            seen.append(entry["item"])
            clock.t += 60.0 * minutes_per_item
            return {"replay_hits": 0, "replay_n": 2048, "replay_match": True, "final": [0, 2048], "stop": "cap"}
        return one
    return runner_for


def test_cumulative_time_cap_stops_between_items_and_resume_skips_finished(tmp_path):
    plan, sel = _plan(5)
    clock, seen = Clock(), []
    res = B.run(plan, sel, _runner(clock, seen), str(tmp_path), max_minutes=2.5, clock=clock)
    assert seen == [0, 1, 2]                              # checked between items: 0, 1, 2 start below 2.5 min
    assert res["not_reached"] == [["J0", 3], ["J0", 4]] and res["complete"] is False
    assert res["minutes"] == pytest.approx(3.0) and len(res["items"]) == 3
    on_disk = json.load(open(tmp_path / "result.json", encoding="utf-8"))
    assert on_disk["not_reached"] == [["J0", 3], ["J0", 4]]
    # same cap, new session: the minutes are cumulative, so nothing more runs
    clock2, seen2 = Clock(), []
    res = B.run(plan, sel, _runner(clock2, seen2), str(tmp_path), max_minutes=2.5, clock=clock2)
    assert seen2 == [] and len(res["items"]) == 3
    # a larger cap resumes at the first unfinished item and skips the finished ones
    res = B.run(plan, sel, _runner(clock2, seen2), str(tmp_path), max_minutes=100, clock=clock2)
    assert seen2 == [3, 4] and [r["item"] for r in res["items"]] == [0, 1, 2, 3, 4]
    assert res["complete"] is True and res["not_reached"] == [] and res["minutes"] == pytest.approx(5.0)


def test_cap_is_checked_before_each_item_not_inside_one(tmp_path):
    plan, sel = _plan(3)
    clock, seen = Clock(), []
    res = B.run(plan, sel, _runner(clock, seen, minutes_per_item=10.0), str(tmp_path), max_minutes=1.0, clock=clock)
    assert seen == [0] and len(res["items"]) == 1          # the running item finishes; the next is not started


def test_result_is_written_after_every_item(tmp_path):
    plan, sel = _plan(3)
    clock, seen = Clock(), []
    inner = _runner(clock, seen)

    def runner_for(domain):
        one = inner(domain)

        def wrapped(entry):
            if entry["item"] == 2:
                on_disk = json.load(open(tmp_path / "result.json", encoding="utf-8"))
                assert [r["item"] for r in on_disk["items"]] == [0, 1]
                raise KeyboardInterrupt
            return one(entry)
        return wrapped
    with pytest.raises(KeyboardInterrupt):
        B.run(plan, sel, runner_for, str(tmp_path), clock=clock)
    seen2 = []
    res = B.run(plan, sel, _runner(clock, seen2), str(tmp_path), clock=clock)
    assert seen2 == [2] and res["complete"]


def test_max_items_limits_a_session(tmp_path):
    plan, sel = _plan(4)
    clock, seen = Clock(), []
    res = B.run(plan, sel, _runner(clock, seen), str(tmp_path), max_items=2, clock=clock)
    assert seen == [0, 1] and res["not_reached"] == [["J0", 2], ["J0", 3]]


def test_a_run_dir_with_another_selection_or_protocol_is_refused(tmp_path):
    plan, sel = _plan(2)
    clock = Clock()
    B.run(plan, sel, _runner(clock, []), str(tmp_path), clock=clock)
    sel2 = copy.deepcopy(sel)
    sel2["J0"]["wide"] = [99]
    with pytest.raises(SystemExit):
        B.run(plan, sel2, _runner(clock, []), str(tmp_path), clock=clock)


def test_result_json_has_no_absolute_paths(tmp_path):
    plan, sel = B.build_plan(str(REAL), only=("A0",))
    clock = Clock()
    B.run(plan[:2], sel, _runner(clock, []), str(tmp_path), clock=clock, extra={"gpu": "cpu", "torch": torch.__version__})
    text = (tmp_path / "result.json").read_text(encoding="utf-8")
    assert str(ROOT) not in text and ":\\" not in text and "/Users" not in text and "\\\\" not in text


def test_end_to_end_item_on_the_released_tinystories_model(tmp_path, monkeypatch):
    """The real runner on CPU at a shortened cut / cap: records the replay flag, final counts and the NLL cross-check."""
    monkeypatch.setattr(B, "CUT", 11)
    monkeypatch.setattr(B, "CAP", 192)
    monkeypatch.setattr(B, "ROUND", 64)
    plan = [{"domain": "tinystories", "model": "A0", "item": 1, "e1": [60, 64], "kind": "wide"}]
    sel = {"A0": {"domain": "tinystories", "delta": 0.05 / 256, "n_items": 64, "censored": [], "wide": [1],
                  "e1_file": "e1_ts__result.json"}}
    res = B.run(plan, sel, B.make_runner_factory("", str(REAL), "cpu"), str(tmp_path))
    r = res["items"][0]
    assert r["stop"] in ("width", "cap") and r["replay_n"] == 64 and r["final"][1] <= 192
    assert isinstance(r["replay_match"], bool) and r["nll_max_abs_diff_vs_e1"] < 1e-2


# ------------------------------------------------------------------------------------------------ AC6: analyze_e1b
@pytest.fixture(scope="module")
def plan_sel():
    return B.build_plan(str(REAL))


def make_e1b(plan_sel, final=lambda e: e["e1"], skip=lambda e: False):
    """A synthetic e1b result: every selected item continued with the given final counts (default: E1's own)."""
    plan, sel = plan_sel
    items = [{"domain": e["domain"], "model": e["model"], "item": e["item"], "kind": e["kind"], "e1": e["e1"],
              "replay_hits": e["e1"][0], "replay_n": e["e1"][1], "replay_match": True, "final": list(final(e)),
              "stop": "cap", "wall_s": 1.0} for e in plan if not skip(e)]
    nr = [[e["model"], e["item"]] for e in plan if skip(e)]
    return {"protocol": B.PROTOCOL, "selection": sel, "items": items, "minutes": 1.0, "not_reached": nr,
            "complete": not nr}


E1_ANALYSIS = json.load(open(E1A_JSON, encoding="utf-8"))


def _p1_numbers(p1):
    return {s: {k: v[k] for k in ("items", "censored", "N_mean", "R_share_lo", "R_share_hi", "T_width_le2_share")} |
               {"e1": v["e1_rule"]["verdict"], "e1a": v["e1a_rule"]["verdict"]} for s, v in p1["sets"].items()}


def test_with_e1b_counts_equal_to_e1_the_readings_equal_the_saved_e1_analysis(plan_sel):
    res = AB.analyse(make_e1b(plan_sel), str(REAL))
    for dom, model in (("tinystories", "A0"), ("wmt14", "J0"), ("sudoku", "S0")):
        want = _p1_numbers(E1_ANALYSIS["domains"][dom]["prediction1"])
        got_e1b, got_e1 = _p1_numbers(res["models"][model]["e1b"]), _p1_numbers(res["models"][model]["e1"])
        assert got_e1 == want and got_e1b == want
        for s in want:
            for k, v in want[s].items():
                if isinstance(v, float):
                    assert got_e1b[s][k] == pytest.approx(v, rel=1e-9), (model, s, k)
    assert set(res["models"]["A0"]["e1b"]["sets"]) == {"all", "untouched"}
    assert res["models"]["A0"]["e1b"]["sets"]["untouched"]["items"] == 61
    assert res["verdict"]["e1_replication_bar_met"] is False                # J0 fails E1's rule on E1's own counts


def test_substitution_changes_only_the_continued_items_and_never_the_saved_files(plan_sel):
    files = {d: json.load(open(REAL / fn, encoding="utf-8")) for d, fn in AE.FILES.items()}
    before = copy.deepcopy(files)
    plan, _ = plan_sel
    chosen = [e for e in plan if e["model"] == "A0" and e["item"] in (0, 7)]
    e1b = make_e1b(plan_sel, final=lambda e: [1000, 2048] if e in chosen else e["e1"], skip=lambda e: e not in chosen)
    rows = files["tinystories"]["rows"]["A0"]
    new = AB.substitute(rows, e1b["items"], "A0")
    for i, (a, b) in enumerate(zip(rows, new)):
        assert (a == b) == (i not in (0, 7)), i
    assert new[0]["mc"]["8"] == [1000, 2048] and new[7]["mc"]["8"] == [1000, 2048]
    assert new[0]["mc"]["9"] == rows[0]["mc"]["9"] and new[0]["level_nll"] == rows[0]["level_nll"]
    assert files == before                                                  # the originals were not edited
    res = AB.analyse(e1b, str(REAL))
    a0, e0 = res["models"]["A0"]["e1b"]["sets"], res["models"]["A0"]["e1"]["sets"]
    assert a0["all"]["censored"] == e0["all"]["censored"] - 1
    assert a0["all"]["R_share_lo"] != e0["all"]["R_share_lo"]
    assert a0["untouched"]["censored"] == e0["untouched"]["censored"] - 1  # item 7 is untouched-set, item 0 is not
    assert res["models"]["A0"]["continuation"]["not_reached"] == sorted(i for i in A0_CENSORED + A0_WIDE if i not in (0, 7))
    # J0 and S0 are unchanged
    assert _p1_numbers(res["models"]["J0"]["e1b"]) == _p1_numbers(res["models"]["J0"]["e1"])
    assert _p1_numbers(res["models"]["S0"]["e1b"]) == _p1_numbers(res["models"]["S0"]["e1"])


def test_item_zero_changes_the_all_set_but_not_the_untouched_set(plan_sel):
    plan, _ = plan_sel
    only0 = [e for e in plan if e["model"] == "A0" and e["item"] == 0]
    e1b = make_e1b(plan_sel, final=lambda e: [1500, 2048], skip=lambda e: e not in only0)
    a0 = AB.analyse(e1b, str(REAL))["models"]["A0"]
    assert a0["e1b"]["sets"]["untouched"] == a0["e1"]["sets"]["untouched"]
    assert a0["e1b"]["sets"]["all"] != a0["e1"]["sets"]["all"]


def test_replication_verdict_follows_the_rule(plan_sel):
    plan, _ = plan_sel
    fixed = lambda e: [16384, 32768] if e["model"] == "J0" else e["e1"]          # rho = 0.5 on every continued J0 item
    res = AB.analyse(make_e1b(plan_sel, final=fixed), str(REAL))
    v = res["verdict"]
    assert res["models"]["J0"]["e1b"]["sets"]["all"]["e1_rule"]["verdict"] == "holds"
    assert v["e1_rule_by_model"] == {"A0": True, "J0": True, "S0": True} and v["e1_replication_bar_met"] is True
    assert v["run_complete"] is True and v["items_not_reached"] == {}
    # A0 fails E1's rule after the continuation (all 14 selected items at rho = e^-300, i.e. T = N): the bar is not met
    spoiled = lambda e: [16384, 32768] if e["model"] == "J0" else [0, 32768]
    res = AB.analyse(make_e1b(plan_sel, final=spoiled), str(REAL))
    assert res["models"]["A0"]["e1_rule_holds_e1b"] is False
    assert res["verdict"]["e1_replication_bar_met"] is False
    assert res["verdict"]["e1_rule_by_model"] == {"A0": False, "J0": True, "S0": True}
    # A0 holds on 'all' but not on the 61 untouched windows: not met either (both sets are read)
    p1 = res["models"]["J0"]["e1b"]
    assert AB.e1_rule_holds(p1) and not AB.e1_rule_holds({"sets": {"all": {"e1_rule": {"verdict": "holds"}},
                                                                   "untouched": {"e1_rule": {"verdict": "neither"}}}})
    # J0 unfixed (E1's own counts): not met
    res = AB.analyse(make_e1b(plan_sel), str(REAL))
    assert res["verdict"]["e1_rule_by_model"] == {"A0": True, "J0": False, "S0": True}


def test_items_not_reached_keep_e1_counts_and_are_reported(plan_sel):
    skip = lambda e: e["model"] == "J0" and e["item"] >= 100
    res = AB.analyse(make_e1b(plan_sel, final=lambda e: [16384, 32768], skip=skip), str(REAL))
    nr = [e["item"] for e in plan_sel[0] if skip(e)]
    assert nr and res["verdict"]["items_not_reached"] == {"J0": nr} and res["verdict"]["run_complete"] is False
    assert res["models"]["J0"]["continuation"]["continued"] == 58 - len(nr)
    # the not-reached censored items (e.g. 103, 110, ...) still count as censored at E1's counts
    assert res["models"]["J0"]["e1b"]["sets"]["all"]["censored"] == len([i for i in nr if i in J0_CENSORED])


def test_analysis_refuses_inconsistent_inputs(plan_sel):
    bad = make_e1b(plan_sel)
    bad["items"][0]["e1"] = [bad["items"][0]["e1"][0] + 1, bad["items"][0]["e1"][1]]
    with pytest.raises(ValueError):
        AB.analyse(bad, str(REAL))
    bad = make_e1b(plan_sel)
    bad["selection"] = copy.deepcopy(bad["selection"])
    bad["selection"]["A0"]["wide"] = bad["selection"]["A0"]["wide"][:-1]
    with pytest.raises(ValueError):
        AB.analyse(bad, str(REAL))


def test_analysis_cli_writes_json_without_absolute_paths(plan_sel, tmp_path):
    src = tmp_path / "e1b.json"
    src.write_text(json.dumps(make_e1b(plan_sel)), encoding="utf-8")
    out = tmp_path / "out" / "a.json"
    import sys
    argv, sys.argv = sys.argv, ["analyze_e1b", "--e1b", str(src), "--e1_dir", str(REAL), "--out", str(out)]
    try:
        AB.main()
    finally:
        sys.argv = argv
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["verdict"]["e1_replication_bar_met"] is False and "A0" in res["models"]
    assert str(ROOT) not in out.read_text(encoding="utf-8")
