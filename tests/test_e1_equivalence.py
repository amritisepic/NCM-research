"""E1 (docs/preregistration.md): scripts/e1_equivalence.py's quantities on tiny models and synthetic rows."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from test_d1_bench import tiny_ts_like, tiny_wmt_like, PAD

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e1_equivalence", ROOT / "scripts" / "e1_equivalence.py")
E1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E1)


def ts_domain():
    tok, net = tiny_ts_like()
    V = int(tok.vq.vqvae_vocab_sizes[0])
    return E1.Domain("tinystories", net, tok, {"unigram": torch.full((V,), 1.0 / V)}), tok, net


def test_level_nll_is_the_teacher_forced_nll_per_level():
    dom, tok, net = ts_domain()
    x = torch.randint(0, 32, (1, 128), generator=torch.Generator().manual_seed(0))
    pre = torch.zeros(1, 1, dtype=torch.long)
    canon = dom.canonical(x)
    got = dom.level_nll(pre, canon)
    lv = [int(l) for l in tok.vq.levels]
    with torch.no_grad():
        lg = net(torch.cat([tok.encoder.wte(pre), tok.vq.idx_to_ncm_input(canon)], 1)).float()
    ce = F.cross_entropy(lg[0], canon[0, 1:], reduction="none")
    assert abs(got[0] - math.log(32)) < 1e-5                              # level 0: the unigram
    off = 0
    for j, l in enumerate(lv[1:], 1):
        assert abs(got[j] - float(ce[off:off + l].sum())) < 1e-3, j
        off += l
    assert len(got) == len(lv)


def test_mc_hits_counts_exact_decodes_and_stops_at_the_hit_target():
    dom, tok, net = ts_domain()
    x = torch.randint(0, 32, (1, 128), generator=torch.Generator().manual_seed(1))
    pre = torch.zeros(1, 1, dtype=torch.long)
    canon = dom.canonical(x)
    target = dom.key(dom.decode_ids(canon)[0].tolist())
    # finest level only, near-zero temperature: the greedy finest codes; a hit iff greedy decodes to x^
    greedy = net(torch.cat([tok.encoder.wte(pre), tok.vq.idx_to_ncm_input(canon)], 1)).float()[:, -128:].argmax(-1)
    c2 = torch.cat([canon[:, :-128], greedy], 1)
    expect_hit = dom.key(dom.decode_ids(c2)[0].tolist()) == target
    h, n = E1.mc_hits(dom, pre, canon, 11, target, 0, 64, 16, 20, temperature=1e-6)
    assert (h, n) == ((32, 32) if expect_hit else (0, 64))      # stops after the round that reaches 20 hits
    # a wrong target never hits, and the sample cap holds
    h, n = E1.mc_hits(dom, pre, canon, 5, ("nope",), 0, 40, 16, 20)
    assert (h, n) == (0, 40)


def test_wmt_key_reads_the_text_up_to_eos():
    tok, net, x = tiny_wmt_like()
    dom = E1.Domain("wmt14", net, tok, {"eos": 3, "pad": PAD})
    assert dom.key([5, PAD, 6, 3, 7, 8]) == (5, 6)
    assert dom.key([5, 6, 3]) == dom.key([5, 6, 3, 9, 9])
    assert dom.key([5, 6]) != dom.key([5, 6, 7])


def _rows(spec):
    """spec: per item (level_nll list, {k: (hits, n)})."""
    return [{"level_nll": nll, "recon_exact": True, "mc": {str(k): list(v) for k, v in mc.items()}} for nll, mc in spec]


def test_summary_decomposition_and_bounds():
    nll = [1.0] * 8 + [10.0, 10.0, 10.0, 10.0]                  # N_8 = 40, A_8 = 8
    rows = _rows([(nll, {8: (20, 40)}), (nll, {8: (10, 1000)})])
    s = E1.summarise(rows, [8], 12)["8"]
    T = np.array([math.log(2), math.log(100)])
    assert abs(s["N_mean"] - 40) < 1e-12 and abs(s["A_mean"] - 8) < 1e-12 and s["hit_share"] == 1.0
    assert abs(s["T_mean"] - T.mean()) < 1e-12 and abs(s["R_mean"] - (40 - T).mean()) < 1e-12
    assert abs(s["U_mean"] - (8 + T).mean()) < 1e-12 and abs(s["R_share_of_N"] - (40 - T.mean()) / 40) < 1e-12
    rows = _rows([(nll, {8: (20, 40)}), (nll, {8: (0, 2048)})])  # a zero-hit item: bounds only
    s = E1.summarise(rows, [8], 12)["8"]
    assert "T_mean" not in s and s["hit_share"] == 0.5 and s["T_median"] is None
    assert abs(s["R_lo_mean"] - (40 - math.log(2)) / 2) < 1e-12
    assert abs(s["R_hi_mean"] - ((40 - math.log(2)) + (40 - math.log(2048 / 3))) / 2) < 1e-12


def test_paired_differences():
    a = _rows([([1.0] * 8 + [5.0] * 4, {8: (20, 40)}), ([1.0] * 8 + [5.0] * 4, {8: (20, 80)})])
    b = _rows([([1.0] * 8 + [4.0] * 4, {8: (20, 20)}), ([1.0] * 8 + [3.0] * 4, {8: (20, 80)})])
    p = E1.paired(a, b, 8)
    assert abs(p["dN_mean"] - (-4 - 8) / 2) < 1e-12
    assert abs(p["dT_mean"] - (-math.log(2) + 0) / 2) < 1e-12
    assert abs(p["dT_share_of_dN"] - abs(p["dT_mean"]) / 6) < 1e-12


def test_coarse_cuts_run_on_the_subset_only():
    dom, tok, net = ts_domain()
    g = torch.Generator().manual_seed(2)
    items = [(torch.zeros(1, 1, dtype=torch.long), torch.randint(0, 32, (1, 128), generator=g)) for _ in range(3)]
    rows = E1.measure(dom, items, [6, 11], 0, max_samples=8, round_=8, hits_needed=1, log_every=99, full_cut_min=8,
                      coarse_n=1)
    assert [sorted(r["mc"]) for r in rows] == [["11", "6"], ["11"], ["11"]]
    s = E1.summarise(rows, [6, 11], 12)
    assert s["6"]["items"] == 1 and s["11"]["items"] == 3


def test_invalid_reconstruction_never_counts_as_hits_and_is_excluded():
    dom, tok, net = ts_domain()
    x = torch.randint(0, 32, (1, 128), generator=torch.Generator().manual_seed(3))
    pre = torch.zeros(1, 1, dtype=torch.long)
    assert E1.mc_hits(dom, pre, dom.canonical(x), 11, None, 0, 64, 16, 20) == (0, 0)
    nll = [1.0] * 8 + [10.0] * 4
    rows = _rows([(nll, {8: (20, 40)}), (nll, {8: (0, 0)})])
    s = E1.summarise(rows, [8], 12)["8"]
    assert s["items"] == 1 and s["hit_share"] == 1.0
    assert E1.paired(rows, rows, 8)["dN_mean"] == 0.0


def test_zero_hit_item_with_small_n_is_clamped_at_r_zero():
    rows = _rows([([1.0] * 8 + [0.75] * 4, {8: (0, 2048)})])           # N_8 = 3 < log(2048 / 3)
    assert E1.summarise(rows, [8], 12)["8"]["R_hi_mean"] == 0.0
