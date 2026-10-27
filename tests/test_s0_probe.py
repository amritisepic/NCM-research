"""S0 neighbour probe (scripts/s0_neighbour_probe.py, docs/preregistration.md "S0"). CPU, synthetic, fast."""
import importlib.util
import os

import numpy as np
import pytest
import torch

import ncmr.ncm_ref_loader  # noqa: F401  (puts the upstream reference code on sys.path)
from models.next_concept import _TiedCodebookHead  # noqa: E402

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "s0_neighbour_probe.py")
_spec = importlib.util.spec_from_file_location("s0_neighbour_probe", _p)
S0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S0)

NE, D, V = 16, 8, 6


def make_head(seed=0, log_inv_t=2.0, calibrated=False):
    """calibrated=True: unit-norm codebook and proj = 0, so the released head is exactly the uniform marginal over the V
    codes (optimal for i.i.d. uniform codes: no term can gain without a real dependence)."""
    g = torch.Generator().manual_seed(seed)
    cb = torch.randn(V, D, generator=g)
    if calibrated:
        cb = cb / cb.norm(dim=-1, keepdim=True)
    head = _TiedCodebookHead(NE, cb, learn_scale=True, trainable=False, use_bias=False)
    with torch.no_grad():
        head.proj.weight.copy_(torch.zeros(D, NE) if calibrated else torch.randn(D, NE, generator=g) / 4)
        head.log_inv_T.fill_(log_inv_t)
    for p in head.parameters():
        p.requires_grad_(False)
    return head, cb


def randomised_term(seed=1):
    t = S0.NeighbourTerm(NE, D, rank=4, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        t.U.copy_(torch.randn(*t.U.shape, generator=g))
        t.U0.copy_(torch.randn(*t.U0.shape, generator=g))
    return t


def test_zero_init_term_returns_exactly_the_base_head_logits():
    head, cb = make_head()
    g = torch.Generator().manual_seed(3)
    h = torch.randn(5, 7, NE, generator=g)
    codes = torch.randint(0, V, (5, 7), generator=g)
    term = S0.NeighbourTerm(NE, D, rank=4, seed=0)
    assert float(term.U.abs().max()) == 0 and float(term.U0.abs().max()) == 0
    assert float(term.A.abs().max()) > 0 and float(term.B.abs().max()) > 0 and float(term.B0.abs().max()) > 0
    base = head(h)                                                      # the released head's own forward
    assert torch.equal(S0.level_logits(head, None, h, codes, cb), base)
    assert torch.equal(S0.level_logits(head, term, h, codes, cb), base)  # step 0 is the released model exactly
    # and a non-zero term does change the logits (the check above is not vacuous)
    assert not torch.allclose(S0.level_logits(head, randomised_term(), h, codes, cb)[:, 1:], base[:, 1:])


def test_neighbour_is_left_code_same_level_and_window_and_target_never_enters():
    head, cb = make_head()
    term = randomised_term()
    g = torch.Generator().manual_seed(4)
    h = torch.randn(3, 6, NE, generator=g)
    codes = torch.randint(0, V, (3, 6), generator=g)
    base = head(h)
    lg = S0.level_logits(head, term, h, codes, cb)
    # position 0 untouched
    assert torch.equal(lg[:, 0], base[:, 0])
    # position p >= 1 equals the hand-built query with the codebook vector of codes[:, p-1] of the same window
    for p in range(1, 6):
        n = cb[codes[:, p - 1]]
        z = head.proj(h[:, p]) + ((h[:, p] @ term.A.t()) * (n @ term.B.t())) @ term.U.t() + (n @ term.B0.t()) @ term.U0.t()
        want = S0.head_logits_from_z(head, z)
        assert torch.allclose(lg[:, p], want, atol=1e-6), p
    # the target code at p never enters features: changing codes[:, p] leaves the logits at positions <= p unchanged
    # and changes the logits at p + 1 only
    for p in range(6):
        c2 = codes.clone()
        c2[:, p] = (c2[:, p] + 1) % V
        lg2 = S0.level_logits(head, term, h, c2, cb)
        assert torch.equal(lg2[:, :p + 1], lg[:, :p + 1]), p
        if p + 1 < 6:
            assert not torch.allclose(lg2[:, p + 1], lg[:, p + 1])
            assert torch.equal(lg2[:, p + 2:], lg[:, p + 2:])
    # other windows do not leak: changing window 0's codes leaves windows 1, 2 unchanged
    c3 = codes.clone()
    c3[0] = (c3[0] + 1) % V
    lg3 = S0.level_logits(head, term, h, c3, cb)
    assert torch.equal(lg3[1:], lg[1:])
    # the NLL sums over every position including position 0
    nll = S0.level_nll(head, term, h, codes, cb)
    ce = -torch.log_softmax(lg, -1).gather(-1, codes[..., None])[..., 0]
    assert torch.allclose(nll, ce.sum(1), atol=1e-5)
    assert nll.shape == (3,)


def _planted(n, length, dependent, seed):
    """Hidden states carry no information about the codes; codes follow c_p = (c_{p-1} + 1) mod V when `dependent`."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, length, NE, generator=g)
    c = torch.randint(0, V, (n, length), generator=g)
    if dependent:
        for p in range(1, length):
            c[:, p] = (c[:, p - 1] + 1) % V
    return h, c


@pytest.mark.parametrize("dependent", [True, False])
def test_trained_term_gains_on_planted_dependence_and_not_without(dependent):
    head, cb = make_head(calibrated=True)
    h_tr, c_tr = _planted(384, 12, dependent, 10)
    h_dv, c_dv = _planted(128, 12, dependent, 11)
    h_te, c_te = _planted(256, 12, dependent, 12)
    term, info = S0.fit_level(head, cb, h_tr, c_tr, h_dv, c_dv, "cpu", seed=0, max_epochs=30, patience=3, lr=1e-2, rank=8)
    base = S0.eval_nll(head, None, h_te, c_te, cb, "cpu")
    with_t = S0.eval_nll(head, term, h_te, c_te, cb, "cpu")
    gain, se = S0.mean_se(base - with_t)
    if dependent:
        # base ~ 12 * log 6 = 21.5 nats; the planted map removes all but position 0 (~11 * 1.79 = 19.7 nats)
        assert gain > 10.0 and gain > 20 * se, (gain, se)
    else:
        assert abs(gain) < 1.0, gain


def test_early_stopping_keeps_best_dev_epoch_and_stops_after_three_non_improving():
    dev = [5.0, 4.0, 3.0, 3.5, 3.0, 3.2, 1.0, 0.5]    # best 3.0 at epoch 3; epochs 4, 5, 6 do not improve -> stop at 6
    state = {"w": 0}
    ep_log = []

    def train_epoch(ep):
        state["w"] = ep
        ep_log.append(ep)

    it = iter(dev)
    best_epoch, best_state, hist, run = S0.early_stop_loop(train_epoch, lambda: next(it), lambda: dict(state), 30, 3)
    assert (best_epoch, run, ep_log) == (3, 6, [1, 2, 3, 4, 5, 6])
    assert best_state == {"w": 3} and hist == dev[:6]
    # an improvement resets the counter: two stalls, then a better epoch, are not a stop
    dev2 = [3.0, 3.1, 3.1, 2.0, 2.1, 2.2, 2.3, 9.0]
    it = iter(dev2)
    be, bs, h2, r2 = S0.early_stop_loop(lambda e: None, lambda: next(it), lambda: {"x": 1}, 30, 3)
    assert (be, r2) == (4, 7)
    # the cap: never more than max_epochs
    it = iter(range(100, 0, -1))
    be, _, h3, r3 = S0.early_stop_loop(lambda e: None, lambda: next(it), lambda: {}, 5, 3)
    assert (be, r3) == (5, 5)


def test_fit_level_restores_best_dev_epoch_parameters():
    head, cb = make_head(calibrated=True)
    # no real dependence and a tiny training set: the term overfits, so dev NLL turns up and the best epoch is not the last
    h_tr, c_tr = _planted(48, 8, False, 20)
    h_dv, c_dv = _planted(64, 8, False, 21)
    term, info = S0.fit_level(head, cb, h_tr, c_tr, h_dv, c_dv, "cpu", seed=0, max_epochs=30, patience=3, lr=3e-2, rank=8)
    hist = info["dev_nll_per_epoch"]
    assert info["best_epoch"] < info["epochs_run"], hist
    assert info["best_epoch"] == int(np.argmin(hist)) + 1
    assert info["epochs_run"] == len(hist) <= 30
    # the returned parameters are the best epoch's: their dev NLL equals the recorded minimum
    d = float(S0.eval_nll(head, term, h_dv, c_dv, cb, "cpu").mean())
    assert abs(d - min(hist)) < 1e-4, (d, min(hist), hist)


def test_gate_arithmetic():
    g = S0.gate_part1
    assert g(10.0, 100.0)["verdict"] == "pass"            # exactly 10 nats and exactly 10% of base
    assert g(9.99, 100.0)["verdict"] == "fail"            # below 10 nats
    assert g(10.0, 200.0)["verdict"] == "pass"            # exactly 5% of base
    assert g(10.0, 200.5)["verdict"] == "fail"            # under 5% of base though >= 10 nats
    assert g(50.0, 2000.0)["verdict"] == "fail"           # 2.5%
    assert g(-3.0, 100.0)["verdict"] == "fail"
    r = g(12.0, 120.0)
    assert r["gain_pct_of_base"] == pytest.approx(10.0) and r["verdict"] == "pass"
