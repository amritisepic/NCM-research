"""Information-theory and target-leakage checks; fitting is run by the P1c script."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("p1c_synthetic", Path(__file__).resolve().parents[1] / "scripts/p1c_synthetic.py")
p1c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1c)


@pytest.mark.parametrize("case", ["independent", "duplicated", "markov_0.10", "markov_0.25"])
def test_true_tc_matches_enumerated_entropy(case):
    codes, prob, truth = p1c.population(case)
    assert prob.sum().item() == pytest.approx(1)
    joint = -(prob[prob > 0] * prob[prob > 0].log()).sum().item()
    marginal = 0.
    for i in range(codes.shape[1]):
        p = torch.stack([prob[codes[:, i] == v].sum() for v in (0, 1)])
        marginal -= (p[p > 0] * p[p > 0].log()).sum().item()
    assert joint == pytest.approx(truth["ar_nll"])
    assert marginal - joint == pytest.approx(truth["tc"])


def test_factorized_probe_never_uses_targets_and_ar_has_no_future_leak():
    fact, ar = p1c.make_pair(0)
    fact.eval(); ar.eval()
    z = torch.randn(2, 5, 8)
    a = torch.zeros(2, 4, dtype=torch.long)
    b = a.clone(); b[:, 2:] = 1
    with torch.no_grad():
        torch.testing.assert_close(fact(z, a), fact(z, b), rtol=0, atol=0)
        torch.testing.assert_close(ar(z, a)[:, :3], ar(z, b)[:, :3], rtol=0, atol=0)


def test_zero_previous_embeddings_do_not_make_masks_equal():
    fact, ar = p1c.make_pair(0)
    for e in ar.prev_emb:
        torch.nn.init.zeros_(e.weight)
    assert not torch.equal(fact.mask, ar.mask)
    z = torch.randn(2, 5, 8)
    targets = torch.zeros(2, 4, dtype=torch.long)
    with torch.no_grad():
        assert not torch.allclose(fact(z, targets), ar(z, targets), rtol=0, atol=1e-7)
