"""Regression tests for Codex's PR 12 findings on W2M (docs/reviews/triage_pr12.md):
4384369537: the flags' contamination footprint must include the residual convolution (quant_resi) that
            idx_to_ncm_input applies after the up-sampling;
4384369539: mechanism measure (a) must be computed in window batches (no full-vocabulary tensor over all windows).
CPU, tiny models."""
import numpy as np
import pytest
import torch

from ncmr import w2m
from ncmr.w2m import context_dependence_kl, make_perturbations, mask_flags
from w2m_support import prefix_fn, tiny_case


def _replace(canon, masks, vocab, g):
    """Every masked slot gets a uniformly random different code."""
    c, acc = canon.clone(), 0
    for k, m in enumerate(masks):
        lk = m.shape[1]
        shift = torch.randint(1, vocab[k], m.shape, generator=g)
        blk = c[:, acc:acc + lk]
        c[:, acc:acc + lk] = torch.where(m, (blk + shift) % vocab[k], blk)
        acc += lk
    return c


@pytest.mark.parametrize("learned", [True, False])
def test_flags_are_positive_exactly_where_the_tokenizers_inputs_change(learned):
    """Ground truth from the tokenizer itself: an input position is contaminated iff replacing the masked codes
    changes it. Exact comparison: unaffected positions are computed from identical values."""
    tok, _ = tiny_case(learned_l0=learned)
    vq = tok.vq
    assert vq.quant_resi_enabled
    levels, vocab = [int(l) for l in vq.levels], [int(v) for v in vq.vqvae_vocab_sizes]
    g = torch.Generator().manual_seed(7)
    B = 24
    canon = torch.cat([torch.randint(0, vocab[k], (B, l), generator=g) for k, l in enumerate(levels)], 1)
    masks = [torch.zeros(B, l, dtype=torch.bool) for l in levels]
    for r in range(B):
        if r == 0:
            continue                                                   # row 0: nothing replaced
        for _ in range(1 + r % 3):
            k = int(torch.randint(1 if r % 4 else 0, len(levels) - 1, (1,), generator=g))
            masks[k][r, int(torch.randint(0, levels[k], (1,), generator=g))] = True
    with torch.no_grad():
        h0 = vq.idx_to_ncm_input(canon)
        h1 = vq.idx_to_ncm_input(_replace(canon, masks, vocab, g))
    changed = (h0 != h1).any(-1)
    flags = mask_flags(vq, masks)
    assert changed.shape == flags.shape
    assert torch.equal(flags > 0, changed), int(((flags > 0) != changed).sum())
    assert float(flags[0].abs().sum()) == 0.0


def test_a_level1_code_contaminates_one_position_beyond_its_linear_support():
    """Codex's case, on the tiny schedule: one level-1 slot replaced; its linear support ends at some position t, and
    the residual convolution carries contamination to t + 1, so a finest-level flag there is > 0."""
    tok, _ = tiny_case(learned_l0=False)
    vq = tok.vq
    levels = [int(l) for l in vq.levels]
    m = [torch.zeros(1, l, dtype=torch.bool) for l in levels]
    m[1][0, 3] = True
    lin = vq._up_to_full(m[1].float()[..., None], 1, levels[-1])[0, :, 0] > 0
    t = int(lin.nonzero().max())
    assert t + 1 < levels[-1]
    f = mask_flags(vq, m)[0, -levels[-1]:]                             # finest input level (length L) is last
    assert float(f[t + 1]) > 0
    assert w2m._resi_halo(vq, 1) == 1


def _kl_full_batch(net, tok, prefix_emb, canon, perturb, device):
    """The pre-fix implementation (all windows at once), kept as the numerical reference."""
    levels = [int(l) for l in tok.vq.levels]
    vocab = [int(v) for v in tok.vq.vqvae_vocab_sizes]
    clean = w2m.forward_logits(net, prefix_emb, tok.vq.idx_to_ncm_input(canon), device, bs=10 ** 6)
    res = {}
    for j, (pos, shift) in perturb.items():
        c = canon.clone()
        rows = torch.arange(canon.shape[0])
        idx = sum(levels[:j]) + pos
        c[rows, idx] = (c[rows, idx] + shift) % vocab[j]
        pert = w2m.forward_logits(net, prefix_emb, tok.vq.idx_to_ncm_input(c), device, bs=10 ** 6)
        lo = sum(levels[1:j + 1])
        lp_c, lp_p = clean[:, lo:].log_softmax(-1), pert[:, lo:].log_softmax(-1)
        p_c = lp_c.exp()
        res[int(j)] = float(torch.where(p_c > 0, p_c * (lp_c - lp_p), torch.zeros_like(p_c)).sum(-1).mean())
    return res


def test_context_dependence_kl_is_batched_over_windows_and_equals_the_full_batch_value(monkeypatch):
    tok, net = tiny_case()
    net.eval()
    levels, vocab = [int(l) for l in tok.vq.levels], [int(v) for v in tok.vq.vqvae_vocab_sizes]
    g = torch.Generator().manual_seed(3)
    N, bs = 37, 8                                                      # not a multiple of bs
    canon = torch.cat([torch.randint(0, vocab[k], (N, l), generator=g) for k, l in enumerate(levels)], 1)
    perturb = make_perturbations(canon, levels, vocab, (1, 2, 3, 4, 5, 6), seed=11)
    pe = prefix_fn(tok)
    want = _kl_full_batch(net, tok, pe, canon, perturb, "cpu")
    seen = []
    orig = w2m.forward_logits

    def spy(net_, prefix_emb, inputs, *a, **k):
        seen.append(int(inputs.shape[0]))
        return orig(net_, prefix_emb, inputs, *a, **k)

    monkeypatch.setattr(w2m, "forward_logits", spy)
    got = context_dependence_kl(net, tok, pe, canon, perturb, "cpu", bs=bs)
    assert max(seen) <= bs, seen                                       # no forward (hence no logits tensor) spans > bs windows
    assert sorted(got) == sorted(want)
    for j in want:
        assert np.isclose(got[j], want[j], rtol=1e-5, atol=1e-7), (j, got[j], want[j])
        assert got[j] > 0


def test_hidden_only_pass_does_not_keep_logits():
    tok, net = tiny_case()
    net.eval()
    levels, vocab = [int(l) for l in tok.vq.levels], [int(v) for v in tok.vq.vqvae_vocab_sizes]
    canon = torch.cat([torch.randint(0, vocab[k], (5, l)) for k, l in enumerate(levels)], 1)
    inp = tok.vq.idx_to_ncm_input(canon)
    lg, hid = w2m.forward_logits(net, prefix_fn(tok), inp, "cpu", hidden=True, keep_logits=False)
    lg2, hid2 = w2m.forward_logits(net, prefix_fn(tok), inp, "cpu", hidden=True)
    assert lg is None and lg2 is not None and torch.equal(hid, hid2)
