"""W1d diagnostic (scripts/diag_w1_frontier.py): its sampler must be train_gen.py's, the oracle curve must supply the
canonical coarse codes, shared seeds must pair samples across models, and the reproduction check must read W1b's
sorted-seed group order. CPU, tiny tokenizer and generator; no released checkpoint needed."""
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

from audit2_support import definitions
from ncmr.data import EOS, PAD, cut_at_eos
from ncmr.models import build_ncm_generator
from ncmr.ncm_ref_loader import ForwardCounter
from test_audit_hierarchy import tiny_text_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def diag():
    spec = importlib.util.spec_from_file_location("diag_w1_frontier", ROOT / "scripts/diag_w1_frontier.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tiny():
    tok = tiny_text_tokenizer(learned_l0=False, seed=3)
    for p in tok.parameters():
        p.requires_grad_(False)
    torch.manual_seed(5)
    net = build_ncm_generator(tok, n_layer=1, n_head=2, n_embd=32, dropout=0.0).eval()
    return tok, net, torch.full((32,), 1 / 32)


def test_draw_equals_train_gen_draw_samples():
    tok, net, L0 = tiny()
    ns = definitions("scripts/train_gen.py", ["draw_samples"], dict(
        torch=torch, time=time, net=net, tok=tok, device="cpu", L0=L0, l0=int(tok.vq.levels[0]), EOS=EOS, PAD=PAD,
        cut_at_eos=cut_at_eos, ForwardCounter=ForwardCounter))
    reference, _, _ = ns["draw_samples"](6, 42, bs=4)
    ours = diag().draw(net, tok, L0, 6, 42, 1.0, "cpu", bs=4)
    assert ours == reference
    assert diag().draw(net, tok, L0, 6, 43, 1.0, "cpu", bs=4) != reference   # the seed matters


def test_oracle_levels_are_canonical_and_full_oracle_is_reconstruction():
    tok, net, L0 = tiny()
    m = diag()
    x = torch.randint(2, 32, (3, 128))
    levels = [int(l) for l in tok.vq.levels]
    off = np.cumsum([0] + levels)
    canon = tok.vq.z_to_idx(tok._encode_pre_quant(x))
    for j in (0, 3, 6):
        _, codes = m.oracle_sample(net, tok, L0, x, j, torch.Generator().manual_seed(1), "cpu", return_codes=True)
        assert torch.equal(codes[:, :off[j + 1]], canon[:, :off[j + 1]])
    ids = m.oracle_sample(net, tok, L0, x, len(levels) - 1, torch.Generator().manual_seed(1), "cpu")
    recon = tok.decoder(tok.vq.idx_to_z_q(canon)).argmax(-1)
    assert ids == [[t for t in cut_at_eos(r) if t != PAD] for r in recon.tolist()]


def test_oracle_samples_are_paired_by_seed():
    tok, net, L0 = tiny()
    m = diag()
    x = torch.randint(2, 32, (4, 128))
    one = m.oracle_sample(net, tok, L0, x, 2, torch.Generator().manual_seed(7), "cpu")
    two = m.oracle_sample(net, tok, L0, x, 2, torch.Generator().manual_seed(7), "cpu")
    other = m.oracle_sample(net, tok, L0, x, 2, torch.Generator().manual_seed(8), "cpu")
    assert one == two and one != other


def test_reproduction_reads_sorted_seed_order():
    expect = {"config": {"sample_eval": {"final_seeds": [42, 0, 1, 2]}},
              "final_eval": {"genppl_gpt2-large_per_group": [10.0, 11.0, 12.0, 13.0]}}   # seeds 0, 1, 2, 42
    rep = diag().reproduction([10.0, 11.5, 13.0], [42, 0, 2], expect)            # ours, sorted: 0, 2, 42
    assert rep["pairs"] == {0: [10.0, 10.0], 2: [11.5, 12.0], 42: [13.0, 13.0]}
    assert rep["max_abs_diff"] == 0.5
