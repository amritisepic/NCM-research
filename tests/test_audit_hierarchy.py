"""Compare the audit hierarchy against real upstream encoding, not a mock."""
import pytest
import torch
from omegaconf import OmegaConf

from ncmr.corruption import CorruptionCfg, build_hierarchy
from ncmr.ncm_ref_loader import VQVAE


def assert_clean_parity(tok, x, cfg):
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        encoded = tok.encode(x)
        upstream_inputs = tok.vq.idx_to_ncm_input(encoded)
        upstream_inputs2, upstream_targets = tok.get_ncm_inputs(x)
    # A caller may be training with bf16, but the hierarchy must stay fp32.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        h = build_hierarchy(tok, x, cfg, generator=torch.Generator().manual_seed(19))
    assert torch.equal(h["canonical"], encoded)
    assert torch.equal(h["targets"], encoded)
    assert torch.equal(h["used"], encoded)
    assert torch.equal(h["targets"][:, tok.vq.levels[0]:], upstream_targets)
    torch.testing.assert_close(h["inputs"], upstream_inputs, rtol=0, atol=0)
    torch.testing.assert_close(h["inputs"], upstream_inputs2, rtol=0, atol=0)
    torch.testing.assert_close(h["z_used"], tok.vq.idx_to_z_q(encoded), rtol=0, atol=0)
    assert h["inputs"].dtype == torch.float32
    assert h["corrupted_frac"] == 0


@pytest.mark.parametrize("cfg", [CorruptionCfg(), CorruptionCfg(mode="uniform", mean_rate=0),
                                  CorruptionCfg(mode="onpolicy", mean_rate=0),
                                  CorruptionCfg(mode="none", retarget=True)])
def test_clean_released_sudoku_equals_upstream(released_sudoku, cfg):
    bundle, puzzles = released_sudoku
    # The prefix has a BOS at each end; the tokenizer encodes the inner 89 positions.
    x = puzzles[:3, 1:-1].contiguous()
    assert x.shape[1] == bundle.vqvae.context_length
    assert_clean_parity(bundle.vqvae, x, cfg)


@pytest.mark.parametrize("learned_l0", [False, True])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_clean_irregular_text_levels_equal_upstream(learned_l0, seed):
    tok = tiny_text_tokenizer(learned_l0, seed)
    x = torch.randint(0, 32, (3, 128))
    assert_clean_parity(tok, x, CorruptionCfg())


def tiny_text_tokenizer(learned_l0=False, seed=0):
    # Same irregular 128-token schedule as W1, but a small upstream tokenizer.
    # This exercises learned L0 sampling, non-integer resampling and quant-resi.
    levels = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
    torch.manual_seed(seed)
    tok = VQVAE(
        in_vocab_size=32, context_length=128, embed_dim=8, vqvae_vocab_size=[32] * len(levels),
        n_layers=2, compression_factor=1, dropout=0, pre_quant_groupnorm=2,
        vector_quantizer_config=OmegaConf.create({
            "clss": "multiscale_residual_vector_quantizer", "decay": 0.99,
            "commitment_cost": 0.25, "learned_l1_sampling": learned_l0,
            "learned_l1_cascade": learned_l0, "learned_l1_cascade_base": 4,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
            "levels": {"use_manual_levels": True, "manual_levels": levels},
        }),
    ).eval()
    return tok
