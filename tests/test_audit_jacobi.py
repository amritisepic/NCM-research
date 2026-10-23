"""Exact fixed-point equality for twenty released hard puzzles, CPU fp32."""
import pytest
import torch

from audit_support import load_jacobi


@pytest.fixture(scope="module")
def decoded(released_sudoku):
    bundle, puzzles = released_sudoku
    net, tok = bundle.net.eval(), bundle.vqvae.eval()
    jacobi = load_jacobi()
    k = len(tok.vq.levels)
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        sequential = net.generate(None, puzzles, temperature=0.0, top_k=0, top_p=1.0,
                                  cfg_scale=0.0, cfg_null_eos=None)
        z_prefix = tok.encoder.wte(puzzles)
        result = jacobi.jacobi_decode(
            net, lambda codes, has_prev: z_prefix, n=20, S=sum(tok.vq.levels),
            device="cpu", max_passes=k + 1, keep_history=True,
        )
        # Independently apply one more parallel update: stopping must mean a true
        # fixed point, not just returning the sequential result or hitting a cap.
        replay = jacobi.parallel_logits(net, z_prefix, result["codes"]).argmax(-1)
        seq_tokens = tok.decode(sequential).argmax(-1)
        jac_tokens = tok.decode(result["codes"]).argmax(-1)
    return sequential, result, replay, seq_tokens, jac_tokens, k


@pytest.mark.parametrize("puzzle", range(20), ids=lambda i: f"released-hard-{i:02d}")
def test_jacobi_fixed_point_equals_upstream_greedy(decoded, puzzle):
    sequential, result, replay, seq_tokens, jac_tokens, k = decoded
    assert result["stop"][puzzle].item() == 1, "must detect fixed point, not stop at the cap"
    assert 2 <= result["nfe"][puzzle].item() <= k + 1
    assert torch.equal(result["codes"][puzzle], replay[puzzle])
    assert torch.equal(result["codes"][puzzle], sequential[puzzle])
    assert torch.equal(jac_tokens[puzzle], seq_tokens[puzzle])
    t = result["nfe"][puzzle].item() - 1
    assert torch.equal(result["history"][t][puzzle], result["codes"][puzzle])
    assert torch.equal(result["history"][t - 1][puzzle], result["codes"][puzzle])
