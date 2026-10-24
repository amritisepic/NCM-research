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
    assert rep["max_abs_diff"] == 0.5 and rep["ok"] is False                   # 0.5 > the 0.01 tolerance
    assert diag().reproduction([10.0, 12.0, 13.0], [42, 0, 2], expect)["ok"] is True


EXPECT = {"config": {"sample_eval": {"final_seeds": [42, 0, 1, 2]}},
          "final_eval": {"genppl_gpt2-large_per_group": [10.0, 11.0, 12.0, 13.0]}}


def test_reproduction_fails_on_unmatched_seed():
    """Codex PR 7 comment 4367597216: a seed with no wave-1 group must not be ignored."""
    rep = diag().reproduction([10.0, 99.0], [0, 999], EXPECT)
    assert rep["missing_seeds"] == [999] and rep["ok"] is False


def test_reproduction_fails_on_substituted_checkpoint_scores():
    """A swapped checkpoint shifts every group; NaN scores must not pass either."""
    assert diag().reproduction([10.3, 11.2, 12.4, 13.1], [42, 0, 1, 2], EXPECT)["ok"] is False
    assert diag().reproduction([10.0, float("nan"), 12.0, 13.0], [42, 0, 1, 2], EXPECT)["ok"] is False


def test_mauve_features_are_fp32_when_amp_is_off(monkeypatch):
    """Codex PR 7 comment 4367597219: W1d featurizes MAUVE in fp32; the default stays bf16 (wave 1's protocol)."""
    from ncmr import eval_text
    seen = []

    class Tok:
        eos_token, eos_token_id = "<eos>", 0

        def __call__(self, text, truncation=True, max_length=256):
            return {"input_ids": [1, 2, 3][:max(1, len(text) % 3 + 1)]}

    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, ids, output_hidden_states=True):
            seen.append(torch.is_autocast_enabled("cpu"))
            h = self.lin(torch.ones(ids.shape[0], ids.shape[1], 4))
            return type("Out", (), {"hidden_states": [h]})()

    monkeypatch.setattr(eval_text, "_load_lm", lambda name, device: (Tok(), LM()))
    monkeypatch.setattr(eval_text, "mauve_score", lambda p, q, seed=0: 0.5)
    eval_text.mauve_features(["a b", "c"], "cpu", amp=False)
    eval_text.mauve_features(["a b", "c"], "cpu")
    assert seen == [False, True]
    seen.clear()
    eval_text.evaluate_samples([[5, 6], [7]], lambda ids: " ".join(map(str, ids)), scorers=(),
                               ref_feats=np.zeros((2, 4)), device="cpu", mauve_amp=False)
    assert seen == [False]


def test_registration_matches_script_defaults_and_notebook():
    """Codex PR 7 comment 4367597223: the W1d registration, the script defaults and the notebook agree."""
    reg = (ROOT / "docs/preregistration.md").read_text(encoding="utf-8").replace("\r\n", "\n").split("## W1d")[1].split("\n## ")[0]
    for phrase in ("T in {0.8, 0.9, 1.0}", "8 groups x 256", "seeds 42, 0-6", "first 1024 val windows",
                   "j in {-1, 3, 5, 7}", "max |diff| <=\n  0.01"):
        assert phrase in reg, phrase
    d = diag().build_parser().parse_args(["--name", "x", "--run_dir", "r"])
    assert (d.temps, d.seeds, d.n, d.n_oracle, d.js, d.tol) == ([0.8, 0.9, 1.0], [42, 0, 1, 2, 3, 4, 5, 6], 256, 1024,
                                                               [-1, 3, 5, 7], 0.01)
    spec = importlib.util.spec_from_file_location("make_notebooks", ROOT / "notebooks/make_notebooks.py")
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    cmds = nb.RUNS["w1d_requant_diag"]["cmds"]
    pairs = {"A0": (None, "L_ts_released16__result.json"), "A1": ("L_ts_ctrl_s0/final.pt", "L_ts_ctrl_s0__result.json"),
             "A2": ("L_ts_requant_s0/final.pt", "L_ts_requant_s0__result.json")}
    assert len(cmds) == 3
    for c in cmds:
        assert not any(f in c for f in ("--temps", "--seeds", "--n ", "--n_oracle", "--js", "--tol")), c
        name = c.split("--name ")[1].split()[0]
        ckpt, expect = pairs[name]
        assert ("--ckpt" in c) == (ckpt is not None) and (ckpt is None or ckpt in c) and expect in c, c
