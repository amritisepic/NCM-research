"""Load the released NCM checkpoints with the vendored, unmodified upstream code (src/ncmr/ncm_ref).

Differences from upstream `scripts/eval_shard.py`, none of which change model outputs:
- device-agnostic (upstream hard-codes .cuda());
- checkpoints are loaded with torch.load(weights_only=True) and pickled BPE files with a
  restricted unpickler (the released files contain no code; this keeps it that way).
"""
from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass

import torch

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncm_ref")
if REF_DIR not in sys.path:
    sys.path.insert(0, REF_DIR)

from omegaconf import OmegaConf  # noqa: E402

from models.next_concept import NextConceptTransformer  # noqa: E402  (upstream)
from models.vqvae import VQVAE  # noqa: E402  (upstream)
from tokenization.bpe_tokenizer import BPETokenizer  # noqa: E402  (upstream)
from tokenization.sudoku_tokenizer import SudokuTokenizer  # noqa: E402  (upstream)


class _NoCodeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a data pickle")


def safe_pickle_load(path):
    with open(path, "rb") as f:
        return _NoCodeUnpickler(f).load()


@dataclass
class NCMBundle:
    cfg: object
    vqvae: VQVAE
    net: NextConceptTransformer
    pre_tokenizer: object
    ckpt_extra: dict  # non-weight entries of generator.pt (e.g. inference.level0_unigram)


def load_ncm(ckpt_dir: str, device: str = "cpu", quiet: bool = True) -> NCMBundle:
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    tcfg = cfg.tokenizer
    if tcfg.pre_tokenizer.name == "bpe":
        vocab = safe_pickle_load(os.path.join(ckpt_dir, cfg.checkpoints.bpe_vocab))
        merges = safe_pickle_load(os.path.join(ckpt_dir, cfg.checkpoints.bpe_merges))
        pre = BPETokenizer(vocab, merges, tcfg.pre_tokenizer.special_tokens)
    elif tcfg.pre_tokenizer.name == "sudoku":
        pre = SudokuTokenizer()
    elif tcfg.pre_tokenizer.name == "hf":
        from tokenization.hf_tokenizer import HFTokenizer  # upstream; downloads the HF tokenizer files
        pre = HFTokenizer.from_config(tcfg.pre_tokenizer)
    else:
        raise NotImplementedError(tcfg.pre_tokenizer.name)

    _stdout = sys.stdout
    if quiet:
        sys.stdout = open(os.devnull, "w")
    try:
        vqvae = VQVAE(**tcfg.vqvae)
        tok_sd = torch.load(os.path.join(ckpt_dir, cfg.checkpoints.tokenizer), map_location="cpu", weights_only=True)
        vqvae.load_state_dict(tok_sd["model"], strict=True)
        vqvae = vqvae.to(device).eval()
        net = NextConceptTransformer(vqvae_local=vqvae, config=cfg.model).to(device)
        gen_sd = torch.load(os.path.join(ckpt_dir, cfg.checkpoints.generator), map_location="cpu", weights_only=True)
        net.load_state_dict(gen_sd["model"], strict=True)
        net.eval()
    finally:
        if quiet:
            sys.stdout.close()
            sys.stdout = _stdout
    extra = {k: v for k, v in gen_sd.items() if k != "model"}
    return NCMBundle(cfg=cfg, vqvae=vqvae, net=net, pre_tokenizer=pre, ckpt_extra=extra)


class ForwardCounter:
    """Counts generator forward passes (NFE) with a hook on the first transformer block."""

    def __init__(self, net: NextConceptTransformer):
        self.n = 0
        self._h = net.transformer.blocks[0].register_forward_hook(self._hook)

    def _hook(self, *_):
        self.n += 1

    def close(self):
        self._h.remove()
