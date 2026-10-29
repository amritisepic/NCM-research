"""Tiny CPU fixtures for the W2M tests (src/ncmr/w2m.py)."""
import torch

from ncmr.corruption import CorruptionCfg
from ncmr.models import build_ncm_generator
from ncmr.w2m import MechArm
from test_audit_hierarchy import tiny_text_tokenizer


def tiny_case(learned_l0=True, seed=42):
    tok = tiny_text_tokenizer(learned_l0=learned_l0, seed=seed)
    for p in tok.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    net = build_ncm_generator(tok, n_layer=1, n_head=2, n_embd=32, dropout=0.1)
    return tok, net


def prefix_fn(tok):
    return lambda b: tok.encoder.wte(torch.zeros(b, 1, dtype=torch.long))


def make_arm(arm, tok, net, seed=0, clean_frac=0.25, rates=None, frozen=None):
    from ncmr.w2m import ALL_ARMS as ARMS
    cfg = CorruptionCfg(mode="mech")
    if frozen is None and ARMS[arm][0] == "frozen_wrong":
        frozen = MechArm.frozen_copy(net)
    V = int(tok.vq.vqvae_vocab_sizes[0])
    return MechArm(arm, tok, net, torch.ones(V) / V, prefix_fn(tok), "cpu", seed,
                   rates if rates is not None else cfg.rates(len(tok.vq.levels)), clean_frac=clean_frac, frozen=frozen)


# ---------------------------------------------------------------------------------------------------------------
# A harness that runs the REAL training loop of scripts/train_gen.py (its `while` node and step functions, executed
# in a namespace of tiny CPU objects), or the pre-W2M copy of it (tests/w2m_ref_train_gen_funcs.txt).
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf

from ncmr.corruption import build_hierarchy, soft_ce
from ncmr.train import Trainer, fp32_eval
from ncmr.w2m import FlagVector, ALL_ARMS as ARMS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def script_nodes(kind="cur"):
    src = (HERE / "w2m_ref_train_gen_funcs.txt" if kind == "ref" else ROOT / "scripts/train_gen.py").read_text(encoding="utf-8")
    tree = ast.parse(src.replace("\r\n", "\n"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    loop = next(n for n in tree.body if isinstance(n, ast.While) and ast.unparse(n.test) == "trainer.step < int(tc.steps)")
    ifs = {ast.unparse(n.test): n for n in tree.body if isinstance(n, ast.If)}
    return funcs, loop, ifs


def run_nodes(nodes, ns, name="train_gen"):
    exec(compile(ast.Module(body=list(nodes), type_ignores=[]), name, "exec"), ns)


class Stream:
    def batch(self, rng, n):
        return rng.integers(0, 32, (n, 128))


class Harness:
    """mode: 'mech' (with `arm`), 'uniform' or 'onpolicy' (the old arms). kind: 'cur' | 'ref' (old arms only)."""

    def __init__(self, tmp, arm=None, mode="mech", kind="cur", accum=2, B=8, ckpt_every=2, seed=0, resume=False,
                 frozen_after_restore=False, at=(), every=0, level_rates=None, flags_w=None):
        torch.manual_seed(1234)                                    # the dropout stream
        tok, net = tiny_case()
        self.tok, self.net = tok, net
        V = int(tok.vq.vqvae_vocab_sizes[0])
        L0 = torch.ones(V) / V
        K = len(tok.vq.levels)
        corr = CorruptionCfg(mode=mode, level_rates=level_rates)
        self.mech = self.flagmod = None
        if mode == "mech":
            frozen = MechArm.frozen_copy(net) if ARMS[arm][0] == "frozen_wrong" else None
            self.mech = MechArm(arm, tok, net, L0, prefix_fn(tok), "cpu", seed, corr.rates(K), clean_frac=0.25, frozen=frozen)
            if self.mech.flag_mode:
                self.flagmod = FlagVector(tok.embed_dim)
                if flags_w is not None:
                    self.flagmod.w.data.copy_(flags_w)
        params = [p for p in net.parameters() if p.requires_grad] + (list(self.flagmod.parameters()) if self.flagmod else [])
        self.trainer = Trainer(str(tmp), params, max_lr=5e-4, total_steps=8, warmup=1, ckpt_every=ckpt_every, device="cpu", amp=False)
        self.modules = {"model": net, **({"flag": self.flagmod} if self.flagmod else {})}
        self.data_rng = np.random.default_rng(seed)
        resume_extra = self.trainer.maybe_resume(self.modules, self.data_rng) if resume else {}
        if frozen_after_restore and self.mech is not None and ARMS[arm][0] == "frozen_wrong":
            self.mech.frozen = MechArm.frozen_copy(net)           # the WRONG order (negative control)
        if self.mech is not None:
            self.mech.fwd = net
        net.train()
        self.sample_calls, self.log = [], []
        funcs, self.loop, ifs = script_nodes(kind)
        self.tc = OmegaConf.create({"steps": 0, "batch_size": B, "eval_every": 1, "ckpt_every": ckpt_every, "log_every": 1})
        self.ns = dict(torch=torch, F=F, np=np, json=json, time=__import__("time"), os=__import__("os"), sys=__import__("sys"),
                       net=net, fnet=net, tok=tok, L0=L0, l0=1, K=K, levels=[int(l) for l in tok.vq.levels], device="cpu",
                       build_hierarchy=build_hierarchy, soft_ce=soft_ce, fp32_eval=fp32_eval, corr=corr, policy_passes=1,
                       policy_temp=1.0, EOS=0, PAD=1, mech=self.mech, flagmod=self.flagmod, trainer=self.trainer,
                       modules=self.modules, data_rng=self.data_rng, stream=Stream(), tc=self.tc,
                       sc=OmegaConf.create({"n": 4}), seed=seed, accum=accum, calib=None, deadline=None, a=SimpleNamespace(max_minutes=0),
                       every_s=every, at_s=set(at), t_last=0.0, s_last=0, fl_step=1.0, params=params, resume_extra=resume_extra,
                       gen=torch.Generator().manual_seed(seed * 7919),
                       policy_gen=torch.Generator().manual_seed(seed * 7919 + 104729),
                       val_nll=lambda: {"val_nll": 0.0},
                       sample_metrics=lambda n, seeds, mauve=False: (self.sample_calls.append(self.trainer.step) or {"gen_ppl": 1.0}, None, None))
        self.ns["prefix_emb"] = prefix_fn(tok)
        defs = [n for k, n in funcs.items() if k in ("policy_codes", "compute_loss", "compute_loss_mech", "save_ckpt")]
        run_nodes(defs, self.ns)
        if resume:
            run_nodes([ifs["trainer.step"]], self.ns)
            if "trainer.step and mech is not None" in ifs:
                run_nodes([ifs["trainer.step and mech is not None"]], self.ns)

    def run(self, until):
        self.tc.steps = until
        run_nodes([self.loop], self.ns)
        return self

    def losses(self):
        import os
        p = os.path.join(self.trainer.run_dir, "metrics.jsonl")
        return [json.loads(l) for l in open(p) if '"loss"' in l]

    def state(self):
        return {"model": {k: v.clone() for k, v in self.net.state_dict().items()},
                "flag": None if self.flagmod is None else self.flagmod.w.detach().clone(),
                "mech": None if self.mech is None else self.mech.state_dict(),
                "torch_rng": torch.get_rng_state().clone(), "gen": self.ns["gen"].get_state(),
                "policy_gen": self.ns["policy_gen"].get_state()}
