"""Minimal resumable training loop shared by all our models (CPU here, CUDA on Colab).

- checkpoints (model, optimizer, step, RNG states, data RNG) every `ckpt_every` steps to `run_dir/ckpt.pt`
  (atomic rename) and resumes from it automatically;
- cosine LR with linear warmup; AdamW; grad clipping;
- autocast: bf16 on CPU and on CUDA GPUs with bf16 support, fp16 + GradScaler on older GPUs (T4);
- appends JSON lines of metrics to `run_dir/metrics.jsonl`.
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast_ctx(device: str, enabled: bool = True):
    if not enabled:
        return nullcontext(), None
    if device == "cuda":
        # native bf16 needs compute capability >= 8.0 (A100/L4); is_bf16_supported() can report emulated
        # support on a T4, which is slow, so use fp16 + loss scaling there.
        if torch.cuda.get_device_capability()[0] >= 8:
            return torch.autocast("cuda", dtype=torch.bfloat16), None
        return torch.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda")
    return torch.autocast("cpu", dtype=torch.bfloat16), None


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_frac: float = 0.1) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


class Trainer:
    def __init__(self, run_dir: str, params, max_lr: float, total_steps: int, warmup: int = 200,
                 weight_decay: float = 0.1, betas=(0.9, 0.95), clip: float = 1.0, ckpt_every: int = 500,
                 device: str = "cpu", amp: bool = True):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir, self.max_lr, self.total, self.warmup = run_dir, max_lr, total_steps, warmup
        self.clip, self.ckpt_every, self.device = clip, ckpt_every, device
        params = [p for p in params if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                      {"params": no_decay, "weight_decay": 0.0}], lr=max_lr, betas=betas)
        self.ctx, self.scaler = autocast_ctx(device, amp)
        self.step = 0
        self.t_start = time.time()
        self.wall_prev = 0.0

    def state(self, modules: dict, data_rng: np.random.Generator, extra: dict | None = None) -> dict:
        return {"step": self.step, "opt": self.opt.state_dict(),
                "modules": {k: m.state_dict() for k, m in modules.items()},
                "torch_rng": torch.get_rng_state(), "np_rng": data_rng.bit_generator.state,
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "wall": self.wall_prev + time.time() - self.t_start, "extra": extra or {},
                "scaler": self.scaler.state_dict() if self.scaler else None}

    def save(self, modules: dict, data_rng, extra=None, name="ckpt.pt"):
        path = os.path.join(self.run_dir, name)
        torch.save(self.state(modules, data_rng, extra), path + ".tmp")
        os.replace(path + ".tmp", path)

    def maybe_resume(self, modules: dict, data_rng: np.random.Generator) -> dict:
        path = os.path.join(self.run_dir, "ckpt.pt")
        if not os.path.exists(path):
            return {}
        st = torch.load(path, map_location="cpu", weights_only=False)  # our own file
        for k, m in modules.items():
            m.load_state_dict(st["modules"][k])
        self.opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        data_rng.bit_generator.state = st["np_rng"]
        if self.scaler and st.get("scaler"):
            self.scaler.load_state_dict(st["scaler"])
        self.step = st["step"]
        self.wall_prev = st.get("wall", 0.0)
        print(f"[resume] step {self.step} from {path}", flush=True)
        return st.get("extra", {})

    def backward_step(self, loss: torch.Tensor, params) -> float:
        for g in self.opt.param_groups:
            g["lr"] = lr_at(self.step, self.max_lr, self.warmup, self.total)
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
        else:
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, self.clip)
        if self.scaler:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.step += 1
        return float(gn)

    def log(self, rec: dict):
        rec = {"step": self.step, "wall_s": round(self.wall_prev + time.time() - self.t_start, 1), **rec}
        with open(os.path.join(self.run_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
