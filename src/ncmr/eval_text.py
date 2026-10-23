"""Evaluation suite for unconditional TinyStories samples.

Every report includes all of these (never generative perplexity alone):
  genppl_<scorer> : generative perplexity under a scorer LM, computed with the upstream NCM function
                    (evaluation/gen_ppl.py: empty prompt, all generated tokens scored, corpus-level).
                    Scorers: gpt2-large (as in the NCM paper) and roneneldan/TinyStories-33M (in-domain).
  entropy         : mean per-sample unigram entropy (nats) of the sample's BPE ids (as in NCM).
  rep_n           : mean per-sample fraction of repeated n-grams, 1 - unique/total (n = 2, 3, 4).
  distinct_n      : corpus-level unique n-grams / total n-grams (n = 2, 3).
  mauve           : MAUVE(samples, human test text) on gpt2-large final-token features.
  len_tokens      : mean sample length in BPE tokens (after cutting at EOS).
  dup_adj1/2      : repeated words at distance 1 ("little little") and 2 ("Sam and Sam") per 1000 words, the
                    artifact NCM's paper reports; compare with the same rate on human text.
Plain-language notes: generative perplexity = how surprised a separate trained LM is by the samples
(lower = more fluent, but a sampler that repeats safe phrases scores well, hence entropy/rep next to it);
MAUVE = how close the distribution of samples is to human text in a learned feature space (0..1).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import torch


def unigram_entropy(ids: list[int]) -> float:
    if not ids:
        return 0.0
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values())


def rep_n(ids: list[int], n: int) -> float:
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def distinct_n(samples: list[list[int]], n: int) -> float:
    grams = [tuple(s[i:i + n]) for s in samples for i in range(len(s) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


_WORD = re.compile(r"[a-z']+")


def word_dups(texts: list[str]) -> dict:
    """Repeated words at distance 1 and 2, per 1000 words (pooled over texts)."""
    n = d1 = d2 = 0
    for t in texts:
        w = _WORD.findall(t.lower())
        n += len(w)
        d1 += sum(w[i] == w[i - 1] for i in range(1, len(w)))
        d2 += sum(w[i] == w[i - 2] for i in range(2, len(w)))
    return {"dup_adj1_per_1k": 1000 * d1 / max(n, 1), "dup_adj2_per_1k": 1000 * d2 / max(n, 1)}


def amp_dtype(device: str):
    """bf16 where it is native (CPU, CUDA capability >= 8), fp16 on older GPUs (T4)."""
    if device == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        return torch.float16
    return torch.bfloat16


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


_LM_CACHE: dict = {}


def _load_lm(name: str, device: str):
    if name not in _LM_CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok_name = "EleutherAI/gpt-neo-125M" if name.startswith("roneneldan/TinyStories") else name
        tok = AutoTokenizer.from_pretrained(tok_name)
        lm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
        _LM_CACHE[name] = (tok, lm)
    return _LM_CACHE[name]


def gen_ppl(texts: list[str], scorer: str, device: str | None = None, batch_size: int = 16) -> float:
    device = device or default_device()
    from ncmr import ncm_ref_loader  # noqa: F401  (puts upstream code on sys.path)
    from evaluation.gen_ppl import generation_perplexity  # upstream NCM implementation
    tok, lm = _load_lm(scorer, device)
    with torch.no_grad():
        r = generation_perplexity(lm, tok, [""] * len(texts), texts, device=device, batch_size=batch_size)
    return float(r.mean_ppl)


@torch.no_grad()
def mauve_features(texts: list[str], device: str | None = None, max_len: int = 256, batch_size: int = 16) -> np.ndarray:
    """gpt2-large final-layer hidden state at the last real token (MAUVE's default featurization).
    Right-padded batches (GPT-2 positions are absolute and attention is causal, so the last real token's
    state is unaffected by padding); bf16 autocast. Every method and the human references go through this
    same function, so the featurization is identical across comparisons."""
    device = device or default_device()
    tok, lm = _load_lm("gpt2-large", device)
    feats = []
    for s in range(0, len(texts), batch_size):
        batch = [t if t.strip() else tok.eos_token for t in texts[s:s + batch_size]]
        enc = [tok(t, truncation=True, max_length=max_len)["input_ids"] for t in batch]
        T = max(len(e) for e in enc)
        ids = torch.full((len(enc), T), tok.eos_token_id, dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e)
        with torch.autocast(device_type=device, dtype=amp_dtype(device)):
            hs = lm(ids.to(device), output_hidden_states=True).hidden_states[-1]
        last = torch.tensor([len(e) - 1 for e in enc], device=hs.device)
        feats.append(hs[torch.arange(len(enc), device=hs.device), last].float().cpu().numpy())
    return np.concatenate(feats)


def mauve_score(p_feats: np.ndarray, q_feats: np.ndarray, seed: int = 0) -> float:
    import mauve
    out = mauve.compute_mauve(p_features=p_feats, q_features=q_feats, seed=seed, verbose=False)
    return float(out.mauve)


def evaluate_samples(sample_ids: list[list[int]], decode, groups: list[int] | None = None,
                     scorers=("gpt2-large", "roneneldan/TinyStories-33M"), ref_feats: np.ndarray | None = None,
                     device: str | None = None) -> dict:
    """sample_ids: BPE ids per sample, already cut at EOS. decode: ids -> str.
    groups: seed index per sample; genppl/MAUVE are computed per group and averaged (mean, sd)."""
    device = device or default_device()
    texts = [decode(s) for s in sample_ids]
    groups = groups if groups is not None else [0] * len(texts)
    res = {"n": len(texts),
           "entropy": float(np.mean([unigram_entropy(s) for s in sample_ids if s])),
           "len_tokens": float(np.mean([len(s) for s in sample_ids])), **word_dups(texts)}
    for n in (2, 3, 4):
        res[f"rep_{n}"] = float(np.mean([rep_n(s, n) for s in sample_ids]))
    for n in (2, 3):
        res[f"distinct_{n}"] = distinct_n(sample_ids, n)
    gs = sorted(set(groups))
    for sc in scorers:
        vals = [gen_ppl([t for t, g in zip(texts, groups) if g == gg], sc, device) for gg in gs]
        key = "genppl_" + sc.split("/")[-1]
        res[key] = float(np.mean(vals))
        res[key + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        res[key + "_per_group"] = vals
    if ref_feats is not None:
        # MAUVE is noisy at a few hundred samples, so pool all samples against an equal-size human set
        # and report the spread over 5 quantization seeds.
        feats = mauve_features(texts, device)
        q = ref_feats[: len(feats)]
        vals = [mauve_score(feats, q, seed=s) for s in range(5)]
        res["mauve"] = float(np.mean(vals))
        res["mauve_sd"] = float(np.std(vals, ddof=1))
        res["mauve_per_seed"] = vals
        res["mauve_n"] = [len(feats), len(q)]
    return res
