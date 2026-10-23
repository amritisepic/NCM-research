"""Run the full text evaluation suite on a samples file (from sample.py / released_generator_eval.py).

  python scripts/eval_samples.py --samples runs/samples/ncm_s.json --ref val --out results/<name>_eval.json

Reference human text for MAUVE: the first N windows of the chosen split (`val` while developing, `test`
only for final numbers), cut at EOS, N = number of samples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.data import cut_at_eos, examples  # noqa: E402
from ncmr.eval_text import evaluate_samples, mauve_features  # noqa: E402
from ncmr.ncm_ref_loader import safe_pickle_load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True)
ap.add_argument("--ref", default="val", choices=["val", "test"])
ap.add_argument("--out", required=True)
ap.add_argument("--no_mauve", action="store_true")
a = ap.parse_args()

from tokenization.bpe_tokenizer import BPETokenizer  # noqa: E402  (upstream; path set by ncmr imports)
D = os.path.join(ROOT, "data", "tinystories")
tok = BPETokenizer(safe_pickle_load(f"{D}/bpe_vocab_8192.pickle"), safe_pickle_load(f"{D}/bpe_merges_8192.pickle"),
                   {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"})
S = json.load(open(a.samples))
t0 = time.time()
ref_feats = None
if not a.no_mauve:
    n = len(S["ids"])
    cache = os.path.join(ROOT, "runs", "cache", f"mauve_ref_{a.ref}_{n}.npy")
    if os.path.exists(cache):
        ref_feats = np.load(cache)
    else:
        refs = [tok.decode(cut_at_eos(r)) for r in examples(a.ref, n)]
        ref_feats = mauve_features(refs)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, ref_feats)
# As in upstream eval_merge.py: token statistics are computed on the BPE re-encoding of the decoded text.
reenc = [tok.encode(tok.decode(ids)) for ids in S["ids"]]
res = evaluate_samples(reenc, tok.decode, S["groups"], ref_feats=ref_feats)
res.update({"samples_file": os.path.relpath(a.samples, ROOT), "ref_split": a.ref,
            "nfe_per_sample": S.get("nfe_per_sample"), "gen_wall_s": S.get("gen_wall_s"),
            "eval_wall_s": time.time() - t0})
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.endswith(("per_group", "per_seed"))}, indent=1))
