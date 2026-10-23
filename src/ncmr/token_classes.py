"""Assign every BPE position of a TinyStories example to a linguistic class, for per-class metrics.

Classes (from spaCy en_core_web_sm POS/NER on the decoded text, aligned to BPE tokens by the first
non-space character of each token):
  function  : DET, ADP, PRON, CCONJ, SCONJ, AUX, PART   (closed-class words)
  noun      : NOUN
  name      : PROPN, or any token inside a PERSON entity (character names: Lily, Tom, ...)
  numeral   : NUM
  verb      : VERB
  modifier  : ADJ, ADV
  punct     : PUNCT
  space     : whitespace-only tokens (newlines)
  other     : INTJ, SYM, X, ...
  special   : EOS / PAD (excluded from all statistics)
Orthogonal flag `rare`: the word containing the token is outside the most frequent word types that
together cover 98% of word occurrences in a 200k-story sample of the training stream (i.e. the
bottom 2% of word mass). Word = lower-cased regex \\w+ match.
"""
from __future__ import annotations

import collections
import json
import os
import re

import numpy as np

FUNCTION_POS = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART"}
CLASSES = ["function", "noun", "name", "numeral", "verb", "modifier", "punct", "space", "other"]
WORD_RE = re.compile(r"\w+", re.UNICODE)


def token_strings(tok) -> list[str]:
    """Surface string of every BPE id (bytes decoded leniently; TinyStories is ~ASCII)."""
    return [tok.vocab[i].decode("utf-8", errors="replace") for i in range(len(tok.vocab))]


def load_spacy():
    import warnings
    warnings.filterwarnings("ignore", message=".*W095.*")
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])


def rare_vocab(tok, train_stream, n_stories: int = 200_000, mass: float = 0.98, seed: int = 0,
               cache: str | None = None) -> set[str]:
    """Word types in the top `mass` of occurrence mass; everything else is 'rare'."""
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return set(json.load(f)["common"])
    strs = token_strings(tok)
    rng = np.random.default_rng(seed)
    idx = rng.choice(train_stream.n_stories, size=n_stories, replace=False)
    cnt = collections.Counter()
    for i in idx:
        ids = train_stream.story(int(i), max_len=10_000)
        cnt.update(w.lower() for w in WORD_RE.findall("".join(strs[t] for t in ids if t > 1)))
    total = sum(cnt.values())
    common, acc = [], 0
    for w, c in cnt.most_common():
        if acc / total >= mass:
            break
        common.append(w)
        acc += c
    if cache:
        with open(cache, "w") as f:
            json.dump({"mass": mass, "n_stories": n_stories, "n_types_total": len(cnt),
                       "n_common": len(common), "common": common}, f)
    return set(common)


def classify(examples: np.ndarray, tok, nlp, common_words: set[str], specials=(0, 1)):
    """Return (cls[N, L] int8 index into CLASSES or -1 for special, rare[N, L] bool)."""
    strs = token_strings(tok)
    N, L = examples.shape
    cls = np.full((N, L), -1, dtype=np.int8)
    rare = np.zeros((N, L), dtype=bool)
    texts, spans = [], []
    for row in examples:
        text, sp = "", []
        for t in row:
            s = "" if t in specials else strs[t]
            sp.append((len(text), len(text) + len(s)))
            text += s
        texts.append(text)
        spans.append(sp)
    for n, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        char2tok = np.full(len(texts[n]) + 1, -1, dtype=np.int32)
        for j, t in enumerate(doc):
            char2tok[t.idx:t.idx + len(t.text)] = j
        for p, (a, b) in enumerate(spans[n]):
            t_id = int(examples[n, p])
            if t_id in specials:
                continue
            seg = texts[n][a:b]
            if seg.strip() == "":
                cls[n, p] = CLASSES.index("space")
                continue
            first = a + (len(seg) - len(seg.lstrip()))
            j = char2tok[first]
            if j < 0:
                cls[n, p] = CLASSES.index("other")
                continue
            st = doc[j]
            if st.pos_ == "PROPN" or st.ent_type_ == "PERSON":
                c = "name"
            elif st.pos_ == "NUM":
                c = "numeral"
            elif st.pos_ in FUNCTION_POS:
                c = "function"
            elif st.pos_ == "NOUN":
                c = "noun"
            elif st.pos_ == "VERB":
                c = "verb"
            elif st.pos_ in ("ADJ", "ADV"):
                c = "modifier"
            elif st.pos_ == "PUNCT":
                c = "punct"
            else:
                c = "other"
            cls[n, p] = CLASSES.index(c)
            words = WORD_RE.findall(st.text.lower())
            rare[n, p] = bool(words) and any(w not in common_words for w in words)
    return cls, rare


def per_class_accuracy(pred: np.ndarray, target: np.ndarray, cls: np.ndarray, rare: np.ndarray) -> dict:
    ok = pred == target
    out = {"all": float(ok[cls >= 0].mean()), "n_all": int((cls >= 0).sum())}
    for i, c in enumerate(CLASSES):
        m = cls == i
        if m.any():
            out[c] = float(ok[m].mean())
            out[f"n_{c}"] = int(m.sum())
    m = rare & (cls >= 0)
    if m.any():
        out["rare"] = float(ok[m].mean())
        out["n_rare"] = int(m.sum())
    return out
