"""Audit 12: independent R062 arithmetic and saved-BPE duplicate counts.

The arithmetic uses standard-library JSON/statistics, never analyze_w2c.
Result digests for seeds 1-3 are shared with test_audit12_readings. Optional
entropy replay uses the two released BPE assets, checked against run manifests.
"""
from collections import Counter
import hashlib
import json
import math
import statistics as st

import pytest

from conftest import ROOT
from test_audit12_readings import pinned_result


SAMPLE_PINS = {
    (2, 0): "f06475e0415931d96ce649c453e99147eed466ff5e5e37a5d9aaf7a7cd786652",
    (3, 0): "09116dd5e652754e926f534877438c02382ba7cdf644964fc83bf064a7042093",
    (2, 1): "7286c7cd1da063c6b726543b7ff32e6652ee8ab52c279340e20ee023b1a31356",
    (3, 1): "2d70433b6a51fd4d9d3937ffec6f088ae4a3bb958aa2b3a32f9c3ed143a6cf2c",
    (2, 2): "bece55767234a2f10a4d94b5daa6b57d651e2911f6a9e5bdfb5f4b91e92d8c07",
    (3, 2): "519fcb0b42d6748bf8d3d29c83a8907893393f525accc23413cebe631635adb2",
    (2, 3): "41354d19ad549ec361e6588b9691be498c3d8642771ae57062c269137de08e1a",
    (3, 3): "43871b7b937387b6e9fe38a0eb9e26c88b9934b8198b670da6114fea8fb4b98b",
}
PROVENANCE_PINS = {
    (2, 1): "1f8a83be1a08e3bff48bd53e9c1c320aa8a69a70d346d07cb01d19b1bef6b4e1",
    (3, 1): "3613cbb67540ecc60d431968a79cbcda393ee77f4460cf4aab6891cd9f7549e0",
    (2, 2): "a74f1fe86a8048296b8e3c20cfc44ab3a69486f5ebb9178187dc8ec428ad9e0c",
    (3, 2): "71b56a3f0f2f626f65741e6a73bad04cfa8a232690ca34768f4bfa69840b582d",
    (2, 3): "0e2d235bcf85a620a268ddf142696665a4405e91f9e227a97a6a9421681b734a",
    (3, 3): "212496f0b875af9ef2af7028aaf6718d11f438f3beb782f44b23379dfb18ebf9",
}
ZERO_PINS = {
    2: "e574efb0b38ee62a5974686f82556a3eff028f5967820dbd8d64ad74587aefd7",
    3: "afd280811fca660f1df3abbe9910b92c91c73869008410bebe2c2149f2e3baae",
}


def path(m, seed, suffix):
    wave = "w2c1" if seed else "w2m"
    cfg = "frozen_pw" if m == 2 else "uniform_wrong"
    return ROOT / f"results/colab/{wave}_L_ts_m{m}/L_ts_m{m}_{cfg}_s{seed}__{suffix}"


def read_pinned(p, digest):
    raw = p.read_bytes()
    if p.suffix != ".pickle":
        raw = raw.replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == digest, p.name
    return raw


def result(m, seed):
    if seed:
        return pinned_result(f"m{m}", seed)
    return json.loads(read_pinned(path(m, seed, "result.json"), ZERO_PINS[m]))


def samples(m, seed):
    return json.loads(read_pinned(path(m, seed, "samples_final.json"), SAMPLE_PINS[m, seed]))


def adjacent_duplicates(ids):
    # w1_sample_stats.scores truncates at W=128, excludes the first position,
    # and pools valid adjacent pairs across sequences. No word tokenization,
    # re-encoding, cross-sample pair, or second EOS/PAD normalization is used.
    rows = [r[:128] for r in ids]
    pairs = sum(max(0, len(r) - 1) for r in rows)
    count = sum(sum(a == b for a, b in zip(r, r[1:])) for r in rows)
    return count, pairs, 1000 * count / pairs


def summarize(seed):
    a, b = [result(m, seed)["final_eval"] for m in (2, 3)]
    out = {}
    for scorer in ("gpt2-large", "TinyStories-33M"):
        k = "genppl_" + scorer
        x, y = a[k + "_per_group"], b[k + "_per_group"]
        assert len(x) == len(y) == 16
        d = [v - w for v, w in zip(x, y)]
        out[scorer] = [st.mean(x), st.mean(y), st.mean(d), st.stdev(d) / 4, sum(v > 0 for v in d)]
    for k in ("val_nll", "mauve", "entropy", "len_tokens"):
        out[k] = [a[k], b[k], a[k] - b[k]]
    out["level_differences"] = [x - y for x, y in zip(a["val_nll_per_level"], b["val_nll_per_level"])]
    out["duplicates"] = [adjacent_duplicates(samples(m, seed)["ids"]) for m in (2, 3)]
    out["duplicate_difference"] = out["duplicates"][0][2] - out["duplicates"][1][2]
    return out


@pytest.mark.parametrize("seed,g,se,t,tse,dnll", [
    (0, 20.936264514681724, .37397848490525193, 33.08414480164973, .7190398005934676, 4.0353515624999545),
    (1, 18.364928244879852, .3246654696388909, 28.54637069551514, .4983916599295046, 3.8617656249999754),
    (2, 19.505983480903076, .30563694905005995, 31.101490537443638, .5289399314021034, 3.9097031249999645),
    (3, 18.75725549630717, .32042041047383507, 29.78307006860512, .5738730457931092, 4.322515624999937),
])
def test_r062_per_seed_arithmetic(seed, g, se, t, tse, dnll):
    row = summarize(seed)
    assert row["gpt2-large"][2:] == pytest.approx([g, se, 16], abs=1e-12)
    assert row["TinyStories-33M"][2:] == pytest.approx([t, tse, 16], abs=1e-12)
    assert row["val_nll"][2] == pytest.approx(dnll, abs=1e-12)
    # Each stored level is rounded to .01; summing those differences is not
    # an exact reconstruction of the independently accumulated total.
    assert abs(sum(row["level_differences"]) - dnll) <= .11


@pytest.mark.parametrize("m,seed", sorted(SAMPLE_PINS))
def test_registered_duplicates_from_saved_ids(m, seed):
    data = samples(m, seed)
    assert len(data["ids"]) == 4096
    assert all(0 not in r and 1 not in r and len(r) <= 128 for r in data["ids"])
    name = f"M{m}_s{seed}" if seed else f"M{m}"
    stats = json.loads((ROOT / f"results/{'w2c1' if seed else 'w2m'}_sample_stats.json").read_text())
    _, _, rate = adjacent_duplicates(data["ids"])
    assert rate == pytest.approx(stats["sets"][name]["adjacent_dup_per_1k"], abs=1e-12)
    assert st.mean(map(len, data["ids"])) == stats["sets"][name]["mean_len"]


def test_r062_aggregate_arithmetic():
    rows = [summarize(s) for s in (1, 2, 3)]
    sp = math.sqrt(sum(st.variance(r["gpt2-large"][arm] for r in rows) for arm in (0, 1)) / 2)
    assert sp == pytest.approx(.5582127099528249, abs=1e-12)
    assert 2 * sp == pytest.approx(1.1164254199056498, abs=1e-12)
    assert st.mean(r["gpt2-large"][2] for r in rows) == pytest.approx(18.8760557406967)
    assert st.mean(r["TinyStories-33M"][2] for r in rows) == pytest.approx(29.81031043385464)
    assert st.mean(r["val_nll"][2] for r in rows) == pytest.approx(4.031328124999959)
    assert st.stdev(r["val_nll"][2] for r in rows) == pytest.approx(.2533122995311128)
    assert st.mean(r["mauve"][2] for r in rows) == pytest.approx(-.40794179970048217)
    assert st.mean(r["entropy"][2] for r in rows) == pytest.approx(.027078075682905805)
    assert st.mean(r["duplicate_difference"] for r in rows) == pytest.approx(12.291083659833248)


def test_r062_provenance_minutes():
    values = {}
    for m, seed in PROVENANCE_PINS:
        raw = read_pinned(path(m, seed, "provenance.jsonl"), PROVENANCE_PINS[m, seed])
        events = [json.loads(line) for line in raw.splitlines()]
        assert [e["event"] for e in events] == ["start", "end"]
        assert events[0]["had_ckpt"] is False and events[1]["exit"] == 0
        assert events[1]["last_logged_step"] == 600
        values[m, seed] = events[1]["minutes"]
    assert [values[2, s] for s in (1, 2, 3)] == [51.85, 50.11, 50.10]
    assert [values[3, s] for s in (1, 2, 3)] == [30.32, 28.32, 28.33]
    assert sum(values.values()) == pytest.approx(239.03)


@pytest.fixture(scope="module")
def released_bpe():
    from ncmr.ncm_ref_loader import BPETokenizer, safe_pickle_load

    names = ("bpe_vocab_8192.pickle", "bpe_merges_8192.pickle")
    pins = json.loads(path(2, 1, "run_manifest.json").read_text())["base_sha256"]
    for directory in (ROOT / "data/tinystories", ROOT / "tests/_assets/tinystories"):
        if all((directory / n).is_file() for n in names):
            for name in names:
                read_pinned(directory / name, pins[name])
            vocab, merges = [safe_pickle_load(directory / n) for n in names]
            return BPETokenizer(vocab, merges, {"eos": "<|endoftext|>", "pad": "<|pad|>"})
    pytest.skip("Released BPE assets absent; independent saved-ID duplicates still run")


@pytest.mark.parametrize("m,seed", sorted(SAMPLE_PINS))
def test_entropy_reencoding_from_pinned_bpe(m, seed, released_bpe):
    reencoded = [released_bpe.encode(released_bpe.decode(r)) for r in samples(m, seed)["ids"]]
    entropies = [-sum((v / len(r)) * math.log(v / len(r)) for v in Counter(r).values()) for r in reencoded if r]
    fe = result(m, seed)["final_eval"]
    assert st.mean(entropies) == pytest.approx(fe["entropy"], abs=1e-12)
    assert st.mean(map(len, reencoded)) == fe["len_tokens"]


if __name__ == "__main__":
    print(json.dumps({s: summarize(s) for s in (0, 1, 2, 3)}, indent=2))
