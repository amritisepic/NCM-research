"""Audit 14: conditional replay contrasts and the candidate claim's scope.

The numeric witnesses use independently calculated means of group log PPL,
not the repository's stored shares or analysis functions.  The document
guards are intentionally narrow: they cover the proposed public claim, not
the registered name of a reading or historical exploratory descriptions.
The unchanged tests/conftest.py supplies the CPU test setup.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st

import pytest


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    1: "46515b911aa803bf1df07dd2ff6fdee0fbe86f9cefebe8ed23850381b644c32c",
    2: "31555fa891fd9eec5ca39aa7799dc561ae92ac841e948c5852b3c9f5c2776ce4",
    3: "daf61e475b3a129f501f6dd7eafd4b3ac757dda58c2afd7262274a23d450de93",
}


def result(seed):
    relative = f"results/colab/w2mgr_eval/w2mgr_eval_s{seed}__result.json"
    raw = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == PINS[seed], relative
    return json.loads(raw)


def losses(seed, scorer):
    cells = result(seed)["cells"]
    return {
        key: st.mean(math.log(x) for x in cell[f"genppl_{scorer}"]["per_group"])
        for key, cell in cells.items()
    }


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("scorer", ["gpt2-large", "TinyStories-33M"])
def test_joint_middle_block_effect_depends_on_surrounding_samplers(seed, scorer):
    """L2-5 has two different conditional effects in the measured cells.

    First hold M2 L1 / M3 L6-11; then hold M3 L1 / M2 L6-11.
    Neither contrast is an unconditional allocation to individual levels.
    """
    ll = losses(seed, scorer)
    delta = ll["m2_m2"] - ll["m3_m3"]
    middle_on_m2_l1_m3_late = (ll["m2to3_c6"] - ll["m2to3_c2"]) / delta
    middle_on_m3_l1_m2_late = (ll["m3to2_c2"] - ll["m3to2_c6"]) / delta
    assert 0.45 < middle_on_m2_l1_m3_late < 0.55
    assert 0.80 < middle_on_m3_l1_m2_late < 0.88
    assert middle_on_m3_l1_m2_late - middle_on_m2_l1_m3_late > 0.28


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_small_l1_prefix_share_is_not_zero_l1_effect_with_m2_suffix(seed):
    ll = losses(seed, "gpt2-large")
    delta = ll["m2_m2"] - ll["m3_m3"]
    l1_effect_on_m3_suffix = (ll["m2to3_c2"] - ll["m3_m3"]) / delta
    l1_effect_on_m2_suffix = (ll["m2_m2"] - ll["m3to2_c2"]) / delta
    assert 0 < l1_effect_on_m3_suffix < 0.03
    assert 0.11 < l1_effect_on_m2_suffix < 0.13
    assert l1_effect_on_m2_suffix - l1_effect_on_m3_suffix > 0.09


def test_cut6_interaction_is_difference_of_conditional_late_effects():
    late_on_m3, late_on_m2, interaction = [], [], []
    for seed in (1, 2, 3):
        ll = losses(seed, "gpt2-large")
        delta = ll["m2_m2"] - ll["m3_m3"]
        on_m3 = (ll["m3to2_c6"] - ll["m3_m3"]) / delta
        on_m2 = (ll["m2_m2"] - ll["m2to3_c6"]) / delta
        late_on_m3.append(on_m3)
        late_on_m2.append(on_m2)
        interaction.append(on_m2 - on_m3)
    assert st.mean(late_on_m3) == pytest.approx(0.05320560198902811)
    assert st.mean(late_on_m2) == pytest.approx(0.4727382156123574)
    assert st.mean(interaction) == pytest.approx(0.41953261362332933)


def effective_r065_claim():
    """Permit an append-only correction without requiring historical edits.

    A later log entry must expressly identify R065's claim/wording and say
    it corrects or supersedes it; an unrelated audit mention is insufficient.
    """
    log = (ROOT / "log.md").read_text(encoding="utf-8")
    entries = re.split(r"(?m)(?=^## R\d+\b)", log)
    original = next(entry for entry in entries if entry.startswith("## R065"))
    effective = original
    for entry in entries[entries.index(original) + 1:]:
        identifies = re.search(r"R065.{0,100}(?:claim|wording)|(?:claim|wording).{0,100}R065", entry, re.I | re.S)
        corrects = re.search(r"correct(?:ion|ed|s)?|supersed(?:e|es|ing)|replace(?:ment|s)?", entry, re.I)
        if identifies and corrects:
            effective = entry
    if effective == original:
        return original.split("- **Status:", 1)[1].split("- Every seed", 1)[0]
    return effective


def astra_candidate():
    prompt = (ROOT / "docs/prompts/gpt6_astra_round4_2026-10-28.md").read_text(encoding="utf-8")
    section = re.search(r"(?ms)^## 4\..*?(?=^## 5\.)", prompt).group()
    return section.split("**C-W2C + W2M-GR**", 1)[1]


def claim_scope_gaps(text):
    """Reviewable semantic guard, accepting alternative corrected prose.

    This does not infer scientific validity from keywords.  The positive
    numeric witnesses above establish why each requested qualification is
    needed; this guard records that the two actual candidate texts omit it.
    """
    flat = " ".join(text.lower().split())
    requirements = {
        "conditional replay interpretation": r"conditional|render(?:ed|ing)? by m3|m3(?:'s)? suffix|on m3(?:'s)? prefix",
        "scorer-defined output quantity": r"log.loss|scorer.defined|decoded.text loss",
        "foreign-prefix response remains open": r"foreign.prefix|out.of.distribution|prefix distribution shift|cross.arm prefix",
        "checkpoint compatibility remains open": r"compatib|co.adapt",
        "semantic attribution is not established": (
            r"(?:not|cannot|does not|no).{0,100}(?:semantic|what.{0,20}levels.{0,15}encode)"
            r"|(?:semantic|what.{0,20}levels.{0,15}encode).{0,100}(?:not established|unproven|unsupported|not shown)"
        ),
    }
    return [name for name, pattern in requirements.items() if not re.search(pattern, flat)]


@pytest.mark.parametrize("source", ["R065", "Astra section 4"])
def test_candidate_claim_states_the_identification_limits(source):
    text = effective_r065_claim() if source == "R065" else astra_candidate()
    assert claim_scope_gaps(text) == [], f"{source}: {claim_scope_gaps(text)}"
