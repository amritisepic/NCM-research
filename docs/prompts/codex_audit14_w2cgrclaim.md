<!-- Run with `scripts/review/codex_local.sh audit audit14 docs/prompts/codex_audit14_w2cgrclaim.md` from a clean
checkout of the audited commit. Codex works on branch `codex/audit14` and commits only the audit doc and its tests. -->
Deep audit 14 before any criterion-(c) claim rests on W2C stage 1 together with W2M-GR (registered "W2C" and "W2M-GR"
in `docs/preregistration.md`). This is a claim audit: the code was audited in 12 (W2C) and 13 (W2M-GR's delta); this
audit checks R065's numbers and the combined wording. Follow "Deep audit protocol" and "Code Review Rules" in AGENTS.md.
Push only `docs/reviews/codex_audit14.md`, `docs/reviews/codex_audit14_numbers.json` (optional) and
`tests/test_audit14_*.py`; change no other file. If your sandbox cannot run the repository's Python, say so in the audit
doc and still write the tests; they will be run for you. The recomputation needs only `json`, `math` and `statistics`.

## Why
R062 (audited in 12, R063) found that M2 (wrong codes drawn from the frozen released model) damages generation relative
to M3 (uniform-wrong codes in the same slots) at seeds 1-3, while canonical teacher-forced NLL barely moves. R065 reports
that W2M-GR's registered reading "localization replicates" holds at seeds 1-3: in a crossed replay of whole level
ranges, M2's prefix through L1 carries ~2% of the M2 - M3 log-loss contrast, M2's prefix through L5 ~53%, M2's levels
6-11 alone ~5%, with an interaction of ~0.42-0.44 at cuts 4 and 6. Together these are believed to meet criterion (c) of
`docs/brief.md` ("a finding about what the levels encode"). No one has independently recomputed R065's numbers from the
raw files under audit, and no one has judged whether the combined wording says more than a crossed replay can show.

## Subject
The audited commit is HEAD of `claude/ncm-research`. Files: `scripts/analyze_w2mgr.py`, `src/ncmr/w2mg.py` (`shares`,
`_shares`, `read_i`, `read_ii`, the (i) ratio and (iii) root-reliance code), `scripts/w2mg_eval.py` (what each cell
samples: `cell_specs`, `protocol_block`, the sanity reference), `notebooks/w2mgr_eval.ipynb`. Existing tests:
`tests/test_w2mg.py`, `tests/test_w2mgr.py`, `tests/test_audit13_*.py`, `tests/test_audit12_*.py`.
Inputs: `results/colab/w2mgr_eval/w2mgr_eval_s{1,2,3}__result.json`, their `__provenance.jsonl` and `_meta.json`;
seed 0 (exploratory, never pooled): `results/colab/w2mg_eval/`; W2C's references: `results/colab/w2c1_L_ts_m2/`,
`results/colab/w2c1_L_ts_m3/`, `results/w2c_analysis.json`. Output under audit: `results/w2mgr_analysis.json`.
Log entries whose numbers to recompute: **R065** (every number, including the seed-0 row it quotes from R058).
Wording under audit: R065's "Status" paragraph; R063's claim wording (the W2C half); section 4 of
`docs/prompts/gpt6_astra_round4_2026-10-28.md` (the combined candidate and its stated limits); and the alternative
wording an external reviewer proposed (quoted in question 6).

## Spec (what correct means)
The W2M-GR registration (last section of `docs/preregistration.md`) is the spec; W2M-G's protocol (registered "W2M-G")
applies unchanged. What it leaves implicit:
- **Cells.** A cell (prefix P, suffix S, cut c) samples levels 1..c-1 with P and levels c..11 with S (the result's
  `protocol.cells.cut_meaning`), ancestrally at T = 1 with common random numbers (one Gumbel generator per group, seeded
  by the group seed), 8 groups x 256 samples. Say exactly what level 0 (the root) is in every cell and whether it is
  shared across cells; the wording "prefix through L1" depends on it.
- **Shares** per cut c from the cells' mean GPT-2 large log loss over the 8 groups: delta = l(M2,M2) - l(M3,M3);
  pi_c = (l(M2->M3, c) - l(M3,M3)) / delta; sigma_c = (l(M3->M2, c) - l(M3,M3)) / delta; interaction = 1 - pi_c -
  sigma_c; jackknife SE over groups. TinyStories-33M beside. Log loss = log Gen-PPL per group.
- **Readings** (seeds 1-3 only; seed 0 never pooled): as registered. Seed SD = sample SD (ddof 1) over seeds 1-3.
- **(i):** ratio r_b of M2-M3 divergence on generated bank b to that on the clean (val) bank, bootstrap CI; replicates
  iff every r_b >= 3 with every lower bound > 1.2 in every seed. **(iii):** rho(M3_s) - rho(M2_s) with paired 95% CI;
  replicates iff every CI lies above 0. **Sanity:** pure cells' GPT-2 large Gen-PPL within 3 SE of W2C's final-eval
  G(M2_s), G(M3_s); say which SE the code uses and whether that is the right one.

## Questions
1. **Recompute every R065 number** from the raw `*__result.json` files with your own code, not the repository's:
   per seed and scorer pi_2, pi_4, pi_6, sigma_6 (and sigma_2, sigma_4), the interactions at cuts 2, 4, 6, their
   jackknife SEs; seed means and SDs; the TinyStories-33M seed means; the margins R065 quotes ("pi_2 <= 0.025",
   "pi_6 >= 0.517", "sigma_6 <= 0.067", "interactions >= 0.416"); (i)'s ratios (6.2-7.0x, 5.4-5.5x), J_clean
   (0.0120-0.0126 nats) and the minimum lower bound (5.16); (iii)'s rho values and CIs (1.38 / 1.40 / 1.38; released
   3.04, M2 1.99-2.05, M3 3.38-3.43); the sanity z values; W2M-G's own (ii) reading per seed; the run times (18.9 /
   18.5 / 18.1 L4-min) from provenance; the seed-0 row from `results/colab/w2mg_eval/`. List every disagreement at
   R065's printed precision. Check that `results/w2mgr_analysis.json` equals what `scripts/analyze_w2mgr.py` produces
   from the committed raw files.
2. **The analysis matches the registration.** Readings, thresholds, inclusive/strict bounds, the TinyStories-33M seed
   mean condition, the overturn rule, seeds 1-3 only, seed 0 never pooled, the 8-group admission rule (A13-1's fix).
   Audit 13 tested the code at `a949044`; confirm the merged code that produced `results/w2mgr_analysis.json` is the
   fixed version, and that the three result files pass its admission checks for the right reasons (identity hashes:
   the M2_s / M3_s checkpoint SHA-256 in each result equal to the checkpoints W2C's runs wrote, where the repository
   records them; same released assets and val hash as W2M-G seed 0; L4; replay seeds 0-7; bank seeds 900, 901).
3. **What a crossed replay of whole level ranges can show.** The candidate wording says "the damage forms at levels
   2-5" and "is amplified by the damaged model's own later levels, which alone carry ~5%". Judge each phrase:
   - The design contains only prefix/suffix splits at cuts 2, 4 and 6; no single-level replay and no cut at 3 or 5.
     Is "levels 2-5" supported, or only "levels 2-5 taken together"? Does pi_6 - pi_2 attribute damage to levels 2-5
     given a 0.42 interaction (the shares are not additive), and given that in a cell M2's levels 2-5 are sampled
     conditioned on M2's own level 1? What about sigma_2 (~0.88): M2's levels 2-11 on M3's L1?
   - "Forms" suggests a time/causal order of generation. Is that justified, or should it be "a prefix sampled by M2
     through L5, rendered by M3, carries ~53% of the contrast"? Does an M3 suffix rendering an M2 prefix measure M2's
     damage, or M3's response to out-of-distribution states (M3 never trained on M2's prefixes)?
   - "Amplified by the damaged model's own later levels": interaction = 1 - pi - sigma > 0 means M2's suffix does more
     harm on M2's prefix than on M3's. Is "amplified" the right word, or "M2's later levels interact with M2's earlier
     levels"? Is there any reading of these numbers in which the interaction reflects co-adaptation within one
     checkpoint (a same-arm cross-seed control, prefix from M2_s and suffix from M2_s', was not run), and what must the
     wording concede because of it?
   - The share denominators use GPT-2 large log loss of decoded text. Does "levels 2-5 carry the damage" survive the
     fact that the shares are of a scorer's log loss, not of a quantity defined on codes? TinyStories-33M agrees; is
     that enough?
   - Is this "a finding about what the levels encode" (criterion (c)), or a finding about where in the hierarchy two
     differently trained checkpoints diverge in a way that matters for generation? Propose the strongest wording the
     evidence supports and say what it must not be read as.
4. **Consistency between the two halves.** W2C's arms are evaluated at their final checkpoint by `train_gen.py`'s final
   evaluation (16 groups x 256, sample seeds 42 and 0-14) on an A100; W2M-GR re-samples the same checkpoints on an L4
   with replay seeds 0-7. Do the pure cells' Gen-PPLs agree with W2C's well enough that the two halves describe the
   same models and the same contrast (delta in log-loss units versus D_s in Gen-PPL)? Is quoting "~53% of the
   log-loss contrast" next to "+18.9 Gen-PPL" misleading in any way?
5. **Alternatives the combined claim must survive or concede.** At least: (a) no frequency-, temperature- or
   length-matched control (audit 12); (b) M2's training includes a proposal forward M3 lacks (unequal compute);
   (c) a marginal-preserving control (same multiset of plausible wrong codes, randomized association to examples) has
   not been run, so "model-specific plausibility" versus "any example-conditioned non-uniform wrong distribution" is
   open; (d) one replay instrument; (e) clean rows (M3 is ~5.5 Gen-PPL above the released model, R059). For each, say
   whether the W2C + W2M-GR data can rule it out, and what the claim must not say.
6. **Judge these wordings** against the code, registration and numbers; for each, say what is supported, what overstates,
   and propose a corrected version:
   - (A, R065 / Astra prompt section 4) "under one continuation protocol, plausible-wrong replacement causes additional
     generation degradation relative to uniform-wrong replacement in the same slots, which canonical teacher-forced NLL
     barely registers; the damage forms at levels 2-5 (M2's prefix through L5 carries ~53% of the log-loss contrast,
     through L1 ~2%) and is amplified by the damaged model's own later levels (interaction ~0.42-0.44), which alone
     carry ~5%."
   - (B, external reviewer) "under this continuation recipe, frozen-proposer wrong replacements worsen quality versus
     same-slot uniform-wrong replacements; crossed replay localizes a middle-level prefix contrast and a suffix
     interaction."
   Also check the limits list in section 4 of the Astra prompt for errors or omissions (e.g. "5.5 Gen-PPL above the
   released model": recompute it from R059 / R062).

## Out of scope
Style, naming, performance; the C-E1 claim; M0, M1, M4-M7, M3-0 and L1R except where a question needs them; W2M-G's
(iv); code defects already triaged in audits 12-13 unless they change a number above.
