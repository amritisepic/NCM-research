<!-- Paste everything below this comment into Codex (desktop app, or `scripts/review/codex_local.sh audit audit11
docs/prompts/codex_audit11_e1claim.md` from a checkout of the audited commit). Work in a new branch `codex/audit11` made
from the audited commit, and commit only the audit doc and its tests there. -->
Deep audit 11 before any criterion-(c) claim rests on E1 (registered "E1", "E1a" and "E1b" in
`docs/preregistration.md`). Follow "Deep audit protocol" and "Code Review Rules" in AGENTS.md. Push only
`docs/reviews/codex_audit11.md` and `tests/test_audit11_*.py`; change no other file. If your sandbox cannot run the
repository's Python, say so in the audit doc and still write the tests; they will be run for you. The confidence
sequence needs only `math.lgamma`, so a standard-library Python is enough to recompute every bound independently.

## Why
R060 reports that E1's registered (c) replication bar is met: at cut 8 the lower confidence bound of mean R_8 / mean
N_8 is >= 0.8 in all three released models (TinyStories 0.894, WMT14 0.859, Sudoku 0.993), after E1b continued the
Monte Carlo on censored and wide items. That reading is computed by `scripts/analyze_e1.py`, which was Codex-reviewed
(PR 15) but never deep-audited; `scripts/analyze_e1b.py` reuses it (audited in 10 for its run-side code only). A claim
will be shown to the user and must survive hostile scrutiny. This audit decides whether the reading is valid as
computed, and whether the proposed claim (below) says more than the evidence.

## Subject
Files: `scripts/analyze_e1.py` (primary: unchanged since `55eded2`), `scripts/analyze_e1b.py` (its use of
analyze_e1: `readings`, `substitute`, `check_selection`, `analyse`). Reference code they depend on:
`scripts/e1_equivalence.py` (`summarise`, `paired`, `mc_hits`, `measure`, `Domain.key`, `Domain.level_nll`),
`scripts/e1b_continue.py` (`e1_delta`, `select_items`, `interval_width`, `continue_item`), `src/ncmr/cached_decode.py`
(`sample_from_cut`, `PrefixCache`, as far as the confidence sequence's i.i.d. assumption and the exp(-N) floor depend
on them), and `src/ncmr/data.py` (`examples`, the val split). Existing tests: `tests/test_analyze_e1.py`,
`tests/test_e1b.py`, `tests/test_audit10_*.py`, `tests/test_audit9_*.py`.
Inputs: `results/colab/e1_equivalence/e1_{ts,wmt,sdk}__result.json` (+ `_meta.json`, `*__provenance.jsonl`),
`results/colab/e1b_continue/e1b__result.json` (+ provenance). Outputs: `results/e1_analysis.json`,
`results/e1b_analysis.json`. Log entries whose numbers to recompute: R054 and R060 (every number), and the R046
pilot counts as far as they identify the pilot windows.

## Spec (what correct means)
The E1, E1a and E1b registrations (and E1b's erratum) are the spec. What they leave implicit:
- **Per item and cut.** S hits in n samples; the beta-mixture martingale for H0: rho = p with a Beta(1/2, 1/2)
  prior, log M_n(p) = log B(1/2 + S, 1/2 + n - S) - log B(1/2, 1/2) - S log p - (n - S) log(1 - p); the confidence
  set is {p : M_n(p) < 1/delta}, valid at any stopping time of the per-sample filtration (Ville). It is then
  intersected with [exp(-N_k), 1], because the canonical codes c*_{>=k} decode to x^ and the sampler draws them with
  probability exp(-N_k). T in [-log p_U, -log p_L], R in [N_k + log p_L, N_k + log p_U], computed in log space.
- **Aggregation.** delta = 0.05 / (items x registered cuts) per model (TinyStories 64 x 4, WMT14 128 x 3, Sudoku
  128 x 4). On the event that every per-item set covers, mean R lies within the means of the per-item endpoints, and
  N_k is exact, so the share bounds are mean(R endpoints) / mean(N_k), with family-wise error <= 0.05 per model.
- **Readings.** E1's rule: holds if the lower share bound >= 0.8; overturned if the upper bound < 0.5; read on all 64
  TinyStories windows and on the 61 untouched by the R046 pilot (items 0-2 excluded). E1a's added rule: on the 61
  untouched TinyStories windows and on all WMT14 / Sudoku items. A model whose mean N_8 < 5 nats is reported, not read.
- **E1b.** An item's E1b final counts at cut 8 replace its E1 counts; nothing else changes; deltas and rules are E1's.
  The (c) bar is met iff E1's rule holds in all three released models (A0 on both item sets).

## Questions
1. **The confidence sequence.** Is `log_mixture` / `cs_log_interval` the set above, for every (S, n) the files contain
   (S = 0, S = n, n up to 32768, p_L down to exp(-N_k) at the largest N_k in the files)? Do the bisections return the roots to well below 1e-6 nats,
   on the correct side, and can the bracket search or the early `break` ever return a narrower interval? Check against
   an independent implementation (dense grid or high precision), not against analyze_e1 itself.
2. **Validity at the stopping rules used.** E1: rounds of 64, stop at >= 20 hits or 2048 samples. E1b: replay of E1's
   generator, then rounds of 512 until the interval is <= 2 nats wide or 32768 samples; E1b was registered and run
   after E1's aggregate result was seen, and it reuses E1's delta. Does anything break Ville's guarantee or the union
   bound: samples within a round that share noise or are not identically distributed (`sample_from_cut`), a hit test
   that depends on the round, a decision to continue that depends on another item's samples, reading E1 for some items
   and E1b for others, or reusing delta across the two phases? Are the deltas per model the registered ones?
3. **The exp(-N_k) floor.** Is rho_k >= exp(-N_k) true as implemented: the sampler's level distributions equal the
   teacher-forced ones that give `level_nll` (same prefix inputs, canonical codes, weights, T = 1, no masking or
   filtering, positions within a level drawn independently), a deterministic decoder, and `Domain.key(D(c*))` equal
   to the target for every domain (WMT14's EOS / pad rule, Sudoku's board parse)? How large can the fp32 gap between
   the sampler's and the teacher-forced log-probabilities be over the files' N_k, and could it move any reading?
4. **Aggregation and rules.** Are the share bounds, E1's and E1a's rules, the not-read rule, the censored share, the
   2-nat width share and the untouched set (`UNTOUCHED_FROM = 3`) exactly as registered? Were the R046 pilot's three
   windows E1's items 0-2 (`examples("val", n)`; note that a window continues into the next story when its first story
   is shorter than 128 tokens)? Is the conjunction "holds in all three models" a valid level-0.05 statement without a
   further correction (intersection-union)?
5. **Integrity checks.** `check_saved`, `monotonicity` (R_k non-increasing and U_k non-decreasing in k; the "beyond
   the intervals" test) and the empty-intersection report: are the inequalities the right way round, and what defects
   would they miss? Do the saved files' provenance (git SHA, GPU, torch, one complete session per domain) match R054
   and R060, and is the run's code identical to the audited `dcdb890` for E1 and to `5ad74b5` for E1b?
6. **analyze_e1b's reuse.** Does `readings` reproduce analyze_e1's cut-8 reading exactly when no count changes? Does
   `substitute` touch only the released model's continued items at cut 8, refuse duplicates, and keep every other
   count? Can an e1b result with an item outside the selection, a missing selected item not listed in `not_reached`,
   or `complete: false` still produce `e1_replication_bar_met: true`?
7. **Recompute** every number in R054 and R060 from `results/**.json` with your own code: both tables (items,
   censored, mean N_8, share bounds, mean T_8 bounds, widths, E1 / E1a verdicts), the deltas, the integrity
   statements (cut pairs, x^ = x counts), prediction 2's intervals and limits, prediction 3's orders, and the
   exploratory statements (0.842-0.913; >= 90% at cuts 9-11; Sudoku >= 93.8% at cuts 5-11; cut 5 T >= 5.1 nats;
   "20 hits in 448 samples give rho in [0.012, 0.108], 2.2 nats wide"; the 18 censored WMT14 pairs carry 21% of
   N_8, mean 198 vs 131; TinyStories 6 windows, 12%; E1b's 72 items, 39 / 19 stops, replays matched). List every
   disagreement.
8. **The proposed claim.** Does anything in the code, the registrations or the numbers contradict, or fail to support,
   this wording? "In each of the three released NCM models (TinyStories, WMT14, Sudoku-hard), given the canonical
   codes of levels 0-7, at least 80% of the canonical code NLL of levels 8 and finer is spent choosing among code
   paths that decode to the same output (lower 95% confidence bounds on mean R_8 / mean N_8: 0.894, 0.859, 0.993 on
   the registered items)." In particular: is the bound conditional on the registered items (Monte Carlo error only),
   and what may it not be read as (all code NLL; text NLL of free generation; the population of windows)?

## Out of scope
Style, naming, performance tuning; the sampler's equality to teacher-forced logits as already tested in audit 9
(re-check only what question 3 needs); E1b's resume and time-cap logic (audit 10).
