Deep audit 3 before any rerun of on-policy corruption (W1 arms A3 / A4 / A5, Sudoku S1 `onpolicy`). Follow
"Deep audit protocol" and "Code Review Rules" in AGENTS.md. Push only `docs/reviews/codex_audit3.md`,
`docs/reviews/codex_audit3_numbers.json` and `tests/test_audit3_*.py`; change no other file.

## Why
Wave 1 (seed 0, A100, `results/colab/w1_*/`) finished. Every on-policy arm collapsed; the others did not:

| arm | steps | Gen-PPL GPT-2 L (16 x 256) | MAUVE | clean val NLL | entropy |
|---|---|---|---|---|---|
| A0 released (`L_ts_released16`) | 0 | 16.16 | 0.827 | 861.8 | 3.970 |
| A1 control, uniform corruption | 1800 | 16.60 | 0.881 | 844.0 | 3.962 |
| A2 uniform + re-quantized labels | 1800 | 16.29 | 0.915 | 749.3 | 3.963 |
| A6 soft labels | 1800 | 17.46 | 0.832 | 888.9 | 3.958 |
| A3 one-step on-policy | 1350 | 41.44 | 0.496 | 885.7 | 4.026 |
| A4 on-policy + re-quantized | 1350 | 36.81 | 0.673 | 707.4 | 4.037 |
| A5 two-pass on-policy | 1080 | 32.17 | 0.543 | 913.9 | 3.990 |

A3's damage is already there at the first eval: val NLL 912.5 at step 300 (released 861.8), sample Gen-PPL 37.3 at
step 600, flat afterwards. Its metrics log has two step-300 eval rows (912.50 and 913.67), so the run probably
resumed once. R014 measured that the released generator's own one-step samples keep 98% of decoded tokens, so a
collapse from training on them is surprising. It is either a defect or a real finding, and the next wave depends on
which.

## Subject
Current head of `claude/ncm-research`. Files: `scripts/train_gen.py` (`policy_codes`, `compute_loss`, `val_nll`,
resume), `src/ncmr/corruption.py` (`build_hierarchy` with `policy_codes`, re-quantized targets), `src/ncmr/train.py`
(`Trainer`, autocast context, checkpointing), `scripts/train_sudoku_L.py` (its on-policy arm), configs
`configs/scale/L_ts_*.yaml`, notebook generator `notebooks/make_notebooks.py`. Registration: W1, W1a, W1b in
`docs/preregistration.md` (W1b defines H8's hierarchy precisely); S1 / S1b / S1c for Sudoku. Log entries: R014, R024.

## Spec (what correct means)
- On-policy hierarchy: at each level the policy code is a sample from the generator given the clean coarser levels
  (pass 1) or the previous pass's full policy hierarchy (pass t > 1), with level 0 drawn once from the unigram;
  the corruption mask then mixes policy codes with clean codes; targets are the clean canonical codes (A3, A5) or
  the re-quantized codes (A4). Inputs, targets and level offsets must line up exactly as in the clean path.
- If the policy returned the canonical codes, the on-policy hierarchy must equal the clean hierarchy bit for bit.
- Dropout, train/eval mode, autocast state and RNG streams after `policy_codes` must be what they were before.
- `val_nll` is the teacher-forced NLL of canonical codes given the clean hierarchy, identical code path for every arm.
- A resumed run equals an uninterrupted one (R024 F1).

## Questions
1. Is there a defect that makes on-policy training inputs differ from what the generator sees at inference
   (alignment / offsets, level 0, prefix, temperature, mask semantics, eval-mode dropout, autocast, `fnet` vs `net`,
   anything compiled)? Write the identity test above; write any other test that would expose a defect.
2. Is the collapse reproducible on CPU at small scale? E.g. a few hundred on-policy steps on the released Sudoku
   model (hash-pinned assets in `tests/conftest.py`) vs uniform corruption: does solve rate or clean NLL degrade
   comparably fast? Report numbers and keep the run under ~20 CPU-minutes.
3. Does the Sudoku tier-L on-policy arm (`train_sudoku_L.py`, greedy policy) share any defect you find?
4. Did A3 resume, and did the resume replay exactly? Which other w1 runs resumed (look for repeated steps)?
5. Re-quantized arms: are labels and unmasked finer inputs both re-quantized as W1b says? Is the A2 / A4 drop in
   clean val NLL (844 -> 749 / 707) possible without a scoring defect?
6. Which code commit did each run use? `_meta.json` has no SHA; infer from `finished`, `wall_s` and fields present
   in the results, and list what the notebooks should record so this is never ambiguous again.

## Numbers
Recompute the table above and every W1b screen quantity from `results/colab/w1_*/**.json` (16 group means, SDs,
differences against each registered comparator, the step-matched A1 curve points at 1080 / 1350 from the metrics
logs, entropy and MAUVE gaps). Save to `docs/reviews/codex_audit3_numbers.json`.

## Out of scope
Style, naming, performance tuning, the WMT14 trainer, tier M.
