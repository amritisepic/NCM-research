# Project brief (condensed) — goals and rules for every session

**Mission.** Act as lead researcher on a small-compute project: invent, test and validate a text-generation
architecture that improves on Next-Concept Modeling (NCM; Kumbong, Khare et al., arXiv 2610.14162, in
`docs/2610.14162v1.pdf`; code github.com/HazyResearch/ncm, MIT, vendored in `src/ncmr/ncm_ref/`; checkpoints
huggingface.co/hazyresearch/ncm-checkpoints). Results must hold up under hostile scrutiny. Careful null
results count.

## Phases
0. Reproduce one released result (done: Sudoku, R004/R007); harness (per-level decoding and metrics,
   per-token-class reconstruction, linear probes, evaluation suite); matched baselines: reimplemented NCM and
   an AR transformer with the same params / data / FLOPs.
1. Literature search and >= 8 candidate architectures (>= 3 not obvious extensions of the user's ideas;
   >= 2 taking their mechanism from outside ML, mapped to concrete losses or sampling changes). Each: mechanism,
   weakness targeted, kill experiment (<= 3 GPU-h), cost, prior. Rank; choose 2-3 incl. one high-risk bet.
   (Done: `docs/phase1_literature_and_candidates.md`, `docs/phase1b_reevaluation.md`, `docs/phase1b_scaleup.md`.)
2. Pre-registered pilots (`docs/preregistration.md`), one axis at a time; kill fast.
3. Synthesis: >= 3 seeds, mean / SD / 95% CI; a gain must exceed 2x the pooled seed SD; actively try to break it.
4. Write-up.

## Success bar (any one, met precisely)
- (a) Better than matched NCM at equal compute across the full suite with <= 1.5x NFE, or matched quality at
  <= 0.6x NFE.
- (b) Close >= half of the NCM -> AR gap with >= 3x fewer NFE than AR.
- (c) A replicated mechanistic finding about what the levels encode.
- (d) An adaptive-compute rule saving >= 25% NFE at < 1 point quality loss on Sudoku AND one text task.

## Evaluation suite (every method)
Gen-PPL under two scorers (GPT-2 large, TinyStories-33M), always with entropy, repetition and MAUVE;
per-token-class accuracy; Sudoku exact solve rate by difficulty; NFE and wall-clock. (`src/ncmr/eval_text.py`.)

## Honesty rules
Report negatives. No cherry-picking. Random samples with stated seeds. Flag every number not run by us.

## Scale and budget
- Original caps: tokenizer <= 10M, generator <= 25M parameters. **Lifted on 2026-10-23 by the user** for tiers
  L/M: the released 88.6M TinyStories generator and 36.4M tokenizer (and the released Sudoku / WMT14 models).
- Colab budget ~200 CU (raised from 100 on 2026-10-23). The user launches Colab notebooks.

## Ask the user only for
(a) any single run > 15 CU; (b) changing the success criteria or scale limits; (c) a Colab run they need to
launch; (d) a result believed to meet the success bar, before spending more compute on it.

## State and continuity
`STATE.md` (read first: phase, hypotheses, killed ideas, compute spent/remaining, next 3 actions), `log.md`
(append-only, one entry per run), `results/`, `configs/`.

## Repository hygiene (public repo)
Commit nothing identifying: no personal names or emails, no absolute local paths, no session links, no
credentials. Commit trailer: the co-author line the harness specifies, without any session link.

## Working style
Matter-of-fact and concise, no flattery. Define domain terms in plain language. Challenge the user's
assumptions and ask conceptual questions about the field.
