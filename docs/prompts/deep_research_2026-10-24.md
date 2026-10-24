# Deep-research brief (paste into ChatGPT deep research), 2026-10-24

You are advising a small-compute research project. Read this brief, then the public files linked below, then
answer the questions in section 4. Primary sources only (papers, official code, reported tables); give numbers and
the exact setting they come from. Keep what a source shows separate from your own reasoning, and label the latter
"speculation". If you cannot find evidence for something, say so instead of filling the gap.

## 1. What we study (Next-Concept Modeling, NCM; Kumbong, Khare et al., arXiv 2610.14162)
- A frozen tokenizer maps each 128-token text window to a 12-level multi-scale **residual-quantized** code hierarchy
  (level sizes 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128; 4096 codes per level). Each level encodes the residual
  left by the coarser levels; summing all levels' embeddings and decoding gives the text back (~99%).
- A transformer generator predicts level k's codes in parallel given the cumulative reconstruction of levels < k
  (block-diagonal attention: each level sees a prefix and its own input only). Sampling: level 0 from a unigram, then
  levels 1-11 one level per forward pass (11 passes for 128 tokens, versus 128 for an autoregressive model).
- NCM trains with **conditioning corruption**: a share of the coarser codes in the input (mean 0.5, decreasing from
  0.75 at level 0 to 0.25 at level 10) is replaced by **uniformly random codes**, and the labels stay the clean
  (canonical) codes. The paper reports this as its largest ablation effect.
- Released checkpoints we fine-tune: TinyStories generator (88.6M), Sudoku-hard (25.5M, 16 codes per level), WMT14
  De-En (120M, 11 levels).

## 2. What we measured (seed 0, released TinyStories model fine-tuned for equal generator FLOPs)
Gen-PPL = GPT-2 large perplexity of 16 x 256 unconditional samples (lower is better; mean ± SE over the 16 groups),
always read with unigram entropy; MAUVE vs 4096 human windows; clean val NLL = teacher-forced NLL of the canonical
codes given the clean hierarchy (nats per 128-token window).

| arm | training hierarchy | Gen-PPL | MAUVE | clean NLL | entropy |
|---|---|---|---|---|---|
| A0 | released model, no training | 16.16 ± 0.11 | 0.827 | 861.8 | 3.970 |
| A1 | control: continue NCM's uniform corruption, clean labels | 16.60 ± 0.08 | 0.881 | 844.0 | 3.962 |
| A2 | uniform corruption + labels re-quantized against the residual left by the corrupted codes | 16.29 ± 0.09 | 0.915 | 749.3 | 3.963 |
| A3 | corruption by the generator's own one-step samples (given the clean coarser levels), clean labels | 41.44 ± 0.43 | 0.496 | 885.7 | 4.026 |
| A4 | A3 + re-quantized labels | 36.81 ± 0.26 | 0.673 | 707.4 | 4.037 |
| A5 | A3 with two-pass (compounded) samples | 32.17 ± 0.18 | 0.543 | 913.9 | 3.990 |
| A6 | A1 + soft labels over the 16 nearest codes | 17.46 ± 0.10 | 0.832 | 888.9 | 3.958 |

Facts that matter for your answer (details and caveats in `log.md` R008, R014, R020, R023, R030, R031):
- Before training, the generator's own one-step samples were the *mildest* corruption in token terms: replacing
  coarse codes by them keeps 98% of decoded tokens, versus 43% for uniform random codes (R014). Yet training on them
  destroyed sample quality within 300 steps in all three on-policy variants; two independent code audits found no
  defect that explains it.
- The damage is local: the collapsed arms' pooled vocabulary is as close to human text as the control's, but local
  order breaks (bigram gain over unigram 2.13 → 1.71 nats/token; doubled, dropped and garbled words, 2.6x more
  adjacent duplicates); story-level structure survives.
- Re-quantized labels cut clean NLL by 95-137 nats, almost all of it in the four finest levels, which hold ~4-7% of
  sample quality in an oracle intervention (supplying real codes for levels 0..j and sampling the rest: levels 1-5
  hold 86% of the quality gap between sampled and real codes). Their sample quality barely moved (A2) or collapsed (A4).
- Exact Jacobi decoding across levels (iterate all levels in parallel to a fixed point) needs 6.41 instead of 12
  passes on Sudoku-hard with identical output, but 11.07 instead of 11 on WMT14 (no saving). A registered experiment
  (J1/J2) fine-tunes the WMT14 model with a CLLM-style consistency loss (map intermediate Jacobi states to the
  model's own fixed point) to reach <= 8.25 passes at < 1 BLEU loss.

Public repository (branch `claude/ncm-research`): https://github.com/amritisepic/NCM-research/tree/claude/ncm-research
- run log: https://github.com/amritisepic/NCM-research/blob/claude/ncm-research/log.md (R030 registered analysis,
  R031 exploratory analysis and our mechanism hypothesis)
- current state and plan: https://github.com/amritisepic/NCM-research/blob/claude/ncm-research/STATE.md
- registrations: https://github.com/amritisepic/NCM-research/blob/claude/ncm-research/docs/preregistration.md
- raw results: https://github.com/amritisepic/NCM-research/tree/claude/ncm-research/results/colab

## 3. Our current reading (attack it)
Our leading explanation for the collapse: uniform random codes are easy to recognise as corrupted, so the model
learns "repair codes that look random, trust codes that look plausible". The model's own samples look exactly like
the codes it sees at inference, so with clean labels it learns to discount every plausible coarse code by roughly the
corruption rate; at inference all coarse codes are its own, so fine codes stop fitting the coarse plan. We read this
as Huszár's (2015) improperness argument against scheduled sampling, made visible by residual quantization.
A CPU test on the released Sudoku model supports the first half: at identical corrupted positions and rates, wrong
codes drawn from the model's own distribution raised clean NLL by 32 nats per puzzle, uniform wrong codes by 2.7, while
greedy solving was unaffected (sampling-time results in R031).
Our plan: (1) finish criterion (d) on text with J1/J2 (WMT14, ~9 CU); J2 also trains on the model's own intermediate
codes, but decodes greedily, which hid the damage in Sudoku; (2) test the mechanism at tier L with four short
TinyStories arms (~12 CU); (3) treat clean code NLL as unreliable for model selection.

## 4. Questions
1. **When does training on a model's own predictions hurt, and why?** Cover scheduled sampling (Bengio et al. 2015;
   Huszár 2015; transformer variants), exposure-bias fixes in diffusion (input perturbation, e.g. Ning et al. 2023),
   masked generative transformers (MaskGIT-style re-masking, token-critic), residual-quantized and multi-scale
   generators (RQ-Transformer, VALL-E's coarse-to-fine stages, VAR, Infinity's bitwise self-correction), and
   consistency training for Jacobi decoding (CLLMs, lookahead decoding). For each: what was corrupted, what the target
   was (true data, the model's own continuation, a re-quantized target), whether the corruption was distinguishable
   from clean input, and the measured effect with numbers. Does any source report a collapse like ours, or a fix?
2. **Is our mechanism right, and what is the cheapest discriminating experiment?** Give the strongest alternative
   explanations (for example optimisation shock, loss of robustness to off-manifold inputs, policy drift, a hidden
   train/inference mismatch specific to residual hierarchies) and what each predicts that ours does not. We plan three
   tests: corruption with the model's own codes plus a "this position was replaced" flag (should not collapse if we are
   right); codes drawn from the model's distribution excluding the correct one at NCM's uniform rates (should collapse);
   uniform corruption at the on-policy arm's lower rates (should not). Are these the right tests? What is missing?
3. **Why do likelihood and sample quality diverge in hierarchical discrete models?** Prior work on likelihood vs
   sample quality (e.g. Theis et al. 2016), on residual / multi-scale VQ codes where many codes decode to nearly the
   same output, and on equivalence-aware or decoded-space likelihoods. Is "fine levels hold most of the NLL and almost
   none of the quality" a known property of RVQ hierarchies (audio codecs, image RQ-VAEs)? Give numbers.
4. **Prior mechanistic findings about what the levels of multi-scale codes encode** (VAR, RQ-VAE, SoundStream/EnCodec
   RVQ, hierarchical VQ-VAE), especially anything measured by interventions rather than probes. What would make our
   own finding new?
5. **Criterion (d): can consistency training make NCM's level-parallel Jacobi decoding converge early?** CLLMs and
   related work report speedups for token-level Jacobi decoding; what do they report for quality and for the number of
   iterations, and what predicts whether a model learns to jump to its fixed point? Anything specific to
   coarse-to-fine or multi-scale models? What would make J2 fail?
6. **Argue against our plan.** Then propose the single most significant experiment that fits ~100 A100 compute units
   (~18 A100-hours) using only the released checkpoints, with the evidence for and against it. "Significant" means one
   of: (a) better than a matched NCM at equal compute across the evaluation suite with <= 1.5x the function
   evaluations (NFE), or matched quality at <= 0.6x NFE; (b) closing >= half the NCM-to-autoregressive gap with
   >= 3x fewer NFE than the autoregressive model; (c) a replicated mechanistic finding about what the levels encode;
   (d) an adaptive-compute rule saving >= 25% NFE at < 1 point quality loss on Sudoku and one text task.

## 5. Output format
- One section per question: findings with citations and numbers first, then "speculation" clearly marked.
- A table of the most relevant papers: setting, what was corrupted or trained on, target type, effect size.
- A final section "What we would have to see to change our minds": for our mechanism, for J2, and for dropping
  on-policy corruption altogether, list concrete observations (with thresholds) that would overturn each.
