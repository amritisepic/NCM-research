# Compute model and budget

All FLOP numbers are analytic "useful" FLOPs from `src/ncmr/flops.py` (multiply-add = 2 FLOPs, training
step = 3 x forward). Dense-mask attention executes more FLOPs than counted; wall-clock captures that.

## Hardware (measured)
| Device | fp32 matmul | bf16 matmul | Notes |
|---|---|---|---|
| Cloud container (this agent), 4-core Xeon w/ AMX | 0.47-0.54 TFLOPS | 1.9-3.2 TFLOPS | n=512..2048 square matmuls; real models reach far less (see "effective") |
| Colab T4 | (to calibrate) | fp16 only | 1.19 CU/h |
| Colab L4 | (to calibrate) | bf16 | 1.71 CU/h |
| Colab A100 40 / 80 GB | (to calibrate) | bf16 | 5.40 / 7.52 CU/h |
CU rates: third-party measurements (March 2026); the Colab resource panel shows the live rate.

Effective training throughput on the container: _to be filled from tok_s / ncm_s step times_.

## Key structural fact: NCM costs more FLOPs than AR at equal parameters
With the NCM TinyStories level schedule (sum of levels = 641 positions for 128 tokens), the generator
processes 641 positions per sequence vs 128 for AR:
- training FLOPs per sequence, NCM / AR = **4.25x** (d512 x 6 layers, codebook-tied heads);
- sampling FLOPs per sample: NCM 25.9 GFLOP vs AR (KV cache) 6.0 GFLOP = **4.3x**.
NCM's advantage is *sequential depth* (11 NFE vs 128), not FLOPs. "Matched FLOPs" and "matched data"
therefore cannot both hold: at matched training FLOPs the AR model sees 4.25x more sequences. We train
FLOP-matched (primary) and read the data-matched point off the AR learning curve.

## Planned runs (estimates; actuals go to log.md)
| Run | Where | Params | Train FLOPs | Est. time | Est. CU |
|---|---|---|---|---|---|
| tok_s (6000 x 64) | CPU / Colab | 8.4M (incl. codebooks) | 2.0 PFLOP | CPU ~1.4 h @0.4 TFLOPS | T4 0.06 |
| ncm_s (6000 x 32) | CPU / Colab | 5.5M | 4.0 PFLOP | CPU ~2.8 h | T4 0.13 |
| ts_fact_s, ts_ar_s (6000 x 32) | Colab | 5.5M + 2.9M emb | 7.9 PFLOP each | CPU ~5.5 h each | T4 0.26 each |
| ar_s (6000 x 128) | CPU / Colab | 6.8M | 4.1 PFLOP | CPU ~2.9 h | T4 0.14 |
| tok_m, 1 epoch | Colab | 8.4M | 13.8 PFLOP | T4 0.38 h | 0.46 |
| Sudoku C4 pilot: tok + ncm_s + rncm_s (20k x 64) | Colab | 3.2M + 2 x 4.8M | ~35 PFLOP | T4 ~1.5-2 h | ~2.5 |
| ncm_m, 1 epoch | Colab | 20.5M | 208 PFLOP | T4 5.8 h / L4 2.3 h | 6.9 / 3.9 |
| ar_m, FLOP-matched | Colab | 23.1M | 208 PFLOP | T4 5.8 h / L4 2.3 h | 6.9 / 3.9 |
T4 assumed 10 TFLOPS effective and L4 25 TFLOPS until calibrated (p1_tc_pilot notebook includes
300-step calibration runs at the m scale). If estimates are off by >2x, investigate before the next run.

## Scale-up tiers (2026-10-23, `docs/phase1b_scaleup.md`)
| Run | Params | FLOPs/run | Est. CU (L4 @ 25-40 TFLOPS useful) |
|---|---|---|---|
| Tier L TinyStories fine-tune, 4000 x 256 (released generator) | 88.6M | 349 PFLOP (on-policy arms: fewer steps, same FLOPs) | 4.5-7 incl. eval |
| Tier M NCM from scratch, 15k x 256 on the released tokenizer | 25.2M body | 411 PFLOP | 6-10 |
| Tier M AR, FLOP-matched (70k x 256; NCM/AR = 4.66x per window) | 25.2M + 4.2M emb | 411 PFLOP | 6-10 |
| Sudoku fine-tune, 5000 x 256 (released generator) | 25.5M | 84 PFLOP | ~1.3 |
| Paper TinyStories generator from scratch (for scale) | 88.6M | ~34 EFLOP | 400+ (not planned) |

## Budget
Raised to ~200 CU by the user on 2026-10-23 (see phase1b_scaleup.md for the allocation). Original note:
Colab: 100 CU total. Planned Phase-0/1 use: ~1 CU (pilot + calibration). Matched baselines with 3 seeds
at the m scale cost ~25 CU on L4 (NCM + AR), so Phase-3 confirmation must be planned against the
remaining ~70 CU; anything >15 CU in one run needs approval.
