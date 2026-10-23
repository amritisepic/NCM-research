# Colab runs: how to launch, and the queue

## How to launch any run (about 1 minute of your time)
1. In Colab: File > Open notebook > GitHub tab > repository `amritisepic/NCM-research`, branch
   `claude/ncm-research`, then pick `notebooks/<run>.ipynb`.
2. Runtime > Change runtime type > **L4** (or A100; T4 works but costs ~2x more per FLOP). Runtime > Run all.
   Approve the Google Drive mount.
3. If the session disconnects, Run all again: every training command resumes from its last checkpoint on
   Drive (`MyDrive/ncm-research/runs/<name>/ckpt.pt`). Released checkpoints are cached in
   `MyDrive/ncm-research/artifacts/` after the first download.
4. More seeds of the same run: edit `SEEDS = [0]` in the first code cell (e.g. `[1, 2]`) and Run all.
5. When the last cell prints "Results copied to ...", copy `MyDrive/ncm-research/results/<run>/` into the
   repo as `results/colab/<run>/` (or tell me it's done and paste/attach the files).
Each notebook pulls the latest code of the branch at start, so re-running after I push a fix picks it up.
If your plan allows several sessions at once, notebooks within the same wave can run in parallel.

## Queue
| Run | Purpose | GPU | Est. CU | Status |
|---|---|---|---|---|
| `p1_tc_pilot` | P1: TC decomposition of the NCM–AR gap (small models) + throughput calibration | T4 | ~3.5 + setup | done (results/colab/) |
| `c4_sudoku_pilot` | C4 kill experiment: Recurrent NCM vs NCM on Sudoku-hard (1 seed each) | T4 | ~5.7 + setup | done (results/colab/) |
| `w0_calib_diag` | **next:** released-model reference with the wave-1 protocol; calibration; where quality is lost; exact Jacobi on WMT14 / TinyStories / Sudoku | L4 | ~2.5 | running (launched 2026-10-23) |
| `w1_L_ts_ctrl` | tier L control (paper recipe, continued) | L4 | 4.5-7 | ready (after w0) |
| `w1_L_ts_onpol` | B1a: on-policy corruption | L4 | 4.5-7 | ready (after w0) |
| `w1_L_ts_onpol2` | B1a-2: two-pass on-policy corruption | L4 | 4.5-7 | ready (after w0) |
| `w1_L_ts_b1` | B1: on-policy + re-quantized labels | L4 | 4.5-7 | ready (after w0) |
| `w1_L_ts_requant` | B1b: re-quantized labels | L4 | 4.5-7 | ready (after w0) |
| `w1_L_ts_soft` | B7: soft labels | L4 | 4.5-7 | ready (after w0) |
| `w2_M_ncm` | tier M NCM from scratch (25M), seeds 0-2 | L4 | 6-10 per seed | ready (any time) |
| `w2_M_ar` | tier M AR, FLOP-matched, seeds 0-2 | L4 | 6-10 per seed | ready (any time) |
| `calib_t4` | old calibration (superseded by w0) | T4 | 0.6 | skip |

Budget: ~200 CU (raised 2026-10-23). Spent: ~10 CU (p1 + c4, T4). Allocation and decision points:
`docs/phase1b_scaleup.md`. Progress check: the notebook prints `{"step": ..., "steps_per_s": ...,
"tflops_eff": ...}`; remaining steps / steps_per_s = time left.
