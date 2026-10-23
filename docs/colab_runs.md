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
Costs below are measured A100-40GB rates from wave 0 (5.4 CU/h); L4 costs about the same per FLOP and takes ~3x longer.
| Run | Purpose | GPU | Est. CU | Status |
|---|---|---|---|---|
| `p1_tc_pilot` | P1: TC decomposition of the NCM–AR gap (small models) + throughput calibration | T4 | ~3.5 + setup | done (R018) |
| `c4_sudoku_pilot` | C4 kill experiment: Recurrent NCM vs NCM on Sudoku-hard (1 seed each) | T4 | ~5.7 + setup | done (R019) |
| `w0_calib_diag` | reference, calibration, where quality is lost, exact Jacobi | A100 | ~2.5 | done (R020) |
| `w1_L_ts_ctrl` | tier L control (paper recipe, continued), 1800 steps (W1a) | A100 / L4 | ~6 | **launch now** |
| `w1_L_ts_onpol` | B1a: on-policy corruption, 1350 steps | A100 / L4 | ~6 | **launch now** |
| `w1_L_ts_onpol2` | B1a-2: two-pass on-policy corruption, 1080 steps | A100 / L4 | ~6 | **launch now** |
| `w1_L_ts_b1` | B1: on-policy + re-quantized labels, 1350 steps | A100 / L4 | ~6 | **launch now** |
| `w1_L_ts_requant` | B1b: re-quantized labels, 1800 steps | A100 / L4 | ~6 | **launch now** |
| `w1_L_ts_soft` | B7: soft labels, 1800 steps | A100 / L4 | ~6 | **launch now** |
| `w2b_L_sdk_ctrl` | Sudoku tier L control (+ released reference A0), S1 | A100 / L4 | ~2.5 | **launch now** |
| `w2b_L_sdk_uniform` | Sudoku: uniform corruption control | A100 / L4 | ~2.5 | **launch now** |
| `w2b_L_sdk_onpolicy` | Sudoku: on-policy greedy corruption (B1a) | A100 / L4 | ~2.5 | **launch now** |
| `w2b_L_sdk_repair` | Sudoku: board-feedback repair passes (B6/C5) | A100 / L4 | ~2.5 | **launch now** |
| `w2c_L_wmt_ctrl` | WMT14 control (+ released reference J0), J1 | A100 / L4 | ~4 | **launch now** |
| `w2c_L_wmt_consist` | WMT14 self-consistency fine-tune (criterion d, text) | A100 / L4 | ~4 | **launch now** |
| `p1b_tc_rerun` | P1b: fair within-level-AR TC probe (reuses tok_s on Drive) | any (T4 ok) | ~1.2 | **launch now** |
| `w2_M_ncm` | tier M NCM from scratch (25M) | — | ~25 per seed at measured throughput | **on hold**: > 15 CU per run; re-plan after wave 1 |
| `w2_M_ar` | tier M AR, FLOP-matched | — | ~14 per seed | **on hold** (re-plan with w2_M_ncm) |
| `calib_t4` | old calibration (superseded by w0) | T4 | 0.6 | skip |

Budget: ~200 CU (raised 2026-10-23). Spent: ~12.5 CU (p1, c4, w0). Queued above: ~56 CU. Allocation and decision points:
`docs/phase1b_scaleup.md`. Progress check: the notebook prints `{"step": ..., "steps_per_s": ...,
"tflops_eff": ...}`; remaining steps / steps_per_s = time left.
