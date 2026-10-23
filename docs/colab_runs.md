# Colab runs: how to launch, and the queue

## How to launch any run (about 1 minute of your time)
1. In Colab: File > Open notebook > GitHub tab > repository `amritisepic/NCM-research`, branch
   `claude/ncm-research`, then pick `notebooks/<run>.ipynb`.
2. Runtime > Change runtime type > T4 GPU (or L4 if offered; any GPU works). Runtime > Run all.
   Approve the Google Drive mount.
3. If the session disconnects, Run all again: every training command resumes from its last checkpoint on
   Drive (`MyDrive/ncm-research/runs/<name>/ckpt.pt`, saved every 500 steps).
4. When the last cell prints "Results copied to ...", copy `MyDrive/ncm-research/results/<run>/` into the
   repo as `results/colab/<run>/` (or just tell me it's done and paste/attach the files).
Each notebook pulls the latest code of the branch at start, so re-running after I push a fix picks it up.

## Queue
| Run | Purpose | GPU | Est. CU | Status |
|---|---|---|---|---|
| `p1_tc_pilot` | P1: TC decomposition of the NCM–AR gap (small models) + 300-step throughput calibration at the matched scale | T4 | ~1.5 | **requested** |
| `c4_sudoku_pilot` | C4 kill experiment: Recurrent NCM vs NCM on Sudoku-hard (1 seed each) | T4 | ~3.0 | **requested** |
| `calib_t4` | calibration only (subset of p1; skip if p1 ran) | T4 | 0.6 | optional |

Budget: 100 CU. Spent: 0. Requested so far: ~4.5.
