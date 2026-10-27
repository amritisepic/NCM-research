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
**Wave 1 (`w1_*`) needs a bf16 GPU: A100 or L4, not T4** (W1b). A run directory that holds a checkpoint from before PR 3 (all wave-1 directories) is refused on resume; use a
fresh directory (new seed) or set `ALLOW_CODE_CHANGE = True` only on purpose.
Final eval per W1 arm: 16 x 256 samples in fp32 (~5-10 min on an A100).
| Run | Purpose | GPU | Est. CU | Status |
|---|---|---|---|---|
| `p1_tc_pilot` | P1: TC decomposition of the NCM–AR gap (small models) + throughput calibration | T4 | ~3.5 + setup | done (R018) |
| `c4_sudoku_pilot` | C4 kill experiment: Recurrent NCM vs NCM on Sudoku-hard (1 seed each) | T4 | ~5.7 + setup | done (R019) |
| `w0_calib_diag` | reference, calibration, where quality is lost, exact Jacobi | A100 | ~2.5 | done (R020) |
| `w1_L_ts_ctrl` | tier L control (paper recipe, continued), 1800 steps (W1a); also A0 with the 16-group protocol (W1b) | A100 / L4 | ~8 (incl. a restart) | done (R030); seeds 1-2 **held** (user, R033: until M1's gate or a new tier-L comparison needs them) |
| `w1_L_ts_onpol` | B1a: on-policy corruption, 1350 steps | A100 / L4 | ~9 (incl. restarts) | done (R030: collapsed); no further seeds |
| `w1_L_ts_onpol2` | B1a-2: two-pass on-policy corruption, 1080 steps | A100 / L4 | ~6.6 | done (R030: collapsed; nominally advances vs A3) -> seeds **dropped** (user, R033) |
| `w1_L_ts_b1` | B1: on-policy + re-quantized labels, 1350 steps | A100 / L4 | ~9.7 (incl. restarts) | done (R030: collapsed; nominally advances vs A3) -> seeds **dropped** (user, R033) |
| `w1_L_ts_requant` | B1b: re-quantized labels, 1800 steps | A100 / L4 | ~6.5 | done (R030: 0.31 below A1, bar 0.35) |
| `w1_L_ts_soft` | B7: soft labels, 1800 steps | A100 / L4 | ~6.5 | done (R030: 0.85 worse than A1) |
| `w1d_requant_diag` | W1d (eval-only): temperature frontier and oracle curve for A0 / A1 / A2 from the wave-1 checkpoints on Drive | **A100** (exact reproduction gate) | ~3 | done (R039: gate exact; "trade-off", A2 not pursued) |
| `w2b_L_sdk_ctrl` | Sudoku tier L control (+ released reference A0), S1 | A100 / L4 | ~2.5 | **hold** (after WMT14; R031) |
| `w2b_L_sdk_uniform` | Sudoku: uniform corruption control | A100 / L4 | ~2.5 | **hold** (after WMT14; R031) |
| `w2b_L_sdk_onpolicy` | Sudoku: on-policy greedy corruption (B1a) | A100 / L4 | ~2.5 | **drop** (recommended, R031: the greedy policy corrupts a few % of codes; the mechanism is tested on CPU instead) |
| `w2b_L_sdk_repair` | Sudoku: board-feedback repair passes (B6/C5) | A100 / L4 | ~2.5 | **hold** (after WMT14; R031) |
| `w2c_L_wmt_ctrl` | WMT14 control (+ released reference J0), J1 + J1c | A100 / L4 | ~4.6 | done (R036: J1 loses 0.8 BLEU, recipe OK); seeds 1-2 only if J2-F advances |
| `w2c_L_wmt_consist` | WMT14 self-consistency fine-tune (criterion d, text); fp32 teacher (J1b), padded degenerate rows (J1c) | A100 / L4 | ~6.4 | done (R036: misses the screen; 16% of test sentences collapse onto two fixed strings); no further seeds |
| `w2c_L_wmt_consist_frozen` | J2-F: J2 with a frozen released-model teacher (contingent arm, triggered by J2's miss) | **A100** | ~6.5 | done (R042: no collapse, 5.6% pass saving; misses the screen; no seeds) |
| `w2m_L_ts_m0` ... `w2m_L_ts_m7`, `w2m_mech_eval` | W2M: eight 600-step tier-L mechanism arms for the on-policy collapse + eval-only mechanism measures | A100 / L4 | ~27 total (< 4 each) | arms **ready** (R041); M5 is a near-replicate of M4 at these rates (R043, W2Ma): skip it if not yet run (user's choice). **Hold `w2m_mech_eval`** for the probe fix (W2Ma), or run it and report probe (b) as uninformative |
| `p1b_tc_rerun` | P1b: within-level-AR TC probe | — | — | **withdrawn** (R023: false premise; notebook removed; P1c to come) |
| `w2_Mc_ncm` | tier M design C (M1): NCM from scratch, d512 x 6, 6k steps, seed 0 | A100 / L4 | ~7.7 | **hold** (criterion b needs a candidate; wave 1 produced none) |
| `w2_Mc_ar` | tier M design C (M1): AR, same body, FLOP-matched, seed 0 | A100 / L4 | ~4.3 | **hold** (as above) |
| `w2_Mc_ar15` | M1 conditional: AR at 1.5x steps, seed 0 | A100 / L4 | ~6.4 | only when asked (before a criterion-(b) claim) |
| `calib_t4` | old calibration (superseded by w0) | T4 | 0.6 | skip |

Budget: ~200 CU (raised 2026-10-23). Spent: ~79.5 CU (p1, c4, w0 ~12.5; wave 1 ~46 incl. restarts, R030; WMT14 pair ~11.9, R036; W1d ~3, R039; J2-F ~6.5, R042).
Next: W2M ~27 (cleared). Proposed in `docs/plan_2026-10-27.md` (awaiting the user): E1/E2 ~2, D1 ~1.5, then W2C / J1d by gate; cut M1, Sudoku arms, A1 seeds 1-2. Reasons and
ranking: `log.md` R031 and `STATE.md`. Every notebook now records the git SHA and each session's resume point
(`_meta.json`, `<run>__provenance.jsonl`) and refuses to resume a run directory under changed code (PR 3). Progress check: the notebook prints `{"step": ..., "steps_per_s": ...,
"tflops_eff": ...}`; remaining steps / steps_per_s = time left.
