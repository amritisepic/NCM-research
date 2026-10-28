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
| `w2m_L_ts_m0` ... `w2m_L_ts_m7`, `w2m_mech_eval` | W2M: eight 600-step tier-L mechanism arms for the on-policy collapse + eval-only mechanism measures | A100 / L4 | ~26 (7 arms) + ~0.5 | arms **done** (R049: M0-M4, M6, M7; M5 skipped); `w2m_mech_eval` **done** (R053, L4: M4 flag use -35 nats/window; W2M's reading final: mixed) |
| `d1_batch1_wallclock` | D1: batch-1 wall-clock gate for criterion (d) (WMT14 J0 / J2-F, Sudoku-hard; fp32 gate, bf16 secondary) | **A100** | ~5.6 (est. 1.5) | **done** (R052: text gate fails, J2-F + stable x1 1.41x J0 sequential; Sudoku verifier 0.44x; (d) closed on text; J1d not run) |
| `e1_equivalence` | E1: representation vs text share of code NLL (TinyStories A0 / A1 / A2 / A4, WMT14 J0 / J1 / J2-F, Sudoku) | A100 / L4 | ~4.3 (L4, incl. one interrupted session) | **done** (R054: prediction 1 holds for TinyStories and Sudoku, not established for WMT14, censoring; analysis PR 15, merged) |
| `w2mg_eval` | W2M-G (eval-only): shared banks, crossed replay M2 x M3 at cuts 2 / 4 / 6, L1 root reliance, M4 flag sweep; decides the L1R gate | **L4** | ~5 (cap 120 L4-min) | **done** (R058: (i) holds, (ii) distributed, (iii) and (iv) hold; L1R gate fails) |
| `w2p_L_ts_m3_c0` | M3-0: M3 without clean rows, seed 0, 600 steps | **A100-40GB** | ~5.4 (cap 45 A100-min) | **done** (R059: 16.50 vs 21.69 Gen-PPL, I = +5.19; holds) |
| `e1b_continue` | E1b: continued E1 sampling on 58 WMT14 J0 and 14 TinyStories A0 items (cut 8) | **L4** (same GPU type every session) | ~6 (cap 150 L4-min) | **done** (R060: E1's (c) replication bar met; claim audit pending) |
| `w2c1_L_ts_m2`, `w2c1_L_ts_m3` | W2C stage 1: M2 / M3 at seeds 1-3, W2M's protocol | **A100-40GB** | ~28 + ~16 (caps 70 / 45 A100-min per run) | **ready: launch next** (one seed per session is fine) |
| `l1r_L_ts_m2r` | L1R kill: M2 with balanced L1 target reassignment, seed 0 (gate script first; A100 > 39 GB only) | **A100-40GB** | ~10 (cap 75 A100-min) | **not run** (R058: gate fails, pi_2 = 0.009 vs 0.25) |
| `p1b_tc_rerun` | P1b: within-level-AR TC probe | — | — | **withdrawn** (R023: false premise; notebook removed; P1c to come) |
| `w2_Mc_ncm` | tier M design C (M1): NCM from scratch, d512 x 6, 6k steps, seed 0 | A100 / L4 | ~7.7 | **hold** (criterion b needs a candidate; wave 1 produced none) |
| `w2_Mc_ar` | tier M design C (M1): AR, same body, FLOP-matched, seed 0 | A100 / L4 | ~4.3 | **hold** (as above) |
| `w2_Mc_ar15` | M1 conditional: AR at 1.5x steps, seed 0 | A100 / L4 | ~6.4 | only when asked (before a criterion-(b) claim) |
| `calib_t4` | old calibration (superseded by w0) | T4 | 0.6 | skip |

Budget: ~200 CU (raised 2026-10-23), raised again by the user's 100-CU purchase (R057). Spent: ~116 CU logged from run time (p1, c4, w0 ~12.5; wave 1 ~46; WMT14 pair ~11.9; W1d ~3; J2-F ~6.5; W2M ~26; D1 ~5.6, mech eval ~0.5, E1 ~4.3), but actual burn ran ~2x (R056). Available now: ~128 CU (27.93 on the panel + 100 bought). Ready queue at the doubled burn: W2M-G ~5, M3-0 ~5.4, E1b ~6, W2C stage 1 ~44, L1R ~10 if gated in (~70-76 total).
Next (R057): launch `w2mg_eval`, `w2p_L_ts_m3_c0` and `e1b_continue` first (they can run in parallel), then `w2c1_L_ts_m2` / `w2c1_L_ts_m3`; `l1r_L_ts_m2r` only if `w2mg_eval`'s gate passes (its first command checks). Reasons and
ranking: `log.md` R031 and `STATE.md`. Every notebook now records the git SHA and each session's resume point
(`_meta.json`, `<run>__provenance.jsonl`) and refuses to resume a run directory under changed code (PR 3). Progress check: the notebook prints `{"step": ..., "steps_per_s": ...,
"tflops_eff": ...}`; remaining steps / steps_per_s = time left.
