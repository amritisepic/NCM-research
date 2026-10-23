"""Generate self-contained Colab notebooks, one per planned run (see docs/colab_runs.md).

Each notebook: mounts Drive -> clones/updates the repo branch -> installs deps -> fetches data (cached on
Drive) -> runs the commands (each resumes automatically from its Drive checkpoint, saved every
`ckpt_every` steps, default 500) -> copies result JSONs to Drive/ncm-research/results/<run>/ for you to
drop into the repo's results/.
Re-running a notebook after a disconnect continues where it stopped.

Usage: python notebooks/make_notebooks.py   (writes notebooks/<run>.ipynb for every RUNS entry)
"""
from __future__ import annotations

import json
import os

REPO = "https://github.com/amritisepic/NCM-research.git"
BRANCH = "claude/ncm-research"

# name -> (description, gpu, est_units, list of commands; {R} = Drive run root, {D} = data dir)
RUNS: dict[str, dict] = {
    "p1_tc_pilot": {
        "desc": "Pilot P1 (small scale) + throughput calibration. Trains the small NCM-style tokenizer, then "
                "the factorized NCM generator, the two-stream TC probes (factorized vs within-level AR), and a "
                "FLOP-matched AR baseline; then 300-step calibration runs at the matched 'm' scale. "
                "Answers: how much of NCM's NLL gap to AR is within-level dependence (TC)?",
        "gpu": "T4 (any GPU works)", "units": 1.5,
        "cmds": [
            "python scripts/train.py tokenizer --config configs/pilot/tok_s.yaml --run_dir {R}/tok_s",
            "python scripts/train.py ncm --config configs/pilot/ncm_s.yaml --tokenizer {R}/tok_s/final.pt --run_dir {R}/ncm_s",
            "python scripts/train.py ts_fact --config configs/pilot/ncm_s.yaml --tokenizer {R}/tok_s/final.pt --run_dir {R}/ts_fact_s",
            "python scripts/train.py ts_ar --config configs/pilot/ncm_s.yaml --tokenizer {R}/tok_s/final.pt --run_dir {R}/ts_ar_s",
            "python scripts/train.py ar --config configs/pilot/ar_s.yaml --run_dir {R}/ar_s",
            "python scripts/train.py tokenizer --config configs/calib/tok_m.yaml --run_dir {R}/calib_tok_m",
            "python scripts/train.py ar --config configs/calib/ar_m.yaml --run_dir {R}/calib_ar_m",
            "python scripts/train.py ncm --config configs/calib/ncm_m.yaml --tokenizer {R}/calib_tok_m/final.pt "
            "--run_dir {R}/calib_ncm_m",
        ],
    },
    "c4_sudoku_pilot": {
        "desc": "Kill experiment for C4 (Recurrent NCM) on Sudoku-hard, small scale, 1 seed per arm. Trains our "
                "Sudoku tokenizer, then (a) upstream-style NCM generator and (b) the same generator with its hidden "
                "state carried across levels and trained with backprop through levels. Same params, same NFE (12). "
                "Pre-registered: (b) beats (a) by >= 3 points on our held-out puzzles, and by >= 10 points on "
                "puzzles needing >= 9 propagation rounds; otherwise C4 is killed. Released test set scored once "
                "at the end (--final), not used for selection.",
        "gpu": "T4 (any GPU works)", "units": 3.0,
        "cmds": [
            "python scripts/train_sudoku.py tokenizer --config configs/sudoku/tok.yaml --level hard "
            "--data assets/sudoku_train/hard.npz --run_dir {R}/sdk_tok_hard",
            "python scripts/train_sudoku.py ncm --config configs/sudoku/gen_s.yaml --level hard "
            "--data assets/sudoku_train/hard.npz --tokenizer {R}/sdk_tok_hard/final.pt --run_dir {R}/sdk_ncm_s_hard --final",
            "python scripts/train_sudoku.py rncm --config configs/sudoku/gen_s.yaml --level hard "
            "--data assets/sudoku_train/hard.npz --tokenizer {R}/sdk_tok_hard/final.pt --run_dir {R}/sdk_rncm_s_hard --final",
        ],
    },
    "calib_t4": {
        "desc": "Throughput calibration: 300 steps each of tokenizer / NCM generator / AR at the 'm' scale. "
                "Measures real TFLOPS on this GPU so all later estimates are grounded.",
        "gpu": "T4", "units": 0.6,
        "cmds": [
            "python scripts/train.py tokenizer --config configs/calib/tok_m.yaml --run_dir {R}/calib_tok_m",
            "python scripts/train.py ar --config configs/calib/ar_m.yaml --run_dir {R}/calib_ar_m",
            "python scripts/train.py ncm --config configs/calib/ncm_m.yaml --tokenizer {R}/calib_tok_m/final.pt "
            "--run_dir {R}/calib_ncm_m",
        ],
    },
}

# ---- Scale-up waves (docs/phase1b_scaleup.md). {S} = seed; SEEDS is set at the top of each notebook.
L_ARMS = {
    "L_ts_ctrl": "control: continue the released recipe (uniform corruption, clean labels)",
    "L_ts_requant": "B1b: uniform corruption + labels re-quantized against the true residual",
    "L_ts_onpol": "B1a: on-policy corruption (the generator's own samples) + clean labels",
    "L_ts_b1": "B1: on-policy corruption + re-quantized labels",
    "L_ts_onpol2": "B1a-2: on-policy corruption with two-pass (compounding) errors + clean labels",
    "L_ts_soft": "B7: control + soft labels over near-equivalent codes",
}
RUNS["w0_calib_diag"] = {
    "desc": "Wave 0 (do this first; ~1-2 CU). (0) Reference: the released TinyStories model evaluated with the "
            "exact protocol of the wave-1 arms (no training). (1) Throughput calibration of the scale-up trainers on this GPU "
            "(60 steps each: paper-scale fine-tune, from-scratch 25M NCM, 25M AR). (2) Where NCM loses quality: "
            "sample with real coarse levels 0..j and generated finer levels (released TinyStories model). "
            "(3) Exact Jacobi decoding across levels on the full WMT14 test set and on TinyStories (training-free "
            "adaptive-compute baseline for criterion d).",
    "gpu": "L4 (or A100)", "units": 1.5, "needs": ["ts", "ckpt", "wmt14"], "seeds": [0],
    "cmds": [
        "python scripts/train_gen.py --config configs/scale/L_ts_ctrl.yaml --run_dir {R}/L_ts_released --eval_only",
        "python scripts/train_gen.py --config configs/scale/L_ts_ctrl.yaml --run_dir {R}/calib_L --calib_steps 60",
        "python scripts/train_gen.py --config configs/scale/M_ncm.yaml --run_dir {R}/calib_M --calib_steps 60",
        "python scripts/train.py ar --config configs/scale/M_ar.yaml --run_dir {R}/calib_M_ar --calib_steps 300",
        "python scripts/tf_levels_diag.py --n 512",
        "python scripts/jacobi_levels.py --task wmt14 --n 3003 --bs 128",
        "python scripts/jacobi_levels.py --task tinystories --n 512 --bs 128 --seed 0",
        "python scripts/jacobi_levels.py --task sudoku --level hard --n 2000 --bs 250",
    ],
}
for arm, what in L_ARMS.items():
    RUNS[f"w1_{arm}"] = {
        "desc": f"Wave 1, tier L (paper scale): fine-tune the released 88.6M TinyStories generator; arm = {what}. "
                "All arms share data order, schedule and training FLOPs (on-policy arms run 3/4 of the steps because "
                "each step has one extra forward pass). Final eval: 4 x 256 samples, Gen-PPL (GPT-2 large and "
                "TinyStories-33M), entropy, repetition, duplicates, MAUVE. Run seed 0 first; seeds 1-2 only for the "
                "arms that survive (edit SEEDS).",
        "gpu": "L4 (A100 is ~3x faster for about the same CU)", "units": 4.5, "needs": ["ts", "ckpt"], "seeds": [0],
        "cmds": [f"python scripts/train_gen.py --config configs/scale/{arm}.yaml --run_dir {{R}}/{arm}_s{{S}} --seed {{S}}"],
    }
RUNS["w2_M_ncm"] = {
    "desc": "Wave 2, tier M: NCM generator from scratch (d512 x 8, 25M body) on the released tokenizer, paper recipe "
            "with conditioning corruption; ~411 PFLOP. Matched baseline for Phase 3 (3 seeds).",
    "gpu": "L4", "units": 7, "needs": ["ts", "ckpt"], "seeds": [0],
    "cmds": ["python scripts/train_gen.py --config configs/scale/M_ncm.yaml --run_dir {R}/M_ncm_s{S} --seed {S}"],
}
RUNS["w2_M_ar"] = {
    "desc": "Wave 2, tier M: AR transformer with the same body (d512 x 8), FLOP-matched to w2_M_ncm (~411 PFLOP), "
            "then 4 x 256 samples at the FLOP-matched end and at the data-matched snapshot (15k steps), full eval suite.",
    "gpu": "L4", "units": 7, "needs": ["ts", "ckpt"], "seeds": [0],
    "cmds": ["python scripts/train.py ar --config configs/scale/M_ar.yaml --run_dir {R}/M_ar_s{S} --seed {S}",
             "python scripts/sample.py --run_dir {R}/M_ar_s{S} --out {R}/M_ar_s{S}/samples_final.json --seeds 42 0 1 2 --n 256",
             "python scripts/eval_samples.py --samples {R}/M_ar_s{S}/samples_final.json --ref val --out {R}/M_ar_s{S}/eval_final.json",
             "python scripts/sample.py --run_dir {R}/M_ar_s{S} --ckpt snap_15000.pt --out {R}/M_ar_s{S}/samples_15k.json --seeds 42 0 1 2 --n 256",
             "python scripts/eval_samples.py --samples {R}/M_ar_s{S}/samples_15k.json --ref val --out {R}/M_ar_s{S}/eval_15k.json"],
}

SETUP = r'''
import os, subprocess, sys, json, shutil, time
T_START = time.time()
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/ncm-research'
R = f'{DRIVE}/runs'; D = f'{DRIVE}/data/tinystories'
os.makedirs(R, exist_ok=True); os.makedirs(D, exist_ok=True)
REPO, BRANCH = '__REPO__', '__BRANCH__'
CMDS = __CMDS__
SEEDS = __SEEDS__   # edit (e.g. [1, 2]) to run more seeds of the same commands
NEEDS = __NEEDS__
NEED_TS = 'ts' in NEEDS or any('scripts/train.py ' in c for c in CMDS)
if not os.path.exists('/content/NCM-research'):
    subprocess.run(['git', 'clone', '-b', BRANCH, REPO, '/content/NCM-research'], check=True)
else:
    subprocess.run(f'cd /content/NCM-research && git fetch origin {BRANCH} && git checkout {BRANCH} && git reset --hard origin/{BRANCH}', shell=True, check=True)
os.chdir('/content/NCM-research')
print(subprocess.run(['git', 'log', '-1', '--oneline'], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'omegaconf', 'hydra-core', 'regex', 'mauve-text', 'faiss-cpu'], check=True)
'''

DATA = r'''
import urllib.request
os.makedirs('data', exist_ok=True)
if NEED_TS:
    B = 'https://huggingface.co/datasets/hazyresearch/ncm-datasets/resolve/main/tinystories/'
    for f in ['bpe_vocab_8192.pickle', 'bpe_merges_8192.pickle', 'valid_bpe_8192.pickle',
              'train_bpe_8192_offsets.npy', 'train_bpe_8192.bin']:
        p = f'{D}/{f}'
        if not os.path.exists(p):
            print('downloading', f, flush=True)
            urllib.request.urlretrieve(B + f, p + '.tmp'); os.replace(p + '.tmp', p)
    if not os.path.exists('data/tinystories'):
        os.symlink(D, 'data/tinystories')
    # local copy of the big stream for fast random access (Drive reads are slow)
    if not os.path.exists('/content/ts'):
        os.makedirs('/content/ts')
        for f in os.listdir(D):
            shutil.copy(f'{D}/{f}', f'/content/ts/{f}')
if 'ckpt' in NEEDS or 'wmt14' in NEEDS:
    ART = f'{DRIVE}/artifacts'; os.makedirs(ART, exist_ok=True)   # released checkpoints, cached on Drive
    if not os.path.exists('artifacts'):
        os.symlink(ART, 'artifacts')
    subprocess.run('bash scripts/fetch_assets.sh checkpoints', shell=True, check=True)
    if 'wmt14' in NEEDS:
        subprocess.run('bash scripts/fetch_assets.sh wmt14', shell=True, check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'sacrebleu'], check=True)
S = 'https://huggingface.co/datasets/hazyresearch/ncm-datasets/resolve/main/sudoku/'
for lvl in ['easy', 'medium', 'hard']:
    os.makedirs(f'data/sudoku/{lvl}', exist_ok=True)
    for f in ['puzzles.pickle', 'solutions.pickle']:
        if not os.path.exists(f'data/sudoku/{lvl}/{f}'):
            urllib.request.urlretrieve(f'{S}{lvl}/{f}', f'data/sudoku/{lvl}/{f}')
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
'''

RUN = r'''
for seed in SEEDS:
    for c in CMDS:
        c = c.format(R=R, D=D, S=seed)
        if 'scripts/train.py ' in c or 'scripts/train_gen.py ' in c:
            c += ' --data_dir /content/ts'
        if '{S}' not in str(CMDS) and seed != SEEDS[0]:
            continue   # seed-independent notebook: run once
        print('>>>', c, flush=True)
        t = time.time()
        p = subprocess.run(c, shell=True)
        print(f'<<< exit {p.returncode} after {(time.time()-t)/60:.1f} min', flush=True)
        if p.returncode != 0:
            raise SystemExit('command failed; fix and re-run this cell (it resumes)')
'''

COLLECT = r'''
out = f'{DRIVE}/results/__NAME__'; os.makedirs(out, exist_ok=True)
for seed in SEEDS:
    for c in CMDS:
        if '--run_dir ' not in c:
            continue
        rd = c.split('--run_dir ')[1].split()[0].format(R=R, S=seed)
        name = os.path.basename(rd)
        for f in ['result.json', 'metrics.jsonl', 'config.yaml', 'calib.json', 'eval_final.json', 'eval_15k.json', 'samples_final.json']:
            if os.path.exists(f'{rd}/{f}'):
                shutil.copy(f'{rd}/{f}', f'{out}/{name}__{f}')
for f in os.listdir('results'):   # diagnostics write straight into the repo's results/
    if f.startswith(('diag_', 'b2_jacobi_', 'sudoku_', 'wmt14_')) and os.path.getmtime(f'results/{f}') > T_START:
        shutil.copy(f'results/{f}', f'{out}/{f}')
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
json.dump({'run': '__NAME__', 'gpu': gpu, 'finished': time.ctime()}, open(f'{out}/_meta.json', 'w'))
print('Results copied to', out, '-> please drop this folder into the repo under results/colab/')
'''


def cell(src: str, kind: str = "code") -> dict:
    lines = src.strip("\n").splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def make(name: str, spec: dict) -> dict:
    head = (f"# NCM research run: `{name}`\n\n{spec['desc']}\n\n"
            f"- Recommended GPU: **{spec['gpu']}**. Estimated cost: **{spec['units']} compute units**.\n"
            "- Runtime > Change runtime type > GPU, then Runtime > Run all.\n"
            "- If Colab disconnects, just Run all again: every command resumes from its last Drive checkpoint.\n"
            f"- When done, copy `MyDrive/ncm-research/results/{name}/` into the repo's `results/colab/`.")
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            cell(head, "markdown"),
            cell(SETUP.replace("__REPO__", REPO).replace("__BRANCH__", BRANCH)
                 .replace("__CMDS__", json.dumps(spec["cmds"], indent=1))
                 .replace("__SEEDS__", json.dumps(spec.get("seeds", [0])))
                 .replace("__NEEDS__", json.dumps(spec.get("needs", [])))),
            cell(DATA),
            cell(RUN),
            cell(COLLECT.replace("__NAME__", name)),
        ],
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for name, spec in RUNS.items():
        with open(os.path.join(here, f"{name}.ipynb"), "w") as f:
            json.dump(make(name, spec), f, indent=1)
        print("wrote", f"notebooks/{name}.ipynb")
