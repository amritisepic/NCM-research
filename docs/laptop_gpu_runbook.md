# Running jobs on a local GPU (e.g. the RTX 4050 laptop)

The cloud agent cannot reach a local machine. GPU work on the laptop runs in a separate Claude Code
session started *on the laptop*, which reads `STATE.md` and picks up from there.

## One-time setup (laptop terminal)
```bash
git clone -b claude/ncm-research https://github.com/amritisepic/NCM-research.git
cd NCM-research
# public repo: commit with GitHub's no-reply address, not a personal email
git config user.email "<your-github-noreply-address>"
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build matching your driver
pip install numpy scipy scikit-learn omegaconf hydra-core regex "transformers>=4.51" datasets sacrebleu mauve-text faiss-cpu
bash scripts/fetch_assets.sh tinystories     # ~1.1 GB; add `sudoku` / `checkpoints` if needed
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
Then start the session with `claude remote-control` in this folder (it appears in the Claude Code app),
or open the folder in the Claude Desktop app, and tell it: "Read STATE.md and run the laptop queue."

## Laptop queue (6 GB VRAM)
Use `grad_accum` in a config's `train:` block if a batch does not fit (same data per optimizer step).
1. P1 / C6 tokenizers and generators at the small scale run as-is (`configs/pilot/*.yaml`).
2. B1 kill test (self-correcting NCM) — configs added when implemented; see `docs/phase1b_reevaluation.md`.
Results: commit `runs/<run>/result.json` + `metrics.jsonl` copies under `results/laptop/<run>/`,
append a `log.md` entry, `git pull --rebase` before every push (two sessions share this branch).
