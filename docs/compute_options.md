# Compute options: launching notebooks from an agent, and GPUs beyond Colab (2026-10-28)

Status: research and a recommendation. Nothing is installed, bought, signed up for or launched. Installing a tool that
holds OAuth tokens for the user's Google account, creating accounts, buying compute and any agent-launched run are
the user's decisions (`docs/brief.md`; the 15-CU per-run rule is unchanged).

## 1. Launching Colab runs from Claude Code

**Today:** the user opens each notebook in `notebooks/`, picks the GPU, runs it, and copies
`results/colab/<notebook>/` into the repository.

**Requirements:** choose the GPU type (A100-40GB, L4; our notebooks refuse others where registered); run a repository
notebook end to end (clone, provenance cell, run, collect, release the runtime); survive a disconnect and resume
(checkpoints and provenance live on Drive); read results back; report CU spent; launch from a script on this Windows
laptop.

| Tool | What it is (checked 2026-10-28) | GPU choice | Windows | Verdict |
|---|---|---|---|---|
| **Colab CLI**, `googlecolab/google-colab-cli` (official, Apache-2.0, ~1.7k stars, last push 2026-10-26) | Provisions runtimes and tears them down (`colab new / stop`, `colab run` = provision, run, release), executes local scripts and notebooks against the VM kernel with outputs written back, uploads/downloads files, mounts Drive, `colab usage` shows the CU rate and balance | **Yes**: `--gpu` with T4, L4, G4, H100, A100 (README) | "supports **Linux and macOS** only. Windows is not supported" (README). WSL not mentioned; this laptop has WSL2 Debian | **Recommended, via WSL, after a pilot** |
| **colab-mcp**, `googlecolab/colab-mcp` (official MCP server, ~1.4k stars) | Bridges a local agent to a Colab notebook already open in the browser; does not provision runtimes | Not offered (GPU is whatever the open tab has) | Open issue reports a failure on Windows 11 | Not recommended. Needs the client to support `notifications/tools/list_changed`; an open issue reports Claude Code cannot connect |
| `mcp-server-colab-exec` (third-party, one maintainer, ~9 stars) | Allocates a runtime and executes Python; reuses the Colab VS Code extension's OAuth client | T4 or L4 only (no A100) | not stated | Rejected: no A100, supply-chain risk for a token on the user's account |
| `claude-colab` (third-party) | A Flask server inside a Colab notebook exposed through a tunnel | T4 (per a listing; repository not located) | n/a | Rejected: not located, tunnel exposes a shell |

**How the Colab CLI would run our notebooks** (from its README and design docs; not tested here):
1. One-time: `uv tool install google-colab-cli` inside WSL; `colab` authenticates with OAuth via a copy-paste code flow
   (works headless; refresh token cached in the WSL user's config folder). Scopes: `colaboratory`, `userinfo.email`,
   `openid` (plus `cloud-platform` under the `adc` provider). The README says some machine shapes need a Pro/Pro+
   entitlement; whether Pay As You Go CU suffice for A100 / L4 is **unverified** (the pilot settles it).
2. Per run: `colab new -s <run> --gpu A100` (or `L4`), `colab drivemount -s <run>`, execute the notebook
   (`colab exec`-style notebook execution, outputs written back to a local `.ipynb`), `colab download` the
   `results/colab/<notebook>/` files, `colab stop -s <run>`; `colab usage` before and after gives the CU spent.
3. **Limit 1: Drive needs a human click per VM.** `colab drivemount` triggers Google's Drive OAuth prompt in the
   browser for each new VM (the CLI's docs give it a 10-minute window). Our resume path keeps checkpoints and
   provenance on Drive, so a launch is "one click by the user" rather than fully unattended. Removing Drive (local
   checkpoints plus periodic `colab download`) would lose the last checkpoint if the VM dies, so it is not proposed.
4. **Limit 2: idle and lifetime rules are Colab's.** The CLI dropped its keep-alive daemon (2026-09-25); liveness
   follows kernel activity, so a long cell keeps the VM, and Colab's runtime lifetime limits still apply. Our
   notebooks' Drive checkpoints and refuse-to-resume-under-changed-code rule carry over unchanged: a resume is a new
   `colab new` + rerun of the same notebook.
5. **Limit 3: Windows.** Native Windows is unsupported; run it in WSL. The agent would call it as
   `wsl.exe -d Debian -- colab ...` from Git Bash.

**Pilot (after the user's go):** install in WSL, authenticate, then run the cheapest eval-only notebook end to end on
an L4 (`w2mgr_eval` for one seed is ~0.5 CU; a re-run of an already-read result is harmless but costs CU, so a smaller
smoke notebook that only runs the provenance cell and `nvidia-smi` is better: < 0.1 CU). Check: the GPU is the one
requested, provenance matches the repository SHA, results download into `results/colab/`, `colab usage` reports the
spend, a deliberate `colab stop` mid-run followed by a relaunch resumes from the Drive checkpoint.

**Rules if adopted** (proposed for the user): the agent launches only runs the user approved, per run or under a
standing rule the user states (for example "eval-only notebooks under 3 CU, once cleared by the Codex protocol"); no
run over 15 CU without per-run approval; every launch, its SHA and its CU spend logged in `docs/colab_runs.md`.

## 2. GPUs beyond Colab

**Our workload:** one GPU; bf16 training plus fp32 evaluation with TF32 off; checkpoint-and-resume (spot capacity is
usable); ~1-2 GB of data; script-driven launches wanted. **Measured:** 37.9 effective TFLOPS training on an A100-40GB
(52 PFLOP per 600-step batch-256 run), ~5.3 CU per A100-hour, ~1.71 per L4-hour (`STATE.md`, Compute).

**Normalisation:** for an all-in hourly price p and useful training throughput f TFLOPS,
$/A100-equivalent hour = p x 37.9 / f and $/EFLOP = p / (0.0036 f). Only the A100 figure is measured. The other f
values are the round-4 reviewer's workload guesses (`docs/research_review_2026-10-28_round4.md`, Part E: L4 15,
RTX A6000 25, RTX 4090 40, RTX 5090 60, H100 65), **not measurements**: any decision needs a benchmark first.

| Option | Price per GPU-hour (official page, read 2026-10-28) | $/A100-eq h; $/EFLOP at the assumed f | Notes |
|---|---|---|---|
| Colab A100-40GB | 5.3 CU/h x price per CU. $9.99 per 100 CU is **unverified** (the official page needs sign-in; secondhand sources agree) | 0.53; 3.88 | Current path; the user's panel gives the real price and balance |
| Colab L4 | 1.71 CU/h | 0.43; 3.16 (f 15 assumed) | Registered device for W2M-G / W2M-GR evals |
| Runpod Community RTX 4090 | $0.34 (Secure $0.89) | 0.32; 2.36 | Cheapest per assumed EFLOP; 24 GB may need a smaller micro-batch (keep the global batch); Community hosts vary in reliability |
| Runpod Community RTX 5090 | $0.69 (Secure $1.19) | 0.44; 3.19 | 32 GB |
| Runpod Community A100 SXM | $1.39 (Secure $1.79) | 1.39; 10.2 | |
| Thunder Compute RTX A6000 | $0.35, 100 GB disk included, egress $0 | 0.53; 3.89 | Persistent single-GPU instances, CLI |
| Thunder Compute A100-80GB / H100 SXM | $1.09 / $3.20 | 1.09; 7.99 / 1.87; 13.7 | |
| Verda A100-40GB spot / on-demand | $0.80 / $1.59 (Astra's $0.67 / $1.33 are out of date) | 0.80; 5.84 | Spot suits our resumable runs |
| Verda H100 SXM spot / on-demand | $2.35 / $4.70 | 1.37; 10.0 | |
| Lambda A100-40GB / H100 SXM / H100 PCIe (1 GPU) | $1.99 / $4.29 / $3.29 | 1.99; 14.6 | Research credits "up to $5,000" for qualifying researchers; eligibility of independent researchers not stated |
| Modal A100-40GB / H100 / L4 (serverless) | $2.10 / $3.95 / $0.80, GPU only; CPU and RAM extra | >= 2.10; >= 15.4 | $30/month free credit on the Starter plan (enough for small eval jobs); academic grants up to $10k, independent eligibility not stated; 24 h per function call |
| Vast.ai marketplace | Host-set; no price readable from the public page | 4090 beats Colab A100 below ~$0.56/h all-in | CLI/API; vet hosts |
| Kaggle | Free weekly GPU quota (T4 x2 / P100); figures not readable from the official page | n/a | P100 is unsupported by CUDA 13; not a target for our stack |
| AWS / GCP / Azure spot | No all-in quote verified | — | Highest setup and quota burden |

**What the money is:** the reviewer's six-run paper-ablation programme (6 x 3.5 EFLOP + 25%) is ~1020 CU on Colab,
i.e. **~$100** at $9.99 / 100 CU, and ~$60-80 on a Community 4090 *if* the assumed 40 TFLOPS held. Cash is not the
binding constraint at this scale; wall time (~25 A100-hours per run), availability, and the gates in
`docs/research_review_2026-10-28_round4.md` are.

## 3. Recommendation
1. **Keep Colab for everything in T0** (the <= 77-CU programme): the notebooks, provenance and resume rules exist and are
   audited. Moving would add porting and review work for no material saving.
2. **Automate Colab with the official CLI in WSL** after a < 0.1-CU smoke pilot the user approves. Expect one Drive
   click per VM. Skip colab-mcp and the third-party servers.
3. **Only if a paper-scale programme is approved:** benchmark one alternative for ~$2 (Runpod Community 4090 or Thunder
   A6000: 200 training steps of the registered trainer plus one fp32 eval group, same pinned PyTorch/CUDA), and run any
   matched comparison entirely on one device type. Never resume an L4 or A100 stream on a different GPU family.
4. **Credits:** the Lambda research-credit application and Modal's free Starter credit are the user's to apply for;
   neither is budgeted.

## 4. Colab CLI pilot (2026-10-28)
- **Install.** `google-colab-cli` 0.7.5 in WSL Debian via `uv tool install --python 3.12`. The CLI needs Python >= 3.12;
  Debian 12 ships 3.11.
- **Sign-in.** OAuth copy-paste flow by the user. The granted scopes are Colab, `drive.file`, email and profile. The token
  is in `~/.config/colab-cli/token.json` (mode 600).
- **Secret leak.** The CLI forces DEBUG logging to `~/.config/colab-cli/colab.log`, which writes the full OAuth token
  response there on every sign-in and refresh. Point the log at `/dev/null` (done here). An earlier grant whose tokens
  were exposed this way was revoked.
- **Smoke run.** `colab run --gpu L4` with a 5-second script, 13 s end to end:
  - it got an NVIDIA L4 (23 GB) with torch 2.11.0+cu130, the stack of the W2M-GR runs;
  - the remote head of `claude/ncm-research` matched the expected SHA;
  - the session was released automatically and no session remained;
  - the cost is below the balance display's resolution (97.55 before and after).
- **Not yet tested.** Drive mount (one browser click per VM), running a repository notebook end to end, and resuming
  after a stop. These get tested on the first approved eval run.
