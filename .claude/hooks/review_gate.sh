#!/usr/bin/env bash
# Claude Code PreToolUse hook (matcher: Bash). Exit 2 blocks the call and shows stderr to Claude.
# 1. Blocks `git push` that would put gated-path changes on the run branch (Colab clones it) without a PR.
# 2. Blocks PR merges that bypass scripts/review/codex_pr.py merge (which enforces the Codex gate).
# Fails closed: if the hook input cannot be parsed and it mentions a push or a merge, the call is blocked.
set -uo pipefail
RUN_BRANCH="claude/ncm-research"
GATED_RE='^(src|scripts|configs|notebooks|assets)/'

block() { echo "BLOCKED by review gate: $*" >&2; echo "See CLAUDE.md, 'Codex review'." >&2; exit 2; }

raw="$(cat)"
field() {  # field command|cwd: read .tool_input.command or .cwd from the hook JSON, via jq or python
  local q
  if [[ "$1" == command ]]; then q='.tool_input.command // ""'; else q='.cwd // ""'; fi
  if command -v jq >/dev/null 2>&1; then printf '%s' "$raw" | jq -r "$q" && return; fi
  local py; for py in python3 python; do
    command -v "$py" >/dev/null 2>&1 || continue
    printf '%s' "$raw" | "$py" -c "import json,sys; d=json.load(sys.stdin); \
print(d.get('tool_input',{}).get('command','') if sys.argv[1]=='command' else d.get('cwd',''))" "$1" \
      2>/dev/null && return
  done
  return 1
}
if ! cmd="$(field command)"; then
  [[ "$raw" =~ push|merge ]] && block "cannot parse the hook input (no jq or python) and it mentions a push or merge."
  exit 0
fi
cwd="$(field cwd 2>/dev/null || true)"

# --- merges must go through the gated merge command ---------------------------------
if [[ "$cmd" =~ gh[[:space:]]+pr[[:space:]]+merge ]] || \
   { [[ "$cmd" =~ gh[[:space:]]+api ]] && [[ "$cmd" =~ /pulls/[0-9]+/merge ]]; }; then
  block "merge PRs with: python scripts/review/codex_pr.py merge <PR>"
fi

# --- pushes to the run branch ----------------------------------------------------
push_re='git[[:space:]]+(-C[[:space:]]+([^[:space:];&|]+)[[:space:]]+)?push([^;&|]*)'
if [[ "$cmd" =~ $push_re ]]; then
  gitdir="${BASH_REMATCH[2]}"; rest="${BASH_REMATCH[3]}"
  # The repo the push runs in: the hook's cwd, then a leading `cd DIR`, then `git -C DIR`.
  isabs() { [[ "$1" == /* || "$1" =~ ^[A-Za-z]: ]]; }
  dir="${cwd:-${CLAUDE_PROJECT_DIR:-.}}"
  cd_re='^[[:space:]]*cd[[:space:]]+("[^"]+"|[^[:space:];&|]+)'
  if [[ "$cmd" =~ $cd_re ]]; then d="${BASH_REMATCH[1]//\"/}"; isabs "$d" && dir="$d" || dir="$dir/$d"; fi
  if [[ -n "$gitdir" ]]; then isabs "$gitdir" && dir="$gitdir" || dir="$dir/$gitdir"; fi
  cd "$dir" 2>/dev/null || block "cannot enter $dir to check the push."
  cur="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo)"
  targets_run=0
  [[ "$rest" == *"$RUN_BRANCH"* || "$rest" == *--all* || "$rest" == *--mirror* ]] && targets_run=1
  # bare `git push` / `git push origin` / `git push origin HEAD` push the current branch
  if [[ "$cur" == "$RUN_BRANCH" ]]; then
    read -r -a toks <<< "$rest"
    args=(); for t in "${toks[@]}"; do [[ "$t" == -* ]] || args+=("$t"); done
    if (( ${#args[@]} < 2 )) || [[ "${args[1]}" == HEAD ]]; then targets_run=1; fi
  fi
  if (( targets_run )); then
    # the pushed commit: `SRC:claude/ncm-research` pushes SRC, anything else pushes HEAD
    src="HEAD"
    refspec_re="([^[:space:]:+]+):(refs/heads/)?$RUN_BRANCH([[:space:]]|$)"
    [[ "$rest" =~ $refspec_re ]] && src="${BASH_REMATCH[1]}"
    git fetch -q origin "$RUN_BRANCH" 2>/dev/null || true
    if ! all="$(git diff --name-only "origin/$RUN_BRANCH...$src" 2>/dev/null)"; then
      block "cannot compare $src with origin/$RUN_BRANCH (fetch failed or no merge base); refusing to push to it."
    fi
    changed="$(printf '%s\n' "$all" | grep -E "$GATED_RE" | grep -v '^scripts/review/' || true)"
    if [[ -n "$changed" ]]; then
      block "this push would put unreviewed code on $RUN_BRANCH:
$(printf '%s\n' "$changed" | head -n 15)
Push to a claude/wip-<topic> branch and open a PR: python scripts/review/codex_pr.py open --head <branch> --title ..."
    fi
  fi
fi
exit 0
