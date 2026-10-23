#!/usr/bin/env python3
"""Codex review over GitHub pull requests (REST only; GraphQL is unavailable in cloud sessions).

Subcommands
  open    --head BRANCH --title T [--body-file F]   open a PR into the run branch
  request PR [--focus TEXT] [--audit-file F]        post "@codex review" (or a deep-audit brief)
  wait    PR [--timeout MIN]                        wait for Codex to respond after the latest request
  status  PR                                        print gate status (exit 0 = merge allowed)
  merge   PR                                        merge only if the gate passes

Gate (status / merge):
  - PRs that touch no gated path (GATED below) pass without review.
  - Otherwise Codex must have responded to the PR's current head commit, and every P0/P1 comment
    Codex left on that commit must be listed (by comment id) in docs/reviews/triage_pr<N>.md on the
    PR head, with a verdict.
Exit codes: 0 ok, 1 gate fails / changes needed, 2 usage error, 3 timeout, 4 Codex never acknowledged.
"""
import argparse, json, re, subprocess, sys, time
from datetime import datetime, timezone

REPO = "amritisepic/NCM-research"
RUN_BRANCH = "claude/ncm-research"           # Colab notebooks clone this branch
GATED = ("src/", "scripts/", "configs/", "notebooks/", "assets/")
UNGATED = ("scripts/review/",)               # review tooling itself; never run by Colab
BOT = re.compile(r"codex", re.I)              # Codex's GitHub bot login contains "codex"
SEVERE = re.compile(r"\bP[01]\b")             # Codex marks findings with P0 / P1 badges
REQ_MARK = "<!-- codex-request -->"


def gh(path, method="GET", data=None, paginate=False):
    cmd = ["gh", "api", "-X", method, "-H", "Accept: application/vnd.github+json"]
    if paginate:
        cmd.append("--paginate")
    cmd.append(path)
    inp = None
    if data is not None:
        cmd += ["--input", "-"]
        inp = json.dumps(data)
    r = subprocess.run(cmd, input=inp, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"gh api {method} {path} failed: {r.stderr.strip() or r.stdout.strip()}")
    out = r.stdout.strip()
    if not out:
        return None
    if paginate and out.startswith("[") and "][" in out:      # gh concatenates pages
        out = "[" + out.replace("][", ",")[1:]
    return json.loads(out)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def pr_info(n):
    pr = gh(f"repos/{REPO}/pulls/{n}")
    head = pr["head"]["sha"]
    commit = gh(f"repos/{REPO}/commits/{head}")
    files = [f["filename"] for f in gh(f"repos/{REPO}/pulls/{n}/files?per_page=100", paginate=True)]
    return pr, head, ts(commit["commit"]["committer"]["date"]), files


def codex_activity(n, head, head_time):
    reviews = [r for r in gh(f"repos/{REPO}/pulls/{n}/reviews?per_page=100", paginate=True)
               if BOT.search(r["user"]["login"])]
    rcomments = [c for c in gh(f"repos/{REPO}/pulls/{n}/comments?per_page=100", paginate=True)
                 if BOT.search(c["user"]["login"])]
    icomments = [c for c in gh(f"repos/{REPO}/issues/{n}/comments?per_page=100", paginate=True)
                 if BOT.search(c["user"]["login"])]
    on_head = ([r for r in reviews if r.get("commit_id") == head] +
               [c for c in icomments if ts(c["created_at"]) >= head_time])
    severe = [c for c in rcomments if c.get("commit_id") == head and SEVERE.search(c["body"])]
    severe += [r for r in reviews if r.get("commit_id") == head and SEVERE.search(r.get("body") or "")]
    return on_head, severe


def triage_ids(n, head):
    path = f"docs/reviews/triage_pr{n}.md"
    r = subprocess.run(["gh", "api", f"repos/{REPO}/contents/{path}?ref={head}",
                        "-H", "Accept: application/vnd.github.raw"], capture_output=True, text=True)
    return (set(re.findall(r"\b\d{6,}\b", r.stdout)), path) if r.returncode == 0 else (None, path)


def gate(n, verbose=True):
    pr, head, head_time, files = pr_info(n)
    say = print if verbose else (lambda *a, **k: None)
    if pr["base"]["ref"] != RUN_BRANCH:
        say(f"PR #{n} targets {pr['base']['ref']}, not {RUN_BRANCH}; gate not applied.")
        return True
    gated = [f for f in files if f.startswith(GATED) and not f.startswith(UNGATED)]
    if not gated:
        say(f"PASS: PR #{n} touches no gated path ({', '.join(GATED)}).")
        return True
    on_head, severe = codex_activity(n, head, head_time)
    if not on_head:
        say(f"FAIL: no Codex response on head {head[:9]}. Run: codex_pr.py request {n} && codex_pr.py wait {n}")
        return False
    if not severe:
        say(f"PASS: Codex reviewed {head[:9]} with no P0/P1 findings.")
        return True
    ids, path = triage_ids(n, head)
    missing = [c for c in severe if ids is None or str(c["id"]) not in ids]
    if missing:
        say(f"FAIL: {len(missing)} P0/P1 finding(s) on {head[:9]} not triaged in {path}:")
        for c in missing:
            first = (c.get("body") or "").strip().splitlines()[0][:140] if c.get("body") else ""
            say(f"  id {c['id']}  {c.get('path', '(review body)')}:{c.get('line') or ''}  {first}")
        return False
    say(f"PASS: all {len(severe)} P0/P1 finding(s) on {head[:9]} triaged in {path}.")
    return True


def cmd_open(a):
    body = open(a.body_file).read() if a.body_file else ""
    pr = gh(f"repos/{REPO}/pulls", "POST", {"title": a.title, "head": a.head, "base": RUN_BRANCH, "body": body})
    print(f"PR #{pr['number']} {pr['html_url']}")


def cmd_request(a):
    if a.audit_file:
        text = "@codex " + open(a.audit_file).read().strip()
    else:
        text = "@codex review" + (f" {a.focus}" if a.focus else "")
    c = gh(f"repos/{REPO}/issues/{a.pr}/comments", "POST", {"body": f"{text}\n\n{REQ_MARK}"})
    print(f"requested (comment {c['id']}) at {c['created_at']}")


def cmd_wait(a):
    reqs = [c for c in gh(f"repos/{REPO}/issues/{a.pr}/comments?per_page=100", paginate=True)
            if REQ_MARK in c["body"]]
    if not reqs:
        sys.exit("no Codex request on this PR; run `request` first")
    req = reqs[-1]
    t0, deadline, acked = ts(req["created_at"]), time.time() + 60 * a.timeout, False
    while time.time() < deadline:
        if not acked:
            reacts = gh(f"repos/{REPO}/issues/comments/{req['id']}/reactions")
            acked = any(BOT.search(r["user"]["login"]) for r in reacts or [])
            if not acked and (datetime.now(timezone.utc) - t0).total_seconds() > 600:
                print("Codex has not acknowledged the request after 10 min: code review is probably not "
                      "enabled for this repository (Codex settings -> Code review).")
                sys.exit(4)
        pr, head, head_time, _ = pr_info(a.pr)
        reviews = [r for r in gh(f"repos/{REPO}/pulls/{a.pr}/reviews?per_page=100", paginate=True)
                   if BOT.search(r["user"]["login"]) and ts(r["submitted_at"]) >= t0]
        icom = [c for c in gh(f"repos/{REPO}/issues/{a.pr}/comments?per_page=100", paginate=True)
                if BOT.search(c["user"]["login"]) and ts(c["created_at"]) >= t0]
        if reviews or icom:
            print(f"Codex responded ({len(reviews)} review(s), {len(icom)} comment(s)).")
            gate(a.pr)
            return
        time.sleep(60)
    print(f"timed out after {a.timeout} min")
    sys.exit(3)


def cmd_status(a):
    sys.exit(0 if gate(a.pr) else 1)


def cmd_merge(a):
    if not gate(a.pr):
        sys.exit(1)
    pr = gh(f"repos/{REPO}/pulls/{a.pr}")
    r = gh(f"repos/{REPO}/pulls/{a.pr}/merge", "PUT",
           {"merge_method": "merge", "sha": pr["head"]["sha"]})
    print(r.get("message", r))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("open"); o.add_argument("--head", required=True); o.add_argument("--title", required=True)
    o.add_argument("--body-file")
    r = sp.add_parser("request"); r.add_argument("pr", type=int); r.add_argument("--focus")
    r.add_argument("--audit-file")
    w = sp.add_parser("wait"); w.add_argument("pr", type=int); w.add_argument("--timeout", type=int, default=90)
    for name in ("status", "merge"):
        s = sp.add_parser(name); s.add_argument("pr", type=int)
    a = p.parse_args()
    {"open": cmd_open, "request": cmd_request, "wait": cmd_wait, "status": cmd_status, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
