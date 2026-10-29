"""Run a complete PD/E0 command under shared cumulative GPU-minute caps, including final evaluation and resumes.

Unclosed reservations are charged at their entire cap. This fails conservatively after a runtime disappears and
also prevents concurrent notebooks from promising the same remaining minutes. Ledger writes use an exclusive lock.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

CAPS = {"pd": ({"p": 25, "d": 25, "u": 15}, 60), "e0": ({"u0": 30, "e0": 30}, 55)}


def available(ledger, experiment, arm):
    per_arm, total = CAPS[experiment]
    events = ledger.get("sessions", [])
    charged = lambda event: event.get("elapsed_s", event["reserved_s"])
    return max(0.0, min(per_arm[arm] * 60 - sum(charged(e) for e in events if e["arm"] == arm),
                        total * 60 - sum(charged(e) for e in events)))


def update(path, transform):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("pilot budget ledger busy; retry after the other notebook finishes its ledger write")
    try:
        os.close(descriptor)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"sessions": []}
        result = transform(data)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        return result
    finally:
        lock.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=CAPS, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.arm not in CAPS[a.experiment][0]:
        ap.error("arm differs from registered experiment")
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        ap.error("a trainer command is required")
    path = Path(a.run_root) / f"{a.experiment}_budget.json"
    identifier = uuid.uuid4().hex

    def reserve(ledger):
        seconds = available(ledger, a.experiment, a.arm)
        if seconds <= 0:
            print("registered cumulative pilot budget exhausted; no command launched", flush=True)
            raise SystemExit(124)
        ledger["experiment"] = a.experiment
        ledger["sessions"].append({"id": identifier, "arm": a.arm, "reserved_s": seconds})
        return seconds

    seconds = update(path, reserve)
    started = time.monotonic()
    process = None
    try:
        process = subprocess.Popen(command, start_new_session=os.name != "nt")
        try:
            status = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
            print("registered cumulative pilot budget reached; final evaluation may be incomplete", flush=True)
            status = 124
    finally:
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
        elapsed = time.monotonic() - started

        def finish(ledger):
            event = next(e for e in ledger["sessions"] if e["id"] == identifier)
            event["elapsed_s"] = elapsed

        update(path, finish)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
