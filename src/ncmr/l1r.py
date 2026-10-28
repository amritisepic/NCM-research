"""L1R (docs/preregistration.md, "L1R -- direction-1 kill: balanced L1 target reassignment"): arm M2R.

M2R is M2 (frozen-wrong proposals, same data order / mask stream / replacement stream / micro-batch partition) except the
level-1 TARGETS of a selected set of root-corrupted rows: a row whose level-0 slot is replaced (canonical root a_i,
supplied root u_i != a_i) gets the canonical L1 block y_j of another selected row j whose canonical root is a_j = u_i, so
the (corrupted root, L1 target) association becomes a canonical one while every corrupted input stays M2's.

  1. `replay` re-runs the run's own data, mask and replacement streams (a fresh numpy stream, a fresh dry `MechArm`, no
     network forward) for every optimizer step and micro-batch and records, for each row with a replaced level-0 code,
     (step, row, a, u, y). The frozen proposer's level >= 1 draws consume the replacement generator by shape only
     (`MechArm.skip_fine`); nothing is drawn from the global torch RNG or the training data stream.
  2. `select_balanced` = the largest balanced subset of the multigraph a -> u (every root has as many selected outgoing as
     incoming rows): a maximum circulation, a linear program on a network matrix (integral), solved by HiGHS; integrality
     and exact flow conservation are verified, else it fails closed. Within each (a, u) pair the rows are drawn at random
     (seeded). `assign_donors` then gives, within each root r, the selected receivers (u_i = r) the blocks of the selected
     donors (a_j = r) by a random permutation. Only roots and masks enter, never target content or a score.
  3. `L1RPlan.reassign` runs in training: it checks every micro-batch's canonical roots, supplied roots and canonical L1
     blocks against the replay (raises `L1RMismatch` on any difference: this guards the shape-only consumption claim on the
     GPU) and then replaces the receivers' L1 targets. Inputs, masks, replacement codes, other targets are untouched.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

SEED_OFFSET = 314159           # selection / assignment stream: np.random.default_rng(seed + SEED_OFFSET)


class L1RMismatch(RuntimeError):
    """The training stream differs from the replay (or the resumed plan from the checkpoint's): stop."""


class L1RCoverageError(RuntimeError):
    """coverage = selected / root-corrupted rows is below the registered minimum: the run is killed before training."""


# ------------------------------------------------------------------------------------------------------- replay
def replay(next_batch, arm, steps: int, accum: int, out_vocab: int | None, canon_fn=None) -> dict:
    """Replay `steps` optimizer steps. `next_batch()` returns the step's full batch of windows (the caller's fresh numpy
    stream, the same call the training loop makes); `arm` is a fresh dry `MechArm` of the run's seed. For each step:
    the full-batch mask plan, then per micro-batch (torch.chunk(batch, accum), as the loop): canonical codes, the level-0
    replacement draw (`MechArm.draw_root`, the code `proposals` runs) and, if `out_vocab` is not None, the level >= 1
    draws' consumption of the replacement generator (`MechArm.skip_fine`). out_vocab=None skips that consumption (the
    CPU coverage estimate, whose replacement stream is simulated). Returns int64 arrays step / row / a / u (N,) and
    y (N, levels[1]) of the rows whose level-0 slot is masked, in (step, row) order, and the full batch size."""
    assert arm.l0 == 1, "L1R is defined for a single level-0 root code"
    canon_fn = canon_fn or arm.canonical
    w1 = arm.levels[1]
    out = {k: [] for k in ("step", "row", "a", "u", "y")}
    B = None
    with torch.no_grad():
        for s in range(steps):
            xb_full = next_batch()
            B = len(xb_full) if B is None else B
            assert len(xb_full) == B
            plan = arm.draw_plan(B)
            row0 = 0
            for xb in torch.chunk(xb_full, accum):
                nb = len(xb)
                canon = canon_fn(xb)
                r0 = arm.draw_root(canon)
                if out_vocab is not None:
                    arm.skip_fine(nb, out_vocab)
                m = plan.masks[0][row0:row0 + nb, 0]
                idx = m.nonzero().flatten()
                out["step"].append(np.full(len(idx), s, dtype=np.int64))
                out["row"].append((idx + row0).cpu().numpy().astype(np.int64))
                out["a"].append(canon[idx, 0].cpu().numpy().astype(np.int64))
                out["u"].append(r0[idx, 0].cpu().numpy().astype(np.int64))
                out["y"].append(canon[idx, 1:1 + w1].cpu().numpy().astype(np.int64))
                row0 += nb
    rec = {k: (np.concatenate(v) if v else np.zeros((0, w1) if k == "y" else (0,), dtype=np.int64)) for k, v in out.items()}
    rec["B"], rec["steps"] = int(B or 0), int(steps)
    return rec


# ------------------------------------------------------------------------------------------------------- selection
def max_circulation(a: np.ndarray, u: np.ndarray):
    """Largest balanced sub-multigraph of the arcs a_i -> u_i (a_i != u_i): returns the distinct arcs (sorted by (a, u)),
    their multiplicities and the integral flow on each (0 <= f <= multiplicity, in = out at every node, total maximal).
    A linear program on a network (node-arc incidence) matrix, hence integral at a vertex; verified, else raises."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    a, u = np.asarray(a, dtype=np.int64), np.asarray(u, dtype=np.int64)
    assert a.shape == u.shape and a.ndim == 1 and bool((a != u).all()), "arcs must have a != u"
    if len(a) == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, z
    base = int(max(a.max(), u.max())) + 1
    arcs, cap = np.unique(a * base + u, return_counts=True)
    ea, eu = arcs // base, arcs % base
    nodes, inv = np.unique(np.concatenate([ea, eu]), return_inverse=True)
    n_arc = len(arcs)
    ia, iu = inv[:n_arc], inv[n_arc:]
    cols = np.arange(n_arc)
    A = coo_matrix((np.concatenate([np.ones(n_arc), -np.ones(n_arc)]), (np.concatenate([ia, iu]), np.concatenate([cols, cols]))),
                   shape=(len(nodes), n_arc)).tocsr()                       # out-arcs +1, in-arcs -1: A f = 0
    res = linprog(-np.ones(n_arc), A_eq=A, b_eq=np.zeros(len(nodes)), bounds=np.stack([np.zeros(n_arc), cap.astype(float)], 1),
                  method="highs")
    if res.status != 0:
        raise RuntimeError(f"max-circulation LP failed: {res.message}")
    f = np.rint(res.x).astype(np.int64)
    if float(np.abs(res.x - f).max()) > 1e-6:
        raise RuntimeError("max-circulation LP solution is not integral")
    bal = np.zeros(len(nodes), dtype=np.int64)
    np.add.at(bal, ia, f)
    np.add.at(bal, iu, -f)
    if bool((bal != 0).any()) or bool((f < 0).any()) or bool((f > cap).any()):
        raise RuntimeError("max-circulation LP solution violates conservation or the capacities")
    return ea, eu, cap, f


def select_balanced(a: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N,) bool: the rows kept. Rows must already be in (step, row) order; within each (a, u) pair, which rows carry
    the pair's flow is drawn at random from `rng`."""
    a, u = np.asarray(a, dtype=np.int64), np.asarray(u, dtype=np.int64)
    sel = np.zeros(len(a), dtype=bool)
    if len(a) == 0:
        return sel
    ea, eu, cap, f = max_circulation(a, u)
    base = int(max(a.max(), u.max())) + 1
    order = np.argsort(a * base + u, kind="stable")                         # rows grouped by pair, (step, row) order inside
    start = np.concatenate([[0], np.cumsum(cap)])
    for p in np.nonzero(f)[0]:                                              # pairs in sorted (a, u) order
        rows = order[start[p]:start[p + 1]]
        sel[rows[rng.choice(len(rows), int(f[p]), replace=False)]] = True
    return sel


def assign_donors(a: np.ndarray, u: np.ndarray, sel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N,) int64: for each selected row i (a receiver of root u_i) the index of the selected donor j with a_j = u_i whose
    L1 block it gets (random permutation within each root, roots in sorted order); -1 for unselected rows. Every selected
    row is a donor exactly once and a receiver exactly once. Raises if a root is unbalanced."""
    a, u = np.asarray(a, dtype=np.int64), np.asarray(u, dtype=np.int64)
    donor = np.full(len(a), -1, dtype=np.int64)
    S = np.nonzero(sel)[0]
    for r in np.unique(np.concatenate([a[S], u[S]])):
        recv, don = S[u[S] == r], S[a[S] == r]
        if len(recv) != len(don):
            raise RuntimeError(f"root {int(r)} is unbalanced: {len(recv)} selected incoming, {len(don)} outgoing")
        donor[recv] = don[rng.permutation(len(don))]
    return donor


class L1RPlan:
    """The replayed records and the selection. Arrays are in (step, row) order; `donor[i] >= 0` marks a receiver."""

    ARRAYS = ("step", "row", "a", "u", "y", "donor")

    def __init__(self, rec: dict, donor: np.ndarray):
        self.B, self.steps = int(rec["B"]), int(rec["steps"])
        self.step, self.row, self.a, self.u, self.y = (np.ascontiguousarray(rec[k], dtype=np.int64) for k in ("step", "row", "a", "u", "y"))
        self.donor = np.ascontiguousarray(donor, dtype=np.int64)
        self.key = self.step * self.B + self.row
        assert bool((np.diff(self.key) > 0).all()), "records must be in strictly increasing (step, row) order"

    # ------------------------------------------------------------------ summary / identity
    @property
    def n_records(self) -> int:
        return len(self.a)

    @property
    def n_selected(self) -> int:
        return int((self.donor >= 0).sum())

    @property
    def coverage(self) -> float:
        return self.n_selected / self.n_records if self.n_records else 0.0

    @property
    def sha256(self) -> str:
        h = hashlib.sha256(f"l1r-plan-v1 B={self.B} steps={self.steps}".encode())
        for k in self.ARRAYS:
            arr = np.ascontiguousarray(getattr(self, k), dtype="<i8")
            h.update(f"|{k}{arr.shape}|".encode() + arr.tobytes())
        return h.hexdigest()

    def summary(self) -> dict:
        return {"coverage": self.coverage, "n_root_corrupted_rows": self.n_records, "n_selected": self.n_selected,
                "n_roots": int(len(np.unique(np.concatenate([self.a, self.u])))) if self.n_records else 0,
                "batch_size": self.B, "steps": self.steps, "plan_sha256": self.sha256}

    def save(self, run_dir: str):
        np.savez_compressed(os.path.join(run_dir, "l1r_plan.npz"), B=self.B, steps=self.steps,
                            **{k: getattr(self, k) for k in self.ARRAYS})
        self.write_summary(run_dir)

    def write_summary(self, run_dir: str, **extra):
        with open(os.path.join(run_dir, "l1r_plan.json"), "w") as f:
            json.dump({**self.summary(), **extra}, f, indent=1)

    # ------------------------------------------------------------------ training
    @torch.no_grad()
    def reassign(self, step: int, mplan, canon: torch.Tensor, repl: torch.Tensor, targets: torch.Tensor, l0: int, w1: int):
        """One micro-batch: `mplan` its MaskPlan (rows row0 .. row0 + nb of the step's batch), `canon` / `repl` the
        canonical and replacement codes (nb, S), `targets` the canonical labels (nb, S). Checks the root-corrupted rows
        against the replay, then returns (targets with the receivers' L1 blocks replaced, number of receivers)."""
        nb = canon.shape[0]
        m = mplan.masks[0][:, 0]
        lo_key = step * self.B + mplan.row0
        lo, hi = np.searchsorted(self.key, [lo_key, lo_key + nb])
        keys = lo_key + m.nonzero().flatten().cpu().numpy().astype(np.int64)
        want = {"row keys": (keys, self.key[lo:hi]),
                "canonical roots": (canon[m, 0].cpu().numpy().astype(np.int64), self.a[lo:hi]),
                "supplied roots": (repl[m, 0].cpu().numpy().astype(np.int64), self.u[lo:hi]),
                "canonical L1 blocks": (canon[m, l0:l0 + w1].cpu().numpy().astype(np.int64), self.y[lo:hi])}
        for name, (got, ref) in want.items():
            if got.shape != ref.shape or not np.array_equal(got, ref):
                raise L1RMismatch(f"M2R: the {name} of step {step}, rows {mplan.row0}..{mplan.row0 + nb - 1} differ from the "
                                  "pre-training replay; the replay no longer describes this run: stop")
        recv = lo + np.nonzero(self.donor[lo:hi] >= 0)[0]
        if len(recv) == 0:
            return targets, 0
        don = self.donor[recv]
        assert bool((self.a[don] == self.u[recv]).all()) and bool((don != recv).all())
        out = targets.clone()
        rows = torch.as_tensor(self.row[recv] - mplan.row0, device=targets.device)
        out[rows, l0:l0 + w1] = torch.as_tensor(self.y[don], dtype=targets.dtype, device=targets.device)
        return out, int(len(recv))


def build_plan(rec: dict, seed: int) -> L1RPlan:
    """Selection and assignment from the replay's records (roots only), seeded by `seed + SEED_OFFSET`."""
    rng = np.random.default_rng(int(seed) + SEED_OFFSET)
    sel = select_balanced(rec["a"], rec["u"], rng)
    return L1RPlan(rec, assign_donors(rec["a"], rec["u"], sel, rng))


def gate(plan: L1RPlan, coverage_min: float, run_dir: str | None = None, write: bool = True):
    """Run-time gate: coverage >= coverage_min, else the summary is written (if `write`) and L1RCoverageError raised.
    On success the plan is saved (if `write`)."""
    ok = plan.coverage >= coverage_min
    if write and run_dir:
        if ok:
            plan.save(run_dir)
        else:
            plan.write_summary(run_dir, gate="FAILED", coverage_min=float(coverage_min))
    if not ok:
        raise L1RCoverageError(f"L1R coverage gate failed: {plan.n_selected} of {plan.n_records} root-corrupted rows "
                               f"selected = {plan.coverage:.4f} < {coverage_min}; the arm is killed before training")
