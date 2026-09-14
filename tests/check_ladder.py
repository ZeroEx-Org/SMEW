# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the fallback ladder, which no benchmark case has ever reached.

Rung 2 had not fired once in the model's history -- not before F3.1, not after
F3.5 -- because the only condition that could trigger it was an absolute
residual nothing came within ten orders of magnitude of. F3.6 gave it the right
trigger, and rungs 3 to 5 besides. That leaves ~60 lines of recovery code that
every benchmark run walks straight past, which is the classic way to ship a
fallback that does not work: it is never wrong until the day it is needed.

So this makes each rung fire on purpose, and checks it produces the right answer
rather than merely returning one:

    1 -> 2  poison rung 1's exit code. Rung 2 must rescue it, and the whole
            trajectory must still match an unpoisoned run.
    2 -> 3  poison both Newton rungs. least_squares must rescue it.
    3 -> 4  force the full-length step to report failure. step() must sub-divide
            the interval, report rung 4, and mark n_substeps -- the mark
            smew.ledger uses to decline to gate a run.
    4 -> 5  make the gate unsatisfiable. The model must RAISE, and say which
            equation and by how much -- not print and carry on, which is what
            TC_SMEW does and is worse than raising, because it produces output
            that looks like a result.

Poisoning is by exit code, not by corrupting the answer: rung 1 is made to
report MINPACK status 5 while returning its starting point. That is exactly the
failure F3.1 measured 11 625 times across the suite and the model could not see,
so the test and the defect are the same shape.

    python tests/check_ladder.py [--t-end 30]
"""
import argparse
import os
import sys
import types
import warnings

import numpy as np

warnings.filterwarnings("ignore")
REPO = os.environ.get("SMEW_REPO", os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from smew import harness                                           # noqa: E402
import smew.biogeochem as bg                                       # noqa: E402

def poisoned(fail_calls):
    """(_step_once, root) where the named `root` calls within each step fail.

    Keyed on call order inside the step, not on the solver options: rungs 1 and
    2 used to be separable by their xtol, and F3.6 made them share one, because
    a fallback that differs in its tolerance as well as its starting point is
    testing two things at once.
    """
    real_root, real_step = bg.root, bg._step_once
    n = {"i": 0}

    def step_once(*a, **k):
        n["i"] = 0
        return real_step(*a, **k)

    def root(fun, y0, **kw):
        n["i"] += 1
        if n["i"] in fail_calls:
            y0 = np.asarray(y0, dtype=float)
            return types.SimpleNamespace(x=y0.copy(), status=5, success=False,
                                         nfev=1, njev=0)
        return real_root(fun, y0, **kw)
    return {"_step_once": step_once, "root": root}


def run(t_end, patch=None):
    saved = {k: getattr(bg, k) for k in ("root", "_step_once", "RES_RTOL")}
    try:
        for k, v in (patch or {}).items():
            setattr(bg, k, v)
        return harness.example_run(t_end=t_end)[0]
    finally:
        for k, v in saved.items():
            setattr(bg, k, v)


def rungs(data):
    r = np.asarray(data["solver_rung"], dtype=float)
    r = r[np.isfinite(r)]
    return {int(k): int(v) for k, v in zip(*np.unique(r, return_counts=True))}


def agrees(a, b, keys=("pH", "Alk", "Ca", "Mg", "Na", "K", "IC_tot"),
           tol=1e-8):
    worst, where = 0.0, ""
    for k in keys:
        x = np.asarray(a[k], dtype=float)
        y = np.asarray(b[k], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        sc = np.max(np.abs(x[m])) if m.any() else 0.0
        if not sc:
            continue
        d = float(np.max(np.abs(x[m] - y[m]))) / sc
        if d > worst:
            worst, where = d, k
    return worst, where, worst <= tol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--t-end", type=float, default=30)
    a = ap.parse_args(argv)
    ok = True

    base = run(a.t_end)
    print(f"baseline: rungs {rungs(base)}\n")

    # --- 1 -> 2 --------------------------------------------------------------
    d2 = run(a.t_end, poisoned({1}))
    r2 = rungs(d2)
    w, where, same = agrees(base, d2)
    hit = set(r2) == {2}
    print(f"1 -> 2  rungs {r2}")
    print(f"        trajectory vs baseline: {w:.2e} on {where or '-'}")
    print(f"        every step escalated: {hit}   answer preserved: {same}"
          f"   -> {'PASS' if hit and same else 'FAIL'}\n")
    ok &= hit and same

    # --- 2 -> 3 --------------------------------------------------------------
    d3 = run(a.t_end, poisoned({1, 2}))
    r3 = rungs(d3)
    w3, where3, same3 = agrees(base, d3)
    hit3 = set(r3) == {3}
    print(f"2 -> 3  rungs {r3}")
    print(f"        trajectory vs baseline: {w3:.2e} on {where3 or '-'}")
    print(f"        least_squares rescued it: {hit3}   answer preserved: {same3}"
          f"   -> {'PASS' if hit3 and same3 else 'FAIL'}\n")
    ok &= hit3 and same3

    # --- 3 -> 4 --------------------------------------------------------------
    # Plumbing, and labelled as such: the full-length step is TOLD it failed, so
    # what is under test is that step() sub-divides, reports rung 4 and marks
    # n_substeps -- not that sub-dividing cures a real non-convergence.
    real_step_once = bg._step_once
    full_dt = [0.0]

    def fail_full_length(state, params, now, prev, dt, post=True, rock=None):
        new, diag = real_step_once(state, params, now, prev, dt, post, rock)
        full_dt[0] = max(full_dt[0], dt)
        if dt >= full_dt[0]:
            diag = dict(diag, unconverged=True)
        return new, diag

    d4 = run(a.t_end, {"_step_once": fail_full_length})
    r4 = rungs(d4)
    # index 0 never solves, so it never sub-steps either; the series carries
    # its initial 1 there and that is not a step this test is about.
    solved = np.isfinite(np.asarray(d4["solver_rung"], dtype=float))
    ns = np.asarray(d4["n_substeps"], dtype=float)[solved]
    w4, where4, same4 = agrees(base, d4)
    hit4 = set(r4) == {4} and bool((ns > 1).all())
    print(f"3 -> 4  rungs {r4}   n_substeps: "
          f"{dict(zip(*[x.tolist() for x in np.unique(ns, return_counts=True)]))}")
    print(f"        trajectory vs baseline: {w4:.2e} on {where4 or '-'}")
    print(f"        sub-divided and marked: {hit4}"
          f"   -> {'PASS' if hit4 else 'FAIL'}\n")
    ok &= hit4

    # --- 4 -> 5 --------------------------------------------------------------
    # An unsatisfiable gate: every rung fails, sub-dividing fails, and the model
    # must refuse to return a number.
    try:
        run(a.t_end, {"RES_RTOL": 0.0})
        print("4 -> 5  returned a result with an unsatisfiable gate -> FAIL\n")
        ok = False
    except ValueError as e:
        msg = str(e)
        informative = ("equation" in msg and "tolerance" in msg
                       and "sub-dividing" in msg)
        print(f"4 -> 5  raised ValueError: {msg[:96]}...")
        print(f"        names the equation, the tolerance and the sub-dividing: "
              f"{informative}   -> {'PASS' if informative else 'FAIL'}\n")
        ok &= informative

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
