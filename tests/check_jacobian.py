# SPDX-License-Identifier: AGPL-3.0-only
"""Gate the hand-written Jacobian of the log-variable system, entry by entry.

F3's single most important test. A hand-differentiated Jacobian fails one entry
at a time -- a sign, a factor of two, a missing chain-rule term on one variable
-- and the solver usually still converges, just slower and from fewer starting
points. That failure is invisible to every other gate in this repo: the answer
is still right, because the Jacobian only steers the search.

So this compares ENTRY BY ENTRY against central differences, never by norm. A
norm hides a wrong entry in a small row, and rows 5 and 16 here are small on
every Al-free feedstock in the suite.

    python tests/check_jacobian.py [--n 100] [--seed 0]

Entries are compared relative to the largest entry in their own COLUMN, not to
themselves. An entry that is legitimately ~0 has no relative scale of its own,
and dividing by it turns finite-difference noise into a failure -- the same
absent-denominator trap as F3.2's row scaling and F2's clip threshold. Column
rather than row because a column is one unknown's influence, which is what a
Newton step actually uses.
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
REPO = os.environ.get("SMEW_REPO", os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from smew import harness                                           # noqa: E402
import smew.biogeochem as bg                                       # noqa: E402

TOL = 1e-6           # the plan's gate
STEP = 1e-6          # central-difference step in the LOG variable
UNK = ["CO2_w", "H", "Al_w", "Mg", "Ca", "Na", "K", "f_Ca"]
EQN = [1, 2, 5, 7, 8, 9, 10, 16]


def harvest(n, seed):
    """(args, scale8, y) for `n` real solves."""
    rec, out = {}, []
    real_eq = bg._biogeochem_equations_8log_numba

    def spy(y, scale8, *args):
        rec["a"] = (scale8, args)
        return real_eq(y, scale8, *args)

    real_root = bg.root

    def spy_root(fun, y0, **kw):
        res = real_root(fun, y0, **kw)
        if "a" in rec:
            out.append((rec["a"][1], rec["a"][0].copy(),
                        np.asarray(res.x, dtype=float).copy()))
        return res

    bg._biogeochem_equations_8log_numba = spy
    bg.root = spy_root
    try:
        harness.example_run()
    finally:
        bg._biogeochem_equations_8log_numba = real_eq
        bg.root = real_root

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(out), size=min(n, len(out)), replace=False)
    return [out[i] for i in sorted(pick)], len(out)


def fd_jacobian(y, scale8, args):
    """Central differences in the log variables."""
    J = np.empty((8, 8))
    for j in range(8):
        h = STEP * max(1.0, abs(y[j]))
        yp = y.copy(); yp[j] += h
        ym = y.copy(); ym[j] -= h
        fp = np.asarray(bg._biogeochem_equations_8log_numba(yp, scale8, *args))
        fm = np.asarray(bg._biogeochem_equations_8log_numba(ym, scale8, *args))
        J[:, j] = (fp - fm) / (2 * h)
    return J


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    states, total = harvest(a.n, a.seed)
    print(f"{len(states)} states sampled from {total} solves of "
          f"harness.example_run()\n")

    worst = np.zeros((8, 8))
    nfail = 0
    for args, scale8, y in states:
        Ja = np.asarray(bg._jacobian_8log_numba(y, scale8, *args))
        Jf = fd_jacobian(y, scale8, args)
        # per-column scale: the influence of one unknown on the whole system
        col = np.maximum(np.abs(Ja).max(axis=0), np.abs(Jf).max(axis=0))
        col = np.where(col > 0, col, 1.0)
        rel = np.abs(Ja - Jf) / col
        worst = np.maximum(worst, rel)
        nfail += int(np.any(rel > TOL))

    print("worst relative disagreement, analytic vs central differences,")
    print("each entry against the largest entry in its own column:\n")
    print("eq  " + "".join(f"{u:>10s}" for u in UNK))
    for i in range(8):
        print(f"{EQN[i]:3d} " + "".join(
            f"{worst[i, j]:10.1e}" if worst[i, j] > 0 else f"{'.':>10s}"
            for j in range(8)))
    w = float(worst.max())
    i, j = np.unravel_index(int(np.argmax(worst)), worst.shape)
    print(f"\nworst {w:.3e} at eq {EQN[i]} / d ln {UNK[j]}   (tol {TOL:.0e})")
    print(f"{nfail} of {len(states)} states carry any entry above tolerance")
    ok = w <= TOL
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
