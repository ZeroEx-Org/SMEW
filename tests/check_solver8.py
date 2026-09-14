# SPDX-License-Identifier: AGPL-3.0-only
"""Gate the reduced 8-equation system against the 16-equation reference.

F3.4 replaced the solved system with an equivalent one in half the unknowns, by
substituting out the eight rows that are already explicit definitions. The
golden masters can no longer guard that change on their own: a different
formulation takes different Newton iterates, so the gate there had to be
loosened from bit-identity to a tolerance, and a tolerance loose enough to admit
the reformulation is loose enough to admit a mistake in it.

This is what does that work instead. Two questions, kept separate because they
fail for different reasons and only one of them is about the derivation:

  (a) ALGEBRAIC. With the eight substitutions inserted, are the eliminated rows
      of the ORIGINAL residual zero? They must be to within a couple of ulp of
      the quantity each row is about -- the substitution IS that row, solved for
      one variable, so anything larger is a transcription error. (Not exactly
      zero: numba compiles `**` a shade differently from numpy, worth ~2 ulp.)

  (b) NUMERICAL. Solved from the same starting point with the same tolerance,
      does the reduced system land where the 16-D system lands?

  (c) ACCURACY. Where they differ, which one is right? Neither answer is exact,
      so (b) alone cannot say: a disagreement at 1e-10 is consistent with either
      solver being the wrong one. Both answers are scored on the SAME sixteen
      equations, by F3.2's scaled residual, and the reduced one must be no
      worse. It is what turns (b) from a comparison into a gate.

    python tests/check_solver8.py [--n 1000] [--seed 0]

States come from a real run rather than from sampling parameter space, because
the states that matter are the ones the model visits. Note the naive form of (a)
-- "does the 16-D solution satisfy the eight definitions?" -- reads ~1e-11 and
looks like a derivation error. It is not: it is fsolve's own residual on the
dimensionless rows. The 16-D answer does not satisfy its own equations exactly
either, which is the whole reason (a) is posed at the substituted vector.
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

from scipy.optimize import fsolve                                  # noqa: E402
from smew import harness                                           # noqa: E402
import smew.biogeochem as bg                                       # noqa: E402

EPS = np.finfo(float).eps
ULP_TOL = 8          # (a): a few ulp, with room for a compiler
# (b) is bounded by the accuracy of the REFERENCE, not of the thing under test:
# the 16-D solve's own worst scaled residual is ~3e-11 on these states, so two
# correct solvers cannot be expected to agree more closely than that. Measured
# worst disagreement is 8.7e-11. The tight gate is (c).
REL_TOL = 1e-9
RES_TOL = 1e-11      # (c): the reduced solve's worst scaled residual

# the 16-vector slots the 8 retained unknowns occupy, and the constants
# _expand_8_numba needs out of the 25-arg list
KEEP = [1, 2, 4, 6, 7, 8, 9, 15]          # CO2_w H Al_w Mg Ca Na K f_Ca
EXP_ARGS = [5, 6, 8, 9, 10, 12, 13, 14, 15, 20, 21, 22, 23, 24]
NAME16 = ["Alk", "CO2_w", "H", "R_alk", "Al_w", "Al", "Mg", "Ca", "Na", "K",
          "f_Al", "f_Mg", "f_Na", "f_K", "f_H", "f_Ca"]
# eliminated row -> the 16-vector slot it defines
ELIM = {3: 0, 4: 3, 6: 5, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14}


def harvest(n, seed):
    """Arguments, starting point and answer of `n` real 8-D solves."""
    rec, out = {}, []
    real_eq = bg._biogeochem_equations_8_numba

    def spy(qv, *args):
        rec["args"] = args
        return real_eq(qv, *args)

    real_fsolve = bg.fsolve

    def spy_fsolve(func, x0, **kw):
        res = real_fsolve(func, x0, **kw)
        x0a = np.atleast_1d(np.asarray(x0, dtype=float))
        if x0a.size == 8 and "args" in rec:
            sol = res[0] if isinstance(res, tuple) else res
            out.append((np.asarray(rec["args"], dtype=float),
                        x0a.copy(), np.asarray(sol, dtype=float).ravel()))
        return res

    bg._biogeochem_equations_8_numba = spy
    bg.fsolve = spy_fsolve
    try:
        harness.example_run()
    finally:
        bg._biogeochem_equations_8_numba = real_eq
        bg.fsolve = real_fsolve

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(out), size=min(n, len(out)), replace=False)
    return [out[i] for i in sorted(pick)], len(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    states, total = harvest(a.n, a.seed)
    print(f"{len(states)} states sampled from {total} solves of "
          f"harness.example_run()\n")

    # ---- (a) algebraic ------------------------------------------------------
    ulp = {j: 0.0 for j in ELIM}
    for args, _q0, q in states:
        p16 = bg._expand_8_numba(q, *[args[i] for i in EXP_ARGS])
        r = np.asarray(bg._biogeochem_equations_numba(p16, *args))
        for j, slot in ELIM.items():
            sc = abs(p16[slot])
            if sc > 0:
                ulp[j] = max(ulp[j], abs(r[j - 1]) / sc / EPS)

    print("(a) eliminated rows at the substituted vector")
    print(f"    {'row':>5s} {'defines':>8s} {'worst':>8s}")
    for j, slot in ELIM.items():
        print(f"    {j:5d} {NAME16[slot]:>8s} {ulp[j]:7.1f} ulp")
    worst_a = max(ulp.values())
    ok_a = worst_a <= ULP_TOL
    print(f"    worst {worst_a:.1f} ulp  (tol {ULP_TOL})  "
          f"-> {'PASS' if ok_a else 'FAIL'}\n")

    # ---- (b) numerical ------------------------------------------------------
    # Compared on the eight RETAINED unknowns only. The eliminated eight are
    # definitions of these, so comparing them too would be counting the same
    # agreement twice -- and would import the 16-D solve's own residual, which
    # is what makes its f_Na differ from its own definition by 1e-11.
    worst = np.zeros(8)
    fails = 0
    r8, r16 = [], []
    for args, q0, q in states:
        a = args
        x0 = bg._expand_8_numba(q0, *[a[i] for i in EXP_ARGS])
        p16 = np.asarray(fsolve(
            lambda pv: bg._biogeochem_equations_numba(pv, *a), x0, xtol=1e-12))
        ref = p16[KEEP]
        sc = np.maximum(np.abs(ref), np.abs(q))
        rel = np.where(sc > 0, np.abs(q - ref) / np.where(sc > 0, sc, 1.0), 0.0)
        worst = np.maximum(worst, rel)
        fails += int(np.any(rel > REL_TOL))

        # (c), on the same states: same equations, same scale vector, two answers
        p8 = bg._expand_8_numba(q, *[a[i] for i in EXP_ARGS])
        scale = bg._residual_scale(a[0], a[4], a[9], a[11], a[16], a[17], a[18],
                                   a[19], a[1] * a[2] * a[3] * 1000)
        r8.append(np.max(np.abs(bg._biogeochem_equations_numba(p8, *a)) / scale))
        r16.append(np.max(np.abs(bg._biogeochem_equations_numba(p16, *a)) / scale))
    r8, r16 = np.array(r8), np.array(r16)

    print("(b) reduced solution vs 16-D solution, same start, same xtol")
    print(f"    {'unknown':>8s} {'worst rel':>11s}")
    for k, slot in enumerate(KEEP):
        print(f"    {NAME16[slot]:>8s} {worst[k]:11.3e}")
    worst_b = float(worst.max())
    ok_b = fails == 0
    print(f"    worst {worst_b:.3e} over {len(states)} states, "
          f"{fails} outside {REL_TOL:.0e}  -> {'PASS' if ok_b else 'FAIL'}\n")

    # ---- (c) accuracy -------------------------------------------------------
    print("(c) worst scaled residual on the sixteen equations, both answers")
    print(f"    {'':>10s} {'median':>11s} {'max':>11s}")
    print(f"    {'16-D':>10s} {np.median(r16):11.3e} {r16.max():11.3e}")
    print(f"    {'8-D':>10s} {np.median(r8):11.3e} {r8.max():11.3e}")
    ok_c = r8.max() <= RES_TOL
    print(f"    reduced better or equal on "
          f"{int((r8 <= r16).sum())}/{len(r8)} states; "
          f"8-D max {r8.max():.3e} vs tol {RES_TOL:.0e} "
          f"-> {'PASS' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
