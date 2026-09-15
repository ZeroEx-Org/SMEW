# SPDX-License-Identifier: AGPL-3.0-only
"""How does this solver scale when the chemistry gets bigger?

Items 4 and 5 of the development plan add Fe/Mn redox couples and Dzombak-Morel
surface complexation. Structurally that is more cation-like pools, each with its
own mass-balance row and its own exchange expression -- exactly what SMEW's
eight-equation system already is, with more of it. The question this answers,
before that work is committed to rather than after, is whether the solver F3.4
and F3.5 built stays affordable at 16, 24, 40, 60 unknowns.

The system here is SYNTHETIC but structurally faithful: same carbonate rows,
same aluminium ladder, same Gaines-Thomas monomials, same closure row, same
sparsity. Only the number of exchangeable cations varies.

    n unknowns = 3 (CO2_w, H, Al_w) + c cations + 1 (f_Ca)

so SMEW itself is c = 4, n = 8.

Each case is built by choosing the answer first and deriving the totals from it.
That guarantees the system has a solution -- a randomly-chosen set of totals
need not -- and it means "did it converge" and "did it converge to the RIGHT
place" are separate questions, both askable.

    python tests/check_scaling.py [--sizes 8 16 24 40 60] [--repeats 200]
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
from numba import njit
from scipy.optimize import brentq, root

warnings.filterwarnings("ignore")
REPO = os.environ.get("SMEW_REPO", os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

STEPS_PER_YEAR = 52559          # Example at dt = 10 min, for the projection


# --- the synthetic system ---------------------------------------------------
# Solver variables, mirroring F3.5: logs everywhere except Al_w, which is
# carried linearly because its true value can be exactly zero.
#   y[0] ln CO2_w   y[1] ln H   y[2] Al_w   y[3] ln Ca
#   y[4 .. 3+c-1] ln of the other cations   y[-1] ln f_Ca

@njit(cache=True)
def _parts(y, c, kc, ex, Kx, K1, K2, K3, K4, KAl, KH, conv_Al, CEC):
    CO2_w = np.exp(y[0]); H = np.exp(y[1]); Al_w = y[2]; Ca = np.exp(y[3])
    f_Ca = np.exp(y[3 + c])
    C = np.empty(c - 1)
    for k in range(c - 1):
        C[k] = np.exp(y[4 + k])

    D = (H**4 + H**3 * K1 + H**2 * K1 * K2 + H * K1 * K2 * K3
         + K1 * K2 * K3 * K4)
    dD = 4 * H**3 + 3 * H**2 * K1 + 2 * H * K1 * K2 + K1 * K2 * K3
    fAl_per = (H**4 / D) / conv_Al * (f_Ca**3 / (KAl * Ca**3))**0.5
    f_Al = fAl_per * Al_w
    f_H = H * (f_Ca / (KH * Ca))**0.5

    f = np.empty(c - 1)
    for k in range(c - 1):
        f[k] = C[k] * (f_Ca / (Kx[k] * Ca))**ex[k]

    Alk = kc[0] * CO2_w / H + 2 * kc[0] * kc[1] * CO2_w / H**2 - H + kc[2] / H
    return CO2_w, H, Al_w, Ca, f_Ca, C, f, f_Al, f_H, fAl_per, Alk, D, dD


@njit(cache=True)
def _residual(y, scale, c, kc, ex, zc, Kx, K1, K2, K3, K4, KAl, KH, conv_Al,
              CEC, N, M, s, Alk_tot, IC_tot, Al_tot, Ctot):
    (CO2_w, H, Al_w, Ca, f_Ca, C, f, f_Al, f_H,
     fAl_per, Alk, D, dD) = _parts(y, c, kc, ex, Kx, K1, K2, K3, K4, KAl, KH,
                                   conv_Al, CEC)
    n = 4 + c
    r = np.empty(n)

    R_alk = f_Ca
    for k in range(c - 1):
        R_alk += f[k]
    R_alk *= CEC

    r[0] = (Alk_tot - R_alk) - Alk * N
    S = s * (1 + kc[0] / H + kc[1] * kc[0] / H**2) + (1 - s) / kc[3]
    r[1] = IC_tot - CO2_w * S * M
    r[2] = Al_w * N + (f_Al / 3) * CEC * conv_Al - Al_tot
    r[3] = Ca * N + f_Ca / 2 * CEC - Ctot[0]
    for k in range(c - 1):
        r[4 + k] = C[k] * N + f[k] / zc[k] * CEC - Ctot[1 + k]

    tot = f_Ca + f_Al + f_H
    for k in range(c - 1):
        tot += f[k]
    r[n - 1] = 1 - tot

    for i in range(n):
        r[i] /= scale[i]
    return r


@njit(cache=True)
def _jacobian(y, scale, c, kc, ex, zc, Kx, K1, K2, K3, K4, KAl, KH, conv_Al,
              CEC, N, M, s, Alk_tot, IC_tot, Al_tot, Ctot):
    (CO2_w, H, Al_w, Ca, f_Ca, C, f, f_Al, f_H,
     fAl_per, Alk, D, dD) = _parts(y, c, kc, ex, Kx, K1, K2, K3, K4, KAl, KH,
                                   conv_Al, CEC)
    n = 4 + c
    J = np.zeros((n, n))
    iCa = 3
    iF = 3 + c

    A1 = kc[0] * CO2_w / H
    A2 = 2 * kc[0] * kc[1] * CO2_w / H**2
    A4 = kc[2] / H
    w = 4 - H * dD / D

    # sum_k e_k f_k -- the recurring term, exactly as in SMEW's rows 1 and 16
    ef = 0.0
    for k in range(c - 1):
        ef += ex[k] * f[k]

    # row 0: alkalinity
    J[0, 0] = -N * (A1 + A2)
    J[0, 1] = -N * (-A1 - 2 * A2 - H - A4)
    J[0, iCa] = CEC * ef
    for k in range(c - 1):
        J[0, 4 + k] = -CEC * f[k]
    J[0, iF] = -CEC * (f_Ca + ef)

    # row 1: inorganic carbon
    S = s * (1 + kc[0] / H + kc[1] * kc[0] / H**2) + (1 - s) / kc[3]
    J[1, 0] = -CO2_w * S * M
    J[1, 1] = CO2_w * M * s * (kc[0] / H + 2 * kc[1] * kc[0] / H**2)

    # row 2: aluminium
    G = CEC * conv_Al / 3
    J[2, 1] = G * f_Al * w
    J[2, 2] = N + G * fAl_per
    J[2, iCa] = -1.5 * G * f_Al
    J[2, iF] = 1.5 * G * f_Al

    # row 3: the reference cation
    J[3, iCa] = Ca * N
    J[3, iF] = 0.5 * CEC * f_Ca

    # rows 4..: the other cations, three entries each whatever n is
    for k in range(c - 1):
        J[4 + k, 4 + k] = C[k] * N + f[k] / zc[k] * CEC
        J[4 + k, iCa] = -ex[k] * f[k] / zc[k] * CEC
        J[4 + k, iF] = ex[k] * f[k] / zc[k] * CEC

    # last row: closure, the only dense one
    J[n - 1, 1] = -(f_Al * w + f_H)
    J[n - 1, 2] = -fAl_per
    J[n - 1, iCa] = 1.5 * f_Al + ef + 0.5 * f_H
    for k in range(c - 1):
        J[n - 1, 4 + k] = -f[k]
    J[n - 1, iF] = -(f_Ca + 1.5 * f_Al + ef + 0.5 * f_H)

    for i in range(n):
        for j in range(n):
            J[i, j] /= scale[i]
    return J


def make_case(c, seed):
    """Choose the answer, then derive the totals that make it the answer."""
    rng = np.random.default_rng(seed)
    conv_mol, conv_Al = 1e6, 1e3
    CEC = 3.0e7
    N = 0.45 * 0.3 * 0.35 * 1000
    M = 0.45 * 0.3 * 1000
    s = 0.35
    kc = np.array([4.5e-7 * conv_mol, 4.7e-11 * conv_mol,
                   1e-14 * conv_mol**2, 1.2])                 # k1 k2 kw kH
    # SMEW's own Al hydrolysis pK ladder, so the stiffest part of the system
    # is the real one rather than an invented one
    K1 = 10**-5.0 * conv_mol
    K2 = 10**-5.1 * conv_mol
    K3 = 10**-6.7 * conv_mol
    K4 = 10**-6.2 * conv_mol
    KAl, KH = 1.0, 1.0

    # charges alternate 2 and 1, as Mg/Ca against Na/K do
    zc = np.array([2.0 if k % 2 == 0 else 1.0 for k in range(c - 1)])
    ex = zc / 2.0
    Kx = np.exp(rng.uniform(-0.5, 0.5, c - 1))

    # the true state
    CO2_w = 60.0 * np.exp(rng.uniform(-0.2, 0.2))
    H = 10**(-6.5) * conv_mol * np.exp(rng.uniform(-0.3, 0.3))
    Al_w = 1e-6 * np.exp(rng.uniform(-0.3, 0.3))
    Ca = 40.0 * np.exp(rng.uniform(-0.2, 0.2))
    C = 20.0 * np.exp(rng.uniform(-0.5, 0.5, c - 1))

    # f_Ca is NOT free. Every other row has a total that can be derived to make
    # the chosen state exact; the closure row has none -- it says the fractions
    # sum to one, and nothing in it is adjustable. So f_Ca is solved for.
    # Every fraction is an increasing monomial in f_Ca, so the sum is monotone
    # and a bracketed solve is exact and cannot fail.
    DD = (H**4 + H**3 * K1 + H**2 * K1 * K2 + H * K1 * K2 * K3
          + K1 * K2 * K3 * K4)

    def closure(lf):
        fc = np.exp(lf)
        tot = fc
        tot += (H**4 / DD) * Al_w / conv_Al * (fc**3 / (KAl * Ca**3))**0.5
        tot += H * (fc / (KH * Ca))**0.5
        for k in range(c - 1):
            tot += C[k] * (fc / (Kx[k] * Ca))**ex[k]
        return tot - 1.0

    f_Ca = float(np.exp(brentq(closure, np.log(1e-12), np.log(1.0),
                               xtol=1e-15, rtol=1e-15)))

    y = np.empty(4 + c)
    y[0] = np.log(CO2_w); y[1] = np.log(H); y[2] = Al_w; y[3] = np.log(Ca)
    y[4:4 + c - 1] = np.log(C)
    y[3 + c] = np.log(f_Ca)

    dummy = np.ones(4 + c)
    zeros = np.zeros(c)
    args = (c, kc, ex, zc, Kx, K1, K2, K3, K4, KAl, KH, conv_Al, CEC, N, M, s,
            0.0, 0.0, 0.0, zeros)
    # the residual with zero totals IS minus the forward map, so the totals
    # that make `y` exact fall straight out of one evaluation
    r = _residual(y, dummy, *args)
    # r = (forward map) - (totals), so with the totals set to zero each row is
    # plus or minus its own forward map, depending which side the total sits on.
    Alk_tot = -r[0]
    IC_tot = -r[1]
    Al_tot = r[2]
    Ctot = np.empty(c)
    Ctot[0] = r[3]
    Ctot[1:] = r[4:4 + c - 1]

    args = (c, kc, ex, zc, Kx, K1, K2, K3, K4, KAl, KH, conv_Al, CEC, N, M, s,
            Alk_tot, IC_tot, Al_tot, Ctot)
    scale = np.empty(4 + c)
    scale[0] = abs(Alk_tot); scale[1] = abs(IC_tot); scale[2] = abs(Al_tot)
    for k in range(c):
        scale[3 + k] = abs(Ctot[k])
    scale[3 + c] = 1.0
    scale = np.maximum(scale, 1e-12 * scale.max())

    # a starting point the size of one timestep's drift, not a random guess
    y0 = y.copy()
    y0[:2] += rng.normal(0, 1e-3, 2)
    y0[3:] += rng.normal(0, 1e-3, 1 + c)
    y0[2] *= 1 + rng.normal(0, 1e-3)
    return args, scale, y, y0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[8, 16, 24, 40, 60])
    ap.add_argument("--repeats", type=int, default=200)
    a = ap.parse_args(argv)

    # --- the Jacobian must be right before any timing means anything ---------
    worst = 0.0
    for n in a.sizes:
        args, scale, y, _ = make_case(n - 4, seed=n)
        Ja = _jacobian(y, scale, *args)
        Jf = np.empty_like(Ja)
        for j in range(len(y)):
            h = 1e-6 * max(1.0, abs(y[j]))
            yp = y.copy(); yp[j] += h
            ym = y.copy(); ym[j] -= h
            Jf[:, j] = (_residual(yp, scale, *args)
                        - _residual(ym, scale, *args)) / (2 * h)
        col = np.maximum(np.abs(Ja).max(axis=0), np.abs(Jf).max(axis=0))
        worst = max(worst, float(np.max(np.abs(Ja - Jf)
                                        / np.where(col > 0, col, 1.0))))
    print(f"synthetic Jacobian vs central differences, worst entry over "
          f"{a.sizes}: {worst:.2e}")
    if worst > 1e-6:
        print("FAIL -- the synthetic Jacobian is wrong, timings would be "
              "meaningless\n")
        return 1
    print()

    print(f"{'n':>4s} {'mode':>9s} {'nfev':>7s} {'njev':>6s} {'ms/solve':>9s} "
          f"{'s/yr':>8s} {'err':>9s}")
    table, nfev_of, njev_of = {}, {}, {}
    for n in a.sizes:
        cases = [make_case(n - 4, seed=1000 + n * 97 + i)
                 for i in range(a.repeats)]
        for mode in ("FD", "analytic"):
            # warm the jit, and check it lands on the planted answer
            args, scale, y, y0 = cases[0]
            kw = {} if mode == "FD" else {
                "jac": lambda yy, s=scale, A=args: _jacobian(yy, s, *A)}
            root(lambda yy, s=scale, A=args: _residual(yy, s, *A), y0,
                 method="hybr", options={"xtol": 1e-12}, **kw)

            nfev = njev = 0
            err = 0.0
            t0 = time.perf_counter()
            for args, scale, y, y0 in cases:
                f = lambda yy, s=scale, A=args: _residual(yy, s, *A)
                kw = {} if mode == "FD" else {
                    "jac": lambda yy, s=scale, A=args: _jacobian(yy, s, *A)}
                r = root(f, y0, method="hybr", options={"xtol": 1e-12}, **kw)
                nfev += r.nfev
                njev += getattr(r, "njev", 0) or 0
                err = max(err, float(np.max(np.abs(r.x - y)
                                            / np.maximum(np.abs(y), 1e-30))))
            el = (time.perf_counter() - t0) / len(cases)
            table[(n, mode)] = el
            nfev_of[(n, mode)] = nfev / len(cases)
            njev_of[(n, mode)] = njev / len(cases)
            print(f"{n:4d} {mode:>9s} {nfev/len(cases):7.1f} "
                  f"{njev/len(cases):6.2f} {el*1e3:9.3f} "
                  f"{el*STEPS_PER_YEAR:8.1f} {err:9.2e}")
    print()

    # --- where the time actually goes ----------------------------------------
    # Wall-clock per solve compresses both curves, so time the two kernels
    # directly: that is what the gate is a statement about.
    #
    # The remainder is NOT all overhead, and calling it that would be wrong.
    # It is scipy's per-call Python wrapping -- which is a fixed cost, and is
    # what F3.5 found dominating the real model -- PLUS MINPACK's own linear
    # algebra, the QR factorisation and dogleg step on an n x n Jacobian, which
    # is genuine necessary work growing as n^2 to n^3. At large n that becomes
    # the binding cost, and no Jacobian can remove it.
    print(f"{'n':>4s} {'t_res us':>9s} {'t_jac us':>9s} {'FD work':>9s} "
          f"{'an. work':>9s} {'not kernel':>11s}")
    work = {}
    for n in a.sizes:
        args, scale, y, _ = make_case(n - 4, seed=n)
        reps = 2000
        t0 = time.perf_counter()
        for _ in range(reps):
            _residual(y, scale, *args)
        tr = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            _jacobian(y, scale, *args)
        tj = (time.perf_counter() - t0) / reps
        wfd = nfev_of[(n, "FD")] * tr
        wan = nfev_of[(n, "analytic")] * tr + njev_of[(n, "analytic")] * tj
        work[n] = (wfd, wan)
        print(f"{n:4d} {tr*1e6:9.2f} {tj*1e6:9.2f} {wfd*1e6:9.1f} "
              f"{wan*1e6:9.1f} {(1 - wan/table[(n,'analytic')])*100:10.0f}%")
    print()

    # --- the gate ------------------------------------------------------------
    print(f"{'n':>4s} {'nfev FD':>8s} {'nfev an.':>9s} {'FD/an. work':>12s} "
          f"{'an. work vs n=8':>16s}")
    base = work[a.sizes[0]][1]
    for n in a.sizes:
        print(f"{n:4d} {nfev_of[(n,'FD')]:8.1f} {nfev_of[(n,'analytic')]:9.1f} "
              f"{work[n][0]/work[n][1]:11.2f}x {work[n][1]/base:15.1f}x")

    # The relationship the gate is really asserting: forward differences cost
    # one residual evaluation per unknown, so their count is n + (the Newton
    # iterations); an analytic Jacobian's count is the Newton iterations alone
    # and does not move with n.
    slope = ((nfev_of[(a.sizes[-1], "FD")] - nfev_of[(a.sizes[0], "FD")])
             / (a.sizes[-1] - a.sizes[0]))
    spread = (max(nfev_of[(n, "analytic")] for n in a.sizes)
              - min(nfev_of[(n, "analytic")] for n in a.sizes))
    print(f"\nnfev(FD) grows at {slope:.2f} per unknown (theory: 1.00)")
    print(f"nfev(analytic) varies by {spread:.1f} across n = "
          f"{a.sizes[0]}..{a.sizes[-1]} (theory: 0)")

    big = 40 if 40 in a.sizes else a.sizes[-1]
    yr = table[(big, "analytic")] * STEPS_PER_YEAR
    ratio_grows = (work[a.sizes[-1]][0] / work[a.sizes[-1]][1]
                   > work[a.sizes[0]][0] / work[a.sizes[0]][1])
    tractable = yr < 600
    print(f"\nanalytic advantage in KERNEL work grows with n: {ratio_grows}")
    print(f"n={big} at Example's {STEPS_PER_YEAR} steps: {yr:.0f} s "
          f"-> tractable (<10 min): {tractable}")
    ok = ratio_grows and tractable
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
