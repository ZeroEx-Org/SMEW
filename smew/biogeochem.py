# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:34:44 2019
"""

import warnings
from dataclasses import fields, replace

import numpy as np
import smew
from smew import activity, weathering_kinec
from smew.state import (SoilParams, SoilState, StepForcing,
                        STATE_1D, STATE_MIN, STATE_PSD, STATE_SEC)
from numba import njit
from scipy.optimize import brentq, fsolve, least_squares, root
#minimize, least_squares, newton_krylov, broyden1, root, broyden2

# --- the carbonate charge balance, solved directly -------------------------
#
# Two places need the H+ that balances a known alkalinity against a known
# dissolved CO2: rainwater entering the soil (every timestep) and the solver's
# rung-2 starting guess. Both asked scipy to find the root of
#
#     f(H) = k1*CO2/H + 2*k1*k2*CO2/H^2 - H + k_w/H - Alk = 0
#
# one scalar call at a time. Multiplying through by H^2 shows what that
# actually is:
#
#     H^3 + Alk*H^2 - (k1*CO2 + k_w)*H - 2*k1*k2*CO2 = 0
#
# a cubic. Its coefficients run + + - -, one sign change, so by Descartes' rule
# it has EXACTLY ONE positive root -- there is never a question of which root is
# meant. f is also strictly decreasing on H > 0 (every term is), which gives a
# bracket and makes Newton safe to use.
#
# The closed form is NOT used, and that is a measurement, not a preference:
# Cardano's formula subtracts Alk/3 from a number very close to Alk/3, and at
# soil alkalinities that cancellation costs nine digits -- worst case 1.8e-2
# relative, checked against 60-digit arithmetic. What is used instead is the
# bracket plus Newton in log H, which is well scaled because every term of f is
# then a positive quantity of comparable size.
#
# Measured against 60-digit bisection over 807 cases, including Alk from 0 to
# 10 mol/l and CO2 down to 1e-30: worst relative error 1.2e-15. On the real
# rainwater workload it converges in TWO iterations and is both 376x faster than
# the scipy loop it replaces (1.07 s -> 2.8 ms per simulated year) and 146x more
# accurate (worst relative residual 3.1e-16 against fsolve's 4.6e-14).

_H_EPS = 4.0 * np.finfo(float).eps


def _H_charge_balance(Alk, k1, k2, CO2_w, k_w):
    """H+ balancing `Alk` against dissolved CO2. Vectorised over any shape.

    Units are whatever the caller works in, provided they are consistent: the
    equation is homogeneous in the concentration unit, so conv_mol cancels.
    """
    Alk, k1, k2, CO2_w, k_w = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (Alk, k1, k2, CO2_w, k_w)])
    B = k1 * CO2_w + k_w                        # coefficient of -H
    C = 2.0 * k1 * k2 * CO2_w                   # the constant, >= 0

    def f_of(H):
        t1 = k1 * CO2_w / H
        t2 = 2.0 * k1 * k2 * CO2_w / (H * H)
        t3 = k_w / H
        return t1 + t2 - H + t3 - Alk, t1, t2, t3

    # Bracket. hi: H^3 >= B*H + C once H >= max(sqrt(2B), cbrt(2C)), and the
    # Alk*H^2 term only reinforces it. lo: the mirror bound, additionally
    # requiring Alk*H^2 to stay under half the right-hand side.
    with np.errstate(divide="ignore", invalid="ignore"):
        hi = np.maximum(np.sqrt(2.0 * B), np.cbrt(2.0 * C))
        a_safe = np.where(Alk > 0.0, Alk, 1.0)
        lo = np.minimum(
            np.maximum(np.sqrt(0.5 * B), np.cbrt(0.5 * C)),
            np.where(Alk > 0.0,
                     np.maximum(B / (2.0 * a_safe), np.sqrt(C / (2.0 * a_safe))),
                     np.inf))
    lo = np.minimum(lo, hi)

    # The bounds above are derived, so trust them -- but verify, because a
    # bracket that excludes the root would send the bisection safeguard to the
    # wrong place silently. 60 doublings cover any input a double can hold.
    for _ in range(60):
        bad_lo = f_of(lo)[0] < 0.0
        bad_hi = f_of(hi)[0] > 0.0
        if not (bad_lo.any() or bad_hi.any()):
            break
        lo = np.where(bad_lo, lo * 0.5, lo)
        hi = np.where(bad_hi, hi * 2.0, hi)
    else:
        raise RuntimeError("carbonate charge balance: could not bracket H+")

    H = np.sqrt(lo * hi)
    for _ in range(100):
        f, t1, t2, t3 = f_of(H)
        # f is decreasing, so f > 0 means the root lies above H.
        lo = np.where(f > 0.0, H, lo)
        hi = np.where(f > 0.0, hi, H)
        if np.all(np.abs(f) <= _H_EPS * (t1 + t2 + t3 + H + np.abs(Alk))):
            break
        dfdu = -(t1 + 2.0 * t2 + t3 + H)        # d f / d ln H, strictly < 0
        Hn = H * np.exp(-f / dfdu)
        # Outside the bracket, Newton is not to be trusted: bisect in log H.
        H = np.where((Hn > lo) & (Hn < hi), Hn, np.sqrt(lo * hi))
    else:
        raise RuntimeError("carbonate charge balance: H+ did not converge")
    return H

#implicit system
@njit
def _biogeochem_equations_numba(
        p, Alk_tot, n, Zr, s, IC_tot, k1, k2, k_H, k_w, CEC_tot, conv_Al, Al_tot, K1, K2, K3, K4, Mg_tot, Ca_tot,
        Na_tot, K_tot, K_Ca_Al, K_Ca_Mg, K_Ca_Na, K_Ca_K, K_Ca_H
):
    Alk, CO2_w, H, R_alk, Al_w, Al, Mg, Ca, Na, K, f_Al, f_Mg, f_Na, f_K, f_H, f_Ca = p

    # Precompute
    nZrs1000 = n * Zr * s * 1000

    return (
        (Alk_tot-R_alk)-Alk*nZrs1000,
        IC_tot-(CO2_w*(1+k1/H+k2*k1/(H**2))*s+(CO2_w/k_H)*(1-s))*(n*Zr*1000),
        (k1*CO2_w/H+2*k1*k2*CO2_w/(H**2)-H+k_w/H)-Alk,
        R_alk-(f_Mg+f_Ca+f_Na+f_K)*CEC_tot,
        Al_w*nZrs1000+(f_Al/3)*CEC_tot*conv_Al-Al_tot,
        Al-(H**4/(H**4+H**3*K1+H**2*K1*K2+H*K1*K2*K3+K1*K2*K3*K4))*Al_w,
        Mg*nZrs1000+f_Mg/2*CEC_tot-Mg_tot,
        Ca*nZrs1000+f_Ca/2*CEC_tot-Ca_tot,
        Na*nZrs1000+f_Na*CEC_tot-Na_tot,
        K*nZrs1000+f_K*CEC_tot-K_tot,
        f_Al - (Al/conv_Al)*(f_Ca**3/(K_Ca_Al*Ca**3))**(1/2),
        f_Mg - Mg*(f_Ca/(K_Ca_Mg*Ca)),
        f_Na - Na*(f_Ca/(K_Ca_Na*Ca))**(1/2),
        f_K - K*(f_Ca/(K_Ca_K*Ca))**(1/2),
        f_H - H*(f_Ca/(K_Ca_H*Ca))**(1/2),
        1-(f_Ca+f_Al+f_Mg+f_Na+f_K+f_H)
    )


# ---------------------------------------------------------------------------
# The same system in eight unknowns
# ---------------------------------------------------------------------------
#
# Eight of the sixteen equations above are already explicit definitions of one
# unknown in terms of the others, so they can be substituted out by hand -- no
# component basis, no reformulation, no choice to make:
#
#   Alk   from eq 3      R_alk from eq 4      Al from eq 6
#   f_Al, f_Mg, f_Na, f_K, f_H  from eqs 11-15
#
# leaving eight unknowns -- CO2_w, H, Al_w, Mg, Ca, Na, K, f_Ca -- against the
# eight remaining equations 1, 2, 5, 7-10 and 16. Exactly determined.
#
# The point is not elegance, it is cost. fsolve builds its Jacobian by forward
# differences, which is n+1 residual evaluations; the 16-D solve measured 34.1
# evaluations per call, i.e. two Jacobians (17 each) plus a few function
# evaluations. At n = 8 a Jacobian is 9 evaluations, and the 16-D solve was 35 %
# of total runtime.
#
# Verified before it was written, at 1 000 states sampled from a real run: with
# the eight substitutions inserted, all eight eliminated rows of the ORIGINAL
# residual are identically zero. That is a transcription check, and
# transcription is the whole risk here -- each substitution IS its row, solved
# for one variable, so anything above a couple of ulp is a typo, not a
# tolerance. `tests/check_solver8.py` is that check, kept.
#
# The naive form of it -- "does the 16-D solution satisfy the eight
# definitions?" -- reads 1e-11, not 1e-16, and would have been mistaken for a
# derivation error. It is not: it is fsolve's own residual on those rows, which
# F3.1 showed is ~1e-11 on the dimensionless rows. The 16-D answer does not
# satisfy its own equations exactly either -- which is also why the goldens
# cannot gate this change bit-identically, and why check_solver8 scores both
# answers on the same sixteen equations rather than against each other.


@njit
def _expand_8_numba(q, k1, k2, k_w, CEC_tot, conv_Al, K1, K2, K3, K4,
                    K_Ca_Al, K_Ca_Mg, K_Ca_Na, K_Ca_K, K_Ca_H):
    """The eight substitutions, in dependency order -> the full 16-vector.

    One source of truth: the reduced residual calls this, and so does the
    expansion of the answer after the solve. The 16-vector it returns is in the
    same order as `_biogeochem_equations_numba`'s unknowns, so the original
    residual can be evaluated at it directly -- which is how the gate, `errors`
    and `res_row` keep meaning exactly what they meant before.
    """
    CO2_w = q[0]; H = q[1]; Al_w = q[2]
    Mg = q[3]; Ca = q[4]; Na = q[5]; K = q[6]; f_Ca = q[7]

    Alk = k1*CO2_w/H + 2*k1*k2*CO2_w/(H**2) - H + k_w/H            # eq 3
    Al = (H**4/(H**4 + H**3*K1 + H**2*K1*K2 + H*K1*K2*K3
                + K1*K2*K3*K4))*Al_w                                # eq 6
    f_Al = (Al/conv_Al)*(f_Ca**3/(K_Ca_Al*Ca**3))**(1/2)            # eq 11
    f_Mg = Mg*(f_Ca/(K_Ca_Mg*Ca))                                   # eq 12
    f_Na = Na*(f_Ca/(K_Ca_Na*Ca))**(1/2)                            # eq 13
    f_K = K*(f_Ca/(K_Ca_K*Ca))**(1/2)                               # eq 14
    f_H = H*(f_Ca/(K_Ca_H*Ca))**(1/2)                               # eq 15
    R_alk = (f_Mg + f_Ca + f_Na + f_K)*CEC_tot                      # eq 4

    p = np.empty(16)
    p[0] = Alk;  p[1] = CO2_w; p[2] = H;    p[3] = R_alk
    p[4] = Al_w; p[5] = Al;    p[6] = Mg;   p[7] = Ca
    p[8] = Na;   p[9] = K;     p[10] = f_Al; p[11] = f_Mg
    p[12] = f_Na; p[13] = f_K; p[14] = f_H; p[15] = f_Ca
    return p


@njit
def _biogeochem_equations_8_numba(
        q, Alk_tot, n, Zr, s, IC_tot, k1, k2, k_H, k_w, CEC_tot, conv_Al, Al_tot,
        K1, K2, K3, K4, Mg_tot, Ca_tot, Na_tot, K_tot, K_Ca_Al, K_Ca_Mg,
        K_Ca_Na, K_Ca_K, K_Ca_H
):
    """Rows 1, 2, 5, 7-10, 16 of the system above, the other eight substituted.

    Same signature as `_biogeochem_equations_numba` apart from the length of the
    unknown vector, so the two can be driven from one call site and compared.
    """
    p = _expand_8_numba(q, k1, k2, k_w, CEC_tot, conv_Al, K1, K2, K3, K4,
                        K_Ca_Al, K_Ca_Mg, K_Ca_Na, K_Ca_K, K_Ca_H)
    Alk = p[0]; CO2_w = p[1]; H = p[2]; R_alk = p[3]
    Al_w = p[4]; Mg = p[6]; Ca = p[7]; Na = p[8]; K = p[9]
    f_Al = p[10]; f_Mg = p[11]; f_Na = p[12]; f_K = p[13]; f_H = p[14]
    f_Ca = p[15]

    nZrs1000 = n * Zr * s * 1000

    return (
        (Alk_tot-R_alk)-Alk*nZrs1000,                                    # 1
        IC_tot-(CO2_w*(1+k1/H+k2*k1/(H**2))*s
                + (CO2_w/k_H)*(1-s))*(n*Zr*1000),                        # 2
        Al_w*nZrs1000+(f_Al/3)*CEC_tot*conv_Al-Al_tot,                   # 5
        Mg*nZrs1000+f_Mg/2*CEC_tot-Mg_tot,                               # 7
        Ca*nZrs1000+f_Ca/2*CEC_tot-Ca_tot,                               # 8
        Na*nZrs1000+f_Na*CEC_tot-Na_tot,                                 # 9
        K*nZrs1000+f_K*CEC_tot-K_tot,                                    # 10
        1-(f_Ca+f_Al+f_Mg+f_Na+f_K+f_H),                                 # 16
    )


# ---------------------------------------------------------------------------
# The same eight equations in log variables, with their analytic Jacobian
# ---------------------------------------------------------------------------
#
# Three changes at once, because each one makes the next cheap and all three
# rewrite the same function:
#
#   1. Solve for ln of seven of the eight unknowns. They span forty orders of
#      magnitude in the model's own units (Al_w ~1e-41 on an Al-free feedstock
#      against CEC_tot ~1e7); the log enforces positivity structurally rather
#      than by patching the answer, and it turns every mass-action term into a
#      monomial, which is why the Jacobian below is mostly multiplication by
#      small integers.
#
#      Al_w is the exception and stays LINEAR. Not a compromise -- a measurement:
#      across the suite's 1 359 727 solves, `Al_tot` is EXACTLY zero on 9 of
#      them, and on those the correct Al_w is exactly zero too. ln cannot
#      represent that. The only way to log it is to floor it, and a floor on a
#      quantity that is legitimately zero is the trap F2 hit on clip_Al ~1e-39
#      and F3.3 hit twice more. So the variable that most wants log conditioning
#      is the one variable that cannot have it.
#
#      (The plan predicted seven logs with f_Ca as the odd one out. It is seven,
#      but f_Ca is not the exception: measured, f_Ca lives in [0.050, 0.930],
#      comfortably inside the interval a logit exists to protect.)
#
#   2. Fold F3.2's row scaling into the residual, not just the gate. F3.2 kept
#      it out because scaling changes the iterates; this rewrite changes them
#      anyway, so it is the cheapest place to do it.
#
#   3. Supply the Jacobian. fsolve was spending 9 of every ~15 residual
#      evaluations building one by forward differences.
#
# If a future case pushes f_Ca toward 1, the logit is the line to revisit.
#
# The chain rule is nearly free in log variables: d r / d(ln x) = x * dr/dx, and
# every term below is a product of powers of the unknowns, so its log-derivative
# is (its own exponent) x (the term itself). That is the whole derivation --
# there is no term in this system whose log-derivative needs more than a
# multiplication, except H, which appears both as a monomial and inside the
# aluminium denominator D(H).


@njit
def _q_from_y_numba(y):
    """Solver variables -> the eight unknowns.

    Seven are logs; slot 2 (Al_w) is carried linearly, for the reason above.
    Short, but it is the definition the Jacobian is differentiated against, so
    it is written once and used by both.
    """
    q = np.empty(8)
    q[0] = np.exp(y[0])
    q[1] = np.exp(y[1])
    q[2] = y[2]                      # Al_w, linear
    q[3] = np.exp(y[3])
    q[4] = np.exp(y[4])
    q[5] = np.exp(y[5])
    q[6] = np.exp(y[6])
    q[7] = np.exp(y[7])
    return q


@njit
def _y_from_q_numba(q):
    """The inverse. Al_w passes through; everything else takes a log."""
    y = np.empty(8)
    y[0] = np.log(q[0])
    y[1] = np.log(q[1])
    y[2] = q[2]
    y[3] = np.log(q[3])
    y[4] = np.log(q[4])
    y[5] = np.log(q[5])
    y[6] = np.log(q[6])
    y[7] = np.log(q[7])
    return y


@njit
def _biogeochem_equations_8log_numba(
        y, scale8, Alk_tot, n, Zr, s, IC_tot, k1, k2, k_H, k_w, CEC_tot,
        conv_Al, Al_tot, K1, K2, K3, K4, Mg_tot, Ca_tot, Na_tot, K_tot,
        K_Ca_Al, K_Ca_Mg, K_Ca_Na, K_Ca_K, K_Ca_H
):
    """The eight residuals, in solver variables, each divided by its own scale."""
    q = _q_from_y_numba(y)
    r = _biogeochem_equations_8_numba(
        q, Alk_tot, n, Zr, s, IC_tot, k1, k2, k_H, k_w, CEC_tot, conv_Al,
        Al_tot, K1, K2, K3, K4, Mg_tot, Ca_tot, Na_tot, K_tot, K_Ca_Al,
        K_Ca_Mg, K_Ca_Na, K_Ca_K, K_Ca_H)
    out = np.empty(8)
    for i in range(8):
        out[i] = r[i] / scale8[i]
    return out


@njit
def _jacobian_8log_numba(
        y, scale8, Alk_tot, n, Zr, s, IC_tot, k1, k2, k_H, k_w, CEC_tot,
        conv_Al, Al_tot, K1, K2, K3, K4, Mg_tot, Ca_tot, Na_tot, K_tot,
        K_Ca_Al, K_Ca_Mg, K_Ca_Na, K_Ca_K, K_Ca_H
):
    """d(scaled residual) / d(solver variable), 8x8, by hand.

    Column order matches the unknowns: CO2_w, H, Al_w, Mg, Ca, Na, K, f_Ca.
    Row order matches the residual: equations 1, 2, 5, 7, 8, 9, 10, 16 of the
    original sixteen.

    Derived term by term rather than by norm-checking the result, and gated
    entry by entry against central differences in tests/check_jacobian.py -- a
    norm hides a wrong entry in a small row, which is exactly the failure mode
    a hand-differentiated Jacobian has.
    """
    q = _q_from_y_numba(y)
    CO2_w = q[0]; H = q[1]; Al_w = q[2]
    Mg = q[3]; Ca = q[4]; Na = q[5]; K = q[6]; f_Ca = q[7]

    N = n * Zr * s * 1000                  # water volume factor [l]
    M = n * Zr * 1000

    # --- the carbonate terms of Alk, each a monomial in CO2_w and H ----------
    A1 = k1 * CO2_w / H                    # HCO3-      d/dlnH = -A1
    A2 = 2 * k1 * k2 * CO2_w / (H ** 2)    # 2*CO3--    d/dlnH = -2*A2
    A4 = k_w / H                           # OH-        d/dlnH = -A4
    dAlk_du = A1 + A2                      # d Alk / d ln CO2_w
    dAlk_dh = -A1 - 2 * A2 - H - A4        # d Alk / d ln H

    # --- the aluminium ladder ------------------------------------------------
    # D(H) is the only non-monomial in the system, so its H-derivative is the
    # only one that is not an integer multiple of the term itself.
    D = H ** 4 + H ** 3 * K1 + H ** 2 * K1 * K2 + H * K1 * K2 * K3 + K1 * K2 * K3 * K4
    dD = 4 * H ** 3 + 3 * H ** 2 * K1 + 2 * H * K1 * K2 + K1 * K2 * K3
    Al = (H ** 4 / D) * Al_w
    w = 4 - H * dD / D                     # d ln(H^4/D) / d ln H

    # --- the exchanger fractions, all monomials ------------------------------
    # f_Al is proportional to Al_w, and column 2 differentiates w.r.t. Al_w
    # itself rather than its log. Carrying the ratio avoids ever dividing by
    # Al_w, which is exactly zero on the nine solves that made it linear.
    fAl_per_Alw = (H ** 4 / D) / conv_Al * (f_Ca ** 3 / (K_Ca_Al * Ca ** 3)) ** (1 / 2)
    f_Al = fAl_per_Alw * Al_w
    f_Mg = Mg * (f_Ca / (K_Ca_Mg * Ca))
    f_Na = Na * (f_Ca / (K_Ca_Na * Ca)) ** (1 / 2)
    f_K = K * (f_Ca / (K_Ca_K * Ca)) ** (1 / 2)
    f_H = H * (f_Ca / (K_Ca_H * Ca)) ** (1 / 2)

    J = np.zeros((8, 8))

    # row 1: (Alk_tot - R_alk) - Alk*N,  R_alk = (f_Mg+f_Ca+f_Na+f_K)*CEC_tot
    J[0, 0] = -N * dAlk_du
    J[0, 1] = -N * dAlk_dh
    J[0, 3] = -CEC_tot * f_Mg
    J[0, 4] = CEC_tot * (f_Mg + 0.5 * f_Na + 0.5 * f_K)
    J[0, 5] = -CEC_tot * f_Na
    J[0, 6] = -CEC_tot * f_K
    J[0, 7] = -CEC_tot * (f_Mg + f_Ca + 0.5 * f_Na + 0.5 * f_K)

    # row 2: IC_tot - CO2_w*S(H)*M
    S = s * (1 + k1 / H + k2 * k1 / (H ** 2)) + (1 - s) / k_H
    J[1, 0] = -CO2_w * S * M
    J[1, 1] = CO2_w * M * s * (k1 / H + 2 * k2 * k1 / (H ** 2))

    # row 5: Al_w*N + (f_Al/3)*CEC_tot*conv_Al - Al_tot
    G = CEC_tot * conv_Al / 3
    J[2, 1] = G * f_Al * w
    J[2, 2] = N + G * fAl_per_Alw          # d/d Al_w, not d/d ln Al_w
    J[2, 4] = -1.5 * G * f_Al
    J[2, 7] = 1.5 * G * f_Al

    # row 7: Mg*N + f_Mg/2*CEC_tot - Mg_tot
    J[3, 3] = Mg * N + 0.5 * CEC_tot * f_Mg
    J[3, 4] = -0.5 * CEC_tot * f_Mg
    J[3, 7] = 0.5 * CEC_tot * f_Mg

    # row 8: Ca*N + f_Ca/2*CEC_tot - Ca_tot
    J[4, 4] = Ca * N
    J[4, 7] = 0.5 * CEC_tot * f_Ca

    # row 9: Na*N + f_Na*CEC_tot - Na_tot
    J[5, 4] = -0.5 * CEC_tot * f_Na
    J[5, 5] = Na * N + CEC_tot * f_Na
    J[5, 7] = 0.5 * CEC_tot * f_Na

    # row 10: K*N + f_K*CEC_tot - K_tot
    J[6, 4] = -0.5 * CEC_tot * f_K
    J[6, 6] = K * N + CEC_tot * f_K
    J[6, 7] = 0.5 * CEC_tot * f_K

    # row 16: 1 - (f_Ca + f_Al + f_Mg + f_Na + f_K + f_H)
    J[7, 1] = -(f_Al * w + f_H)
    J[7, 2] = -fAl_per_Alw                 # d/d Al_w
    J[7, 3] = -f_Mg
    J[7, 4] = 1.5 * f_Al + f_Mg + 0.5 * f_Na + 0.5 * f_K + 0.5 * f_H
    J[7, 5] = -f_Na
    J[7, 6] = -f_K
    J[7, 7] = -(f_Ca + 1.5 * f_Al + f_Mg + 0.5 * f_Na + 0.5 * f_K + 0.5 * f_H)

    for i in range(8):
        for j in range(8):
            J[i, j] /= scale8[i]
    return J

# ---------------------------------------------------------------------------
# Resolving the constants
# ---------------------------------------------------------------------------

def _resolve_params(n, Zr, CEC_tot, k_v, RAI, root_d, K_CEC, mineral,
                    M_rock_in, t_app, d_in, dt, diss_f, conv_mol, conv_Al,
                    gamma_on=False, gamma_picard=0,
                    mu_ref_CEC=activity.MU_REF_CEC, T_ref_CEC=283.15,
                    mg_phase="magnesite", pH_method="solution", reapply=None,
                    ic_al_cap=None, ic_al_hold="totals",
                    sec_phases=(), sec_A_seed=None, fe_mode="tracer"):
    """Every constant the run needs, looked up once.

    Previously these were ~30 locals resolved inline in the middle of
    biogeochem_balance, which meant a depth- or restart-capable model would have
    re-derived them per column or per restart. They do not depend on state.
    """
    has_rock = M_rock_in > 0
    number_min = len(mineral) if has_rock else 1
    if ic_al_cap is not None:
        smew.log_K_AlOH3(ic_al_cap, 298.15)     # unknown phase: fail here, not at t = 0
    if ic_al_hold not in ("totals", "solution", "K_Ca_Al"):
        raise ValueError("ic_al_hold must be 'totals', 'solution' or 'K_Ca_Al', "
                         f"got {ic_al_hold!r}")

    # S1: secondary phases. Checked here so a typo fails before t = 0.
    sec_phases = tuple(sec_phases)
    if len(set(sec_phases)) != len(sec_phases):
        raise ValueError(f"sec_phases lists a phase twice: {sec_phases}")
    sec_A_seed = dict(sec_A_seed or {})
    for name in sec_A_seed:
        if name not in sec_phases:
            raise ValueError(f"sec_A_seed names {name!r}, which is not in "
                             f"sec_phases {sec_phases}")
    # S3: what released Fe is
    if fe_mode not in ("tracer", "oxic", "reduced"):
        raise ValueError(f"fe_mode must be 'tracer', 'oxic' or 'reduced', got {fe_mode!r}")
    fe_on = [n for n in sec_phases if n in smew.FE_PHASES]
    if fe_on and fe_mode != "oxic":
        raise ValueError(f"{fe_on} are Fe(III) phases: they need fe_mode='oxic' "
                         f"(got {fe_mode!r})")
    if fe_mode == "oxic" and "Fe(OH)3(a)" not in sec_phases:
        raise ValueError("fe_mode='oxic' needs 'Fe(OH)3(a)' in sec_phases: the "
                         "Fe(III) it makes has nowhere else to go")
    if "goethite" in sec_phases and "Fe(OH)3(a)" not in sec_phases:
        raise ValueError("goethite forms only by ferrihydrite ageing: add 'Fe(OH)3(a)'")
    sec_st = np.zeros([len(sec_phases), 7])
    A_seed = np.zeros(len(sec_phases))
    for j, name in enumerate(sec_phases):
        ph = smew.sec_phase(name)
        sec_st[j] = ph["st"]
        A_seed[j] = float(sec_A_seed.get(name, ph["A_seed"]))
        if not A_seed[j] >= 0:
            raise ValueError(f"sec_A_seed for {name} must be >= 0, got {A_seed[j]}")

    # second application (see SoilParams.n_blocks)
    n_blocks, tt_app_re = 1, -1
    if reapply is not None:
        t_app_re, M_rock_re = reapply
        if not has_rock:
            raise ValueError("reapply needs a first application (M_rock_in > 0)")
        if not M_rock_re > 0:
            raise ValueError(f"reapply mass must be > 0 [g/m2], got {M_rock_re}")
        if not t_app_re > t_app:
            raise ValueError(f"reapply time {t_app_re} d must be after t_app {t_app} d")
        n_blocks, tt_app_re = 2, int(t_app_re / dt)

    MM_min = np.zeros(number_min)
    min_st = np.zeros([number_min, 7])   # [Ca, Mg, K, Na, Al, Si, Fe]
    K_sp = np.zeros(number_min)
    if has_rock:
        for j in range(0, number_min):
            MM_min[j], min_st[j, :] = smew.min_const(mineral[j], conv_mol)

    # [g/mol-conv]: Molar masses
    [MM_Mg, MM_Ca, MM_Na, MM_K, MM_Si, MM_C, MM_Anions, MM_Al] = smew.MM(conv_mol)

    # Aluminium speciation. Discarded here on purpose -- temperature-dependent
    # as of P5b, rebuilt per timestep in biogeochem_balance.
    _ = smew.K_Al(conv_mol)

    # CEC Gaines-Thomas constants
    #
    # These arrive as CONCENTRATION-based selectivity coefficients -- de Vries &
    # Posch (2003) state that convention explicitly, and say a model working in
    # activities has to correct them. So on a gamma_on run they are lifted once
    # to thermodynamic constants at the calibration data's own ionic strength;
    # activity.effective_constants then brings them back down at each step's mu.
    # The net correction is only the DIFFERENCE between the two ionic strengths.
    # See the long note in smew/activity.py. Exact identity when gamma_on is off.
    [K_Ca_Mg, K_Ca_K, K_Ca_Na, K_Ca_Al, K_Ca_H] = K_CEC
    if gamma_on:
        g_ref = activity.gammas(mu_ref_CEC, T_ref_CEC)
        (K_Ca_Mg, K_Ca_K, K_Ca_Na, K_Ca_Al, K_Ca_H) = \
            activity.exchange_to_thermodynamic(
                g_ref, K_Ca_Mg, K_Ca_K, K_Ca_Na, K_Ca_Al, K_Ca_H)

    # nutrient uptake by plants
    [v_f_Ca, v_f_Mg, v_f_K, v_f_Si] = smew.plant_nutr_f()
    dry_perc = 0.1  # percent of dry mass
    xi = dry_perc * np.array([v_f_Ca / MM_Ca, v_f_Mg / MM_Mg,
                              v_f_K / MM_K, v_f_Si / MM_Si])  # [mol-conv/g_biomass]

    # carb weathering constants. The two solubility products are discarded
    # here on purpose -- they are temperature-dependent as of P5 and are
    # rebuilt per timestep in biogeochem_balance, alongside k1/k2/k_w.
    [_, _, r_CaCO3, r_MgCO3, tau_CaCO3, tau_MgCO3] = \
        smew.carb_weath_const(conv_mol)

    # depth over which the soil-air CO2 gradient is taken
    Z_CO2 = Zr / 2 if Zr <= 0.3 else 0.15

    # rock surface fractality (Beerling 2020). `a` is refined against a measured
    # SSA at application, so it is finalised in _initial_state.
    b = 0.35                     # [-]
    a = (1 / (2 * 1e-10)) ** b   # [1/m^b]

    return SoilParams(
        conv_mol=conv_mol, conv_Al=conv_Al,
        n=n, Zr=Zr, CEC_tot=CEC_tot, Z_CO2=Z_CO2,
        k_v=k_v, RAI=RAI, root_d=root_d, xi=xi,
        CO2_atm=smew.CO2_atm(conv_mol),
        K_Ca_Mg=K_Ca_Mg, K_Ca_K=K_Ca_K, K_Ca_Na=K_Ca_Na,
        K_Ca_Al=K_Ca_Al, K_Ca_H=K_Ca_H,
        r_CaCO3=r_CaCO3, r_MgCO3=r_MgCO3,
        tau_CaCO3=tau_CaCO3, tau_MgCO3=tau_MgCO3,
        mineral=tuple(mineral), number_min=number_min, MM_min=MM_min,
        min_st=min_st, K_sp=K_sp, diss_f=diss_f,
        rho_rock=3 * 1e6, a=a, b=b, M_iner=0.0,
        n_d_cl=(len(d_in) * n_blocks) if has_rock else 1,
        tt_app=int(t_app / dt) if has_rock else 0,
        has_rock=has_rock,
        gamma_on=bool(gamma_on), gamma_picard=int(gamma_picard),
        mu_ref_CEC=float(mu_ref_CEC), mg_phase=str(mg_phase),
        pH_method=str(pH_method),
        ic_al_cap=ic_al_cap, ic_al_hold=str(ic_al_hold),
        n_blocks=n_blocks, tt_app_re=tt_app_re,
        sec_phases=sec_phases, n_sec=len(sec_phases), sec_st=sec_st,
        sec_A_seed=A_seed, dt_run=float(dt), fe_mode=str(fe_mode),
    )


def _forcing(s, v, I, L, T, Dw, D, r_het, r_aut, temp_soil, T_K,
             k1, k2, k_w, k_H, K_CaCO3, K_MgCO3, K1, K2, K3, K4, DIC_rain):
    """One StepForcing per timestep.

    Built up front rather than on demand because the freezing logic needs to
    reach an arbitrary earlier index: after a frost the model resumes from the
    last unfrozen step, and indexing a list makes that a lookup rather than a
    rebuild.
    """
    return [StepForcing(s=s[i], v=v[i], I=I[i], L=L[i], T=T[i],
                        Dw=Dw[i], D=D[i], r_het=r_het[i], r_aut=r_aut[i],
                        temp_soil=temp_soil[i], T_K=T_K[i],
                        k1=k1[i], k2=k2[i], k_w=k_w[i], k_H=k_H[i],
                        K_CaCO3=K_CaCO3[i], K_MgCO3=K_MgCO3[i],
                        K1=K1[i], K2=K2[i], K3=K3[i], K4=K4[i],
                        DIC_rain=DIC_rain[i])
            for i in range(len(s))]


_FORCING_FIELDS = tuple(f.name for f in fields(StepForcing))


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

# How far the particle size distribution may miss summing to 1 before it is
# worth saying so out loud. 1e-9 is roundoff on a hand-typed list; the real
# cases that fail do so at 1e-3, three orders of magnitude clear of it.
PSD_SUM_TOL = 1e-9


def _block_widths(d, n_blocks):
    """Class widths per application block: each block's own insert(diff(d), 0, d[0]).

    One application is one block and that formula applied to the whole array. With a
    second application the classes are two blocks of sorted diameters, and the
    formula across the seam would make the first class of block 2 as wide as the
    jump from block 1's largest grain to block 2's smallest.
    """
    return np.concatenate([np.insert(np.diff(b), 0, b[0]) for b in np.split(d, n_blocks)])


def _fresh_block(p, M_rock, rock_f_in, d_in, psd_perc_in):
    """Geometry of a freshly applied rock mass, as _initial_state builds it, with `a` as
    already refined there (it does not depend on the applied mass)."""
    rock_f = np.asarray(rock_f_in, dtype=float)
    d_b, psd_perc_b = _validate_psd(d_in, psd_perc_in)
    delta_d_b = np.insert(np.diff(d_b), 0, d_b[0])
    psd_b = psd_perc_b * M_rock / delta_d_b
    lamb_b = p.a * d_b ** p.b
    SSA_b = 6 / (d_b * p.rho_rock) * lamb_b
    return (rock_f * M_rock, d_b, delta_d_b, lamb_b, SSA_b, psd_b,
            M_rock * (1 - np.sum(rock_f)))


def _reapply(st, p):
    """Land the second application on state `st`; returns (state, params).

    The new block's classes take the fresh geometry, the minerals and the inert mass
    gain the new rock, and EW is recomputed from the new surface so that both rock
    accounts (stoichiometric and geometric, tests/check_psd.py) lose mass from the
    new rock on the very next step.
    """
    M_min_add, d_b, delta_d_b, lamb_b, SSA_b, psd_b, M_iner_add = p.rock_re
    k = p.n_d_cl - len(d_b)                                 # first class of block 2
    p = replace(p, M_iner=p.M_iner + M_iner_add)
    M_min = st.M_min + M_min_add
    M_rock = np.sum(M_min) + p.M_iner
    rock_f = M_min / M_rock
    d, delta_d, lamb, SSA, psd = (x.copy() for x in (st.d, st.delta_d, st.lamb, st.SSA, st.psd))
    d[k:], delta_d[k:], lamb[k:], SSA[k:], psd[k:] = d_b, delta_d_b, lamb_b, SSA_b, psd_b
    SA = np.sum(SSA * psd * delta_d)
    return replace(st, M_min=M_min, M_rock=M_rock, rock_f=rock_f, d=d, delta_d=delta_d,
                   lamb=lamb, SSA=SSA, psd=psd, SA=SA, EW=st.Wr * SA * rock_f), p


def _validate_psd(d_in, psd_perc_in):
    """Put the diameter classes in order and check they describe one rock.

    Two assumptions have always been made about `d_in` and `psd_perc_in` and
    neither has ever been checked, because nothing downstream fails loudly when
    they are broken:

      sorted   `delta_d = np.insert(np.diff(d), 0, d[0])` is a bin WIDTH, so it
               is only a width if `d` ascends. Out of order it goes negative,
               and so does `psd = psd_perc * M_rock / delta_d`. Today that is
               invisible, because `delta_d` cancels in every one of its four
               uses -- it is created as 1/delta_d inside `psd` and every reader
               takes the product `psd * delta_d`, which is just class mass. The
               cancellation is exact: sorting the real unsorted cases moves the
               answer by at most 3e-14, pure summation order. But a negative
               bin width is a trap for the first reader that does not multiply
               it back out, so the order is fixed here rather than relied upon.
               Measured: all five Bottles_tePas treatments arrive unsorted, from
               WebPlotDigitizer CDF data read in file order.

      sums to 1 `psd_perc` is the mass fraction per class, so the geometric rock
               `sum(psd * delta_d)` equals `M_rock` only if it sums to 1. When it
               does not, surface area -- and every dissolution flux through it --
               is computed for a rock of the wrong mass, permanently and
               silently. Measured: Bottles_tePas anorthite sums to 0.997506, so
               its geometric rock starts 0.25% light and stays there, which is
               178x the mass that actually dissolves in that case.

    The sum is a warning, not an error: 0.9975 is a digitisation artefact, not a
    different rock, and refusing to run would break a published benchmark over
    a rounding error. Order is repaired silently because it is provably
    result-neutral. Duplicates are fatal -- a zero bin width divides by zero.
    """
    d = np.asarray(d_in, dtype=float).ravel()
    w = np.asarray(psd_perc_in, dtype=float).ravel()
    if d.shape != w.shape:
        raise ValueError(
            "d_in and psd_perc_in must be the same length, got %d and %d"
            % (d.size, w.size))
    if d.size == 0:
        raise ValueError("d_in is empty: the rock has no diameter classes")
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("d_in must be finite and strictly positive, got %r"
                         % (d,))

    order = np.argsort(d, kind="stable")
    d, w = d[order], w[order]
    if np.any(np.diff(d) == 0.0):
        raise ValueError(
            "d_in has repeated diameters, which give a zero bin width in "
            "delta_d = diff(d) and divide by zero in psd: %r" % (d,))

    total = float(np.sum(w))
    if abs(total - 1.0) > PSD_SUM_TOL:
        warnings.warn(
            "psd_perc_in sums to %.6f, not 1: the particle size distribution "
            "accounts for %.4f%% of M_rock, so surface area and every "
            "dissolution flux through it are computed for a rock that much "
            "lighter than the one the mass balance carries."
            % (total, 100.0 * total),
            RuntimeWarning, stacklevel=3)
    return d, w


def _cap_initial_al(p, f0, H, gam, conc_in, f_CEC_in, den):
    """S1.0: hold the initial dissolved Al to what an Al(OH)3 phase allows.

    The IC helpers in ic.py (Amann, f_CEC_to_conc, Kelland, ...) derive the
    dissolved Al from the input exchanger fraction f_Al: free Al3+ from the
    Gaines-Thomas constant, then Al_w = Al3+ / (free fraction of the hydrolysis
    ladder). Nothing bounds it, and at pH 7 the free fraction is ~1e-7, so any
    f_Al above ~1e-5 becomes millimoles per litre of dissolved Al -- Amann
    starts at 27.6 mmol/l, 10^4-10^5 times gibbsite saturation (S0, vault note
    "Secondary mineral precipitation" section 5.3). A sink added later would
    remove that on its first step and be credited for what is really an
    initial-condition correction. This makes the correction once, at t = 0,
    where it belongs and where it can be measured on its own.

    What is held, and what gives. The measured pH is held (it is a measurement;
    H here is already the concentration the pH loop settled on). The TOTAL of
    each base cation implied by (conc_in, f_CEC_in) is held, so no Ca, Mg, Na or
    K is created or destroyed. Free Al3+ is set to saturation:
        a_Al3+ = K(T) * a_H+^3   ->   [Al3+] = K (gamma_H [H])^3 / gamma_Al
    and the exchanger re-partitions around it under the model's own equations
    11-15 with the same constants the first step will use, so step 1 starts at
    equilibrium instead of relaxing off a state its own chemistry rejects. The
    Al that leaves -- dissolved and exchanged -- is the amount precipitated as
    the phase at t = 0; it is reported (data["Al_ic_removed"]) rather than
    carried, since no solid Al pool exists until S1.

    f_Al falls, and the sites it frees go to the base cations and H+. WHERE the
    base cations come from is p.ic_al_hold, because it depends on what the
    dataset measured, and one of the two inputs has to give:
      "totals"   (default) each base cation's total is held; the sites are filled
                 FROM SOLUTION, so the aqueous concentrations fall. Right when
                 totals were measured (tePas: total_in + acid saturation). An and
                 the background inputs, which are scaled from the initial aqueous
                 composition, fall with them.
      "solution" the aqueous concentrations are held; the exchanger's base-cation
                 share grows to fill the sites, so base-cation totals rise. Right
                 when the solution was measured (Amann: Mg_in; Kelland: conc_in).
      "K_Ca_Al"  solution AND exchanger held; the literature selectivity K_Ca_Al
                 gives instead, recalibrated so the input f_Al sits in equilibrium
                 with saturated Al. Only the dissolved Al changes. Right when
                 f_Al was measured on an acid soil, where real exchangers hold a
                 lot of Al against water near Al(OH)3 saturation -- i.e. where
                 the constant, not the measurement, is the suspect.
    In the first two the clay's reserve of base cations (R_alk) grows by what
    f_Al lost, and so Alk_tot rises: with the pH held, sites that cannot be Al
    are base-cation sites, and that is reserve alkalinity.

    Returns None when the input is already at or below saturation -- the input
    is left exactly as given -- else a dict with the new conc and f_CEC.
    """
    cm, ca = p.conv_mol, p.conv_Al
    gH, gAl = gam[activity.SP_H], gam[activity.SP_AL]
    logK = smew.log_K_AlOH3(p.ic_al_cap, f0.T_K)
    Al_sat = 10 ** logK * (gH * H / cm) ** 3 / gAl * cm * ca    # [mol-conv_Al-conv/l]

    Ca0, Mg0, K0, Na0, Al_w0 = (float(np.ravel(x)[0]) for x in conc_in)
    Al_free0 = (H ** 4 / den) * Al_w0
    if not Al_free0 > Al_sat:
        return None
    Al_w = Al_sat * den / H ** 4

    # the constants step 1 will use, as concentration quotients
    (_, _, _, _, _, _, _, K_Mg, K_K, K_Na, K_Al, K_H) = activity.effective_constants(
        gam, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        p.K_Ca_Mg, p.K_Ca_K, p.K_Ca_Na, p.K_Ca_Al, p.K_Ca_H)

    V = p.n * f0.s * p.Zr * 1000
    C = p.CEC_tot
    fCa0, fMg0, fK0, fNa0, fAl0, fH0 = (float(np.ravel(x)[0]) for x in f_CEC_in)

    # A cap removes aluminium; it never adds it. The input pair (Al_w, f_Al)
    # need not satisfy equation 11 as the model writes it -- ic.py builds it on
    # the fixed 25 C hydrolysis ladder, the model has used the T-dependent one
    # since P5b -- so re-solving the exchanger at saturation can put MORE Al on
    # the clay than the input carried. Then the input's Al, re-equilibrated,
    # sits below saturation: nothing would precipitate, and relaxing the
    # partition is step 1's job, not the cap's. Leave the input alone.
    Al_tot0 = Al_w0 * V + fAl0 / 3 * C * ca

    def _only_removes(out):
        Al_tot_new = out["conc"][4] * V + out["f_CEC"][4] / 3 * C * ca
        return out if Al_tot_new < Al_tot0 else None
    tot = np.array([Ca0 * V + fCa0 / 2 * C, Mg0 * V + fMg0 / 2 * C,
                    Na0 * V + fNa0 * C, K0 * V + fK0 * C])

    def fractions(Ca, Mg, Na, K, fCa):
        r = np.sqrt(fCa / Ca)
        return (fCa, Mg * fCa / (K_Mg * Ca), K * r / np.sqrt(K_K), Na * r / np.sqrt(K_Na),
                (Al_sat / ca) * r ** 3 / np.sqrt(K_Al), H * r / np.sqrt(K_H))

    def res(y):
        Ca, Mg, Na, K, fCa = np.exp(y)
        fCa, fMg, fK, fNa, fAl, fH = fractions(Ca, Mg, Na, K, fCa)
        return np.array([(Ca * V + fCa / 2 * C) / tot[0] - 1,
                         (Mg * V + fMg / 2 * C) / tot[1] - 1,
                         (Na * V + fNa * C) / tot[2] - 1,
                         (K * V + fK * C) / tot[3] - 1,
                         fCa + fMg + fK + fNa + fAl + fH - 1])

    if p.ic_al_hold == "K_Ca_Al":
        # Every measurement held -- solution, exchanger, pH -- and the one
        # LITERATURE number gives: K_Ca_Al is recalibrated so that equation 11
        # puts the input f_Al in equilibrium with saturated Al, exactly as
        # ic.f_CEC_and_conc_to_K calibrates from paired measurements. Only the
        # dissolved Al changes at t = 0; the new constant holds for the run.
        out = {"conc": (Ca0, Mg0, K0, Na0, Al_w), "f_CEC": list(f_CEC_in)}
        if fAl0 > 0:
            K_eff = (Al_sat / ca) ** 2 * fCa0 ** 3 / (fAl0 ** 2 * Ca0 ** 3)
            out["K_Ca_Al"] = K_eff * gAl ** 2 / gam[activity.SP_CA] ** 3   # thermodynamic
        return _only_removes(out)

    if p.ic_al_hold == "solution":
        # The solution as given; only the exchanger re-partitions. Sum-to-one in
        # f_Ca alone -- ic.conc_to_f_CEC's own equation, with the capped Al.
        def res1(z):
            return np.array([sum(fractions(Ca0, Mg0, Na0, K0, np.exp(z[0]))) - 1])
        z, info, ier, msg = fsolve(res1, [np.log(fCa0)], full_output=True, xtol=1e-14)
        if not abs(res1(z)[0]) <= 1e-10:
            raise RuntimeError("ic_al_cap: exchanger re-partition did not converge "
                               "(residual %.3g): %s" % (abs(res1(z)[0]), msg))
        fCa, fMg, fK, fNa, fAl, fH = fractions(Ca0, Mg0, Na0, K0, np.exp(z[0]))
        return _only_removes({"conc": (Ca0, Mg0, K0, Na0, Al_w),
                              "f_CEC": [fCa, fMg, fK, fNa, fAl, fH]})

    y0 = np.log([Ca0, Mg0, Na0, K0, fCa0])
    y, info, ier, msg = fsolve(res, y0, full_output=True, xtol=1e-14)
    # Judged on the residual, not on ier: at xtol=1e-14 fsolve reports "no
    # further improvement" on states it has already solved to 1e-16.
    if not np.max(np.abs(res(y))) <= 1e-10:
        # Usually not a solver failure but an impossible request: with the pH
        # and every base-cation total held, the sites f_Al gives up must be
        # filled from solution, and the solution may simply not hold that much
        # charge. Say so, with the numbers, rather than blame convergence.
        have = (2 * Ca0 + 2 * Mg0 + Na0 + K0) * V           # [mol_c-conv]
        need = fAl0 * C                                      # upper bound
        raise ValueError(
            "ic_al_cap=%r with ic_al_hold='totals' has no solution: the exchanger "
            "gives up up to %.3g mol_c of Al sites, and holding pH and every "
            "base-cation total, they can only be refilled from solution, which "
            "holds %.3g mol_c (max residual %.3g). The input exchanger and "
            "solution cannot both be right here -- use ic_al_hold='solution' or "
            "'K_Ca_Al'." % (p.ic_al_cap, need / p.conv_mol, have / p.conv_mol,
                             np.max(np.abs(res(y)))))
    Ca, Mg, Na, K, fCa = np.exp(y)
    fCa, fMg, fK, fNa, fAl, fH = fractions(Ca, Mg, Na, K, fCa)
    return _only_removes({"conc": (Ca, Mg, K, Na, Al_w),
                          "f_CEC": [fCa, fMg, fK, fNa, fAl, fH]})


def _initial_state(p, f0, pH_in, conc_in, f_CEC_in, Si_in, CaCO3_in, MgCO3_in,
                   M_rock_in, rock_f_in, d_in, psd_perc_in, SSA_in, t_app,
                   s, T, L, keyword_add):
    """The state at t = 0, plus the two constants that only the ICs can fix.

    Returns (state, params, applied). `params` comes back refined because two of
    its fields cannot be known before this point: the fractal prefactor `a`,
    which is calibrated against a measured specific surface area, and the
    background solute inputs I_*, which are set from the initial concentrations.
    `applied` is the rock geometry at the application timestep, which the caller
    writes at index tt_app -- that index is not necessarily 0.
    """
    # --- pH: what was measured, and what the model needs -------------------
    #
    # Two conversions sit between a reported soil pH and the [H+] this function
    # has to start from, and until P6 the model did neither.
    #
    # 1. WHICH SOLUTION WAS MEASURED. `pH_method` says. A pH read in 0.01 M
    #    CaCl2 is not the same number as one read in water, and neither is
    #    automatically the pH of the soil solution at field moisture, which is
    #    what the model means by pH. `_PH_TO_SOLUTION` carries the offsets and
    #    the reasoning; "solution" is the default and subtracts nothing, which
    #    is what every run before P6 assumed.
    #
    # 2. ACTIVITY VERSUS CONCENTRATION. A pH electrode measures the ACTIVITY of
    #    H+. The model's H is a CONCENTRATION -- it has to be, because the
    #    Gaines-Thomas constants below are concentration-based (F4.6) and so is
    #    the aluminium ladder. So [H+] = 10^-pH / gamma_H, not 10^-pH.
    #
    # gamma_H needs the ionic strength, which needs the composition, which needs
    # H. That circularity is why this was deferred at F4.5, and it is resolved
    # here by the obvious means: iterate. Convergence is immediate because the
    # ionic strength is dominated by Ca, Mg, Na and K, which are INPUTS and do
    # not move between passes; only An shifts, and weakly. Two passes are
    # enough and three is the setting, so the loop is cheap insurance rather
    # than a solve.
    #
    # With gamma_on=False, gamma_H is exactly 1 and the loop exits on the first
    # pass with H = 10^-pH_in * conv_mol -- bit-identical to every pre-P6 run.
    # That is not a special case bolted on: an "activity pH" and a
    # "concentration pH" ARE the same number when every activity coefficient
    # is one.
    pH_sol = pH_in - _PH_TO_SOLUTION[p.pH_method]
    gamma_H_ic = 1.0
    gam_ic = activity.GAMMA_ONE
    al_cap = None
    for _pass in range(_IC_PH_MAX_PASSES):
        H = 10 ** (-pH_sol) / gamma_H_ic * p.conv_mol

        # pCO2
        CO2_air = (f0.r_het + f0.r_aut) / (f0.D * 1000 / (p.Z_CO2)) + p.CO2_atm
        Fs = f0.D / (p.Z_CO2) * (CO2_air - p.CO2_atm) * 1000   # [mol-conv/d]
        CO2_w = f0.k_H * CO2_air                               # Henry's law

        # carbonate system
        HCO3 = f0.k1 * CO2_w / H                               # [mol/l]
        CO3 = f0.k2 * f0.k1 * CO2_w / (H ** 2)                 # [mol/l]
        DIC = HCO3 + CO3 + CO2_w
        IC_tot = (DIC * f0.s + CO2_air * (1 - f0.s)) * (p.n * p.Zr * 1000)   # [mol]

        # Alk
        Alk = HCO3 + 2 * CO3 - H + f0.k_w / H

        # cations (mol/l)
        [Ca, Mg, K, Na, Al_w] = conc_in
        Al_w_in_ic = Al_w

        # anions (mol_c/l)
        An = 2 * Mg + 2 * Ca + Na + K - Alk                    # [mol_c/l]
        if An < 0:
            print(An)
            raise ValueError("Not enough cations for this alkalinity")

        # aluminium speciation
        K1, K2, K3, K4 = f0.K1, f0.K2, f0.K3, f0.K4
        den = H ** 4 + H ** 3 * K1 + H ** 2 * K1 * K2 + H * K1 * K2 * K3 + K1 * K2 * K3 * K4
        Al = (H ** 4 / den) * Al_w                             # mol/l
        AlOH = (H ** 3 * K1 / den) * Al_w
        AlOH2 = (H ** 2 * K1 * K2 / den) * Al_w
        AlOH3 = (H * K1 * K2 * K3 / den) * Al_w
        AlOH4 = Al_w - (Al + AlOH + AlOH2 + AlOH3)
        f_CEC = f_CEC_in

        # S1.0: dissolved Al that its own solubility cannot hold
        if p.ic_al_cap is not None:
            al_cap = _cap_initial_al(p, f0, H, gam_ic, conc_in, f_CEC_in, den)
            if al_cap is not None and "K_Ca_Al" in al_cap:
                p = replace(p, K_Ca_Al=al_cap["K_Ca_Al"])
            if al_cap is not None:
                Ca, Mg, K, Na, Al_w, f_CEC = al_cap["conc"] + (al_cap["f_CEC"],)
                Al = (H ** 4 / den) * Al_w
                AlOH = (H ** 3 * K1 / den) * Al_w
                AlOH2 = (H ** 2 * K1 * K2 / den) * Al_w
                AlOH3 = (H * K1 * K2 * K3 / den) * Al_w
                AlOH4 = Al_w - (Al + AlOH + AlOH2 + AlOH3)
                An = 2 * Mg + 2 * Ca + Na + K - Alk
                if An < 0:
                    raise ValueError(
                        "Not enough cations for this alkalinity after capping the "
                        "initial Al (ic_al_cap=%r): the exchanger took up %.3g of "
                        "the base cations' solution charge" % (p.ic_al_cap, -An))

        if not p.gamma_on:
            break
        OH_ic = f0.k_w / H if H > 0.0 else 0.0
        mu_ic = activity.ionic_strength(Ca, Mg, Na, K, H, OH_ic, HCO3, CO3, An,
                                        Al, AlOH, AlOH2, AlOH4,
                                        p.conv_mol, p.conv_Al)
        gam_ic = activity.gammas(mu_ic, f0.T_K)
        _g = gam_ic[activity.SP_H]
        if abs(_g - gamma_H_ic) <= _IC_PH_RTOL * _g:
            break                       # the state in hand is consistent with _g
        gamma_H_ic = _g
    else:
        raise ValueError(
            "initial-condition pH did not converge in %d passes: gamma_H is "
            "still moving by more than %g relative. This is the fixed point "
            "H = 10^-pH / gamma_H(mu(H)); a very low pH makes H itself a large "
            "part of the ionic strength, which is what slows it."
            % (_IC_PH_MAX_PASSES, _IC_PH_RTOL))

    # Silicon
    Si = Si_in

    # Background inputs (rain, litterfall, background weathering..)
    if keyword_add == 1:
        mTL, ms = np.mean(T + L), np.mean(s)
        I_An = mTL * 1000 * An * f0.s / ms                 # [mol_c d-1]
        I_Ca = mTL * 1000 * Ca * f0.s / ms                 # [mol d-1]
        I_Mg = mTL * 1000 * Mg * f0.s / ms
        I_Na = mTL * 1000 * Na * f0.s / ms
        I_K = mTL * 1000 * K * f0.s / ms
        I_Si = mTL * 1000 * Si * f0.s / ms
    elif keyword_add == 0:
        I_An = I_Ca = I_Mg = I_K = I_Na = I_Si = 0

    # CEC adsorbed species
    [f_Ca, f_Mg, f_K, f_Na, f_Al, f_H] = f_CEC

    # reserve of alkalinity
    R_alk = (f_Mg + f_Ca + f_Na + f_K) * p.CEC_tot         # [mol_c]

    # total amounts (solution and adsorbed)
    #
    # Written out rather than factored through a common `n*s*Zr*1000`: floating
    # point multiplication is not associative, so Ca*n*s*Zr*1000 (left to right)
    # and Ca*(n*s*Zr*1000) differ in the last bits. That difference is real --
    # it moved the golden masters by up to 1e-11 relative over a run -- and the
    # groupings below are the ones the model has always used. Note that the
    # aluminium and silicon lines further down order it n*Zr*s, not n*s*Zr, and
    # that the Alk_tot line does parenthesise the product. Both are deliberate
    # here, and neither is worth "tidying".
    Ca_tot = Ca * p.n * f0.s * p.Zr * 1000 + f_Ca / 2 * p.CEC_tot     # [mol]
    Mg_tot = Mg * p.n * f0.s * p.Zr * 1000 + f_Mg / 2 * p.CEC_tot     # [mol]
    K_tot = K * p.n * f0.s * p.Zr * 1000 + f_K * p.CEC_tot            # [mol]
    Na_tot = Na * p.n * f0.s * p.Zr * 1000 + f_Na * p.CEC_tot         # [mol]
    Alk_tot = (2 * Mg_tot + 2 * Ca_tot + Na_tot + K_tot
               - An * (p.n * f0.s * p.Zr * 1000))                     # [mol_c]
    An_tot = An * p.n * f0.s * p.Zr * 1000                            # [mol_c]
    Al_tot = Al_w * p.n * p.Zr * f0.s * 1000 + (f_Al / 3) * p.CEC_tot * p.conv_Al
    Si_tot = Si * p.n * p.Zr * f0.s * 1000

    # Carbonate minerals (considered as an additional pool)
    CaCO3, MgCO3 = CaCO3_in, MgCO3_in

    # --- F4: activities at t = 0 ---------------------------------------------
    #
    # A caveat that belongs here and not in a commit message. The initial
    # condition is resolved by ic.py, whose Gaines-Thomas conversions
    # (conc_to_f_CEC and friends) are written on CONCENTRATIONS. So on a
    # gamma_on run the exchanger composition handed in at t = 0 is not quite the
    # one the model's own equilibrium would produce from the same totals, and
    # the first few steps relax off it. That is a property of the IC being an
    # INPUT, not a solve: conc_in and f_CEC_in are measurements, and the six
    # conversion helpers in ic.py are still written on concentrations.
    #
    # P6 closed the pH half of that gap -- the loop above now reads pH_in as the
    # activity it is -- but conc_in and f_CEC_in are still taken at face value.
    if p.gamma_on:
        OH0 = f0.k_w / H if H > 0.0 else 0.0
        mu = activity.ionic_strength(Ca, Mg, Na, K, H, OH0, HCO3, CO3, An,
                                     Al, AlOH, AlOH2, AlOH4,
                                     p.conv_mol, p.conv_Al)
        gam = activity.gammas(mu, f0.T_K)
    else:
        mu = 0.0
        gam = activity.GAMMA_ONE
    gamma_H = gam[activity.SP_H]
    # Both pH series, from the H the loop above settled on. `pH` is the ACTIVITY
    # pH and should come back equal to pH_sol -- the measured value, method-
    # corrected -- because that is what H was derived from; the identity is
    # asserted rather than assumed, since it is the cheapest possible check that
    # the iteration converged and the two conversions did not cancel wrongly.
    pH_conc = -np.log10(H / p.conv_mol)
    pH = -np.log10(gamma_H * H / p.conv_mol)
    assert abs(pH - pH_sol) < 1e-8, (
        "initial-condition pH did not converge: asked for activity pH %.6f, "
        "the derived state has %.6f" % (pH_sol, pH))

    # Carbonate weathering
    Omega_CaCO3 = (gam[activity.SP_CA] * Ca) * (gam[activity.SP_CO3] * CO3) / f0.K_CaCO3
    Omega_MgCO3 = (gam[activity.SP_MG] * Mg) * (gam[activity.SP_CO3] * CO3) / f0.K_MgCO3
    [W_CaCO3, W_MgCO3] = smew.carb_W(CaCO3, MgCO3, Omega_CaCO3, Omega_MgCO3,
                                     f0.s, p.Zr, p.r_CaCO3, p.r_MgCO3,
                                     p.tau_CaCO3, p.tau_MgCO3)

    # zero rock geometry, replaced below when there is rock
    nm, nd = p.number_min, p.n_d_cl
    M_min = np.zeros(nm); rock_f = np.zeros(nm)
    EW = np.zeros(nm); Wr = np.zeros(nm); Omega = np.zeros(nm)
    d = np.zeros(nd); delta_d = np.zeros(nd)
    lamb = np.zeros(nd); SSA = np.zeros(nd); psd = np.zeros(nd)
    M_rock = 0.0; SA = 0.0
    applied = None

    # Silicate weathering
    if p.has_rock:
        # rock composition
        M_rock_a = M_rock_in                               # [g/m2]
        rock_f_a = np.asarray(rock_f_in, dtype=float).copy()
        M_min_a = rock_f_a * M_rock_a                      # [g/m2]
        M_iner = M_rock_a * (1 - np.sum(rock_f_a))         # [g/m2]

        # diameter classes. Sorted and checked first -- see _validate_psd.
        d_a, psd_perc_a = _validate_psd(d_in, psd_perc_in)  # [m], [-]
        delta_d_a = np.insert(np.diff(d_a), 0, d_a[0])

        # particle size distribution
        psd_a = psd_perc_a * M_rock_a / delta_d_a          # [g/m]

        # refinement of fractal constant based on measured SSA
        a = p.a
        if SSA_in > 0:
            a = (SSA_in * p.rho_rock * M_rock_a / 6) / np.sum(
                d_a ** (p.b - 1) * psd_a * delta_d_a)      # [m**-b]

        # surface area
        lamb_a = a * d_a ** p.b                            # [-]
        SSA_a = 6 / (d_a * p.rho_rock) * lamb_a            # [m2/g]
        SA_a = np.sum(SSA_a * psd_a * delta_d_a)           # [m2]
        if p.n_blocks > 1:
            # room for the second application's block, empty (d = 0) until it lands
            d_a, delta_d_a, lamb_a, SSA_a, psd_a = (
                np.concatenate([x, np.zeros(p.n_d_cl - len(x))])
                for x in (d_a, delta_d_a, lamb_a, SSA_a, psd_a))

        p = replace(p, a=a, M_iner=M_iner)
        applied = (M_min_a, M_rock_a, rock_f_a, d_a, delta_d_a,
                   lamb_a, SSA_a, psd_a, SA_a)

        # mineral weathering. Only when the rock goes on at t = 0; otherwise
        # index 0 keeps the zeros allocated above, as it always has.
        if t_app == 0:
            M_min, M_rock, rock_f = M_min_a, M_rock_a, rock_f_a
            d, delta_d, lamb, SSA, psd, SA = \
                d_a, delta_d_a, lamb_a, SSA_a, psd_a, SA_a
            a_Ca0 = gam[activity.SP_CA] * Ca
            a_Mg0 = gam[activity.SP_MG] * Mg
            a_K0 = gam[activity.SP_K] * K
            a_Na0 = gam[activity.SP_NA] * Na
            a_H0 = gamma_H * H
            a_Al0 = gam[activity.SP_AL] * Al
            for j in range(0, p.number_min):
                Omega[j] = weathering_kinec.Omega_sil(
                    p.mineral[j], a_Ca0, a_Mg0, a_K0, a_Na0, Si, a_H0, a_Al0,
                    _fe2_for_omega(p, 0.0, mu, f0.T_K), p.K_sp[j], p.conv_mol,
                    p.conv_Al)  # [-]
                Wr[j] = f0.s * p.diss_f * weathering_kinec.mineral_weathering(
                    p.mineral[j], f0.T_K, Omega[j], a_H0, a_Al0,
                    p.conv_mol, p.conv_Al)
                EW[j] = Wr[j] * SA * rock_f[j]             # [mol-conv/d]

    p = replace(p, I_Ca=I_Ca, I_Mg=I_Mg, I_K=I_K, I_Na=I_Na, I_Si=I_Si, I_An=I_An)

    # S1: no secondary solid at t = 0 (no initial inventory yet, plan D2), and
    # the first step's rates from the initial solution
    M_sec = np.zeros(p.n_sec)
    Omega_sec = W_sec = lim_sec = np.zeros(0)
    if p.n_sec:
        Omega_sec, W_sec, lim_sec = _secondary(
            p, f0.T_K, f0.s, f0.L, f0.T, gam, H, Ca, Mg, K, Na, Al,
            Al / Al_w if Al_w > 0 else 0.0, Si, M_sec,
            p.n * p.Zr * f0.s * 1000, Fe=0.0, mu=mu)

    state = SoilState(
        pH=pH, pH_conc=pH_conc, H=H, f_H=f_H, Alk=Alk, Alk_tot=Alk_tot, An=An, An_tot=An_tot,
        CO2_w=CO2_w, CO2_air=CO2_air, HCO3=HCO3, CO3=CO3, DIC=DIC,
        IC_tot=IC_tot, Fs=Fs, ADV=0.0,
        Ca=Ca, Ca_tot=Ca_tot, f_Ca=f_Ca, Mg=Mg, Mg_tot=Mg_tot, f_Mg=f_Mg,
        K=K, K_tot=K_tot, f_K=f_K, Na=Na, Na_tot=Na_tot, f_Na=f_Na,
        Si=Si, Si_tot=Si_tot,
        # No Fe in the initial condition: there is no Fe_in argument, and the
        # pool exists to account what the ROCK releases, which is zero at t=0.
        Fe=0.0, Fe_tot=0.0,
        Al=Al, AlOH=AlOH, AlOH2=AlOH2, AlOH3=AlOH3, AlOH4=AlOH4,
        Al_w=Al_w, Al_tot=Al_tot, f_Al=f_Al, R_alk=R_alk,
        mu=mu, gamma_H=gamma_H,
        CaCO3=CaCO3, MgCO3=MgCO3,
        Omega_CaCO3=Omega_CaCO3, Omega_MgCO3=Omega_MgCO3,
        W_CaCO3=W_CaCO3, W_MgCO3=W_MgCO3,
        M_rock=M_rock, SA=SA, M_min=M_min, rock_f=rock_f,
        EW=EW, Wr=Wr, Omega=Omega,
        d=d, delta_d=delta_d, lamb=lamb, SSA=SSA, psd=psd,
        clip_M_min=np.zeros(p.number_min),
        clip_psd=0.0,
        M_sec=M_sec, W_sec=W_sec, Omega_sec=Omega_sec,
        clip_sec=np.zeros(p.n_sec), lim_sec=lim_sec,
    )
    if al_cap is not None:
        p = replace(p, ic_al_removed=(Al_w_in_ic * p.n * p.Zr * f0.s * 1000
                                      + (f_CEC_in[4] / 3) * p.CEC_tot * p.conv_Al
                                      - Al_tot))
    return state, p, applied


# ---------------------------------------------------------------------------
# One timestep
# ---------------------------------------------------------------------------

def _gammas_for(state, p, T_K, k_w):
    """(mu, gamma) from a state. One definition, used by both coupling modes.

    Which STATE is passed is the whole difference between lagged and Picard:
    lagged passes the step being advanced from, Picard passes the step's own
    provisional answer. Everything downstream is identical, which is why the
    two can share a residual, a Jacobian and a gate.
    """
    if not p.gamma_on:
        return 0.0, activity.GAMMA_ONE
    OH = k_w / state.H if state.H > 0.0 else 0.0
    mu = activity.ionic_strength(
        state.Ca, state.Mg, state.Na, state.K, state.H, OH,
        state.HCO3, state.CO3, state.An,
        state.Al, state.AlOH, state.AlOH2, state.AlOH4,
        p.conv_mol, p.conv_Al)
    return mu, activity.gammas(mu, T_K)


def _fe2_for_omega(p, Fe, mu, T_K):
    """The Fe+2 activity Omega_sil gets for its Fe end-members (augite's
    ferrosilite, olivine's fayalite) [mol-conv/l], by fe_mode.

    "tracer": None, which Omega_sil reads as its historical literal 1 --
    bit-identical, and 1 mol/l, i.e. not a concentration at all.
    "oxic": 0. Released Fe is Fe(III) at once; Fe+2 is not there.
    "reduced": the dissolved Fe, which is all Fe+2, times its activity
    coefficient (phreeqc.dat ion size 6.0).
    """
    if p.fe_mode == "tracer":
        return None
    if p.fe_mode == "oxic":
        return 0.0
    g = activity.gamma_ion(mu, T_K, 2, activity.A0_FE2) if p.gamma_on else 1.0
    return g * Fe


def _secondary(p, T_K, s, L, T, gam, H, Ca, Mg, K, Na, Al, Al_free_frac, Si, M_sec,
               nZrs, Fe=0.0, mu=0.0):
    """S1: saturation and rate of every secondary phase, from one solution.

    Returns (Omega, W, lim), one entry per phase in p.sec_phases. W is the
    DISSOLUTION rate [mol-conv/d] -- negative is precipitation, the carbonate
    sign convention -- to be applied over the NEXT step.

    THE RATE. Transition-state theory, R = s k(T, a_H) A (1 - Omega), the same
    form in both directions -- unless the phase carries its own growth law
    (kaolinite: R = -s k_ppt A ln Omega, Yang & Steefel 2008). s because the model scales every mineral surface
    by wetness (silicates, carbonates); A [m2 per m2 of ground] is growth on
    what has formed, floored by a nucleation surface:
        A = max(A_seed Zr, SSA MM M_sec)
    An equilibrium phase (kin None) has no rate: the limiter alone sets W.

    THE LIMITER (plan D1). An explicit rate at Omega = 1e3 overshoots past
    saturation in one step and oscillates; F2 stops a pool going below zero,
    and this stops a solution going below saturation, from the other side.
    Precipitation is capped at what would bring the DISSOLVED element to the
    phase's saturation at this state's pH, net of what the next step's
    drainage removes anyway; dissolution at what would bring it
    up to saturation, and at the solid present. Sized for one full run step
    (p.dt_run), so a shorter sub-step can only take less.

    Why the dissolved excess and not a re-solved equilibrium: SMEW's
    alkalinity is the Al(OH)3-referenced one (S0.2), so removing dissolved Al
    leaves Alk_tot alone; the exchanger then gives some Al back and the pH
    falls, and both push Omega UP. So the closed form undershoots equilibrium
    and never overshoots it -- the next steps finish the approach. That is
    measured, not assumed: tests/check_secondary.py.

    The price, also measured. Where the exchanger holds much Al -- acid soil --
    it hands back most of what each step removes, so an "equilibrium" phase
    approaches saturation over many steps, at a pace set by dt and the
    exchanger's share of the pool, not by any rate constant. Sizing the step
    on the whole pool instead (the exchanger scaled with Al3+) was tried and
    is far worse: stripping exchanger Al moves base cations onto the clay, the
    pH falls 1.3 units in one step and Omega swings to 1e-6 -- the overshoot
    the limiter exists to prevent. Above pH ~6.5, where the clay holds almost
    no Al (S0.3), the dissolved excess IS the whole excess and one step lands
    on saturation.

    Phases are visited in p.sec_phases order and share each element's excess:
    whatever an earlier phase took is no longer there for a later one. With
    Al(OH)3(a) first and gibbsite second, the amorphous ceiling is enforced at
    once, gibbsite grows slowly below it, and the amorphous solid then
    redissolves into the deficit gibbsite leaves -- Ostwald ripening, from
    nothing more than the order.
    """
    n = p.n_sec
    Omega, W, lim = np.zeros(n), np.zeros(n), np.zeros(n)
    cm, ca = p.conv_mol, p.conv_Al
    a_H = gam[activity.SP_H] * H / cm
    # activities [mol/l] over min_st's columns. Si is H4SiO4 and uncharged; Fe
    # has no aqueous species yet (p.Fe), the same caveat as Omega_sil.
    g = (gam[activity.SP_CA], gam[activity.SP_MG], gam[activity.SP_K],
         gam[activity.SP_NA], gam[activity.SP_AL], 1.0, 1.0)
    # S3: under fe_mode="oxic" the dissolved Fe pool is Fe(III), and its free
    # Fe+3 is a tiny, pH-set fraction of it (Fe3_free_fraction, ~1e-9 at pH 7)
    g_fe, fe_frac = 1.0, 1.0
    if p.fe_mode == "oxic":
        g_spec = None
        if p.gamma_on:
            g_fe = activity.gamma_ion(mu, T_K, 3, activity.A0_FE3)
            g_spec = (g_fe,) + tuple(activity.gamma_ion(mu, T_K, z, a0)
                                     for z, a0 in smew.FE3_SPECIES_Z_A0)
        fe_frac = smew.Fe3_free_fraction(a_H, T_K, g_spec)
    g = g[:6] + (g_fe,)
    a = np.array([g[0] * Ca / cm, g[1] * Mg / cm, g[2] * K / cm, g[3] * Na / cm,
                  g[4] * Al / (cm * ca), Si / cm, g_fe * fe_frac * Fe / cm])
    # dissolved total per element [model units/l], and model units per mol-conv
    w = (Ca, Mg, K, Na, Al / Al_free_frac if Al_free_frac > 0 else 0.0, Si, Fe)
    unit = (1.0, 1.0, 1.0, 1.0, ca, 1.0, 1.0)
    removed = np.zeros(7)              # taken out of solution so far [mol-conv]
    frac = (1.0, 1.0, 1.0, 1.0, Al_free_frac, 1.0, fe_frac)  # free / dissolved total
    for j, name in enumerate(p.sec_phases):
        ph = smew.sec_phase(name)
        st = p.sec_st[j]
        els = np.flatnonzero(st)
        logK = smew.sec_log_K(name, T_K)
        if np.all(a[els] > 0):
            Omega[j] = 10 ** (np.sum(st[els] * np.log10(a[els]))
                              - ph["nu_H"] * np.log10(a_H) - logK)
        # Xp: how much of the PHASE [mol-conv] would bring the water to its
        # saturation -- positive, precipitable; negative, the deficit a solid
        # could dissolve into. Net of what the next step drains anyway
        # (drainage takes Al; drainage and transpiration take the rest, as
        # the balances in _step_once do) and of what earlier phases took.
        if len(els) == 1:
            c = int(els[0])
            nu = st[c]
            # closed form: dissolved total at saturation from the free
            # activity, through the activity coefficient and, for Al, the
            # hydrolysis ladder's free fraction
            a_sat = (10 ** logK * a_H ** ph["nu_H"]) ** (1 / nu)
            free_sat = a_sat / g[c] * cm * unit[c]
            w_sat = free_sat / frac[c] if c in (4, 6) else free_sat
            # drainage takes Al and Fe; drainage and transpiration the rest
            out_l = (L if c in (4, 6) else L + T) * 1000 * p.dt_run   # [l per m2]
            X = ((w[c] - w_sat) * nZrs - w[c] * out_l) / unit[c] - removed[c]
            Xp = X / nu
        else:
            Xp = _sec_extent(st, els, ph["nu_H"], logK, a_H, g, frac, w, unit,
                             removed, nZrs, L, T, cm, M_sec[j], p.dt_run)
        if ph.get("from_solution", True) is False:
            continue                     # forms only by ageing (goethite): below
        cap_pre = max(Xp, 0.0) / p.dt_run                    # [mol-conv/d] phase
        sat_dis = max(-Xp, 0.0)
        cap_dis = min(sat_dis, M_sec[j]) / p.dt_run
        # what bound a dissolution: 1 saturation, 2 the solid running out
        why_dis = 1.0 if sat_dis <= M_sec[j] else 2.0
        if ph["kin"] is None:
            W[j] = cap_dis if Xp < 0 else -cap_pre
            if W[j] != 0.0:
                lim[j] = why_dis if Xp < 0 else 1.0
        else:
            A = max(p.sec_A_seed[j] * p.Zr, ph["SSA"] * ph["MM"] * M_sec[j] / cm)
            k_ppt = smew.sec_k_ppt(name, T_K)
            if Omega[j] > 1 and k_ppt is not None:
                # growth by its own law (kaolinite: Yang & Steefel 2008 eq. 9)
                R = -s * k_ppt * 86400 * cm * A * np.log(Omega[j])
            else:
                k = smew.sec_k(name, T_K, a_H) * 86400 * cm  # [mol-conv/m2/d]
                R = s * k * A * (1 - Omega[j])
            if R < -cap_pre:
                R, lim[j] = -cap_pre, 1.0
            elif R > cap_dis and M_sec[j] > 0:
                R, lim[j] = cap_dis, why_dis
            elif R > 0 and M_sec[j] == 0:
                R = 0.0                  # nothing to dissolve: not a limit
            W[j] = R
        for c in els:
            removed[c] -= W[j] * p.dt_run * st[c]

    # S3: ageing, solid to solid (ferrihydrite -> goethite). Dissolution-
    # reprecipitation is the mechanism, so it is written as exactly that: the
    # parent dissolves k M and the product precipitates the same amount, in the
    # same step. The water sees nothing, and every ledger row already covers it.
    # Capped at what the parent will still hold after its own W this step.
    if any(smew.sec_phase(n).get("ages_to") for n in p.sec_phases):
        pH_a = -np.log10(a_H)
        for j, name in enumerate(p.sec_phases):
            k_age = smew.sec_k_age(name, pH_a)
            if k_age is None:
                continue
            target = smew.sec_phase(name)["ages_to"][0]
            if target not in p.sec_phases:
                continue
            left = max(M_sec[j] - max(W[j], 0.0) * p.dt_run, 0.0)
            tr = min(k_age * M_sec[j], left / p.dt_run)       # [mol-conv/d]
            W[j] += tr
            W[p.sec_phases.index(target)] -= tr
    return Omega, W, lim


def _sec_extent(st, els, nu_H, logK, a_H, g, frac, w, unit, removed, nZrs, L, T,
                cm, M_sol, dt_run):
    """S2: the limiter for a phase that holds more than one element.

    Precipitating xi moles of kaolinite takes 2 xi Al AND 2 xi Si, so no
    single element's excess says how far is too far. Instead solve
        log Omega(xi) = sum_c nu_c log a_c(xi) - nu_H log a_H - log K = 0,
    with each element's dissolved pool reduced by nu_c xi. log Omega falls
    monotonically in xi, so a bracketed root is exact, and the same sign
    conventions as the closed form hold: xi > 0 precipitable, xi < 0 the
    deficit a solid could dissolve into (bracket stopped at the solid present).
    At fixed pH and free fraction, as the closed form also assumes.
    """
    # dissolved amount per element left for this phase [mol-conv]
    D = {}
    for c in els:
        out_l = (L if c in (4, 6) else L + T) * 1000 * dt_run
        D[c] = (w[c] * nZrs - w[c] * out_l) / unit[c] - removed[c]

    def logO(xi):
        tot = -nu_H * np.log10(a_H) - logK
        for c in els:
            amt = D[c] - st[c] * xi
            if amt <= 0:
                return -np.inf
            # activity: dissolved amount -> concentration -> free -> activity;
            # the model's unit factors (conv_Al for Al) cancel
            tot += st[c] * np.log10(g[c] * frac[c] * amt / (nZrs * cm))
        return tot

    hi = min(D[c] / st[c] for c in els)                   # all of one element gone
    f0 = logO(0.0)
    if f0 > 0:
        return brentq(logO, 0.0, hi * (1 - 1e-12), xtol=1e-14 * max(hi, 1e-300),
                      rtol=1e-12) if hi > 0 else 0.0
    if not M_sol > 0:
        return 0.0
    if logO(-M_sol) <= 0:
        return -M_sol                       # all the solid fits under saturation
    return brentq(logO, -M_sol, 0.0, xtol=1e-14 * M_sol, rtol=1e-12)


def _step_once(state, params, now, prev, dt, post_application=True,
               rock_now=None, gam_in=None):
    """One forward-Euler-plus-implicit-solve timestep. See step() for the wrapper.

    Pure: `state` is never modified.

        state   the state to advance FROM. Not necessarily the previous index:
                through a frost the model holds, then resumes from the last
                unfrozen state, and the caller passes that one.
        now     this timestep's forcing
        prev    the forcing at the step `state` belongs to. The explicit
                balances evaluate their loss terms at that step, not at this
                one, so both are needed.
        post_application
                whether the rock geometry evolves this step. False before and
                exactly at the application timestep, where the geometry is the
                applied one and `rock_now` supplies it.
        gam_in  (mu, gamma) to use instead of computing them from `state`.
                None is the lagged coupling and the default; step() passes this
                when iterating gamma to self-consistency (F4.7).

    Returns (new_state, diagnostics). diagnostics carries the uptake fluxes and
    the solver residuals; see biogeochem_balance for why uptake is not simply
    part of the returned state.
    """
    p = params

    # --- F4: activities ------------------------------------------------------
    #
    # gamma is LAGGED -- computed from `state`, the step being advanced FROM,
    # and therefore constant through this step's nonlinear solve. That is what
    # lets every activity correction fold into the eleven constants below
    # without touching the residual or F3.5's analytic Jacobian; see the long
    # note in smew/activity.py. Picard iteration would preserve that property
    # too; only fully-coupled gamma would break it.
    #
    # OH- is estimated from the UNCORRECTED k_w, because the corrected one needs
    # the gamma this is an input to. The circularity is not worth resolving: at
    # soil pH, OH- is ~1e-7 mol/l against ~1e-3 for calcium, so its contribution
    # to mu is six orders below the terms that set it.
    if gam_in is not None:
        mu, gam = gam_in                     # a Picard iterate; see step()
    else:
        mu, gam = _gammas_for(state, p, now.T_K, now.k_w)
    # Concentration quotients, i.e. exactly what the residual has always taken.
    # At gamma = 1 this is an exact identity, not an approximation to it.
    (k1, k2, k_w, K1, K2, K3, K4,
     K_Ca_Mg, K_Ca_K, K_Ca_Na, K_Ca_Al, K_Ca_H) = activity.effective_constants(
        gam, now.k1, now.k2, now.k_w, now.K1, now.K2, now.K3, now.K4,
        p.K_Ca_Mg, p.K_Ca_K, p.K_Ca_Na, p.K_Ca_Al, p.K_Ca_H)
    gamma_H = gam[activity.SP_H]

    # CO2 advection due to moisture variation
    ADV = 0.0
    if now.s < prev.s:
        ADV = p.n * p.Zr * 1000 * (now.s - prev.s) * p.CO2_atm      # [mol]
    elif now.s > prev.s:
        ADV = p.n * p.Zr * 1000 * (now.s - prev.s) * state.CO2_air

    # active uptake [Ca, Mg, K, Si]
    UP_act = smew.up_act(now.v, (now.v - prev.v), p.xi, dt, prev.T,
                         state.Ca, state.Mg, state.K, state.Si, prev.Dw,
                         p.Zr, p.k_v, p.RAI, p.root_d)
    UP_Ca, UP_Mg, UP_K, UP_Si = UP_act                              # [mol-conv/d]

    # explicit mass balances # [mol]
    min_st, EW_p = p.min_st, state.EW
    LT = (prev.L + prev.T) * 1000
    Ca_tot = state.Ca_tot + (p.I_Ca + np.sum(min_st[:, 0] * EW_p) + state.W_CaCO3
                             - LT * state.Ca - UP_Ca) * dt
    Mg_tot = state.Mg_tot + (p.I_Mg + np.sum(min_st[:, 1] * EW_p) + state.W_MgCO3
                             - LT * state.Mg - UP_Mg) * dt
    K_tot = state.K_tot + (p.I_K + np.sum(min_st[:, 2] * EW_p)
                           - LT * state.K - UP_K) * dt
    Na_tot = state.Na_tot + (p.I_Na + np.sum(min_st[:, 3] * EW_p)
                             - LT * state.Na) * dt
    # Al_w, not (Al + AlOH4). Drainage removes whatever is dissolved, and Al_w
    # IS the dissolved total -- the expansion defines AlOH4 = Al_w - (Al + AlOH
    # + AlOH2 + AlOH3), so the four hydroxide species and the free ion sum to it
    # exactly. Removing only the free ion and the fully hydrolysed anion left
    # AlOH, AlOH2 and AlOH3 in the soil forever: 0.2 % to 68 % of dissolved
    # aluminium depending on pH, since the middle of the ladder dominates
    # between about pH 5 and 6.5. It is the only cation whose drainage term was
    # written species by species rather than as the total, which is how the
    # three got lost.
    #
    # Al still has no transpiration sink while every other cation does. That is
    # deliberately NOT fixed here: whether transpiration removes solutes at all
    # is one question asked of every species at once, and answering it for
    # aluminium alone would bury it.
    Al_tot = state.Al_tot + (np.sum(min_st[:, 4] * EW_p) * p.conv_Al
                             - prev.L * 1000 * state.Al_w) * dt
    Si_tot = state.Si_tot + (p.I_Si + np.sum(min_st[:, 5] * EW_p)
                             - LT * state.Si - UP_Si) * dt
    # --- P3: iron ------------------------------------------------------------
    #
    # A MASS TRACER, deliberately, and it is worth being explicit about what
    # that means because the temptation is to do more.
    #
    # Iron is in three of the database's formulas -- augite Fe0.275, basalt
    # glass Fe0.19, olivine Fe0.2 -- and min_st had no column for it, so the
    # rock lost its full formula mass while only part of its cations reached
    # solution. Measured: 27.5 % of the cations released by tePas's augite and
    # 10.0 % of Kelland's olivine simply vanished.
    #
    # What this pool does: accounts the release, removes it by drainage,
    # closes the rock-to-solution balance.
    #
    # What it deliberately does NOT do, and why. It does not enter alkalinity.
    # Fe2+ leaving a silicate consumes 2 H+ exactly as Mg2+ does, so it
    # generates alkalinity -- but if it then oxidises and precipitates as
    # Fe(OH)3 it consumes that alkalinity straight back, and which happens is a
    # redox question this model cannot answer. Crediting the alkalinity would
    # be a guess at the favourable end; leaving the mass unaccounted, as
    # before, was a guess at nothing. So the pool bounds the open question
    # instead: the alkalinity Fe is worth lies between zero (full oxidation and
    # precipitation) and 2 eq/mol (none), and the size of that interval is now
    # a number rather than an omission. It also stays out of the exchanger, out
    # of the charge balance and out of the ionic strength, since all three need
    # a valence this model has not got.
    #
    # Fe is treated as fully aqueous and fully mobile, which is the LEAST
    # conservative assumption for how fast it leaves and therefore the right
    # one for a tracer whose job is to bound a missing term.
    Fe_tot = state.Fe_tot + (np.sum(min_st[:, 6] * EW_p)
                             - prev.L * 1000 * state.Fe) * dt
    An_tot = state.An_tot + (p.I_An - (prev.L + prev.T) * state.An * 1000) * dt
    IC_tot = (state.IC_tot + now.I * 1000 * now.DIC_rain - ADV
              + (state.W_CaCO3 + state.W_MgCO3 + prev.r_het + prev.r_aut
                 - state.Fs - prev.L * 1000 * state.DIC) * dt)

    # --- S1: secondary minerals ----------------------------------------------
    # What the phases dissolved (W > 0) or precipitated (W < 0) over this step,
    # at the rates the previous state set -- the carbonate scheme. A separate
    # statement, after the totals are formed, so that with no phase on not one
    # floating-point operation above has moved (the golden masters are
    # bit-identical). Before F2, so the positivity limiter sees it.
    if p.n_sec:
        sec_src = state.W_sec @ p.sec_st                       # [mol-conv/d]
        Ca_tot = Ca_tot + sec_src[0] * dt
        Mg_tot = Mg_tot + sec_src[1] * dt
        K_tot = K_tot + sec_src[2] * dt
        Na_tot = Na_tot + sec_src[3] * dt
        Al_tot = Al_tot + sec_src[4] * p.conv_Al * dt
        Si_tot = Si_tot + sec_src[5] * dt
        Fe_tot = Fe_tot + sec_src[6] * dt

    # --- F2: positivity ------------------------------------------------------
    # Forward Euler with loss terms evaluated at the previous step has no lower
    # bound: if dt x loss exceeds the pool, the pool goes negative and the
    # implicit solve below is handed a negative total, which either fails or
    # returns a state that propagates.
    #
    # The limiter is written as a clamp on the ALREADY-COMPUTED total rather
    # than as a rearrangement into sources-minus-sinks, deliberately. Splitting
    # `pool + (a + b + c - d - e)*dt` into `pool + (src - loss)*dt` re-associates
    # the sum and changes the last bits -- which is exactly the mistake F1 made
    # and the golden masters caught. This form is bit-identical whenever no
    # clipping occurs, which is every step of every benchmark case.
    #
    # dt_max is the largest step that would have kept every pool non-negative.
    # It follows from the totals already in hand: if a pool falls from `pool` to
    # `tent` over dt, the net loss rate is (pool - tent)/dt, so the pool would
    # reach zero at dt * pool / (pool - tent). Reported per step; the minimum
    # over a run is the model's documented stable dt, which nothing has ever
    # stated before.
    clip_Ca = clip_Mg = clip_K = clip_Na = 0.0
    clip_Al = clip_Si = clip_An = clip_C = clip_Fe = 0.0
    #
    # `_pool > 0.0` is not a strong enough test, and F3.5 is what showed it.
    # On a forsterite feedstock there is no aluminium, so `Al_tot` is float
    # noise -- 1e-56 -- and it still satisfies `> 0`. Every one of the two
    # hundred smallest dt_max values on `Example` is set by that pool. The
    # number F2 introduced as "the model's documented stable dt" was therefore,
    # on any Al-free run, a statement about a pool that is not there: it moved
    # by six orders of magnitude (0.023 d -> 4.3e-08 d) purely because F3.5's
    # last bits landed differently, while every real pool moved by ~1e-12.
    #
    # The fix is the floor this repo already uses in three other places --
    # F2's CLIP_REL, the ledger's ACTIVITY_FLOOR, F3.2's SCALE_FLOOR: a pool is
    # only worth limiting the timestep if it is a pool at all, judged relative
    # to the largest one in the system rather than against an absolute constant.
    # Never gate on, and never divide by, a quantity that can legitimately be
    # zero.
    _pools = ((state.Ca_tot, Ca_tot), (state.Mg_tot, Mg_tot),
              (state.K_tot, K_tot), (state.Na_tot, Na_tot),
              (state.Al_tot, Al_tot), (state.Si_tot, Si_tot),
              (state.An_tot, An_tot), (state.IC_tot, IC_tot),
              (state.Fe_tot, Fe_tot))
    _floor = SCALE_FLOOR * max(abs(_p) for _p, _ in _pools)
    dt_max = float("inf")
    for _pool, _tent in _pools:
        if _tent < _pool and _pool > _floor:
            _lim = dt * _pool / (_pool - _tent)
            if _lim < dt_max:
                dt_max = _lim
    if Ca_tot < 0.0:
        clip_Ca = -Ca_tot; Ca_tot = 0.0
    if Mg_tot < 0.0:
        clip_Mg = -Mg_tot; Mg_tot = 0.0
    if K_tot < 0.0:
        clip_K = -K_tot; K_tot = 0.0
    if Na_tot < 0.0:
        clip_Na = -Na_tot; Na_tot = 0.0
    if Al_tot < 0.0:
        clip_Al = -Al_tot; Al_tot = 0.0
    if Si_tot < 0.0:
        clip_Si = -Si_tot; Si_tot = 0.0
    if Fe_tot < 0.0:
        clip_Fe = -Fe_tot; Fe_tot = 0.0
    if An_tot < 0.0:
        clip_An = -An_tot; An_tot = 0.0
    if IC_tot < 0.0:
        clip_C = -IC_tot; IC_tot = 0.0

    # Alk_tot is DEFINED from the cation pools rather than integrated, so it is
    # formed after they are final. It is a charge balance and may legitimately
    # be negative; it is never clamped.
    Alk_tot = 2 * Mg_tot + 2 * Ca_tot + Na_tot + K_tot - An_tot     # [mol_c]
    # S3: under fe_mode="reduced" the released Fe stays Fe+2, and Fe+2 leaving a
    # silicate generates alkalinity exactly as Mg+2 does (P3's upper bound,
    # now a run). It is held aqueous -- not on the exchanger, not in the ionic
    # strength -- the tracer's assumption, so this is the bound, not a model of
    # Fe2+ chemistry. A separate statement: "tracer"/"oxic" are bit-identical.
    if p.fe_mode == "reduced":
        Alk_tot = Alk_tot + 2 * Fe_tot

    # implicit system.
    #
    # This closure used to assign its trial vector into the output arrays --
    # `Alk[i], CO2_w[i], ... = p` -- so the answer was stored as a side effect of
    # the LAST residual evaluation happening to be at the solution. That made the
    # residual non-reentrant, wrote 16 array slots on each of ~35 evaluations per
    # step, and would have silently stored a trial iterate under any solver that
    # does not finish by evaluating at its own answer (least_squares, F3's
    # planned fallback). It is a pure function now, and the solution is unpacked
    # explicitly below.
    # k_H is Henry's constant for CO2(g)/CO2(aq). CO2 is uncharged, so it is
    # the one constant here that activities leave alone.
    _args = (Alk_tot, p.n, p.Zr, now.s, IC_tot, k1, k2, now.k_H,
             k_w, p.CEC_tot, p.conv_Al, Al_tot, K1, K2, K3, K4,
             Mg_tot, Ca_tot, Na_tot, K_tot, K_Ca_Al, K_Ca_Mg, K_Ca_Na,
             K_Ca_K, K_Ca_H)

    # The reduced system is what is solved; the original is what the answer is
    # judged by. `equations` (16 unknowns) is kept as the reference definition of
    # the chemistry -- `tests/check_solver8.py` drives both from the same states
    # and compares -- and is also what produces `errors` below, so the gate,
    # `res_max` and `res_row` are still statements about the sixteen equations
    # the model is actually asserting.
    def equations(pv):
        return _biogeochem_equations_numba(pv, *_args)

    def equations8(qv):
        return _biogeochem_equations_8_numba(qv, *_args)

    def equations8log(yv):
        return _biogeochem_equations_8log_numba(yv, res_scale8, *_args)

    def jacobian8log(yv):
        return _jacobian_8log_numba(yv, res_scale8, *_args)

    def expand8(qv):
        return _expand_8_numba(qv, k1, k2, k_w, p.CEC_tot,
                               p.conv_Al, K1, K2, K3, K4, K_Ca_Al,
                               K_Ca_Mg, K_Ca_Na, K_Ca_K, K_Ca_H)

    # initial guess
    nZrs = p.n * p.Zr * now.s * 1000
    Alk0 = (Alk_tot - state.R_alk) / nZrs
    CO2_w0 = IC_tot / (p.n * p.Zr * 1000) * 1 / (
        now.s * (1 + k1 / state.H + k2 * k1 / (state.H ** 2))
        + (1 - now.s) / now.k_H)
    Al_w0 = (Al_tot - (state.f_Al / 3) * p.CEC_tot * p.conv_Al) / nZrs
    # R_alk0, Al0 and the eight starting fractions are gone with the equations
    # that defined them: the reduced system starts from the retained unknowns
    # only, and the previous guesses for the eliminated ones were never
    # independent of these in the first place.
    Mg0 = (Mg_tot - state.f_Mg / 2 * p.CEC_tot) / nZrs
    Na0 = (Na_tot - state.f_Na * p.CEC_tot) / nZrs
    Ca0 = (Ca_tot - state.f_Ca / 2 * p.CEC_tot) / nZrs
    K0 = (K_tot - state.f_K * p.CEC_tot) / nZrs
    H0 = state.H

    # solution 1
    #
    # full_output=True is not cosmetic. Without it fsolve's exit code is thrown
    # away, and MINPACK's exit code 5 -- "the iteration is not making good
    # progress", i.e. the solver stopped because it was stuck rather than
    # because it was done -- fires on 3.6 % of steps in Example and 7.9 % in the
    # synthetic case. SMEW has never seen one. scipy does emit a RuntimeWarning
    # in that case, ~1900 of them on a one-year run, but only when full_output
    # is False; requesting the code replaces a stream of warnings with a series
    # we can count. Here those abandoned iterates happen to sit on the answer
    # anyway (worst scaled residual measured anywhere is 1e-11), but nothing
    # guarantees that -- it is a property of this system at these conditions.
    q0 = np.array([CO2_w0, H0, Al_w0, Mg0, Ca0, Na0, K0, state.f_Ca])
    res_scale = _residual_scale(Alk_tot, IC_tot, p.CEC_tot, Al_tot, Mg_tot,
                                Ca_tot, Na_tot, K_tot, nZrs)
    # The eight retained rows of F3.2's sixteen scales, so there is one scale
    # vector in the model and not two that can drift apart.
    res_scale8 = res_scale[[0, 1, 4, 6, 7, 8, 9, 15]]

    # nfev counts evaluations of the EIGHT-equation residual, and from F3.5 on
    # the Jacobian is analytic, so it no longer contains hidden Jacobian builds.
    # njev counts those separately. Neither is comparable with F3.1's numbers.
    sol_r = root(equations8log, _y_from_q_numba(q0), jac=jacobian8log,
                 method="hybr", options={"xtol": 1e-12})
    q = _q_from_y_numba(sol_r.x)
    ier = sol_r.status
    sol = expand8(q)
    errors = np.asarray(equations(sol))                    # residuals, all 16
    nfev = sol_r.nfev
    njev = sol_r.njev

    # --- the ladder ----------------------------------------------------------
    #
    # F3.1 measured the trigger that was missing, and F3.2 could not yet use it:
    # MINPACK's own exit code. The gate was the residual alone, so a solver that
    # stopped because it was stuck -- 11 625 times across the suite before F3.4 --
    # was accepted in silence provided the iterate it abandoned happened to sit
    # near the answer. Both conditions route to the next rung now.
    #
    # Rung 2 had never fired once in the model's history, before or after F3.5,
    # because the only thing that could trigger it was an absolute residual
    # nothing came within ten orders of magnitude of.

    def _resid_bad(e):
        return bool(np.any(abs(e) / res_scale > RES_RTOL))

    def _bad(e, code):
        return code not in _IER_OK or _resid_bad(e)

    rung = 1
    if _bad(errors, ier):
        # Rung 2 -- the alternative pH guess, from the carbonate system alone.
        #
        # Solved HERE and not once per timestep. Until F3.6 `H0_2` cost a full
        # 1-D fsolve on every step -- 52 559 a year on Example, ~5 % of the
        # run -- to seed a rung that had never been taken. Since P8 it is not a
        # 1-D solve at all: the same cubic the rainwater uses, done directly.
        H0_2 = float(_H_charge_balance(Alk0, k1, k2, CO2_w0, k_w))

        # Same xtol as rung 1. The rung differs in its STARTING POINT, which
        # is the whole idea of it; the 1e-14 it inherited made it differ in
        # tolerance too, and that tolerance is below what a double can deliver
        # as a relative step -- measured, it made MINPACK return code 3 on
        # 4 017 of 4 319 calls while the residuals were 1.5e-15.
        q0 = np.array([CO2_w0, H0_2, Al_w0, Mg0, Ca0, Na0, K0, state.f_Ca])
        sol_r = root(equations8log, _y_from_q_numba(q0), jac=jacobian8log,
                     method="hybr", options={"xtol": 1e-12})
        q = _q_from_y_numba(sol_r.x)
        ier = sol_r.status
        sol = expand8(q)
        errors = np.asarray(equations(sol))
        nfev += sol_r.nfev
        njev += sol_r.njev
        rung = 2

    if _bad(errors, ier):
        # Rung 3 -- least squares with bounds, which cannot fail to return
        # something, and minimises the residual rather than requiring a root.
        #
        # The bounds are nearly free in solver variables: six of the eight are
        # logs, so positivity is already structural and their bounds are
        # infinite. Only two are real constraints -- Al_w is carried linearly
        # (see F3.5) and is a concentration, so it is bounded below by zero, not
        # by a floor; and f_Ca is an equivalent fraction, so ln f_Ca <= 0.
        lo = np.array([-np.inf, -np.inf, 0.0, -np.inf, -np.inf, -np.inf,
                       -np.inf, -np.inf])
        hi = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                       np.inf, 0.0])
        # A starting point outside its own bounds is a starting point, not a
        # result: clipping it decides nothing about the answer. Al_w0 is the one
        # that needs it -- it is negative on 10 of the suite's 1.36 M solves,
        # always on a feedstock with no aluminium in it.
        y0 = np.clip(_y_from_q_numba(q0), lo, hi)
        ls = least_squares(equations8log, y0, jac=jacobian8log, bounds=(lo, hi),
                           method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14)
        nfev += ls.nfev
        njev += ls.njev

        # least_squares minimises a SUM of squares, so it is free to trade one
        # row off against the rest -- and the row it gives up is whichever
        # contributes least, which on an Al-free feedstock is equation 5. Over
        # 400 cold starts it leaves that row at 6.9e-08 (median AND worst, so it
        # is systematic, not occasional) while every other row sits at machine
        # precision. x_scale='jac' changes nothing. A single Newton solve from
        # its answer does: 4.8e-16 median, 1.1e-15 worst.
        #
        # So rung 3 is two moves, not one -- least_squares to find the basin,
        # Newton to land in it. The cost is irrelevant: this rung does not run.
        pol = root(equations8log, ls.x, jac=jacobian8log, method="hybr",
                   options={"xtol": 1e-12})
        nfev += pol.nfev
        njev += pol.njev

        # Take the best of the three, scored the same way the gate scores them.
        # least_squares returns its best point whether or not that point is a
        # root, so "it returned" is not evidence of anything.
        best = np.max(abs(errors) / res_scale)
        for _y in (ls.x, pol.x):
            _q = _q_from_y_numba(_y)
            _sol = expand8(_q)
            _e = np.asarray(equations(_sol))
            _s = np.max(abs(_e) / res_scale)
            if _s < best:
                best, q, sol, errors = _s, _q, _sol, _e
        rung = 3
        # From rung 3 on, `ier` is normalised: 1 if the answer met the gate and
        # 0 if it did not. least_squares' status codes do not share MINPACK's
        # convention, and `n_stall` counts ier != 1, so passing them through
        # would silently change what that number means.
        ier = 0 if _resid_bad(errors) else 1

    # Rung 4 is sub-stepping the interval, and it lives in step() -- F2 already
    # has that machinery and a second copy would be a second thing to get wrong.
    # Rung 5 is raising, also in step(), once sub-dividing has been tried. So
    # this returns the failure rather than raising on it: _step_once is one
    # Euler step and is not the place that decides an interval is hopeless.
    unconverged = bool(_bad(errors, ier))

    (Alk, CO2_w, H, R_alk, Al_w, Al, Mg, Ca, Na, K,
     f_Al, f_Mg, f_Na, f_K, f_H, f_Ca) = sol

    # pH and C
    #
    # F4.5: the reported pH is the ACTIVITY pH, -log10(a_H+) -- the quantity a
    # pH electrode measures, and therefore the one every Exp_Data overlay has
    # been compared against all along. The concentration pH is kept alongside
    # it rather than replaced, so a pre-F4 run stays reproducible and the
    # offset is inspectable rather than inferred.
    #
    # Measured across all seven cases the offset is +0.011 to +0.094 pH units,
    # median 0.042, largest on Vulkaneifel where the ionic strength is highest.
    # Small, always positive, and it moves every model-vs-experiment pH figure.
    #
    # gamma_H is exactly 1.0 with the flag off, so this is bit-identical there:
    # multiplying by 1.0 is exact in IEEE.
    pH_conc = -np.log10(H / p.conv_mol)                    # [-]
    pH = -np.log10(gamma_H * H / p.conv_mol)               # [-]
    CO2_air = CO2_w / now.k_H                              # [mol/l]
    HCO3 = k1 * CO2_w / H
    CO3 = k2 * k1 * CO2_w / (H ** 2)
    DIC = CO2_w + HCO3 + CO3

    # Al speciation
    den = (H ** 4 + H ** 3 * K1 + H ** 2 * K1 * K2 + H * K1 * K2 * K3
           + K1 * K2 * K3 * K4)
    AlOH = (H ** 3 * K1 / den) * Al_w                      # [mol/l]
    AlOH2 = (H ** 2 * K1 * K2 / den) * Al_w
    AlOH3 = (H * K1 * K2 * K3 / den) * Al_w
    AlOH4 = Al_w - (Al + AlOH + AlOH2 + AlOH3)

    # concentrations
    Si = Si_tot / nZrs                                     # [mol-conv/l]
    An = An_tot / nZrs                                     # [mol_c-conv/l]
    Fe = Fe_tot / nZrs                                     # [mol-conv/l], tracer

    # CO2 diff flux
    Fs = now.D / (p.Z_CO2) * (CO2_air - p.CO2_atm) * 1000  # [mol/d]

    # Carbonate minerals
    #
    # The dissolution branch of carb_W is proportional to the pool
    # (W = s*CaCO3*(1-Omega)/tau), so this is an exponential decay integrated
    # explicitly: once s*(1-Omega)*dt/tau exceeds 1 the pool overshoots through
    # zero. Same clamp, same recording.
    CaCO3 = state.CaCO3 - state.W_CaCO3 * dt               # [mol-conv]
    MgCO3 = state.MgCO3 - state.W_MgCO3 * dt
    clip_CaCO3 = clip_MgCO3 = 0.0
    if CaCO3 < 0.0:
        clip_CaCO3 = -CaCO3; CaCO3 = 0.0
    if MgCO3 < 0.0:
        clip_MgCO3 = -MgCO3; MgCO3 = 0.0

    # S1: the secondary solids, and the next step's rates from this solution
    M_sec = state.M_sec - state.W_sec * dt                 # [mol-conv]
    clip_sec = np.maximum(-M_sec, 0.0)
    M_sec = np.maximum(M_sec, 0.0)
    Omega_sec = W_sec = lim_sec = np.zeros(0)
    if p.n_sec:
        Omega_sec, W_sec, lim_sec = _secondary(
            p, now.T_K, now.s, now.L, now.T, gam, H, Ca, Mg, K, Na, Al,
            Al / Al_w if Al_w > 0 else 0.0,
            Si, M_sec, nZrs, Fe=Fe, mu=mu)

    # Carbonate weathering
    #
    # Outside the implicit solve the natural quantity is the ACTIVITY, so these
    # take activities directly rather than through an effective constant -- the
    # same correction, written where it reads as what it is. K_CaCO3 and
    # K_MgCO3 are solubility products and are already defined on activities.
    #
    # This is where F4 changes the model most. carb_W has NO clamp: its
    # precipitation branch is r*Zr*(1-Omega), linear and unbounded, and
    # Omega_CaCO3 sits close to 1 in exactly the cases that precipitate. The
    # gamma_Ca*gamma_CO3 factor measures 0.17-0.78 across the benchmark suite,
    # which is enough to move Omega from just above 1 to below it at a given
    # state. PHREEQC agrees on the direction independently: at Example step
    # 29861 SMEW reads Omega = 1.17 where PHREEQC reports SI_calcite = -0.16.
    #
    # But that is a post-hoc reading of frozen output, and coupled the effect is
    # much smaller: precipitating less leaves more Ca and CO3 in solution, which
    # pushes Omega back up. Measured (tests/golden/F4_shift.txt, % of steps with
    # W_CaCO3 < 0, gamma off -> on), precipitation shrinks but does not stop:
    # Example 42.0 -> 33.6, Vulkaneifel raw 28.2 -> 12.1, tePas olivine
    # 52.2 -> 46.0, Amann 98-99.6 essentially unchanged.
    Omega_CaCO3 = (gam[activity.SP_CA] * Ca) * (gam[activity.SP_CO3] * CO3) / now.K_CaCO3
    Omega_MgCO3 = (gam[activity.SP_MG] * Mg) * (gam[activity.SP_CO3] * CO3) / now.K_MgCO3
    [W_CaCO3, W_MgCO3] = smew.carb_W(CaCO3, MgCO3, Omega_CaCO3, Omega_MgCO3,
                                     now.s, p.Zr, p.r_CaCO3, p.r_MgCO3,
                                     p.tau_CaCO3, p.tau_MgCO3)

    # Silicate weathering
    M_min, M_rock, rock_f = state.M_min, state.M_rock, state.rock_f
    d, delta_d = state.d, state.delta_d
    lamb, SSA, psd, SA = state.lamb, state.SSA, state.psd, state.SA
    clip_M_min = np.zeros(p.number_min)
    clip_psd = 0.0
    Omega = np.zeros(p.number_min)
    Wr = np.zeros(p.number_min)
    EW = np.zeros(p.number_min)

    if p.has_rock:
        # saturation and weathering rate
        #
        # Al, not Al[0]: saturation must use the current aluminium, as every
        # other species here does. Caveat unchanged from F0 -- there is no Al
        # sink in the model, so dissolved Al accumulates and the solution sits
        # supersaturated with respect to gibbsite. Needs a solubility control.
        # Activities, not concentrations. Every Omega_sil branch is a
        # monomial in the species handed to it, and every KINEC rate law is a
        # power of H+ (basalt_glass, of H+ and Al+3), so multiplying the
        # ARGUMENTS by their activity coefficients converts all twenty-odd
        # branches at once and exactly. The alternative -- a hand-transcribed
        # table of per-mineral stoichiometry to build a correction factor from
        # -- would be ~130 exponents copied by hand for no gain, and a single
        # typo in it would be silent.
        #
        # Si is H4SiO4 and Fe is not modelled: both uncharged, gamma = 1.
        a_Ca = gam[activity.SP_CA] * Ca
        a_Mg = gam[activity.SP_MG] * Mg
        a_K = gam[activity.SP_K] * K
        a_Na = gam[activity.SP_NA] * Na
        a_H = gamma_H * H
        a_Al = gam[activity.SP_AL] * Al
        for j in range(0, p.number_min):
            Omega[j] = weathering_kinec.Omega_sil(
                p.mineral[j], a_Ca, a_Mg, a_K, a_Na, Si, a_H, a_Al,
                _fe2_for_omega(p, Fe, mu, now.T_K),
                p.K_sp[j], p.conv_mol, p.conv_Al)          # [-]
            Wr[j] = now.s * p.diss_f * weathering_kinec.mineral_weathering(
                p.mineral[j], now.T_K, Omega[j], a_H, a_Al,
                p.conv_mol, p.conv_Al)

        if post_application:
            # mineral fractions in rock
            M_min = state.M_min - state.EW * p.MM_min * dt         # [g]
            # This clamp is not new -- it has always been here. What is new is
            # that the amount is recorded. It is the mass the rock is credited
            # with losing but does not have, and F0 flagged it as crediting
            # cations to solution that never left the rock.
            clip_M_min = np.maximum(-M_min, 0.0)
            M_min = np.maximum(M_min, 0)
            M_rock = np.sum(M_min) + p.M_iner                      # [g]
            # rock_f stays zero, not held, when the rock is gone: that is what
            # the preallocated array did.
            rock_f = M_min / M_rock if M_rock > 0 else np.zeros(p.number_min)

            # diameter variation
            d_shrink = np.sum(state.rock_f * state.Wr * p.MM_min
                              / p.rho_rock) * dt                   # [m]
            d = state.d - 2 * d_shrink * state.lamb                # [m]
            # A class whose particles have shrunk to nothing. psd_evol's
            # `if d[k]>0` guard leaves such a class at psd = 0, and that cutoff
            # is the one place the geometric account of the rock can lose mass
            # the stoichiometric account keeps. So it is recorded, the same way
            # F2 records clip_M_min, instead of being assumed harmless.
            #
            # Measured, it is zero everywhere, and for a better reason than the
            # classes being small: psd scales as (d/d_0)**3, so by the time d
            # crosses zero the cube has already underflowed. Vulkaneifel is the
            # only case that exhausts any class -- 16 of its 100 -- and each one
            # arrives at the cutoff with d ~ 3e-15 m and psd already exactly 0.
            # The guard discards nothing; the mass left continuously. Keeping
            # the measurement is what makes that a fact rather than a hope.
            exhausted = d <= 0.0
            clip_psd = float(np.sum(state.psd[exhausted]
                                    * state.delta_d[exhausted]))   # [g]
            d = np.where(d < 0, 0.0, d)
            if p.n_blocks == 1:
                delta_d = np.insert(np.diff(d), 0, d[0])           # [m]
            else:
                delta_d = _block_widths(d, p.n_blocks)             # [m]
            [lamb, SSA, psd, SA] = smew.psd_evol(
                d, delta_d, state.d, state.delta_d, state.psd,
                p.n_d_cl, p.a, p.b, p.rho_rock)
        elif rock_now is not None:
            # at or before the application step the geometry is whatever was
            # written at this index, not something evolved from the last state
            (M_min, M_rock, rock_f, d, delta_d, lamb, SSA, psd, SA) = rock_now

        # weathering fluxes
        EW = Wr * SA * rock_f                                      # [mol/d]

    new = SoilState(
        pH=pH, pH_conc=pH_conc, H=H, f_H=f_H, Alk=Alk, Alk_tot=Alk_tot, An=An, An_tot=An_tot,
        CO2_w=CO2_w, CO2_air=CO2_air, HCO3=HCO3, CO3=CO3, DIC=DIC,
        IC_tot=IC_tot, Fs=Fs, ADV=ADV,
        Ca=Ca, Ca_tot=Ca_tot, f_Ca=f_Ca, Mg=Mg, Mg_tot=Mg_tot, f_Mg=f_Mg,
        K=K, K_tot=K_tot, f_K=f_K, Na=Na, Na_tot=Na_tot, f_Na=f_Na,
        Si=Si, Si_tot=Si_tot, Fe=Fe, Fe_tot=Fe_tot,
        Al=Al, AlOH=AlOH, AlOH2=AlOH2, AlOH3=AlOH3, AlOH4=AlOH4,
        Al_w=Al_w, Al_tot=Al_tot, f_Al=f_Al, R_alk=R_alk,
        mu=mu, gamma_H=gamma_H,
        CaCO3=CaCO3, MgCO3=MgCO3,
        Omega_CaCO3=Omega_CaCO3, Omega_MgCO3=Omega_MgCO3,
        W_CaCO3=W_CaCO3, W_MgCO3=W_MgCO3,
        M_rock=M_rock, SA=SA, M_min=M_min, rock_f=rock_f,
        EW=EW, Wr=Wr, Omega=Omega,
        d=d, delta_d=delta_d, lamb=lamb, SSA=SSA, psd=psd,
        UP_Ca=UP_Ca, UP_Mg=UP_Mg, UP_K=UP_K, UP_Si=UP_Si,
        clip_Ca=clip_Ca, clip_Mg=clip_Mg, clip_K=clip_K, clip_Na=clip_Na,
        clip_Al=clip_Al, clip_Si=clip_Si, clip_An=clip_An, clip_C=clip_C,
        clip_Fe=clip_Fe,
        clip_CaCO3=clip_CaCO3, clip_MgCO3=clip_MgCO3, clip_M_min=clip_M_min,
        clip_psd=clip_psd,
        dt_max=dt_max, n_substeps=1.0,
        M_sec=M_sec, W_sec=W_sec, Omega_sec=Omega_sec, clip_sec=clip_sec,
        lim_sec=lim_sec,
    )
    # Is that clipping meaningful, or is it roundoff on an empty pool?
    #
    # This distinction is not pedantry. A forsterite run carries no aluminium at
    # all, so Al_tot sits at zero plus float noise and goes very slightly
    # negative on occasional steps -- clip_Al ~ 1e-39. Treating that as a
    # positivity failure sent the sub-stepper through 2, 4, 8, 16, 32 -- 62
    # extra implicit solves -- to "fix" a denormal, and then reported failure.
    # So significance is judged against the largest pool in the system, the same
    # way smew.ledger's ACTIVITY_FLOOR refuses to gate a pool that is entirely
    # noise. Rock is compared against rock: it is grams, not moles, and summing
    # the two would be dimensionally meaningless.
    scale_mol = max(abs(state.Ca_tot), abs(state.Mg_tot), abs(state.K_tot),
                    abs(state.Na_tot), abs(state.Al_tot), abs(state.Si_tot),
                    abs(state.An_tot), abs(state.IC_tot), abs(state.CaCO3),
                    abs(state.MgCO3))
    clipped_mol = (clip_Ca + clip_Mg + clip_K + clip_Na + clip_Al + clip_Si
                   + clip_An + clip_C + clip_CaCO3 + clip_MgCO3)
    clipped_rock = float(np.sum(clip_M_min))
    significant = (clipped_mol > CLIP_REL * scale_mol
                   or clipped_rock > CLIP_REL * max(abs(state.M_rock), 0.0))
    return new, {"up_act": UP_act, "errors": errors, "rung": rung,
                 "res_scale": res_scale,
                 "ier": int(ier), "nfev": int(nfev), "njev": int(njev),
                 "unconverged": unconverged,
                 "clipped": clipped_mol + clipped_rock,
                 "significant": bool(significant)}


# The positivity limiter above is a last resort: it keeps the run alive and
# leaves an auditable record, but it destroys mass. Sub-stepping is the fix --
# the step was simply too long for the state it started from, so shorten it.
#
# CLIP_REL is what counts as a real clip rather than float noise, as a fraction
# of the largest pool in the system. 1e-12 is far above double-precision noise
# on a pool (~1e-16 relative) and far below any amount of mass worth arguing
# about.
# ---------------------------------------------------------------------------
# Residual scaling
# ---------------------------------------------------------------------------
#
# The sixteen residuals do not share units. Rows 1-10 are in mol or mol_c, rows
# 11-16 are dimensionless. The original gate compared all of them against one
# absolute number, 1e-1, which therefore meant something different on every row
# and on some rows nothing at all: on row 16 -- the statement that the exchanger
# fractions sum to one -- it accepted a 10 % violation of CEC closure, while on
# a row whose natural magnitude is ~1e7 mol_c it was never binding.
#
# The fix is to divide each residual by a characteristic scale for its own
# equation, so that one relative tolerance means the same thing everywhere. The
# scale chosen here is THE POOL THE EQUATION BALANCES, which makes the scaled
# residual read as "the fraction of this element's mass that is unaccounted
# for" -- the same normalisation smew.ledger uses, and a statement with a
# physical meaning rather than a numerical one.
#
# Every scale is a quantity known BEFORE the solve, never a function of the
# iterate. That keeps the scale fixed while fsolve runs, which is what makes it
# safe to fold into the residual itself later (F3.5).

SCALE_FLOOR = 1e-12     # see _residual_scale

# What counts as converged, as a fraction of the pool each equation balances.
#
# Chosen from the data, not from taste. The worst scaled residual over the whole
# benchmark suite -- 21 case-variants, 1.41 M solves -- is 3.66e-9, on Amann's
# fine_nocrop. 1e-7 clears that by 27x while still being six orders of magnitude
# tighter than the 1e-1 it replaces on the dimensionless rows, and about 1e14
# tighter on the mol rows, where the old gate was never binding at all.
#
# F3.6 tightened it from 1e-7 to 1e-9, which is the value F3.2 originally
# proposed and had to abandon. Two things had to change first. The residuals
# themselves: the suite's worst was 3.66e-09 under the 16-equation system, which
# 1e-9 would have tripped on three case-variants, and is 2.89e-13 now -- so 1e-9
# clears the measured worst by ~3 500x instead of failing it. And the
# consequence of tripping: it used to mean rung 2 and then a crash, so an
# over-tight gate turned a working run into a failure. It now means the next
# rung, and the last rung is the one that raises.
# MINPACK exit codes the ladder treats as "this rung is done". 1 is a clean
# convergence. 3 is "xtol is too small, no further improvement in x is
# possible" -- a statement about how far the SOLUTION VECTOR can still move in
# double precision, not about the residual, which is checked separately and on
# its own terms. Reading 3 as failure sends converged steps down the ladder:
# measured, it did so on 93 % of them.
#
# Everything else -- 2 (out of evaluations), 4 and 5 (not making progress) --
# is a real failure and escalates.
_IER_OK = (1, 3)

RES_RTOL = 1e-9


def _residual_scale(Alk_tot, IC_tot, CEC_tot, Al_tot, Mg_tot, Ca_tot, Na_tot,
                    K_tot, nZrs):
    """Characteristic magnitude of each of the sixteen equations.

    The floor is the part that is easy to get wrong. On a forsterite run there
    is no aluminium at all, so Al_tot sits at denormal noise -- median 5e-32 --
    and dividing row 5's residual by it produces a "relative residual" of 1e+261.
    That is the same trap F2 hit when sub-stepping fired on clip_Al ~ 1e-39, and
    the same one smew.ledger's ACTIVITY_FLOOR exists to avoid: any absolute
    threshold on a quantity that can legitimately be zero is a bug waiting to
    happen, and so is any DENOMINATOR.

    So each pool scale is floored against the largest pool in the system, at the
    same 1e-12 relative F2 uses for CLIP_REL. The families are mixed in that max
    (mol, mol_c and mol-conv_Al all appear), which is loose; it is defensible
    because the floor is a noise guard rather than a normalisation, and 1e-12 of
    the largest pool is far above float noise and far below any mass worth
    arguing about. Rows 3 and 6 are in concentration units, so they take the
    floored pool scale divided by the water volume rather than a floor of their
    own.
    """
    pools = (abs(Alk_tot), abs(IC_tot), CEC_tot, abs(Al_tot),
             abs(Mg_tot), abs(Ca_tot), abs(Na_tot), abs(K_tot))
    floor = SCALE_FLOOR * max(pools)
    Alk_s, IC_s, CEC_s, Al_s, Mg_s, Ca_s, Na_s, K_s = (
        max(x, floor) for x in pools)
    return np.array([
        Alk_s,          # 1  charge balance on the exchanger + solution  [mol_c]
        IC_s,           # 2  inorganic carbon                            [mol]
        Alk_s / nZrs,   # 3  alkalinity definition                       [mol/l]
        CEC_s,          # 4  R_alk definition                            [mol_c]
        Al_s,           # 5  aluminium balance                   [mol-conv_Al]
        Al_s / nZrs,    # 6  Al speciation                     [mol-conv_Al/l]
        Mg_s,           # 7  magnesium balance                          [mol]
        Ca_s,           # 8  calcium
        Na_s,           # 9  sodium
        K_s,            # 10 potassium
        1.0, 1.0, 1.0, 1.0, 1.0,   # 11-15 Gaines-Thomas rows, dimensionless
        1.0,                       # 16 sum of exchanger fractions = 1
    ])


CLIP_REL = 1e-12
MAX_SUBSTEPS = 32       # 2, 4, 8, 16, 32; beyond this, accept the clip


def _unconverged_msg(diag, dt, extra=""):
    """Say which equation failed and by how much, not just that something did.

    `res_max` and `res_row` are derived in biogeochem_balance rather than
    carried in diag, so they are rebuilt here from the two things that are:
    the sixteen residuals and the scale vector they are judged against.
    """
    scaled = np.abs(np.asarray(diag["errors"])) / diag["res_scale"]
    j = int(np.argmax(scaled))
    return ("Solver did not converge after rung %d: scaled residual %.3e on "
            "equation %d (tolerance %.0e), MINPACK exit code %d, dt = %g d%s"
            % (diag["rung"], scaled[j], j + 1, RES_RTOL, diag["ier"], dt, extra))


def _lerp_forcing(a, b, f):
    """Forcing part-way through an interval, for sub-stepping.

    Linear in every driver, so a single sub-step (f = 1) reproduces `b` exactly
    and the no-sub-stepping path is untouched. `I` is handled by the caller: it
    is a per-step DEPTH of infiltrating water, not a rate, so splitting an
    interval must divide it among the sub-steps rather than interpolate it.
    """
    if f >= 1.0:
        return b
    return StepForcing(**{k: getattr(a, k) + (getattr(b, k) - getattr(a, k)) * f
                          for k in _FORCING_FIELDS})


GAMMA_PICARD_RTOL = 1e-10

# --- P6: reading the measured pH ------------------------------------------
#
# Reconciling [H+] with gamma_H at t = 0 is a fixed point:
#     H = 10^-pH / gamma_H(mu(H))
# It is iterated rather than solved. Convergence is fast because the ionic
# strength is dominated by Ca, Mg, Na and K, which are inputs and do not move
# between passes -- but it is NOT immediate at low pH, where H is itself a
# significant part of mu, so the loop runs to a tolerance rather than to a
# fixed count.
_IC_PH_MAX_PASSES = 40
_IC_PH_RTOL = 1e-13

# What to subtract from a reported pH to get the pH of the SOIL SOLUTION at
# field moisture, which is the quantity the model means by pH.
#
#   "solution"  the caller has already done this, or is supplying a porewater
#               pH. Subtracts nothing, and is the default: it is what every run
#               before P6 assumed, so the default path does not move.
#
#   "CaCl2"     0.01 M CaCl2, the standard agronomic method. Subtracts nothing
#               either, but for a reason rather than by default. Aitken & Moody
#               (1991), Aust. J. Soil Res. 29, 483-491, measured pH in several
#               electrolytes against soil solution pH directly and found 0.01 M
#               CaCl2 the best approximation to it for CULTIVATED soils of
#               ionic strength above 0.018 M. That is the population this model
#               is applied to, and SMEW's own benchmark cases sit at ionic
#               strengths of 1e-2 to 1.6e-1. (For their whole set, including
#               uncultivated soils of lower ionic strength, 0.002 M CaCl2 was
#               closer -- so this mapping is right for the intended use, not
#               universally.)
#
#   "H2O"      soil:water suspension. The dilution lowers the ionic strength and
#               leaves exchangeable H+ and Al3+ on the exchanger, so the reading
#               comes out HIGHER than the soil solution: reported differences
#               against CaCl2 run 0.4-0.8 pH units, commonly quoted near 0.5.
#               0.5 is used here, and it is the crudest number in this file: the
#               real offset is not a constant, it shrinks as soil EC rises and
#               the relationship is sigmoidal rather than linear. Treat it as a
#               default to override per site, not as a property of soils.
_PH_TO_SOLUTION = {"solution": 0.0, "CaCl2": 0.0, "H2O": 0.5}


def step(state, params, now, prev, dt, post_application=True, rock_now=None,
         max_substeps=MAX_SUBSTEPS):
    """Advance the chemistry by one timestep, sub-stepping if it is too long.

        state   the state to advance FROM. Not necessarily the previous index:
                through a frost the model holds, then resumes from the last
                unfrozen state, and the caller passes that one.
        now     this timestep's forcing
        prev    the forcing at the step `state` belongs to. The explicit
                balances evaluate their loss terms at that step, not at this
                one, so both are needed.
        post_application
                whether the rock geometry evolves this step. False before and
                exactly at the application timestep, where the geometry is the
                applied one and `rock_now` supplies it.
        max_substeps
                give up sub-dividing beyond this and accept the clip. Set to 1
                to disable sub-stepping entirely and clamp instead.

    A full-length step is tried first, so when nothing is wrong -- every step of
    every benchmark case -- this is exactly _step_once and the result is
    bit-identical. Only if that step had to clip is the interval sub-divided,
    which is why F2 changes results only where the old scheme was already
    producing a negative pool.

    Returns (new_state, diagnostics).
    """
    new, diag = _step_once(state, params, now, prev, dt, post_application, rock_now)

    # --- F4.7: optionally iterate gamma to self-consistency ------------------
    #
    # Lagged gamma (params.gamma_picard == 0) is the default and it is what the
    # whole of F4 rests on: gamma constant within the solve is what keeps
    # F3.5's analytic Jacobian valid. Picard does NOT give that up -- gamma is
    # still constant within each inner solve, it is just a better constant --
    # so this is a free upgrade if the lag ever turns out to matter.
    #
    # Measured on Example, over a full year. One sweep is enough -- three
    # sweeps differ from one by 0.4 % of the gap between one and none, so the
    # iteration converges immediately -- and against the LAGGED answer it moves
    # the output series by:
    #
    #     key        p50        p99        max  | the activity effect itself
    #     pH      1.8e-06    2.2e-05    6.3e-03 |          1.9e-02
    #     Alk     1.5e-05    5.2e-04    6.5e-01 |          3.1e-01
    #     Ca      5.1e-05    1.3e-04    1.8e-02 |          6.8e-02
    #     CaCO3   1.2e-05    1.5e-04    1.6e-02 |          3.2e-01
    #
    # So the lag error is the same order as the dt discretisation error F2
    # documented (~1e-5) at the median, about 10x above it at p99, and THREE TO
    # FOUR ORDERS OF MAGNITUDE below the activity effect it is a part of. The
    # max column is a handful of drainage steps where mu moves >1 % in one
    # timestep -- 0.18 % of steps -- and it is transient.
    #
    # Against that, one sweep costs 35 % of the run (5.8 s -> 7.9 s per
    # simulated year). Lagged is the default on those numbers. This stays in
    # the tree rather than being deleted because the argument for lagging is a
    # MEASUREMENT, and measurements expire: a saltier application, a longer dt,
    # or F4.6's recalibration could each move it.
    if params.gamma_on and params.gamma_picard > 0 and not diag["unconverged"]:
        for _ in range(params.gamma_picard):
            gam_new = _gammas_for(new, params, now.T_K, now.k_w)
            if abs(gam_new[0] - new.mu) <= GAMMA_PICARD_RTOL * max(new.mu, 1e-30):
                break
            cand, cdiag = _step_once(state, params, now, prev, dt,
                                     post_application, rock_now, gam_in=gam_new)
            if cdiag["unconverged"]:
                break                        # keep the lagged answer, not a bad one
            new, diag = cand, cdiag

    if (not diag["significant"] and not diag["unconverged"]) or max_substeps <= 1:
        if diag["unconverged"]:
            raise ValueError(_unconverged_msg(diag, dt))
        return new, diag

    # Sub-stepping now has two triggers. F2's is a pool that went negative;
    # F3.6 adds a solve that did not converge, on the same reasoning -- a
    # shorter interval is a smaller change in the totals and therefore a
    # starting point closer to its own answer. Both use this one loop, and both
    # land in the ledger the same way, through n_substeps: smew.ledger already
    # declines to gate a run containing a sub-divided interval, and a
    # solver-triggered one must not quietly escape that.
    solver_triggered = diag["unconverged"]

    n = 2
    while n <= max_substeps:
        st = state
        acc = None
        ok = True
        for m in range(n):
            sub_prev = _lerp_forcing(prev, now, m / n)
            sub_now = _lerp_forcing(prev, now, (m + 1) / n)
            # the interval's infiltration is shared out, not repeated
            sub_now = replace(sub_now, I=now.I / n)
            st, d = _step_once(st, params, sub_now, sub_prev, dt / n,
                               post_application, rock_now)
            acc = d if acc is None else {
                "up_act": acc["up_act"],          # the first sub-step's, see below
                "errors": d["errors"],
                "rung": max(acc["rung"], d["rung"]),
                # Diagnostics over a sub-divided interval report the interval,
                # not the last sub-step: nfev is the total cost paid, and ier is
                # the worst code seen, so one stalled sub-step is not hidden by
                # the ones after it that went fine. (MINPACK codes are ordered
                # 1 = converged, everything else a different way of not
                # converging, so max() is "worst" only by convention -- but 1 is
                # the only value that means success, which is what matters.)
                "ier": max(acc["ier"], d["ier"]),
                "nfev": acc["nfev"] + d["nfev"],
                "njev": acc["njev"] + d["njev"],
                "res_scale": d["res_scale"],

                "clipped": acc["clipped"] + d["clipped"],
                "significant": False,
                "unconverged": False,
            }
            if d["significant"] or d["unconverged"]:
                ok = False
                break
        if ok:
            st.n_substeps = float(n)
            st.dt_max = new.dt_max
            acc["substeps"] = n
            # An interval that only converged because it was sub-divided is
            # rung 4, however easily each sub-step then went. Reporting the
            # sub-steps' own rung would hide the one fact worth recording.
            if solver_triggered:
                acc["rung"] = max(acc["rung"], 4)
            return st, acc
        n *= 2

    # Rung 5. Sub-dividing did not rescue it.
    #
    # For a clipped pool, keeping the clamped full step is the established
    # behaviour: it records what it destroyed and n_substeps == 0 marks it. For
    # a solve that will not converge there is nothing to keep -- the numbers are
    # not a solution to anything -- so this raises. TC_SMEW prints and carries
    # on at the end of its ladder, which is worse than raising, because it
    # produces output that looks like a result.
    if diag["unconverged"]:
        raise ValueError(_unconverged_msg(
            diag, dt, ", and sub-dividing it into %d did not help"
            % max_substeps))
    new.n_substeps = 0.0
    diag["substeps"] = 0
    return new, diag


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def biogeochem_balance(n, s, L, T, I, v, k_v, RAI, root_d, Zr, r_het, r_aut, D, temp_soil, pH_in, conc_in, f_CEC_in, K_CEC, CEC_tot, Si_in, CaCO3_in, MgCO3_in, M_rock_in, t_app, mineral, rock_f_in, d_in, psd_perc_in, SSA_in, diss_f, dt, conv_Al, conv_mol, keyword_add, *, max_substeps=MAX_SUBSTEPS, gamma_on=False, gamma_picard=0, mu_ref_CEC=activity.MU_REF_CEC, mg_phase="magnesite", pH_method="solution", reapply=None, ic_al_cap=None, ic_al_hold="totals", sec_phases=(), sec_A_seed=None, fe_mode="tracer"):
    '''Run the soil biogeochemistry over the whole forcing series.

    !!! Fe not modeled !!! -> set to zero to make the model run!!

    The 34-argument signature is unchanged and the returned series are unchanged.
    Internally this is now a loop over `step`, against one `SoilParams` resolved
    once and one `StepForcing` per timestep; see smew/state.py.

    mg_phase is keyword-only and selects which magnesium carbonate may
    precipitate. "magnesite" is the default and is bit-identical to every
    result produced before it existed. "nesquehonite" is the phase that
    actually forms in soil below ~100 C -- magnesite is kinetically inhibited
    there -- and is ~1.8 log units more soluble, so it precipitates far less
    readily. It is off by default deliberately: switch it on for a site where
    an Mg carbonate has actually been identified, not on the general argument.
    Note that r_MgCO3 and tau_MgCO3 were tuned under the magnesite assumption
    and are NOT adjusted with the phase.

    gamma_on is keyword-only and defaults to False, which reproduces every
    result produced before F4 BIT FOR BIT -- `activity.effective_constants` at
    gamma = 1 is an exact identity, not an approximation to one. True turns on
    ionic strength and activity coefficients (smew/activity.py): the eleven
    equilibrium constants of the implicit system are rescaled, and the
    saturation states and rate laws outside it take activities directly. Output
    series stay in concentrations; `mu` and `gamma_H` are added to the contract.

    max_substeps is keyword-only and defaults to the module setting, so the 34
    positional arguments are exactly as they were. Set it to 1 to disable
    sub-stepping and fall back to clamping, which is how the clipping rows in
    smew.ledger are exercised -- with sub-stepping on, no case in the suite ever
    reaches the clamp.

    `reapply = (t_app_re [d], M_rock_re [g/m2])` lands a second application of the
    same feedstock (same minerals, PSD and SSA) at t_app_re. None, the default, is the
    single application and is bit-identical to it. With it, the PSD output arrays
    have twice as many classes: block 1 is the first application, block 2 the second
    (zero until it lands). See SoilParams.n_blocks for what is pooled and what is not.

    `sec_phases` (S1) lists the secondary minerals allowed to form in the soil, by
    constants._SEC_PHASES name: "Al(OH)3(a)", "gibbsite", "SiO2(a)". () is the
    default and is bit-identical. Order matters -- phases sharing an element share
    its excess in this order, so put the more soluble first, e.g.
    ("Al(OH)3(a)", "gibbsite", "SiO2(a)"). `sec_A_seed = {name: m2 per m3 of soil}`
    overrides a phase's nucleation surface, the parameter the data cannot pin down
    and the one to sweep. See _secondary for the rate law and the limiter.

    The `D` argument is accepted and discarded. It has never been used: the CO2
    diffusivity is recomputed below from the moisture series before D is read,
    so the value smew.respiration returns and every notebook passes in has no
    effect. Left in place because removing it would change the signature, which
    F1 is not allowed to do; worth removing deliberately later.
    '''
    nt = len(s)
    p = _resolve_params(n, Zr, CEC_tot, k_v, RAI, root_d, K_CEC, mineral,
                        M_rock_in, t_app, d_in, dt, diss_f, conv_mol, conv_Al,
                        gamma_on=gamma_on, gamma_picard=gamma_picard,
                        mu_ref_CEC=mu_ref_CEC, mg_phase=mg_phase,
                        pH_method=pH_method, reapply=reapply,
                        ic_al_cap=ic_al_cap, ic_al_hold=ic_al_hold,
                        sec_phases=sec_phases, sec_A_seed=sec_A_seed,
                        fe_mode=fe_mode)

    # --- forcing derived from the inputs -------------------------------------
    T_K = temp_soil + 273.15

    # soil CO2 diffusivity. NOTE: this shadows the D argument (see docstring).
    D_0 = smew.D_0()                                       # free-air [m2/d]
    D = D_0 * (1 - s) ** (10 / 3) * n ** (4 / 3)           # Mill-Quirk (1961)

    # solute diffusivity in soil water
    Dw_0 = smew.Dw_0()
    Dw = Dw_0 * (n * s) ** 2      # Archie 1942, Grathwohl 1998 (book)

    # carbonate spec
    [k1, k2, k_w, k_H] = smew.K_C(T_K, conv_mol)
    # P5: calcite's solubility product varies with temperature like the rest of
    # the carbonate system. Built here, per step, for the same reason k1 is.
    [K_CaCO3_t, K_MgCO3_t, _, _, _, _] = smew.carb_weath_const(conv_mol, T_K,
                                                              mg_phase)
    # P5b: the aluminium ladder, likewise.
    [K1_t, K2_t, K3_t, K4_t] = smew.K_Al(conv_mol, T_K)

    # --- rainwater ------------------------------------------------------------
    Alk_rain = 0                                           # alk
    CO2_w_rain = k_H * p.CO2_atm                           # [mol/l] Henry's law
    # P8: one vectorised solve for the whole time vector, where this used to be
    # `nt` separate scipy calls -- the single largest avoidable cost in the run.
    H_rain = _H_charge_balance(Alk_rain, k1, k2, CO2_w_rain, k_w)  # [mol/l]
    DIC_rain = (CO2_w_rain + k1 * CO2_w_rain / H_rain
                + k2 * k1 * CO2_w_rain / (H_rain ** 2))

    forcing = _forcing(s, v, I, L, T, Dw, D, r_het, r_aut, temp_soil, T_K,
                       k1, k2, k_w, k_H, K_CaCO3_t, K_MgCO3_t,
                       K1_t, K2_t, K3_t, K4_t, DIC_rain)

    # --- storage --------------------------------------------------------------
    out = {k: np.zeros(nt) for k in STATE_1D}
    for k in STATE_MIN:
        out[k] = np.zeros([p.number_min, nt])
    for k in STATE_PSD:
        out[k] = np.zeros([p.n_d_cl, nt])
    for k in STATE_SEC:
        out[k] = np.zeros([p.n_sec, nt])
    for k in ("UP_Ca", "UP_Mg", "UP_K", "UP_Si"):
        out[k] = np.zeros(nt)
    errors = np.zeros([16, nt])

    # Solver diagnostics, one entry per timestep.
    #
    # NaN rather than zero where no solve happened. Through a frost the model
    # holds the chemistry rather than integrating it, so those indices never
    # reach the solver at all -- and 0 is a meaningful MINPACK exit code, so
    # zero-filling would make "no solve" indistinguishable from "improper input
    # parameters". `errors` keeps its zeros, which is what that array has always
    # held at frozen indices.
    solver = {k: np.full(nt, np.nan)
              for k in ("solver_ier", "solver_nfev", "solver_njev",
                        "solver_rung",
                        "res_max", "res_row")}

    # --- initial conditions ---------------------------------------------------
    state, p, applied = _initial_state(
        p, forcing[0], pH_in, conc_in, f_CEC_in, Si_in, CaCO3_in, MgCO3_in,
        M_rock_in, rock_f_in, d_in, psd_perc_in, SSA_in, t_app, s, T, L,
        keyword_add)
    state.write_into(out, 0)

    # A second application: its block geometry needs `a`, refined just above.
    if p.n_blocks > 1:
        p = replace(p, rock_re=_fresh_block(p, reapply[1], rock_f_in, d_in, psd_perc_in))
    reapplied = p.n_blocks == 1

    # The rock lands at index tt_app, which is not necessarily 0.
    if applied is not None:
        ta = p.tt_app
        (out["M_min"][:, ta], out["M_rock"][ta], out["rock_f"][:, ta],
         out["d"][:, ta], out["delta_d"][:, ta], out["lamb"][:, ta],
         out["SSA"][:, ta], out["psd"][:, ta], out["SA"][ta]) = applied

    # --- the time loop --------------------------------------------------------
    # Freezing is handled here rather than inside step(): below zero the model
    # holds, leaving this index at its preallocated zeros, and on the thaw it
    # resumes from the last unfrozen state. That makes "the previous step" an
    # arbitrary earlier index, which is a property of the schedule, not of the
    # chemistry -- so the loop picks the state and step() just advances it.
    #
    # fallback for a run that STARTS frozen: idx_before_freezing is only assigned
    # at a thaw-to-freeze transition, but is read at every freeze-to-thaw one, so
    # with no unfrozen step before the first frost it was read unbound
    # (UnboundLocalError). 1 -> last = 0, i.e. resume from the initial conditions.
    idx_before_freezing = 1
    prev_state, prev_idx = state, 0

    for i in range(1, nt):

        last = i - 1

        if temp_soil[i] > 0:

            # Modified by ZeroEx to shut down model at below zero temperatures
            # and continue at last value above zero
            if i < nt - 1:
                if temp_soil[i + 1] < 0:   # if the next value is below zero
                    idx_before_freezing = i  # remember index to continue from

            if temp_soil[i - 1] < 0:       # If last value was during freezing
                last = idx_before_freezing - 1   # use last index before freezing
            else:
                last = i - 1

            # The common case is last == i-1, where the state is already in hand.
            # Only a thaw needs it read back out of the arrays.
            st = prev_state if last == prev_idx else SoilState.from_arrays(out, last)

            # Second application: added to the state this step advances from, and
            # written back at that index so the outputs (and the ledger, which reads
            # EW there) show it. At the first unfrozen step on or after tt_app_re.
            if not reapplied and i >= p.tt_app_re and last == i - 1:
                st, p = _reapply(st, p)
                st.write_into(out, last)
                reapplied = True

            post = i > p.tt_app
            rock_now = None
            if p.has_rock and not post:
                rock_now = (out["M_min"][:, i].copy(), out["M_rock"][i],
                            out["rock_f"][:, i].copy(), out["d"][:, i].copy(),
                            out["delta_d"][:, i].copy(), out["lamb"][:, i].copy(),
                            out["SSA"][:, i].copy(), out["psd"][:, i].copy(),
                            out["SA"][i])

            new, diag = step(st, p, forcing[i], forcing[last], dt, post, rock_now,
                             max_substeps=max_substeps)
            new.write_into(out, i)

            # Uptake is stored at `last`, not at i. That is what the model has
            # always done (and it means the returned UP_* series is overwritten
            # at pre-freeze indices and cannot be integrated -- smew.ledger
            # recomputes up_act instead). Preserved deliberately: changing it
            # would change the output, which is F2's call, not F1's.
            (out["UP_Ca"][last], out["UP_Mg"][last],
             out["UP_K"][last], out["UP_Si"][last]) = diag["up_act"]
            errors[:, i] = diag["errors"]
            solver["solver_ier"][i] = diag["ier"]
            solver["solver_nfev"][i] = diag["nfev"]
            solver["solver_njev"][i] = diag["njev"]
            solver["solver_rung"][i] = diag["rung"]
            # Largest SCALED residual and the row carrying it. Scaled is
            # what makes these two comparable across rows and across cases:
            # res_max reads as the fraction of a pool left unaccounted for by
            # the worst equation, and res_row says which equation that was.
            _a = np.abs(diag["errors"]) / diag["res_scale"]
            _j = int(np.argmax(_a))
            solver["res_max"][i] = _a[_j]
            solver["res_row"][i] = _j + 1          # 1-based, as the note numbers them

            prev_state, prev_idx = new, i

    _solved = np.isfinite(solver["solver_ier"])
    n_solve = int(_solved.sum())
    n_stall = int((solver["solver_ier"][_solved] != 1).sum())
    nfev_total = float(np.nansum(solver["solver_nfev"]))
    njev_total = float(np.nansum(solver["solver_njev"]))

    # The output contract, stated rather than dumped.
    #
    # This used to be `{k: v for k, v in locals().items()}`, which returned 192
    # names: 34 arguments, the real state, loop temporaries, and two fsolve
    # closures. Nothing could be renamed safely because everything was public,
    # and nothing was documented because nothing was chosen.
    #
    # These 76 are what the notebooks, the ZeroEx wrapper and smew.ledger
    # actually read -- see smew.harness.CONTRACT, which is the same list and is
    # gated by harness.verify_contract(). Adding a process means adding its
    # names here AND there, deliberately.
    return {
        # --- acid-base, carbonate system, CO2 fluxes ---
        "pH": out["pH"], "pH_conc": out["pH_conc"],
        "H": out["H"], "f_H": out["f_H"], "Alk": out["Alk"],
        "Alk_tot": out["Alk_tot"], "An": out["An"], "An_tot": out["An_tot"],
        "CO2_w": out["CO2_w"], "CO2_air": out["CO2_air"], "HCO3": out["HCO3"],
        "CO3": out["CO3"], "DIC": out["DIC"], "IC_tot": out["IC_tot"],
        "Fs": out["Fs"], "ADV": out["ADV"], "DIC_rain": DIC_rain,
        # --- cations ---
        "Ca": out["Ca"], "Ca_tot": out["Ca_tot"], "f_Ca": out["f_Ca"],
        "UP_Ca": out["UP_Ca"],
        "Mg": out["Mg"], "Mg_tot": out["Mg_tot"], "f_Mg": out["f_Mg"],
        "UP_Mg": out["UP_Mg"],
        "K": out["K"], "K_tot": out["K_tot"], "f_K": out["f_K"],
        "Na": out["Na"], "Na_tot": out["Na_tot"], "f_Na": out["f_Na"],
        "Si": out["Si"], "Si_tot": out["Si_tot"], "UP_Si": out["UP_Si"],
        # --- iron: a mass tracer, not a modelled species ---
        "Fe": out["Fe"], "Fe_tot": out["Fe_tot"],
        # --- aluminium ---
        "Al": out["Al"], "AlOH": out["AlOH"], "AlOH2": out["AlOH2"],
        "AlOH3": out["AlOH3"], "AlOH4": out["AlOH4"], "Al_w": out["Al_w"],
        "Al_tot": out["Al_tot"], "f_Al": out["f_Al"],
        # S1.0: Al the initial-condition cap took out at t = 0 (0 when off).
        # Outside the output contract on purpose: a scalar about the input, not
        # a series, and the golden masters compare the contract.
        "Al_ic_removed": p.ic_al_removed,
        # the K_Ca_Al the run used (thermodynamic when gamma_on); differs from
        # the input only under ic_al_hold="K_Ca_Al"
        "K_Ca_Al_used": p.K_Ca_Al,
        # --- S1: secondary minerals, (n_sec, nt); empty when sec_phases=() ---
        # Outside the output contract for now, as Al_ic_removed is: the golden
        # masters compare the contract, and a phase-off run must stay
        # bit-identical in it. W_sec < 0 is precipitation, applied over the
        # step AFTER the index it is stored at (the W_CaCO3 convention).
        "sec_phases": p.sec_phases, "sec_st": p.sec_st, "fe_mode": p.fe_mode,
        "M_sec": out["M_sec"], "W_sec": out["W_sec"],
        "Omega_sec": out["Omega_sec"], "clip_sec": out["clip_sec"],
        "lim_sec": out["lim_sec"],
        # --- F4 activities ---
        # mu is ionic strength [mol/l]; gamma_H is the H+ activity coefficient,
        # which is exactly the activity-vs-concentration pH offset. Every other
        # gamma is a pure function of (mu, temp_soil): activity.gammas(mu, T_K).
        # Zero and one respectively on a gamma_on=False run.
        "mu": out["mu"], "gamma_H": out["gamma_H"],
        # --- solids and their dissolution ---
        "M_rock": out["M_rock"], "CaCO3": out["CaCO3"], "MgCO3": out["MgCO3"],
        "W_CaCO3": out["W_CaCO3"], "W_MgCO3": out["W_MgCO3"],
        "EW": out["EW"], "Wr": out["Wr"], "Omega": out["Omega"],
        "M_min": out["M_min"],
        # --- forcing carried through, so a result is self-contained ---
        "s": s, "temp_soil": temp_soil, "v": v, "I": I, "L": L, "T": T,
        "Dw": Dw, "r_het": r_het, "r_aut": r_aut,
        # --- fixed-shape arrays ---
        "min_st": p.min_st, "xi": p.xi,
        # --- background solute input (zero when keyword_add == 0) ---
        "I_Ca": p.I_Ca, "I_Mg": p.I_Mg, "I_K": p.I_K, "I_Na": p.I_Na,
        "I_Si": p.I_Si, "I_An": p.I_An,
        # --- geometry, units, vegetation constants ---
        "n": n, "Zr": Zr, "dt": dt, "conv_mol": conv_mol, "conv_Al": conv_Al,
        "k_v": k_v, "RAI": RAI, "root_d": root_d,
        # --- what was applied ---
        "mineral": mineral,
        # --- F2 positivity bookkeeping ---
        # Mass the explicit update could not remove without driving a pool
        # negative. Zero on every step of every benchmark case; non-zero means
        # conservation was broken here, deliberately and visibly, and
        # smew.ledger carries each of these as its own row.
        "clip_Ca": out["clip_Ca"], "clip_Mg": out["clip_Mg"],
        "clip_K": out["clip_K"], "clip_Na": out["clip_Na"],
        "clip_Al": out["clip_Al"], "clip_Si": out["clip_Si"],
        "clip_An": out["clip_An"], "clip_C": out["clip_C"],
        "clip_Fe": out["clip_Fe"],
        "clip_CaCO3": out["clip_CaCO3"], "clip_MgCO3": out["clip_MgCO3"],
        "clip_M_min": out["clip_M_min"],
        # clip_psd is the geometric twin of clip_M_min [g]: mass dropped from
        # the particle size distribution because a diameter class shrank to
        # zero. It is NOT a ledger row -- the ledger balances moles of elements,
        # and the stoichiometric rock (M_rock) is what feeds it. This is the
        # surface-area account's own loss, and the only way the two accounts of
        # the rock can drift apart. See tests/check_psd.py.
        "clip_psd": out["clip_psd"],
        # dt_max: the largest step that would have kept every pool
        # non-negative [d]. n_substeps: how many sub-intervals the step needed
        # (1 = none, 0 = sub-dividing failed and the clamp was used).
        "dt_max": out["dt_max"], "n_substeps": out["n_substeps"],
        # --- F3.1 solver diagnostics ---
        # errors is [16, nt]: the residual of every equation at every timestep,
        # which the model has always computed and always discarded. The rest are
        # per-timestep, NaN where a frost meant no solve happened:
        #   solver_ier   MINPACK exit code. 1 = converged; 5 = "not making good
        #                progress", the solver giving up. Anything but 1 is a
        #                step whose answer nothing has checked.
        #   solver_nfev  residual evaluations spent, summed over sub-steps.
        #   solver_rung  which fallback rung produced the answer.
        #   res_max      largest |residual| and res_row the equation carrying it.
        "errors": errors,
        # Run-level solver totals, computed on the FULL record before anything
        # thins it. The per-step series above are decimated by harness.collect
        # and go NaN at frozen indices, so neither the stall count nor the cost
        # is recoverable from them; these three are. n_stall is the headline
        # number -- steps whose answer the solver itself declined to vouch for.
        "n_solve": float(n_solve), "n_stall": float(n_stall),
        "nfev_total": float(nfev_total), "njev_total": float(njev_total),
        "solver_ier": solver["solver_ier"],
        "solver_nfev": solver["solver_nfev"],
        "solver_rung": solver["solver_rung"],
        "res_max": solver["res_max"],
        "res_row": solver["res_row"],
    }
