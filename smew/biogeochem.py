# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:34:44 2019
"""

from dataclasses import fields, replace

import numpy as np
import smew
from smew import weathering_kinec
from smew.state import (SoilParams, SoilState, StepForcing,
                        STATE_1D, STATE_MIN, STATE_PSD)
from numba import njit
from scipy.optimize import fsolve, root
#minimize, least_squares, newton_krylov, broyden1, root, broyden2

@njit
def _equations_water_numba(p, Alk_rain, k1, k2, CO2_w_rain, k_w):
    H_rain = p[0]
    return Alk_rain-(k1*CO2_w_rain/H_rain+2*k1*k2*CO2_w_rain/(H_rain**2)-H_rain+k_w/H_rain)

@njit
def _eqH_numba(p, k1, k2, CO2_w0, k_w, Alk0):
    H0 = p[0]
    return (k1*CO2_w0/H0+2*k1*k2*CO2_w0/(H0**2)-H0+k_w/H0)-Alk0

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
                    M_rock_in, t_app, d_in, dt, diss_f, conv_mol, conv_Al):
    """Every constant the run needs, looked up once.

    Previously these were ~30 locals resolved inline in the middle of
    biogeochem_balance, which meant a depth- or restart-capable model would have
    re-derived them per column or per restart. They do not depend on state.
    """
    has_rock = M_rock_in > 0
    number_min = len(mineral) if has_rock else 1

    MM_min = np.zeros(number_min)
    min_st = np.zeros([number_min, 6])
    K_sp = np.zeros(number_min)
    if has_rock:
        for j in range(0, number_min):
            MM_min[j], min_st[j, :] = smew.min_const(mineral[j], conv_mol)

    # [g/mol-conv]: Molar masses
    [MM_Mg, MM_Ca, MM_Na, MM_K, MM_Si, MM_C, MM_Anions, MM_Al] = smew.MM(conv_mol)

    # Aluminium speciation
    [K1, K2, K3, K4] = smew.K_Al(conv_mol)

    # CEC Gaines-Thomas constants
    [K_Ca_Mg, K_Ca_K, K_Ca_Na, K_Ca_Al, K_Ca_H] = K_CEC

    # nutrient uptake by plants
    [v_f_Ca, v_f_Mg, v_f_K, v_f_Si] = smew.plant_nutr_f()
    dry_perc = 0.1  # percent of dry mass
    xi = dry_perc * np.array([v_f_Ca / MM_Ca, v_f_Mg / MM_Mg,
                              v_f_K / MM_K, v_f_Si / MM_Si])  # [mol-conv/g_biomass]

    # carb weathering constants
    [K_CaCO3, K_MgCO3, r_CaCO3, r_MgCO3, tau_CaCO3, tau_MgCO3] = \
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
        K1=K1, K2=K2, K3=K3, K4=K4,
        K_Ca_Mg=K_Ca_Mg, K_Ca_K=K_Ca_K, K_Ca_Na=K_Ca_Na,
        K_Ca_Al=K_Ca_Al, K_Ca_H=K_Ca_H,
        K_CaCO3=K_CaCO3, K_MgCO3=K_MgCO3, r_CaCO3=r_CaCO3, r_MgCO3=r_MgCO3,
        tau_CaCO3=tau_CaCO3, tau_MgCO3=tau_MgCO3,
        mineral=tuple(mineral), number_min=number_min, MM_min=MM_min,
        min_st=min_st, K_sp=K_sp, diss_f=diss_f,
        rho_rock=3 * 1e6, a=a, b=b, M_iner=0.0,
        n_d_cl=len(d_in) if has_rock else 1,
        tt_app=int(t_app / dt) if has_rock else 0,
        has_rock=has_rock,
    )


def _forcing(s, v, I, L, T, Dw, D, r_het, r_aut, temp_soil, T_K,
             k1, k2, k_w, k_H, DIC_rain):
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
                        DIC_rain=DIC_rain[i])
            for i in range(len(s))]


_FORCING_FIELDS = tuple(f.name for f in fields(StepForcing))


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

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
    # pH
    pH = pH_in
    H = 10 ** (-pH) * p.conv_mol

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

    # anions (mol_c/l)
    An = 2 * Mg + 2 * Ca + Na + K - Alk                    # [mol_c/l]
    if An < 0:
        print(An)
        raise ValueError("Not enough cations for this alkalinity")

    # aluminium speciation
    K1, K2, K3, K4 = p.K1, p.K2, p.K3, p.K4
    den = H ** 4 + H ** 3 * K1 + H ** 2 * K1 * K2 + H * K1 * K2 * K3 + K1 * K2 * K3 * K4
    Al = (H ** 4 / den) * Al_w                             # mol/l
    AlOH = (H ** 3 * K1 / den) * Al_w
    AlOH2 = (H ** 2 * K1 * K2 / den) * Al_w
    AlOH3 = (H * K1 * K2 * K3 / den) * Al_w
    AlOH4 = Al_w - (Al + AlOH + AlOH2 + AlOH3)

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
    [f_Ca, f_Mg, f_K, f_Na, f_Al, f_H] = f_CEC_in

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

    # Carbonate weathering
    Omega_CaCO3 = Ca * CO3 / p.K_CaCO3                     # [-]
    Omega_MgCO3 = Mg * CO3 / p.K_MgCO3
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

        # diameter classes
        d_a = np.asarray(d_in, dtype=float).copy()         # [m]
        delta_d_a = np.insert(np.diff(d_a), 0, d_a[0])

        # particle size distribution
        psd_a = psd_perc_in * M_rock_a / delta_d_a         # [g/m]

        # refinement of fractal constant based on measured SSA
        a = p.a
        if SSA_in > 0:
            a = (SSA_in * p.rho_rock * M_rock_a / 6) / np.sum(
                d_a ** (p.b - 1) * psd_a * delta_d_a)      # [m**-b]

        # surface area
        lamb_a = a * d_a ** p.b                            # [-]
        SSA_a = 6 / (d_a * p.rho_rock) * lamb_a            # [m2/g]
        SA_a = np.sum(SSA_a * psd_a * delta_d_a)           # [m2]

        p = replace(p, a=a, M_iner=M_iner)
        applied = (M_min_a, M_rock_a, rock_f_a, d_a, delta_d_a,
                   lamb_a, SSA_a, psd_a, SA_a)

        # mineral weathering. Only when the rock goes on at t = 0; otherwise
        # index 0 keeps the zeros allocated above, as it always has.
        if t_app == 0:
            M_min, M_rock, rock_f = M_min_a, M_rock_a, rock_f_a
            d, delta_d, lamb, SSA, psd, SA = \
                d_a, delta_d_a, lamb_a, SSA_a, psd_a, SA_a
            for j in range(0, p.number_min):
                Omega[j] = weathering_kinec.Omega_sil(
                    p.mineral[j], Ca, Mg, K, Na, Si, H, Al, p.Fe, p.K_sp[j],
                    p.conv_mol, p.conv_Al)                 # [-]
                Wr[j] = f0.s * p.diss_f * weathering_kinec.mineral_weathering(
                    p.mineral[j], f0.T_K, Omega[j], H, Al, p.conv_mol, p.conv_Al)
                EW[j] = Wr[j] * SA * rock_f[j]             # [mol-conv/d]

    p = replace(p, I_Ca=I_Ca, I_Mg=I_Mg, I_K=I_K, I_Na=I_Na, I_Si=I_Si, I_An=I_An)

    state = SoilState(
        pH=pH, H=H, f_H=f_H, Alk=Alk, Alk_tot=Alk_tot, An=An, An_tot=An_tot,
        CO2_w=CO2_w, CO2_air=CO2_air, HCO3=HCO3, CO3=CO3, DIC=DIC,
        IC_tot=IC_tot, Fs=Fs, ADV=0.0,
        Ca=Ca, Ca_tot=Ca_tot, f_Ca=f_Ca, Mg=Mg, Mg_tot=Mg_tot, f_Mg=f_Mg,
        K=K, K_tot=K_tot, f_K=f_K, Na=Na, Na_tot=Na_tot, f_Na=f_Na,
        Si=Si, Si_tot=Si_tot,
        Al=Al, AlOH=AlOH, AlOH2=AlOH2, AlOH3=AlOH3, AlOH4=AlOH4,
        Al_w=Al_w, Al_tot=Al_tot, f_Al=f_Al, R_alk=R_alk,
        CaCO3=CaCO3, MgCO3=MgCO3,
        Omega_CaCO3=Omega_CaCO3, Omega_MgCO3=Omega_MgCO3,
        W_CaCO3=W_CaCO3, W_MgCO3=W_MgCO3,
        M_rock=M_rock, SA=SA, M_min=M_min, rock_f=rock_f,
        EW=EW, Wr=Wr, Omega=Omega,
        d=d, delta_d=delta_d, lamb=lamb, SSA=SSA, psd=psd,
        clip_M_min=np.zeros(p.number_min),
    )
    return state, p, applied


# ---------------------------------------------------------------------------
# One timestep
# ---------------------------------------------------------------------------

def _step_once(state, params, now, prev, dt, post_application=True, rock_now=None):
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

    Returns (new_state, diagnostics). diagnostics carries the uptake fluxes and
    the solver residuals; see biogeochem_balance for why uptake is not simply
    part of the returned state.
    """
    p = params

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
    Al_tot = state.Al_tot + (np.sum(min_st[:, 4] * EW_p) * p.conv_Al
                             - prev.L * 1000 * (state.Al + state.AlOH4)) * dt
    Si_tot = state.Si_tot + (p.I_Si + np.sum(min_st[:, 5] * EW_p)
                             - LT * state.Si - UP_Si) * dt
    An_tot = state.An_tot + (p.I_An - (prev.L + prev.T) * state.An * 1000) * dt
    IC_tot = (state.IC_tot + now.I * 1000 * now.DIC_rain - ADV
              + (state.W_CaCO3 + state.W_MgCO3 + prev.r_het + prev.r_aut
                 - state.Fs - prev.L * 1000 * state.DIC) * dt)

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
    clip_Al = clip_Si = clip_An = clip_C = 0.0
    dt_max = float("inf")
    for _pool, _tent in ((state.Ca_tot, Ca_tot), (state.Mg_tot, Mg_tot),
                         (state.K_tot, K_tot), (state.Na_tot, Na_tot),
                         (state.Al_tot, Al_tot), (state.Si_tot, Si_tot),
                         (state.An_tot, An_tot), (state.IC_tot, IC_tot)):
        if _tent < _pool and _pool > 0.0:
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
    if An_tot < 0.0:
        clip_An = -An_tot; An_tot = 0.0
    if IC_tot < 0.0:
        clip_C = -IC_tot; IC_tot = 0.0

    # Alk_tot is DEFINED from the cation pools rather than integrated, so it is
    # formed after they are final. It is a charge balance and may legitimately
    # be negative; it is never clamped.
    Alk_tot = 2 * Mg_tot + 2 * Ca_tot + Na_tot + K_tot - An_tot     # [mol_c]

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
    _args = (Alk_tot, p.n, p.Zr, now.s, IC_tot, now.k1, now.k2, now.k_H,
             now.k_w, p.CEC_tot, p.conv_Al, Al_tot, p.K1, p.K2, p.K3, p.K4,
             Mg_tot, Ca_tot, Na_tot, K_tot, p.K_Ca_Al, p.K_Ca_Mg, p.K_Ca_Na,
             p.K_Ca_K, p.K_Ca_H)

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
        return _expand_8_numba(qv, now.k1, now.k2, now.k_w, p.CEC_tot,
                               p.conv_Al, p.K1, p.K2, p.K3, p.K4, p.K_Ca_Al,
                               p.K_Ca_Mg, p.K_Ca_Na, p.K_Ca_K, p.K_Ca_H)

    # initial guess
    nZrs = p.n * p.Zr * now.s * 1000
    Alk0 = (Alk_tot - state.R_alk) / nZrs
    CO2_w0 = IC_tot / (p.n * p.Zr * 1000) * 1 / (
        now.s * (1 + now.k1 / state.H + now.k2 * now.k1 / (state.H ** 2))
        + (1 - now.s) / now.k_H)
    Al_w0 = (Al_tot - (state.f_Al / 3) * p.CEC_tot * p.conv_Al) / nZrs
    K1, K2, K3, K4 = p.K1, p.K2, p.K3, p.K4     # also used by the Al ladder below
    # R_alk0, Al0 and the eight starting fractions are gone with the equations
    # that defined them: the reduced system starts from the retained unknowns
    # only, and the previous guesses for the eliminated ones were never
    # independent of these in the first place.
    Mg0 = (Mg_tot - state.f_Mg / 2 * p.CEC_tot) / nZrs
    Na0 = (Na_tot - state.f_Na * p.CEC_tot) / nZrs
    Ca0 = (Ca_tot - state.f_Ca / 2 * p.CEC_tot) / nZrs
    K0 = (K_tot - state.f_K * p.CEC_tot) / nZrs
    H0 = state.H

    def eqH(pv):
        return _eqH_numba(pv, now.k1, now.k2, CO2_w0, now.k_w, Alk0)
    H0_2 = fsolve(eqH, state.H)[0]

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

    # solution 2
    #
    # The gate is now scaled: each residual divided by the pool its own equation
    # balances, against one relative tolerance that therefore means the same
    # thing on every row. It replaces `np.any(abs(errors) > 1e-1)`, which on row
    # 16 -- the exchanger fractions summing to one -- accepted a 10 % violation
    # of CEC closure, and on the mol rows was never binding at all.
    #
    # What this does NOT do is change which steps take rung 2. Nothing on any
    # benchmark case reaches either threshold, so rung 2 still never fires and
    # the output is unchanged. The gate is now capable of firing for a defensible
    # reason, which is the whole of F3.2; making it fire usefully is F3.6, whose
    # real trigger is ier != 1 -- measured on 0.96 % of the suite's 1.41 M solves
    # and, on Example, carrying residuals four orders of magnitude worse than the
    # converged steps (median 4.7e-12 against 2.2e-16).
    rung = 1
    if np.any(abs(errors) / res_scale > RES_RTOL):
        q0 = np.array([CO2_w0, H0_2, Al_w0, Mg0, Ca0, Na0, K0, state.f_Ca])
        sol_r = root(equations8log, _y_from_q_numba(q0), jac=jacobian8log,
                     method="hybr", options={"xtol": 1e-14})
        q = _q_from_y_numba(sol_r.x)
        ier = sol_r.status
        sol = expand8(q)
        errors = np.asarray(equations(sol))
        nfev += sol_r.nfev
        njev += sol_r.njev
        rung = 2
        if np.any(abs(errors) / res_scale > RES_RTOL):
            worst = int(np.argmax(abs(errors) / res_scale))
            raise ValueError(
                "Solution not converging: scaled residual %.3e on equation %d "
                "(tolerance %.0e)" % ((abs(errors) / res_scale)[worst],
                                      worst + 1, RES_RTOL))

    (Alk, CO2_w, H, R_alk, Al_w, Al, Mg, Ca, Na, K,
     f_Al, f_Mg, f_Na, f_K, f_H, f_Ca) = sol

    # pH and C
    pH = -np.log10(H / p.conv_mol)                         # [-]
    CO2_air = CO2_w / now.k_H                              # [mol/l]
    HCO3 = now.k1 * CO2_w / H
    CO3 = now.k2 * now.k1 * CO2_w / (H ** 2)
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

    # Carbonate weathering
    Omega_CaCO3 = Ca * CO3 / p.K_CaCO3                     # [-]
    Omega_MgCO3 = Mg * CO3 / p.K_MgCO3
    [W_CaCO3, W_MgCO3] = smew.carb_W(CaCO3, MgCO3, Omega_CaCO3, Omega_MgCO3,
                                     now.s, p.Zr, p.r_CaCO3, p.r_MgCO3,
                                     p.tau_CaCO3, p.tau_MgCO3)

    # Silicate weathering
    M_min, M_rock, rock_f = state.M_min, state.M_rock, state.rock_f
    d, delta_d = state.d, state.delta_d
    lamb, SSA, psd, SA = state.lamb, state.SSA, state.psd, state.SA
    clip_M_min = np.zeros(p.number_min)
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
        for j in range(0, p.number_min):
            Omega[j] = weathering_kinec.Omega_sil(
                p.mineral[j], Ca, Mg, K, Na, Si, H, Al, p.Fe, p.K_sp[j],
                p.conv_mol, p.conv_Al)                     # [-]
            Wr[j] = now.s * p.diss_f * weathering_kinec.mineral_weathering(
                p.mineral[j], now.T_K, Omega[j], H, Al, p.conv_mol, p.conv_Al)

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
            d = np.where(d < 0, 0.0, d)
            delta_d = np.insert(np.diff(d), 0, d[0])               # [m]
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
        pH=pH, H=H, f_H=f_H, Alk=Alk, Alk_tot=Alk_tot, An=An, An_tot=An_tot,
        CO2_w=CO2_w, CO2_air=CO2_air, HCO3=HCO3, CO3=CO3, DIC=DIC,
        IC_tot=IC_tot, Fs=Fs, ADV=ADV,
        Ca=Ca, Ca_tot=Ca_tot, f_Ca=f_Ca, Mg=Mg, Mg_tot=Mg_tot, f_Mg=f_Mg,
        K=K, K_tot=K_tot, f_K=f_K, Na=Na, Na_tot=Na_tot, f_Na=f_Na,
        Si=Si, Si_tot=Si_tot,
        Al=Al, AlOH=AlOH, AlOH2=AlOH2, AlOH3=AlOH3, AlOH4=AlOH4,
        Al_w=Al_w, Al_tot=Al_tot, f_Al=f_Al, R_alk=R_alk,
        CaCO3=CaCO3, MgCO3=MgCO3,
        Omega_CaCO3=Omega_CaCO3, Omega_MgCO3=Omega_MgCO3,
        W_CaCO3=W_CaCO3, W_MgCO3=W_MgCO3,
        M_rock=M_rock, SA=SA, M_min=M_min, rock_f=rock_f,
        EW=EW, Wr=Wr, Omega=Omega,
        d=d, delta_d=delta_d, lamb=lamb, SSA=SSA, psd=psd,
        UP_Ca=UP_Ca, UP_Mg=UP_Mg, UP_K=UP_K, UP_Si=UP_Si,
        clip_Ca=clip_Ca, clip_Mg=clip_Mg, clip_K=clip_K, clip_Na=clip_Na,
        clip_Al=clip_Al, clip_Si=clip_Si, clip_An=clip_An, clip_C=clip_C,
        clip_CaCO3=clip_CaCO3, clip_MgCO3=clip_MgCO3, clip_M_min=clip_M_min,
        dt_max=dt_max, n_substeps=1.0,
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
# Not tighter than that, yet. Tripping this gate today sends the step to rung 2
# and, if that also fails, RAISES -- so an over-tight gate converts a run that
# worked into a crash. F3.6 replaces that with a ladder that degrades instead of
# raising; tightening belongs there, once failing is survivable. res_max is in
# the output, so how close any run came is now an observable rather than a
# guess.
RES_RTOL = 1e-7


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
    if not diag["significant"] or max_substeps <= 1:
        return new, diag

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
            }
            if d["significant"]:
                ok = False
                break
        if ok:
            st.n_substeps = float(n)
            st.dt_max = new.dt_max
            acc["substeps"] = n
            return st, acc
        n *= 2

    # Sub-dividing did not rescue it. Keep the clamped full step, which at least
    # records what it destroyed, and let the caller see n_substeps == 0 as the
    # marker that the limiter was reached rather than avoided.
    new.n_substeps = 0.0
    diag["substeps"] = 0
    return new, diag


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def biogeochem_balance(n, s, L, T, I, v, k_v, RAI, root_d, Zr, r_het, r_aut, D, temp_soil, pH_in, conc_in, f_CEC_in, K_CEC, CEC_tot, Si_in, CaCO3_in, MgCO3_in, M_rock_in, t_app, mineral, rock_f_in, d_in, psd_perc_in, SSA_in, diss_f, dt, conv_Al, conv_mol, keyword_add, *, max_substeps=MAX_SUBSTEPS):
    '''Run the soil biogeochemistry over the whole forcing series.

    !!! Fe not modeled !!! -> set to zero to make the model run!!

    The 34-argument signature is unchanged and the returned series are unchanged.
    Internally this is now a loop over `step`, against one `SoilParams` resolved
    once and one `StepForcing` per timestep; see smew/state.py.

    max_substeps is keyword-only and defaults to the module setting, so the 34
    positional arguments are exactly as they were. Set it to 1 to disable
    sub-stepping and fall back to clamping, which is how the clipping rows in
    smew.ledger are exercised -- with sub-stepping on, no case in the suite ever
    reaches the clamp.

    The `D` argument is accepted and discarded. It has never been used: the CO2
    diffusivity is recomputed below from the moisture series before D is read,
    so the value smew.respiration returns and every notebook passes in has no
    effect. Left in place because removing it would change the signature, which
    F1 is not allowed to do; worth removing deliberately later.
    '''
    nt = len(s)
    p = _resolve_params(n, Zr, CEC_tot, k_v, RAI, root_d, K_CEC, mineral,
                        M_rock_in, t_app, d_in, dt, diss_f, conv_mol, conv_Al)

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

    # --- rainwater ------------------------------------------------------------
    Alk_rain = 0                                           # alk
    CO2_w_rain = k_H * p.CO2_atm                           # [mol/l] Henry's law
    H_rain = np.zeros(nt)
    DIC_rain = np.zeros(nt)
    for i in range(0, nt):
        def equations(pv):
            return _equations_water_numba(pv, Alk_rain, k1[i], k2[i],
                                          CO2_w_rain[i], k_w[i])
        H_rain[i] = fsolve(equations, 10 ** -6 * conv_mol)[0]      # [mol/l]
        DIC_rain[i] = (CO2_w_rain[i] + k1[i] * CO2_w_rain[i] / H_rain[i]
                       + k2[i] * k1[i] * CO2_w_rain[i] / (H_rain[i] ** 2))

    forcing = _forcing(s, v, I, L, T, Dw, D, r_het, r_aut, temp_soil, T_K,
                       k1, k2, k_w, k_H, DIC_rain)

    # --- storage --------------------------------------------------------------
    out = {k: np.zeros(nt) for k in STATE_1D}
    for k in STATE_MIN:
        out[k] = np.zeros([p.number_min, nt])
    for k in STATE_PSD:
        out[k] = np.zeros([p.n_d_cl, nt])
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
        "pH": out["pH"], "H": out["H"], "f_H": out["f_H"], "Alk": out["Alk"],
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
        # --- aluminium ---
        "Al": out["Al"], "AlOH": out["AlOH"], "AlOH2": out["AlOH2"],
        "AlOH3": out["AlOH3"], "AlOH4": out["AlOH4"], "Al_w": out["Al_w"],
        "Al_tot": out["Al_tot"], "f_Al": out["f_Al"],
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
        "clip_CaCO3": out["clip_CaCO3"], "clip_MgCO3": out["clip_MgCO3"],
        "clip_M_min": out["clip_M_min"],
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
