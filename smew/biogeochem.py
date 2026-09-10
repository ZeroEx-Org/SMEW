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
from scipy.optimize import fsolve
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
    def equations(pv):
        return _biogeochem_equations_numba(
            pv, Alk_tot, p.n, p.Zr, now.s, IC_tot, now.k1, now.k2, now.k_H,
            now.k_w, p.CEC_tot, p.conv_Al, Al_tot, p.K1, p.K2, p.K3, p.K4,
            Mg_tot, Ca_tot, Na_tot, K_tot, p.K_Ca_Al, p.K_Ca_Mg, p.K_Ca_Na,
            p.K_Ca_K, p.K_Ca_H
        )

    # initial guess
    nZrs = p.n * p.Zr * now.s * 1000
    Alk0 = (Alk_tot - state.R_alk) / nZrs
    CO2_w0 = IC_tot / (p.n * p.Zr * 1000) * 1 / (
        now.s * (1 + now.k1 / state.H + now.k2 * now.k1 / (state.H ** 2))
        + (1 - now.s) / now.k_H)
    R_alk0 = state.R_alk
    Al_w0 = (Al_tot - (state.f_Al / 3) * p.CEC_tot * p.conv_Al) / nZrs
    H_ = state.H
    K1, K2, K3, K4 = p.K1, p.K2, p.K3, p.K4
    Al0 = (H_ ** 4 / (H_ ** 4 + H_ ** 3 * K1 + H_ ** 2 * K1 * K2
                      + H_ * K1 * K2 * K3 + K1 * K2 * K3 * K4)) * Al_w0
    Mg0 = (Mg_tot - state.f_Mg / 2 * p.CEC_tot) / nZrs
    Na0 = (Na_tot - state.f_Na * p.CEC_tot) / nZrs
    Ca0 = (Ca_tot - state.f_Ca / 2 * p.CEC_tot) / nZrs
    K0 = (K_tot - state.f_K * p.CEC_tot) / nZrs
    H0 = state.H

    def eqH(pv):
        return _eqH_numba(pv, now.k1, now.k2, CO2_w0, now.k_w, Alk0)
    H0_2 = fsolve(eqH, state.H)[0]

    # solution 1
    x0 = np.array([Alk0, CO2_w0, H0, R_alk0, Al_w0, Al0, Mg0, Ca0, Na0, K0,
                   state.f_Al, state.f_Mg, state.f_Na, state.f_K,
                   state.f_H, state.f_Ca])
    sol = fsolve(equations, x0, xtol=1e-12)
    errors = np.asarray(equations(sol))                    # residuals

    # solution 2
    res_threshold = 1e-1
    rung = 1
    if np.any(abs(errors) > res_threshold):
        x0 = np.array([Alk0, CO2_w0, H0_2, R_alk0, Al_w0, Al0, Mg0, Ca0, Na0,
                       K0, state.f_Al, state.f_Mg, state.f_Na, state.f_K,
                       state.f_H, state.f_Ca])
        sol = fsolve(equations, x0, xtol=1e-14)
        errors = np.asarray(equations(sol))
        rung = 2
        if np.any(abs(errors) > res_threshold):
            raise ValueError("Solution not converging")

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

            prev_state, prev_idx = new, i

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
    }
