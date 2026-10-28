# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""Ionic strength and activity coefficients. The F4 piece.

WHY THIS EXISTS
---------------
Every thermodynamic constant SMEW uses -- carbonate dissociation, aluminium
hydrolysis, mineral solubility products, Gaines-Thomas selectivity, and the H+
dependence in the KINEC rate laws -- is defined on ACTIVITIES. Until F4 the
model evaluated all of them on raw concentrations, which is the same as
asserting that every activity coefficient is 1. `weathering_kinec.py` says so in
its own docstring. Enhanced rock weathering raises ionic strength on purpose, so
the error grows in exactly the direction the model is used to explore.

Measured across the seven benchmark cases before any of this was written, the
ionic strength spans 1.3e-3 (Bottles_tePas, untreated) to 1.6e-1 (Vulkaneifel,
a real field site). At the top of that range a trivalent ion sits at ~13 % of
its nominal activity.

WHY EXTENDED DEBYE-HUCKEL AND NOT DAVIES
----------------------------------------
The Debye-Huckel limiting law is good to about I = 5e-3 and dies well below this
model's range. Davies reaches I ~ 0.5 but carries no ion-specific parameter at
all, and its error grows fastest above I ~ 0.1 -- which is precisely where the
one real field case lives. So the primary model is the extended form with an
ion-size parameter and a linear term, the same one PHREEQC applies to its major
ions (Truesdell & Jones 1974; WATEQ):

    log10 gamma_i = -A z_i^2 sqrt(mu) / (1 + B a_i sqrt(mu))  +  b_i mu

`a_i` (angstrom) and `b_i` are taken FROM `phreeqc.dat`, verbatim, and the
provenance of each is recorded in the table below. That is deliberate and it is
what makes the cross-check meaningful: PHREEQC is this block's oracle, so using
its functional form and its parameters means a disagreement is a bug in this
file, not a comparison of two different approximations. Davies is the fallback
for any species with no tabulated `a_i`.

    Measured against PHREEQC at mu = 2e-3 .. 3e-1 and 13 / 25 C, over
    H+, Na+, Ca+2, CO3-2 and Al+3: worst disagreement 0.36 %, on Al+3 at
    mu = 0.3. See tests/check_activity.py.

WHAT IS NOT HERE
----------------
Ion pairing. CaHCO3+, CaCO3(aq), MgHCO3+ and the sulfate complexes tie up free
Ca+2 and CO3-2, and SMEW models none of them. That is a LARGER correction to
calcite saturation than activity coefficients are: at one `Example` timestep
SMEW reads Omega_calcite = 1.17 on concentrations, this module's coefficients
bring it to ~0.28, and PHREEQC -- which has both -- says 0.69. Activities close
part of the gap and overshoot the part they close. Naming that here because a
quantified omission is a much better position than an unexplained residual.

A NOTE ON THE NAME `mu`
-----------------------
Ionic strength is conventionally `I`, and `I` is taken twice over in this
codebase: infiltration in the output contract (`harness.py`, `moisture.py`) and
`I_Ca .. I_An` for the background solute input fluxes (`biogeochem.py`). `mu` is
PHREEQC's own name for it and it collides with nothing.

A NOTE ON UNITS
---------------
Everything entering `ionic_strength` is in the model's own units -- [mol-conv/l]
for the major ions, [mol-conv_Al/l] for the five aluminium species, because
`conv_Al` MULTIPLIES `conv_mol` rather than replacing it. The returned `mu` is in
mol/l, which is what the Debye-Huckel expression needs and what PHREEQC reports.
Getting this wrong is silent: it produces a plausible-looking gamma from a
nonsense ionic strength.
"""
import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# The species table
# ---------------------------------------------------------------------------
#
# Index order is fixed and is the return order of `gammas`. The module-level
# names are the only supported way to index it -- a bare integer at a call site
# is how these tables drift out of step with their consumers.
#
# `a0` and `b0` are lifted from phreeqc.dat's SOLUTION_SPECIES `-gamma` lines.
# Where a species carries two -gamma lines, PHREEQC uses the LAST, so Na+ takes
# (4.08, 0.082) -- the halite-solubility pair -- not (4.0, 0.075).
#
# Neutral species (z = 0) get gamma = 1 from the expression itself, with no
# special case: CO2(aq), H4SiO4 and Al(OH)3 are all uncharged. A Setchenow
# salting-out term would make them slightly greater than 1 at high mu; at this
# model's ionic strengths that is a sub-percent effect on a species whose
# activity enters nothing that is currently calibrated, so it is not modelled.
#
# `An` is the lumped anion tracer. It has no identity in the model, but it is
# not undetermined either: `constants.py` sets MM_Anions = 62 g/mol_c, which is
# NITRATE, so monovalent is the model's own standing assumption rather than a
# new one made here, and NO3- is what its a0 is taken from. This matters more
# than it looks -- An carries up to 44 % of the ionic strength on the field
# case, the largest single contribution of any species.

SP_H, SP_OH, SP_HCO3, SP_CO3 = 0, 1, 2, 3
SP_CA, SP_MG, SP_NA, SP_K, SP_AN = 4, 5, 6, 7, 8
SP_AL, SP_ALOH, SP_ALOH2, SP_ALOH3, SP_ALOH4 = 9, 10, 11, 12, 13
SP_SI, SP_CO2 = 14, 15
N_SP = 16

SP_NAMES = ("H+", "OH-", "HCO3-", "CO3-2", "Ca+2", "Mg+2", "Na+", "K+",
            "An-", "Al+3", "AlOH+2", "Al(OH)2+", "Al(OH)3", "Al(OH)4-",
            "H4SiO4", "CO2")

#                    H+   OH-  HCO3  CO3   Ca   Mg   Na   K   An   Al  AlOH AlOH2 AlOH3 AlOH4 Si  CO2
_Z = np.array([       1.0, -1., -1.,  -2.,  2.,  2.,  1.,  1., -1.,  3.,  2.,  1.,   0.,  -1.,  0., 0.])
_A0 = np.array([      9.0,  3.5, 5.4,  5.4, 5.0, 5.5, 4.08, 3.5, 3.0, 9.0, 5.4, 5.4,  0.0,  4.5, 0., 0.])
_B0 = np.array([      0.0,  0.0, 0.0,  0.0, .165, .20, .082, .015, 0., 0.,  0.,  0.,   0.,   0.,  0., 0.])
# Whether phreeqc.dat tabulates an ion-size parameter for this species. False
# means Davies, and in this table that is only ever a charged species with no
# entry -- there are none today, so the branch exists for a species added later.
_HAS_A0 = _A0 > 0.0

GAMMA_ONE = np.ones(N_SP)     # the gamma_on=False vector, allocated once


# ---------------------------------------------------------------------------
# Water properties and the Debye-Huckel coefficients
# ---------------------------------------------------------------------------

@njit(cache=True)
def _rho_w(t_C):
    """Density of pure water [g/cm3]. Kell (1975), 0-100 C at 1 atm."""
    return ((999.83952 + 16.945176 * t_C - 7.9870401e-3 * t_C ** 2
             - 46.170461e-6 * t_C ** 3 + 105.56302e-9 * t_C ** 4
             - 280.54253e-12 * t_C ** 5)
            / (1.0 + 16.879850e-3 * t_C) / 1000.0)


@njit(cache=True)
def _eps_w(t_C):
    """Static dielectric constant of water [-]. Malmberg & Maryott (1956)."""
    return 87.740 - 0.40008 * t_C + 9.398e-4 * t_C ** 2 - 1.410e-6 * t_C ** 3


@njit(cache=True)
def dh_coeffs(T_K):
    """(A, B) of the Debye-Huckel expression at temperature T_K.

    A in [kg^0.5 mol^-0.5], B in [kg^0.5 mol^-0.5 angstrom^-1], so `a0` is in
    angstrom as phreeqc.dat tabulates it. A = 0.511, B = 0.329 at 25 C.

    These are not fitted: they follow from the density and dielectric constant
    of water. `K_C(T_K, conv_mol)` is the existing pattern for a temperature-
    dependent constant in this codebase, and this follows it.
    """
    t_C = T_K - 273.15
    r = _rho_w(t_C)
    e = _eps_w(t_C)
    A = 1.82483e6 * np.sqrt(r) / (e * T_K) ** 1.5
    B = 50.2916 * np.sqrt(r) / np.sqrt(e * T_K)
    return A, B


# ---------------------------------------------------------------------------
# Ionic strength
# ---------------------------------------------------------------------------

@njit(cache=True)
def ionic_strength(Ca, Mg, Na, K, H, OH, HCO3, CO3, An,
                   Al, AlOH, AlOH2, AlOH4, conv_mol, conv_Al):
    """mu = 1/2 sum(c_i z_i^2) over the modelled species [mol/l].

    Inputs are in the model's own units: [mol-conv/l] for everything except the
    four aluminium species, which are [mol-conv_Al/l].

    Al(OH)3 is uncharged and contributes nothing, so it is not an argument.
    H4SiO4 likewise. Al(OH)4- IS charged and is included -- leaving it out was
    tempting because the expansion computes it by difference, but it is the
    dominant aluminium species above pH 6 and dropping it would understate mu on
    every alkaline step.
    """
    cm = conv_mol
    cA = conv_mol * conv_Al
    return 0.5 * (
        4.0 * Ca / cm + 4.0 * Mg / cm + Na / cm + K / cm
        + H / cm + OH / cm + HCO3 / cm + 4.0 * CO3 / cm
        + An / cm                         # monovalent: see the table's note
        + 9.0 * Al / cA + 4.0 * AlOH / cA + AlOH2 / cA + AlOH4 / cA
    )


@njit(cache=True)
def gammas(mu, T_K):
    """Activity coefficients for all N_SP species at (mu, T_K).

    Extended Debye-Huckel where phreeqc.dat tabulates an ion size, Davies
    otherwise. Returns a length-N_SP array indexed by the SP_* constants.

    mu <= 0 returns all ones rather than a nan. That is not defensive
    programming for its own sake: a frozen step holds the chemistry and can
    present an all-zero state, and `sqrt` of a negative would propagate a nan
    into every constant in the model rather than failing where it happened.
    """
    g = np.ones(N_SP)
    if mu <= 0.0:
        return g
    A, B = dh_coeffs(T_K)
    sq = np.sqrt(mu)
    for i in range(N_SP):
        z2 = _Z[i] * _Z[i]
        if z2 == 0.0:
            continue                      # neutral: gamma = 1, see the header
        if _HAS_A0[i]:
            lg = -A * z2 * sq / (1.0 + B * _A0[i] * sq) + _B0[i] * mu
        else:
            lg = -A * z2 * (sq / (1.0 + sq) - 0.3 * mu)          # Davies
        g[i] = 10.0 ** lg
    return g


def gamma_ion(mu, T_K, z, a0, b0=0.0):
    """Activity coefficient of one ion outside the species table, by the same
    extended Debye-Huckel as gammas().

    For the iron ions S3 needs (phreeqc.dat ion sizes: Fe+2 6.0, Fe+3 9.0),
    which are deliberately NOT in the table: Fe stays out of the ionic strength
    and the implicit solve, as P3 decided, so adding them to SP_* would change
    what every other gamma means for nothing. mu <= 0 gives 1, as gammas() does.
    """
    if mu <= 0.0 or z == 0:
        return 1.0
    A, B = dh_coeffs(T_K)
    sq = np.sqrt(mu)
    return 10.0 ** (-A * z * z * sq / (1.0 + B * a0 * sq) + b0 * mu)


A0_FE2, A0_FE3 = 6.0, 9.0     # phreeqc.dat -gamma for Fe+2 and Fe+3


# ---------------------------------------------------------------------------
# Effective constants: how activities reach the implicit solve
# ---------------------------------------------------------------------------
#
# THE POINT OF THIS FUNCTION, which is the whole reason F4 is cheap.
#
# The implicit system solves for CONCENTRATIONS. Its residual is a set of mass
# balances plus mass-action expressions, and every mass-action expression is a
# product of powers of the unknowns divided by a constant. Introducing activity
# coefficients multiplies each such product by a fixed factor -- fixed, because
# under LAGGED coupling gamma is computed from the previous state and is
# therefore not a function of the unknowns. So every one of those factors can be
# folded into the constant it sits next to, and the residual algebra is
# untouched.
#
# Three things follow, and the third is the valuable one:
#
#   1. `_biogeochem_equations_8_numba` and `_biogeochem_equations_8log_numba`
#      need no edit. They already take all eleven of these as arguments.
#   2. Every output series stays in concentrations, which is what the mass
#      balances, the ledger and every existing plot expect.
#   3. `_jacobian_8log_numba` -- F3.5's hand-derived analytic Jacobian and the
#      entry-by-entry finite-difference gate that guards it -- remains EXACTLY
#      valid, because it differentiates with respect to the unknowns at fixed
#      constants, and a lagged gamma is a fixed constant. That was the largest
#      implementation risk in this block and it is not present.
#
# The same argument holds under Picard iteration (gamma is still constant within
# each inner solve) and fails under fully-coupled gamma, which is the cost
# argument against fully-coupled: it would mean re-deriving and re-gating the
# hardest thing F3 produced, to buy a second-order accuracy improvement.
#
# TWO ROWS ARE NEARLY OR EXACTLY IMMUNE, and it is worth knowing which.
# Magnesium exchange is homovalent -- Ca+2 against Mg+2 -- so its gamma ratio is
# gamma_Ca/gamma_Mg, which would be exactly 1 in any charge-only model (Davies).
# It is not exactly 1 here only because phreeqc.dat gives Ca and Mg different ion
# sizes; the residual correction is ~4 % at mu = 0.1, against ~150 % for the
# Na row and ~4x for the Al row. K3 is exactly immune: both sides of
# Al(OH)2+ = Al(OH)3 + H+ change charge by the same amount.

@njit(cache=True)
def effective_constants(g, k1, k2, k_w, K1, K2, K3, K4,
                        K_Ca_Mg, K_Ca_K, K_Ca_Na, K_Ca_Al, K_Ca_H):
    """The eleven constants of the implicit system, corrected for activities.

    Each is returned in the form the residual already uses it -- a quotient of
    CONCENTRATIONS -- so the caller substitutes and changes nothing else.

    Carbonate system (CO2 is neutral, so gamma_CO2 = 1):
        k1  = [H][HCO3]/[CO2]        K1_true = a_H a_HCO3 / a_CO2
        k2  = [H][CO3]/[HCO3]        K2_true = a_H a_CO3  / a_HCO3
        k_w = [H][OH]                Kw_true = a_H a_OH

    Aluminium ladder, each rung Al(OH)n + H2O = Al(OH)n+1 + H+, as
    `_expand_8_numba` uses them (K1 = [AlOH][H]/[Al], and so on up):

    Gaines-Thomas exchange, in the algebraic form of equations 11-15. The
    exchanger fractions f_x are equivalent fractions on the solid and are NOT
    activities, so only the aqueous side of each ratio is corrected:
        K_Ca_Mg = [Mg] f_Ca / (f_Mg [Ca])
        K_Ca_Na = [Na]^2 f_Ca / (f_Na^2 [Ca])          (K, H the same shape)
        K_Ca_Al = ([Al]/conv_Al)^2 f_Ca^3 / (f_Al^2 [Ca]^3)
    """
    gH = g[SP_H]; gOH = g[SP_OH]; gHCO3 = g[SP_HCO3]; gCO3 = g[SP_CO3]
    gCa = g[SP_CA]; gMg = g[SP_MG]; gNa = g[SP_NA]; gK = g[SP_K]
    gAl = g[SP_AL]; gAlOH = g[SP_ALOH]; gAlOH2 = g[SP_ALOH2]
    gAlOH3 = g[SP_ALOH3]; gAlOH4 = g[SP_ALOH4]

    k1e = k1 / (gH * gHCO3)
    k2e = k2 * gHCO3 / (gH * gCO3)
    k_we = k_w / (gH * gOH)

    K1e = K1 * gAl / (gAlOH * gH)
    K2e = K2 * gAlOH / (gAlOH2 * gH)
    K3e = K3 * gAlOH2 / (gAlOH3 * gH)
    K4e = K4 * gAlOH3 / (gAlOH4 * gH)

    K_Ca_Mg_e = K_Ca_Mg * gCa / gMg
    K_Ca_K_e = K_Ca_K * gCa / (gK * gK)
    K_Ca_Na_e = K_Ca_Na * gCa / (gNa * gNa)
    K_Ca_Al_e = K_Ca_Al * gCa ** 3 / (gAl * gAl)
    K_Ca_H_e = K_Ca_H * gCa / (gH * gH)

    return (k1e, k2e, k_we, K1e, K2e, K3e, K4e,
            K_Ca_Mg_e, K_Ca_K_e, K_Ca_Na_e, K_Ca_Al_e, K_Ca_H_e)


# ---------------------------------------------------------------------------
# The calibration convention of K_GT_CEC, which is not the model's convention
# ---------------------------------------------------------------------------
#
# `effective_constants` above assumes every constant handed to it is a
# THERMODYNAMIC constant, defined on activities. For the carbonate and aluminium
# constants that is true -- Stumm & Morgan and Weil & Brady both tabulate
# activity constants. For the Gaines-Thomas selectivity coefficients it is NOT,
# and the source says so in as many words.
#
# constants.py cites a meta-analysis of Dutch soils; that is de Vries & Posch
# (2003), Alterra-rapport 701, and SMEW's three soil classes are its 0-30 cm
# rows verbatim -- sand (0.557, -1.157, 0.745), loess->"loam" (0.108, -2.072,
# 0.378) and clay (0.385, -2.428, 0.774) for Ca-Mg, Ca-K and Ca-Na. Page 13:
#
#     "Although from a theoretical standpoint activities would be more
#      appropriate, we only consider exchange equations involving
#      concentrations, since they are mostly used in simple models."
#
# and, on p.29, about a model that does use activities:
#
#     "It should be noted, however, that in the MAGIC model activities are used
#      instead of concentrations. Thus the exchange constants presented here
#      would have to be corrected with the proper activity coefficients."
#
# So those five numbers are concentration quotients measured at the ionic
# strength of Dutch forest soil solutions. Feeding them to
# `effective_constants` unchanged would apply an activity correction to a
# constant that already has one baked in at the calibration conditions -- a
# double correction, and in the wrong direction.
#
# The fix is a conversion, NOT a refit. Lift each measured constant to its
# thermodynamic value once, using gamma at the ionic strength of the
# calibration data; `effective_constants` then brings it back down to a
# concentration quotient at the model's own ionic strength, step by step. The
# net correction is therefore only the DIFFERENCE between the two ionic
# strengths, and it vanishes identically when the model sits where the
# calibration data sat. That is the physically right answer, and it is also why
# this is cheap: nothing is being re-estimated.
#
# MU_REF_CEC is an estimate and is flagged as one. Alterra-rapport 701 gives
# adsorbed fractions and exchange constants but does not tabulate the solution
# composition, so its ionic strength is not recoverable from the report. 5e-3
# mol/l is representative of the soils it describes -- Dutch non-agricultural
# forest and heath, soil solution extracted by centrifugation, pH 3-5, under
# the high N and S deposition of the 1990s. Sensitivity over 2e-3 to 2e-2 is
# reported in tests/check_activity.py cec.
#
# Worth keeping in proportion. The report's own standard deviations on these
# constants are 0.35 to 0.79 log units -- factors of 2 to 6 -- while this
# correction is 0.1 to 0.2 log units. It is made because being knowingly
# inconsistent is not defensible when consistency is free, not because it is
# large compared with what is known about these parameters.

MU_REF_CEC = 5.0e-3          # [mol/l]  see above; an estimate, not a datum


@njit(cache=True)
def exchange_to_thermodynamic(g_ref, K_Ca_Mg, K_Ca_K, K_Ca_Na, K_Ca_Al,
                              K_Ca_H):
    """Concentration-based selectivity coefficients -> thermodynamic ones.

    The exact inverse of the five exchange lines of `effective_constants`,
    evaluated at the activity coefficients of the CALIBRATION data rather than
    of the model state. Applied once per run, not once per step.

    At g_ref = 1 this is an exact identity, so a gamma_on=False run is
    untouched and the conversion cannot leak into the bit-identity gate.
    """
    gH = g_ref[SP_H]; gCa = g_ref[SP_CA]; gMg = g_ref[SP_MG]
    gNa = g_ref[SP_NA]; gK = g_ref[SP_K]; gAl = g_ref[SP_AL]
    return (K_Ca_Mg * gMg / gCa,
            K_Ca_K * (gK * gK) / gCa,
            K_Ca_Na * (gNa * gNa) / gCa,
            K_Ca_Al * (gAl * gAl) / gCa ** 3,
            K_Ca_H * (gH * gH) / gCa)
