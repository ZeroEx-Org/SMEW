# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""The state a SMEW timestep advances, and the constants it advances against.

Three containers, separated by how often each changes:

    SoilParams   resolved once per run and then never again -- unit
                 conversions, equilibrium constants, mineral stoichiometry,
                 rock geometry.
    StepForcing  one timestep's drivers, produced upstream by the hydroclimatic,
                 moisture and respiration stages: wetness, temperature, the
                 water fluxes, respiration.
    SoilState    the prognostic state at one instant -- everything the model has
                 to carry forward in order to take another step.

WHY THIS SPLIT, BEYOND TIDINESS
-------------------------------
biogeochem_balance could not advance a single timestep, could not be restarted
from a saved state, and could not carry a depth index, because all three of
these lived tangled together in one 460-line function's locals. With them
separated, one step is `step(state, params, now, prev, dt)`, and a depth-
resolved model (parent plan item 6) is N SoilStates against one SoilParams.

A NOTE ON numba
---------------
These are orchestration-layer objects and never cross the jit boundary. The
jitted helpers (_biogeochem_equations_numba and friends) still take plain floats
and plain arrays, exactly as before.

A NOTE ON MUTATION
------------------
SoilParams is frozen: it is shared by every step of a run, so nothing may edit
it in place. SoilState is not frozen, because `step` builds a new one field by
field and freezing would cost a dict round-trip per field; purity is enforced
instead by `step` never writing to its input, which tests/check_step.py checks
directly by round-tripping a state through step twice.

`slots=True` on all three: it makes attribute access materially cheaper in a
loop that runs ~5e4 times per simulated year, and it turns a typo in a field
name into an AttributeError instead of a silently-added attribute.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class SoilParams:
    """Everything resolved once at the start of a run.

    Two of these fields exist to kill a live footgun. `conv_mol` and `conv_Al`
    are the unit conversions (moles -> micromoles, and micromoles -> nanomoles
    for aluminium). They currently have to be passed identically to six separate
    call sites, and nothing checks that they match; resolving them here once
    means a single source of truth for the whole run.
    """
    # --- units -----------------------------------------------------------
    conv_mol: float
    conv_Al: float
    # --- geometry and exchange capacity ----------------------------------
    n: float                     # porosity [-]
    Zr: float                    # rooting depth [m]
    CEC_tot: float               # cation exchange capacity [mol_c]
    Z_CO2: float                 # depth of the CO2 diffusion gradient [m]
    # --- vegetation ------------------------------------------------------
    k_v: float
    RAI: float
    root_d: float
    xi: np.ndarray               # nutrient content per g biomass [mol-conv/g]
    # --- atmosphere ------------------------------------------------------
    CO2_atm: float
    # --- aluminium hydrolysis --------------------------------------------
    K1: float
    K2: float
    K3: float
    K4: float
    # --- Gaines-Thomas cation exchange selectivity ------------------------
    K_Ca_Mg: float
    K_Ca_K: float
    K_Ca_Na: float
    K_Ca_Al: float
    K_Ca_H: float
    # --- carbonate minerals ----------------------------------------------
    K_CaCO3: float
    K_MgCO3: float
    r_CaCO3: float
    r_MgCO3: float
    tau_CaCO3: float
    tau_MgCO3: float
    # --- silicate minerals ------------------------------------------------
    mineral: tuple               # names, in the order every per-mineral array uses
    number_min: int
    MM_min: np.ndarray           # molar mass per mineral [g/mol-conv]
    min_st: np.ndarray           # stoichiometry [Ca, Mg, K, Na, Al, Si] per mineral
    K_sp: np.ndarray             # solubility product per mineral
    diss_f: float                # per-experiment dissolution scaling [-]
    # --- rock geometry ----------------------------------------------------
    rho_rock: float
    a: float                     # fractal prefactor, refined from measured SSA
    b: float                     # fractal exponent
    M_iner: float                # inert (non-dissolving) rock mass [g/m2]
    n_d_cl: int                  # number of particle-diameter classes
    tt_app: int                  # timestep index at which rock is applied
    has_rock: bool
    # --- background solute input, replacing what leaching removes ---------
    # Resolved from the initial state rather than from constants, so these are
    # filled in by a second dataclasses.replace() once the ICs exist.
    I_Ca: float = 0.0
    I_Mg: float = 0.0
    I_K: float = 0.0
    I_Na: float = 0.0
    I_Si: float = 0.0
    I_An: float = 0.0
    # --- iron --------------------------------------------------------------
    # Not modelled. Hardcoded to zero exactly as biogeochem.py always did; named
    # here so that it is a stated assumption rather than a bare literal.
    Fe: float = 0.0


@dataclass(slots=True)
class StepForcing:
    """One timestep's drivers. Read-only in practice; built once per timestep.

    Held as a per-step object rather than as arrays-plus-an-index because the
    freezing logic makes the "previous" step an arbitrary earlier index, not
    i-1: after a frost the model resumes from the last unfrozen step. Passing
    two of these makes that jump explicit at the call site instead of hiding it
    behind index arithmetic inside the physics.
    """
    s: float                     # relative soil moisture [-]
    v: float                     # above-ground biomass [g/m2]
    I: float                     # infiltration [m]
    L: float                     # leaching [m/d]
    T: float                     # transpiration [m/d]
    Dw: float                    # solute diffusivity in soil water [m2/d]
    D: float                     # CO2 diffusivity in soil air [m2/d]
    r_het: float                 # heterotrophic respiration [mol-conv/d]
    r_aut: float                 # autotrophic respiration [mol-conv/d]
    temp_soil: float             # [C]
    T_K: float                   # [K]
    k1: float                    # carbonate dissociation constants at T_K
    k2: float
    k_w: float
    k_H: float
    DIC_rain: float              # DIC of incoming rainwater [mol-conv/l]


# Series written once per executed timestep. Kept as module constants so the
# allocator, the writer and the reader cannot drift apart.
STATE_1D = (
    "pH", "H", "f_H", "Alk", "Alk_tot", "An", "An_tot",
    "CO2_w", "CO2_air", "HCO3", "CO3", "DIC", "IC_tot", "Fs", "ADV",
    "Ca", "Ca_tot", "f_Ca", "Mg", "Mg_tot", "f_Mg",
    "K", "K_tot", "f_K", "Na", "Na_tot", "f_Na", "Si", "Si_tot",
    "Al", "AlOH", "AlOH2", "AlOH3", "AlOH4", "Al_w", "Al_tot", "f_Al",
    "R_alk", "M_rock", "CaCO3", "MgCO3", "W_CaCO3", "W_MgCO3",
    "Omega_CaCO3", "Omega_MgCO3", "SA",
)
STATE_MIN = ("EW", "Wr", "Omega", "M_min", "rock_f")      # (n_mineral, n_steps)
STATE_PSD = ("d", "delta_d", "lamb", "SSA", "psd")        # (n_class,   n_steps)


@dataclass(slots=True)
class SoilState:
    """The prognostic state at one instant.

    "Prognostic" means: enough to take the next step from, with nothing else
    carried over. That is a stronger claim than "the variables the model plots",
    and it is what makes restart, sub-stepping (F2) and depth resolution
    possible at all.

    UP_Ca / UP_Mg / UP_K / UP_Si are the plant uptake fluxes of the transition
    INTO this state, not a property of the instant. They are carried here only
    so the caller can reproduce the legacy output layout -- see the note in
    biogeochem.biogeochem_balance about uptake being written at `last`.
    """
    # acid-base and the carbonate system
    pH: float; H: float; f_H: float; Alk: float; Alk_tot: float
    An: float; An_tot: float
    CO2_w: float; CO2_air: float; HCO3: float; CO3: float; DIC: float
    IC_tot: float; Fs: float; ADV: float
    # cations
    Ca: float; Ca_tot: float; f_Ca: float
    Mg: float; Mg_tot: float; f_Mg: float
    K: float; K_tot: float; f_K: float
    Na: float; Na_tot: float; f_Na: float
    Si: float; Si_tot: float
    # aluminium
    Al: float; AlOH: float; AlOH2: float; AlOH3: float; AlOH4: float
    Al_w: float; Al_tot: float; f_Al: float
    # exchange reserve
    R_alk: float
    # carbonate solids
    CaCO3: float; MgCO3: float
    Omega_CaCO3: float; Omega_MgCO3: float
    W_CaCO3: float; W_MgCO3: float
    # silicate solids, per mineral
    M_rock: float; SA: float
    M_min: np.ndarray; rock_f: np.ndarray
    EW: np.ndarray; Wr: np.ndarray; Omega: np.ndarray
    # particle size distribution, per diameter class
    d: np.ndarray; delta_d: np.ndarray
    lamb: np.ndarray; SSA: np.ndarray; psd: np.ndarray
    # uptake fluxes of the transition into this state
    UP_Ca: float = 0.0; UP_Mg: float = 0.0; UP_K: float = 0.0; UP_Si: float = 0.0

    # -- array <-> state ---------------------------------------------------
    # These loop over the STATE_* name tuples with getattr rather than assigning
    # 56 fields by hand, so that adding a field cannot leave the writer and the
    # reader disagreeing. Measured at 0.26 s of a 6.2 s year-long run (~4%),
    # against 1.5 s saved by making the residual pure -- so the generic version
    # is affordable. Revisit if a field count or a step count grows a lot.

    def write_into(self, out, i):
        """Store this state at index i of the output arrays.

        Deliberately does NOT write the UP_* fluxes: biogeochem.py writes those
        at `last`, not at i, and that layout is part of the frozen output.
        """
        for k in STATE_1D:
            out[k][i] = getattr(self, k)
        for k in STATE_MIN + STATE_PSD:
            out[k][:, i] = getattr(self, k)

    @classmethod
    def from_arrays(cls, out, i):
        """Rebuild the state stored at index i.

        Needed only where the previous step is not i-1: after a frost the model
        resumes from the last unfrozen step, and that state has to be read back
        out of the output arrays.
        """
        kw = {k: out[k][i] for k in STATE_1D}
        kw.update({k: out[k][:, i].copy() for k in STATE_MIN + STATE_PSD})
        return cls(**kw)

    def copy(self):
        from dataclasses import replace
        return replace(self)
