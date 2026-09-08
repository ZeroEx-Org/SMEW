# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""Mass-balance ledger: rebuild every source and sink of every element pool and
check that they add up.

WHAT THIS IS FOR
----------------
tests/golden/*.npz answer a comparison question -- "did anything change since
the reference commit?" -- and need a stored reference to answer it. This module
answers an invariant question -- "does this run conserve mass?" -- and needs no
reference at all, because the right answer is known in advance: zero.

That makes it the only check here that keeps working as new processes are added.
Every new source or sink becomes one more named row, and the total still has to
balance.

HOW IT WORKS, AND WHY IT IS DONE THIS WAY
-----------------------------------------
biogeochem_balance ends with `data = {k: v for k, v in locals().items()}`, so
every ingredient of every balance survives the run. This module reads that
dictionary and rebuilds each balance from the outside.

The obvious alternative -- having the model add up its own fluxes as it steps --
cannot fail. Writing `ledger_Ca += flux*dt` beside `Ca_tot[i] = Ca_tot[last] +
flux*dt` re-uses the same `flux`, so the two agree whatever `flux` is, including
when it is wrong. It would confirm that addition works and nothing else.

Rebuilding from the outside is a second, independent reading of the scheme. If
it agrees to roundoff, then the balance in the code and the balance we believe
is in the code are the same balance.

WHAT EACH POOL IS
-----------------
For each element the model carries ONE total per square metre of soil surface,
in moles: dissolved in the soil water PLUS held on the exchange sites of clay
and organic matter. Concentrations are recovered from that total each step by
the implicit solver. So wetting and drying dilute or concentrate the solution
without changing the pool -- which is why no dilution term appears anywhere
below. The pool is an inventory, not a concentration.

USAGE
-----
    from smew import ledger
    led = ledger.build(data, water={"E": E}, oc={"SOC": SOC})
    print(ledger.report(led))
    ok = ledger.passed(led)
"""
import numpy as np

import smew

# The eight pools with a balance of their own. Alkalinity is deliberately absent:
# biogeochem.py:400 DEFINES it from these, so it gets an identity check instead
# (see _alkalinity_identity).
ELEMENTS = ("Ca", "Mg", "K", "Na", "Al", "Si", "An", "C")

# column of min_st (the mineral formula written as coefficients) per element
MIN_ST_COL = {"Ca": 0, "Mg": 1, "K": 2, "Na": 3, "Al": 4, "Si": 5}

# Cations the model does not track, per formula unit, with their charge. The rock
# loses its full formula mass when it dissolves, but these never arrive in
# solution: Fe is hardcoded to 0 at biogeochem.py:59 and there is no Ti or P at
# all. Maintained by hand from the formula comments in constants.min_const --
# min_st itself has no column for them.
UNMODELLED = {
    # mineral        : {species: (stoichiometry, charge)}
    "augite":         {"Fe": (0.275, 2)},   # Mg0.45Fe0.275Ca0.275SiO3
    "olivine":        {"Fe": (0.2, 2)},     # Mg1.8Fe0.2SiO4
    "basalt_glass":   {"Fe": (0.19, 2),     # SiTi0.02Al0.36Fe0.19Mg0.28Ca0.26Na0.08K0.008O3.364
                       "Ti": (0.02, 4)},
    "hydroxyapatite": {"PO4": (3, -3),      # Ca5(OH)(PO4)3
                       "OH": (1, -1)},
}

# Closure gates. The reconstruction re-uses the same stored floats, so the only
# difference from the model's own arithmetic is the order of additions -- both
# residuals are therefore pure roundoff, and the thresholds are set from what
# roundoff can actually do rather than picked round.
#
# RTOL_STEP  per-step, normalised by the POOL as well as by the step's own
#            throughput. It has to include the pool because the storage change
#            is X[i] - X[last], a difference of two numbers ~1e5 times larger
#            than the change itself: that subtraction alone carries ~eps*pool of
#            roundoff, which is ~1e-11 of one step's throughput and has nothing
#            to do with whether the balance is right. Observed ~1e-16.
# RTOL_RUN   cumulative, normalised by throughput. Roundoff accumulates roughly
#            as sqrt(N)*eps, which for 5e4 steps is ~5e-14. Observed 1e-15 to
#            1e-17, so 1e-11 leaves room for longer runs and is still far below
#            any physically meaningful error.
RTOL_STEP = 1e-12
RTOL_RUN = 1e-11

# A pool whose entire activity sits below this fraction of the run's largest pool
# is at that pool's own roundoff and cannot be checked at all -- aluminium in a
# forsterite-only run, carbonate carbon with no carbonates. Reported as inactive
# rather than gated, so noise divided by noise is never called a failure.
ACTIVITY_FLOOR = 1e-16


# ---------------------------------------------------------------------------
# the freezing index mapping
# ---------------------------------------------------------------------------

def executed_steps(temp_soil):
    """Which steps the model actually integrated, and what each stepped FROM.

    Below zero the model stops the chemistry rather than integrating through it
    (biogeochem.py:370-379). The mechanism: on the last unfrozen step before a
    frost it records that index in idx_before_freezing; frozen steps are skipped
    entirely, so those array positions keep the zeros they were allocated with;
    on the first unfrozen step afterwards it steps from idx_before_freezing-1
    instead of from i-1.

    Three consequences the ledger has to respect:
      1. the pool is HELD -- state after thaw equals state before the freeze;
      2. the flux at idx_before_freezing-1 is consumed TWICE, once into the
         freeze and once out of it;
      3. the executed sequence is not contiguous and jumps BACKWARDS, so adding
         up per-step changes does not give final minus initial.

    (3) is why the gate is per-step. That statement holds however the indices
    jump, so freezing cannot raise a false alarm.

    Returns (steps, lasts): the integrated indices i, and their source index.
    """
    temp_soil = np.asarray(temp_soil, dtype=float)
    n = len(temp_soil)
    steps, lasts = [], []
    # same fallback as biogeochem.py:367 -- a run that starts frozen resumes
    # from the initial conditions
    idx_before_freezing = 1
    for i in range(1, n):
        if temp_soil[i] > 0:
            # the model mutates idx_before_freezing BEFORE reading it, so the
            # order of these two blocks matters
            if i < n - 1 and temp_soil[i + 1] < 0:
                idx_before_freezing = i
            last = idx_before_freezing - 1 if temp_soil[i - 1] < 0 else i - 1
            steps.append(i)
            lasts.append(last)
    return np.asarray(steps, dtype=np.int64), np.asarray(lasts, dtype=np.int64)


def _uptake(data, steps, lasts):
    """Recompute active plant uptake for every executed step.

    Why recompute rather than read data["UP_Ca"]: the model writes uptake at
    index `last`, not `i` (biogeochem.py:390). At a thaw step last is
    idx_before_freezing-1, so that slot is OVERWRITTEN with a value computed
    from the thaw step's biomass, and the value actually used one step before
    the freeze is gone. Reading the returned array would therefore reconstruct
    the wrong number at exactly the pre-freeze steps.

    smew.up_act is pure and deterministic, so recomputing reproduces it exactly
    -- and, being a second reading, it also checks the call site.
    """
    v, xi, Dw = data["v"], data["xi"], data["Dw"]
    T, dt, Zr = data["T"], data["dt"], data["Zr"]
    k_v, RAI, root_d = data["k_v"], data["RAI"], data["root_d"]
    Ca, Mg, K, Si = data["Ca"], data["Mg"], data["K"], data["Si"]

    up = np.zeros((len(steps), 4))  # [Ca, Mg, K, Si]
    for k, (i, j) in enumerate(zip(steps, lasts)):
        up[k] = smew.up_act(v[i], v[i] - v[j], xi, dt, T[j], Ca[j], Mg[j],
                            K[j], Si[j], Dw[j], Zr, k_v, RAI, root_d)
    return up


# ---------------------------------------------------------------------------
# the ledger itself
# ---------------------------------------------------------------------------

def build(data, water=None, oc=None):
    """Rebuild every balance in biogeochem.py:392-401 as named per-step series.

    data  : the dict biogeochem_balance returns
    water : {"E": E} -- bare evaporation, the one series the model never sees
            (it is not an argument to biogeochem_balance). Enables the water
            balance; omit to skip it.
    oc    : {"SOC": SOC} -- soil organic carbon stock. Enables the organic
            carbon balance; omit to skip it.

    Every term is in moles per square metre PER STEP (already multiplied by dt
    where the model multiplies by dt), so a term and a storage change are
    directly comparable.
    """
    d = data
    steps, lasts = executed_steps(d["temp_soil"])
    i, j = steps, lasts                      # short names, used heavily below
    dt, n_por, Zr = d["dt"], d["n"], d["Zr"]
    L, T, I = d["L"], d["T"], d["I"]
    min_st, EW = np.atleast_2d(d["min_st"]), np.atleast_2d(d["EW"])
    up = _uptake(d, steps, lasts)

    def mineral(el):
        """Element released by silicate dissolution, mol/m2 per step.

        EW[m, k] = Wr * SA * rock_f is moles of FORMULA UNITS of mineral m
        dissolving per day. min_st[m, col] is that formula written as
        coefficients [Ca, Mg, K, Na, Al, Si], so forsterite Mg2SiO4 is
        [0,2,0,0,0,1]: two magnesium and one silicon per unit dissolved.
        """
        return (min_st[:, MIN_ST_COL[el]] @ EW[:, j]) * dt

    # water carried out of the layer, per step, in litres per m2
    drain = L[j] * 1000 * dt                 # below the root zone
    transp = T[j] * 1000 * dt                # pulled through the plant
    # evaporation is deliberately absent: it removes pure water and leaves the
    # solutes behind

    el = {}

    # --- calcium ------------------------------------------------------------
    # W_CaCO3 is calcite dissolving: CaCO3 + CO2 + H2O -> Ca2+ + 2HCO3-. One
    # mole of calcium here and one mole of carbon in the carbon pool below. The
    # CO2 the reaction consumes needs no term of its own: it was already inside
    # the carbon pool. Only the carbon that came OUT OF THE MINERAL is new.
    el["Ca"] = _pool(
        d["Ca_tot"], i, j,
        sources={"background": d["I_Ca"] * dt * np.ones(len(i)),
                 "mineral": mineral("Ca"),
                 "carbonate": d["W_CaCO3"][j] * dt},
        sinks={"drainage": drain * d["Ca"][j],
               "transpiration": transp * d["Ca"][j],
               "uptake_active": up[:, 0] * dt})

    # --- magnesium ----------------------------------------------------------
    el["Mg"] = _pool(
        d["Mg_tot"], i, j,
        sources={"background": d["I_Mg"] * dt * np.ones(len(i)),
                 "mineral": mineral("Mg"),
                 "carbonate": d["W_MgCO3"][j] * dt},
        sinks={"drainage": drain * d["Mg"][j],
               "transpiration": transp * d["Mg"][j],
               "uptake_active": up[:, 1] * dt})

    # --- potassium ----------------------------------------------------------
    el["K"] = _pool(
        d["K_tot"], i, j,
        sources={"background": d["I_K"] * dt * np.ones(len(i)),
                 "mineral": mineral("K")},
        sinks={"drainage": drain * d["K"][j],
               "transpiration": transp * d["K"][j],
               "uptake_active": up[:, 2] * dt})

    # --- sodium -------------------------------------------------------------
    # no uptake row of either kind: plants do not require sodium
    el["Na"] = _pool(
        d["Na_tot"], i, j,
        sources={"background": d["I_Na"] * dt * np.ones(len(i)),
                 "mineral": mineral("Na")},
        sinks={"drainage": drain * d["Na"][j],
               "transpiration": transp * d["Na"][j]})

    # --- aluminium ----------------------------------------------------------
    # The ragged one (biogeochem.py:397). conv_Al because aluminium is carried in
    # different units from the other cations. No background input, no
    # transpiration, and drainage removes only Al + AlOH4 -- three of the five
    # dissolved species are missing. Quantified in diagnostics, not judged here.
    el["Al"] = _pool(
        d["Al_tot"], i, j,
        sources={"mineral": mineral("Al") * d["conv_Al"]},
        sinks={"drainage": drain * (d["Al"][j] + d["AlOH4"][j])})

    # --- silicon ------------------------------------------------------------
    el["Si"] = _pool(
        d["Si_tot"], i, j,
        sources={"background": d["I_Si"] * dt * np.ones(len(i)),
                 "mineral": mineral("Si")},
        sinks={"drainage": drain * d["Si"][j],
               "transpiration": transp * d["Si"][j],
               "uptake_active": up[:, 3] * dt})

    # --- lumped anion charge ------------------------------------------------
    # A bookkeeping device, in moles of CHARGE. At the start An[0] = 2Mg + 2Ca +
    # Na + K - Alk: whatever negative charge makes the solution electrically
    # neutral given the cations and the alkalinity implied by the starting pH.
    # Physically chloride, sulfate and nitrate; the model does not resolve them,
    # so it carries one number. It has no chemistry -- one input, one washout.
    # Its real job is setting pH: given total carbon and total cations, the size
    # of this pool fixes how much alkalinity is left over.
    el["An"] = _pool(
        d["An_tot"], i, j,
        sources={"background": d["I_An"] * dt * np.ones(len(i))},
        sinks={"drainage": drain * d["An"][j],
               "transpiration": transp * d["An"][j]})

    # --- inorganic carbon ---------------------------------------------------
    # IC_tot is dissolved CO2 + bicarbonate + carbonate in the water, PLUS CO2
    # in the air-filled pores.
    #
    # Two terms carry no dt, and that is correct. moisture.py:60 updates wetness
    # as s[i+1] = s[i] + rain[i+1]/(n*Zr) - (E+T+L)/(n*Zr)*dt: rain is added
    # without dt and the fluxes are multiplied by it, so rain -- and therefore
    # I = rain - Q -- is a DEPTH PER STEP in metres, while L and T are RATES in
    # m/d. ADV is likewise already a per-step amount, being a difference.
    #
    # ADV is labelled advection but is the displacement of soil gas by water:
    # drying opens air-filled pore space that fills with outside air (carbon in),
    # wetting squeezes that air out at the current soil concentration (carbon
    # out). Signed, so it appears as a negative sink when it is a source.
    el["C"] = _pool(
        d["IC_tot"], i, j,
        sources={"rain": I[i] * 1000 * d["DIC_rain"][i],
                 "respiration_het": d["r_het"][j] * dt,
                 "respiration_aut": d["r_aut"][j] * dt,
                 "carbonate": (d["W_CaCO3"][j] + d["W_MgCO3"][j]) * dt},
        sinks={"efflux": d["Fs"][j] * dt,
               "drainage": drain * d["DIC"][j],
               "gas_displacement": d["ADV"][i]})

    led = {"steps": steps, "lasts": lasts, "elements": el,
           "n_steps": len(d["temp_soil"]), "dt": dt,
           "rtol_step": RTOL_STEP, "rtol_run": RTOL_RUN}
    led["identities"] = _identities(d, i, j)
    led["charge"] = _charge(d, i, j, min_st, EW, drain, transp, up, dt)
    led["diagnostics"] = _diagnostics(d, i, j, min_st, EW, dt)
    if water is not None:
        led["water"] = _water(d, water)
    if oc is not None:
        led["organic_carbon"] = _organic_carbon(d, oc)
    return led


def _pool(tot, i, j, sources, sinks):
    """One pool: named per-step sources and sinks, and the storage change they
    are supposed to explain.

    Storage change is X[i] - X[last], NOT X[i] - X[i-1]: at a thaw step the
    model steps from before the freeze, so i-1 would be a skipped index still
    holding its allocated zero.
    """
    tot = np.asarray(tot, dtype=float)
    return {"sources": {k: np.asarray(v, dtype=float) for k, v in sources.items()},
            "sinks": {k: np.asarray(v, dtype=float) for k, v in sinks.items()},
            "delta": tot[i] - tot[j],
            "storage": tot}


# ---------------------------------------------------------------------------
# alkalinity, charge, diagnostics
# ---------------------------------------------------------------------------

def _identities(d, i, j):
    """Relations the code asserts by construction rather than by integrating.

    Alkalinity: biogeochem.py:400 is Alk_tot = 2Mg_tot + 2Ca_tot + Na_tot +
    K_tot - An_tot. That is the charge-balance definition -- alkalinity is the
    excess of conservative cation charge over conservative anion charge. Because
    it is DEFINED from pools that are integrated, it never gets a balance of its
    own; it is recomputed arithmetically each step. So it gets an exact identity
    check, at step 0 and at every executed step.
    """
    def alk_from_cations(k):
        return (2 * d["Mg_tot"][k] + 2 * d["Ca_tot"][k]
                + d["Na_tot"][k] + d["K_tot"][k] - d["An_tot"][k])

    res_run = d["Alk_tot"][i] - alk_from_cations(i)
    res_ic = float(d["Alk_tot"][0] - alk_from_cations(0))
    scale = max(float(np.max(np.abs(d["Alk_tot"][i]))) if len(i) else 0.0, 1e-300)
    return {"alkalinity_definition": {
        "max_abs": float(np.max(np.abs(res_run))) if len(i) else 0.0,
        "at_step_0": res_ic,
        "rel": max(float(np.max(np.abs(res_run))) if len(i) else 0.0,
                   abs(res_ic)) / scale}}


def _charge(d, i, j, min_st, EW, drain, transp, up, dt):
    """Where alkalinity comes from and goes, in moles of charge.

    This is the heart of the enhanced-weathering claim. Silicate dissolution
    consumes protons in proportion to the cation charge released, and because
    those protons come from carbonic acid, consuming them converts CO2 into
    bicarbonate. In the code it happens implicitly: released cations raise
    Alk_tot at line 400, and the implicit solver then redistributes carbon among
    dissolved CO2, bicarbonate and carbonate. Nothing names the step.

    Arithmetically this is a linear combination of the element balances above,
    so it cannot fail independently -- it is here because it is the row a reader
    actually wants, and because it exposes two things the element rows hide:

      * aluminium charge, which the alkalinity definition excludes entirely;
      * cations the model drops at dissolution (see UNMODELLED).
    """
    cat_charge = 2 * min_st[:, 0] + 2 * min_st[:, 1] + min_st[:, 2] + min_st[:, 3]
    # Al charge released, in the SAME units as everything else here (i.e. NOT
    # scaled by conv_Al, unlike the aluminium pool itself) so it is comparable
    al_charge = 3 * min_st[:, 4]

    conc = d["Ca"], d["Mg"], d["K"], d["Na"]
    leached = (drain + transp) * (2 * conc[0][j] + 2 * conc[1][j]
                                 + conc[2][j] + conc[3][j])

    rows = {
        "from_minerals": (cat_charge @ EW[:, j]) * dt,
        "from_carbonates": 2 * (d["W_CaCO3"][j] + d["W_MgCO3"][j]) * dt,
        "from_background": (2 * d["I_Ca"] + 2 * d["I_Mg"] + d["I_K"]
                            + d["I_Na"] - d["I_An"]) * dt * np.ones(len(i)),
        "lost_cations_leached": leached,
        "lost_uptake_active": (2 * up[:, 0] + 2 * up[:, 1] + up[:, 2]) * dt,
        "regained_anions_leached": (drain + transp) * d["An"][j],
    }
    delta = d["Alk_tot"][i] - d["Alk_tot"][j]
    net = (rows["from_minerals"] + rows["from_carbonates"] + rows["from_background"]
           - rows["lost_cations_leached"] - rows["lost_uptake_active"]
           + rows["regained_anions_leached"])

    out = {"rows": {k: float(np.sum(v)) for k, v in rows.items()},
           "delta_alkalinity": float(np.sum(delta)),
           "residual": float(np.max(np.abs(net - delta))) if len(i) else 0.0}

    # charge released as aluminium, which generates no alkalinity here
    out["aluminium_charge_excluded"] = float(np.sum((al_charge @ EW[:, j]) * dt))

    # cations the rock loses but the solution never receives
    dropped_mol, dropped_chg = {}, 0.0
    minerals = list(d.get("mineral", []) or [])
    for m_idx, name in enumerate(minerals):
        # with M_rock_in == 0 the model allocates EW as a single zero row
        # (biogeochem.py:128) while `mineral` may still name several minerals
        if m_idx >= EW.shape[0]:
            break
        for sp, (st, chg) in UNMODELLED.get(str(name), {}).items():
            amount = float(np.sum(EW[m_idx, j] * dt) * st)
            dropped_mol[f"{name}.{sp}"] = amount
            dropped_chg += amount * chg
    out["dropped_species_mol"] = dropped_mol
    out["dropped_charge"] = dropped_chg
    return out


def _diagnostics(d, i, j, min_st, EW, dt):
    """Numbers that are NOT gated, because they measure approximations the model
    makes on purpose, or gaps that are a modelling question rather than a bug.
    """
    dg = {}
    temp = np.asarray(d["temp_soil"], dtype=float)
    dt_, L, T = d["dt"], d["L"], d["T"]

    # --- freezing -----------------------------------------------------------
    frozen = np.where(temp[1:] <= 0)[0] + 1
    dg["n_frozen_steps"] = int(len(frozen))
    dg["frozen_fraction"] = float(len(frozen) / max(len(temp) - 1, 1))
    # Each thaw step is where the mapping jumps backwards. There j = h - 1, so
    # h = j + 1 is the index whose state the pool is held at -- read off the
    # mapping itself rather than re-derived from temperature.
    thaws = np.where(j != i - 1)[0]
    dg["n_freeze_intervals"] = int(len(thaws))

    # What the skipped steps WOULD have moved had they been integrated: the size
    # of the approximation. Water fluxes are computed straight through frost
    # (moisture_balance knows nothing about freezing), so leaching uses the real
    # L and T over the interval at the HELD concentration -- concentrations at
    # frozen indices are the allocated zeros and cannot be used. Dissolution
    # likewise uses the held rate. Respiration needs no row: f_T is clipped at
    # zero (organic_carbon.py:41), so decomposition genuinely stops.
    # Each interval is the indices strictly between the previous executed step
    # and the thaw, and the state to evaluate it at is j -- the same index the
    # model's own balance reads (Ca[last] etc). j is always a valid state: an
    # unfrozen step, or 0 when the run STARTS frozen and resumes from the
    # initial conditions. Using held = j+1 instead would land on a skipped index
    # holding its allocated zero in exactly that start-frozen case.
    unacc = {}
    dg["frozen_steps_covered"] = 0
    if len(thaws):
        spans = [(int(i[k - 1]) + 1 if k > 0 else 1, int(i[k]), int(j[k]))
                 for k in thaws]
        for name, col, conc in (("Ca", 0, d["Ca"]), ("Mg", 1, d["Mg"]),
                                ("K", 2, d["K"]), ("Na", 3, d["Na"]),
                                ("Si", 5, d["Si"])):
            leach = rel = 0.0
            for f0, f1, h in spans:
                leach += float(np.sum((L[f0:f1] + T[f0:f1]) * 1000 * dt_)) * float(conc[h])
                rel += float(min_st[:, col] @ EW[:, h]) * dt_ * (f1 - f0)
            unacc[name] = {"leached_if_run": leach, "released_if_run": rel}
        c_leach = c_eff = 0.0
        for f0, f1, h in spans:
            c_leach += float(np.sum(L[f0:f1] * 1000 * dt_)) * float(d["DIC"][h])
            c_eff += float(d["Fs"][h]) * dt_ * (f1 - f0)
        unacc["C"] = {"drained_if_run": c_leach, "effluxed_if_run": c_eff}
        dg["frozen_steps_covered"] = int(sum(f1 - f0 for f0, f1, _ in spans))
    dg["frozen_flux_unaccounted"] = unacc

    # --- pools going negative ----------------------------------------------
    # the balances are explicit Euler with losses at the previous step and no
    # positivity clamp, so if dt x loss exceeds the pool it goes negative and
    # the implicit solve receives a negative total
    pools = {name: np.asarray(d[name], dtype=float)
             for name in ("Ca_tot", "Mg_tot", "K_tot", "Na_tot", "Al_tot",
                          "Si_tot", "An_tot", "IC_tot")}
    run_scale = max(float(np.max(np.abs(v))) for v in pools.values())
    neg = {}
    for name, tot in pools.items():
        # a pool that is zero plus roundoff reports spurious negatives -- the
        # same activity floor the closure gate uses
        if float(np.max(np.abs(tot))) < ACTIVITY_FLOOR * run_scale:
            continue
        seen = tot[i]
        bad = seen < 0
        if bad.any():
            neg[name] = {"n_steps": int(bad.sum()), "min": float(seen.min()),
                         "first_step": int(i[np.argmax(bad)])}
    dg["negative_pools"] = neg

    # --- aluminium species missing from drainage ---------------------------
    # biogeochem.py:397 removes Al + AlOH4 only
    missing = d["L"][j] * 1000 * dt * (d["AlOH"][j] + d["AlOH2"][j] + d["AlOH3"][j])
    removed = d["L"][j] * 1000 * dt * (d["Al"][j] + d["AlOH4"][j])
    dg["aluminium_drainage_omitted"] = {
        "mol_omitted": float(np.sum(missing)) * d["conv_Al"],
        "mol_removed": float(np.sum(removed)) * d["conv_Al"],
        "no_transpiration_term_mol": float(np.sum(T[j] * 1000 * dt * d["Al_w"][j])) * d["conv_Al"],
    }

    # --- carbon leaves by drainage, alkalinity leaves by drainage AND
    #     transpiration ----------------------------------------------------
    dg["carbon_transpiration_omitted_mol"] = float(np.sum(T[j] * 1000 * dt * d["DIC"][j]))

    # --- how much of the carbon input is the rain term ----------------------
    # moisture.py assigns I two different meanings: `I = rain - Q` in the
    # dynamic branch is a depth per step, but `I = E+T+L` in the constant-
    # moisture branch (moisture.py:88) is a RATE. biogeochem.py:401 consumes
    # I[i] without dt, so in constant-moisture runs this term is larger than the
    # depth interpretation by 1/dt. Reported as a share of total carbon input so
    # the consequence is visible either way.
    rain_c = float(np.sum(d["I"][i] * 1000 * d["DIC_rain"][i]))
    other_c = float(np.sum((d["r_het"][j] + d["r_aut"][j]
                            + d["W_CaCO3"][j] + d["W_MgCO3"][j]) * dt))
    dg["carbon_input_share"] = {
        "rain_mol": rain_c,
        "respiration_and_carbonate_mol": other_c,
        "rain_fraction": rain_c / max(abs(rain_c) + abs(other_c), 1e-300),
        "rain_mol_if_dt_applied": rain_c * dt,
    }
    return dg


def _water(d, water):
    """The water balance, and the ceiling that quietly discards water.

    moisture.py:63-69 runs two clamps in sequence. If wetness would exceed 1 the
    excess is recorded as runoff Q and wetness set to 1 -- properly accounted,
    and already inside I = rain - Q. Then, unconditionally,

        if s[i+1] >= 0.98: s[i+1] = 0.98

    with no record anywhere. Every solute concentration is a pool divided by a
    water volume, so water removed without a record concentrates everything left
    behind.

    The identity, from moisture.py:60 with I = rain - Q:

        (s[i+1] - s[i]) * n * Zr = I[i+1] - (E[i] + T[i] + L[i]) * dt - ceiling

    so the residual IS the discarded water. The test is therefore not that the
    residual is zero, but that it is non-negative and non-zero ONLY at steps
    where the ceiling actually fired.
    """
    s = np.asarray(d["s"], dtype=float)
    E = np.asarray(water["E"], dtype=float)
    if len(E) != len(s):
        return {"skipped": f"E has length {len(E)}, s has {len(s)}"}
    if np.allclose(s, s[0]):
        return {"skipped": "constant-moisture run (keyword_wb=0): "
                           "no water balance is integrated"}

    nZr, dt = d["n"] * d["Zr"], d["dt"]
    L, T = np.asarray(d["L"], dtype=float), np.asarray(d["T"], dtype=float)
    I = np.asarray(d["I"], dtype=float)

    k = np.arange(0, len(s) - 1)
    residual = (I[k + 1] - (E[k] + T[k] + L[k]) * dt) - (s[k + 1] - s[k]) * nZr
    at_ceiling = np.isclose(s[k + 1], 0.98, rtol=0, atol=1e-15)

    scale = max(float(np.max(np.abs(I))), 1e-300)
    off = residual[~at_ceiling]
    return {
        "n_steps": len(k),
        "n_at_ceiling": int(at_ceiling.sum()),
        "ceiling_water_m": float(np.sum(residual[at_ceiling])),
        "ceiling_water_mm_per_event": (float(np.mean(residual[at_ceiling])) * 1000
                                       if at_ceiling.any() else 0.0),
        "unexplained_max_m": float(np.max(np.abs(off))) if off.size else 0.0,
        "unexplained_rel": (float(np.max(np.abs(off))) / scale) if off.size else 0.0,
        "any_negative_ceiling": bool(np.any(residual[at_ceiling] < -1e-15)),
    }


def _organic_carbon(d, oc):
    """The soil organic carbon balance.

    organic_carbon.py:117 steps the stock as
        SOC[i] = SOC[i-1] + (ADD/Zr - r*DEC[i-1])*dt
    and reports respiration as r_het = r*DEC*Zr/MM_C. Substituting,

        (SOC[i] - SOC[i-1]) * Zr = (ADD - r_het[i-1]*MM_C) * dt

    which needs no knowledge of ADD, r or DEC -- rearranging gives ADD back, and
    ADD is a constant, so checking that the implied value does not vary is a
    STRONGER test than comparing against a number we were handed.

    Two things worth stating so nobody later "fixes" them:
      * DEC is a GROSS decomposition rate, of which only r = 0.7 is realised as
        loss. The other 30% is not a leak; it is simply never subtracted,
        because DEC is not itself a flux out of the pool.
      * r_aut (root respiration) is a carbon source to the soil air with NO soil
        pool behind it -- that carbon comes from the plant, which the model does
        not track. So it is legitimately external, and soil carbon plus gas
        carbon cannot be closed as one system.
    """
    SOC = np.asarray(oc["SOC"], dtype=float)
    r_het = np.asarray(d["r_het"], dtype=float)
    if len(SOC) != len(r_het):
        return {"skipped": f"SOC has length {len(SOC)}, r_het has {len(r_het)}"}
    Zr, dt = d["Zr"], d["dt"]
    MM_C = smew.MM(d["conv_mol"])[5]

    k = np.arange(1, len(SOC))
    stock_step = (SOC[k] - SOC[k - 1]) * Zr          # gC/m2 gained this step
    resp_step = r_het[k - 1] * MM_C * dt             # gC/m2 respired this step
    add_step = stock_step + resp_step                # = ADD*dt if the balance holds

    # Two things must be in the denominator, for two different reasons.
    #
    # The fluxes, because all four comparison notebooks set ADD = 0: add_step is
    # then zero plus roundoff, and judging its spread against its own size
    # divides noise by noise -- the same trap as an inactive pool.
    #
    # The STOCK, because stock_step is SOC[i] - SOC[i-1], a difference of two
    # numbers ~5e5 times larger than the difference itself. That subtraction
    # carries eps*stock of roundoff whatever the balance does. Measured on
    # Vials_Dietzen: worst add_step 1.45e-12 against eps*stock = 2.93e-12, i.e.
    # below one unit in the last place of the stock, yet 1.0e-10 of a step's
    # respiration. Exactly the same arithmetic as the pool differences in
    # closure(), in a second place.
    stock_scale = float(np.max(np.abs(SOC))) * Zr
    flux_scale = max(float(np.max(np.abs(resp_step))),
                     float(np.max(np.abs(stock_step))), 1e-300)
    scale = max(flux_scale, stock_scale, 1e-300)
    spread = float(np.max(add_step) - np.min(add_step))
    return {
        "add_implied_g_m2_d": float(np.mean(add_step)) / dt,
        "add_implied_spread_g_m2_d": spread / dt,
        "add_implied_rel": spread / scale,
        # how many times larger the stock is than one step's respiration: the
        # sharpness limit of this check, same meaning as `resolution` in closure()
        "resolution": stock_scale / flux_scale,
        "stock_change_gC_m2": float((SOC[-1] - SOC[0]) * Zr),
        "respired_gC_m2": float(np.sum(resp_step)),
        "root_respiration_no_pool_gC_m2": float(np.sum(d["r_aut"][:-1]) * MM_C * dt),
    }


# ---------------------------------------------------------------------------
# closure
# ---------------------------------------------------------------------------

def closure(led):
    """Per-element residuals, gated per-step and cumulatively.

    Normalisation matters more than the tolerance, and it is different for the
    two gates.

    Cumulative: over a long run, drainage can exceed the pool many times over --
    the pool turns over. Dividing by pool size would read as failure caused by
    cancellation between two large numbers rather than by any error. So the
    denominator is actual throughput.

    Per-step: the storage change is X[i] - X[last], and the pool is ~1e5 times
    larger than what moves in one step, so that subtraction carries ~eps*pool of
    roundoff all on its own. Measured on the example case, the worst per-step
    residual is 4.7e-10 against eps*pool = 1.3e-9 -- below one unit in the last
    place of the pool. Normalising by throughput alone would therefore report a
    floor of ~1e-11 that is a property of the arithmetic, not of the balance. The
    denominator includes the pool.

    A consequence worth knowing: because the pool dwarfs the per-step change, a
    systematic per-step error smaller than ~1e-11 of a step's throughput is
    invisible to the per-step gate. It is not invisible to the cumulative one,
    which is why both are checked. `resolution` records that ratio per element.
    """
    scales, per = {}, {}
    for name in ELEMENTS:
        p = led["elements"][name]
        src = sum(p["sources"].values()) if p["sources"] else np.zeros(len(led["steps"]))
        snk = sum(p["sinks"].values()) if p["sinks"] else np.zeros(len(led["steps"]))
        per[name] = (src, snk, src - snk - p["delta"])
        scales[name] = max(float(np.sum(np.abs(src))), float(np.sum(np.abs(snk))),
                           abs(float(np.sum(p["delta"]))),
                           float(np.max(np.abs(p["storage"]))), 0.0)
    run_scale = max(scales.values()) if scales else 0.0

    rows = []
    for name in ELEMENTS:
        p = led["elements"][name]
        src, snk, step_res = per[name]
        tot_src, tot_snk = float(np.sum(src)), float(np.sum(snk))
        tot_del = float(np.sum(p["delta"]))
        residual = tot_src - tot_snk - tot_del
        scale = max(scales[name], 1e-300)
        inactive = scales[name] < ACTIVITY_FLOOR * run_scale

        # per-step denominator: the step's own throughput OR the pool it is
        # differenced from, whichever limits resolution
        pool = np.maximum(np.abs(p["storage"][led["steps"]]),
                          np.abs(p["storage"][led["lasts"]]))
        step_scale = np.maximum(np.maximum(np.abs(src), np.abs(snk)), pool)
        step_scale = np.maximum(step_scale, scale * 1e-16)
        thru = np.maximum(np.maximum(np.abs(src), np.abs(snk)), 1e-300)

        has = len(step_res) > 0
        step_rel = float(np.max(np.abs(step_res) / step_scale)) if has else 0.0
        rows.append({
            "element": name,
            "sources": tot_src,
            "sinks": tot_snk,
            "delta": tot_del,
            "residual": residual,
            "rel": abs(residual) / scale,
            "step_max_abs": float(np.max(np.abs(step_res))) if has else 0.0,
            "step_rel": step_rel,
            "worst_step": int(led["steps"][np.argmax(np.abs(step_res))]) if has else -1,
            # how many times larger the pool is than one step's throughput: the
            # sharpness limit of the per-step gate
            "resolution": float(np.median(pool / thru)) if has else 0.0,
            "inactive": bool(inactive),
            "ok": bool(inactive or (step_rel <= led["rtol_step"]
                                    and abs(residual) / scale <= led["rtol_run"])),
        })
    return rows


def failures(led, rows=None):
    """Every gated check that did not clear its threshold, as readable reasons."""
    rows = rows if rows is not None else closure(led)
    out = []
    for r in rows:
        if r["inactive"] or r["ok"]:
            continue
        why = []
        if r["step_rel"] > led["rtol_step"]:
            why.append(f"per-step {r['step_rel']:.2e} at step {r['worst_step']}")
        if r["rel"] > led["rtol_run"]:
            why.append(f"cumulative {r['rel']:.2e}")
        out.append(f"{r['element']}: " + ", ".join(why))

    ai = led["identities"]["alkalinity_definition"]
    if ai["rel"] > led["rtol_step"]:
        out.append(f"alkalinity definition: {ai['rel']:.2e} relative")

    w = led.get("water", {})
    if w and "skipped" not in w:
        if w["unexplained_rel"] > 1e-10:
            out.append(f"water: unexplained {w['unexplained_rel']:.2e} relative")
        if w["any_negative_ceiling"]:
            out.append("water: the 0.98 ceiling ADDED water at some step")

    o = led.get("organic_carbon", {})
    if o and "skipped" not in o:
        if o["add_implied_rel"] > 1e-10:
            out.append(f"organic carbon: implied input varies by "
                       f"{o['add_implied_rel']:.2e} relative")
    return out


def passed(led, rows=None):
    """True when every gated check clears its threshold."""
    return not failures(led, rows)


def report(led, terms=True):
    """Human-readable ledger."""
    rows = closure(led)
    dg = led["diagnostics"]
    out = []

    out.append(f"executed {len(led['steps'])} of {led['n_steps']-1} steps "
               f"({dg['n_frozen_steps']} frozen, {dg['frozen_fraction']*100:.1f}%, "
               f"{dg['n_freeze_intervals']} freeze/thaw transitions)")
    out.append("")
    out.append(f"{'element':>8} {'sum sources':>14} {'sum sinks':>14} "
               f"{'d storage':>14} {'run rel':>10} {'step rel':>10} {'pool/step':>10}")
    for r in rows:
        tag = ("  inactive" if r["inactive"]
               else ("" if r["ok"] else "  <-- FAIL"))
        out.append(f"{r['element']:>8} {r['sources']:>14.6e} {r['sinks']:>14.6e} "
                   f"{r['delta']:>14.6e} {r['rel']:>10.2e} "
                   f"{r['step_rel']:>10.2e} {r['resolution']:>10.1e}{tag}")
    out.append(f"{'':>8} gates: run <= {led['rtol_run']:.0e}, "
               f"per-step <= {led['rtol_step']:.0e}")

    if terms:
        out.append("")
        out.append("named terms (mol/m2 over the run; C in mol-conv)")
        for name in ELEMENTS:
            p = led["elements"][name]
            if not p["sources"] and not p["sinks"]:
                continue
            out.append(f"  {name}")
            for k, v in p["sources"].items():
                out.append(f"      + {k:<22s} {float(np.sum(v)):>14.6e}")
            for k, v in p["sinks"].items():
                out.append(f"      - {k:<22s} {float(np.sum(v)):>14.6e}")

    ai = led["identities"]["alkalinity_definition"]
    out.append("")
    out.append(f"alkalinity definition (biogeochem.py:400) holds to "
               f"{ai['rel']:.2e} relative (step 0: {ai['at_step_0']:.3e})")

    c = led["charge"]
    out.append("")
    out.append("charge / alkalinity generation [mol_c/m2]")
    for k, v in c["rows"].items():
        out.append(f"      {k:<28s} {v:>14.6e}")
    out.append(f"      {'= d alkalinity':<28s} {c['delta_alkalinity']:>14.6e} "
               f"(residual {c['residual']:.2e})")
    out.append(f"      {'aluminium charge excluded':<28s} "
               f"{c['aluminium_charge_excluded']:>14.6e}")
    if c["dropped_species_mol"]:
        out.append(f"      {'cations dropped at dissolution':<28s} "
                   f"{c['dropped_charge']:>14.6e} mol_c")
        for k, v in c["dropped_species_mol"].items():
            out.append(f"          {k:<24s} {v:>14.6e} mol")

    out.append("")
    out.append("diagnostics (not gated)")
    if dg["negative_pools"]:
        for k, v in dg["negative_pools"].items():
            out.append(f"      NEGATIVE POOL {k:<12s} {v['n_steps']} steps, "
                       f"min {v['min']:.3e}, first at step {v['first_step']}")
    else:
        out.append("      no pool went negative")
    al = dg["aluminium_drainage_omitted"]
    out.append(f"      Al drained {al['mol_removed']:.3e}, "
               f"omitted species {al['mol_omitted']:.3e}, "
               f"no transpiration term {al['no_transpiration_term_mol']:.3e} mol")
    out.append(f"      carbon not removed by transpiration "
               f"{dg['carbon_transpiration_omitted_mol']:.3e} mol")
    ci = dg["carbon_input_share"]
    out.append(f"      carbon input: rain {ci['rain_mol']:.3e} "
               f"({ci['rain_fraction']*100:.1f}%), "
               f"respiration+carbonate {ci['respiration_and_carbonate_mol']:.3e}; "
               f"rain x dt would be {ci['rain_mol_if_dt_applied']:.3e}")
    if dg["frozen_flux_unaccounted"]:
        out.append(f"      flux the frozen steps skipped, over the "
                   f"{dg['frozen_steps_covered']} of {dg['n_frozen_steps']} frozen "
                   f"steps that end in a thaw (held state x the water flux that "
                   f"still ran):")
        for k, v in dg["frozen_flux_unaccounted"].items():
            terms = "  ".join(f"{kk} {vv:.4e}" for kk, vv in v.items())
            out.append(f"          {k:<4s} {terms}")

    w = led.get("water")
    if w:
        out.append("")
        if "skipped" in w:
            out.append(f"water balance: skipped -- {w['skipped']}")
        else:
            out.append(f"water balance: {w['n_at_ceiling']} of {w['n_steps']} steps hit "
                       f"the 0.98 ceiling, discarding {w['ceiling_water_m']*1000:.2f} mm "
                       f"total ({w['ceiling_water_mm_per_event']:.2f} mm per event); "
                       f"unexplained {w['unexplained_rel']:.2e} relative")

    o = led.get("organic_carbon")
    if o:
        out.append("")
        if "skipped" in o:
            out.append(f"organic carbon: skipped -- {o['skipped']}")
        else:
            out.append(f"organic carbon: implied input "
                       f"{o['add_implied_g_m2_d']:.6g} g/m2/d, constant to "
                       f"{o['add_implied_rel']:.2e}; stock change "
                       f"{o['stock_change_gC_m2']:.4g} gC/m2, respired "
                       f"{o['respired_gC_m2']:.4g}, root respiration with no pool "
                       f"{o['root_respiration_no_pool_gC_m2']:.4g} "
                       f"(stock/step {o['resolution']:.1e})")

    out.append("")
    out.append("CLOSED" if passed(led, rows) else "NOT CLOSED")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# proving the ledger can actually fail
# ---------------------------------------------------------------------------

def self_test(led, factor=1.001):
    """Perturb each term in turn and check that closure breaks.

    A ledger that closes proves nothing on its own: it might close because a row
    is quietly zero, or because two mistakes cancel. So multiply the LEDGER's
    copy of one term by a small factor and require the gate to fail. Any term
    that can be perturbed without breaking closure is not doing its job.

    Same trick harness.py already uses with --diss-f to prove the frozen files
    detect change.

    Returns (survivors, inactive): terms that failed to break closure, and terms
    that are identically zero in this case and therefore legitimately cannot be
    detected (mineral release with no rock applied, carbonate weathering with no
    carbonates, and so on).
    """
    survivors, inactive = [], []
    # a pool the gate skips as inactive cannot have its terms detected either,
    # by construction -- report those as inactive rather than as holes
    skip = {r["element"] for r in closure(led) if r["inactive"]}
    for name in ELEMENTS:
        pool = led["elements"][name]
        for kind in ("sources", "sinks"):
            for term in list(pool[kind]):
                label = f"{name}.{kind}.{term}"
                original = pool[kind][term]
                if name in skip or not np.any(original):
                    inactive.append(label)
                    continue
                pool[kind][term] = original * factor
                try:
                    still_ok = all(r["ok"] for r in closure(led)
                                   if r["element"] == name)
                finally:
                    pool[kind][term] = original
                if still_ok:
                    survivors.append(label)
    return survivors, inactive
