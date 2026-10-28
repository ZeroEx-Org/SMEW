# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""S5: the CDR ledger -- where the alkalinity went, and what that is worth in CO2.

WHAT THIS IS FOR
----------------
SMEW reports alkalinity and pH. An enhanced-weathering claim is a carbon number,
and between the two sits a budget nobody had written down: of the alkalinity the
rock generated, how much left in drainage, how much the exchanger is holding,
how much went into secondary carbonates, how much the plants took. This module
writes that budget, per step and cumulatively, from a biogeochem_balance result.

It is built the way smew.ledger is, and for the same reason: rebuilt from the
outside, on the steps the model actually executed (ledger.executed_steps), so it
is a second reading of the scheme, not the scheme adding up its own fluxes. Its
rows are charge-weighted combinations of the ledger's element rows, so it closes
on the change in Alk_tot to roundoff -- and that closure is gated.

THE ALKALINITY BUDGET  [eq per m2]
---------------------------------
Alk_tot = 2 Ca_tot + 2 Mg_tot + K_tot + Na_tot - An_tot (+ 2 Fe_tot under
fe_mode="reduced"): the conservative-ion definition the model carries.

    sources   silicates       cation charge released by dissolving minerals
                              (+ 2 per Fe under "reduced"; 0 otherwise -- P3)
              carbonates_dis  carbonate solids dissolving (2 eq per CO3)
              background      keyword_add's replacement inputs, net of anions
              clipping        F2's retained mass (zero on every benchmark)
    sinks     carbonates_ppt  carbonate solids forming (2 eq per CO3)
              secondary_other cations taken by a non-carbonate secondary phase
                              (none in the current table; Al, Si and Fe phases
                              take no alkalinity -- S0.2)
              drainage        leaving below the root zone: EXPORTED
              transpiration   solutes carried into the plant with water
              uptake          active nutrient uptake
    storage   aqueous         Alk in the soil water
              exchanger       base cations on the clay (Alk_tot - aqueous)

THE CARBON NUMBER  [mol CO2 per m2]
-----------------------------------
    co2_removed = drainage alkalinity exported  (1 CO2 per eq, as HCO3-)
                + carbonate C formed            (1 C per mol CO3 precipitated)
                - carbonate C dissolved         (inventory: liming, etc.)

Each term is a carbon balance, which is why they add. Silicate alkalinity
exported as bicarbonate holds one CO2 per equivalent. Silicate alkalinity that
ends in a carbonate holds one C per two equivalents -- the other CO2 is released
on precipitation -- and is counted by the carbonate C term. A carbonate that
dissolves and is exported gives 2 eq for 1 C of its own, so it is worth one CO2
net: +2 exported, -1 dissolved. Dissolve-and-reprecipitate nets zero. So the
number is right whatever mixture of silicate and carbonate the alkalinity came
from, without attributing it.

What it is NOT, stated so it is not mistaken for more:
  * Ocean storage efficiency: exported alkalinity is counted at 1 CO2/eq.
    `eta` scales it if a downstream efficiency is wanted; default 1.
  * A treatment effect: background inputs and initial carbonate inventories
    contribute too. The CDR of an application is `compare(treated, control)`.
  * Alkalinity still in the soil is not in co2_removed: it has removed nothing
    until it leaves. That makes co2_removed TRANSIENTLY NEGATIVE after liming --
    the carbonate's own carbon is subtracted when it dissolves, its alkalinity
    only counted when it drains (+1 CO2 per 2 eq once it has). So a second
    number is reported beside it:
        co2_potential = co2_removed + alkalinity still in the soil
    -- the value if what the soil holds now is eventually exported. The
    long-run claim lies between the two; their gap is the soil's storage.

USAGE
-----
    from smew import cdr
    c = cdr.build(data)
    print(cdr.report(c))
    delta = cdr.compare(cdr.build(treated), cdr.build(control))
"""
import numpy as np

from smew import ledger

RTOL_STEP = 1e-12        # per-step closure, as smew.ledger gates it


def build(data, eta=1.0):
    """The alkalinity budget and the carbon number of one biogeochem_balance run.

    Every series is per executed step [eq or mol per m2, i.e. already / conv_mol];
    `totals` holds their sums. `eta` scales exported alkalinity into CO2.
    """
    d = data
    cm = float(d["conv_mol"])
    steps, lasts = ledger.executed_steps(d["temp_soil"])
    i, j = steps, lasts
    dt = d["dt"]
    L, T = d["L"], d["T"]
    min_st, EW = np.atleast_2d(d["min_st"]), np.atleast_2d(d["EW"])
    up = ledger._uptake(d, steps, lasts)
    reduced = d.get("fe_mode", "tracer") == "reduced"
    fe = 2.0 if reduced else 0.0

    drain = L[j] * 1000 * dt                     # [l per m2 per step]
    transp = T[j] * 1000 * dt

    def alk_conc(k, with_fe=True):
        """Conservative alkalinity of the soil water [eq-conv per l]. Fe+2 counts
        under "reduced" -- but leaves by drainage only, as the model's Fe
        balance has it (no transpiration term), hence `with_fe`."""
        a = 2 * d["Ca"][k] + 2 * d["Mg"][k] + d["K"][k] + d["Na"][k] - d["An"][k]
        return a + fe * d["Fe"][k] if (reduced and with_fe) else a

    def clip(name):
        x = d.get("clip_" + name)
        return np.zeros(len(i)) if x is None else np.asarray(x, dtype=float)[i]

    cat_charge = 2 * min_st[:, 0] + 2 * min_st[:, 1] + min_st[:, 2] + min_st[:, 3]
    cat_charge = cat_charge + fe * min_st[:, 6]

    # carbonates and other secondary phases, signed: W > 0 dissolution
    sec_names = tuple(d.get("sec_phases", ()))
    carb_W = np.zeros(len(i))                   # [eq-conv per step], signed
    carb_C = np.zeros(len(i))                   # [mol-conv C per step], signed
    other = np.zeros(len(i))
    if d.get("carb_scheme", "legacy") == "legacy":
        w = (np.asarray(d["W_CaCO3"])[j] + np.asarray(d["W_MgCO3"])[j]) * dt
        carb_W += 2 * w
        carb_C += w
    if sec_names:
        sec_st = np.asarray(d["sec_st"], dtype=float)
        sec_C = np.asarray(d.get("sec_C", np.zeros(len(sec_names))), dtype=float)
        W = np.asarray(d["W_sec"], dtype=float)[:, j] * dt
        chg = 2 * sec_st[:, 0] + 2 * sec_st[:, 1] + sec_st[:, 2] + sec_st[:, 3]
        for m in range(len(sec_names)):
            if sec_C[m] > 0:
                carb_W += chg[m] * W[m]
                carb_C += sec_C[m] * W[m]
            elif chg[m] > 0:
                other += chg[m] * W[m]

    rows = {
        "silicates": (cat_charge @ EW[:, j]) * dt,
        "carbonates_dis": np.maximum(carb_W, 0.0),
        "background": (2 * d["I_Ca"] + 2 * d["I_Mg"] + d["I_K"] + d["I_Na"]
                       - d["I_An"]) * dt * np.ones(len(i)),
        "clipping": (2 * clip("Ca") + 2 * clip("Mg") + clip("K") + clip("Na")
                     - clip("An") + (fe * clip("Fe") if reduced else 0.0)),
        "carbonates_ppt": np.maximum(-carb_W, 0.0),
        "secondary_other": -other,
        "drainage": drain * alk_conc(j),
        "transpiration": transp * alk_conc(j, with_fe=False),
        "uptake": (2 * up[:, 0] + 2 * up[:, 1] + up[:, 2]) * dt,
    }
    SOURCES = ("silicates", "carbonates_dis", "background", "clipping")
    SINKS = ("carbonates_ppt", "secondary_other", "drainage", "transpiration", "uptake")
    delta = np.asarray(d["Alk_tot"])[i] - np.asarray(d["Alk_tot"])[j]
    resid = sum(rows[k] for k in SOURCES) - sum(rows[k] for k in SINKS) - delta
    # as smew.ledger gates it: the step's throughput OR the pool it is
    # differenced from, whichever limits resolution -- Alk_tot[i] - Alk_tot[j]
    # carries ~eps * Alk_tot of roundoff all on its own
    scale = np.maximum.reduce([np.abs(rows[k]) for k in rows] +
                              [np.abs(np.asarray(d["Alk_tot"])[i]),
                               np.abs(np.asarray(d["Alk_tot"])[j])])
    step_rel = float(np.max(np.abs(resid) / np.maximum(scale, 1e-300))) if len(i) else 0.0

    # storage split at the start and end of the run
    nZrs = np.asarray(d["n"]) * np.asarray(d["Zr"]) * np.asarray(d["s"]) * 1000
    aq = np.asarray(d["Alk"]) * nZrs
    k0, k1 = 0, int(steps[-1]) if len(steps) else 0
    stored = {"aqueous": float(aq[k1] - aq[k0]) / cm,
              "exchanger": float((d["Alk_tot"][k1] - aq[k1]) - (d["Alk_tot"][k0] - aq[k0])) / cm}

    tot = {k: float(np.sum(v)) / cm for k, v in rows.items()}
    c_formed = float(np.sum(np.maximum(-carb_C, 0.0))) / cm
    c_dissolved = float(np.sum(np.maximum(carb_C, 0.0))) / cm
    co2 = {"exported_alkalinity": eta * tot["drainage"],
           "carbonate_C_formed": c_formed,
           "carbonate_C_dissolved": -c_dissolved}
    co2["co2_removed"] = sum(co2.values())
    co2["stored_alkalinity"] = stored["aqueous"] + stored["exchanger"]
    co2["co2_potential"] = co2["co2_removed"] + co2["stored_alkalinity"]
    return {"series": {k: v / cm for k, v in rows.items()},
            "carbonate_C": -carb_C / cm,                 # formed > 0, per step
            "steps": steps, "totals": tot, "stored": stored,
            "delta_alk_tot": float(np.sum(delta)) / cm,
            "closure_step_rel": step_rel, "ok": step_rel <= RTOL_STEP,
            "co2": co2, "eta": eta, "fe_mode": d.get("fe_mode", "tracer"),
            "carb_scheme": d.get("carb_scheme", "legacy")}


def compare(treated, control):
    """Treatment effect: every cumulative number, treated minus control."""
    out = {"totals": {k: treated["totals"][k] - control["totals"][k]
                      for k in treated["totals"]},
           "stored": {k: treated["stored"][k] - control["stored"][k]
                      for k in treated["stored"]},
           "co2": {k: treated["co2"][k] - control["co2"][k] for k in treated["co2"]}}
    return out


def report(c, label=""):
    """Human-readable budget [eq or mol per m2]."""
    t, co2 = c["totals"], c["co2"]
    gross = t["silicates"] + t["carbonates_dis"] + t["background"]
    pct = (lambda x: 100 * x / gross) if gross else (lambda x: float("nan"))
    out = [f"CDR ledger {label}".rstrip() + f"  (fe_mode={c['fe_mode']}, "
           f"carbonates={c['carb_scheme']}, closure {c['closure_step_rel']:.1e} "
           f"{'ok' if c['ok'] else 'FAILED'})",
           f"  alkalinity in   [eq/m2]   silicates {t['silicates']:.4g}  "
           f"carbonates {t['carbonates_dis']:.4g}  background {t['background']:.4g}",
           f"  alkalinity out  [% of in] drainage {pct(t['drainage']):.1f}  "
           f"carbonates {pct(t['carbonates_ppt']):.1f}  uptake {pct(t['uptake']):.1f}  "
           f"transpiration {pct(t['transpiration']):.1f}  other {pct(t['secondary_other']):.1f}",
           f"  still in soil   [eq/m2]   aqueous {c['stored']['aqueous']:+.4g}  "
           f"exchanger {c['stored']['exchanger']:+.4g}",
           f"  CO2 removed     [mol/m2]  {co2['co2_removed']:.4g}  = exported "
           f"{co2['exported_alkalinity']:.4g} (eta {c['eta']}) + carbonate C formed "
           f"{co2['carbonate_C_formed']:.4g} {co2['carbonate_C_dissolved']:+.4g} dissolved",
           f"  CO2 potential   [mol/m2]  {co2['co2_potential']:.4g}  (+ {co2['stored_alkalinity']:.4g} "
           f"eq still in the soil, if it is exported)"]
    return "\n".join(out)
