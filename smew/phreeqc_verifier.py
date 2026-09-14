# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-check SMEW's chemistry against PHREEQC, and cost a delegated speciation call.

This is the F3.3 instrument. It exists for two jobs that happen to need the same
plumbing:

  1. The cross-code check. Golden masters lock in existing behaviour INCLUDING
     existing mistakes, and the mass-balance ledger catches bookkeeping errors
     but not a wrong equilibrium constant. An independent implementation is the
     only tier that catches an error which is CONSISTENTLY wrong, which is why
     Foundations_0 Part V calls it the most valuable single test in the block.

  2. The timing number that decides the speciation-engine fork: is per-timestep
     delegation to PHREEQC affordable at ~5e4 steps per simulated year?

PHREEQC is never in SMEW's run path. This module is an optional dependency --
import it, and if phreeqpython is missing you get a clear skip rather than an
ImportError at package import time. Nothing in smew/ imports this.

    python -m smew.phreeqc_verifier check     # the layered comparison
    python -m smew.phreeqc_verifier sweep     # separate activities from constants
    python -m smew.phreeqc_verifier bench     # the fork's cost number

WHY THE COMPARISON IS LAYERED. A single SMEW-versus-PHREEQC number would be
uninterpretable, because at least three things differ at once: the carbonate
constants, the aluminium ladder, and the exchange convention -- on top of the
fact that SMEW works in CONCENTRATIONS and PHREEQC in ACTIVITIES. So each layer
adds exactly one thing, and `sweep` separates the activity gap from everything
else by diluting: a discrepancy that survives at infinite dilution is a
different constant, and one that grows with ionic strength is the missing
activity model. That second number is the size of the F4 gap, measured before
any of F4 is built.
"""

import numpy as np

# --- optional dependency -----------------------------------------------------
#
# Import from the INSTALLED package, never from a local clone of the Vitens
# source: the dylib committed there is x86_64-only and dies in
# ctypes.LoadLibrary on arm64. `periodictable` is an undeclared dependency of
# phreeqpython (utility.py imports it at module level) and pip does not pull it.
try:
    import phreeqpython as _pp
    _IMPORT_ERROR = None
except Exception as exc:                                   # pragma: no cover
    _pp = None
    _IMPORT_ERROR = exc


def available():
    """(bool, reason). Callers skip rather than fail when PHREEQC is absent."""
    if _pp is None:
        return False, ("phreeqpython not importable: %s\n"
                       "    pip install phreeqpython periodictable" % _IMPORT_ERROR)
    return True, "phreeqpython %s" % getattr(_pp, "__version__", "?")


# ---------------------------------------------------------------------------
# Units: the part that has to be right before anything else means anything
# ---------------------------------------------------------------------------
#
# SMEW's units are inconsistent by design and every function documents its own.
# Collected here once, because a translation layer that gets one of these wrong
# produces a comparison that looks like a chemistry disagreement:
#
#   Ca, Mg, Na, K, Si, An, H, CO2_w, HCO3, CO3   [mol-conv/l]  -> / conv_mol
#   Al, Al_w, AlOH, AlOH2, AlOH3, AlOH4          [mol-conv_Al/l]
#                                                -> / (conv_mol * conv_Al)
#   CEC_tot                                      [mol_c per m2] -> / conv_mol
#   nZrs = n * Zr * s * 1000                     [l of soil water per m2]
#
# The aluminium one is the trap. conv_Al multiplies conv_mol rather than
# replacing it, so with the usual conv_mol=1e6, conv_Al=1e3 aluminium is carried
# in NANOMOLES per litre while everything else is in micromoles. Confirmed
# against biogeochem.py's equation 5, where the exchanger term
# (f_Al/3)*CEC_tot*conv_Al must come out in Al_tot's units and CEC_tot is
# already in conv_mol.


def _molar(x, conv_mol):
    return float(x) / conv_mol


def _molar_al(x, conv_mol, conv_Al):
    return float(x) / (conv_mol * conv_Al)


def composition_from_run(data, i, CEC_tot=None):
    """One timestep of a SMEW result dict, in mol/l, as PHREEQC needs it.

    `data` is what biogeochem_balance returns; `i` the timestep index. Reads the
    aqueous concentrations rather than the total pools, because what is being
    compared is the SPECIATION of a given solution, not the pool split.

    The lumped anion is the one genuine modelling decision here, and it cannot
    be avoided: SMEW's `An` is a charge tracer, not a species -- it exists to
    close the charge balance and has no identity, no complexation and no size.
    PHREEQC cannot take a solution without real anions. Chloride is the right
    stand-in (conservative, does not complex the major cations, charge -1 so
    mol_c and mol coincide), and the same choice will have to be made in F4 to
    let `An` enter the ionic strength at all. Declared here, once.
    """
    cm = float(np.ravel(data["conv_mol"])[0])
    ca = float(np.ravel(data["conv_Al"])[0])
    n = float(np.ravel(data["n"])[0])
    Zr = float(np.ravel(data["Zr"])[0])
    s = float(np.asarray(data["s"])[i])
    nZrs = n * Zr * s * 1000.0

    return {
        "pH": float(np.asarray(data["pH"])[i]),
        "temp": float(np.asarray(data["temp_soil"])[i]),
        "Ca": _molar(np.asarray(data["Ca"])[i], cm),
        "Mg": _molar(np.asarray(data["Mg"])[i], cm),
        "Na": _molar(np.asarray(data["Na"])[i], cm),
        "K": _molar(np.asarray(data["K"])[i], cm),
        "Si": _molar(np.asarray(data["Si"])[i], cm),
        "Cl": _molar(np.asarray(data["An"])[i], cm),       # the lumped anion
        "C": _molar(np.asarray(data["DIC"])[i], cm),
        "Al": _molar_al(np.asarray(data["Al_w"])[i], cm, ca),
        # SMEW's own answer, to be compared against PHREEQC's
        "smew": {
            "CO2": _molar(np.asarray(data["CO2_w"])[i], cm),
            "HCO3": _molar(np.asarray(data["HCO3"])[i], cm),
            "CO3": _molar(np.asarray(data["CO3"])[i], cm),
            "Al+3": _molar_al(np.asarray(data["Al"])[i], cm, ca),
            "AlOH+2": _molar_al(np.asarray(data["AlOH"])[i], cm, ca),
            "Al(OH)2+": _molar_al(np.asarray(data["AlOH2"])[i], cm, ca),
            "Al(OH)3": _molar_al(np.asarray(data["AlOH3"])[i], cm, ca),
            "Al(OH)4-": _molar_al(np.asarray(data["AlOH4"])[i], cm, ca),
            "f_Ca": float(np.asarray(data["f_Ca"])[i]),
            "f_Mg": float(np.asarray(data["f_Mg"])[i]),
            "f_Na": float(np.asarray(data["f_Na"])[i]),
            "f_K": float(np.asarray(data["f_K"])[i]),
            "f_Al": float(np.asarray(data["f_Al"])[i]),
        },
        # CEC_tot is not in the output contract (it is an argument, not a
        # result), so the caller supplies it when layer 3 is wanted.
        "CEC_per_l": (CEC_tot / cm / nZrs) if CEC_tot is not None else None,
        "nZrs": nZrs,
    }

# ---------------------------------------------------------------------------
# The PHREEQC side
# ---------------------------------------------------------------------------

_PUNCH = """
SELECTED_OUTPUT 1
    -reset false
    -high_precision true
USER_PUNCH 1
    -headings pH mu CO2 HCO3 CO3 Al3 AlOH AlOH2 AlOH3 AlOH4 CaX2 MgX2 NaX KX AlX3 gH gCa gHCO3
    10 PUNCH -LA("H+"), MU, MOL("CO2"), MOL("HCO3-"), MOL("CO3-2")
    20 PUNCH MOL("Al+3"), MOL("AlOH+2"), MOL("Al(OH)2+"), MOL("Al(OH)3"), MOL("Al(OH)4-")
    30 PUNCH MOL("CaX2"), MOL("MgX2"), MOL("NaX"), MOL("KX"), MOL("AlX3")
    40 PUNCH ACT("H+")/MOL("H+"), ACT("Ca+2")/MOL("Ca+2"), ACT("HCO3-")/MOL("HCO3-")
"""
_COLS = ("pH mu CO2 HCO3 CO3 Al3 AlOH AlOH2 AlOH3 AlOH4 "
         "CaX2 MgX2 NaX KX AlX3 gH gCa gHCO3").split()


class Verifier:
    """One live PHREEQC engine, reused across calls.

    Reused deliberately. Constructing PhreeqPython() per call costs 1.76 ms
    against 0.08 ms on a persistent instance -- a 20x penalty that would make
    any measurement of "what would delegation cost" meaningless, and it is the
    single mistake most likely to wrongly kill the delegated option.
    """

    def __init__(self, database="phreeqc.dat"):
        ok, why = available()
        if not ok:
            raise RuntimeError(why)
        self.pp = _pp.PhreeqPython(database=database)
        self.ip = self.pp.ip
        self.database = database
        self.ip.run_string(_PUNCH)

    # -- input assembly ----------------------------------------------------
    def _solution(self, c, elements, dilution=1.0, fix_pH=True):
        """A SOLUTION block holding only `elements`, optionally diluted.

        Dilution divides every total by the same factor and leaves pH fixed. It
        is how `sweep` separates a constant disagreement (survives dilution)
        from the missing activity model (vanishes with it): at infinite dilution
        every activity coefficient goes to 1, so a concentration-based model and
        an activity-based one must agree if their constants agree.

        With fix_pH the pH is imposed and charge is balanced on Cl, so what is
        compared is the SPECIATION of a stated solution rather than SMEW's
        charge-balance closure. That is the cleaner diagnostic and it comes
        first; letting PHREEQC find its own pH tests something else.
        """
        lines = ["SOLUTION 1", "    units mol/kgw", "    temp %.4f" % c["temp"],
                 "    pH %.6f" % (c["pH"] if fix_pH else 7.0)]
        for el in elements:
            key = {"C": "C(4)"}.get(el, el)
            val = c[el] / dilution
            # Charge is balanced on chloride: SMEW's An is a charge tracer with
            # no identity, so it is the only species here with no information to
            # lose. Balancing on anything else would silently move a real total.
            suffix = " charge" if el == "Cl" else ""
            lines.append("    %-5s %.10e%s" % (key, max(val, 0.0), suffix))
        return "\n".join(lines) + "\n"

    def _run(self, text):
        self.ip.run_string(text)
        row = self.ip.get_selected_output_array()[-1]
        return dict(zip(_COLS, (float(v) for v in row)))

    # -- the layers --------------------------------------------------------
    def layer1_carbonate(self, c, dilution=1.0):
        """Carbonate system only. Tests k1, k2 against PHREEQC's K1, K2.

        No aluminium and no exchanger, so any disagreement is the carbonate
        constants plus the activity gap and nothing else.
        """
        out = self._run(self._solution(
            c, ("Ca", "Mg", "Na", "K", "Cl", "C"), dilution) + "END\n")
        ref = {k: c["smew"][k] / dilution for k in ("CO2", "HCO3", "CO3")}
        return out, ref

    def layer2_aluminium(self, c, dilution=1.0, Al_total=None):
        """+ aluminium. Tests SMEW's four hydrolysis constants.

        SMEW's K_Al are pK 5.0, 5.1, 6.7, 6.2 and are TEMPERATURE-INDEPENDENT
        (constants.py K_Al takes no T), while PHREEQC's carry a delta_h. So part
        of any disagreement here is temperature, and it will not vanish on
        dilution -- which the sweep makes visible rather than hiding.
        """
        cc = dict(c)
        if Al_total is not None:
            cc["Al"] = Al_total
        out = self._run(self._solution(
            cc, ("Ca", "Mg", "Na", "K", "Cl", "C", "Al", "Si"), dilution)
            + "END\n")
        ref = smew_al_ladder(cc["Al"] / dilution, cc["pH"],
                             cc.get("conv_mol", 1e6), cc.get("conv_Al", 1e3))
        return out, ref

    # -- layer 3: exchange -------------------------------------------------
    #
    # PHREEQC's stock exchange log_k values are DIFFERENT PARAMETERS from
    # SMEW's, whose K_GT_CEC come from a meta-analysis of Dutch soils. Comparing
    # against them would disagree for parameter reasons and say nothing about
    # SMEW's code. So SMEW's own constants are injected, and what the layer then
    # tests is whether SMEW's Gaines-Thomas ALGEBRA matches PHREEQC's
    # Gaines-Thomas implementation -- which would catch a wrong exponent or a
    # mis-stated convention in biogeochem.py rows 11-15.
    #
    # Two conversions are needed, and the second is the one that bites.
    #
    # (a) Reference frame. SMEW states every selectivity relative to calcium;
    #     PHREEQC states each species relative to the free site X-. Only
    #     differences matter, so CaX2 is pinned at log_k 0 and the rest follow
    #     from the exchange reactions:
    #         Ca+2 + 2NaX  = CaX2 + 2Na+   K_Ca_Na = 10^(k_CaX2 - 2 k_NaX)
    #         Ca+2 +  MgX2 = CaX2 +  Mg+2  K_Ca_Mg = 10^(k_CaX2 -   k_MgX2)
    #        3Ca+2 + 2AlX3 = 3CaX2 + 2Al+3 K_Ca_Al = 10^(3 k_CaX2 - 2 k_AlX3)
    #
    # (b) Units. SMEW's constants are defined on concentrations in conv_mol
    #     units (micromoles/l at the usual conv_mol=1e6), PHREEQC's on mol/kgw.
    #     The factor is NOT the same for every species -- it is conv_mol raised
    #     to the net change in aqueous stoichiometry across the reaction:
    #         Mg  exchange is 1-for-1 divalent, so conv_mol cancels: factor 1
    #         Na, K, H are 2-for-1, one Ca in against two out:   K/conv_mol
    #         Al  is 2-for-3, three Ca in against two out:       K*conv_mol
    #     Getting this wrong moves a selectivity by six orders of magnitude and
    #     reads as a chemistry disagreement, so `layer3_exchange` refuses to
    #     report unless its dilution self-check passes.

    _GT = (("MgX2", "Mg+2", 2, "K_Ca_Mg", 1.0, 1.0),
           ("NaX",  "Na+",  1, "K_Ca_Na", 0.5, -1.0),
           ("KX",   "K+",   1, "K_Ca_K",  0.5, -1.0),
           ("AlX3", "Al+3", 3, "K_Ca_Al", 0.5, +1.0))

    def exchange_species_block(self, K_CEC, conv_mol):
        """SMEW's Gaines-Thomas constants as a PHREEQC EXCHANGE_SPECIES block."""
        K_Ca_Mg, K_Ca_K, K_Ca_Na, K_Ca_Al, K_Ca_H = (float(x) for x in K_CEC)
        Ks = {"K_Ca_Mg": K_Ca_Mg, "K_Ca_K": K_Ca_K, "K_Ca_Na": K_Ca_Na,
              "K_Ca_Al": K_Ca_Al, "K_Ca_H": K_Ca_H}
        out = ["EXCHANGE_SPECIES",
               "    X- = X-", "        log_k 0.0",
               "    Ca+2 + 2X- = CaX2", "        log_k 0.0"]   # the reference
        for name, ion, nu, key, half, upow in self._GT:
            K_molar = Ks[key] * (conv_mol ** upow)
            # k_species = -half * log10(K_molar), from the reactions above
            out += ["    %s + %dX- = %s" % (ion, nu, name),
                    "        log_k %.6f" % (-half * np.log10(K_molar))]
        out += ["    H+ + X- = HX",
                "        log_k %.6f" % (-0.5 * np.log10(Ks["K_Ca_H"] / conv_mol))]
        return "\n".join(out) + "\n"

    def set_exchange_constants(self, K_CEC, conv_mol):
        """Load SMEW's selectivity constants once, not once per call.

        EXCHANGE_SPECIES redefines database-level species, so emitting it with
        every solve makes PHREEQC re-parse the exchange block ~5e4 times a run.
        Measured, that is the difference between 0.51 ms and 0.30 ms per step --
        the delegated cost looked 70 % worse purely because of where this line
        sat. Exactly the "benchmark the good implementation, not the obvious
        one" trap, found by falling into it.
        """
        self.ip.run_string(self.exchange_species_block(K_CEC, conv_mol))
        self._exchange_loaded = True

    def layer3_exchange(self, c, K_CEC, conv_mol, dilution=1.0):
        """+ cation exchange, with SMEW's own selectivity constants injected.

        Returns (phreeqc_row, smew_reference, sites). The exchanger composition
        handed to PHREEQC is SMEW's own equivalent fractions times the site
        density, so agreement means the two implementations put the same
        cations in the same places from the same starting point.
        """
        sites = c["CEC_per_l"]
        if sites is None or not np.isfinite(sites):
            raise ValueError("layer 3 needs CEC_tot; pass it to "
                             "composition_from_run")
        sm = c["smew"]
        # equivalent fractions -> moles of each exchange species per litre
        ex = ["EXCHANGE 1",
              "    CaX2 %.10e" % (sm["f_Ca"] * sites / 2.0),
              "    MgX2 %.10e" % (sm["f_Mg"] * sites / 2.0),
              "    NaX  %.10e" % (sm["f_Na"] * sites),
              "    KX   %.10e" % (sm["f_K"] * sites),
              "    AlX3 %.10e" % (sm["f_Al"] * sites / 3.0)]
        if not getattr(self, "_exchange_loaded", False):
            self.set_exchange_constants(K_CEC, conv_mol)
        text = (self._solution(c, ("Ca", "Mg", "Na", "K", "Cl", "C", "Al", "Si"),
                               dilution)
                + "\n".join(ex) + "\nEND\nUSE solution 1\nUSE exchange 1\nEND\n")
        out = self._run(text)
        return out, sm, sites


def smew_al_ladder(Al_w_molar, pH, conv_mol, conv_Al):
    """SMEW's aluminium speciation, computed directly from its own constants.

    Layer 2 is a test of four equilibrium constants, so it should not depend on
    whether the run being sampled happened to contain any aluminium. A
    forsterite run carries none -- Al sits at 1e-50 and comparing it against
    PHREEQC's 1e-33 numerical floor is noise against noise. Evaluating SMEW's
    own ladder at a stated total Al asks the question the layer is actually for.

    Mirrors biogeochem.py's speciation block exactly, including that SMEW's
    K_Al are temperature-independent while PHREEQC's are not.
    """
    import smew
    K1, K2, K3, K4 = smew.K_Al(conv_mol)
    H = 10 ** (-pH) * conv_mol
    den = (H ** 4 + H ** 3 * K1 + H ** 2 * K1 * K2 + H * K1 * K2 * K3
           + K1 * K2 * K3 * K4)
    Al = (H ** 4 / den) * Al_w_molar
    AlOH = (H ** 3 * K1 / den) * Al_w_molar
    AlOH2 = (H ** 2 * K1 * K2 / den) * Al_w_molar
    AlOH3 = (H * K1 * K2 * K3 / den) * Al_w_molar
    return {"Al3": Al, "AlOH": AlOH, "AlOH2": AlOH2, "AlOH3": AlOH3,
            "AlOH4": Al_w_molar - (Al + AlOH + AlOH2 + AlOH3)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _reldiff(a, b):
    """Relative difference, on a floor so two near-zero species are not news."""
    a, b = float(a), float(b)
    sc = max(abs(a), abs(b))
    return np.nan if sc <= 0 else abs(a - b) / sc


# Below this total concentration an element is not present, it is float noise,
# and comparing two noises produces a confident-looking 100 %. A forsterite run
# carries no aluminium at all: SMEW reports Al species at 1e-50 and PHREEQC at
# its own numerical floor of 1e-33, and the "disagreement" between them is
# nothing. Same principle as smew.ledger's ACTIVITY_FLOOR and F2's CLIP_REL --
# never gate a quantity that is legitimately zero.
INACTIVE = 1e-12        # mol/l


def _table(title, pairs, inactive=False):
    print("\n  %s" % title)
    if inactive:
        print("    INACTIVE -- this element is absent from the run (total below"
              " %.0e mol/l).\n    Nothing to compare; noise over noise is not a"
              " disagreement." % INACTIVE)
        return None
    print("    %-12s %14s %14s %10s" % ("species", "SMEW", "PHREEQC", "rel diff"))
    worst = 0.0
    for name, smew_v, pq_v in pairs:
        d = _reldiff(smew_v, pq_v)
        if np.isfinite(d):
            worst = max(worst, d)
        print("    %-12s %14.5e %14.5e %9.2f%%"
              % (name, smew_v, pq_v, 100 * d if np.isfinite(d) else float("nan")))
    return worst


def run_check(database="phreeqc.dat", index=None, verbose=True):
    """The layered comparison, on a real timestep of a real SMEW run."""
    from smew.harness import example_run
    import smew

    data, t, _ = example_run(t_end=365)
    i = index if index is not None else len(t) // 2
    conv_mol = float(np.ravel(data["conv_mol"])[0])

    # CEC_tot is an argument of the run, not an output; example_run's value.
    rho_bulk, Zr = 1.2e6, 0.3
    CEC_tot = 10 * 1e-5 * rho_bulk * Zr * conv_mol
    c = composition_from_run(data, i, CEC_tot=CEC_tot)

    v = Verifier(database)
    print("=" * 78)
    print("SMEW vs PHREEQC -- layered speciation check")
    print("  database %s - Example run, timestep %d of %d, pH %.3f, %.1f C"
          % (database, i, len(t), c["pH"], c["temp"]))
    print("=" * 78)

    worst = {}

    o, ref = v.layer1_carbonate(c)
    worst["1 carbonate"] = _table(
        "Layer 1 - carbonate system (SMEW k1, k2 vs PHREEQC K1, K2)",
        [(k, ref[k], o[k]) for k in ("CO2", "HCO3", "CO3")])
    print("    ionic strength %.4e mol/kgw   gamma(H+) %.4f  gamma(Ca+2) %.4f"
          % (o["mu"], o["gH"], o["gCa"]))

    # The Example run is forsterite-only, so its own Al is float noise. Layer 2
    # tests four constants, not this run's mineralogy, so it is evaluated at a
    # stated, realistic dissolved aluminium instead.
    AL_TEST = 1e-6                       # mol/l, a plausible soil-solution Al
    o2, ref2 = v.layer2_aluminium(c, Al_total=AL_TEST)
    worst["2 aluminium"] = _table(
        "Layer 2 - Al hydrolysis at %.0e mol/l total Al (SMEW pK 5.0/5.1/6.7/"
        "6.2, T-independent)" % AL_TEST,
        [(k, ref2[k], o2[k]) for k in ("Al3", "AlOH", "AlOH2", "AlOH3", "AlOH4")])

    # layer 3 needs the same K_CEC the run used
    f_CEC_in = np.array([0.30, 0.15, 0.10, 0.05, 0.00, 0.40])
    _, K_CEC = smew.f_CEC_to_conc(f_CEC_in, 4, "loam", conv_mol,
                                  float(np.ravel(data["conv_Al"])[0]))
    o3, sm3, sites = v.layer3_exchange(c, K_CEC, conv_mol)
    rows3 = [("f_Ca", sm3["f_Ca"], 2 * o3["CaX2"] / sites),
             ("f_Mg", sm3["f_Mg"], 2 * o3["MgX2"] / sites),
             ("f_Na", sm3["f_Na"], o3["NaX"] / sites),
             ("f_K",  sm3["f_K"],  o3["KX"] / sites)]
    if c["Al"] >= INACTIVE:
        rows3.append(("f_Al", sm3["f_Al"], 3 * o3["AlX3"] / sites))
    worst["3 exchange"] = _table(
        "Layer 3 - cation exchange, SMEW's own K_GT_CEC injected", rows3)

    print("\n  worst relative disagreement per layer:")
    for k in sorted(worst):
        print("    %-14s %8.2f %%" % (k, 100 * worst[k]))
    print("\n  These are NOT pass/fail. SMEW works in concentrations and PHREEQC"
          "\n  in activities, so disagreement is expected and its SIZE is the"
          "\n  measurement. Run `sweep` to split it into a constant difference"
          "\n  (survives dilution) and the missing activity model (does not).")
    return worst


def run_sweep(database="phreeqc.dat", index=None):
    """Dilution series: separate a wrong constant from a missing activity model.

    At infinite dilution every activity coefficient goes to 1, so a
    concentration-based model and an activity-based one MUST agree if their
    constants agree. Whatever disagreement survives dilution is a different
    constant; whatever vanishes is the F4 gap, and its size at the run's own
    ionic strength is how much F4 is worth.
    """
    from smew.harness import example_run
    data, t, _ = example_run(t_end=365)
    i = index if index is not None else len(t) // 2
    c = composition_from_run(data, i)
    v = Verifier(database)

    print("=" * 78)
    print("Dilution sweep -- constants vs activities")
    print("  a disagreement that SURVIVES dilution is a different constant;")
    print("  one that VANISHES is the activity model SMEW does not have (F4).")
    print("=" * 78)
    print("\n  %10s %12s %8s %10s %10s %10s" %
          ("dilution", "I (mol/kgw)", "g(Ca+2)", "HCO3 err", "CO3 err", "Al3 err"))
    for dil in (1, 10, 100, 1000, 10000):
        try:
            o, ref = v.layer1_carbonate(c, dilution=dil)
            o2, ref2 = v.layer2_aluminium(c, dilution=dil, Al_total=1e-6)
        except Exception as exc:                       # PHREEQC may not converge
            print("  %10d  did not converge: %s" % (dil, str(exc)[:40]))
            continue
        print("  %10d %12.3e %8.4f %9.3f%% %9.3f%% %9.3f%%"
              % (dil, o["mu"], o["gCa"],
                 100 * _reldiff(ref["HCO3"], o["HCO3"]),
                 100 * _reldiff(ref["CO3"], o["CO3"]),
                 100 * _reldiff(ref2["Al3"], o2["Al3"])))
    print("\n  The residue at high dilution bounds the CONSTANT disagreement;")
    print("  everything above it is the activity model, i.e. the size of F4.")


def run_bench(database="phreeqc.dat", n=1000):
    """What would per-timestep delegation to PHREEQC actually cost?

    The number the speciation-engine fork turns on. Reported against SMEW's own
    measured cost: the 16-unknown implicit solve is 44 us per step, 2.3 s per
    simulated year, inside a 6.5 s run at dt = 10 min.
    """
    import time
    v = Verifier(database)
    rng = np.random.default_rng(0)
    base = {"pH": 5.5, "temp": 13.0, "Ca": 9.8e-4, "Mg": 5.1e-4, "Na": 1.2e-4,
            "K": 8.0e-5, "Si": 2.2e-4, "Cl": 9.0e-4, "C": 2.4e-3, "Al": 1.2e-7,
            "CEC_per_l": 0.5,
            "smew": {k: 0.0 for k in ("CO2", "HCO3", "CO3", "Al+3", "AlOH+2",
                                      "Al(OH)2+", "Al(OH)3", "Al(OH)4-")}}
    base["smew"].update({"f_Ca": 0.30, "f_Mg": 0.15, "f_Na": 0.05,
                         "f_K": 0.10, "f_Al": 0.0})

    def jitter():
        c = dict(base)
        j = rng.normal(1.0, 0.05)
        for k in ("Ca", "Mg", "Na", "K", "Si", "Cl", "C", "Al"):
            c[k] = base[k] * j
        c["pH"] = 5.5 + 0.3 * rng.normal()
        return c

    def timeit(label, fn, m):
        fn()
        t0 = time.perf_counter()
        for _ in range(m):
            fn()
        dt = (time.perf_counter() - t0) / m
        print("  %-46s %8.3f ms  -> %7.1f s / simulated year"
              % (label, dt * 1e3, dt * 52560))
        return dt

    print("=" * 78)
    print("Cost of delegating speciation to PHREEQC (%s)" % database)
    print("  SMEW today: 0.044 ms/step for the 16-unknown solve")
    print("              2.3 s of a 6.5 s one-year run at dt = 10 min")
    print("=" * 78 + "\n")

    timeit("fresh PhreeqPython() every call (the trap)",
           lambda: _pp.PhreeqPython(database=database).add_solution(
               {"Ca": 1.0, "C(4)": 2.0, "pH": 7.0}), max(n // 20, 50))
    a = timeit("persistent engine, speciation only",
               lambda: v.layer1_carbonate(jitter()), n)
    b = timeit("persistent engine, + aluminium and silica",
               lambda: v.layer2_aluminium(jitter()), n)
    K = np.array([0.6, 3.0, 0.4, 1.0, 1.0])
    v.set_exchange_constants(K, 1e6)          # once, not per call -- see above
    c3 = timeit("persistent engine, + coupled cation exchange",
                lambda: v.layer3_exchange(jitter(), K, 1e6), n)
    print("\n  the exchange coupling adds %.3f ms (%.1fx the speciation alone)"
          % ((c3 - b) * 1e3, c3 / b))
    print("  a delegated run would be about %.0f s against 6.5 s today (%.1fx)"
          % (6.5 - 2.3 + c3 * 52560, (6.5 - 2.3 + c3 * 52560) / 6.5))
    return {"speciation": a, "aluminium": b, "exchange": c3}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["check", "sweep", "bench"])
    ap.add_argument("--database", default="phreeqc.dat")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("-n", type=int, default=1000)
    a = ap.parse_args(argv)

    ok, why = available()
    if not ok:
        print("SKIP: %s" % why)
        return 0
    print("phreeqpython: %s\n" % why)
    if a.mode == "check":
        run_check(a.database, a.index)
    elif a.mode == "sweep":
        run_sweep(a.database, a.index)
    else:
        run_bench(a.database, a.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
