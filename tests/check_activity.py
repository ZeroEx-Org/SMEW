# SPDX-License-Identifier: AGPL-3.0-only
"""F4 gates for smew/activity.py.

    python tests/check_activity.py identity   # gamma_on=False is BIT-identical
    python tests/check_activity.py phreeqc    # gamma_i within 1 % of PHREEQC
    python tests/check_activity.py units      # mu against an independent sum
    python tests/check_activity.py shift      # F4.4: what the flag moves, per case
    python tests/check_activity.py cec        # F4.6: sensitivity to MU_REF_CEC

`identity` is the gate that matters most and it is re-run after every later F4
step, not only after the first. The whole audit trail of this block is the
claim that the flag-off path did not move; the moment that stops being checked,
an activity effect and a bug become indistinguishable.
"""
import io
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from smew import activity, harness           # noqa: E402

# Keys F4 ADDS. A golden frozen before F4 cannot contain them, so they are the
# only permitted absences; anything else failing is a regression.
NEW_KEYS = ("mu", "gamma_H", "pH_conc")
SUFFIXES = ("", "__min", "__max", "__mean", "__sum", "__final")


def _is_new(key):
    base = key.rsplit(".", 1)[-1]
    for s in SUFFIXES:
        if s and base.endswith(s):
            base = base[: -len(s)]
            break
    return base in NEW_KEYS


def check_identity(golden_dir=None, rtol=0.0):
    """Every pre-F4 key, bit-identical, on every frozen case."""
    golden_dir = golden_dir or os.path.join(REPO, "tests", "golden")
    sys.path.insert(0, os.path.join(REPO, "tests"))
    import freeze_cases as fc

    rc = 0
    for stem, builder in (("Vials_Dietzen", None), ("Mesocosm_Kelland", None),
                          ("Mesocosm_Amann", None), ("Bottles_tePas", None),
                          ("Example", None), ("Vulkaneifel", "wrapper"),
                          ("Synthetic_application", "synthetic")):
        dest = os.path.join(golden_dir, stem + ".npz")
        if not os.path.exists(dest):
            print("%-22s MISS" % stem); rc = 1; continue
        if builder == "wrapper":
            new = fc.build_wrapper(500)
        elif builder == "synthetic":
            new = fc.build_synthetic(500)
        else:
            g = fc.exec_notebook(os.path.join(REPO, "Examples", stem + ".ipynb"))
            new = fc.build(g, 500)
        res = harness.compare(harness.load(dest), new, rtol=rtol)
        bad = [r for r in res if not r[2]]
        unexpected = [r for r in bad if not _is_new(r[0])]
        added = len(bad) - len(unexpected)
        ok = not unexpected
        print("%-22s %s  %d/%d keys identical (rtol=%g), %d new keys added"
              % (stem, "PASS" if ok else "FAIL", len(res) - len(bad), len(res),
                 rtol, added))
        for k, d, _ in unexpected[:8]:
            print("      %-30s max|diff| %12.6g" % (k, d))
        rc |= 0 if ok else 1
    return rc


def check_phreeqc(tol_pct=1.0):
    """gamma_i within tol_pct of PHREEQC over the model's whole mu range.

    A formula-against-formula comparison on a STATED composition, which is why
    1 % is achievable here. Saturation indices are deliberately not in this
    gate: SMEW has no ion pairing, so an SI comparison measures that far larger
    and separate omission rather than this module. See activity.py's header.
    """
    try:
        import phreeqpython
    except Exception as exc:
        print("SKIP  phreeqpython not importable: %s" % exc)
        return 0

    pp = phreeqpython.PhreeqPython(database="phreeqc.dat")
    ip = pp.ip
    ip.run_string(
        'SELECTED_OUTPUT 1\n -reset false\n -high_precision true\n'
        'USER_PUNCH 1\n -headings mu gH gOH gHCO3 gCO3 gCa gMg gNa gK gAl gAlOH4\n'
        ' 10 PUNCH MU, ACT("H+")/MOL("H+"), ACT("OH-")/MOL("OH-")\n'
        ' 20 PUNCH ACT("HCO3-")/MOL("HCO3-"), ACT("CO3-2")/MOL("CO3-2")\n'
        ' 30 PUNCH ACT("Ca+2")/MOL("Ca+2"), ACT("Mg+2")/MOL("Mg+2")\n'
        ' 40 PUNCH ACT("Na+")/MOL("Na+"), ACT("K+")/MOL("K+")\n'
        ' 50 PUNCH ACT("Al+3")/MOL("Al+3"), ACT("Al(OH)4-")/MOL("Al(OH)4-")\n')
    COLS = "mu gH gOH gHCO3 gCO3 gCa gMg gNa gK gAl gAlOH4".split()
    IDX = {"gH": activity.SP_H, "gOH": activity.SP_OH,
           "gHCO3": activity.SP_HCO3, "gCO3": activity.SP_CO3,
           "gCa": activity.SP_CA, "gMg": activity.SP_MG,
           "gNa": activity.SP_NA, "gK": activity.SP_K,
           "gAl": activity.SP_AL, "gAlOH4": activity.SP_ALOH4}

    worst = {k: (0.0, 0.0, 0.0) for k in IDX}
    n = 0
    # The suite's measured mu range is 1.3e-3 to 1.6e-1. This sweep spans
    # 2e-3 to 5e-1, i.e. past the top of it, so the gate still means something
    # for an application rate heavier than anything benchmarked so far.
    for t_C in (2.0, 13.0, 25.0, 35.0):
        for c in (1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1):
            # Chloride is stated and SODIUM balances the charge. Balancing
            # on Cl instead leaves PHREEQC unable to converge on Ca at the
            # dilute end, because the stated Cl is then not enough to carry the
            # divalent cations and the solver is asked to satisfy a fixed pH
            # and a charge deficit at once.
            ip.run_string(
                "SOLUTION 1\n units mol/kgw\n temp %.2f\n pH 7.2\n"
                " Cl %.6e\n Na %.6e charge\n Ca %.6e\n Mg %.6e\n K %.6e\n"
                " C(4) 2e-3\n Al 1e-7\n END\n"
                % (t_C, 2.0 * c, c, c / 3.0, c / 5.0, c / 10.0))
            row = dict(zip(COLS, (float(v) for v in
                                  ip.get_selected_output_array()[-1])))
            mu = row["mu"]
            g = activity.gammas(mu, t_C + 273.15)
            n += 1
            for key, i in IDX.items():
                ref = row[key]
                if ref <= 0:
                    continue
                err = 100.0 * abs(g[i] - ref) / ref
                if err > worst[key][0]:
                    worst[key] = (err, mu, t_C)

    rc = 0
    print("gamma vs PHREEQC over %d compositions, mu = 2e-3 .. 5e-1, 2-35 C" % n)
    for key in COLS[1:]:
        err, mu, t_C = worst[key]
        ok = err <= tol_pct
        rc |= 0 if ok else 1
        print("   %-8s %s worst %6.3f %%  at mu = %.4f, %.0f C"
              % (activity.SP_NAMES[IDX[key]], "PASS" if ok else "FAIL", err,
                 mu, t_C))
    return rc


def check_units():
    """mu from the model's own units against a plain mol/l sum.

    The trap this exists for: conv_Al MULTIPLIES conv_mol, so aluminium is
    carried in nanomoles while everything else is in micromoles. A translation
    that misses that produces a plausible-looking gamma from a nonsense mu.
    """
    cm, cA = 1e6, 1e3
    # mol/l, a realistic soil solution
    m = dict(Ca=1.0e-3, Mg=6.0e-4, Na=4.0e-4, K=1.0e-4, H=10 ** -6.5,
             OH=1e-14 / 10 ** -6.5, HCO3=1.2e-3, CO3=3.0e-6, An=9.0e-4,
             Al=2.0e-9, AlOH=5.0e-9, AlOH2=2.0e-8, AlOH4=3.0e-7)
    z = dict(Ca=2, Mg=2, Na=1, K=1, H=1, OH=-1, HCO3=-1, CO3=-2, An=-1,
             Al=3, AlOH=2, AlOH2=1, AlOH4=-1)
    want = 0.5 * sum(m[k] * z[k] ** 2 for k in m)

    got = activity.ionic_strength(
        m["Ca"] * cm, m["Mg"] * cm, m["Na"] * cm, m["K"] * cm, m["H"] * cm,
        m["OH"] * cm, m["HCO3"] * cm, m["CO3"] * cm, m["An"] * cm,
        m["Al"] * cm * cA, m["AlOH"] * cm * cA, m["AlOH2"] * cm * cA,
        m["AlOH4"] * cm * cA, cm, cA)
    rel = abs(got - want) / want
    ok = rel < 1e-14
    print("mu units   %s  model-units %.8e vs mol/l sum %.8e  rel %.2e"
          % ("PASS" if ok else "FAIL", got, want, rel))

    # gamma = 1 must be an exact identity, not an approximation to one -- this
    # is what the bit-identity gate rests on.
    base = (1e-2, 1e-8, 1e-8, 1e-5, 1e-5, 1e-6, 1e-6, 3.6, 1e-2, 5.6, 2e-2, 4e-6)
    eff = activity.effective_constants(activity.GAMMA_ONE, *base)
    exact = all(a == b for a, b in zip(base, eff))
    print("gamma=1    %s  effective_constants is an exact identity"
          % ("PASS" if exact else "FAIL"))
    return 0 if (ok and exact) else 1


# ---------------------------------------------------------------------------
# F4.4 -- what turning the flag on actually does, case by case
# ---------------------------------------------------------------------------

def _force_gamma_on(flag):
    """Run every case with gamma_on=flag without editing a notebook.

    The notebooks call smew.biogeochem_balance positionally and know nothing
    about F4, so the flag is injected by wrapping the package-level name. This
    is the F3 Part-I pattern: measure by wrapping, not by editing the tree, so
    the numbers describe the committed code exactly.
    """
    import smew
    import smew.biogeochem as bg
    orig = bg.biogeochem_balance

    def wrapped(*a, **kw):
        kw.setdefault("gamma_on", flag)
        return orig(*a, **kw)

    smew.biogeochem_balance = wrapped
    bg.biogeochem_balance = wrapped
    return orig


def _restore(orig):
    import smew
    import smew.biogeochem as bg
    smew.biogeochem_balance = orig
    bg.biogeochem_balance = orig


CASES = (("Vials_Dietzen", None), ("Mesocosm_Kelland", None),
         ("Mesocosm_Amann", None), ("Bottles_tePas", None),
         ("Example", None), ("Vulkaneifel", "wrapper"),
         ("Synthetic_application", "synthetic"))


def _build(stem, kind, n_out=500):
    import freeze_cases as fc
    if kind == "wrapper":
        return fc.build_wrapper(n_out)
    if kind == "synthetic":
        return fc.build_synthetic(n_out)
    g = fc.exec_notebook(os.path.join(REPO, "Examples", stem + ".ipynb"))
    return fc.build(g, n_out)


def check_shift(n_out=500, out_path=None):
    """Table of what gamma_on=True moves, per case and per series.

    Not a pass/fail. F4.3 and F4.5 are supposed to change results; this is the
    record of BY HOW MUCH, which is the deliverable even if the recalibration
    in F4.6 is never done.
    """
    sys.path.insert(0, os.path.join(REPO, "tests"))
    import freeze_cases  # noqa: F401  (sets up sys.path for the notebooks)

    pay = {}
    for flag in (False, True):
        orig = _force_gamma_on(flag)
        try:
            for stem, kind in CASES:
                pay[(stem, flag)] = _build(stem, kind, n_out)
        finally:
            _restore(orig)

    def ser(pl):
        return sorted({k.rsplit(".", 1)[0] for k in pl
                       if k.endswith(".pH") and not k.startswith("_")})

    def g(pl, pre, k):
        key = pre + "." + k
        return np.asarray(pl[key], dtype=float) if key in pl else None

    rows = []
    for stem, _ in CASES:
        a, b = pay[(stem, False)], pay[(stem, True)]
        for pre in ser(a):
            ok = (g(a, pre, "pH") > 0) & np.isfinite(g(a, pre, "pH"))
            if not ok.any():
                continue
            m = lambda pl, k: np.median(g(pl, pre, k)[ok])
            Wa, Wb = g(a, pre, "W_CaCO3"), g(b, pre, "W_CaCO3")
            pa = 100.0 * np.mean(Wa[ok] < 0) if Wa is not None else np.nan
            pb = 100.0 * np.mean(Wb[ok] < 0) if Wb is not None else np.nan
            ca, cb = g(a, pre, "CaCO3"), g(b, pre, "CaCO3")
            ew_a, ew_b = g(a, pre, "EW__sum"), g(b, pre, "EW__sum")
            rows.append(dict(
                case=stem, series=pre,
                mu=m(b, "mu"), gH=m(b, "gamma_H"),
                dpH=m(b, "pH") - m(a, "pH"),
                r_Alk=m(b, "Alk") / m(a, "Alk") if m(a, "Alk") else np.nan,
                r_DIC=m(b, "DIC") / m(a, "DIC") if m(a, "DIC") else np.nan,
                r_Ca=m(b, "Ca") / m(a, "Ca") if m(a, "Ca") else np.nan,
                cc_a=np.max(ca[ok]), cc_b=np.max(cb[ok]),
                prec_a=pa, prec_b=pb,
                r_EW=(float(np.sum(ew_b)) / float(np.sum(ew_a))
                      if ew_a is not None and float(np.sum(ew_a)) else np.nan),
                r_Mrock=(m(b, "M_rock") / m(a, "M_rock")
                         if m(a, "M_rock") else np.nan),
            ))

    hdr = ("%-21s %-17s %8s %6s %7s %6s %6s %6s %11s %11s %6s %6s %6s %6s"
           % ("case", "series", "mu", "g(H)", "d pH", "xAlk", "xDIC", "xCa",
              "CaCO3 off", "CaCO3 on", "%pr o", "%pr on", "xEW", "xRock"))
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append("%-21s %-17s %8.2e %6.3f %+7.3f %6.3f %6.3f %6.3f "
                     "%11.4e %11.4e %6.1f %6.1f %6.3f %6.3f"
                     % (r["case"], r["series"], r["mu"], r["gH"], r["dpH"],
                        r["r_Alk"], r["r_DIC"], r["r_Ca"], r["cc_a"], r["cc_b"],
                        r["prec_a"], r["prec_b"], r["r_EW"], r["r_Mrock"]))
    txt = "\n".join(lines)
    print(txt)
    if out_path:
        io.open(out_path, "w", encoding="utf-8").write(txt + "\n")
        print("\nwritten to %s" % out_path)
    return 0


def check_cec(t_end=365):
    """How much the model depends on MU_REF_CEC, which is an estimate.

    The Gaines-Thomas selectivity coefficients are concentration-based by their
    source's own statement (de Vries & Posch 2003, p.13), so a model working in
    activities has to lift them to thermodynamic constants at the ionic
    strength of the calibration data. That ionic strength is not recoverable
    from the report, so it is an estimate -- and an estimate that feeds a
    constant deserves to have its sensitivity stated rather than assumed small.
    """
    from smew.harness import example_run
    rows = []
    for mr in (2e-3, 5e-3, 1e-2, 2e-2):
        d, _, _ = example_run(t_end=t_end, gamma_on=True, mu_ref_CEC=mr)
        ok = d["pH"] > 0
        rows.append((mr, float(np.median(d["f_Ca"][ok])),
                     float(np.median(d["f_Na"][ok])),
                     float(np.median(d["Alk"][ok])),
                     float(np.median(d["pH"][ok]))))
    print("%-11s %9s %9s %11s %9s" % ("mu_ref_CEC", "f_Ca", "f_Na", "Alk", "pH"))
    for r in rows:
        print("%-11.1e %9.5f %9.5f %11.4e %9.5f" % r)
    span = max(abs(a[i] / b[i] - 1.0) for i in (1, 2, 3)
               for a in rows for b in rows if b[i])
    dpH = max(abs(a[4] - b[4]) for a in rows for b in rows)
    ok = span < 0.05 and dpH < 0.02
    print("\n%s  over a 10x range in mu_ref_CEC: worst series shift %.2f %%, "
          "pH shift %.4f" % ("PASS" if ok else "FAIL", 100 * span, dpH))
    print("      The estimate is not a material uncertainty. For scale, the "
          "report's own\n      standard deviations on these constants are "
          "0.35-0.79 log units (factors of 2-6).")
    return 0 if ok else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else "units"
    if mode == "identity":
        return check_identity(rtol=float(argv[1]) if len(argv) > 1 else 0.0)
    if mode == "phreeqc":
        return check_phreeqc()
    if mode == "units":
        return check_units()
    if mode == "cec":
        return check_cec()
    if mode == "shift":
        return check_shift(out_path=os.path.join(REPO, "tests", "golden",
                                                 "F4_shift.txt"))
    if mode == "all":
        return check_units() | check_phreeqc() | check_identity()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
