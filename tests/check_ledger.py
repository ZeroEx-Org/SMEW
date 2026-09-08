# SPDX-License-Identifier: AGPL-3.0-only
"""Run the mass-balance ledger on every benchmark case and gate on closure.

This is a checker, not a fixture: nothing is written to disk and no frozen file
is read. The ledger asks an invariant question -- does this run conserve mass --
whose answer is known in advance, so it needs no stored reference. That is also
why it is unaffected by the open k_dec question.

    python tests/check_ledger.py                  # all cases
    python tests/check_ledger.py --only example
    python tests/check_ledger.py --self-test      # prove the ledger can fail
    python tests/check_ledger.py --quiet          # one line per case

SMEW_REPO=<worktree> runs it against another code state, same as freeze_cases.py.
"""
import argparse
import os
import sys
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")
REPO = os.environ.get("SMEW_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
from smew import harness, ledger  # noqa: E402

NOTEBOOKS = ["Vials_Dietzen.ipynb", "Mesocosm_Kelland.ipynb", "Mesocosm_Amann.ipynb",
             "Bottles_tePas.ipynb", "Example.ipynb"]


def example_case():
    """harness.example_run -- the seeded, self-contained case. Fastest to iterate on."""
    data, t, extra = harness.example_run(with_water=True)
    yield "example", data, {"E": extra["E"]}, {"SOC": extra["SOC"]}


def cold_case():
    """The same case driven below freezing. Two configurations.

    Not benchmarks: synthetic cases that exist to exercise the one branch no
    comparison notebook reaches. None of the five ever drops below zero, so the
    freeze/thaw index mapping -- skipped steps, the state held at
    idx_before_freezing, the flux consumed twice, the uptake array overwritten at
    `last` -- would otherwise be reconstructed but never tested.

    Soil temperature at depth Zr is seasonally damped, so a one-year run crosses
    zero at most once. Hence two cases:

      example_cold        starts frozen, which is the only way to reach the
                          idx_before_freezing = 1 fallback (biogeochem.py:367)
                          where the model resumes from the initial conditions
      example_freezethaw  two winters, so two mid-run freeze/thaw transitions
                          with a real held state on either side
    """
    data, t, extra = harness.example_run(with_water=True, temp_av=2, day1=1)
    yield "example_cold", data, {"E": extra["E"]}, {"SOC": extra["SOC"]}
    data, t, extra = harness.example_run(with_water=True, temp_av=2, day1=250,
                                         t_end=730)
    yield "example_freezethaw", data, {"E": extra["E"]}, {"SOC": extra["SOC"]}


def notebook_cases(nb):
    """Every biogeochem result dict a notebook leaves behind.

    Water and organic carbon live outside biogeochem_balance, so they come from
    the notebook namespace. A notebook with several treatments leaves only the
    LAST moisture and carbon series in its globals, which would silently belong
    to a different case -- so hand them over only when the namespace's moisture
    series is bit-identical to the one this result was actually computed with.
    """
    import freeze_cases
    g = freeze_cases.exec_notebook(os.path.join(REPO, "Examples", nb))
    results, _ = freeze_cases.payload_for(g)
    for name in sorted(results):
        data = results[name]
        water = oc = None
        s_nb = g.get("s")
        same_moisture = (isinstance(s_nb, np.ndarray)
                         and np.array_equal(np.asarray(s_nb, dtype=float),
                                            np.asarray(data["s"], dtype=float)))
        if same_moisture and isinstance(g.get("E"), np.ndarray):
            water = {"E": g["E"]}
        if same_moisture and isinstance(g.get("SOC"), np.ndarray):
            oc = {"SOC": g["SOC"]}
        yield f"{nb.replace('.ipynb','')}:{name}", data, water, oc


def run(a):
    cases = []
    if a.only:
        for want in a.only:
            if want.lower() == "example":
                cases.append(("example", example_case, None))
            elif want.lower() in ("cold", "example_cold"):
                cases.append(("example_cold", cold_case, None))
            else:
                nb = want if want.endswith(".ipynb") else want + ".ipynb"
                cases.append((nb, notebook_cases, nb))
    else:
        cases.append(("example", example_case, None))
        cases.append(("example_cold", cold_case, None))
        for nb in NOTEBOOKS:
            cases.append((nb, notebook_cases, nb))

    rc = 0
    n_case = n_ok = 0
    for label, fn, arg in cases:
        t0 = time.perf_counter()
        try:
            produced = list(fn(arg) if arg else fn())
        except Exception as e:
            print(f"{label:28s} ERROR  {type(e).__name__}: {e}")
            print("   " + "\n   ".join(traceback.format_exc().strip().splitlines()[-3:]))
            rc = 1
            continue
        el = time.perf_counter() - t0

        for name, data, water, oc in produced:
            n_case += 1
            led = ledger.build(data, water=water, oc=oc)
            rows = ledger.closure(led)
            ok = ledger.passed(led, rows)
            n_ok += bool(ok)
            worst = max((r for r in rows if not r["inactive"]),
                        key=lambda r: max(r["step_rel"], r["rel"]), default=None)
            gate = (f"worst {worst['element']} step {worst['step_rel']:.1e} "
                    f"run {worst['rel']:.1e}" if worst else "no active pool")
            print(f"{name:28s} {'CLOSED' if ok else 'NOT CLOSED':10s} "
                  f"{gate}  ({el:5.1f}s run)")
            if not ok:
                for why in ledger.failures(led, rows):
                    print(f"      FAIL  {why}")
                rc = 1
            if not a.quiet:
                print()
                print("  " + ledger.report(led).replace("\n", "\n  "))
                print()
            if a.self_test:
                surv, inactive = ledger.self_test(led)
                print(f"    self-test: {len(surv)} undetectable, "
                      f"{len(inactive)} inactive (zero in this case)")
                for s in surv:
                    print(f"      UNDETECTABLE  {s}")
                    rc = 1
                if not a.quiet and inactive:
                    print(f"      inactive: {', '.join(inactive)}")
            el = 0.0  # run cost belongs to the first result of a notebook

    print()
    print(f"{n_ok}/{n_case} cases closed")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help="case names: example, or a notebook stem")
    ap.add_argument("--quiet", action="store_true", help="one line per case")
    ap.add_argument("--self-test", action="store_true",
                    help="perturb every term and require closure to break")
    a = ap.parse_args()
    os.chdir(REPO)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
