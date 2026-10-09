# SPDX-License-Identifier: AGPL-3.0-only
"""The rock is represented twice. Gate on the two accounts staying together.

SMEW carries the amendment in two independent ways and nothing has ever checked
that they agree:

  stoichiometric   M_min -= EW * MM_min * dt, summed to M_rock [g]. This is the
                   account the chemistry runs on: every mole of cation released
                   to solution is subtracted from here, and smew.ledger balances
                   against it.

  geometric        each diameter class shrinks by d_shrink, and psd_evol
                   rescales its mass as (d/d_0)**3. The total is
                   sum(psd * delta_d) [g]. This is the account the SURFACE AREA
                   runs on, and surface area multiplies every dissolution rate.

They are not two views of one number -- they are two separate integrations of
one rock, and a model can happily dissolve mass out of one while the other goes
its own way. F0 flagged this as an open thread.

THEY DO AGREE, AND NOT BY LUCK

    route A   sum_m EW_m * MM_m * dt  =  SA * rho_rock * d_shrink
    route B   sum_k |dmass_k|         =  6 * d_shrink * sum_k lamb_k*mass_k/d_k
    with      SA = sum_k SSA_k*mass_k =  (6/rho_rock) * sum_k lamb_k*mass_k/d_k
    => identical to first order in d_shrink.

So what is tested here is not equality -- that is algebra -- but DRIFT: the
first-order Euler error, and any term that breaks the identity. There is exactly
one such term, clip_psd, the mass dropped when a class shrinks to d <= 0, so it
is added back before the comparison. If the two accounts ever part company for
any other reason, this is what says so.

    python tests/check_psd.py                  # all cases
    python tests/check_psd.py --only example
    python tests/check_psd.py --quiet

WHY DRIFT AND NOT THE ABSOLUTE GAP
    The gap at the application step is set by whether psd_perc_in sums to 1, a
    property of the INPUT, not of the integration. Bottles_tePas anorthite sums
    to 0.997506, so its geometric rock starts 0.25% light and stays there --
    _validate_psd warns about that, and it is reported here separately as the
    static offset. Folding it into the drift would hide the thing being tested.

SMEW_REPO=<worktree> runs it against another code state, same as freeze_cases.py.
"""
import argparse
import os
import sys
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")
REPO = os.environ.get("SMEW_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import smew  # noqa: E402
from smew import biogeochem, harness  # noqa: E402

NOTEBOOKS = ["Vials_Dietzen.ipynb", "Mesocosm_Kelland.ipynb", "Mesocosm_Amann.ipynb",
             "Bottles_tePas.ipynb", "Example.ipynb"]

# Drift allowed between the two accounts, as a fraction of rock mass. The scheme
# is first-order explicit in the diameter, so the error is O(dt) and the worst
# measured case -- Example, which dissolves 61% of its rock -- sits at 5.6e-6.
# 1e-4 is ~20x that: loose enough not to fire on a discretisation choice, tight
# enough that any real decoupling of the two accounts trips it.
DRIFT_TOL = 1e-4

# A static offset larger than this is reported as a finding. It is not a gate:
# it is an input defect, _validate_psd already warns about it on load, and
# failing here would break a published benchmark a second time for one cause.
# 1e-6 is above the ~1e-8 summation roundoff a hundred-class distribution
# carries, and 2500x below the one real offender (Bottles_tePas at 2.5e-3).
OFFSET_REPORT = 1e-6


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
# The particle size distribution is deliberately NOT in the output contract
# (smew.harness.CONTRACT) -- notebooks plot surface area, not psd. Rather than
# widen the contract for one test, the step function is wrapped: every state the
# model builds, including sub-stepped intermediates and candidates the
# sub-stepper later rejects, is recorded as it is made.

_TRACE = []


def _install():
    """Wrap _step_once to record both accounts, and biogeochem_balance to segment."""
    step0 = biogeochem._step_once
    bal0 = smew.biogeochem_balance

    def step(state, params, *a, **k):
        new, diag = step0(state, params, *a, **k)
        if params.has_rock and _TRACE:
            _TRACE[-1].append((float(new.M_rock),
                               float(np.sum(new.psd * new.delta_d)),
                               float(new.clip_psd)))
        return new, diag

    def balance(*a, **k):
        _TRACE.append([])
        try:
            return bal0(*a, **k)
        finally:
            pass

    biogeochem._step_once = step
    smew.biogeochem_balance = balance
    biogeochem.biogeochem_balance = balance


def _take():
    """Hand back the traces recorded since the last call, and clear."""
    out = [t for t in _TRACE if t]
    _TRACE.clear()
    return out


def measure(trace):
    """Drift, static offset and dissolved mass for one biogeochem_balance call."""
    M = np.array([r[0] for r in trace])
    G = np.array([r[1] for r in trace])
    clip = np.cumsum([r[2] for r in trace])

    live = M > 0
    if not live.any():
        return None
    M, G, clip = M[live], G[live], clip[live]

    gap = G - M
    # clip_psd leaves the geometric account only; add it back so what remains is
    # integration error alone.
    drift = (gap + clip) - (gap[0] + clip[0])
    scale = M[0] if M[0] > 0 else 1.0
    dissolved = M[0] - M[-1]
    return {"drift": float(np.max(np.abs(drift)) / scale),
            "offset": float(gap[0] / scale),
            "clip": float(clip[-1] / scale),
            "dissolved": float(dissolved / scale),
            "M0": float(M[0]), "n": int(M.size)}


# ---------------------------------------------------------------------------
# Cases -- the same set check_ledger.py runs
# ---------------------------------------------------------------------------

def example_case():
    harness.example_run(with_water=True)
    yield "example", _take()


def vulkaneifel_case():
    sys.path.insert(0, os.path.join(REPO, "wrappers_postprocessing"))
    import zeroex_input_data_wrapper as wp
    inp = wp.read_input_data(project_name="Vulkaneifel", value_col="Field C")
    wp.run_SMEW(project_name="Vulkaneifel", input_data=inp,
                interpolate_frozen=False, seed=42)
    yield "Vulkaneifel", _take()


def notebook_cases(nb):
    import freeze_cases
    freeze_cases.exec_notebook(os.path.join(REPO, "Examples", nb))
    yield nb.replace(".ipynb", ""), _take()


def run(a):
    sources = []
    if a.only:
        for want in a.only:
            if want.lower() == "example":
                sources.append(("example", example_case))
            elif want.lower().startswith("vulk"):
                sources.append(("Vulkaneifel", vulkaneifel_case))
            else:
                nb = next((n for n in NOTEBOOKS if want.lower() in n.lower()), None)
                if nb is None:
                    raise SystemExit(f"no case matching {want!r}")
                sources.append((nb, lambda nb=nb: notebook_cases(nb)))
    else:
        sources = ([("example", example_case)]
                   + [(nb, lambda nb=nb: notebook_cases(nb)) for nb in NOTEBOOKS]
                   + [("Vulkaneifel", vulkaneifel_case)])

    _install()
    bad, offsets, clipped, rows = [], [], [], []
    for label, fn in sources:
        try:
            for name, traces in fn():
                for j, tr in enumerate(traces):
                    m = measure(tr)
                    if m is None:
                        continue
                    tag = name if len(traces) == 1 else f"{name}[{j}]"
                    rows.append((tag, m))
                    if m["drift"] > DRIFT_TOL:
                        bad.append((tag, m))
                    if abs(m["offset"]) > OFFSET_REPORT:
                        offsets.append((tag, m))
                    if abs(m["clip"]) > 0:
                        clipped.append((tag, m))
        except Exception:
            print(f"  {label}: FAILED TO RUN")
            traceback.print_exc()
            bad.append((label, None))

    if not rows:
        raise SystemExit("no case carried rock: nothing was tested")

    w = max(len(t) for t, _ in rows)
    if not a.quiet:
        print(f"{'case':<{w}}  {'steps':>6}  {'M_rock0 [g]':>12}  "
              f"{'dissolved':>10}  {'drift':>10}  {'offset':>10}  {'clip_psd':>10}")
        print("-" * (w + 68))
    for tag, m in rows:
        print(f"{tag:<{w}}  {m['n']:>6}  {m['M0']:>12.4g}  "
              f"{m['dissolved']:>9.2%}  {m['drift']:>10.2e}  "
              f"{m['offset']:>10.2e}  {m['clip']:>10.2e}")

    print()
    print(f"drift   max |d(sum psd*delta_d - M_rock)| / M_rock, "
          f"clip_psd added back.  tol {DRIFT_TOL:.0e}")
    print("offset  the two accounts' gap at application. Non-zero means "
          "psd_perc_in does not sum to 1.")
    print("clip_psd  mass dropped because a diameter class reached d <= 0.")
    print()

    if offsets:
        print("STATIC OFFSET (an input defect, not a drift):")
        for tag, m in offsets:
            print(f"  {tag}: psd accounts for {1 + m['offset']:.6f} of M_rock")
        print()
    if clipped:
        print("CLASSES EXHAUSTED (geometric account lost mass the other kept):")
        for tag, m in clipped:
            print(f"  {tag}: {m['clip']:.3e} of M_rock")
        print()

    if bad:
        print(f"FAIL: {len(bad)} case(s) above tolerance")
        for tag, m in bad:
            if m is not None:
                print(f"  {tag}: drift {m['drift']:.3e} > {DRIFT_TOL:.0e}")
        return 1
    print(f"PASS: {len(rows)} case(s), the two accounts of the rock agree")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help="case name fragments; default is all")
    ap.add_argument("--quiet", action="store_true", help="skip the header")
    raise SystemExit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
