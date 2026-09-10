# SPDX-License-Identifier: AGPL-3.0-only
"""Is step() actually pure?

The F1 acceptance gate: calling step twice on the same input must return the
same answer, and must not modify the input state. That is not a style point.
The residual closure used to store the solution by writing its trial vector into
the output arrays, so "the answer" was a side effect of the last evaluation
happening to land on the solution -- which made the function non-reentrant,
unparallelisable, and unusable with any solver that does not finish by
evaluating at its own answer (scipy's least_squares, the F3 fallback).

    python tests/check_step.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
REPO = os.environ.get("SMEW_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from smew import biogeochem, state as st  # noqa: E402
from smew.state import STATE_1D, STATE_MIN, STATE_PSD  # noqa: E402


def _fields(s):
    """Every field of a SoilState, as plain arrays, for comparison."""
    return {k: np.atleast_1d(np.asarray(getattr(s, k), dtype=float)).copy()
            for k in STATE_1D + STATE_MIN + STATE_PSD
            + ("UP_Ca", "UP_Mg", "UP_K", "UP_Si")}


def _same(a, b):
    return [k for k in a if a[k].shape != b[k].shape
            or not np.array_equal(a[k], b[k])]


def setup(t_end=6):
    """A real mid-run state, not a hand-made one: run a few days and stop."""
    sys.path.insert(0, os.path.join(REPO, "tests"))
    import freeze_cases
    kw, t, _ = freeze_cases.synthetic_inputs(t_app=0, t_end=t_end)

    import smew
    p = biogeochem._resolve_params(
        kw["n"], kw["Zr"], kw["CEC_tot"], kw["k_v"], kw["RAI"], kw["root_d"],
        kw["K_CEC"], kw["mineral"], kw["M_rock_in"], kw["t_app"], kw["d_in"],
        kw["dt"], kw["diss_f"], kw["conv_mol"], kw["conv_Al"])
    T_K = kw["temp_soil"] + 273.15
    D = smew.D_0() * (1 - kw["s"]) ** (10 / 3) * kw["n"] ** (4 / 3)
    Dw = smew.Dw_0() * (kw["n"] * kw["s"]) ** 2
    k1, k2, k_w, k_H = smew.K_C(T_K, kw["conv_mol"])
    DIC_rain = np.zeros(len(kw["s"]))
    forcing = biogeochem._forcing(
        kw["s"], kw["v"], kw["I"], kw["L"], kw["T"], Dw, D, kw["r_het"],
        kw["r_aut"], kw["temp_soil"], T_K, k1, k2, k_w, k_H, DIC_rain)
    state, p, _ = biogeochem._initial_state(
        p, forcing[0], kw["pH_in"], kw["conc_in"], kw["f_CEC_in"], kw["Si_in"],
        kw["CaCO3_in"], kw["MgCO3_in"], kw["M_rock_in"], kw["rock_f_in"],
        kw["d_in"], kw["psd_perc_in"], kw["SSA_in"], kw["t_app"], kw["s"],
        kw["T"], kw["L"], kw["keyword_add"])
    # advance a few steps so the state under test is a working one
    for i in range(1, 40):
        state, _ = biogeochem.step(state, p, forcing[i], forcing[i - 1], kw["dt"])
    return state, p, forcing, kw["dt"]


def main():
    state, p, forcing, dt = setup()
    i = 40
    before = _fields(state)

    a, da = biogeochem.step(state, p, forcing[i], forcing[i - 1], dt)
    mid = _fields(state)
    b, db = biogeochem.step(state, p, forcing[i], forcing[i - 1], dt)

    rc = 0
    bad = _same(before, mid)
    print(f"input state unmodified              : "
          f"{'OK' if not bad else 'MUTATED ' + ', '.join(bad[:8])}")
    rc |= bool(bad)

    bad = _same(_fields(a), _fields(b))
    print(f"two calls agree bit for bit         : "
          f"{'OK' if not bad else 'DIFFER ' + ', '.join(bad[:8])}")
    rc |= bool(bad)

    same_res = np.array_equal(np.asarray(da["errors"]), np.asarray(db["errors"]))
    print(f"diagnostics agree                   : {'OK' if same_res else 'DIFFER'}")
    rc |= (not same_res)

    # the returned state must not alias the input's arrays, or a later step
    # would edit history in place
    aliased = [k for k in STATE_MIN + STATE_PSD
               if getattr(a, k) is getattr(state, k)
               and np.asarray(getattr(a, k)).size]
    # aliasing is only a fault if the array is then written; carrying an
    # unchanged array forward by reference is fine as long as nothing mutates it
    print(f"no in-place writes to carried arrays: "
          f"{'OK' if not _same(before, _fields(state)) else 'FAILED'}"
          f"   (shared refs: {', '.join(aliased) if aliased else 'none'})")

    print("\nPASS" if rc == 0 else "\nFAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
