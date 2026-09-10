# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze golden masters for every comparison notebook, by EXECUTING the notebook
and capturing the result dicts it leaves in its namespace.

This is deliberately not a re-implementation of each pipeline: a hand-written case
module can silently drift from the notebook it is meant to mirror, and then the
golden master guards the copy rather than the real thing.

    python freeze_cases.py freeze [--out DIR] [--n-out N]
    python freeze_cases.py check  [--out DIR] [--n-out N]

Notebooks with unseeded stochastic rainfall are skipped for `check` unless
--allow-unseeded, since they cannot be reproduced.
"""
import argparse
import io
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")
# SMEW_REPO lets the same freezer run against an older code state in a git
# worktree, so pre-change fixtures come from genuinely old code rather than a
# monkeypatch.
REPO = os.environ.get("SMEW_REPO", "/Users/estebangaitan/Documents/Github/SMEW")
sys.path.insert(0, REPO)
from smew import harness  # noqa: E402

# a dict is a biogeochem result if it carries these
MARKERS = {"pH", "Alk", "M_rock", "Ca_tot"}

NOTEBOOKS = ["Vials_Dietzen.ipynb", "Mesocosm_Kelland.ipynb", "Mesocosm_Amann.ipynb",
             "Bottles_tePas.ipynb", "Example.ipynb"]
# not a notebook: driven through wrappers_postprocessing. See build_wrapper.
WRAPPER_CASE = "Vulkaneifel"
# Notebooks whose rainfall is stochastic AND unseeded cannot be reproduced, so
# there is nothing for `check` to compare against. Empty since F0.1: Example.ipynb
# now passes seed=42 to rain_stoc. Kept as the mechanism, not as a fact -- a new
# notebook may well arrive unseeded.
UNSEEDED = set()


def exec_notebook(path):
    import matplotlib
    matplotlib.use("Agg")
    nb = json.load(io.open(path, encoding="utf-8"))
    g = {"__name__": "__main__", "__file__": path}
    for i, c in enumerate([c for c in nb["cells"] if c["cell_type"] == "code"]):
        src = "".join(c["source"])
        if src.strip():
            exec(compile(src, f"<cell {i}>", "exec"), g)
    return g


def payload_for(g):
    """One flat payload covering every result dict in the notebook namespace."""
    t = g.get("t")
    if t is None:
        raise RuntimeError("notebook left no time vector 't'")
    t = np.asarray(t, dtype=float)
    results = {k: v for k, v in g.items()
               if not k.startswith("_") and isinstance(v, dict) and MARKERS <= set(v)}
    if not results:
        raise RuntimeError("notebook left no biogeochem result dict")
    return results, t


def build(g, n_out):
    results, t = payload_for(g)
    merged = {}
    for name in sorted(results):
        # Extras that live outside biogeochem_balance but are plotted.
        #
        # Only rain. "s" and "v" used to be taken from here too, and that was
        # wrong: a notebook namespace holds only the LAST value assigned, so
        # every case got the last treatment's moisture and biomass. Harmless
        # where all treatments share one forcing, but Amann's three _nocrop
        # cases have their own -- the frozen v was off by the whole 500 g/m2 of
        # biomass and s by 0.176. Both are in harness.CONTRACT now, so they come
        # from each case's own result dict, which is correct by construction.
        # check_ledger.notebook_cases has always guarded this; this did not.
        extra = {k: np.asarray(g[k], dtype=float) for k in ("rain",)
                 if isinstance(g.get(k), np.ndarray)
                 and np.asarray(g[k]).shape[-1:] == (len(t),)}
        p = harness.collect(results[name], t, extra=extra, n_out=n_out)
        for k, v in p.items():
            merged[k if k.startswith("_") or k == "t" else f"{name}.{k}"] = v
    merged["_cases"] = np.array(sorted(results), dtype="U32")
    return merged


# ---------------------------------------------------------------------------
# The wrapper path
# ---------------------------------------------------------------------------
# Vulkaneifel is not a notebook case. Its notebook is two calls into
# wrappers_postprocessing, so driving those directly is both faster and a truer
# test: it is the only case that exercises read_input_data, the xlsx data sheet,
# run_SMEW, day1=291 (an October start, which is what the seasonal rain rotation
# was added for) and four minerals at once.
#
# Two payloads, because run_SMEW does two separable things:
#   raw     -- interpolate_frozen=False, the model's own output. This is what F1
#              must preserve, and the only one a model regression can be read off.
#   interp  -- the default, with the 13 series rewritten across frozen steps.
#              Frozen separately so a change in the interpolation shows up as a
#              change in `interp` alone, instead of being indistinguishable from
#              a change in the model.
VULKANEIFEL = ("Vulkaneifel", "Field C")

# A synthetic case, for the branches the real ones do not reach.
#
# Every notebook and the Vulkaneifel sheet apply rock at t_app = 0, so tt_app is
# 0 and the loop is post-application from its very first step. The code that
# reads the rock geometry AT the application index therefore had no coverage at
# all -- it was verified once, against the pre-F1 implementation, and that
# comparison is not repeatable because the old code is gone. This fixture makes
# it permanent. It also carries several particle-diameter classes, a measured
# SSA (which recalibrates the fractal prefactor) and two minerals, none of which
# the single-mineral cases exercise together.
SYNTHETIC_CASE = "Synthetic_application"


def synthetic_inputs(t_app=30, t_end=90, dt=1 / (24 * 6), seed=42,
                     mineral=("forsterite", "anorthite"),
                     d_in=None, psd_perc_in=None, SSA_in=0.5):
    """Inputs for the synthetic case, stated explicitly like harness.example_run."""
    import smew
    t = np.arange(0, t_end, dt)
    conv_mol, conv_Al = 1e6, 1e3
    soil, Zr, rho_bulk = "loam", 0.3, 1.2e6
    latitude, altitude = 40 * np.pi / 180, 33
    wind = 1 * np.ones(len(t))
    temp_air, temp_soil, temp_min, temp_max = smew.temp(
        latitude, 13, 11, 5, Zr, t_end, dt, 1)
    ET0 = smew.ET0(latitude, altitude, temp_air, temp_soil, temp_min, temp_max,
                   wind, 0.25, Zr, False, t_end, dt, 1)
    rain = smew.rain_stoc(0.25, (1.2 / 0.25) / 365, t_end, dt, seed=seed)
    v = smew.veg(3000, 100, 3000, 0, temp_soil, dt)
    s, s_w, s_i, I, L, T, E, Q, Irr, n = smew.moisture_balance(
        rain, Zr, soil, ET0, v, 3000, 1, 0.5, t_end, dt)
    SOC, r_het, r_aut, D = smew.respiration(
        1, rho_bulk * 0.05 / 100, 10 * smew.CO2_atm(conv_mol), 1, soil, s, v,
        3000, Zr, temp_soil, dt, conv_mol)
    f_CEC_in = np.array([0.30, 0.15, 0.10, 0.05, 0.00, 0.40])
    conc_in, K_CEC = smew.f_CEC_to_conc(f_CEC_in, 4, soil, conv_mol, conv_Al)
    if d_in is None:
        d_in = np.array([50, 100, 200, 400]) * 1e-6
        psd_perc_in = np.array([0.25, 0.25, 0.25, 0.25])
    return dict(n=n, s=s, L=L, T=T, I=I, v=v, k_v=3000, RAI=10, root_d=0.4e-3,
                Zr=Zr, r_het=r_het, r_aut=r_aut, D=D, temp_soil=temp_soil,
                pH_in=4, conc_in=conc_in, f_CEC_in=f_CEC_in, K_CEC=K_CEC,
                CEC_tot=10 * 1e-5 * rho_bulk * Zr * conv_mol,
                Si_in=0, CaCO3_in=0, MgCO3_in=0, M_rock_in=1000, t_app=t_app,
                mineral=list(mineral),
                rock_f_in=np.ones(len(mineral)) / len(mineral),
                d_in=d_in, psd_perc_in=psd_perc_in, SSA_in=SSA_in,
                diss_f=1.0, dt=dt, conv_Al=conv_Al, conv_mol=conv_mol,
                keyword_add=1), t, {"rain": rain, "s": s, "v": v}


def build_synthetic(n_out, seed=42):
    import smew
    kw, t, extra = synthetic_inputs(seed=seed)
    data = smew.biogeochem_balance(**kw)
    p = harness.collect(data, t, extra=extra, n_out=n_out)
    merged = {k if k.startswith("_") or k == "t" else f"application.{k}": v
              for k, v in p.items()}
    merged["_cases"] = np.array(["application"], dtype="U32")
    return merged


def build_wrapper(n_out, seed=42):
    sys.path.insert(0, os.path.join(REPO, "wrappers_postprocessing"))
    import zeroex_input_data_wrapper as wp

    project, field = VULKANEIFEL
    inp = wp.read_input_data(project_name=project, value_col=field)
    merged = {}
    for name, interp in (("raw", False), ("interp", True)):
        data = wp.run_SMEW(project_name=project, input_data=inp,
                           interpolate_frozen=interp, seed=seed)
        t = np.asarray(data["t"], dtype=float)
        extra = {k: np.asarray(data[k], dtype=float)
                 for k in ("rain", "s_filtered") if k in data}
        p = harness.collect(data, t, extra=extra, n_out=n_out)
        for k, v in p.items():
            merged[k if k.startswith("_") or k == "t" else f"{name}.{k}"] = v
    merged["_cases"] = np.array(["raw", "interp"], dtype="U32")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "check"])
    ap.add_argument("--out", default=os.path.join(REPO, "tests", "golden"))
    ap.add_argument("--n-out", type=int, default=500)
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--allow-unseeded", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    os.chdir(REPO)
    todo = a.only or NOTEBOOKS + [WRAPPER_CASE, SYNTHETIC_CASE]
    rc = 0
    for nb in todo:
        stem = nb.replace(".ipynb", "")
        dest = os.path.join(a.out, stem + ".npz")
        if a.mode == "check" and nb in UNSEEDED and not a.allow_unseeded:
            print(f"{stem:20s} SKIP   unseeded rainfall, not reproducible")
            continue
        t0 = time.perf_counter()
        try:
            if nb == WRAPPER_CASE:
                p = build_wrapper(a.n_out)
            elif nb == SYNTHETIC_CASE:
                p = build_synthetic(a.n_out)
            else:
                g = exec_notebook(os.path.join(REPO, "Examples", nb))
                p = build(g, a.n_out)
        except Exception as e:
            print(f"{stem:20s} ERROR  {type(e).__name__}: {e}")
            print("   " + "\n   ".join(traceback.format_exc().strip().splitlines()[-3:]))
            rc = 1
            continue
        el = time.perf_counter() - t0
        cases = ", ".join(p["_cases"].tolist())
        if a.mode == "freeze":
            sz = harness.save(dest, p)
            print(f"{stem:20s} FROZE  {el:5.1f}s  {sz/1e3:7.1f} kB  "
                  f"{len(p['t'])} steps  [{cases}]")
        else:
            if not os.path.exists(dest):
                print(f"{stem:20s} MISS   no golden at {dest}")
                rc = 1
                continue
            res = harness.compare(harness.load(dest), p, rtol=a.rtol)
            bad = [r for r in res if not r[2]]
            print(f"{stem:20s} {'PASS' if not bad else 'FAIL'}   {el:5.1f}s  "
                  f"{len(res)-len(bad)}/{len(res)} keys  [{cases}]")
            for k, d, ok in bad[:6]:
                print(f"      {k:28s} max|diff| {d:12.6g}")
            if bad:
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
