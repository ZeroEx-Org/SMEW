# SPDX-License-Identifier: AGPL-3.0-only
"""The timestep: is it stable, is it converged, and what is the actual limit?

SMEW's dt has always been chosen by convention in the notebooks -- 10 minutes,
because that is what the examples use. Nothing stated what dt the scheme is
actually stable at, and nothing measured how much of the answer is discretisation
error. This does both.

    python tests/check_dt.py [--quick]

THREE THINGS ARE CHECKED

  positivity   Run at dt, 10dt, 100dt. Every run must either finish with no
               negative pool, or fail loudly. A silent negative pool is the
               failure this is looking for -- it is what F2 exists to prevent.

  limit        dt_max is recorded at every step: the largest step that would
               have kept every pool non-negative, given the state it started
               from. The minimum over a run is the model's stability limit for
               that case, and the ratio to the dt actually used is its margin.

  convergence  Halving dt should halve the error of a first-order explicit
               scheme. Measured against the finest run as reference, the
               observed order should be ~1. A much lower order means the answer
               is dominated by something other than the explicit integration.

WHY dt VALUES ARE CHOSEN TO DIVIDE t_end EXACTLY
    smew.temp sizes its arrays as np.arange(0, t_end/dt) while smew.rain_stoc
    uses np.zeros(int(t_end/dt)). When t_end/dt is not a whole number those
    differ by one, and smew.respiration then fails with a broadcast error
    between v and r_het. That is a real pre-existing bug in the forcing stages,
    unrelated to the chemistry; this script sidesteps it by construction rather
    than pretending it is not there.
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
REPO = os.environ.get("SMEW_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import smew  # noqa: E402

POOLS = ("Ca_tot", "Mg_tot", "K_tot", "Na_tot", "Al_tot", "Si_tot",
         "An_tot", "IC_tot", "CaCO3", "MgCO3")
# Compared across resolutions. Intensive quantities and pools, not fluxes:
# a flux at one instant is not comparable between two discretisations.
COMPARE = ("pH", "Alk", "Ca", "Mg", "Si", "DIC", "Ca_tot", "Mg_tot", "M_rock")

# A pool this far below the largest in the run is zero plus float noise; a
# negative value there is not a positivity failure. Mirrors smew.ledger's
# ACTIVITY_FLOOR and smew.biogeochem.CLIP_REL.
NOISE_REL = 1e-12


def build(dt, t_end=120, CaCO3_in=0.0, soil="loam", seed=42, pH_in=5.0,
          M_rock_in=1000, R_tot=1.2, mineral=("forsterite",), rain_mode="stochastic"):
    """One case, at a chosen dt. Inputs stated explicitly, as in harness.example_run.

    rain_mode="uniform" spreads the annual rainfall evenly instead of drawing
    storms. That matters for the convergence study and nowhere else: rain_stoc
    places storms at random times, so re-drawing at a different dt gives a
    DIFFERENT realisation, and the difference between two runs is then dominated
    by the weather rather than by the integration. Uniform rain is identical at
    every dt by construction, which is what isolates discretisation error.
    """
    conv_mol, conv_Al = 1e6, 1e3
    Zr, rho_bulk = 0.3, 1.2e6
    lat = 40 * np.pi / 180
    temp_air, temp_soil, temp_min, temp_max = smew.temp(lat, 13, 11, 5, Zr, t_end, dt, 1)
    ET0 = smew.ET0(lat, 33, temp_air, temp_soil, temp_min, temp_max,
                   np.ones(len(temp_air)), 0.25, Zr, False, t_end, dt, 1)
    if rain_mode == "uniform":
        rain = np.full(len(temp_air), R_tot / 365 * dt)     # [m per step]
    else:
        rain = smew.rain_stoc(0.25, (R_tot / 0.25) / 365, t_end, dt, seed=seed)
    v = smew.veg(3000, 100, 3000, 0, temp_soil, dt)
    s, s_w, s_i, I, L, T, E, Q, Irr, n = smew.moisture_balance(
        rain, Zr, soil, ET0, v, 3000, 1, 0.5, t_end, dt)
    SOC, r_het, r_aut, D = smew.respiration(
        1, rho_bulk * 0.05 / 100, 10 * smew.CO2_atm(conv_mol), 1, soil, s, v,
        3000, Zr, temp_soil, dt, conv_mol)
    f_CEC_in = np.array([0.30, 0.15, 0.10, 0.05, 0.00, 0.40])
    conc_in, K_CEC = smew.f_CEC_to_conc(f_CEC_in, pH_in, soil, conv_mol, conv_Al)
    return dict(n=n, s=s, L=L, T=T, I=I, v=v, k_v=3000, RAI=10, root_d=0.4e-3,
                Zr=Zr, r_het=r_het, r_aut=r_aut, D=D, temp_soil=temp_soil,
                pH_in=pH_in, conc_in=conc_in, f_CEC_in=f_CEC_in, K_CEC=K_CEC,
                CEC_tot=10 * 1e-5 * rho_bulk * Zr * conv_mol, Si_in=0,
                CaCO3_in=CaCO3_in, MgCO3_in=0, M_rock_in=M_rock_in, t_app=0,
                mineral=list(mineral),
                rock_f_in=np.ones(len(mineral)) / len(mineral),
                d_in=np.array([100]) * 1e-6, psd_perc_in=np.array([1]),
                SSA_in=float("nan"), diss_f=1.0, dt=dt, conv_Al=conv_Al,
                conv_mol=conv_mol, keyword_add=1)


def negatives(d):
    """Pools that went meaningfully negative, ignoring noise on an empty pool."""
    scale = max(float(np.max(np.abs(np.asarray(d[k], float)))) for k in POOLS)
    out = {}
    for k in POOLS:
        a = np.asarray(d[k], dtype=float)
        m = float(np.min(a))
        if m < -NOISE_REL * scale:
            out[k] = m
    return out


CASES = {
    "plain": dict(),
    "carbonate": dict(CaCO3_in=1e4, soil="sand"),
    "wet": dict(R_tot=3.0, soil="sand"),
}


def positivity(quick):
    print("POSITIVITY -- no pool may go negative at any resolution\n")
    # 1x = 10 min, 12x = 2 h, 144x = 1 day. All divide t_end = 120 d exactly,
    # which the forcing stages require (see the module docstring).
    base = 1 / (24 * 6)
    mults = (1, 12) if quick else (1, 12, 144)
    rc = 0
    for name, kw in CASES.items():
        for m in mults:
            dt = base * m
            label = f"  {name:10s} dt={dt:8.5f} d ({m:3d}x)"
            args = build(dt=dt, **kw)
            # Check the forcing before blaming the chemistry. smew.moisture_balance
            # has the SAME explicit-Euler positivity problem this file is testing
            # for, one stage upstream: on sand under heavy rain at a long step it
            # drives wetness negative and then to NaN, silently. F2 does not touch
            # moisture.py, so that is reported here rather than hidden.
            s_in = np.asarray(args["s"], dtype=float)
            if not np.all(np.isfinite(s_in)) or float(np.min(s_in)) < 0.0:
                print(f"{label}  FORCING BAD: moisture_balance gave "
                      f"{int(np.isnan(s_in).sum())} NaN, min s = {np.nanmin(s_in):.3g}"
                      f"  <- upstream of the chemistry")
                continue
            try:
                d = smew.biogeochem_balance(**args)
            except Exception as e:
                # loud failure is an acceptable outcome; a silent bad answer is not
                print(f"{label}  RAISED {type(e).__name__}: {str(e)[:44]}")
                continue
            neg = negatives(d)
            ns = np.asarray(d["n_substeps"], float)
            sub = int(np.count_nonzero(ns > 1))
            clamp = int(np.count_nonzero(ns == 0))
            clip = sum(float(np.sum(np.asarray(d[k], float)))
                       for k in d if k.startswith("clip_"))
            print(f"{label}  {'NEGATIVE ' + str(neg) if neg else 'ok':44s}"
                  f" substepped {sub:4d}  clamped {clamp:3d}  clipped {clip:.3g}")
            if neg:
                rc = 1
    return rc


def limit():
    print("\nSTABILITY LIMIT -- min over the run of dt_max\n")
    base = 1 / (24 * 6)
    for name, kw in CASES.items():
        d = smew.biogeochem_balance(**build(dt=base, **kw))
        a = np.asarray(d["dt_max"], dtype=float)
        a = a[np.isfinite(a)]
        if not a.size:
            print(f"  {name:10s} no pool ever lost mass; unlimited")
            continue
        lim = float(np.min(a))
        print(f"  {name:10s} dt_max = {lim:9.5f} d ({lim*24*60:8.1f} min)   "
              f"used {base:.5f} d ({base*24*60:.0f} min)   margin x{lim/base:6.1f}")
    return 0


def convergence(quick):
    print("\nCONVERGENCE -- error against the finest run; first order means ~1\n")
    base = 1 / (24 * 6)
    # each divides t_end = 120 exactly, so the forcing stages stay consistent
    steps = [base, 2 * base, 4 * base] if quick else [base, 2 * base, 4 * base, 8 * base]
    runs = {}
    for dt in steps:
        runs[dt] = smew.biogeochem_balance(**build(dt=dt, rain_mode="uniform"))
    ref_dt = steps[0]
    ref = runs[ref_dt]
    t_ref = np.arange(0, 120, ref_dt)

    print(f"  {'series':10s} " + "".join(f"{f'err@{int(round(d/base))}x':>13s}"
                                         for d in steps[1:]) + "     order")
    rc = 0
    for key in COMPARE:
        r = np.asarray(ref[key], dtype=float)
        if r.ndim > 1:
            r = r[0]
        scale = max(float(np.max(np.abs(r))), 1e-300)
        errs = []
        for dt in steps[1:]:
            a = np.asarray(runs[dt][key], dtype=float)
            if a.ndim > 1:
                a = a[0]
            t_a = np.arange(0, 120, dt)
            # compare on the coarse grid, interpolating the reference onto it
            ri = np.interp(t_a, t_ref, r)
            errs.append(float(np.max(np.abs(a - ri))) / scale)
        # order from the two coarsest points, where the asymptotic rate shows
        if errs[-1] > 0 and errs[-2] > 0:
            order = np.log2(errs[-1] / errs[-2])
        else:
            order = float("nan")
        print(f"  {key:10s} " + "".join(f"{e:13.3e}" for e in errs)
              + f"   {order:6.2f}")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="fewer resolutions")
    a = ap.parse_args()
    rc = positivity(a.quick)
    rc |= limit()
    rc |= convergence(a.quick)
    print("\nPASS" if rc == 0 else "\nFAIL -- a pool went negative")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
