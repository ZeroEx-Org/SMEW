# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""Serialise a SMEW run to .npz at a chosen output resolution, and compare runs.

biogeochem_balance returns ~192 names via locals(); 76 of them are read by the
notebooks, the wrapper or smew.ledger. This module pins that subset (CONTRACT),
thins the time series among them, and writes the result to a compressed .npz.

Why thin at all: a 1-year run at dt = 10 min is 52560 steps = 13.5 MB of float64
for the schema. Writing that costs ~0.1% of the simulation's own runtime
uncompressed, so thinning is NOT a speed optimisation -- it is about disk and git
size when several cases are re-frozen on every change.

Why thinning does not cost detection power: every series is ALSO reduced to
aggregates (min/max/mean/sum/final) computed on the FULL record before thinning.
A perturbation that only shows up between two kept samples still moves the
aggregates, so a 500-point golden master catches what a 52560-point one does.

    from smew import harness
    payload = harness.collect(data, t, n_out=500)     # or dt_out=1.0, or stride=100
    harness.save("tests/golden/example.npz", payload)
    ref = harness.load("tests/golden/example.npz")
    print(harness.report(harness.compare(ref, harness.collect(data2, t, n_out=500))))
"""
import os
import numpy as np

# ---------------------------------------------------------------------------
# The output contract
# ---------------------------------------------------------------------------
# What biogeochem_balance must return. It currently returns ~192 names via
# locals(); these 76 are the ones something actually reads. F1 replaces that
# locals() dump with an explicit dict built from exactly this list, so anything
# missing here is a name F1 would silently drop.
#
# There are three consumers, not one:
#   Examples/*.ipynb, Projects/*/*.ipynb, smew.complementary   the plots
#   wrappers_postprocessing/zeroex_input_data_wrapper.py       run_SMEW
#   smew.ledger                                                the mass balance
#
# The ledger is by far the most demanding, and it was written after the first
# version of this schema. It rebuilds every source and sink from the outside, so
# it reads the background-input scalars, the per-mineral stoichiometry and the
# raw fluxes -- 54 names, most of which no plot ever touches. Established
# empirically, by recording every __getitem__ the ledger performs across all 21
# of its cases; the set does not vary by case.
#
# Entries are grouped by what they are, not alphabetically, because the grouping
# is the part that has to be maintained when a process is added.

SCHEMA_1D = [                                        # shape (n_steps,)
    # acid-base, the carbonate system, and the CO2 fluxes in and out of it
    "pH", "H", "f_H", "Alk", "Alk_tot", "An", "An_tot",
    "CO2_w", "CO2_air", "HCO3", "CO3", "DIC", "IC_tot", "Fs", "ADV", "DIC_rain",
    # cations: aqueous concentration, total pool, exchanger fraction, uptake
    "Ca", "Ca_tot", "f_Ca", "UP_Ca",
    "Mg", "Mg_tot", "f_Mg", "UP_Mg",
    "K", "K_tot", "f_K",
    "Na", "Na_tot", "f_Na",
    "Si", "Si_tot", "UP_Si",
    # aluminium: free ion, the four hydroxide species, dissolved and total
    "Al", "AlOH", "AlOH2", "AlOH3", "AlOH4", "Al_w", "Al_tot", "f_Al",
    # solid phases and their dissolution fluxes
    "M_rock", "CaCO3", "MgCO3", "W_CaCO3", "W_MgCO3",
    # forcing carried in from the moisture, respiration and vegetation stages
    "s", "temp_soil", "v", "I", "L", "T", "Dw", "r_het", "r_aut",
    # F2: what the positivity limiter had to do. All zero on a healthy run, so
    # these are also the cheapest assertion that a run stayed well-posed.
    "clip_Ca", "clip_Mg", "clip_K", "clip_Na", "clip_Al", "clip_Si",
    "clip_An", "clip_C", "clip_CaCO3", "clip_MgCO3",
    "dt_max", "n_substeps",
]
SCHEMA_2D = ["EW", "Wr", "Omega", "M_min", "clip_M_min"]   # (n_mineral, n_steps)

# Shape is set by the run, not by the time vector, so these are stored whole:
# thinning and aggregating them would be meaningless.
SCHEMA_FIXED = ["min_st", "xi"]                      # (n_mineral, 6) and (4,)
SCHEMA_SCALAR = [
    # background solute input, replacing what leaching and transpiration remove
    "I_Ca", "I_Mg", "I_K", "I_Na", "I_Si", "I_An",
    # geometry, units, and the vegetation constants the uptake recompute needs
    "n", "Zr", "dt", "conv_mol", "conv_Al", "k_v", "RAI", "root_d",
]
SCHEMA_TEXT = ["mineral"]                            # list of mineral names

SCHEMA_SERIES = SCHEMA_1D + SCHEMA_2D    # thinned and aggregated by collect()
SCHEMA = SCHEMA_SERIES                   # kept: the pre-F0.1 name for this set
CONTRACT = SCHEMA_SERIES + SCHEMA_FIXED + SCHEMA_SCALAR + SCHEMA_TEXT


def verify_contract(data):
    """Names in CONTRACT that this result dict does not carry.

    The gate is that this comes back empty for every case, so that the contract
    describes the output rather than wishing for it. It is also the F1 gate:
    F1 swaps locals() for an explicit dict, and this is what says the swap left
    nothing behind.
    """
    return [k for k in CONTRACT if k not in data]

AGGS = {"min": np.min, "max": np.max, "mean": np.mean, "sum": np.sum}


def provenance():
    """Which code state produced this payload. Recorded so a failing check can
    say what it is diffing against, instead of just that something moved.

    git is optional: if it is unavailable the fields come back as "unknown".
    """
    import datetime
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*a):
        try:
            return subprocess.run(("git",) + a, cwd=here, capture_output=True,
                                  text=True, timeout=10).stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    sha = git("rev-parse", "HEAD")
    return {
        "_git_sha": np.array([sha], dtype="U40"),
        # dirty means the working tree differed from that commit, so the payload
        # is NOT reproducible from the sha alone
        # ":/smew" is a repo-root-relative pathspec: cwd here is the package
        # directory, so a plain "smew" would resolve to smew/smew and match nothing
        "_git_dirty": np.array([bool(git("status", "--porcelain", "--", ":/smew")
                                     not in ("", "unknown"))]),
        "_git_branch": np.array([git("rev-parse", "--abbrev-ref", "HEAD")], dtype="U64"),
        "_created": np.array([datetime.datetime.now().isoformat(timespec="seconds")],
                             dtype="U32"),
        "_numpy": np.array([np.__version__], dtype="U16"),
    }


def describe(payload):
    """One-line summary of where a payload came from."""
    def g(k, d="?"):
        v = payload.get(k)
        return d if v is None else (v[0].item() if hasattr(v[0], "item") else v[0])
    sha = str(g("_git_sha"))[:8]
    dirty = " +dirty" if g("_git_dirty", False) else ""
    kept = len(payload["t"]) if "t" in payload else "?"
    full = g("_n_steps_full", "?")
    return (f"{sha}{dirty} on {g('_git_branch')}, {kept} of {full} steps, "
            f"created {g('_created')}, numpy {g('_numpy')}")


def out_index(n_steps, n_out=None, stride=None, dt_out=None, dt=None):
    """Timestep indices to keep. Pass exactly one of n_out / stride / dt_out.

    n_out  : target number of samples, evenly spread -- this is the size knob
    stride : keep every stride-th step
    dt_out : output interval in days; requires dt (the simulation timestep)

    Index 0 and the final index are always kept, so initial and final state are
    exact rather than interpolated or dropped.
    """
    if sum(x is not None for x in (n_out, stride, dt_out)) != 1:
        raise ValueError("give exactly one of n_out, stride, dt_out")

    if dt_out is not None:
        if dt is None:
            raise ValueError("dt_out requires dt (the simulation timestep, in days)")
        stride = max(1, int(round(dt_out / dt)))

    if stride is not None:
        idx = np.arange(0, n_steps, int(stride), dtype=np.int64)
    else:
        n = int(np.clip(n_out, 2, n_steps))
        idx = np.unique(np.linspace(0, n_steps - 1, n).round().astype(np.int64))

    if idx[-1] != n_steps - 1:
        idx = np.append(idx, n_steps - 1)  # never lose the final state
    return idx


def collect(data, t, n_out=None, stride=None, dt_out=None, dt=None, extra=None):
    """Pull SCHEMA out of a biogeochem_balance result and thin it.

    data  : dict returned by smew.biogeochem_balance
    t     : the run's time vector
    extra : optional {name: array} for series produced outside biogeochem_balance
            (rain, s, ET0, SOC, ...) -- thinned and aggregated identically

    Returns a flat dict ready for np.savez_compressed:
        t, <series>...                                thinned
        <series>__min/max/mean/sum/final              from the FULL record
        _idx, _n_steps_full, _schema, _missing        provenance
    """
    t = np.asarray(t, dtype=float)
    n_steps = len(t)
    idx = out_index(n_steps, n_out, stride, dt_out, dt)

    series, missing = {}, []
    for k in SCHEMA_SERIES:
        if k in data:
            series[k] = np.asarray(data[k], dtype=float)
        else:
            missing.append(k)
    for k, a in (extra or {}).items():
        series[k] = np.asarray(a, dtype=float)

    out = {"t": t[idx]}
    for k, a in series.items():
        if a.ndim == 1:
            if len(a) != n_steps:
                raise ValueError(f"{k}: length {len(a)} != len(t) {n_steps}")
            out[k] = a[idx]
        elif a.ndim == 2:
            if a.shape[1] != n_steps:
                raise ValueError(f"{k}: shape {a.shape} does not end in len(t) {n_steps}")
            out[k] = a[:, idx]
        else:
            raise ValueError(f"{k}: unsupported ndim {a.ndim}")

        # Aggregates over the FULL record -- this is what makes thinning safe.
        # Any mass-balance ledger term belongs here too, for the same reason: it
        # cannot be reconstructed from decimated samples.
        axis = None if a.ndim == 1 else 1
        for name, fn in AGGS.items():
            out[f"{k}__{name}"] = np.atleast_1d(np.asarray(fn(a, axis=axis), dtype=float))
        out[f"{k}__final"] = np.atleast_1d(np.asarray(a[..., -1], dtype=float))

    # Everything whose shape is not (..., n_steps): stored whole, since there is
    # nothing to thin and no aggregate that would mean anything. Cheap, and worth
    # pinning -- a changed conv_mol, a changed min_st lookup or a changed
    # background input I_Ca moves every number downstream, and these are the only
    # keys that would say so directly rather than as a diff in fifty series.
    for k in SCHEMA_FIXED + SCHEMA_SCALAR:
        if k in data:
            out[k] = np.atleast_1d(np.asarray(data[k], dtype=float))
        else:
            missing.append(k)
    for k in SCHEMA_TEXT:
        if k in data:
            out[k] = np.asarray(data[k], dtype="U32").reshape(-1)
        else:
            missing.append(k)

    out.update(provenance())
    out["_idx"] = idx
    out["_n_steps_full"] = np.array([n_steps])
    out["_schema"] = np.array(sorted(series), dtype="U32")
    if missing:
        out["_missing"] = np.array(sorted(missing), dtype="U32")
    return out


def save(path, payload, compress=True):
    """Write a collect() payload. compress=False is ~40x faster and ~1.4x bigger."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    (np.savez_compressed if compress else np.savez)(path, **payload)
    return os.path.getsize(path if path.endswith(".npz") else path + ".npz")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def compare(ref, new, rtol=1e-12, atol=0.0):
    """Compare two payloads key by key -> [(key, max_abs_diff, ok), ...].

    Keys present in only one payload, or with mismatched shapes, report nan/False.
    """
    out = []
    for k in sorted(set(ref) | set(new)):
        if k.startswith("_"):
            continue
        if k not in ref or k not in new:
            out.append((k, float("nan"), False))
            continue
        a, b = np.asarray(ref[k]), np.asarray(new[k])
        if a.shape != b.shape:
            out.append((k, float("nan"), False))
            continue
        if a.dtype.kind in "USO" or b.dtype.kind in "USO":
            # text (mineral names): equal or not, there is no tolerance to apply
            same = bool(np.array_equal(a, b))
            out.append((k, 0.0 if same else float("nan"), same))
            continue
        a, b = a.astype(float), b.astype(float)
        if not a.size:
            out.append((k, 0.0, True))
            continue

        # Non-finite entries need handling before any subtraction: inf - inf is
        # nan, so a series that legitimately carries inf (dt_max is inf wherever
        # no pool lost mass in that step) compared against itself would report
        # nan and fail. Two entries match if they are both nan, or equal
        # infinities of the same sign; anything else is a real difference.
        fa, fb = np.isfinite(a), np.isfinite(b)
        both_finite = fa & fb
        agree_nonfinite = ((np.isnan(a) & np.isnan(b))
                           | (~fa & ~fb & ~np.isnan(a) & ~np.isnan(b) & (a == b)))
        if np.any(~both_finite & ~agree_nonfinite):
            out.append((k, float("inf"), False))
            continue

        av, bv = a[both_finite], b[both_finite]
        d = float(np.max(np.abs(av - bv))) if av.size else 0.0
        scale = max(float(np.max(np.abs(av))) if av.size else 0.0, 1e-300)
        out.append((k, d, bool(d <= max(atol, rtol * scale))))
    return out


def report(results, show_pass=False):
    bad = [r for r in results if not r[2]]
    lines = [f"   {'PASS' if ok else 'FAIL'}  {k:26s} max|diff| {d:12.6g}"
             for k, d, ok in results if show_pass or not ok]
    head = f"{len(results) - len(bad)}/{len(results)} keys match"
    return "\n".join([head] + lines) if lines else head


# ---------------------------------------------------------------------------
# Example usage
#
#     python -m smew.harness freeze              # write the golden master
#     python -m smew.harness check               # re-run and compare
#     python -m smew.harness check --diss-f 1.00001   # prove it detects change
#     python -m smew.harness sweep               # size/time vs n_out
#
# example_run() below is the pattern to copy for a new case: declare the inputs,
# wire the pipeline stages in order, return (data, t, extra).
# ---------------------------------------------------------------------------

def example_run(seed=42, t_end=365, dt=1 / (24 * 6), diss_f=1.0, day1=1,
                with_water=False, temp_av=13):
    """One full SMEW pipeline with every input stated explicitly.

    Deterministic given seed: rain_stoc is the only stochastic stage.
    Returns (data, t, extra) -- exactly what collect() consumes.
    """
    import numpy as np
    import smew

    t = np.arange(0, t_end, dt)

    # --- units (must stay identical across every call below) ---
    conv_mol, conv_Al = 1e6, 1e3

    # --- site / soil ---
    soil, Zr, rho_bulk = "loam", 0.3, 1.2e6      # [-], [m], [g/m3]
    latitude, altitude = 40 * np.pi / 180, 33     # [rad], [m]
    s_in = 0.5                                    # initial relative moisture [-]

    # --- climate ---
    # temp_av is a parameter so a caller can drive the run below freezing:
    # biogeochem.py holds the chemistry through frost instead of integrating it
    # (biogeochem.py:370-379), and no comparison notebook ever goes below zero,
    # so that branch is otherwise never exercised.
    temp_ampl_yr, temp_ampl_d = 11, 5                   # [C]
    albedo, coastal = 0.25, False
    wind = 1 * np.ones(len(t))                          # [m/s]
    R_tot, lamda = 1.2, 0.25                            # [m/yr], [1/d]
    alfa = (R_tot / lamda) / 365                        # mean storm depth [m]

    # --- vegetation ---
    k_v, T_v, RAI, root_d = 3000, 100, 10, 0.4e-3       # [g/m2], [d], [m2/m2], [m]
    v_in, t0_v = 1 * k_v, 0

    # --- organic carbon ---
    ADD, SOC_perc, ratio_aut_het = 1, 0.05, 1           # [gOC/(m2 d)], [%], [-]

    # --- initial chemistry ---
    pH_in = 4
    CEC_tot = 10 * 1e-5 * rho_bulk * Zr * conv_mol      # from mmol_c/100g -> mol_c
    f_CEC_in = np.array([0.30, 0.15, 0.10, 0.05, 0.00, 0.40])  # Ca Mg K Na Al H
    assert abs(f_CEC_in.sum() - 1) <= 1e-3, "f_CEC_in must sum to 1"
    Si_in = CaCO3_in = MgCO3_in = 0

    # --- ERW application (parallel arrays, one entry per mineral) ---
    mineral = ["forsterite"]
    M_rock_in, t_app = 1000, 0                          # [g/m2], [d]
    rock_f_in = np.array([1])
    d_in = np.array([100]) * 1e-6                       # particle diameter [m]
    psd_perc_in = np.array([1])
    SSA_in = np.nan

    keyword_wb, keyword_add = 1, 1   # dynamic moisture, replace background losses

    # --- 1. hydroclimatic forcing ---
    temp_air, temp_soil, temp_min, temp_max = smew.temp(
        latitude, temp_av, temp_ampl_yr, temp_ampl_d, Zr, t_end, dt, day1)
    ET0 = smew.ET0(latitude, altitude, temp_air, temp_soil, temp_min, temp_max,
                   wind, albedo, Zr, coastal, t_end, dt, day1)
    rain = smew.rain_stoc(lamda, alfa, t_end, dt, seed=seed)

    # --- 2. vegetation ---
    v = smew.veg(v_in, T_v, k_v, t0_v, temp_soil, dt)

    # --- 3. moisture balance ---
    s, s_w, s_i, I, L, T, E, Q, Irr, n = smew.moisture_balance(
        rain, Zr, soil, ET0, v, k_v, keyword_wb, s_in, t_end, dt)

    # --- 4. organic carbon / respiration ---
    SOC, r_het, r_aut, D = smew.respiration(
        ADD, rho_bulk * SOC_perc / 100, 10 * smew.CO2_atm(conv_mol),
        ratio_aut_het, soil, s, v, k_v, Zr, temp_soil, dt, conv_mol)

    # --- 5. initial conditions ---
    conc_in, K_CEC = smew.f_CEC_to_conc(f_CEC_in, pH_in, soil, conv_mol, conv_Al)

    # --- 6. biogeochemistry ---
    data = smew.biogeochem_balance(
        n, s, L, T, I, v, k_v, RAI, root_d, Zr, r_het, r_aut, D, temp_soil,
        pH_in, conc_in, f_CEC_in, K_CEC, CEC_tot, Si_in, CaCO3_in, MgCO3_in,
        M_rock_in, t_app, mineral, rock_f_in, d_in, psd_perc_in, SSA_in,
        diss_f, dt, conv_Al, conv_mol, keyword_add)

    # series the notebooks plot that biogeochem_balance does not return
    extra = {"rain": rain, "s": s, "ET0": ET0, "SOC": SOC,
             "temp_soil": temp_soil, "L": L, "Q": Q, "v": v}
    # bare evaporation is the one series biogeochem_balance never sees (it is not
    # one of its arguments), and the water balance in smew.ledger needs it.
    # Off by default so freeze/check payloads keep exactly the keys they had --
    # compare() treats a key present in only one payload as a failure, so adding
    # it unconditionally would invalidate every frozen file.
    if with_water:
        extra["E"] = E
    return data, t, extra


def main(argv=None):
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["freeze", "check", "sweep"])
    # this module lives in smew/, so the repo root is one level up. Goldens are
    # kept outside the package (tests/golden/) so they are not shipped on install.
    # "harness_example", not "example": tests/golden/Example.npz is the fixture for
    # Examples/Example.ipynb, and on a case-insensitive filesystem (macOS default)
    # "example.npz" and "Example.npz" are the same file -- they silently overwrite
    # each other. This one is harness.example_run(), which is seeded and therefore
    # checkable; the notebook is not (it calls rain_stoc without a seed).
    ap.add_argument("--golden", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tests", "golden", "harness_example.npz"))
    ap.add_argument("--n-out", type=int, default=500,
                    help="timesteps to keep in the .npz (size knob)")
    ap.add_argument("--dt-out", type=float, default=None,
                    help="output interval in days, instead of --n-out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-end", type=float, default=365)
    ap.add_argument("--diss-f", type=float, default=1.0,
                    help="perturb dissolution to check the master detects it")
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--no-compress", action="store_true")
    a = ap.parse_args(argv)

    dt = 1 / (24 * 6)
    thin = ({"dt_out": a.dt_out, "dt": dt} if a.dt_out else {"n_out": a.n_out})

    if a.mode == "sweep":
        data, t, extra = example_run(seed=a.seed, t_end=a.t_end, dt=dt)
        print(f"{'n_out':>8} {'kept':>7} {'kB':>9} {'save s':>8}")
        for n in [50, 200, 500, 2000, 10000, None]:
            kw = {"stride": 1} if n is None else {"n_out": n}
            p = collect(data, t, extra=extra, **kw)
            t0 = time.perf_counter()
            sz = save(a.golden + ".sweep", p)
            el = time.perf_counter() - t0
            print(f"{str(n):>8} {len(p['t']):>7} {sz/1e3:>9.1f} {el:>8.3f}")
        os.remove(a.golden + ".sweep.npz")
        return 0

    t0 = time.perf_counter()
    data, t, extra = example_run(seed=a.seed, t_end=a.t_end, dt=dt, diss_f=a.diss_f)
    print(f"run: {time.perf_counter()-t0:.2f} s, {len(t)} steps")
    payload = collect(data, t, extra=extra, **thin)

    if a.mode == "freeze":
        sz = save(a.golden, payload, compress=not a.no_compress)
        print(f"froze {len(payload['t'])} of {len(t)} steps -> {a.golden} "
              f"({sz/1e3:.1f} kB)")
        print(f"  {describe(payload)}")
        return 0

    if not os.path.exists(a.golden):
        print(f"no golden master at {a.golden}; run 'freeze' first")
        return 2
    ref = load(a.golden)
    print(f"baseline: {describe(ref)}")
    print(f"current:  {describe(payload)}")
    res = compare(ref, payload, rtol=a.rtol)
    print(report(res))
    return 0 if all(ok for _, _, ok in res) else 1


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
