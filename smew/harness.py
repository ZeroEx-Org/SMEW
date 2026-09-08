# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""Serialise a SMEW run to .npz at a chosen output resolution, and compare runs.

biogeochem_balance returns ~192 names via locals(); only ~32 of them are series
the notebooks actually consume. This module pins that subset (SCHEMA), thins it
to a requested number of timesteps, and writes it to a compressed .npz.

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

# series consumed by Examples/*.ipynb, Projects/*/*.ipynb and smew.complementary
SCHEMA_1D = [
    "pH", "Alk", "Ca", "Mg", "K", "Na", "Si", "Al", "Al_w", "H",
    "DIC", "CO2_w", "CO2_air", "M_rock",
    "Ca_tot", "Mg_tot", "Si_tot", "Al_tot", "CaCO3", "MgCO3",
    "UP_Ca", "UP_Mg", "UP_Si",
    "f_Ca", "f_Mg", "f_K", "f_Na", "f_Al", "f_H",
]
SCHEMA_2D = ["EW", "Wr", "Omega"]  # shape (n_mineral, n_steps)
SCHEMA = SCHEMA_1D + SCHEMA_2D

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
    for k in SCHEMA:
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
        a, b = np.asarray(ref[k], dtype=float), np.asarray(new[k], dtype=float)
        if a.shape != b.shape:
            out.append((k, float("nan"), False))
            continue
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        scale = max(float(np.max(np.abs(a))) if a.size else 0.0, 1e-300)
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

def example_run(seed=42, t_end=365, dt=1 / (24 * 6), diss_f=1.0, day1=1):
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
    temp_av, temp_ampl_yr, temp_ampl_d = 13, 11, 5      # [C]
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
