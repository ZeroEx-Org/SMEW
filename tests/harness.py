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

    import harness
    payload = harness.collect(data, t, n_out=500)     # or dt_out=1.0, or stride=100
    harness.save("golden/example.npz", payload)
    ref, new = harness.load("golden/example.npz"), harness.collect(data2, t, n_out=500)
    print(harness.report(harness.compare(ref, new)))
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
