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
    todo = a.only or NOTEBOOKS + [WRAPPER_CASE]
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
