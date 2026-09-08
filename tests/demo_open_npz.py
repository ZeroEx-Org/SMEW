# SPDX-License-Identifier: AGPL-3.0-only
"""Answers two questions: how to open a golden .npz in a notebook, and how the
harness payload differs from the dict run_SMEW returns."""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import harness         # noqa: E402
import case_example    # noqa: E402

GOLD = os.path.join(HERE, "golden", "example.npz")

print("=" * 70)
print("1. OPENING A GOLDEN .npz  (exactly what you would type in a notebook)")
print("=" * 70)
z = np.load(GOLD)
print(f"   z = np.load(path)        -> {type(z).__name__}, lazy zip archive")
print(f"   len(z.files)             -> {len(z.files)} arrays")
print(f"   z.files[:6]              -> {z.files[:6]}")
pH = z["pH"]
print(f"   z['pH']                  -> {type(pH).__name__} {pH.shape} {pH.dtype}")
print(f"   z['pH'][:4]              -> {np.round(pH[:4], 4)}")
print(f"   z['Omega'].shape         -> {z['Omega'].shape}   (n_mineral, n_kept)")
print(f"   z['pH__final'][0]        -> {z['pH__final'][0]:.6f}  (exact, full-record)")
print(f"   z['pH__min/max']         -> {z['pH__min'][0]:.4f} / {z['pH__max'][0]:.4f}"
      "   (over ALL 52560 steps, not just the kept ones)")
z.close()

print("\n   whole thing as a plain dict (what harness.load does):")
d = harness.load(GOLD)
print(f"   harness.load(path)       -> dict, {len(d)} keys, values are real ndarrays")
series = [k for k in d if not k.startswith("_") and "__" not in k and d[k].ndim == 1]
print(f"   1-D series for a DataFrame: {len(series)}")
try:
    import pandas as pd
    df = pd.DataFrame({k: d[k] for k in sorted(series)})
    print(f"   pd.DataFrame(...)        -> {df.shape[0]} rows x {df.shape[1]} cols")
    print(df[["t", "pH", "Alk", "M_rock"]].head(3).to_string(index=False))
except ImportError:
    print("   (pandas not installed in this venv)")

print("\n" + "=" * 70)
print("2. HARNESS PAYLOAD  vs  THE DICT run_SMEW RETURNS")
print("=" * 70)
data, t, extra = case_example.run(seed=42, t_end=120, day1=1)
print(f"   raw biogeochem_balance dict: {len(data)} names (locals())")

kinds = {}
for k, v in data.items():
    a = np.asarray(v, dtype=object) if isinstance(v, (list, str)) else np.asarray(v)
    key = ("time series" if getattr(a, "ndim", 0) >= 1 and getattr(a, "shape", (0,))[-1] == len(t)
           else f"{type(v).__name__}")
    kinds[key] = kinds.get(key, 0) + 1
print("   composition:", dict(sorted(kinds.items(), key=lambda x: -x[1])))

print("\n   (a) can you just np.savez the whole dict?")
try:
    np.savez(os.path.join(HERE, "_x.npz"), **data)
    sz = os.path.getsize(os.path.join(HERE, "_x.npz"))
    print(f"       it writes ({sz/1e6:.1f} MB) but see below")
    try:
        np.load(os.path.join(HERE, "_x.npz"), allow_pickle=False)["mineral"]
        print("       reload of 'mineral' with allow_pickle=False: ok")
    except ValueError as e:
        print(f"       reload FAILS with allow_pickle=False: {str(e)[:72]}...")
        print("       -> object arrays need pickle; pickled fixtures are not safe to")
        print("          load from a repo and are not reproducible across numpy versions")
    os.remove(os.path.join(HERE, "_x.npz"))
except Exception as e:
    print(f"       raises {type(e).__name__}: {str(e)[:60]}")

print("\n   (b) the freezing interpolation -- the difference that matters most")
WRAP = ['pH', 'Alk', 'M_rock', 'f_Ca', 'f_Mg', 'f_K', 'f_Na', 'f_H', 'f_Al',
        'HCO3', 'CO3', 'CO2_air', 'CO2_w']
# Vulkaneifel-like climate, January start -> a real frost period
data, t, extra = case_example.run(seed=42, t_end=365, day1=1,
                                  temp_av=8.45, temp_ampl_yr=16.35,
                                  latitude=50 * np.pi / 180, altitude=400)
raw = np.asarray(data['pH'], dtype=float).copy()
y = raw.copy()
x = np.arange(len(y))
mask = y != 0
interp = np.interp(x, x[mask], y[mask])
print(f"       run_SMEW overwrites {len(WRAP)} series with np.interp over frozen steps")
print(f"       for pH here: {int((~mask).sum())} of {len(y)} steps are zero "
      f"({100*(~mask).mean():.1f}%) and get replaced")
print(f"       max|raw - interpolated| = {np.max(np.abs(raw - interp)):.4f} pH units")
print("       -> a master frozen from run_SMEW output pins model AND interpolation")
print("          together; a regression in either looks identical. The harness")
print("          stores raw biogeochem output, so the two are separable.")
