# SPDX-License-Identifier: AGPL-3.0-only
"""Demonstrate the n_out knob: size/time trade-off, and that thinning keeps
detection power because aggregates are computed on the full record."""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import harness            # noqa: E402
import case_example       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

t0 = time.perf_counter()
data, t, extra = case_example.run(seed=42)
t_sim = time.perf_counter() - t0
print(f"run: {t_sim:.2f} s, {len(t)} steps, "
      f"{len(harness.SCHEMA)} schema series + {len(extra)} extra\n")

print("=== size / time vs the n_out knob ===")
print(f"{'n_out':>8} {'kept':>7} {'dt_out':>9} {'raw kB':>9} {'gz kB':>9} "
      f"{'gz save s':>10} {'% of sim':>9}")
for n_out in [50, 200, 500, 2000, 10000, None]:
    kw = dict(stride=1) if n_out is None else dict(n_out=n_out)
    p = harness.collect(data, t, extra=extra, **kw)
    raw = harness.save(os.path.join(HERE, "_a.npz"), p, compress=False)
    t0 = time.perf_counter()
    gz = harness.save(os.path.join(HERE, "_b.npz"), p, compress=True)
    el = time.perf_counter() - t0
    kept = len(p["t"])
    dt_out = (t[-1] - t[0]) / max(kept - 1, 1)
    print(f"{str(n_out):>8} {kept:>7} {dt_out:>8.3f}d {raw/1e3:>9.1f} {gz/1e3:>9.1f} "
          f"{el:>10.3f} {el/t_sim*100:>8.1f}%")

print("\n=== golden master at n_out=500: freeze, re-run, check ===")
ref_payload = harness.collect(data, t, extra=extra, n_out=500)
gold = os.path.join(HERE, "golden", "example.npz")
sz = harness.save(gold, ref_payload)
print(f"   frozen: tests/golden/example.npz  {sz/1e3:.1f} kB")
d2, t2, e2 = case_example.run(seed=42)
print("  ", harness.report(harness.compare(
    harness.load(gold), harness.collect(d2, t2, extra=e2, n_out=500))))

print("\n=== does thinning lose detection power? diss_f perturbed by 0.001% ===")
d3, t3, e3 = case_example.run(seed=42, diss_f=1.00001)
for n_out in [50, 200, 500, 52560]:
    ref = harness.collect(data, t, extra=extra, n_out=n_out)
    new = harness.collect(d3, t3, extra=e3, n_out=n_out)
    res = harness.compare(ref, new)
    ser = [r for r in res if "__" not in r[0]]
    agg = [r for r in res if "__" in r[0]]
    print(f"   n_out={n_out:>6}: thinned series caught "
          f"{sum(1 for r in ser if not r[2]):>2}/{len(ser)}   "
          f"full-record aggregates caught {sum(1 for r in agg if not r[2]):>3}/{len(agg)}")
print(f"   pH moved by only "
      f"{np.max(np.abs(np.asarray(data['pH']) - np.asarray(d3['pH']))):.3g} units "
      "-- invisible on any figure, caught by the harness")

for f in ("_a.npz", "_b.npz"):
    os.remove(os.path.join(HERE, f))
