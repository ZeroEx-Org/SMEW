# SPDX-License-Identifier: AGPL-3.0-only
"""Unit checks for harness.out_index / collect -- no simulation needed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
n, dt = 52560, 1 / (24 * 6)

print("out_index modes on a 1-year, dt=10 min run (52560 steps):")
for lbl, kw in [("n_out=500", dict(n_out=500)),
                ("dt_out=1.0 d", dict(dt_out=1.0, dt=dt)),
                ("dt_out=0.25 d", dict(dt_out=0.25, dt=dt)),
                ("stride=1000", dict(stride=1000))]:
    i = harness.out_index(n, **kw)
    print(f"   {lbl:14s} -> {len(i):6d} samples, first {i[0]}, last {i[-1]} "
          f"(= n-1: {i[-1] == n - 1})")

print("\nguard rails:")
for lbl, fn in [("no knob", lambda: harness.out_index(n)),
                ("two knobs", lambda: harness.out_index(n, n_out=10, stride=5)),
                ("dt_out without dt", lambda: harness.out_index(n, dt_out=1.0)),
                ("n_out > n_steps", lambda: harness.out_index(10, n_out=999))]:
    try:
        r = fn()
        print(f"   {lbl:20s} -> ok, {len(r)} samples (clamped)")
    except ValueError as e:
        print(f"   {lbl:20s} -> ValueError: {e}")

print("\nlength mismatch is caught, not silently broadcast:")
try:
    harness.collect({"pH": np.zeros(99)}, np.arange(100), n_out=10)
except ValueError as e:
    print("   ValueError:", e)

gold = os.path.join(HERE, "golden", "example.npz")
if os.path.exists(gold):
    g = harness.load(gold)
    print(f"\nfrozen master: {len(g)} arrays, n_steps_full={g['_n_steps_full'][0]}, "
          f"kept={len(g['t'])}, missing={list(g.get('_missing', []))}")
