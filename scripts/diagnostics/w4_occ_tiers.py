#!/usr/bin/env python
"""Where does the board's occupancy_dice loss actually come from?  (doc sec.14)

Board facts used here (hyperbolic skill, `common.core_metrics.skill`):
  heart interp  floor/ceiling  ODS (0.8100, 0.9528)   d2 (0.04610, 0.00213)
  live  file 11  ODS 43.5  d2 69.7      -> dice 0.7673  d2 0.02124
  file 05 spantrim ODS 47.2 d2 63.8     -> dice 0.7931  d2 0.02709   (nocc 289 -> 246)
`occupancy_dice` runs `_match_n`, and the heart truth has MORE cells than we submit, so our
side is used in full (row-order invariant, measured sd 0.00000) while the TRUTH is subsampled
to our n. Both board files therefore saw the same k = |occ(truth)|, which makes the pair
(dice, nocc) a two-equation system in (I, k) -- solved below.

Then the mechanism question: our 289 voxels vs the truth's ~244. Which of the excess voxels
can be identified WITHOUT the truth? A voxel that neither bracketing real stage occupies
cannot be occupied by an interpolation target ("tier 1"). This measures, on three windows
with a REAL truth, how often that rule is wrong, and how much of our excess it explains.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
import numpy as np, anndata as ad
from common.shape_metrics import _canonicalise, occupancy_dice, d2_distance

FLIPS = [np.array(f, float) for f in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]
N_GRID, EXTENT = 16, 3.0
FLOOR, CEIL = 0.81, 0.9528
D2F, D2C = 0.0461, 0.00213
sk_ods = lambda d: min(100.0, 100.0 * (FLOOR - 0) / ((FLOOR) + (CEIL - d)) * (CEIL - 0) / (CEIL - 0)) if False else min(100.0, 100.0 * (CEIL - FLOOR) / ((CEIL - FLOOR) + (CEIL - d)))
sk_d2 = lambda r: 100.0 * (D2F - D2C) / ((D2F - D2C) + (r - D2C))
inv_ods = lambda s: CEIL - (CEIL - FLOOR) * (100.0 / s - 1.0)
inv_d2 = lambda s: D2C + (D2F - D2C) * (100.0 / s - 1.0)


def coords(p):
    return np.asarray(ad.read_h5ad(p).obsm["spatial_3D"], float)


def sub(C, n=5000, seed=0):
    C = np.asarray(C, float)
    return C if len(C) <= n else C[np.random.default_rng(seed).choice(len(C), n, replace=False)]


def canon(C):
    Z, rms = _canonicalise(np.asarray(C, float))
    return Z, float(rms)


def vox(Z):
    v = np.floor((np.clip(Z, -EXTENT, EXTENT - 1e-9) + EXTENT) / (2 * EXTENT) * N_GRID).astype(int)
    return set(map(tuple, v))


def best_flip(Zp, Zt):
    """Flip of the PRED cloud that maximises dice against Zt (what the metric does)."""
    vt = vox(Zt); best, bf = -1.0, None
    for f in FLIPS:
        d = 2 * len(vox(Zp * f) & vt) / (len(vox(Zp * f)) + len(vt))
        if d > best: best, bf = d, f
    return best, bf


def tiers(Zp, anchors):
    """Voxel tiers of Zp against a list of anchor clouds, each in its own best-flip frame."""
    vp = vox(Zp); sets = []
    for Za in anchors:
        _, f = best_flip(Zp, Za)
        sets.append(vox(Za * f))      # anchor expressed in the pred's frame
    t1 = {v for v in vp if not any(v in s for s in sets)}
    t3 = {v for v in vp if all(v in s for s in sets)}
    t2 = vp - t1 - t3
    return t1, t2, t3, sets


def cellcount(Z, vs):
    v = np.floor((np.clip(Z, -EXTENT, EXTENT - 1e-9) + EXTENT) / (2 * EXTENT) * N_GRID).astype(int)
    return int(sum(1 for t in map(tuple, v) if t in vs))


print("=" * 100)
print("A. board arithmetic: solve for k = |occ(truth at our n)| from two submissions")
rows = [("live 11 (nocc 289)", 43.5, 289), ("live 08 (nocc 289)", 43.3, 289),
        ("05 spantrim (nocc 246)", 47.2, 246), ("09 cigeo (nocc 289)", 45.8, 289)]
for tag, s, m in rows:
    print(f"  {tag:26s} ODS {s:5.1f} -> dice {inv_ods(s):.4f}   "
          f"d2 skill -> {inv_d2({'live 11 (nocc 289)': 69.7, 'live 08 (nocc 289)': 70.4, '05 spantrim (nocc 246)': 63.8, '09 cigeo (nocc 289)': 68.9}[tag]):.5f}")
# two-equation solve: I = dice*(m+k)/2, and spantrim only REMOVED voxels (I2 <= I1)
d1, m1 = inv_ods(43.5), 289
d2_, m2 = inv_ods(47.2), 246
print("\n  feasible k range from I<=k and I<=m:")
lo = max(d1 * m1 / (2 - d1), d2_ * m2 / (2 - d2_))
hi = min(m1 * (2 - d1) / d1, m2 * (2 - d2_) / d2_)
print(f"    k in [{lo:.0f}, {hi:.0f}]")
for k in (230, 238, 244, 251, 260, 280, 300):
    I1, I2 = d1 * (m1 + k) / 2, d2_ * (m2 + k) / 2
    print(f"    k={k:4d}: I_live {I1:6.1f} (spurious {m1-I1:5.1f}, truth-missed {k-I1:5.1f}) | "
          f"I_spantrim {I2:6.1f} | true voxels LOST by spantrim {I1-I2:5.1f} of {m1-m2} removed "
          f"-> precision {1-(I1-I2)/(m1-m2):.2f}")
print(f"\n  anchor nocc at n=5000 (the truth is subsampled to our n, so k is on THIS scale):")
ANCH = {"E8.25_late": "data/E8.25_late.h5ad", "E8.75": "data/E8.75.h5ad", "E9.5": "data/E9.5.h5ad"}
Z5 = {}
for nm, p in ANCH.items():
    Z5[nm], _ = canon(sub(coords(p)))
    print(f"    {nm:12s} nocc {len(vox(Z5[nm])):4d}")

print()
print("=" * 100)
print("B. is the tier-1 rule ('occupied by NEITHER bracketing real stage') safe?  REAL-truth windows")
WINDOWS = [
    ("heart board  E8.25+E8.75 -> E8.5 (NO truth)", "data/E8.25_late.h5ad", "data/E8.75.h5ad", None,
     "outputs/t2/submit/T2_heart_val_interp.h5ad"),
    ("heart W2     E8.25+E9.5  -> E8.75 (truth)", "data/E8.25_late.h5ad", "data/E9.5.h5ad", "data/E8.75.h5ad",
     "outputs/t2/heart/w2_pipeline.h5ad"),
    ("heart narrow E8.0? n/a", None, None, None, None),
    ("embryo board E7.25+E8.0  -> E7.5?", "data/E7.25.h5ad", "data/E8.0.h5ad", None,
     "outputs/t2/submit/T2_embryo_val_interp.h5ad"),
    ("embryo W     E6.75+E8.0  -> E7.25 (truth)", "data/E6.75.h5ad", "data/E8.0.h5ad", "data/E7.25.h5ad", None),
]
for tag, pa, pb, pt, plive in WINDOWS:
    if pa is None:
        continue
    Za, _ = canon(sub(coords(pa))); Zb, _ = canon(sub(coords(pb)))
    line = f"  {tag}\n"
    if pt is not None:
        Zt, _ = canon(sub(coords(pt)))
        T = vox(Zt)
        _, fa = best_flip(Zt, Za); _, fb = best_flip(Zt, Zb)
        UA = vox(Za * fa) | vox(Zb * fb)
        outside = T - UA
        line += (f"    truth nocc {len(T):4d}; truth voxels outside the anchor union {len(outside):4d}"
                 f" = {100*len(outside)/len(T):5.1f}%  <- tier-1 FALSE-POSITIVE rate\n")
        # how well does the rule find a synthetic cloud's spurious voxels?
    if plive is not None:
        Zl, _ = canon(coords(plive))
        t1, t2, t3, sets = tiers(Zl, [Za, Zb])
        UA = sets[0] | sets[1]
        line += (f"    live  nocc {len(vox(Zl)):4d} = tier3(both) {len(t3):4d} + tier2(one) {len(t2):4d} "
                 f"+ tier1(neither) {len(t1):4d}   | anchor union {len(UA):4d}\n")
        line += (f"    cells living in tier1 voxels: {cellcount(Zl, t1):5d} of {len(Zl)} "
                 f"({100*cellcount(Zl,t1)/len(Zl):4.1f}%);  tier2 {cellcount(Zl,t2):5d}, tier3 {cellcount(Zl,t3):5d}\n")
        if pt is not None:
            Zt, _ = canon(sub(coords(pt)))
            T = vox(Zt)
            _, f = best_flip(Zl, Zt)
            vp = vox(Zl * f)
            spur = vp - T
            line += (f"    REAL check: live spurious voxels {len(spur):4d}; of those, tier1 catches "
                     f"{len(spur & t1):4d} ({100*len(spur&t1)/max(len(spur),1):.0f}%), tier2 {len(spur&t2):4d}, "
                     f"tier3 {len(spur&t3):4d}\n")
            line += (f"    REAL cost : tier1 voxels that are actually IN the truth: {len(t1 & T):4d} of {len(t1):4d}"
                     f" -> tier-1 removal precision {1-len(t1&T)/max(len(t1),1):.2f}")
    print(line)
