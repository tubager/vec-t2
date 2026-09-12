#!/usr/bin/env python
"""Occupancy pruning with FULL anchor clouds: how many of our spurious voxels are identifiable?

w4_occ_tiers.py showed why the tier rule looked weak: occupancy at n=5000 is a SPARSE SAMPLE of
the tissue, so a voxel the truth occupies can easily be missed by a 5000-cell anchor (8-10% of
real truth voxels sit outside the 5000-cell anchor union). The anchors are released in full
(E8.25_late 58716, E8.75 24826, E9.5, E7.25 13295, E8.0, E6.75 cells), so "occupied by neither
FULL anchor" is a far stronger statement than "occupied by neither 5000-cell subsample".

Board arithmetic (w4 part A, k = |occ(truth subsampled to our n=5000)| ~ 244):
  live heart  nocc 289, I ~ 204.5 -> ~85 spurious voxels, ~40 truth voxels we miss
  file 05 removed 43 voxels at 0.76 precision and bought ODS +3.9 but paid d2 -6.6 (clumping).
Here we measure only the IDENTIFICATION question, on windows with a real truth:
  precision(R) = 1 - |removed voxels that are really in the truth| / R
  recall(R)    = |removed spurious| / |all spurious|
and the implied board ODS skill, for a legitimacy ranking of our voxels.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
import numpy as np, anndata as ad
from common.shape_metrics import _canonicalise, occupancy_dice

FLIPS = [np.array(f, float) for f in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]
N_GRID, EXTENT = 16, 3.0
FLOOR, CEIL = 0.81, 0.9528
sk_ods = lambda d: min(100.0, 100.0 * (CEIL - FLOOR) / ((CEIL - FLOOR) + (CEIL - d)))
inv_ods = lambda s: CEIL - (CEIL - FLOOR) * (100.0 / s - 1.0)


def coords(p):
    return np.asarray(ad.read_h5ad(p).obsm["spatial_3D"], float)


def sub(C, n=5000, seed=0):
    C = np.asarray(C, float)
    return C if len(C) <= n else C[np.random.default_rng(seed).choice(len(C), n, replace=False)]


def canon(C):
    Z, rms = _canonicalise(np.asarray(C, float)); return Z


def vidx(Z):
    return np.floor((np.clip(Z, -EXTENT, EXTENT - 1e-9) + EXTENT) / (2 * EXTENT) * N_GRID).astype(int)


def vox(Z):
    return set(map(tuple, vidx(Z)))


def flip_to(Zp, Zt):
    """Best proper sign flip of Zp against Zt, with the runner-up margin (frame ambiguity check)."""
    vt = vox(Zt); sc = []
    for f in FLIPS:
        vp = vox(Zp * f); sc.append(2 * len(vp & vt) / (len(vp) + len(vt)))
    o = np.argsort(sc)[::-1]
    return FLIPS[o[0]], sc[o[0]], sc[o[1]]


def legitimacy(Zp, anchors_full):
    """Rank our voxels: how many FULL anchor clouds occupy it (in the common frame), then cell count."""
    vp = vidx(Zp)
    cnt = {}
    for t in map(tuple, vp):
        cnt[t] = cnt.get(t, 0) + 1
    sup = {}
    for Za in anchors_full:
        s = vox(Za)
        for t in cnt:
            sup[t] = sup.get(t, 0) + (1 if t in s else 0)
    order = sorted(cnt, key=lambda t: (sup[t], cnt[t]))     # least legitimate first
    return order, sup, cnt


def evaluate(tag, Zp, anchors_full, Zt=None, k_truth=None, m_live=None, I_live=None):
    order, sup, cnt = legitimacy(Zp, anchors_full)
    print(f"  {tag}")
    hist = {}
    for t in sup:
        hist[sup[t]] = hist.get(sup[t], 0) + 1
    print(f"    our {len(order)} voxels by FULL-anchor support: " +
          "  ".join(f"sup={s}:{hist.get(s,0)}" for s in sorted(hist)))
    n_by_sup = {s: sum(cnt[t] for t in sup if sup[t] == s) for s in sorted(hist)}
    print("    cells in those voxels:                    " +
          "  ".join(f"sup={s}:{n_by_sup[s]}" for s in sorted(n_by_sup)))
    if Zt is None:
        return
    T = vox(Zt)
    spur = vox(Zp) - T
    print(f"    REAL truth: nocc {len(T)}, our spurious {len(spur)}, our shared {len(vox(Zp)&T)}")
    print(f"    {'R':>4s} {'removed':>8s} {'真spurious':>10s} {'误删real':>8s} {'precision':>9s} "
          f"{'recall':>7s} {'dice':>7s} {'ODS':>6s} {'ΔODS':>6s}")
    base_d = 2 * len(vox(Zp) & T) / (len(vox(Zp)) + len(T))
    base_s = sk_ods(base_d)
    for R in (10, 20, 30, 40, 50, 60, 70, 80, 100, 120):
        rem = set(order[:R]); hit = len(rem & spur); miss = len(rem & T)
        m = len(vox(Zp)) - R; I = len(vox(Zp) & T) - miss
        d = 2 * I / (m + len(T)); s = sk_ods(d)
        print(f"    {R:4d} {R:8d} {hit:10d} {miss:8d} {hit/R:9.2f} {hit/len(spur):7.2f} "
              f"{d:7.4f} {s:6.1f} {s-base_s:+6.1f}")


print("=" * 100)
print("HEART board window (anchors FULL E8.25_late + E8.75, truth unknown -> identification only)")
LIVE = canon(coords("outputs/t2/submit/T2_heart_val_interp.h5ad"))
A_full = canon(coords("data/E8.25_late.h5ad")); B_full = canon(coords("data/E8.75.h5ad"))
f, s1, s2 = flip_to(LIVE, A_full)
print(f"  frame check: dice(LIVE,E8.25_full) best {s1:.4f} runner-up {s2:.4f}; flip {f}")
f2, s1b, s2b = flip_to(LIVE, B_full)
print(f"               dice(LIVE,E8.75_full) best {s1b:.4f} runner-up {s2b:.4f}; flip {f2}")
evaluate("live heart (n=5000, nocc 289)", LIVE * f, [A_full, B_full * np.array([1, 1, 1])])

print()
print("=" * 100)
print("HEART W2 window: anchors FULL E8.25_late + E9.5, REAL truth E8.75 @5000")
W2 = canon(np.asarray(ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad").obsm["spatial_3D"], float))
A2 = canon(coords("data/E8.25_late.h5ad")); B2 = canon(coords("data/E9.5.h5ad"))
T2 = canon(sub(coords("data/E8.75.h5ad")))
fw, w1, w2 = flip_to(W2, T2)
print(f"  frame check: dice(W2,truth) best {w1:.4f} runner-up {w2:.4f}")
fa, _, _ = flip_to(W2 * fw, A2); fb, _, _ = flip_to(W2 * fw, B2)
evaluate("W2 pipeline cloud", W2 * fw, [A2 * fa, B2 * fb], T2)

print()
print("HEART W2 stress: real truth cloud given the LIVE radial profile (board-like over-dispersion)")
r = np.linalg.norm(W2 * fw, axis=1); rL = np.sort(np.linalg.norm(LIVE, axis=1))
rr = np.linalg.norm(T2, axis=1); o = np.argsort(rr, kind="stable")
new_r = np.interp((np.arange(len(rr)) + .5) / len(rr), (np.arange(len(rL)) + .5) / len(rL), rL)
ZS = np.empty_like(T2); ZS[o] = T2[o] / np.maximum(rr[o, None], 1e-12) * new_r[:, None]
ZS /= np.sqrt((ZS ** 2).sum(1).mean())
fs, _, _ = flip_to(ZS, T2)
fa, _, _ = flip_to(ZS * fs, A2); fb, _, _ = flip_to(ZS * fs, B2)
evaluate("stress cloud (truth angles+ranks, live radii)", ZS * fs, [A2 * fa, B2 * fb], T2)

print()
print("=" * 100)
print("EMBRYO board window (anchors FULL E7.25 + E8.0, truth unknown)")
LE = canon(coords("outputs/t2/submit/T2_embryo_val_interp.h5ad"))
AE = canon(coords("data/E7.25.h5ad")); BE = canon(coords("data/E8.0.h5ad"))
fe, e1, e2 = flip_to(LE, AE)
print(f"  frame check: dice(live,E7.25_full) best {e1:.4f} runner-up {e2:.4f}  <- small margin = degenerate PCA")
fe2, e1b, e2b = flip_to(LE, BE)
print(f"               dice(live,E8.0_full)  best {e1b:.4f} runner-up {e2b:.4f}")
evaluate("live embryo (n=5000, nocc 204)", LE * fe, [AE, BE * fe2 if False else BE])

print()
print("EMBRYO W window: anchors FULL E6.75 + E8.0, REAL truth E7.25 @5000")
AW = canon(coords("data/E6.75.h5ad")); BW = canon(coords("data/E8.0.h5ad")); TW = canon(sub(coords("data/E7.25.h5ad")))
ZW = canon(sub(coords("data/E6.75.h5ad")))  # placeholder cloud = one anchor (a 'copy_last'-like prediction)
fz, _, _ = flip_to(ZW, TW); fa, _, _ = flip_to(ZW * fz, AW); fb, _, _ = flip_to(ZW * fz, BW)
evaluate("copy_last-like cloud (E6.75@5000) vs truth E7.25", ZW * fz, [AW * fa, BW * fb], TW)
