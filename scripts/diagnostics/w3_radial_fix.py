"""Honest test of the monotone radial-transport fix for the shape group.

The live heart cloud is 2.2x farther from the anchor-interpolated radial profile than
either real anchor is, and the excess sits entirely in the tail (q98 +0.29 RMS units),
which is exactly what produces 53-59 spurious voxels (ODS precision 0.80) and excess
large-distance mass (d2). The fix is a monotone radial transport: keep every angle and
the rank order of radii, map the radius onto the target radial quantile function, then
renormalise the RMS radius. X is untouched, so expr/state are frozen by construction;
angles and radial rank are preserved, so neighbourhoods (NFS) deform smoothly.

Test 1 (W2, real truth E8.75): does anchor-interpolated radial matching improve dice/d2?
Test 2 (stress, the regime that matters): impose the LIVE cloud's excess tail on the real
truth cloud, confirm dice/d2 collapse, then confirm an anchor-only fix recovers them.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import numpy as np, anndata as ad
from common.shape_metrics import _canonicalise, occupancy_dice, d2_distance

FLOOR, CEIL = 0.81, 0.9528
D2F, D2C = 0.0461, 0.00213
sk_ods = lambda d: min(100.0, 100.0 * (d - FLOOR) / (CEIL - FLOOR))
sk_d2 = lambda r: 100.0 * (D2F - r) / (D2F - D2C)


def coords(p):
    return np.asarray(ad.read_h5ad(p).obsm["spatial_3D"], float)


def sub5k(C, seed=0):
    C = np.asarray(C, float)
    return C if len(C) <= 5000 else C[np.random.default_rng(seed).choice(len(C), 5000, replace=False)]


def canon(C):
    Z, rms = _canonicalise(np.asarray(C, float))
    return Z, rms


def nocc(Z):
    v = np.floor((np.clip(Z, -3.0, 3.0 - 1e-9) + 3.0) / 6.0 * 16).astype(int)
    return len(set(map(tuple, v)))


def sorted_r(Z):
    return np.sort(np.linalg.norm(np.asarray(Z, float), axis=1))


def radial_match(Z, target_sorted_r):
    """Monotone radial transport onto a target radius distribution; angles + rank preserved."""
    Z = np.asarray(Z, float)
    r = np.linalg.norm(Z, axis=1)
    n = len(r); m = len(target_sorted_r)
    u = (np.arange(n) + 0.5) / n
    um = (np.arange(m) + 0.5) / m
    new_r = np.interp(u, um, target_sorted_r)
    order = np.argsort(r, kind="stable")
    unit = Z[order] / np.maximum(r[order, None], 1e-12)
    out = np.empty_like(Z)
    out[order] = unit * new_r[:, None]
    return out / max(np.sqrt((out ** 2).sum(1).mean()), 1e-12)


def report(tag, Zp, Zt):
    dice, _ = occupancy_dice(Zp, Zt, seed=0)
    d2 = d2_distance(Zp, Zt, seed=0)
    print(f"  {tag:44s} nocc {nocc(canon(Zp)[0]):4d}  dice {dice:.4f} (ODS {sk_ods(dice):5.1f})"
          f"  d2 {d2:.5f} (skill {sk_d2(d2):5.1f})  shape_grp {(sk_ods(dice)+sk_d2(d2)+100)/3:5.1f}")
    return dice, d2


A, B, T = coords("data/E8.25_late.h5ad"), coords("data/E9.5.h5ad"), coords("data/E8.75.h5ad")
LIVE = coords("outputs/t2/submit/T2_heart_val_interp.h5ad")
ZA, _ = canon(sub5k(A)); ZB, _ = canon(sub5k(B)); ZT, rmsT = canon(sub5k(T))
ZL, _ = canon(LIVE)
rA, rB, rT, rL = sorted_r(ZA), sorted_r(ZB), sorted_r(ZT), sorted_r(ZL)

print("=" * 104)
print("radial-profile distances (L2 over 5000 sorted radii, RMS-normalised units)")
print(f"  truth E8.75 vs E8.25            {np.linalg.norm(rT-rA)/np.sqrt(len(rT)):.4f}")
print(f"  truth E8.75 vs E9.5             {np.linalg.norm(rT-rB)/np.sqrt(len(rT)):.4f}")
w = 0.4
print(f"  truth E8.75 vs interp(w=0.4)    {np.linalg.norm(rT-(0.6*rA+0.4*rB))/np.sqrt(len(rT)):.4f}"
      "   <- best anchor-only estimate of the truth profile")
print(f"  LIVE heart vs interp(w=0.5)     {np.linalg.norm(rL-(0.5*sorted_r(canon(sub5k(A))[0])+0.5*sorted_r(canon(sub5k(coords('data/E8.75.h5ad')))[0])))/np.sqrt(len(rL)):.4f}")

print("\n" + "=" * 104)
print("TEST 1 - W2 (truth E8.75): radial-match the live W2 pipeline cloud")
P = ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad")
ZW, _ = canon(np.asarray(P.obsm["spatial_3D"], float))
rW = sorted_r(ZW)
print(f"  W2 pipeline radial mismatch vs interp(w=0.4): {np.linalg.norm(rW-(0.6*rA+0.4*rB))/np.sqrt(len(rW)):.4f}"
      f" | vs truth: {np.linalg.norm(rW-rT)/np.sqrt(len(rW)):.4f}")
report("baseline (pipeline cloud)", ZW, ZT)
report("radial-matched to interp(w=0.4) anchors", radial_match(ZW, 0.6 * rA + 0.4 * rB), ZT)
report("radial-matched to E8.25 only", radial_match(ZW, rA), ZT)
report("radial-matched to E9.5 only", radial_match(ZW, rB), ZT)
report("[oracle] radial-matched to TRUTH profile", radial_match(ZW, rT), ZT)

print("\n" + "=" * 104)
print("TEST 2 - stress: impose the LIVE cloud's excess tail on the REAL truth cloud")
ZS = radial_match(ZT, rL)                      # truth angles, live radii -> board-like over-dispersion
print(f"  truth profile -> live profile: nocc {nocc(ZT)} -> {nocc(ZS)}")
report("real truth cloud (ceiling reference)", ZT, ZT)
report("truth + LIVE radial tail (board-like defect)", ZS, ZT)
report("  fixed with interp(w=0.4) anchors", radial_match(ZS, 0.6 * rA + 0.4 * rB), ZT)
report("  fixed with E8.25 only", radial_match(ZS, rA), ZT)
report("  fixed with E9.5 only", radial_match(ZS, rB), ZT)
report("  [oracle] fixed with truth profile", radial_match(ZS, rT), ZT)

print("\n" + "=" * 104)
print("TEST 3 - does the transport preserve local structure (NFS proxy: kNN membership)?")
from sklearn.neighbors import NearestNeighbors
def knn_overlap(C1, C2, k=15):
    _, i1 = NearestNeighbors(n_neighbors=k + 1).fit(C1).kneighbors(C1)
    _, i2 = NearestNeighbors(n_neighbors=k + 1).fit(C2).kneighbors(C2)
    s1 = [set(r[1:]) for r in i1]; s2 = [set(r[1:]) for r in i2]
    return float(np.mean([len(a & b) / k for a, b in zip(s1, s2)]))
Zfix = radial_match(ZW, 0.6 * rA + 0.4 * rB)
print(f"  kNN-15 membership overlap, baseline vs radial-matched: {knn_overlap(ZW, Zfix):.4f}")
Zsh = ZW[np.random.default_rng(0).permutation(len(ZW))]
print(f"  (reference) baseline vs random re-coordinate:          {knn_overlap(ZW, Zsh):.4f}")
