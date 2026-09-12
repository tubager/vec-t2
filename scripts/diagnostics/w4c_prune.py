#!/usr/bin/env python
"""Prune the voxels that NO full-resolution anchor stage occupies -- and price it honestly.

w4b: the live heart cloud has 289 occupied voxels, of which 39 are occupied by neither FULL
anchor (E8.25_late 58716 cells, E8.75 24826 cells) and hold only 180 cells (3.6%). On the
over-dispersed real-truth stress window that tier prunes at 0.90-0.97 precision and is worth
up to +22 ODS skill; on a cloud that is NOT over-dispersed it is worth +0.2 (self-disabling).

Two candidate mechanisms, both keeping n=5000, X byte-frozen and the RMS radius exact:
  ray     move each unsupported cell inward along ITS OWN ray to the outermost supported
          voxel, spread inside it -> only 180 cells move, no near-duplicates by construction
  profile global monotone radial transport onto the supported-cell radial profile (angles and
          radial rank preserved) -> every cell moves a little
Span-trim (file 05) failed because it teleported cells onto the NEAREST SURVIVOR + jitter,
which manufactures near-duplicate pairs: d2 0.0206 -> 0.0271 (-6.6 skill) while ODS gained
+3.9. Both metrics below are measured on windows with a REAL truth.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
import numpy as np, anndata as ad
from common.shape_metrics import _canonicalise, occupancy_dice, d2_distance
from common.core_metrics import mmd_unbiased
from sklearn.neighbors import NearestNeighbors

FLIPS = [np.array(f, float) for f in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]
N_GRID, EXTENT = 16, 3.0
FLOOR, CEIL = 0.81, 0.9528
D2F, D2C = 0.0461, 0.00213
NFLOOR, NCEIL = 0.10737, 0.00047
sk = lambda v, f, c: min(100.0, 100.0 * abs(c - f) / (abs(c - f) + abs(v - c)))
sk_ods = lambda d: sk(d, FLOOR, CEIL)
sk_d2 = lambda r: sk(r, D2F, D2C)
sk_nfs = lambda r: sk(r, NFLOOR, NCEIL)


def coords(p):
    return np.asarray(ad.read_h5ad(p).obsm["spatial_3D"], float)


def sub(C, n=5000, seed=0):
    C = np.asarray(C, float)
    return C if len(C) <= n else C[np.random.default_rng(seed).choice(len(C), n, replace=False)]


def canon(C):
    return _canonicalise(np.asarray(C, float))[0]


def vidx(Z):
    return np.floor((np.clip(Z, -EXTENT, EXTENT - 1e-9) + EXTENT) / (2 * EXTENT) * N_GRID).astype(int)


def vtup(Z):
    return list(map(tuple, vidx(Z)))


def vtup_arr(Z, mask=None):
    V = vidx(Z)
    return V if mask is None else V[np.asarray(mask)]


def nvox(Z, mask=None):
    V = vidx(Z)
    if mask is not None:
        V = V[np.asarray(mask)]
    return len(set(map(tuple, V)))


def flip_to(Zp, Zt):
    vt = set(vtup(Zt)); sc = []
    for f in FLIPS:
        vp = set(vtup(Zp * f)); sc.append(2 * len(vp & vt) / (len(vp) + len(vt)))
    return FLIPS[int(np.argmax(sc))]


def support(Z, anchors):
    """Per-cell support count: how many FULL anchor clouds occupy the cell's voxel."""
    t = vtup(Z); sets = [set(vtup(a)) for a in anchors]
    return np.array([sum(1 for s in sets if v in s) for v in t])


def renorm(Z):
    Z = Z - Z.mean(0)
    return Z / max(np.sqrt((Z ** 2).sum(1).mean()), 1e-12)


def prune_ray(Z, anchors, eps=0.03, steps=400):
    """Move unsupported cells inward along their own ray to the outermost supported shell."""
    sup = support(Z, anchors)
    bad = np.where(sup == 0)[0]
    out = Z.copy()
    if len(bad) == 0:
        return out, 0
    r = np.linalg.norm(Z, axis=1)
    grid = np.linspace(0.0, 1.0, steps)
    sets = [set(vtup(a)) for a in anchors]
    supported = set.union(*[s for s in sets]) if sets else set()
    # group by voxel so cells spread inside the landing voxel instead of stacking
    byv = {}
    for i in bad:
        byv.setdefault(vtup(Z)[i], []).append(i)
    moved = 0
    for v, idxs in byv.items():
        for j, i in enumerate(idxs):
            u = Z[i] / max(r[i], 1e-12)
            cand = grid[grid < r[i]][::-1]
            land = None
            for rr in cand:
                if tuple(map(int, vidx((u * rr)[None])[0])) in supported:
                    land = rr; break
            if land is None:
                land = np.quantile(r[sup > 0], 0.95)
            land *= 1.0 - eps * (j + 0.5) / max(len(idxs), 1)
            out[i] = u * land
            moved += 1
    return renorm(out), moved


def prune_profile(Z, anchors):
    """Global monotone radial transport onto the radial profile of the SUPPORTED cells."""
    sup = support(Z, anchors)
    r = np.linalg.norm(Z, axis=1)
    tgt = np.sort(r[sup > 0])
    n, m = len(r), len(tgt)
    new_r = np.interp((np.arange(n) + .5) / n, (np.arange(m) + .5) / m, tgt)
    o = np.argsort(r, kind="stable")
    out = np.empty_like(Z)
    out[o] = Z[o] / np.maximum(r[o, None], 1e-12) * new_r[:, None]
    return renorm(out), int((sup == 0).sum())


def knn_overlap(C1, C2, k=15):
    _, i1 = NearestNeighbors(n_neighbors=k + 1).fit(C1).kneighbors(C1)
    _, i2 = NearestNeighbors(n_neighbors=k + 1).fit(C2).kneighbors(C2)
    return float(np.mean([len(set(a[1:]) & set(b[1:])) / k for a, b in zip(i1, i2)]))


def report(tag, Z, X, Zt, Xt):
    n_occ = nvox(Z)
    d, _ = occupancy_dice(Z, Zt, seed=0)
    r2 = d2_distance(Z, Zt, seed=0)
    line = (f"  {tag:42s} nocc {n_occ:4d}  dice {d:.4f} ODS {sk_ods(d):5.1f} | "
            f"d2 {r2:.5f} skill {sk_d2(r2):5.1f} | shape {(sk_ods(d)+sk_d2(r2)+100)/3:5.1f}")
    if X is not None:
        nb_p = NearestNeighbors(n_neighbors=16).fit(Z).kneighbors(Z)[1][:, 1:]
        nb_t = NearestNeighbors(n_neighbors=16).fit(Zt).kneighbors(Zt)[1][:, 1:]
        pb_p = X[nb_p].mean(1); pb_t = Xt[nb_t].mean(1)
        nfs = mmd_unbiased(pb_p, pb_t, n=2000)
        line += f" | NFS {nfs:.5f} skill {sk_nfs(nfs):5.1f}"
    print(line)
    return d, r2


A = canon(coords("data/E8.25_late.h5ad"))       # FULL 58716
B = canon(coords("data/E9.5.h5ad"))             # FULL
T = canon(sub(coords("data/E8.75.h5ad")))       # truth @5000
XT = None

print("=" * 108)
print("W2 stress window: REAL truth E8.75 coordinates, live heart radial profile imposed (board-like defect)")
P = ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad")
LIVE = canon(coords("outputs/t2/submit/T2_heart_val_interp.h5ad"))
rL = np.sort(np.linalg.norm(LIVE, axis=1))
rr = np.linalg.norm(T, axis=1); o = np.argsort(rr, kind="stable")
new_r = np.interp((np.arange(len(rr)) + .5) / len(rr), (np.arange(len(rL)) + .5) / len(rL), rL)
ZS = np.empty_like(T); ZS[o] = T[o] / np.maximum(rr[o, None], 1e-12) * new_r[:, None]
ZS = renorm(ZS)
f = flip_to(ZS, T); ZS = ZS * f
fa = flip_to(ZS, A); fb = flip_to(ZS, B)
ANCH = [A * fa, B * fb]
sup = support(ZS, ANCH)
print(f"  support census: sup0 {int((sup==0).sum())} cells / {nvox(ZS, sup==0)} voxels, "
      f"sup1 {int((sup==1).sum())}, sup2 {int((sup==2).sum())}")
report("stress baseline", ZS, None, T, None)
Z1, m1 = prune_ray(ZS, ANCH)
report(f"stress + prune_ray (moved {m1} cells)", Z1, None, T, None)
Z2, m2 = prune_profile(ZS, ANCH)
report(f"stress + prune_profile ({m2} unsupported)", Z2, None, T, None)
print(f"  kNN-15 membership overlap vs baseline: ray {knn_overlap(ZS, Z1):.4f}  profile {knn_overlap(ZS, Z2):.4f}")
print(f"  min pairwise distance: base {NearestNeighbors(n_neighbors=2).fit(ZS).kneighbors(ZS)[0][:,1].min():.4f}"
      f"  ray {NearestNeighbors(n_neighbors=2).fit(Z1).kneighbors(Z1)[0][:,1].min():.4f}"
      f"  profile {NearestNeighbors(n_neighbors=2).fit(Z2).kneighbors(Z2)[0][:,1].min():.4f}")

print()
print("=" * 108)
print("W2 real pipeline cloud (NOT over-dispersed): the rule must self-disable")
ZW = canon(np.asarray(P.obsm["spatial_3D"], float))
XW = P.X.toarray() if hasattr(P.X, "toarray") else np.asarray(P.X)
fw = flip_to(ZW, T); ZW = ZW * fw
fa = flip_to(ZW, A); fb = flip_to(ZW, B)
ANCHW = [A * fa, B * fb]
supW = support(ZW, ANCHW)
print(f"  support census: sup0 {int((supW==0).sum())} cells / {nvox(ZW, supW==0)} voxels")
XT2 = None
Tad = ad.read_h5ad("data/E8.75.h5ad")
idx = np.random.default_rng(0).choice(len(Tad), 5000, replace=False)
XT2 = Tad.X.toarray()[idx] if hasattr(Tad.X, "toarray") else np.asarray(Tad.X)[idx]
report("W2 pipeline baseline", ZW, XW, T, XT2)
ZW1, n1 = prune_ray(ZW, ANCHW)
report(f"W2 pipeline + prune_ray (moved {n1})", ZW1, XW, T, XT2)
ZW2, n2 = prune_profile(ZW, ANCHW)
report(f"W2 pipeline + prune_profile", ZW2, XW, T, XT2)
print(f"  kNN-15 overlap: ray {knn_overlap(ZW, ZW1):.4f}  profile {knn_overlap(ZW, ZW2):.4f}")

print()
print("=" * 108)
print("BOARD file (no truth): what would we actually ship?")
ZL = canon(coords("outputs/t2/submit/T2_heart_val_interp.h5ad"))
A2 = canon(coords("data/E8.25_late.h5ad")); B2 = canon(coords("data/E8.75.h5ad"))
fA = flip_to(ZL, A2); fB = flip_to(ZL, B2)
print(f"  frame: flip vs E8.25_full {fA}, vs E8.75_full {fB}")
ANCHL = [A2 * fA, B2 * fB]
supL = support(ZL, ANCHL)
bad = supL == 0
r = np.linalg.norm(ZL, axis=1)
print(f"  nocc {nvox(ZL)}; unsupported cells {bad.sum()} in {nvox(ZL, bad)} voxels")
print(f"  radius quantiles of unsupported cells {np.round(np.quantile(r[bad],[.1,.5,.9,1.0]),3)}"
      f" vs supported {np.round(np.quantile(r[~bad],[.1,.5,.9,1.0]),3)}")
print(f"  cells per unsupported voxel: {np.round(np.bincount(np.unique(vtup_arr(ZL, bad),axis=0,return_inverse=True)[1]),1)[:20]}")
for nm, fn in (("ray", prune_ray), ("profile", prune_profile)):
    Zo, mv = fn(ZL, ANCHL)
    so = support(Zo, ANCHL)
    dA, _ = occupancy_dice(Zo, A2 * fA, seed=0); dB, _ = occupancy_dice(Zo, B2 * fB, seed=0)
    dA0, _ = occupancy_dice(ZL, A2 * fA, seed=0); dB0, _ = occupancy_dice(ZL, B2 * fB, seed=0)
    print(f"  {nm:8s}: moved {mv:4d}  nocc {nvox(ZL):3d} -> {nvox(Zo):3d}  "
          f"still-unsupported cells {int((so==0).sum()):3d}  "
          f"dice vs E8.25_full {dA0:.4f}->{dA:.4f}  vs E8.75_full {dB0:.4f}->{dB:.4f}  "
          f"d2 vs E8.25_full {d2_distance(ZL,A2*fA,seed=0):.5f}->{d2_distance(Zo,A2*fA,seed=0):.5f}  "
          f"kNN15 {knn_overlap(ZL,Zo):.4f}")
