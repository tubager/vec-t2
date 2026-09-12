#!/usr/bin/env python
"""W2 coupling shootout: does keeping each real cell's OWN coordinates fix neighborhood_mmd?

Window: heart anchors E8.25 + E9.5 -> real truth E8.75 (the only heart window with a real truth).
The truth's FOV differs from the anchors (rms 217 vs 354/335), so shape metrics are reported but not
used for decisions; NFS / mmd_u / variogram / de are the informative ones here.

X is held IDENTICAL across variants (the same sampled real cells), only the coordinate assignment
changes, so any NFS difference is pure X<->coord coupling:
  own      each cell keeps its own source coordinate, per-anchor rescaled to the clock-interpolated size
  own_aniso same, per-axis rescaled
  shuffle  coordinates randomly re-paired (coupling destroyed)
  pipeline the production recipe (13_predict_t2.py, OT-place + X pick)
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import numpy as np
import anndata as ad

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, "/tmp/t2shim")

from t2.expression import CompositionT2, labels_to_clusters  # noqa: E402
from common.core_metrics import mmd_unbiased, variogram_score, de_score, de_direction  # noqa: E402
from common.shape_metrics import d2_distance, occupancy_dice, scale_log_ratio  # noqa: E402
from T2.metrics import neighborhood_mmd  # noqa: E402

A_PATH, B_PATH, T_PATH = "data/E8.25_late.h5ad", "data/E9.5.h5ad", "data/E8.75.h5ad"
TA, TB, TT = 8.25, 9.5, 8.75
N = 5000
SEED = 0


def dense(a):
    X = a.X
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)


def rms_axes(C):
    Z = C - C.mean(0)
    return float(np.sqrt((Z ** 2).sum(1).mean())), np.sqrt((Z ** 2).mean(0))


def sample_cells(w_right: float, comp_interp: str = "lin"):
    """Clock composition from the two anchors only (target is NOT fitted -> leakage-free)."""
    A, B = ad.read_h5ad(A_PATH), ad.read_h5ad(B_PATH)
    comp = CompositionT2("heart", {"E8.25": TA, "E9.5": TB}, comp_interp=comp_interp)
    comp.fit({"E8.25": A.obs["celltype"], "E9.5": B.obs["celltype"]}, {"E8.25": TA, "E9.5": TB})
    pi = comp.probs(TT)
    rng = np.random.default_rng(SEED)
    keys = [k for k, p in pi.items() if p > 0]
    p = np.array([pi[k] for k in keys]); p /= p.sum()
    counts = rng.multinomial(N, p)
    clA = np.array(labels_to_clusters(A.obs["celltype"], "heart"))
    clB = np.array(labels_to_clusters(B.obs["celltype"], "heart"))
    XA, CA = dense(A), np.asarray(A.obsm["spatial_3D"], float)
    XB, CB = dense(B), np.asarray(B.obsm["spatial_3D"], float)
    rowsA, rowsB = [], []
    for k, c in zip(keys, counts):
        nb = int(round(c * w_right)); na = int(c) - nb
        ia = np.flatnonzero(clA == k); ib = np.flatnonzero(clB == k)
        if len(ia) == 0 and len(ib) == 0:
            continue
        take = lambda idx, m: idx[rng.choice(len(idx), m, replace=len(idx) < m)] if len(idx) else np.zeros(0, int)
        rowsA.append(take(ia, na)); rowsB.append(take(ib, nb))
    ia = np.concatenate(rowsA).astype(int); ib = np.concatenate(rowsB).astype(int)
    X = np.vstack([XA[ia], XB[ib]])
    Ca = CA[ia]; Cb = CB[ib]
    return X, Ca, Cb


def place(X, Ca, Cb, w_right, mode):
    ra, aa = rms_axes(Ca) if len(Ca) else (1.0, np.ones(3))
    rb, ab = rms_axes(Cb) if len(Cb) else (1.0, np.ones(3))
    if mode == "own_aniso":
        ta = np.exp((1 - w_right) * np.log(aa) + w_right * np.log(ab))
        out = []
        for C, ax in ((Ca, aa), (Cb, ab)):
            if not len(C):
                continue
            Z = C - C.mean(0)
            out.append(Z * (ta / ax) + C.mean(0) * 0.0)
        return np.vstack(out) if out else np.zeros((0, 3))
    rt = float(np.exp((1 - w_right) * np.log(ra) + w_right * np.log(rb)))
    out = []
    for C, r in ((Ca, ra), (Cb, rb)):
        if not len(C):
            continue
        Z = C - C.mean(0)
        out.append(Z * (rt / r))
    own = np.vstack(out) if out else np.zeros((0, 3))
    if mode == "own":
        return own
    if mode == "shuffle":
        return own[np.random.default_rng(7).permutation(len(own))]
    raise ValueError(mode)


def score(name, X, C, tX, tC, refX):
    d = {
        "neighborhood_mmd": neighborhood_mmd(X, C, tX, tC),
        "mmd_u": mmd_unbiased(X, tX, seed=0),
        "variogram": variogram_score(X, tX, seed=0),
        "de_score": de_score(X, tX, refX)["score"],
        "de_direction": de_direction(X, tX, refX),
        "d2": d2_distance(C, tC, seed=0),
        "occ_dice": occupancy_dice(C, tC, seed=0)[0],
        "scale_lr": scale_log_ratio(C, tC),
    }
    print(f"{name:26s} " + "  ".join(f"{k}={v:.5f}" for k, v in d.items()))
    return d


def main():
    T = ad.read_h5ad(T_PATH)
    tX, tC = dense(T), np.asarray(T.obsm["spatial_3D"], float)
    refX = dense(ad.read_h5ad(A_PATH))
    w = (TT - TA) / (TB - TA)
    print(f"# W2 window: anchors {TA}+{TB} -> truth {TT}, clock w={w:.2f}, n={N}")
    X, Ca, Cb = sample_cells(w)
    print(f"# sampled: {len(Ca)} from E8.25, {len(Cb)} from E9.5, X={X.shape}")
    res = {}
    for mode in ("own", "own_aniso", "shuffle"):
        res[mode] = score(f"real cells / {mode}", X, place(X, Ca, Cb, w, mode), tX, tC, refX)
    # truth-vs-truth ceiling for NFS at matched count
    rng = np.random.default_rng(0); i = rng.choice(len(tX), N, replace=False)
    res["ceiling"] = score("CEILING real truth 5000", tX[i], tC[i], tX, tC, refX)
    return res


if __name__ == "__main__":
    main()
