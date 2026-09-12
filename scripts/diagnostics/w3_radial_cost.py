"""Safety gate for the radial transport: what does it cost NFS / the X-side metrics?

Radial matching keeps X byte-identical, so de_score / de_direction / mmd_u / variogram
are frozen by construction. It changes the coordinate point set, so d2 / ODS move and
scale_log_ratio must be restored by renormalising the RMS radius. neighborhood_mmd is
the only metric that reads the X<->coord pairing, so it is the one thing to measure.
Also traces WHERE the live cloud's excess tail is created inside the pipeline.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import numpy as np, anndata as ad
from common.shape_metrics import _canonicalise, occupancy_dice, d2_distance
from common.core_metrics import mmd_unbiased, variogram_score, de_score, de_direction
from T2.metrics import neighborhood_mmd
from t2.io import to_dense

def canon(C):
    Z, rms = _canonicalise(np.asarray(C, float)); return Z, rms
def nocc(Z):
    v = np.floor((np.clip(Z, -3.0, 3.0-1e-9) + 3.0)/6.0*16).astype(int); return len(set(map(tuple, v)))
def sorted_r(Z): return np.sort(np.linalg.norm(np.asarray(Z, float), axis=1))
def radial_match(C, target_sorted_r, out_rms):
    """Monotone radial transport in the CANONICAL frame, returned in the original frame's scale."""
    C = np.asarray(C, float); mu = C.mean(0)
    Z, rms = _canonicalise(C)
    r = np.linalg.norm(Z, axis=1); n = len(r); m = len(target_sorted_r)
    new_r = np.interp((np.arange(n)+0.5)/n, (np.arange(m)+0.5)/m, target_sorted_r)
    o = np.argsort(r, kind="stable")
    Z2 = np.empty_like(Z); Z2[o] = (Z[o]/np.maximum(r[o, None], 1e-12)) * new_r[:, None]
    Z2 /= max(np.sqrt((Z2**2).sum(1).mean()), 1e-12)
    return Z2 * float(out_rms)          # canonical frame, RMS = out_rms

def coords(p): return np.asarray(ad.read_h5ad(p).obsm["spatial_3D"], float)
def sub5k(C, seed=0):
    C = np.asarray(C, float)
    return C if len(C) <= 5000 else C[np.random.default_rng(seed).choice(len(C), 5000, replace=False)]

A, B, T = coords("data/E8.25_late.h5ad"), coords("data/E9.5.h5ad"), coords("data/E8.75.h5ad")
ZA, _ = canon(sub5k(A)); ZB, _ = canon(sub5k(B)); ZT, _ = canon(sub5k(T))
rA, rB, rT = sorted_r(ZA), sorted_r(ZB), sorted_r(ZT)

P = ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad")
X = to_dense(P.X).astype(np.float64); C0 = np.asarray(P.obsm["spatial_3D"], float)
XT = to_dense(T if False else ad.read_h5ad("data/E8.75.h5ad").X).astype(np.float64)
XA = to_dense(ad.read_h5ad("data/E8.25_late.h5ad").X).astype(np.float64)
CT = np.asarray(ad.read_h5ad("data/E8.75.h5ad").obsm["spatial_3D"], float)
rms0 = float(np.sqrt(((C0 - C0.mean(0))**2).sum(1).mean()))

print("=" * 100)
print("NFS / expr safety gate on W2 (truth E8.75). X is byte-identical in every row below.")
def gate(tag, C):
    Zc, rms = canon(C)
    print(f"  {tag:40s} rms {rms:7.2f} nocc {nocc(Zc):4d} dice {occupancy_dice(C,CT,seed=0)[0]:.4f}"
          f" d2 {d2_distance(C,CT,seed=0):.5f} NFS {neighborhood_mmd(X,C,XT,CT):.5f}"
          f" mmd {mmd_unbiased(X,XT,seed=0):.5f} de {de_score(X,XT,XA)['score']:.4f}")
gate("baseline (live W2 cloud)", C0)
for nm, tr in (("radial -> interp(w=0.4)", 0.6*rA+0.4*rB), ("radial -> E8.25", rA),
               ("radial -> E9.5", rB), ("[oracle] radial -> truth", rT)):
    gate(nm, radial_match(C0, tr, rms0))

print("\n" + "=" * 100)
print("stress regime (impose the LIVE heart tail on the real truth cloud) - NFS cost there")
LIVE = coords("outputs/t2/submit/T2_heart_val_interp.h5ad")
rL = sorted_r(canon(LIVE)[0])
print(f"  radial profile distances: ||live - truth875|| {np.linalg.norm(rL-rT)/np.sqrt(len(rT)):.4f}"
      f"  ||live - interp0.4|| {np.linalg.norm(rL-(0.6*rA+0.4*rB))/np.sqrt(len(rT)):.4f}")
ZS = radial_match(sub5k(T), rL, rms0)
gate("truth + LIVE tail (board-like defect)", ZS)
gate("  fixed -> interp(w=0.4)", radial_match(ZS, 0.6*rA+0.4*rB, rms0))
gate("  fixed -> E8.25", radial_match(ZS, rA, rms0))

print("\n" + "=" * 100)
print("root cause trace: radial tail of the live board cloud vs its own source anchor")
for nm, p in (("E8.25 raw (5k sub)", None), ("pipeline w0.25 base", "outputs/t2/heart/pred_E8.5_full_anisotropic_otinterp_xpick_xyzleft_w0.25_n5000_rms255.h5ad"),
              ("pipeline xw025_cw05 (68.41)", "outputs/t2/heart/pred_E8.5_full_anisotropic_otinterp_xpick_xyzleft_xw025_cw05_n5000_rms255.h5ad"),
              ("LIVE 68.43", "outputs/t2/submit/T2_heart_val_interp.h5ad")):
    C = sub5k(A) if p is None else coords(p)
    Z, rms = canon(C); q = np.quantile(np.linalg.norm(Z, axis=1), [0.1, 0.5, 0.9, 0.98])
    print(f"  {nm:30s} rms {rms:7.2f} nocc {nocc(Z):4d} q10 {q[0]:.3f} q50 {q[1]:.3f} q90 {q[2]:.3f} q98 {q[3]:.3f}")
