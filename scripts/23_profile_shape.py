"""Shape-group headroom: occupancy_dice is a DENSITY-PROFILE statistic, and the shape
group (d2 + ODS + scale)/3 = 25% of the total is the only group whose headroom is still
large (ODS 43.5 with a ceiling of 100) AND whose fix touches coordinates only, so X,
expr, state and NFS are untouched by construction.

A: board-implied dice arithmetic -> how many voxels are spurious, what is the ceiling.
B: profile comparison of the live cloud against both anchors and the interpolated target
   (radial quantiles, per-axis std, nocc).
C: honest W2 test (real truth cloud E8.75) of monotone radial-CDF matching, plus a
   stress test that first imposes the live cloud's over-dispersion on the real truth.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import numpy as np, anndata as ad
from common.shape_metrics import _canonicalise, occupancy_dice, d2_distance, scale_log_ratio

FLOOR, CEIL = 0.81, 0.9528          # board occupancy_dice floor/ceiling (reverse engineered)
D2F, D2C = 0.0461, 0.00213          # board d2 floor/ceiling
skill_ods = lambda d: 100.0 * (d - FLOOR) / (CEIL - FLOOR)
skill_d2 = lambda r: 100.0 * (D2F - r) / (D2F - D2C)
rng = np.random.default_rng(0)


def sub5k(C, seed=0):
    C = np.asarray(C, float)
    return C if len(C) <= 5000 else C[np.random.default_rng(seed).choice(len(C), 5000, replace=False)]


def nocc(Z):
    v = np.floor((np.clip(Z, -3.0, 3.0 - 1e-9) + 3.0) / 6.0 * 16).astype(int)
    return len(set(map(tuple, v)))


def canon(C):
    Z, rms = _canonicalise(np.asarray(C, float))
    return Z, rms


def radial_q(Z, qs=np.linspace(0.02, 0.98, 49)):
    r = np.linalg.norm(Z, axis=1)
    return np.quantile(r, qs), r


def radial_match(Z, target_q, qs=np.linspace(0.02, 0.98, 49)):
    """Monotone radial transport onto a target radial quantile function; angles preserved."""
    r = np.linalg.norm(Z, axis=1)
    n = len(r)
    order = np.argsort(r)
    u = (np.arange(n) + 0.5) / n
    new_r = np.interp(u, qs, target_q)
    out = Z.copy()
    scale = np.zeros(n)
    nz = r > 1e-12
    scale[nz] = 1.0
    Z2 = Z[order] * (new_r[:, None] / np.maximum(r[order, None], 1e-12))
    out[order] = Z2
    Z2 = Z2 / max(np.sqrt((Z2 ** 2).sum(1).mean()), 1e-12)      # keep RMS radius = 1
    return Z2


def load_coords(path):
    return np.asarray(ad.read_h5ad(path).obsm["spatial_3D"], float)


print("=" * 84)
print("A. board-implied occupancy arithmetic (heart interp, live 68.43)")
ours = nocc(canon(sub5k(load_coords("outputs/t2/submit/T2_heart_val_interp.h5ad")))[0])
a25 = nocc(canon(sub5k(load_coords("data/E8.25_late.h5ad")))[0])
a75 = nocc(canon(sub5k(load_coords("data/E8.75.h5ad")))[0])
print(f"  nocc: ours {ours} | E8.25 {a25} | E8.75 {a75}")
for dice_skill, label in ((43.5, "live 68.43"), (47.2, "span-trim 05 (68.22)")):
    dice = FLOOR + dice_skill / 100 * (CEIL - FLOOR)
    for nt in (a25, a75, (a25 + a75) // 2):
        inter = dice * (ours + nt) / 2
        print(f"  {label:22s} dice {dice:.4f} vs n_true={nt}: |A^B| {inter:6.1f}"
              f"  recall {inter/nt:.3f}  precision {inter/ours:.3f}  spurious {ours-inter:5.1f}")
nt = (a25 + a75) // 2
dice_now = FLOOR + 0.435 * (CEIL - FLOOR)
inter_now = dice_now * (ours + nt) / 2
for keep_extra in (0.5, 0.25, 0.0):
    A = inter_now + (ours - inter_now) * keep_extra
    d = 2 * A / (A + nt)
    print(f"  if spurious voxels reduced to {keep_extra:.0%}: |A| {A:.0f} -> dice {d:.4f}"
          f" -> ODS skill {min(skill_ods(d),100):.1f}  (+{min(skill_ods(d),100)-43.5:.1f} skill,"
          f" +{(min(skill_ods(d),100)-43.5)/12:.2f} total)")

print("\n" + "=" * 84)
print("B. density profile: live cloud vs anchors vs interpolated target (t=8.5, w=0.5)")
qs = np.linspace(0.02, 0.98, 49)
Zc = {}
for nm, p in (("OURS", "outputs/t2/submit/T2_heart_val_interp.h5ad"),
              ("E8.25", "data/E8.25_late.h5ad"), ("E8.75", "data/E8.75.h5ad"), ("E9.5", "data/E9.5.h5ad")):
    Z, rms = canon(sub5k(load_coords(p)))
    q, r = radial_q(Z, qs)
    Zc[nm] = (Z, q, rms)
    ax = Z.std(0)
    print(f"  {nm:6s} rms {rms:7.2f} nocc {nocc(Z):4d} axis-std {ax.round(4)} ratio {ax[0]/ax[2]:.3f}"
          f" | r quantiles q10 {q[4]:.3f} q50 {q[24]:.3f} q90 {q[44]:.3f} q98 {q[48]:.3f}"
          f" | kurtosis {((r**4).mean()/ (r**2).mean()**2):.3f}")
tgt = 0.5 * Zc["E8.25"][1] + 0.5 * Zc["E8.75"][1]
print(f"  TARGET(w=0.5)        q10 {tgt[4]:.3f} q50 {tgt[24]:.3f} q90 {tgt[44]:.3f} q98 {tgt[48]:.3f}")
d = Zc["OURS"][1] - tgt
print(f"  OURS - TARGET        q10 {d[4]:+.3f} q50 {d[24]:+.3f} q90 {d[44]:+.3f} q98 {d[48]:+.3f}"
      f"   (positive = our cloud is WIDER at that quantile)")
print(f"  radial L2 mismatch  ours-vs-target {np.linalg.norm(d):.4f}"
      f" | E8.25-vs-target {np.linalg.norm(Zc['E8.25'][1]-tgt):.4f}"
      f" | E8.75-vs-target {np.linalg.norm(Zc['E8.75'][1]-tgt):.4f}")
