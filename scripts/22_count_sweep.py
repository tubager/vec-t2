"""Cell-count (n) sweep: occupancy_dice is a COUNT statistic, and the panel subsamples
mmd/variogram/NFS to min(2000|1500, n) rows, so for n >= 2000 shrinking n changes the
shape metrics' count-matching without changing the expression metrics' sample size.

Part 1 (proxy): nocc/dice/d2 of our live cloud vs BOTH anchors at matched n.
Part 2 (honest W2): every metric vs the real truth E8.75 as n shrinks.

Nothing is moved or clumped -- rows are dropped, so X<->coord pairing is untouched and
d2 (scale-free, RMS-normalised) should be flat. Uniform rescale keeps rms_radius fixed
(both occupancy_dice and d2 canonicalise by RMS, so the rescale is metric-neutral).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import numpy as np, anndata as ad
from common.shape_metrics import occupancy_dice, d2_distance, scale_log_ratio
from t2.geometry import rms_radius, scale_cloud
from t2.io import to_dense

# board-reverse-engineered floor/ceiling (heart val_interp)
FLOOR_CEIL = {"d2": (0.0461, 0.00213), "ods": (0.81, 0.9528)}
skill_d2 = lambda raw: 100.0 * (FLOOR_CEIL["d2"][0] - raw) / (FLOOR_CEIL["d2"][0] - FLOOR_CEIL["d2"][1])
skill_ods = lambda dice: 100.0 * (dice - FLOOR_CEIL["ods"][0]) / (FLOOR_CEIL["ods"][1] - FLOOR_CEIL["ods"][0])


def nocc(C, n_grid=16, extent=3.0):
    from common.shape_metrics import _canonicalise
    Z, _ = _canonicalise(np.asarray(C, float))
    v = np.floor((np.clip(Z, -extent, extent - 1e-9) + extent) / (2 * extent) * n_grid).astype(int)
    return len(set(map(tuple, v)))


def sub(C, n, seed=0):
    C = np.asarray(C, float)
    if n >= C.shape[0]:
        return C
    return C[np.random.default_rng(seed).choice(C.shape[0], n, replace=False)]


def part1():
    print("=" * 78)
    print("PART 1 - shape proxy: our live cloud vs both anchors at matched n")
    for tag, live, anchors in (
        ("heart interp (live 68.43)", "outputs/t2/submit/T2_heart_val_interp.h5ad",
         [("E8.25", "data/E8.25_late.h5ad"), ("E8.75", "data/E8.75.h5ad")]),
        ("embryo interp (live 66.41)", "outputs/t2/submit/T2_embryo_val_interp.h5ad",
         [("E7.25", "data/E7.25.h5ad"), ("E8.0", "data/E8.0.h5ad")]),
    ):
        P = ad.read_h5ad(live)
        Cp = np.asarray(P.obsm["spatial_3D"], float)
        full = {nm: np.asarray(ad.read_h5ad(p).obsm["spatial_3D"], float) for nm, p in anchors}
        print(f"\n--- {tag}: live n={len(Cp)}, rms={rms_radius(Cp):.2f} ---")
        print(f"  nocc(live full {len(Cp)}) = {nocc(Cp)} | anchors: "
              + " ".join(f"{nm}(5000)={nocc(sub(c,5000))} full={nocc(c)}" for nm, c in full.items()))
        print(f"  {'n':>6s} {'nocc_ours':>9s} " + " ".join(f"{'nocc_'+nm:>11s} {'dice':>7s} {'d2':>8s}" for nm in full)
              + f" {'ODSskill':>9s} {'d2skill':>8s} {'rms':>7s}")
        for n in (5000, 4500, 4000, 3500, 3000, 2500, 2000):
            if n > len(Cp):
                continue
            Co = sub(Cp, n)
            Co = np.asarray(scale_cloud(Co, rms_radius(Cp)), float)   # keep rms fixed (metric-neutral)
            row, ds, os_ = [], [], []
            for nm, c in full.items():
                dice, _ = occupancy_dice(Co, c, seed=0)
                raw = d2_distance(Co, c, seed=0)
                row.append(f"{nocc(sub(c,n)):>11d} {dice:7.4f} {raw:8.5f}")
                ds.append(skill_d2(raw)); os_.append(skill_ods(dice))
            print(f"  {n:6d} {nocc(Co):9d} " + " ".join(row)
                  + f" {np.mean(os_):9.2f} {np.mean(ds):8.2f} {rms_radius(Co):7.2f}")


def part2():
    print("\n" + "=" * 78)
    print("PART 2 - honest W2 (truth E8.75): do the expression/state/NFS metrics move with n?")
    from common.core_metrics import mmd_unbiased, variogram_score, de_score, de_direction
    from T2.metrics import neighborhood_mmd
    P = ad.read_h5ad("outputs/t2/heart/w2_pipeline.h5ad")
    T = ad.read_h5ad("data/E8.75.h5ad"); A = ad.read_h5ad("data/E8.25_late.h5ad")
    X = to_dense(P.X).astype(np.float64); C = np.asarray(P.obsm["spatial_3D"], float)
    XT = to_dense(T.X).astype(np.float64); CT = np.asarray(T.obsm["spatial_3D"], float)
    XA = to_dense(A.X).astype(np.float64)
    print(f"  {'n':>6s} {'de':>7s} {'ddir':>7s} {'mmd':>8s} {'var':>8s} {'NFS':>8s} {'dice':>7s} {'d2':>8s}")
    for n in (5000, 4500, 4000, 3500, 3000, 2500, 2000):
        idx = np.random.default_rng(0).choice(X.shape[0], n, replace=False) if n < X.shape[0] else np.arange(X.shape[0])
        Xn, Cn = X[idx], np.asarray(scale_cloud(C[idx], rms_radius(C)), float)
        dice, _ = occupancy_dice(Cn, CT, seed=0)
        print(f"  {n:6d} {de_score(Xn,XT,XA)['score']:7.4f} {de_direction(Xn,XT,XA):7.4f}"
              f" {mmd_unbiased(Xn,XT,seed=0):8.5f} {variogram_score(Xn,XT,seed=0):8.5f}"
              f" {neighborhood_mmd(Xn,Cn,XT,CT):8.5f} {dice:7.4f} {d2_distance(Cn,CT,seed=0):8.5f}")


if __name__ == "__main__":
    part1()
    part2()
