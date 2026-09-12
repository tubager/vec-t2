#!/usr/bin/env python
"""Full board objective (4 equal groups) on honest leave-out windows.

The leaderboard total is NOT the mean of the eight displayed skills.  Fitting every
submission recorded in doc/t2_p2_val_2026-09-05.md reproduces it exactly as four
equally weighted groups (residual <= 0.25 skill, and exactly 0.00 for the files whose
X was byte-frozen):

    expr  = (de_score + de_direction) / 2          weight 1/4
    state = (mmd_u + variogram) / 2                weight 1/4
    shape = (d2_shape + occupancy_dice + scale_log_ratio) / 3   weight 1/4
    local = neighborhood_mmd                       weight 1/4

so a single skill point of `neighborhood_mmd` is worth 3x a single point of `d2_shape`.
skill() is the official hyperbolic map (common/core_metrics.py) with floor = copy_last
and ceiling = split-half of the truth, which is how the boards are calibrated.

Windows (truth is a released stage, never used to build the prediction):
    w1 embryo interp   E6.75 + E8.0  -> E7.25      (ref E6.75)
    w2 heart  interp   E8.25 + E9.5  -> E8.75      (ref E8.25)
    w3 heart  extrap   E8.25 + E8.75 -> E9.5       (ref E8.75)  <- the E10.5 analog
    w4 embryo extrap   E6.75 + E7.25 -> E8.0       (ref E7.25)

Usage:
    .venv/bin/python scripts/diagnostics/27_full_panel.py --window w3 fileA.h5ad name=fileB.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from common.core_metrics import (  # noqa: E402
    de_direction, de_score, energy_distance, mmd_unbiased, skill, split_half, variogram_score,
)
from common.shape_metrics import (  # noqa: E402
    d2_distance, median_nn_distance, occupancy_dice, scale_log_ratio,
)
from T1.metrics import composition_jsd, pseudobulk_pearson, train_frozen_probe  # noqa: E402
from T2.metrics import neighborhood_mmd  # noqa: E402
from common.core_metrics import pb_rel_err, variance_ratio  # noqa: E402
from t2.io import to_dense  # noqa: E402

WINDOWS = {
    "w1": dict(note="embryo interp  E6.75+E8.0 -> E7.25", target="E7.25", ref="E6.75", src="E6.75"),
    "w2": dict(note="heart interp   E8.25+E9.5 -> E8.75", target="E8.75", ref="E8.25_late", src="E8.25_late"),
    "w3": dict(note="heart extrap   E8.25+E8.75 -> E9.5", target="E9.5", ref="E8.75", src="E8.75"),
    "w4": dict(note="embryo extrap  E6.75+E7.25 -> E8.0", target="E8.0", ref="E7.25", src="E7.25"),
}
LOWER = dict(de_score=False, de_direction=False, mmd_u=True, variogram=True, d2_shape=True,
             occupancy_dice=False, scale_log_ratio=True, neighborhood_mmd=True,
             energy_distance=True, pb_rel_err=True, composition_JSD=True,
             pseudobulk_pearson=False, variance_ratio=True)
GROUPS = dict(expr=["de_score", "de_direction"], state=["mmd_u", "variogram"],
              shape=["d2_shape", "occupancy_dice", "scale_log_ratio"], local=["neighborhood_mmd"])
EXTRA = ["energy_distance", "pb_rel_err", "composition_JSD", "pseudobulk_pearson", "variance_ratio"]


def _load(path: Path):
    a = ad.read_h5ad(path)
    X = to_dense(a.X).astype(np.float64)
    C = np.asarray(a.obsm["spatial_3D"], dtype=np.float64)[:, :3] if "spatial_3D" in a.obsm else None
    ct = np.asarray(a.obs["celltype"]).astype(str) if "celltype" in a.obs else np.array(["NA"] * a.n_obs)
    return X, C, ct, [str(g) for g in a.var_names]


def _align(X, genes, panel):
    if genes == panel:
        return X
    idx = {g: i for i, g in enumerate(genes)}
    miss = [g for g in panel if g not in idx]
    if miss:
        raise SystemExit(f"missing panel genes: {miss[:5]}")
    return X[:, [idx[g] for g in panel]]


def _rms(C):
    X = C - C.mean(0)
    return float(np.sqrt((X ** 2).sum(1).mean()))


def measure(Xp, Cp, Xt, Ct, tt_ct, probe, Xr):
    de = de_score(Xp, Xt, Xr)
    dice, vox = occupancy_dice(Cp, Ct, seed=0)
    out = dict(
        de_score=float(de["score"]), de_direction=de_direction(Xp, Xt, Xr),
        mmd_u=mmd_unbiased(Xp, Xt, seed=0), variogram=variogram_score(Xp, Xt, seed=0),
        d2_shape=d2_distance(Cp, Ct, seed=0), occupancy_dice=dice,
        scale_log_ratio=scale_log_ratio(Cp, Ct),
        neighborhood_mmd=neighborhood_mmd(Xp, Cp, Xt, Ct),
        energy_distance=energy_distance(Xp, Xt, seed=0), pb_rel_err=pb_rel_err(Xp, Xt),
        composition_JSD=composition_jsd(Xp, probe, tt_ct),
        pseudobulk_pearson=pseudobulk_pearson(Xp, Xt), variance_ratio=variance_ratio(Xp, Xt),
    )
    out["_nocc_pred"] = len(np.unique(np.floor((np.clip(
        (lambda Z: Z)( (lambda A: (A - A.mean(0)) @ np.linalg.svd(A - A.mean(0), full_matrices=False)[2].T /
                                  max(_rms(A), 1e-12))(Cp)), -3, 3 - 1e-9) + 3) / 6 * 16).astype(int), axis=0))
    out["_nn_pred"] = median_nn_distance(Cp, 0)
    out["_rms_pred"] = _rms(Cp)
    out["_vox_ratio"] = vox
    return out


def rescale(C, target_rms):
    if target_rms is None:
        return C
    C = np.asarray(C, float)
    return (C - C.mean(0)) * (target_rms / _rms(C))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True, choices=sorted(WINDOWS))
    ap.add_argument("--match-rms", action="store_true",
                    help="rescale each prediction's coords to the truth RMS before scoring "
                         "(emulates a submission whose scale_log_ratio is already dialled in)")
    ap.add_argument("--floor-n", type=int, default=3000)
    ap.add_argument("preds", nargs="*", help="path or name=path")
    args = ap.parse_args()

    spec = WINDOWS[args.window]
    Xt, Ct, tt_ct, panel = _load(ROOT / "data" / f"{spec['target']}.h5ad")
    Xr, _, _, gr = _load(ROOT / "data" / f"{spec['ref']}.h5ad")
    Xr = _align(Xr, gr, panel)
    Xa, Ca, _, ga = _load(ROOT / "data" / f"{spec['src']}.h5ad")
    Xa = _align(Xa, ga, panel)
    probe = train_frozen_probe(Xt, tt_ct)

    rms_true = _rms(Ct)
    h1, h2 = split_half(Xt.shape[0], 0)
    ceil = measure(Xt[h1], Ct[h1], Xt, Ct, tt_ct[h2], train_frozen_probe(Xt[h2], tt_ct[h2]), Xr)
    sel = np.random.default_rng(0).choice(Xa.shape[0], min(args.floor_n, Xa.shape[0]), replace=False)
    floor = measure(Xa[sel], Ca[sel], Xt, Ct, tt_ct, probe, Xr)

    rows = [("floor_copy_last", floor), ("ceil_splithalf", ceil)]
    for item in args.preds:
        name, _, p = item.partition("=")
        path = Path(p if p else name)
        if not path.is_absolute():
            path = ROOT / path
        Xp, Cp, _, gp = _load(path)
        Xp = _align(Xp, gp, panel)
        Cp = rescale(Cp, rms_true if args.match_rms else None)
        rows.append((Path(name).stem if p == "" else name, measure(Xp, Cp, Xt, Ct, tt_ct, probe, Xr)))

    keys = list(GROUPS["expr"] + GROUPS["state"] + GROUPS["shape"] + GROUPS["local"] + EXTRA)
    print(f"\n# window {args.window}: {spec['note']}   truth n={Xt.shape[0]} rms={rms_true:.1f} "
          f"nn={median_nn_distance(Ct,0):.2f}")
    hdr = f"{'variant':30s}" + "".join(f"{k[:9]:>10s}" for k in keys)
    print(hdr)
    table = {}
    for name, r in rows:
        sk = {}
        for k in keys:
            s = skill(r[k], floor[k], ceil[k], lower_is_better=LOWER[k])
            sk[k] = float("nan") if s != s else 100.0 * s
        table[name] = (r, sk)
        print(f"{name[:30]:30s}" + "".join(f"{sk[k]:10.1f}" for k in keys))
    print()
    print(f"{'variant':30s}{'expr':>8s}{'state':>8s}{'shape':>8s}{'local':>8s}{'TOTAL':>9s}"
          f"{'nocc':>6s}{'nn':>7s}{'rms':>7s}")
    ranked = []
    for name, (r, sk) in table.items():
        g = {gn: float(np.mean([sk[k] for k in ks])) for gn, ks in GROUPS.items()}
        tot = float(np.mean(list(g.values())))
        ranked.append((tot, name, g, r))
        print(f"{name[:30]:30s}{g['expr']:8.2f}{g['state']:8.2f}{g['shape']:8.2f}{g['local']:8.2f}"
              f"{tot:9.2f}{r['_nocc_pred']:6d}{r['_nn_pred']:7.2f}{r['_rms_pred']:7.1f}")
    print("\n# ranked by TOTAL")
    for tot, name, g, r in sorted(ranked, reverse=True):
        print(f"  {tot:7.2f}  {name}   expr {g['expr']:.1f} state {g['state']:.1f} "
              f"shape {g['shape']:.1f} local {g['local']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
