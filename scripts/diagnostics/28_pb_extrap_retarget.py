#!/usr/bin/env python
"""W3 test: multiplicative per-gene pseudobulk EXTRAPOLATION for extrapolation targets.

Interpolation windows have no estimable pb target better than verbatim anchor mixing
(doc 12.2: clock/geo targets all lose).  EXTRAPOLATION is different: with anchors at
t_A < t_B and a target at t_C > t_B the log-linear continuation

    log pb_C = log pb_B + k * (log pb_B - log pb_A),   k = (t_C - t_B) / (t_B - t_A)

is the unique degree-1 estimator, and it is applied to the prediction *multiplicatively
per gene*, so the per-cell (X, xyz) pairing and every gene's within-cloud shape are
preserved -- unlike the additive per-cluster delta, which doc 12/W3 measured as a net
loss (state group -6 for expr +1).

W3 window: truth E9.5, anchors E8.25 (A) + E8.75 (B), k = (9.5-0.75)/0.5 = 1.5.
Board extrap: truth E10.5, anchors E8.75 (A) + E9.5 (B), k = (10.5-9.5)/0.75 = 1.333.
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
    de_direction, de_score, mmd_unbiased, pb_rel_err, skill, split_half, variogram_score,
)
from common.shape_metrics import d2_distance, occupancy_dice, scale_log_ratio  # noqa: E402
from T1.metrics import composition_jsd, train_frozen_probe  # noqa: E402
from T2.metrics import neighborhood_mmd  # noqa: E402

GROUPS = {
    "expr": ["de_score", "de_direction"],
    "state": ["mmd_u", "variogram"],
    "shape": ["d2_shape", "occupancy_dice", "scale_log_ratio"],
    "local": ["neighborhood_mmd"],
}
LOWER = dict(de_score=False, de_direction=False, mmd_u=True, variogram=True, d2_shape=True,
             occupancy_dice=False, scale_log_ratio=True, neighborhood_mmd=True)


def dense(a):
    X = a.X
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)


def load(name):
    a = ad.read_h5ad(ROOT / "data" / f"{name}.h5ad")
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/t2/heart/w3sweep/w3_nodelta.h5ad")
    ap.add_argument("--anchor-a", default="E8.25_late")
    ap.add_argument("--anchor-b", default="E8.75")
    ap.add_argument("--truth", default="E9.5")
    ap.add_argument("--ref", default="E8.75")
    ap.add_argument("--ta", type=float, default=8.25)
    ap.add_argument("--tb", type=float, default=8.75)
    ap.add_argument("--tc", type=float, default=9.5)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--lams", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--damp", type=float, default=1.0,
                    help="shrink the extrapolated log-ratio toward 0 (0=freeze at anchor B)")
    args = ap.parse_args()

    A, B, T, R = load(args.anchor_a), load(args.anchor_b), load(args.truth), load(args.ref)
    panel = list(T.var_names)
    def align(a):
        X = dense(a)
        idx = {g: i for i, g in enumerate(a.var_names)}
        return X[:, [idx[g] for g in panel]]
    XA, XB, XT, XR = align(A), align(B), align(T), align(R)
    tt_ct = np.asarray(T.obs[T.obs.columns[T.obs.columns.str.contains("type|cluster|cell")][0]].values)

    P = ad.read_h5ad(ROOT / args.pred)
    X = align(P)
    C = np.asarray(P.obsm["spatial_3D"], float)
    Ct = np.asarray(T.obsm["spatial_3D"], float)

    k = (args.tc - args.tb) / (args.tb - args.ta)
    e = args.eps
    lpa, lpb = np.log(XA.mean(0) + e), np.log(XB.mean(0) + e)
    lpt = lpb + args.damp * k * (lpb - lpa)
    tgt = np.exp(lpt) - e
    tgt = np.maximum(tgt, 0.0)
    print(f"# k={k:.3f} damp={args.damp}  rel_err(target_pb, truth_pb)="
          f"{np.linalg.norm(tgt - XT.mean(0)) / np.linalg.norm(XT.mean(0)):.4f}")
    print(f"# rel_err(ours)={pb_rel_err(X, XT):.4f}")

    # floor / ceiling for skill
    sel = np.random.default_rng(0).choice(XB.shape[0], 3000, replace=False)
    h1, h2 = split_half(XT.shape[0], 0)
    probe_t = train_frozen_probe(XT, tt_ct)

    def row(Xc, Cc):
        return dict(
            de_score=float(de_score(Xc, XT, XR)["score"]), de_direction=de_direction(Xc, XT, XR),
            mmd_u=mmd_unbiased(Xc, XT, seed=0), variogram=variogram_score(Xc, XT, seed=0),
            d2_shape=d2_distance(Cc, Ct, seed=0), occupancy_dice=occupancy_dice(Cc, Ct, seed=0)[0],
            scale_log_ratio=scale_log_ratio(Cc, Ct), neighborhood_mmd=neighborhood_mmd(Xc, Cc, XT, Ct),
            pb_rel_err=pb_rel_err(Xc, XT),
        )

    fl = row(XB[sel], np.asarray(B.obsm["spatial_3D"], float)[sel])
    ce = row(XT[h1], Ct[h1])
    keys = list(GROUPS["expr"] + GROUPS["state"] + GROUPS["shape"] + GROUPS["local"])
    out = [("floor", fl), ("ceil", ce)]
    for lam in [float(v) for v in args.lams.split(",")]:
        lr = np.log(tgt + e) - np.log(X.mean(0) + e)
        Xl = np.maximum(np.exp(np.log(X + e) + lam * lr) - e, 0.0)
        out.append((f"lam{lam}", row(Xl, C)))

    print(f"\n{'variant':10s}" + "".join(f"{x[:9]:>10s}" for x in keys) + f"{'pb_rel':>9s}")
    tab = {}
    for nm, r in out:
        sk = {x: 100.0 * skill(r[x], fl[x], ce[x], lower_is_better=LOWER[x]) for x in keys}
        tab[nm] = (r, sk)
        print(f"{nm:10s}" + "".join(f"{sk[x]:10.1f}" for x in keys) + f"{r['pb_rel_err']:9.4f}")
    print(f"\n{'variant':10s}{'expr':>8s}{'state':>8s}{'shape':>8s}{'local':>8s}{'TOTAL':>9s}")
    for nm, (r, sk) in tab.items():
        g = {gn: float(np.mean([sk[x] for x in ks])) for gn, ks in GROUPS.items()}
        print(f"{nm:10s}{g['expr']:8.2f}{g['state']:8.2f}{g['shape']:8.2f}{g['local']:8.2f}"
              f"{float(np.mean(list(g.values()))):9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
