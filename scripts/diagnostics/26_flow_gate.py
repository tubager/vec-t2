#!/usr/bin/env python
"""Honest flow gate: mix (OT-place pick) vs left-only CFM, scored on a held-out midpoint.

Embryo (priority, analog of the 66.41 board):
  train:  15_train_flow_t2.py --setting embryo --hop E6.75,E8.0 --exclude E7.25 \\
          --out-dir outputs/t2/embryo/gate725 --epochs 80
  mix:    13_predict_t2.py --setting embryo --target 7.25 --method full --shape tps \\
          --ot-interp --ot-x pick --ot-xyz left --n 5000 --ot-n-pair 6000 --ot-unique \\
          --flow-dir outputs/t2/embryo/gate725 --out .../mix.h5ad
  flow:   same + --use-flow --out .../flow.h5ad
  score:  26_flow_gate.py --window embryo --mix .../mix.h5ad --flow .../flow.h5ad

Heart W2 (E8.25+E9.5 → true E8.75; labels poorly aligned, stress test):
  train:  --setting heart --hop E8.25,E9.5 --exclude E8.75 --out-dir outputs/t2/heart/gate_w2
  mix/flow: --setting heart --target 8.75 --w2 --shape anisotropic --ot-interp --ot-x pick \\
            --ot-xyz left --n 5000 --flow-dir outputs/t2/heart/gate_w2

Gate: flow must beat mix on NFS and mmd_u, and de_score must not drop. Otherwise do not submit.
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

from T2.metrics import neighborhood_mmd  # noqa: E402
from common.core_metrics import de_direction, de_score, mmd_unbiased, variogram_score  # noqa: E402
from common.shape_metrics import d2_distance, occupancy_dice  # noqa: E402
from t2.io import to_dense  # noqa: E402

WINDOWS = {
    "embryo": {
        "target": ROOT / "data" / "E7.25.h5ad",
        "reference": ROOT / "data" / "E6.75.h5ad",
        "note": "leave-out E7.25 from E6.75+E8.0",
    },
    "w2": {
        "target": ROOT / "data" / "E8.75.h5ad",
        "reference": ROOT / "data" / "E8.25_late.h5ad",
        "note": "leave-out E8.75 from E8.25+E9.5",
    },
}


def _xy(path: Path):
    a = ad.read_h5ad(path)
    X = to_dense(a.X).astype(np.float64)
    C = np.asarray(a.obsm["spatial_3D"], dtype=np.float64)
    genes = np.asarray(a.var_names)
    return X, C, genes


def _align(X: np.ndarray, genes: np.ndarray, panel: np.ndarray) -> np.ndarray:
    if list(genes) == list(panel):
        return X
    idx = {g: i for i, g in enumerate(genes)}
    miss = [g for g in panel if g not in idx]
    if miss:
        raise SystemExit(f"prediction missing {len(miss)} panel genes e.g. {miss[:3]}")
    return X[:, [idx[g] for g in panel]]


def _row(name: str, X, C, XT, CT, XR) -> dict:
    de = de_score(X, XT, XR)
    score = float(de["score"] if isinstance(de, dict) else de)
    return {
        "name": name,
        "nfs": float(neighborhood_mmd(X, C, XT, CT)),
        "mmd": float(mmd_unbiased(X, XT, seed=0)),
        "var": float(variogram_score(X, XT, seed=0)),
        "de": score,
        "ddir": float(de_direction(X, XT, XR)),
        "ods": float(occupancy_dice(C, CT, seed=0)[0]),
        "d2": float(d2_distance(C, CT, seed=0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=sorted(WINDOWS), required=True)
    ap.add_argument("--mix", required=True, type=Path)
    ap.add_argument("--flow", required=True, type=Path)
    ap.add_argument("--copy", type=Path, default=None, help="optional copy_last / scale_copy file")
    args = ap.parse_args()
    spec = WINDOWS[args.window]
    XT, CT, panel = _xy(spec["target"])
    XR, _, pref = _xy(spec["reference"])
    XR = _align(XR, pref, panel)

    print(f"# {args.window}: {spec['note']}")
    print(f"{'variant':28s} {'NFS↓':>9s} {'mmd_u↓':>9s} {'vario↓':>9s} {'de_sc↑':>8s} {'de_dir↑':>8s}")
    rows = []
    files = [("mix", args.mix), ("flow", args.flow)]
    if args.copy:
        files.insert(0, ("copy", args.copy))
    for name, path in files:
        X, C, genes = _xy(path)
        X = _align(X, genes, panel)
        r = _row(name, X, C, XT, CT, XR)
        rows.append(r)
        print(
            f"{r['name']:28s} {r['nfs']:9.5f} {r['mmd']:9.5f} {r['var']:9.5f} "
            f"{r['de']:8.4f} {r['ddir']:8.4f}"
        )

    by = {r["name"]: r for r in rows}
    mix, flow = by["mix"], by["flow"]
    ok_nfs = flow["nfs"] < mix["nfs"]
    ok_mmd = flow["mmd"] < mix["mmd"]
    ok_de = flow["de"] >= mix["de"] - 0.02
    passed = ok_nfs and ok_mmd and ok_de
    print(
        f"\nGATE  NFS {flow['nfs']:.5f} vs mix {mix['nfs']:.5f}  {'PASS' if ok_nfs else 'FAIL'}"
        f"  |  mmd {flow['mmd']:.5f} vs {mix['mmd']:.5f}  {'PASS' if ok_mmd else 'FAIL'}"
        f"  |  de {flow['de']:.4f} vs {mix['de']:.4f}  {'PASS' if ok_de else 'FAIL'}"
    )
    print("RESULT", "PASS — eligible for a slot" if passed else "FAIL — do not submit this flow")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
