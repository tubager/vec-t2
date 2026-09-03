#!/usr/bin/env python
"""Local T2 veckit / geometry-gate scoring. Not official validation scores."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.geometry import rms_radius  # noqa: E402
from t2.io import load_adata, spatial_xyz, write_t2  # noqa: E402
from t2.neighborhood import shuffle_xyz  # noqa: E402
from t2.paths import ROOT as PROJ, load_config, pred_dir, resolve  # noqa: E402


PROXIES = {
    "embryo_interp": {
        "setting": "embryo",
        "pred_default": "scale_copy_proxy_E725.h5ad",
        "target": "embryo/E7.25",
        "reference": "embryo/E6.75",
        "note": "Leave-out E7.25; w=(7.25-6.75)/(8.0-6.75)=0.4. Floor: subsample E6.75.",
    },
    "heart_extrap": {
        "setting": "heart",
        "pred_default": "scale_copy_proxy_E95.h5ad",
        "target": "heart/E9.5",
        "reference": "heart/E8.75",
        "note": "One-step E8.75→E9.5. Labels almost relabelled; MMD/NFS will look bad.",
    },
    "heart_growth": {
        "setting": "heart",
        "pred_default": "scale_copy_growth_E875.h5ad",
        "target": "heart/E8.75",
        "reference": "heart/E8.25",
        "note": "Growth diagnostic only (types aligned, geometry shrinks). Do not treat as atrophy.",
    },
}


def _path(cfg: dict, dotted: str) -> Path:
    setting, key = dotted.split("/")
    return resolve(cfg, setting, key)


def _score(task_setting: str, pred: Path, target: Path, reference: Path):
    try:
        from veckit import score
    except ImportError:
        print("veckit is not installed.  pip install veckit")
        print("CLI equivalent:")
        print(
            f"  veckit --task T2 --setting {task_setting} "
            f"--input {pred} --target {target} --reference {reference}"
        )
        return None
    print(f"input     {pred}")
    print(f"target    {target}")
    print(f"reference {reference}")
    result = score(
        task="T2",
        setting=task_setting,
        input=str(pred),
        target=str(target),
        reference=str(reference),
    )
    metrics = result.get("metrics", result) if isinstance(result, dict) else result
    print(json.dumps(metrics, indent=2, default=str))
    return metrics


def _heart_interp_gate(pred: Path) -> None:
    """No E8.5 truth locally. Check RMS lands between E8.25 (354) and E8.75 (217)."""
    adata = load_adata(pred)
    r = rms_radius(spatial_xyz(adata))
    lo, hi = 216.9, 354.1
    target = 277.0
    print(f"heart interp RMS gate: r_pred={r:.1f}  expected in ({lo}, {hi}), log-lin ~{target}")
    if not (lo < r < hi):
        print("  FAIL: RMS outside (216.9, 354.1) — interpolation must shrink E8.25, not grow it")
    else:
        print(f"  ok: |log(r/277)| = {abs(np.log(r / target)):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", choices=list(PROXIES) + ["heart_interp_gate"], default=None)
    parser.add_argument("--pred", default=None, help="prediction .h5ad")
    parser.add_argument("--setting", choices=["embryo", "heart"], default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--shuffle", action="store_true", help="also score xyz-shuffled copy (NFS diagnostic)")
    args = parser.parse_args()

    cfg = load_config()

    if args.proxy == "heart_interp_gate":
        pred = Path(args.pred) if args.pred else (pred_dir(cfg, "heart") / "scale_copy_E85.h5ad")
        if not pred.is_absolute():
            pred = (Path.cwd() / pred).resolve()
        _heart_interp_gate(pred)
        return 0

    if args.proxy:
        info = PROXIES[args.proxy]
        setting = info["setting"]
        pred = Path(args.pred) if args.pred else (pred_dir(cfg, setting) / info["pred_default"])
        target = _path(cfg, info["target"])
        reference = _path(cfg, info["reference"])
        print(info["note"])
    else:
        if not args.pred or not args.setting:
            parser.error("need --proxy or (--pred and --setting)")
        setting = args.setting
        pred = Path(args.pred)
        target = Path(args.target) if args.target else None
        reference = Path(args.reference) if args.reference else None
        if target is None or reference is None:
            parser.error("custom scoring needs --target and --reference")

    if not pred.is_absolute():
        pred = (Path.cwd() / pred).resolve()

    metrics = _score(setting, pred, target, reference)
    if args.shuffle and metrics is not None:
        adata = load_adata(pred)
        rng = np.random.default_rng(0)
        xyz = shuffle_xyz(adata.obsm["spatial_3D"], rng)
        with tempfile.TemporaryDirectory() as td:
            shuffled = Path(td) / "shuffled.h5ad"
            write_t2(adata.X, xyz, list(adata.var_names), shuffled)
            print("\n--- xyz shuffled (NFS should get worse) ---")
            _score(setting, shuffled, target, reference)
    return 0 if metrics is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
