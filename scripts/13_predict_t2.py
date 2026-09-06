#!/usr/bin/env python
"""Generate a T2 prediction .h5ad for embryo or heart at a target time."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.expression import CompositionT2  # noqa: E402
from t2.flow import flow_artifact_paths, load_velocity  # noqa: E402
from t2.geometry import rms_radius  # noqa: E402
from t2.infer import (  # noqa: E402
    generate,
    load_panel_for,
    load_stages,
    target_spec,
    write_pred,
)
from t2.paths import ensure_out_dirs, load_config, pred_dir  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["embryo", "heart"], required=True)
    parser.add_argument("--target", type=float, required=True, help="7.5 | 7.75 | 8.5 | 10.5 | 12.5 | proxies 7.25/9.5/8.75")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument(
        "--method",
        choices=["copy_last", "scale_copy", "shift_scale", "full"],
        default="full",
    )
    parser.add_argument("--shape", choices=["isotropic", "anisotropic", "tps"], default=None)
    parser.add_argument("--mix-anchors", action="store_true")
    parser.add_argument(
        "--mix-expr",
        action="store_true",
        help="full/interp: mix left+right expression libraries by time weight; apply +wΔ / +(w-1)Δ per source",
    )
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="full: skip residual Δ (extrap: composition + cluster place + scale only)",
    )
    parser.add_argument("--use-flow", action="store_true", help="full method only: panel OT-CFM then residual shift")
    parser.add_argument("--jitter", type=float, default=None, help="override jitter_frac (relative to median NN)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    ensure_out_dirs(cfg)
    shape_mode = args.shape or (cfg.get("shape") or {}).get("mode") or "isotropic"
    if str(shape_mode).lower() == "tps" and not (cfg.get("shape") or {}).get("allow_tps"):
        raise SystemExit(
            "TPS is gated off (anisotropic already won local SDD/ODS). "
            "Set shape.allow_tps: true in configs/t2.yaml only after a new local gate."
        )
    n = args.n or int(cfg["n_submit"])
    seed = cfg["seed"] if args.seed is None else args.seed
    spec = target_spec(args.setting, args.target, cfg)
    panel = load_panel_for(spec, cfg)
    stages = load_stages(args.setting, cfg, panel)

    composition = None
    comp_path = pred_dir(cfg, args.setting) / "composition.json"
    if args.method == "full" and comp_path.exists():
        composition = CompositionT2.load(comp_path)

    use_flow = bool(args.use_flow or (cfg.get("flow") or {}).get("use"))
    pca = None
    flow_model = None
    trained = None
    if use_flow:
        if args.method != "full":
            raise SystemExit("--use-flow / flow.use only applies to --method full")
        pca_path, ckpt_path = flow_artifact_paths(pred_dir(cfg, args.setting))
        if not pca_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"Need {pca_path} and {ckpt_path}; run "
                f"python scripts/15_train_flow_t2.py --setting {args.setting}"
            )
        pca = ExpressionPCA.load(pca_path)
        flow_model, extra = load_velocity(ckpt_path)
        trained = set(extra.get("trained_clusters") or [])
        print(f"loaded T2 OT-CFM  clusters={sorted(trained)}")

    adata = generate(
        spec,
        stages,
        panel,
        cfg,
        n=n,
        seed=seed,
        method=args.method,
        shape_mode=args.shape,
        mix_anchors=True if args.mix_anchors else None,
        mix_expr=True if args.mix_expr else None,
        no_delta=bool(args.no_delta),
        composition=composition,
        use_flow=use_flow,
        pca=pca,
        flow_model=flow_model,
        trained_clusters=trained,
        jitter_frac=args.jitter,
    )

    tag = args.method
    if args.method == "full":
        tag = f"full_{(args.shape or cfg.get('shape', {}).get('mode') or 'isotropic')}"
        if use_flow:
            tag = f"{tag}_flow"
        if args.mix_expr:
            tag = f"{tag}_mixexpr"
        if args.no_delta:
            tag = f"{tag}_nodelta"
        if args.jitter is not None:
            tag = f"{tag}_j{args.jitter:g}"
    default_name = f"pred_E{args.target:g}_{tag}.h5ad"
    out = Path(args.out) if args.out else (pred_dir(cfg, args.setting) / default_name)
    write_pred(adata, out, spec, clip_min=float(cfg["shift"]["clip_min"]))
    r = rms_radius(adata.obsm["spatial_3D"])
    print(
        f"wrote {out}  n={adata.n_obs} vars={adata.n_vars}  "
        f"rms={r:.1f}  X.min={float(np.asarray(adata.X.min())):.4f}"
    )
    if spec.proxy:
        print("This is a local proxy target, not an official leaderboard file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
