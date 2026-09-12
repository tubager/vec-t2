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
        "--mix-xyz",
        choices=["paired", "left", "right"],
        default="paired",
        help="mix-expr: keep each cell's own xyz (paired) or place everyone on the left/right cloud",
    )
    parser.add_argument(
        "--geom-src",
        choices=["left", "right"],
        default=None,
        help="full/interp: which anchor supplies xyz (default: nearest side, ties left). Expression stays on the nearest side.",
    )
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="full: skip residual Δ (extrap: composition + cluster place + scale only)",
    )
    parser.add_argument(
        "--delta-alpha",
        type=float,
        default=None,
        help="override expression residual step α (extrap: prefer 0.3–0.5; α=1.0 crashed "
        "online to 42.5). Separate from growth.heart_extrap_alpha_* (RMS only). "
        "Also configurable as shift.extrap_delta_alpha.",
    )
    parser.add_argument(
        "--ot-interp",
        action="store_true",
        help="full/interp: per-cluster Hungarian pairing then place cells (no extra Δ)",
    )
    parser.add_argument(
        "--ot-x",
        choices=["lerp", "pick", "slice"],
        default="lerp",
        help="ot-interp: lerp paired genes, pick a real left/right cell (Bernoulli w), "
        "or slice (pick only from cells near clock progress within each type)",
    )
    parser.add_argument(
        "--slice-keep",
        type=float,
        default=0.4,
        help="ot-x slice: fraction of each side's cells to keep (default 0.4)",
    )
    parser.add_argument(
        "--slice-mode",
        choices=["band", "tail"],
        default="band",
        help="ot-x slice: band = closest to clock w; tail = late left + early right",
    )
    parser.add_argument(
        "--slice-stratify",
        choices=["none", "spatial", "fill"],
        default="none",
        help="ot-x slice: none = global expression band; spatial = band inside voxels; "
        "fill = global band plus one best-progress cell per missed occupied voxel",
    )
    parser.add_argument(
        "--slice-bins",
        type=int,
        default=8,
        help="ot-x slice --slice-stratify spatial|fill: bins per canonical axis (default 8)",
    )
    parser.add_argument(
        "--slice-fill-frac",
        type=float,
        default=0.08,
        help="ot-x slice --slice-stratify fill: max extra cells per cluster as a fraction (default 0.08)",
    )
    parser.add_argument(
        "--ot-xyz",
        choices=["lerp", "left", "right", "morph", "occ", "own"],
        default="lerp",
        help="ot-interp: lerp/left/right/morph/own (native pairing), or occ",
    )
    parser.add_argument(
        "--ot-w",
        type=float,
        default=None,
        help="ot-interp pick: fraction of right-anchor cells (default: time weight). Does not change xyz lerp weight.",
    )
    parser.add_argument(
        "--ot-n-pair",
        type=int,
        default=None,
        help="ot-interp: Hungarian pairs per cluster (default: shape.ot_n_pair/tps_n=800). "
        "Raise above the largest per-cluster allocation to stop reusing the same real cell.",
    )
    parser.add_argument(
        "--ot-unique",
        action="store_true",
        help="ot-interp: exhaust the paired templates before reusing one (no duplicate cells)",
    )
    parser.add_argument(
        "--geom-w",
        type=float,
        default=None,
        help="anisotropic interp: override the axis-ratio weight (clock weight by default). "
        "Online TSR puts heart E8.5 at ~0.67 of the E8.25->E8.75 log-RMS path.",
    )
    parser.add_argument(
        "--tps-w",
        type=float,
        default=None,
        help="TPS morph weight toward the right anchor (default: clock weight). "
        "Embryo E7.5 clock is 1/3; try 0.5 if left-like occupancy is too weak.",
    )
    parser.add_argument(
        "--use-flow",
        action="store_true",
        help="full: Euler-integrate panel OT-CFM on left-anchor cells (replaces mix/Δ). "
        "Combine with --ot-interp to keep OT-place xyz; ot-w is forced to 0.",
    )
    parser.add_argument(
        "--flow-dir",
        default=None,
        help="directory with pca.joblib + ckpt/flow.pt (default: outputs/t2/<setting>/)",
    )
    parser.add_argument(
        "--w2",
        action="store_true",
        help="heart --target 8.75: honest leave-out (anchors E8.25+E9.5, no E8.75 leak)",
    )
    parser.add_argument("--jitter", type=float, default=None, help="override jitter_frac (relative to median NN)")
    parser.add_argument(
        "--dens-keep",
        type=float,
        default=None,
        help="keep this fraction of highest kNN-density points then OT-assign onto the core (1.0 = no-op)",
    )
    parser.add_argument(
        "--crop-to-rms",
        action="store_true",
        help="FOV-crop leftmost cloud then OT-assign cells onto it (keeps pick X)",
    )
    parser.add_argument(
        "--crop-rms",
        type=float,
        default=None,
        help="crop-to-rms target RMS (default: log-linear r_target). Heart interp: 255",
    )
    parser.add_argument(
        "--crop-src",
        type=float,
        default=None,
        help="FOV crop by SELECTION: keep only the geometry donor cells inside this raw RMS "
        "radius, then allocate from them (native spacing, no OT re-placement). Heart interp: 255",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--crop-keep",
        type=float,
        default=None,
        help="FOV crop by REJECTION after allocation: over-allocate, then keep only the cells "
        "already inside this RMS radius (own X and xyz kept, native spacing). Heart interp: 255",
    )
    parser.add_argument(
        "--over-alloc",
        type=float,
        default=2.0,
        help="with --crop-keep: allocate n x this many cells before rejecting the periphery",
    )
    parser.add_argument(
        "--comp-interp",
        choices=["lin", "geo", "hell", "blend"],
        default=None,
        help="how the two anchor cluster proportions are interpolated along the simplex: "
             "lin = arithmetic (default, historical), geo = log-linear/exponential, "
             "hell = Hellinger chord, blend = renormalize(lin * geo**geo_damp) hedge. "
             "Only the real-truth heart window (E8.25+E9.5 -> E8.75) "
             "separates them: geo beats lin on mmd_u/variogram/de_direction at a matched pick "
             "weight, while both embryo windows prefer lin -- so this is a per-setting knob.",
    )
    parser.add_argument(
        "--geo-damp",
        type=float,
        default=None,
        help="damping exponent in [0,1] for --comp-interp blend (default 0.5 there, 1.0 elsewhere)",
    )
    parser.add_argument(
        "--comp-trend",
        type=float,
        default=None,
        help="damped continuation (0..1] of the last composition slope for extrapolation targets "
        "(default: freeze at last fitted stage; freeze beat all trends on the E8.75->E9.5 backtest)",
    )
    args = parser.parse_args()

    cfg = load_config()
    ensure_out_dirs(cfg)
    shape_mode = args.shape or (cfg.get("shape") or {}).get("mode") or "isotropic"
    if str(shape_mode).lower() == "tps" and not (cfg.get("shape") or {}).get("allow_tps"):
        if not args.ot_interp:
            raise SystemExit(
                "TPS is gated off (anisotropic already won local SDD/ODS). "
                "Use --ot-interp --shape tps after a local proxy gate, "
                "or set shape.allow_tps: true in configs/t2.yaml."
            )
    n = args.n or int(cfg["n_submit"])
    seed = cfg["seed"] if args.seed is None else args.seed
    if args.w2 and (args.setting != "heart" or abs(float(args.target) - 8.75) > 1e-9):
        raise SystemExit("--w2 requires --setting heart --target 8.75")
    spec = target_spec(args.setting, args.target, cfg, w2=bool(args.w2))
    panel = load_panel_for(spec, cfg)
    stages = load_stages(args.setting, cfg, panel)

    composition = None
    comp_path = pred_dir(cfg, args.setting) / "composition.json"
    if args.method == "full" and comp_path.exists():
        composition = CompositionT2.load(comp_path)
    if args.comp_interp is not None:
        if args.method != "full":
            raise SystemExit("--comp-interp only applies with --method full")
        cfg.setdefault("composition", {})["comp_interp"] = str(args.comp_interp)
        if args.geo_damp is not None:
            if not (0.0 <= float(args.geo_damp) <= 1.0):
                raise SystemExit(f"--geo-damp must be in [0,1], got {args.geo_damp}")
            cfg["composition"]["geo_damp"] = float(args.geo_damp)
        elif str(args.comp_interp) == "blend":
            cfg["composition"]["geo_damp"] = 0.5
        if composition is not None:
            composition.comp_interp = str(args.comp_interp)
            if cfg["composition"].get("geo_damp") is not None:
                composition.geo_damp = float(cfg["composition"]["geo_damp"])
    if args.comp_trend is not None:
        if args.method != "full":
            raise SystemExit("--comp-trend only applies with --method full")
        if not (0.0 < args.comp_trend <= 1.0):
            raise SystemExit(f"--comp-trend must be in (0, 1], got {args.comp_trend}")
        cfg.setdefault("composition", {})["extrap_comp_trend"] = float(args.comp_trend)
        if composition is not None:
            composition.extrap_trend = float(args.comp_trend)

    use_flow = bool(args.use_flow or (cfg.get("flow") or {}).get("use"))
    if args.ot_interp and use_flow:
        if args.ot_w is not None and abs(float(args.ot_w)) > 1e-12:
            raise SystemExit(
                "--use-flow with --ot-interp forces ot-w=0 (left-only X); do not pass --ot-w"
            )
        print("ot-interp + flow: X from left-anchor cells, then CFM; xyz from OT-place")
    if args.ot_x != "lerp" and not args.ot_interp:
        raise SystemExit("--ot-x only applies with --ot-interp")
    if args.ot_w is not None and not args.ot_interp:
        raise SystemExit("--ot-w only applies with --ot-interp")
    if args.ot_x == "slice":
        if not args.ot_interp:
            raise SystemExit("--ot-x slice requires --ot-interp")
        if not (0.0 < float(args.slice_keep) <= 1.0):
            raise SystemExit(f"--slice-keep must be in (0, 1], got {args.slice_keep}")
        if args.slice_stratify != "none" and int(args.slice_bins) < 2:
            raise SystemExit(f"--slice-bins must be >= 2, got {args.slice_bins}")
    elif args.slice_stratify != "none":
        raise SystemExit("--slice-stratify only applies with --ot-x slice")
    if args.crop_to_rms and args.dens_keep is not None:
        raise SystemExit("--crop-to-rms and --dens-keep cannot be combined")
    if args.crop_to_rms and str(shape_mode).lower() == "tps":
        raise SystemExit("--crop-to-rms cannot combine with --shape tps")
    if args.crop_to_rms and args.ot_xyz == "occ":
        raise SystemExit("--crop-to-rms cannot combine with --ot-xyz occ")
    if args.crop_rms is not None and not args.crop_to_rms:
        raise SystemExit("--crop-rms only applies with --crop-to-rms")
    if args.ot_unique and not args.ot_interp:
        raise SystemExit("--ot-unique only applies with --ot-interp")
    if args.ot_n_pair is not None and not args.ot_interp:
        raise SystemExit("--ot-n-pair only applies with --ot-interp")
    if args.geom_w is not None and (args.geom_w < 0.0 or args.geom_w > 1.0):
        raise SystemExit(f"--geom-w must be in [0, 1], got {args.geom_w}")
    if args.tps_w is not None and (args.tps_w < 0.0 or args.tps_w > 1.0):
        raise SystemExit(f"--tps-w must be in [0, 1], got {args.tps_w}")
    if args.tps_w is not None and str(shape_mode).lower() != "tps":
        raise SystemExit("--tps-w only applies with --shape tps")
    if args.delta_alpha is not None and args.no_delta:
        raise SystemExit("--delta-alpha and --no-delta cannot be combined")
    if args.delta_alpha is not None and args.delta_alpha < 0.0:
        raise SystemExit(f"--delta-alpha must be >= 0, got {args.delta_alpha}")
    pca = None
    flow_model = None
    trained = None
    flow_dir = Path(args.flow_dir) if args.flow_dir else pred_dir(cfg, args.setting)
    if use_flow:
        if args.method != "full":
            raise SystemExit("--use-flow / flow.use only applies to --method full")
        pca_path, ckpt_path = flow_artifact_paths(flow_dir)
        if not pca_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"Need {pca_path} and {ckpt_path}; run "
                f"python scripts/15_train_flow_t2.py --setting {args.setting}"
            )
        pca = ExpressionPCA.load(pca_path)
        flow_model, extra = load_velocity(ckpt_path)
        trained = set(extra.get("trained_clusters") or [])
        print(f"loaded T2 OT-CFM  clusters={sorted(trained)}")
    elif args.ot_interp:
        if args.method != "full":
            raise SystemExit("--ot-interp only applies to --method full")
        pca_path, _ = flow_artifact_paths(flow_dir)
        if pca_path.exists():
            pca = ExpressionPCA.load(pca_path)
            print(f"loaded panel PCA for OT pairing  {pca_path}")
        else:
            print("no pca.joblib; OT pairing uses raw expression")

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
        mix_xyz=args.mix_xyz,
        geom_src=args.geom_src,
        ot_interp=bool(args.ot_interp),
        ot_xyz=args.ot_xyz,
        ot_x=args.ot_x,
        ot_w=args.ot_w,
        no_delta=bool(args.no_delta),
        composition=composition,
        use_flow=use_flow,
        pca=pca,
        flow_model=flow_model,
        trained_clusters=trained,
        jitter_frac=args.jitter,
        dens_keep=args.dens_keep,
        crop_to_rms=bool(args.crop_to_rms),
        crop_rms=args.crop_rms,
        ot_n_pair=args.ot_n_pair,
        ot_unique=bool(args.ot_unique),
        geom_w=args.geom_w,
        crop_src=args.crop_src,
        crop_keep=args.crop_keep,
        over_alloc=args.over_alloc,
        delta_alpha=args.delta_alpha,
        tps_w=args.tps_w,
        slice_keep=float(args.slice_keep),
        slice_mode=str(args.slice_mode),
        slice_stratify=str(args.slice_stratify),
        slice_bins=int(args.slice_bins),
        slice_fill_frac=float(args.slice_fill_frac),
    )

    tag = args.method
    if args.method == "full":
        tag = f"full_{(args.shape or cfg.get('shape', {}).get('mode') or 'isotropic')}"
        if use_flow:
            tag = f"{tag}_flow"
        if args.ot_interp:
            tag = f"{tag}_otinterp"
            if args.ot_x != "lerp":
                tag = f"{tag}_x{args.ot_x}"
                if args.ot_x == "slice":
                    tag = f"{tag}_{args.slice_mode}k{args.slice_keep:g}"
                    if args.slice_stratify != "none":
                        tag = f"{tag}_{args.slice_stratify}{args.slice_bins}"
                        if args.slice_stratify == "fill":
                            tag = f"{tag}f{args.slice_fill_frac:g}"
            if args.ot_xyz != "lerp":
                tag = f"{tag}_xyz{args.ot_xyz}"
            if args.ot_w is not None:
                tag = f"{tag}_w{args.ot_w:g}"
            if args.ot_unique:
                tag = f"{tag}_uniq"
            if args.ot_n_pair is not None:
                tag = f"{tag}_np{args.ot_n_pair}"
        if args.mix_expr:
            tag = f"{tag}_mixexpr"
            if args.mix_xyz != "paired":
                tag = f"{tag}_xyz{args.mix_xyz}"
        if args.geom_src:
            tag = f"{tag}_geom{args.geom_src}"
        if args.geom_w is not None:
            tag = f"{tag}_gw{args.geom_w:g}"
        if args.tps_w is not None:
            tag = f"{tag}_tpsw{args.tps_w:g}"
        if args.no_delta:
            tag = f"{tag}_nodelta"
        elif args.delta_alpha is not None:
            tag = f"{tag}_da{args.delta_alpha:g}"
        if args.comp_interp is not None:
            tag = f"{tag}_ci{args.comp_interp}"
        if args.comp_trend is not None:
            tag = f"{tag}_ct{args.comp_trend:g}"
        if args.jitter is not None:
            tag = f"{tag}_j{args.jitter:g}"
    if args.dens_keep is not None:
        tag = f"{tag}_dk{args.dens_keep:g}"
    if args.crop_to_rms:
        tag = f"{tag}_crop"
        if args.crop_rms is not None:
            tag = f"{tag}{args.crop_rms:g}"
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
