#!/usr/bin/env python
"""Generate a T1 prediction .h5ad for E9.5 (local proxy), E10.5, or E12.5."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1.composition import CompositionModel  # noqa: E402
from t1.infer import build_libraries, generate  # noqa: E402
from t1.io import align_to_panel, assert_t1_submission, load_adata, load_panel, mean_X, write_submission  # noqa: E402
from t1.paths import ROOT as PROJ, ensure_out_dirs, load_config, resolve  # noqa: E402
from t1.pca import ExpressionPCA  # noqa: E402


def _panel(cfg, adata):
    panel = load_panel(resolve(cfg, "panel")) or load_panel(resolve(cfg, "panel_fallback"))
    return panel or list(adata.var_names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, required=True, help="9.5, 10.5, or 12.5")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--use-flow", action="store_true")
    parser.add_argument("--source", choices=["8.5", "9.5"], default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    ensure_out_dirs(cfg)
    n = args.n or int(cfg["n_submit"])
    seed = cfg["seed"] if args.seed is None else args.seed
    target = float(args.target)

    if args.source:
        source = args.source
    else:
        source = "8.5" if target <= 9.5 else "9.5"

    a85 = load_adata(PROJ / cfg["paths"]["adata_85"])
    a95 = load_adata(PROJ / cfg["paths"]["adata_95"])
    panel = _panel(cfg, a85)
    a85 = align_to_panel(a85, panel)
    a95 = align_to_panel(a95, panel)

    pca = ExpressionPCA.load(resolve(cfg, "pca"))
    composition = CompositionModel.load(resolve(cfg, "composition"))

    z85_path = PROJ / "outputs/t1/z_85.npy"
    z95_path = PROJ / "outputs/t1/z_95.npy"
    if z85_path.exists() and z95_path.exists():
        z85 = np.load(z85_path)
        z95 = np.load(z95_path)
    else:
        print("encoding stages (z_*.npy missing) ...")
        z85 = pca.encode_adata(a85)
        z95 = pca.encode_adata(a95)

    libraries = build_libraries(z85, z95, a85.obs["celltype"], a95.obs["celltype"])
    delta_path = PROJ / "outputs/t1/delta.npy"
    if delta_path.exists():
        delta = np.load(delta_path)
    else:
        delta = mean_X(a95) - mean_X(a85)

    model = None
    if args.use_flow:
        from t1.flow import load_velocity

        ckpt = resolve(cfg, "ckpt")
        if not ckpt.exists():
            raise FileNotFoundError(f"Need {ckpt}; run scripts/02_train_flow.py first")
        model, _ = load_velocity(ckpt)

    tag = "flow" if args.use_flow else "comp"
    default_name = f"pred_E{target:g}_{tag}_from{source}.h5ad"
    out = Path(args.out) if args.out else (resolve(cfg, "preds") / default_name)

    adata = generate(
        target_t=target,
        n=n,
        cfg=cfg,
        composition=composition,
        pca=pca,
        libraries=libraries,
        delta=delta,
        var_names=panel,
        source=source,
        use_flow=args.use_flow,
        model=model,
        seed=seed,
    )
    write_submission(adata.X, panel, out, clip_min=float(cfg["shift"]["clip_min"]))
    saved = load_adata(out)
    assert_t1_submission(saved, cfg)
    print(f"wrote {out}  n={saved.n_obs} vars={saved.n_vars} min={float(np.asarray(saved.X.min())):.4f}")
    if target == 9.5:
        print("score with:")
        print(f"  python scripts/04_score_local.py --pred {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
