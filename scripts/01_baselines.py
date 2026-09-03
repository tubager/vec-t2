#!/usr/bin/env python
"""Write T1 floor / shift / noise baselines under outputs/t1/preds/."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1.baselines import (  # noqa: E402
    copy_last,
    pseudobulk_shift,
    resample_types_then_noise,
    save_pred,
)
from t1.io import align_to_panel, load_adata, load_panel  # noqa: E402
from t1.paths import ROOT as PROJ, ensure_out_dirs, load_config, resolve  # noqa: E402


def _panel(cfg, adata) -> list[str]:
    panel = load_panel(resolve(cfg, "panel")) or load_panel(resolve(cfg, "panel_fallback"))
    return panel or list(adata.var_names)


def main() -> int:
    cfg = load_config()
    ensure_out_dirs(cfg)
    n = int(cfg["n_submit"])
    seed = int(cfg["seed"])
    clip = float(cfg["shift"]["clip_min"])
    pred_dir = resolve(cfg, "preds")

    a85 = load_adata(PROJ / cfg["paths"]["adata_85"])
    a95 = load_adata(PROJ / cfg["paths"]["adata_95"])
    panel = _panel(cfg, a85)
    a85 = align_to_panel(a85, panel)
    a95 = align_to_panel(a95, panel)

    jobs = {
        # Fair one-step (score against E9.5, reference E8.5)
        "copy_last_E95.h5ad": copy_last(a85, n=n, seed=seed),
        "resample_noise_E95.h5ad": resample_types_then_noise(a85, n=n, seed=seed, clip_min=clip),
        # Leaky diagnostic: E8.5 + true E9.5-E8.5 mean (uses E9.5 answers)
        "shift_on_E85_to_E95.h5ad": pseudobulk_shift(a85, a85, a95, n=n, alpha=1.0, seed=seed, clip_min=clip),
        # Extrapolation submissions (no local biological score for E10.5)
        "copy_E95.h5ad": copy_last(a95, n=n, seed=seed),
        "E95_plus_delta.h5ad": pseudobulk_shift(a95, a85, a95, n=n, alpha=1.0, seed=seed, clip_min=clip),
    }
    for name, adata in jobs.items():
        path = pred_dir / name
        save_pred(adata, path, clip_min=clip)
        print(f"wrote {path}  n={adata.n_obs} vars={adata.n_vars}")
    print("\nLocal one-step scoring (fair):")
    print("  python scripts/04_score_local.py --pred outputs/t1/preds/copy_last_E95.h5ad")
    print("P2 first official file:")
    print("  outputs/t1/preds/E95_plus_delta.h5ad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
