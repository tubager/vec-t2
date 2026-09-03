#!/usr/bin/env python
"""Write T2 copy_last / scale_copy / shift_scale baselines (n=3000, panel-aligned, +spatial_3D)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.infer import (  # noqa: E402
    generate,
    load_panel_for,
    load_stages,
    target_spec,
    write_pred,
)
from t2.paths import ensure_out_dirs, load_config, pred_dir  # noqa: E402


JOBS = [
    # Official leaderboards
    ("embryo", 7.5, "copy_last", "copy_last_E75.h5ad"),
    ("embryo", 7.5, "scale_copy", "scale_copy_E75.h5ad"),
    ("embryo", 7.5, "shift_scale", "shift_scale_E75.h5ad"),
    ("heart", 8.5, "copy_last", "copy_last_E85.h5ad"),
    ("heart", 8.5, "scale_copy", "scale_copy_E85.h5ad"),
    ("heart", 8.5, "shift_scale", "shift_scale_E85.h5ad"),
    ("heart", 10.5, "copy_last", "copy_last_E105.h5ad"),
    ("heart", 10.5, "scale_copy", "scale_copy_E105.h5ad"),
    ("heart", 10.5, "shift_scale", "shift_scale_E105.h5ad"),
    # Local proxies
    ("embryo", 7.25, "copy_last", "copy_last_proxy_E725.h5ad"),
    ("embryo", 7.25, "scale_copy", "scale_copy_proxy_E725.h5ad"),
    ("embryo", 7.25, "shift_scale", "shift_scale_proxy_E725.h5ad"),
    ("heart", 9.5, "copy_last", "copy_last_proxy_E95.h5ad"),
    ("heart", 9.5, "scale_copy", "scale_copy_proxy_E95.h5ad"),
    ("heart", 9.5, "shift_scale", "shift_scale_proxy_E95.h5ad"),
    ("heart", 8.75, "copy_last", "copy_last_growth_E875.h5ad"),
    ("heart", 8.75, "scale_copy", "scale_copy_growth_E875.h5ad"),
]


def main() -> int:
    cfg = load_config()
    ensure_out_dirs(cfg)
    n = int(cfg["n_submit"])
    seed = int(cfg["seed"])
    clip = float(cfg["shift"]["clip_min"])

    cache: dict[str, object] = {}

    def stages_for(setting: str, panel):
        if setting not in cache:
            cache[setting] = load_stages(setting, cfg, panel)
        return cache[setting]

    for setting, t, method, name in JOBS:
        spec = target_spec(setting, t, cfg)
        panel = load_panel_for(spec, cfg)
        stages = stages_for(setting, panel)
        adata = generate(spec, stages, panel, cfg, n=n, seed=seed, method=method)
        out = pred_dir(cfg, setting) / name
        write_pred(adata, out, spec, clip_min=clip)
        xyz = adata.obsm["spatial_3D"]
        from t2.geometry import rms_radius

        print(
            f"wrote {out}  n={adata.n_obs} vars={adata.n_vars}  "
            f"rms={rms_radius(xyz):.1f}  method={method}  t={t}"
        )

    print("\nLocal proxy scoring:")
    print("  python scripts/14_score_local_t2.py --proxy embryo_interp")
    print("  python scripts/14_score_local_t2.py --proxy heart_extrap")
    print("  python scripts/14_score_local_t2.py --proxy heart_growth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
