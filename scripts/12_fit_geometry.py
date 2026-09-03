#!/usr/bin/env python
"""Fit T2 geometry (RMS, axes) and working-cluster composition; write outputs/t2/{setting}/."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.expression import CompositionT2  # noqa: E402
from t2.growth import r_interp, save_geometry, target_rms  # noqa: E402
from t2.infer import axis_map, load_panel_for, load_stages, rms_map, stage_times, target_spec  # noqa: E402
from t2.io import mean_X, spatial_xyz  # noqa: E402
from t2.paths import ensure_out_dirs, load_config, pred_dir  # noqa: E402


def _fit_setting(setting: str, cfg: dict) -> None:
    times = stage_times(setting, cfg)
    # Use the interpolation panel for fitting (embryo 498 / heart 500). Heart
    # interp and extrap panels are identical.
    dummy_t = 7.5 if setting == "embryo" else 8.5
    spec = target_spec(setting, dummy_t, cfg)
    panel = load_panel_for(spec, cfg)
    stages = load_stages(setting, cfg, panel)

    rms_by_t = rms_map(stages, times)
    std_by_t = axis_map(stages, times)
    geom = {
        "setting": setting,
        "stages": {},
        "r_targets": {},
        "axis_targets": {},
    }
    for name, adata in stages.items():
        t = times[name]
        xyz = spatial_xyz(adata)
        geom["stages"][name] = {
            "t": t,
            "n": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "rms": float(rms_by_t[t]),
            "axis_std": [float(x) for x in std_by_t[t]],
            "centroid": [float(x) for x in np.asarray(xyz).mean(axis=0)],
        }

    if setting == "embryo":
        targets = [7.25, 7.5, 7.75]
    else:
        targets = [8.5, 8.75, 9.5, 10.5, 12.5]
    for t in targets:
        try:
            r = target_rms(setting, t, rms_by_t, cfg)
        except Exception:
            if setting == "embryo":
                t0, t1 = cfg["growth"]["embryo_interp_anchors"]
                r = r_interp(t0, t1, t, rms_by_t[t0], rms_by_t[t1])
            else:
                r = float("nan")
        geom["r_targets"][str(t)] = float(r)
        print(f"  r({setting}, t={t}) = {r:.2f}")

    out_dir = pred_dir(cfg, setting)
    save_geometry(geom, out_dir / "geometry.json")

    comp = CompositionT2(setting, cfg["time"][setting], eps=float(cfg["composition"]["eps"]))
    comp.fit(
        {name: stages[name].obs["celltype"] for name in stages},
        {name: times[name] for name in stages},
    )
    comp.save(out_dir / "composition.json")
    for t in targets:
        pi = comp.probs(t)
        top = sorted(((k, v) for k, v in pi.items() if v > 1e-4), key=lambda kv: -kv[1])[:8]
        print(f"  π({setting}, t={t}): " + ", ".join(f"{k}={v:.3f}" for k, v in top))

    # Residual Δ on panel (for shift).
    if setting == "embryo":
        delta = mean_X(stages["E8.0"]) - mean_X(stages["E7.25"])
        np.save(out_dir / "delta_725_to_80.npy", delta)
        delta_proxy = mean_X(stages["E8.0"]) - mean_X(stages["E6.75"])
        np.save(out_dir / "delta_675_to_80.npy", delta_proxy)
    else:
        delta_i = mean_X(stages["E8.75"]) - mean_X(stages["E8.25"])
        delta_e = mean_X(stages["E9.5"]) - mean_X(stages["E8.75"])
        np.save(out_dir / "delta_825_to_875.npy", delta_i)
        np.save(out_dir / "delta_875_to_95.npy", delta_e)

    print(f"wrote {out_dir / 'geometry.json'}")
    print(f"wrote {out_dir / 'composition.json'}")


def main() -> int:
    cfg = load_config()
    ensure_out_dirs(cfg)
    for setting in ("embryo", "heart"):
        print(f"=== {setting} ===")
        _fit_setting(setting, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
