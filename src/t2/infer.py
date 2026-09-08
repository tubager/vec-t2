"""Generate n T2 cells for a setting + target time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np

from t1.baselines import add_delta
from t2.clusters import birth_clusters, birth_progenitor, labels_to_clusters
from t2.expression import (
    CompositionT2,
    apply_cluster_delta,
    cluster_mean_delta,
    combine_delta,
    copy_last,
    project_t1_delta,
    sample_paired_cells,
    scale_copy,
    shift_scale,
    transport_expression,
)
from t2.geometry import axis_stds, canonicalize, rms_radius, scale_cloud
from t2.growth import axis_interp, interp_weight, r_interp, target_rms
from t2.io import (
    assert_t2_submission,
    load_adata,
    load_panel,
    mean_X,
    reindex_adata,
    spatial_xyz,
    to_dense,
    write_t2,
)
from t2.neighborhood import add_jitter
from t2.ot import ot_interpolate_alloc
from t2.paths import ROOT, resolve
from t2.shape import (
    crop_place_xyz,
    dens_keep_cloud,
    match_axis_flips,
    nearest_side,
    occupancy_interpolate_cloud,
    ot_assign_xyz,
    ot_interpolate_clouds,
    radius_crop_indices,
    radius_crop_resample,
    transform_cloud,
)


@dataclass
class TargetSpec:
    setting: str
    t: float
    mode: str  # interp | extrap
    split: str | None
    panel_key: str
    n_genes: int
    n_min: int
    n_max: int
    left: str | None
    right: str | None
    src: str
    geom_src: str
    t_left: float | None
    t_right: float | None
    allowed_times: list[float] | None = None
    proxy: bool = False


def _lb(cfg: dict, key: str) -> dict:
    return cfg["leaderboards"][key]


def target_spec(setting: str, t: float, cfg: dict | None = None) -> TargetSpec:
    """Resolve an official or local-proxy target."""
    t = float(t)
    if setting == "embryo":
        te = cfg["time"]["embryo"] if cfg else {}
        if abs(t - 7.5) < 1e-9:
            lb = _lb(cfg, "embryo_val_interp")
            return TargetSpec(
                setting="embryo", t=7.5, mode="interp", split="val_interp",
                panel_key="panel_val_interp", n_genes=lb["n_genes"],
                n_min=lb["n_min"], n_max=lb["n_max"],
                left="E7.25", right="E8.0", src="E7.25", geom_src="E7.25",
                t_left=7.25, t_right=8.0, allowed_times=[7.25, 8.0],
            )
        if abs(t - 7.75) < 1e-9:
            lb = _lb(cfg, "embryo_val_interp")
            return TargetSpec(
                setting="embryo", t=7.75, mode="interp", split=None,
                panel_key="panel_val_interp", n_genes=lb["n_genes"],
                n_min=lb["n_min"], n_max=lb["n_max"],
                left="E7.25", right="E8.0", src="E7.25", geom_src="E8.0",
                t_left=7.25, t_right=8.0, allowed_times=[7.25, 8.0],
            )
        if abs(t - 7.25) < 1e-9:
            # Local proxy: leave out E7.25, interpolate from E6.75 + E8.0.
            lb = _lb(cfg, "embryo_val_interp")
            return TargetSpec(
                setting="embryo", t=7.25, mode="interp", split=None,
                panel_key="panel_val_interp", n_genes=lb["n_genes"],
                n_min=lb["n_min"], n_max=lb["n_max"],
                left="E6.75", right="E8.0", src="E6.75", geom_src="E6.75",
                t_left=6.75, t_right=8.0, allowed_times=[6.75, 8.0],
                proxy=True,
            )
    if setting == "heart":
        if abs(t - 8.5) < 1e-9:
            lb = _lb(cfg, "heart_val_interp")
            return TargetSpec(
                setting="heart", t=8.5, mode="interp", split="val_interp",
                panel_key="panel_val_interp", n_genes=lb["n_genes"],
                n_min=lb["n_min"], n_max=lb["n_max"],
                left="E8.25", right="E8.75", src="E8.25", geom_src="E8.25",
                t_left=8.25, t_right=8.75, allowed_times=[8.25, 8.75],
            )
        if abs(t - 10.5) < 1e-9:
            lb = _lb(cfg, "heart_val_extrap")
            return TargetSpec(
                setting="heart", t=10.5, mode="extrap", split="val_extrap",
                panel_key="panel_val_extrap", n_genes=lb["n_genes"],
                n_min=lb["n_min"], n_max=lb["n_max"],
                left="E8.75", right="E9.5", src="E9.5", geom_src="E9.5",
                t_left=8.75, t_right=9.5, allowed_times=[8.75, 9.5],
            )
        if abs(t - 12.5) < 1e-9:
            lb = _lb(cfg, "heart_val_extrap")
            return TargetSpec(
                setting="heart", t=12.5, mode="extrap", split=None,
                panel_key="panel_val_extrap", n_genes=lb["n_genes"],
                n_min=lb["n_min"], n_max=lb["n_max"],
                left="E8.75", right="E9.5", src="E9.5", geom_src="E9.5",
                t_left=8.75, t_right=9.5, allowed_times=[8.75, 9.5],
            )
        if abs(t - 9.5) < 1e-9:
            lb = _lb(cfg, "heart_val_extrap")
            return TargetSpec(
                setting="heart", t=9.5, mode="extrap", split=None,
                panel_key="panel_val_extrap", n_genes=lb["n_genes"],
                n_min=lb["n_min"], n_max=lb["n_max"],
                left="E8.25", right="E8.75", src="E8.75", geom_src="E8.75",
                t_left=8.25, t_right=8.75, allowed_times=[8.25, 8.75],
                proxy=True,
            )
        if abs(t - 8.75) < 1e-9:
            # Growth diagnostic: predict E8.75 from E8.25 (types aligned, geometry shrinks).
            lb = _lb(cfg, "heart_val_interp")
            return TargetSpec(
                setting="heart", t=8.75, mode="interp", split=None,
                panel_key="panel_val_interp", n_genes=lb["n_genes"],
                n_min=lb["n_min"], n_max=lb["n_max"],
                left="E8.25", right="E8.75", src="E8.25", geom_src="E8.25",
                t_left=8.25, t_right=8.75, allowed_times=[8.25],
                proxy=True,
            )
    raise ValueError(f"Unsupported target setting={setting} t={t}")


def stage_files(setting: str, cfg: dict) -> dict[str, Path]:
    keys = cfg["paths"][setting]
    out = {}
    for k, v in keys.items():
        if k.startswith("panel"):
            continue
        out[k] = ROOT / v
    return out


def stage_times(setting: str, cfg: dict) -> dict[str, float]:
    t = cfg["time"][setting]
    if setting == "embryo":
        return {"E6.75": t["t_675"], "E7.25": t["t_725"], "E8.0": t["t_80"]}
    return {"E8.25": t["t_825"], "E8.75": t["t_875"], "E9.5": t["t_95"]}


def load_panel_for(spec: TargetSpec, cfg: dict) -> list[str]:
    path = resolve(cfg, spec.setting, spec.panel_key)
    panel = load_panel(path)
    if panel is None:
        raise FileNotFoundError(f"Missing panel {path}")
    if len(panel) != spec.n_genes:
        raise ValueError(f"Panel {path} has {len(panel)} genes, expected {spec.n_genes}")
    return panel


def load_stages(
    setting: str,
    cfg: dict,
    panel: list[str],
    names: list[str] | None = None,
) -> dict[str, ad.AnnData]:
    files = stage_files(setting, cfg)
    names = names or list(files)
    out = {}
    for name in names:
        adata = load_adata(files[name])
        out[name] = reindex_adata(adata, panel)
    return out


def rms_map(stages: dict[str, ad.AnnData], times: dict[str, float]) -> dict[float, float]:
    out = {}
    for name, adata in stages.items():
        out[float(times[name])] = rms_radius(spatial_xyz(adata))
    return out


def axis_map(stages: dict[str, ad.AnnData], times: dict[str, float]) -> dict[float, np.ndarray]:
    out = {}
    for name, adata in stages.items():
        out[float(times[name])] = axis_stds(spatial_xyz(adata))
    return out


def _delta_for(spec: TargetSpec, stages: dict[str, ad.AnnData], panel: list[str], cfg: dict) -> tuple[np.ndarray, float]:
    if spec.mode == "interp":
        prev = stages[spec.left]
        nxt = stages[spec.right]
        w = interp_weight(spec.t_left, spec.t_right, spec.t)
        delta = mean_X(nxt) - mean_X(prev)
        return delta, w
    # extrap
    if spec.proxy and spec.t == 9.5:
        # Fair one-step: last observed MERFISH Δ (E8.75 − E8.25), α=1.
        delta = mean_X(stages["E8.75"]) - mean_X(stages["E8.25"])
        return delta, 1.0
    prev = stages.get("E8.75") or stages[spec.left]
    nxt = stages.get("E9.5") or stages[spec.src]
    delta = mean_X(nxt) - mean_X(prev)
    if abs(spec.t - 10.5) < 1e-9:
        alpha = 1.0
    elif abs(spec.t - 12.5) < 1e-9:
        alpha = 1.0  # P3 grid; do not guess a larger step
    else:
        alpha = 1.0
    w_t1 = float(cfg["shift"].get("t1_delta_weight") or 0.0)
    if w_t1 > 0:
        t1_delta = project_t1_delta(
            panel,
            ROOT / cfg["paths"]["t1_delta"],
            ROOT / cfg["paths"]["t1_genes"],
        )
        delta = combine_delta(delta, t1_delta, w_t1)
    return delta, alpha


def _r_target(spec: TargetSpec, rms_by_t: dict[float, float], cfg: dict) -> float:
    if spec.mode == "interp":
        return r_interp(spec.t_left, spec.t_right, spec.t, rms_by_t[spec.t_left], rms_by_t[spec.t_right])
    if spec.proxy and spec.t == 9.5:
        # Growth diagnostic uses true E9.5 RMS if measured; else freeze E8.75.
        if 9.5 in rms_by_t:
            return float(rms_by_t[9.5])
        return float(rms_by_t[spec.t_right] if spec.t_right in rms_by_t else list(rms_by_t.values())[-1])
    if spec.proxy and spec.t == 8.75:
        return float(rms_by_t[8.75]) if 8.75 in rms_by_t else r_interp(8.25, 8.75, 8.75, rms_by_t[8.25], rms_by_t.get(8.75, rms_by_t[8.25]))
    return target_rms(spec.setting, spec.t, rms_by_t, cfg)


def _flow_clock(spec: TargetSpec, times: dict[str, float]) -> tuple[float, float]:
    """(t_start, dt) for Euler integration. Interp from left; extrap from spec.src."""
    if spec.mode == "interp":
        t_start = float(spec.t_left)
        return t_start, float(spec.t) - t_start
    t_start = float(times[spec.src])
    return t_start, float(spec.t) - t_start


def generate(
    spec: TargetSpec,
    stages: dict[str, ad.AnnData],
    panel: list[str],
    cfg: dict,
    n: int,
    seed: int = 0,
    method: str = "full",
    shape_mode: str | None = None,
    mix_anchors: bool | None = None,
    mix_expr: bool | None = None,
    mix_xyz: str = "paired",
    geom_src: str | None = None,
    ot_interp: bool = False,
    ot_xyz: str = "lerp",
    ot_x: str = "lerp",
    ot_w: float | None = None,
    no_delta: bool = False,
    composition: CompositionT2 | None = None,
    use_flow: bool = False,
    pca=None,
    flow_model=None,
    trained_clusters: set[str] | None = None,
    jitter_frac: float | None = None,
    dens_keep: float | None = None,
    crop_to_rms: bool = False,
    crop_rms: float | None = None,
    ot_n_pair: int | None = None,
    ot_unique: bool = False,
    geom_w: float | None = None,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    morph_rng = np.random.default_rng(int(seed) + 2_000_007)
    times = stage_times(spec.setting, cfg)
    rms_by_t = rms_map(stages, times)
    r_star = _r_target(spec, rms_by_t, cfg)
    clip = float(cfg["shift"]["clip_min"])
    shape_mode = (shape_mode or cfg.get("shape", {}).get("mode") or "isotropic").lower()
    mix = cfg.get("shape", {}).get("mix_anchors", False) if mix_anchors is None else mix_anchors
    jitter = float(cfg["jitter_frac"] if jitter_frac is None else jitter_frac)
    dens_keep = 1.0 if dens_keep is None else float(dens_keep)
    if crop_to_rms and dens_keep < 1.0 - 1e-12:
        raise ValueError("crop_to_rms and dens_keep cannot be combined")
    if crop_to_rms and shape_mode == "tps":
        raise ValueError("crop_to_rms cannot combine with TPS")
    if crop_to_rms and (ot_xyz or "").lower() == "occ":
        raise ValueError("crop_to_rms cannot combine with occupancy xyz")
    r_geom = r_star if crop_rms is None else float(crop_rms)
    if crop_rms is not None and not crop_to_rms:
        raise ValueError("crop_rms requires crop_to_rms")
    if crop_to_rms and r_geom <= 0:
        raise ValueError(f"crop RMS must be positive, got {r_geom}")

    if method == "copy_last":
        src = stages[spec.src]
        out = copy_last(src, n=n, seed=seed)
        xyz = dens_keep_cloud(out.obsm["spatial_3D"], dens_keep, morph_rng)
        return _finalize(out.X, xyz, panel, spec)

    if method == "scale_copy":
        src = stages[spec.src]
        if crop_to_rms:
            out = copy_last(src, n=n, seed=seed)
            xyz = radius_crop_resample(out.obsm["spatial_3D"], r_geom, morph_rng)
            return _finalize(out.X, xyz, panel, spec)
        out = scale_copy(src, n=n, r_target=r_star, seed=seed)
        xyz = dens_keep_cloud(out.obsm["spatial_3D"], dens_keep, morph_rng)
        if dens_keep < 1.0 - 1e-12:
            xyz = scale_cloud(xyz, r_star)
        return _finalize(out.X, xyz, panel, spec)

    if method == "shift_scale":
        src = stages[spec.src]
        if spec.mode == "interp":
            prev, nxt = stages[spec.left], stages[spec.right]
            w = interp_weight(spec.t_left, spec.t_right, spec.t)
            out = shift_scale(src, prev, nxt, r_star, n=n, alpha=w, seed=seed, clip_min=clip)
        else:
            delta, alpha = _delta_for(spec, stages, panel, cfg)
            out = scale_copy(src, n=n, r_target=r_star, seed=seed)
            out.X = add_delta(to_dense(out.X), delta, alpha=alpha, clip_min=clip)
        xyz = dens_keep_cloud(out.obsm["spatial_3D"], dens_keep, morph_rng)
        if dens_keep < 1.0 - 1e-12:
            xyz = scale_cloud(xyz, r_star)
        return _finalize(out.X, xyz, panel, spec)

    # --- full: composition + cluster placement --------------------------------
    if composition is None:
        composition = CompositionT2(spec.setting, cfg["time"][spec.setting], eps=float(cfg["composition"]["eps"]))
        composition.fit(
            {name: stages[name].obs["celltype"] for name in stages},
            {name: times[name] for name in stages},
        )

    alloc = composition.allocate(spec.t, n, rng, allowed_times=spec.allowed_times)
    got = sum(alloc.values())
    if got != n:
        raise RuntimeError(f"allocation {got} != n {n}")

    mix_expr = bool((cfg.get("composition") or {}).get("mix_expr", False)) if mix_expr is None else bool(mix_expr)
    per_cluster = bool((cfg.get("shift") or {}).get("per_cluster", True))
    rescale_after = bool((cfg.get("shape") or {}).get("rescale_after_place", True))
    progs = birth_progenitor(spec.setting)
    birth = birth_clusters(spec.setting)

    if spec.mode == "interp":
        w = interp_weight(spec.t_left, spec.t_right, spec.t)
        default_side = nearest_side(spec.t, spec.t_left, spec.t_right)
        side = (geom_src or default_side).lower()
        if side not in {"left", "right"}:
            raise ValueError(f"geom_src must be left|right, got {side}")
        geom_stage = spec.left if side == "left" else spec.right
        expr_stage = spec.left if default_side == "left" else spec.right
        mix_xyz = (mix_xyz or "paired").lower()
        if mix_xyz not in {"paired", "left", "right"}:
            raise ValueError(f"mix_xyz must be paired|left|right, got {mix_xyz}")
        if mix_xyz == "left":
            geom_stage = spec.left
        elif mix_xyz == "right":
            geom_stage = spec.right
        if mix:
            expr_w = {spec.left: 1.0 - w, spec.right: w}
        elif mix_expr:
            expr_w = {spec.left: 1.0 - w, spec.right: w}
        else:
            # Geometry override must not switch the expression library.
            expr_w = {expr_stage: 1.0}
        global_delta, delta_by_k = cluster_mean_delta(stages[spec.left], stages[spec.right], spec.setting)
        _, alpha = _delta_for(spec, stages, panel, cfg)
        # Δ transports left → right. Flip only when expression was drawn from the
        # right (not merely when the geometry template is the right cloud).
        expr_only = [s for s, wt in expr_w.items() if wt > 0]
        if len(expr_only) == 1 and expr_only[0] == spec.right:
            alpha = float(alpha) - 1.0
    else:
        geom_stage = spec.geom_src
        expr_w = {spec.src: 1.0}
        global_delta, alpha = _delta_for(spec, stages, panel, cfg)
        delta_by_k = None
        if per_cluster and spec.left in stages and spec.right in stages:
            global_delta, delta_by_k = cluster_mean_delta(stages[spec.left], stages[spec.right], spec.setting)

    if use_flow:
        if pca is None or flow_model is None:
            raise ValueError("use_flow requires pca and flow_model (run scripts/15_train_flow_t2.py)")
        # Type-internal transport from a single source; mixing right-anchor
        # cells would push them past the target.
        if spec.mode == "interp":
            geom_stage = spec.left
            expr_w = {spec.left: 1.0}
        else:
            geom_stage = spec.src
            expr_w = {spec.src: 1.0}

    axis_tgt = None
    if shape_mode == "anisotropic" and spec.t_left is not None and spec.right in stages:
        std_map = axis_map(stages, times)
        if spec.mode == "interp":
            # geom_w overrides the axis-ratio weight only. The online TSR can
            # imply a different effective weight than the clock (heart E8.5:
            # RMS 255 sits at w~0.67 of the E8.25->E8.75 log path, not 0.5).
            t_axis = spec.t
            if geom_w is not None:
                t_axis = spec.t_left + float(geom_w) * (spec.t_right - spec.t_left)
            axis_tgt = axis_interp(
                spec.t_left, spec.t_right, t_axis,
                std_map[spec.t_left], std_map[spec.t_right],
            )
        else:
            axis_tgt = std_map[times[spec.geom_src]]

    geom_mode = "isotropic" if shape_mode == "tps" else shape_mode
    needed = {geom_stage}
    needed.update(s for s in expr_w if expr_w[s] > 0)
    if spec.left:
        needed.add(spec.left)
    if spec.right:
        needed.add(spec.right)

    stage_X, stage_xyz, stage_cl = {}, {}, {}
    use_occ = (ot_xyz or "").lower() == "occ"
    for name in needed:
        if name not in stages:
            continue
        C = spatial_xyz(stages[name])
        stage_X[name] = to_dense(stages[name].X)
        stage_cl[name] = np.asarray(labels_to_clusters(stages[name].obs["celltype"], spec.setting))
        if use_occ:
            Z, r, _ = canonicalize(C)
            stage_xyz[name] = (Z / max(r, 1e-8)).astype(np.float32)
        elif shape_mode == "tps" and spec.mode == "interp" and spec.left in stages and spec.right in stages and name == spec.left:
            w = interp_weight(spec.t_left, spec.t_right, spec.t)
            tps_n = int(cfg.get("shape", {}).get("tps_n") or 800)
            stage_xyz[name] = ot_interpolate_clouds(
                C, spatial_xyz(stages[spec.right]), w, r_star, n_pair=tps_n, rng=morph_rng
            )
        else:
            stage_xyz[name] = transform_cloud(C, r_star, mode=geom_mode, axis_std_target=axis_tgt)

    mixed_expr = len([s for s, wt in expr_w.items() if wt > 0]) > 1
    pair_xyz = mixed_expr and mix_xyz == "paired"
    xyz_rng = np.random.default_rng(int(seed) + 1_000_003)
    if ot_interp:
        if spec.mode != "interp" or spec.left is None or spec.right is None:
            raise ValueError("ot_interp only applies to interpolation targets")
        w_ot = interp_weight(spec.t_left, spec.t_right, spec.t)
        if ot_w is not None:
            w_ot = float(ot_w)
            if not (0.0 <= w_ot <= 1.0):
                raise ValueError(f"ot_w must be in [0, 1], got {ot_w}")
        n_pair = int(
            ot_n_pair
            or (cfg.get("shape") or {}).get("ot_n_pair")
            or (cfg.get("shape") or {}).get("tps_n")
            or 800
        )
        w_time = interp_weight(spec.t_left, spec.t_right, spec.t)
        xyz_for_alloc = "left" if (ot_xyz or "").lower() == "occ" else ot_xyz
        if (ot_xyz or "lerp").lower() in {"lerp", "occ"} and spec.left in stage_xyz and spec.right in stage_xyz:
            stage_xyz[spec.right] = match_axis_flips(
                stage_xyz[spec.left], stage_xyz[spec.right], morph_rng, n_pair=n_pair
            )
        X, xyz, expr_clusters = ot_interpolate_alloc(
            stage_X,
            stage_xyz,
            stage_cl,
            alloc,
            spec.left,
            spec.right,
            w_ot,
            rng,
            pca=pca,
            birth=birth,
            progenitors=progs,
            n_pair=n_pair,
            sigma_birth=float(cfg["composition"]["sigma_birth"]),
            xyz_mode=xyz_for_alloc,
            x_mode=ot_x,
            xyz_w=w_time,
            xyz_rng=morph_rng,
            unique_pick=bool(ot_unique),
        )
        if (ot_xyz or "").lower() == "occ":
            occ_bins = int((cfg.get("shape") or {}).get("occ_bins") or 48)
            occ_xyz = occupancy_interpolate_cloud(
                stage_xyz[spec.left],
                stage_xyz[spec.right],
                w_time,
                n=len(X),
                rng=morph_rng,
                n_bins=occ_bins,
            )
            xyz = ot_assign_xyz(xyz, occ_xyz)
        expr_src = np.full(len(X), spec.left, dtype=object)
        no_delta = True
    else:
        X, xyz, expr_clusters, expr_src = sample_paired_cells(
            stage_X,
            stage_xyz,
            stage_cl,
            alloc,
            rng,
            geom_stage=geom_stage,
            expr_weights=expr_w,
            progenitors=progs,
            birth=birth,
            sigma_birth=float(cfg["composition"]["sigma_birth"]),
            pair_xyz_to_expr=pair_xyz,
            xyz_rng=xyz_rng,
        )

    if use_flow:
        t_start, dt_flow = _flow_clock(spec, times)
        trained = trained_clusters or set()
        X = transport_expression(
            X,
            expr_clusters,
            pca,
            flow_model,
            trained,
            spec.setting,
            t=t_start,
            dt=dt_flow,
            steps=int(cfg.get("flow", {}).get("euler_steps") or 10),
        )

    if not no_delta:
        mixed = spec.mode == "interp" and len([s for s, w in expr_w.items() if w > 0]) > 1
        if mixed:
            w_delta = interp_weight(spec.t_left, spec.t_right, spec.t)
            X_out = np.empty_like(X)
            for st in np.unique(expr_src):
                mask = expr_src == st
                if st == spec.right:
                    a = float(w_delta) - 1.0
                elif st == spec.left:
                    a = float(w_delta)
                else:
                    a = float(alpha)
                if per_cluster and delta_by_k is not None:
                    X_out[mask] = apply_cluster_delta(
                        X[mask], expr_clusters[mask], global_delta, delta_by_k, a, clip
                    )
                else:
                    X_out[mask] = add_delta(X[mask], global_delta, alpha=a, clip_min=clip)
            X = X_out
        elif per_cluster and delta_by_k is not None:
            X = apply_cluster_delta(X, expr_clusters, global_delta, delta_by_k, alpha, clip)
        else:
            X = add_delta(X, global_delta, alpha=alpha, clip_min=clip)

    if crop_to_rms:
        tmpl_name = geom_stage
        tmpl = spatial_xyz(stages[tmpl_name])
        keep_n = len(radius_crop_indices(tmpl, r_geom))
        print(f"crop_xyz {tmpl_name}: {len(tmpl)} -> {keep_n} (r_target={r_geom:.1f})")
        xyz = crop_place_xyz(
            xyz,
            tmpl,
            r_geom,
            morph_rng,
            mode=geom_mode,
            axis_std_target=axis_tgt,
        )
    if rescale_after:
        xyz = scale_cloud(xyz, r_geom if crop_to_rms else r_star)
    xyz = dens_keep_cloud(xyz, dens_keep, morph_rng)
    xyz = add_jitter(xyz, rng, jitter)
    if rescale_after:
        xyz = scale_cloud(xyz, r_geom if crop_to_rms else r_star)
    X = np.clip(np.asarray(X, dtype=np.float32), clip, None)
    return _finalize(X, xyz, panel, spec)


def _finalize(X, xyz, panel: list[str], spec: TargetSpec) -> ad.AnnData:
    X = np.asarray(X, dtype=np.float32)
    xyz = np.asarray(xyz, dtype=np.float32)
    adata = ad.AnnData(X)
    adata.var_names = list(panel)
    adata.obs_names = [f"pred_{i}" for i in range(X.shape[0])]
    adata.obsm["spatial_3D"] = xyz
    assert_t2_submission(adata, spec.n_genes, spec.n_min, spec.n_max)
    return adata


def write_pred(adata: ad.AnnData, path: Path, spec: TargetSpec, clip_min: float = 0.0) -> Path:
    write_t2(adata.X, adata.obsm["spatial_3D"], list(adata.var_names), path, clip_min=clip_min)
    saved = load_adata(path)
    assert_t2_submission(saved, spec.n_genes, spec.n_min, spec.n_max)
    return path
