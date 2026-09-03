#!/usr/bin/env python
"""Audit T2 training files, panels, RMS, and working-cluster maps → outputs/t2/audit.json."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters  # noqa: E402
from t2.geometry import canonicalize, rms_radius  # noqa: E402
from t2.io import load_adata, load_panel, spatial_xyz, to_dense  # noqa: E402
from t2.paths import ROOT as PROJ, ensure_out_dirs, load_config, resolve  # noqa: E402


def _x_stats(adata, n=3000) -> dict:
    sl = slice(0, min(n, adata.n_obs))
    X = to_dense(adata[sl].X)
    return {
        "sample_cells": int(X.shape[0]),
        "min": float(X.min()),
        "max": float(X.max()),
        "mean": float(X.mean()),
        "frac_zero": float((X == 0).mean()),
        "has_nan": bool(np.isnan(X).any()),
        "has_inf": bool(np.isinf(X).any()),
        "has_negative": bool((X < -1e-8).any()),
    }


def _type_report(adata, setting: str) -> dict:
    labels = adata.obs["celltype"].astype(str)
    vc = labels.value_counts()
    clusters = labels_to_clusters(labels, setting)
    cc = Counter(clusters)
    n = len(labels)
    return {
        "n_types": int(labels.nunique()),
        "counts": {k: int(v) for k, v in vc.items()},
        "fractions": {k: float(v / n) for k, v in vc.items()},
        "cluster_counts": {k: int(v) for k, v in cc.items()},
        "cluster_fractions": {k: float(v / n) for k, v in cc.items()},
    }


def _geom(adata) -> dict:
    xyz = spatial_xyz(adata)
    Z, r, _ = canonicalize(xyz)
    std = Z.std(axis=0, ddof=1)
    return {
        "shape": list(xyz.shape),
        "dtype": str(xyz.dtype),
        "has_nan": bool(np.isnan(xyz).any()),
        "rms": float(r),
        "axis_std": [float(x) for x in std],
        "centroid": [float(x) for x in xyz.mean(axis=0)],
        "rms_raw": float(rms_radius(xyz)),
    }


def _panel_equal(panel: list[str], names: list[str]) -> bool:
    return list(panel) == list(names)


def main() -> int:
    cfg = load_config()
    ensure_out_dirs(cfg)
    warnings: list[str] = []

    stages = {
        "E6.75": ("embryo", resolve(cfg, "embryo", "E6.75")),
        "E7.25": ("embryo", resolve(cfg, "embryo", "E7.25")),
        "E8.0": ("embryo", resolve(cfg, "embryo", "E8.0")),
        "E8.25": ("heart", resolve(cfg, "heart", "E8.25")),
        "E8.75": ("heart", resolve(cfg, "heart", "E8.75")),
        "E9.5": ("heart", resolve(cfg, "heart", "E9.5")),
    }

    loaded = {}
    report = {}
    for name, (setting, path) in stages.items():
        if not path.exists():
            warnings.append(f"missing {path}")
            continue
        adata = load_adata(path, backed="r")
        loaded[name] = adata
        exp = cfg["expected"][name]
        rec = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "obs_columns": list(adata.obs.columns),
            "obsm_keys": list(adata.obsm.keys()),
            "X": _x_stats(adata),
            "geometry": _geom(adata),
        }
        try:
            rec["celltype"] = _type_report(adata, setting)
        except KeyError as e:
            rec["celltype_error"] = str(e)
            warnings.append(str(e))
        if adata.n_obs != exp["n_obs"]:
            warnings.append(f"{name} n_obs {adata.n_obs} != expected {exp['n_obs']}")
        if adata.n_vars != exp["n_vars"]:
            warnings.append(f"{name} n_vars {adata.n_vars} != expected {exp['n_vars']}")
        if abs(rec["geometry"]["rms"] - exp["rms"]) > 2.0:
            warnings.append(f"{name} RMS {rec['geometry']['rms']:.1f} != expected {exp['rms']}")
        if "bytes" in exp and abs(rec["bytes"] - exp["bytes"]) > 5_000_000:
            warnings.append(f"{name} file size {rec['bytes']} != expected ~{exp['bytes']}")
        if rec["X"]["has_negative"] or rec["X"]["has_nan"] or rec["X"]["has_inf"]:
            warnings.append(f"{name} X has negative/NaN/Inf")
        if rec["geometry"]["has_nan"]:
            warnings.append(f"{name} spatial_3D has NaN")
        report[name] = rec
        print(
            f"{name:6s}  {adata.n_obs:6d} x {adata.n_vars:3d}  "
            f"rms={rec['geometry']['rms']:.1f}  "
            f"axes={[round(x, 1) for x in rec['geometry']['axis_std']]}  "
            f"MB={rec['bytes']/1e6:.1f}"
        )

    genes = {name: list(loaded[name].var_names) for name in loaded}
    embryo_panel = load_panel(resolve(cfg, "embryo", "panel_val_interp"))
    heart_pi = load_panel(resolve(cfg, "heart", "panel_val_interp"))
    heart_pe = load_panel(resolve(cfg, "heart", "panel_val_extrap"))
    if embryo_panel is None:
        warnings.append("missing embryo val_interp panel")
        embryo_panel = genes.get("E6.75", [])
    if heart_pi is None:
        warnings.append("missing heart val_interp panel")
        heart_pi = genes.get("E8.25", [])
    if heart_pe is None:
        warnings.append("missing heart val_extrap panel")
        heart_pe = heart_pi

    extra_e80 = [g for g in genes.get("E8.0", []) if g not in genes.get("E6.75", [])]
    e80_minus = [g for g in genes.get("E8.0", []) if g in set(genes.get("E6.75", []))]
    embryo_498_ok = e80_minus == genes.get("E6.75", [])
    heart_same = genes.get("E8.25") == genes.get("E8.75") == genes.get("E9.5")
    e80_heart_same = genes.get("E8.0") == genes.get("E8.25")
    e675_e725_same = genes.get("E6.75") == genes.get("E7.25")

    types = {name: set(loaded[name].obs["celltype"].astype(str)) for name in loaded}

    def _share(a, b):
        sa, sb = types[a], types[b]
        na = loaded[b].n_obs
        frac = float(loaded[b].obs["celltype"].astype(str).isin(sa).mean()) if na else 0.0
        return {
            "shared_names": sorted(sa & sb),
            "only_left": sorted(sa - sb),
            "only_right": sorted(sb - sa),
            "frac_right_cells_with_left_name": frac,
        }

    gene_block = {
        "E6.75_E7.25_same_order": e675_e725_same,
        "E8.0_minus_Casp4_Pnliprp1_equals_E675": embryo_498_ok,
        "E8.0_extra": extra_e80,
        "heart_three_same_order": heart_same,
        "E8.0_equals_heart_500": e80_heart_same,
        "panel_embryo_n": len(embryo_panel),
        "panel_heart_interp_n": len(heart_pi),
        "panel_heart_extrap_n": len(heart_pe),
        "panel_embryo_equals_E675": _panel_equal(embryo_panel, genes.get("E6.75", [])),
        "panel_embryo_equals_E725": _panel_equal(embryo_panel, genes.get("E7.25", [])),
        "panel_heart_interp_equals_E825": _panel_equal(heart_pi, genes.get("E8.25", [])),
        "panel_heart_interp_equals_extrap": heart_pi == heart_pe,
        "need_reindex_embryo_E80": not _panel_equal(embryo_panel, genes.get("E8.0", [])),
    }
    if extra_e80 != ["Casp4", "Pnliprp1"] and set(extra_e80) != {"Casp4", "Pnliprp1"}:
        warnings.append(f"E8.0 extra genes {extra_e80} (expected Casp4, Pnliprp1)")
    if not gene_block["panel_embryo_equals_E675"]:
        warnings.append("embryo panel != E6.75 var_names; submissions will reindex")
    if not gene_block["panel_heart_interp_equals_E825"]:
        warnings.append("heart panel != E8.25 var_names; submissions will reindex")
    if not gene_block["panel_heart_interp_equals_extrap"]:
        warnings.append("heart val_interp panel != val_extrap panel")

    # Cluster maps used (for audit.json when mappings change).
    cluster_maps = {
        "embryo": {
            name: report[name]["celltype"]["cluster_counts"]
            for name in ("E6.75", "E7.25", "E8.0")
            if name in report and "celltype" in report[name]
        },
        "heart": {
            name: report[name]["celltype"]["cluster_counts"]
            for name in ("E8.25", "E8.75", "E9.5")
            if name in report and "celltype" in report[name]
        },
    }

    audit = {
        "stages": report,
        "genes": gene_block,
        "overlap": {
            "E6.75_E7.25": _share("E6.75", "E7.25") if "E6.75" in types and "E7.25" in types else {},
            "E7.25_E8.0": _share("E7.25", "E8.0") if "E7.25" in types and "E8.0" in types else {},
            "E8.25_E8.75": _share("E8.25", "E8.75") if "E8.25" in types and "E8.75" in types else {},
            "E8.75_E9.5": _share("E8.75", "E9.5") if "E8.75" in types and "E9.5" in types else {},
        },
        "cluster_maps": cluster_maps,
        "warnings": warnings,
    }

    out = PROJ / cfg["paths"]["preds"] / "audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2))
    print(f"\nWrote {out}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(" -", w)
    return 1 if any("n_vars" in w or "Unmapped" in w or "missing" in w for w in warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
