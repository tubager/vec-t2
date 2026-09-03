#!/usr/bin/env python
"""Audit T1 training files, dump gene order, write outputs/t1/audit.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1.clusters import labels_to_clusters  # noqa: E402
from t1.io import load_adata, load_panel, to_dense, write_gene_order  # noqa: E402
from t1.paths import ROOT as PROJ, ensure_out_dirs, load_config, resolve  # noqa: E402


def _x_stats(adata, n=500) -> dict:
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


def _type_report(adata) -> dict:
    labels = adata.obs["celltype"].astype(str)
    vc = labels.value_counts()
    clusters = labels_to_clusters(labels)
    from collections import Counter

    cc = Counter(clusters)
    n = len(labels)
    return {
        "n_types": int(labels.nunique()),
        "counts": {k: int(v) for k, v in vc.items()},
        "fractions": {k: float(v / n) for k, v in vc.items()},
        "cluster_counts": {k: int(v) for k, v in cc.items()},
    }


def main() -> int:
    cfg = load_config()
    ensure_out_dirs(cfg)
    p85 = PROJ / cfg["paths"]["adata_85"]
    p95 = PROJ / cfg["paths"]["adata_95"]
    a85 = load_adata(p85, backed="r")
    a95 = load_adata(p95, backed="r")

    genes85 = list(a85.var_names)
    genes95 = list(a95.var_names)
    genes_equal = genes85 == genes95

    panel_path = resolve(cfg, "panel")
    panel = load_panel(panel_path)
    panel_source = str(panel_path) if panel else "training_var_names"
    if panel is None:
        panel = genes85
        print(f"[warn] {panel_path} missing; using E8.5 var_names as gene order")
    panel_match_85 = panel == genes85
    panel_match_95 = panel == genes95

    fallback = resolve(cfg, "panel_fallback")
    write_gene_order(panel, fallback)

    types85 = set(a85.obs["celltype"].astype(str))
    types95 = set(a95.obs["celltype"].astype(str))
    nt85 = float((a85.obs["celltype"].astype(str) == "Neural Tube").mean())
    nt95 = float((a95.obs["celltype"].astype(str) == "Neural Tube").mean())

    exp = cfg["expected"]
    warnings = []
    if a85.n_obs != exp["n_obs_85"]:
        warnings.append(f"E8.5 n_obs {a85.n_obs} != expected {exp['n_obs_85']}")
    if a95.n_obs != exp["n_obs_95"]:
        warnings.append(f"E9.5 n_obs {a95.n_obs} != expected {exp['n_obs_95']}")
    if a85.n_vars != exp["n_vars"] or a95.n_vars != exp["n_vars"]:
        warnings.append(f"n_vars {a85.n_vars}/{a95.n_vars} != {exp['n_vars']}")
    if not genes_equal:
        warnings.append("E8.5 and E9.5 var_names differ")
    if not panel_match_85:
        warnings.append("Panel does not match E8.5 gene order; submissions will reindex")
    if abs(nt85 - 0.046) > 0.005:
        warnings.append(f"Neural Tube E8.5 fraction {nt85:.4f} (expected ~0.046)")

    audit = {
        "E8.5": {
            "path": str(p85),
            "n_obs": int(a85.n_obs),
            "n_vars": int(a85.n_vars),
            "obs_columns": list(a85.obs.columns),
            "obsm_keys": list(a85.obsm.keys()),
            "X": _x_stats(a85),
            "celltype": _type_report(a85),
        },
        "E9.5": {
            "path": str(p95),
            "n_obs": int(a95.n_obs),
            "n_vars": int(a95.n_vars),
            "obs_columns": list(a95.obs.columns),
            "obsm_keys": list(a95.obsm.keys()),
            "X": _x_stats(a95),
            "celltype": _type_report(a95),
        },
        "genes": {
            "same_order": genes_equal,
            "n": len(genes85),
            "first": genes85[:8],
            "last": genes85[-5:],
            "panel_source": panel_source,
            "panel_n": len(panel),
            "panel_matches_E85": panel_match_85,
            "panel_matches_E95": panel_match_95,
            "gene_order_path": str(fallback),
        },
        "overlap": {
            "shared_types": sorted(types85 & types95),
            "only_E85": sorted(types85 - types95),
            "only_E95": sorted(types95 - types85),
            "frac_E85_type_in_E95": float(a85.obs["celltype"].astype(str).isin(types95).mean()),
            "frac_E95_type_in_E85": float(a95.obs["celltype"].astype(str).isin(types85).mean()),
            "neural_tube_frac_E85": nt85,
            "neural_tube_frac_E95": nt95,
        },
        "warnings": warnings,
    }

    out = resolve(cfg, "audit")
    out.write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: audit[k] if k != "E8.5" and k != "E9.5" else {"n_obs": audit[k]["n_obs"], "n_vars": audit[k]["n_vars"], "n_types": audit[k]["celltype"]["n_types"]} for k in audit}, indent=2))
    print(f"\nWrote {out}")
    print(f"Wrote gene order to {fallback} ({len(panel)} genes)")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(" -", w)
        return 1 if any("var_names differ" in w or "n_vars" in w for w in warnings) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
