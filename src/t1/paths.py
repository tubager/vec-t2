from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or (ROOT / "configs" / "t1.yaml")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def ensure_out_dirs(cfg: dict) -> None:
    for key in ("audit", "pca", "composition", "ckpt", "preds", "panel_fallback"):
        path = resolve(cfg, key)
        path.parent.mkdir(parents=True, exist_ok=True)
