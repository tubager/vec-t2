from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or (ROOT / "configs" / "t2.yaml")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve(cfg: dict, *keys: str) -> Path:
    node = cfg["paths"]
    for k in keys:
        node = node[k]
    return ROOT / node


def pred_dir(cfg: dict, setting: str) -> Path:
    path = ROOT / cfg["paths"]["preds"] / setting
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_out_dirs(cfg: dict) -> None:
    root = ROOT / cfg["paths"]["preds"]
    for sub in ("embryo", "heart"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "audit.json").parent.mkdir(parents=True, exist_ok=True)
