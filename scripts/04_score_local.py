#!/usr/bin/env python
"""Local T1 veckit score: pred vs E9.5, reference E8.5."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1.paths import ROOT as PROJ, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="prediction .h5ad")
    parser.add_argument("--target", default=None, help="default: data/E9.5_RNA.h5ad")
    parser.add_argument("--reference", default=None, help="default: data/E8.5_RNA.h5ad")
    args = parser.parse_args()

    cfg = load_config()
    pred = Path(args.pred)
    if not pred.is_absolute():
        pred = (Path.cwd() / pred).resolve()
    target = Path(args.target) if args.target else (PROJ / cfg["paths"]["adata_95"])
    reference = Path(args.reference) if args.reference else (PROJ / cfg["paths"]["adata_85"])

    try:
        from veckit import score
    except ImportError:
        print("veckit is not installed.  pip install veckit")
        print("CLI equivalent:")
        print(f"  veckit --task T1 --input {pred} --target {target} --reference {reference}")
        return 2

    print(f"input     {pred}")
    print(f"target    {target}")
    print(f"reference {reference}")
    result = score(task="T1", input=str(pred), target=str(target), reference=str(reference))
    metrics = result.get("metrics", result)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
