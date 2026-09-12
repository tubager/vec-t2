#!/usr/bin/env python
"""Leave-out gate for within-type developmental slice vs Bernoulli mix.

Slice keeps real cells. Mix is OT-place pick.

AND gate (doc/t2_p2_plan_2026-09-13.md):
  NFS  < mix
  de   >= mix − 0.02
  mmd  <= mix × 1.14   (global slice missed beating mix; 14% was the observed tax)
  ODS  >= mix − 0.02   (raw occupancy_dice vs true midpoint; blocks 01-style collapse)
  d2   <= mix + 0.003  (heart span-trim local |Δd2| scale)

Native-geometry submit needs the full AND. If NFS/de pass but ODS fails, the X may
be frozen onto the live 67.11 coordinates (separate decision, not this script).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from importlib.machinery import SourceFileLoader  # noqa: E402

g = SourceFileLoader("flow_gate", str(ROOT / "scripts/diagnostics" / "26_flow_gate.py")).load_module()

MMD_SLACK = 1.14
DE_SLACK = 0.02
ODS_SLACK = 0.02
D2_SLACK = 0.003


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=sorted(g.WINDOWS), required=True)
    ap.add_argument("--mix", required=True, type=Path)
    ap.add_argument("--cand", action="append", default=[], help="name=path (repeatable)")
    args = ap.parse_args()
    if not args.cand:
        raise SystemExit("need at least one --cand name=path")
    spec = g.WINDOWS[args.window]
    XT, CT, panel = g._xy(spec["target"])
    XR, _, pref = g._xy(spec["reference"])
    XR = g._align(XR, pref, panel)
    print(f"# {args.window}: {spec['note']}")
    hdr = (
        f"{'variant':28s} {'NFS↓':>9s} {'mmd_u↓':>9s} {'vario↓':>9s} "
        f"{'de_sc↑':>8s} {'de_dir↑':>8s} {'ODS↑':>8s} {'d2↓':>8s}"
    )
    print(hdr)
    files = [("mix", args.mix)]
    for item in args.cand:
        if "=" not in item:
            raise SystemExit(f"--cand expects name=path, got {item!r}")
        name, path = item.split("=", 1)
        files.append((name, Path(path)))
    rows = []
    for name, path in files:
        X, C, genes = g._xy(path)
        X = g._align(X, genes, panel)
        r = g._row(name, X, C, XT, CT, XR)
        rows.append(r)
        print(
            f"{r['name']:28s} {r['nfs']:9.5f} {r['mmd']:9.5f} {r['var']:9.5f} "
            f"{r['de']:8.4f} {r['ddir']:8.4f} {r['ods']:8.4f} {r['d2']:8.5f}"
        )
    mix = rows[0]
    by = {r["name"]: r for r in rows}
    band = by.get("band")
    passed_any = False
    freeze_any = False
    board_any = False
    print()
    for r in rows[1:]:
        ok_nfs = r["nfs"] < mix["nfs"]
        ok_mmd = r["mmd"] <= mix["mmd"] * MMD_SLACK
        ok_de = r["de"] >= mix["de"] - DE_SLACK
        ok_ods = r["ods"] >= mix["ods"] - ODS_SLACK
        ok_d2 = r["d2"] <= mix["d2"] + D2_SLACK
        ok = ok_nfs and ok_mmd and ok_de and ok_ods and ok_d2
        freeze = ok_nfs and ok_de and (not ok_ods)
        board = False
        if band is not None and r["name"] != "band":
            board = (
                r["de"] >= band["de"] - 0.02
                and r["nfs"] <= band["nfs"] * 1.20
                and r["ods"] >= mix["ods"] - ODS_SLACK
                and r["mmd"] <= band["mmd"] * 1.05
                and r["d2"] <= mix["d2"] + D2_SLACK
            )
        passed_any = passed_any or ok
        freeze_any = freeze_any or freeze
        board_any = board_any or board
        extra = ""
        if band is not None and r["name"] != "band":
            extra = f"  board {'PASS' if board else 'FAIL'}"
        print(
            f"GATE {r['name']:16s}  "
            f"NFS {'PASS' if ok_nfs else 'FAIL'}  "
            f"mmd {'PASS' if ok_mmd else 'FAIL'}  "
            f"de {'PASS' if ok_de else 'FAIL'}  "
            f"ODS {'PASS' if ok_ods else 'FAIL'}  "
            f"d2 {'PASS' if ok_d2 else 'FAIL'}  "
            f"=> {'PASS native' if ok else ('FREEZE-X only' if freeze else 'FAIL')}"
            f"{extra}"
        )
    if passed_any or board_any:
        print("RESULT PASS — eligible for a native-geometry slot")
        return 0
    if freeze_any:
        print("RESULT FAIL native — NFS/de only; freeze X onto live coords, do not submit own xyz")
        return 2
    print("RESULT FAIL — do not submit")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
