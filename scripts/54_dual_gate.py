#!/usr/bin/env python
"""Dual gate for a discrete-X candidate.

Gate 1 (native): vs mix on NFS/de; vs lo-pm on variogram.
  NFS  < mix
  de   >= mix − 0.02
  var  <= pm × 1.10
Native ODS is reported but not required — freeze path is allowed to fail ODS.

Gate 2 (same X frozen onto k04b6 / 09a analog): vs lo-pm, no anti-transfer.
  NFS  <= pm × 1.05
  mmd  <= pm × 1.05
  var  <= pm × 1.10
  de   >= pm − 0.02
  ODS  >= k04b6 − 0.01   (shape must lock)

Both gates PASS → eligible to freeze onto live 09a (still need board > 68.26).
Gate 2 FAIL → do not submit (ticket 20/52/110/170 fingerprint).
Do not use heart W2 as a substitute window.

  .venv/bin/python scripts/54_dual_gate.py --window embryo \\
    --mix outputs/t2/embryo/gate725/mix.h5ad \\
    --pm outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --k04b6 outputs/t2/embryo/gate725/slice_spatial_k0.4_b6.h5ad \\
    --native outputs/t2/embryo/gate725_pick/lo_k04b6/native.h5ad \\
    --freeze outputs/t2/embryo/gate725_pick/lo_k04b6/freeze.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from importlib.machinery import SourceFileLoader  # noqa: E402

g = SourceFileLoader("flow_gate", str(ROOT / "scripts/diagnostics" / "26_flow_gate.py")).load_module()

DE_SLACK = 0.02
VAR_SLACK = 1.10
NFS_SLACK = 1.05
MMD_SLACK = 1.05
ODS_LOCK = 0.01


def _print_row(r):
    print(
        f"{r['name']:20s} {r['nfs']:9.5f} {r['mmd']:9.5f} {r['var']:9.5f} "
        f"{r['de']:8.4f} {r['ddir']:8.4f} {r['ods']:8.4f} {r['d2']:8.5f}"
    )


def _score(name, path, XT, CT, XR, panel):
    X, C, genes = g._xy(path)
    X = g._align(X, genes, panel)
    return g._row(name, X, C, XT, CT, XR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=sorted(g.WINDOWS), default="embryo")
    ap.add_argument("--mix", type=Path, required=True)
    ap.add_argument("--pm", type=Path, required=True)
    ap.add_argument("--k04b6", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--freeze", type=Path, required=True)
    args = ap.parse_args()

    spec = g.WINDOWS[args.window]
    XT, CT, panel = g._xy(spec["target"])
    XR, _, pref = g._xy(spec["reference"])
    XR = g._align(XR, pref, panel)

    print(f"# dual-gate {args.window}: {spec['note']}")
    print(
        f"{'variant':20s} {'NFS↓':>9s} {'mmd_u↓':>9s} {'vario↓':>9s} "
        f"{'de_sc↑':>8s} {'de_dir↑':>8s} {'ODS↑':>8s} {'d2↓':>8s}"
    )
    mix = _score("mix", args.mix, XT, CT, XR, panel)
    pm = _score("pm", args.pm, XT, CT, XR, panel)
    k04 = _score("k04b6", args.k04b6, XT, CT, XR, panel)
    native = _score("native", args.native, XT, CT, XR, panel)
    freeze = _score("freeze", args.freeze, XT, CT, XR, panel)
    for r in (mix, pm, k04, native, freeze):
        _print_row(r)

    g1_nfs = native["nfs"] < mix["nfs"]
    g1_de = native["de"] >= mix["de"] - DE_SLACK
    g1_var = native["var"] <= pm["var"] * VAR_SLACK
    gate1 = g1_nfs and g1_de and g1_var

    g2_nfs = freeze["nfs"] <= pm["nfs"] * NFS_SLACK
    g2_mmd = freeze["mmd"] <= pm["mmd"] * MMD_SLACK
    g2_var = freeze["var"] <= pm["var"] * VAR_SLACK
    g2_de = freeze["de"] >= pm["de"] - DE_SLACK
    g2_ods = freeze["ods"] >= k04["ods"] - ODS_LOCK
    gate2 = g2_nfs and g2_mmd and g2_var and g2_de and g2_ods

    print()
    print(
        f"GATE1 native  NFS {'PASS' if g1_nfs else 'FAIL'}  "
        f"de {'PASS' if g1_de else 'FAIL'}  "
        f"var-vs-pm {'PASS' if g1_var else 'FAIL'}  "
        f"=> {'PASS' if gate1 else 'FAIL'}"
    )
    print(
        f"GATE2 freeze  NFS {'PASS' if g2_nfs else 'FAIL'}  "
        f"mmd {'PASS' if g2_mmd else 'FAIL'}  "
        f"var {'PASS' if g2_var else 'FAIL'}  "
        f"de {'PASS' if g2_de else 'FAIL'}  "
        f"ODS-lock {'PASS' if g2_ods else 'FAIL'}  "
        f"=> {'PASS' if gate2 else 'FAIL'}"
    )
    if gate1 and gate2:
        print("RESULT PASS — freeze X onto live 09a is eligible; still need board > 68.26")
        return 0
    if gate1 and not gate2:
        print("RESULT FAIL gate2 — anti-transfer on freeze; do not submit")
        return 2
    print("RESULT FAIL — do not submit")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
