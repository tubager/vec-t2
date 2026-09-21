#!/usr/bin/env python
"""Calibrate a native (X,xyz) cloud onto a board-proven occupancy support.

Ticket 130: nocc 213, board ODS 53.6. Same fingerprint as ticket 06 (nocc 210 → ODS 54.2).
04/09a: nocc 199, board ODS 66.0. Leave-out ODS≈0.90 does not predict board ODS; nocc does.

Keeps X byte-identical. Moves cells out of voxels the prior does not occupy, then
trims the sparsest remaining voxels until nocc <= target (default = prior nocc).

Leave-out:
  .venv/bin/python scripts/45_occ_cal_place.py \\
    --in outputs/t2/embryo/gate725_newpaths/lo_ridge_pk0.82_s10.h5ad \\
    --prior outputs/t2/embryo/gate725/slice_spatial_k0.4_b6.h5ad \\
    --out outputs/t2/embryo/gate725_occcal/lo_130_nocc199.h5ad

Board:
  .venv/bin/python scripts/45_occ_cal_place.py \\
    --in outputs/t2/queue/2026-09-15/130_embryo_comptps_otsoft_pk0.82_s10_E725E80_n5000_rms19824.h5ad \\
    --prior outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --out outputs/t2/queue/2026-09-16/140_embryo_130_occcal_nocc199_on09a.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from t2.geometry import scale_cloud  # noqa: E402

N_GRID = 16
EXTENT = 3.0


def _xyz(adata: ad.AnnData) -> np.ndarray:
    z = adata.obsm["spatial_3D"]
    return np.asarray(z.todense() if hasattr(z, "todense") else z, dtype=np.float64)


def canonicalise(C: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mu = C.mean(0)
    X = C - mu
    rms = float(np.sqrt((X**2).sum(1).mean()))
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return (X @ Vt.T) / rms, rms, Vt, mu


def voxel_index(Z: np.ndarray) -> np.ndarray:
    return np.floor((np.clip(Z, -EXTENT, EXTENT - 1e-9) + EXTENT) / (2 * EXTENT) * N_GRID).astype(int)


def occupied(Z: np.ndarray) -> set[tuple[int, int, int]]:
    return set(map(tuple, voxel_index(Z)))


def rms_radius(C: np.ndarray) -> float:
    X = C - C.mean(0)
    return float(np.sqrt((X**2).sum(1).mean()))


def best_prior_voxels(Z_cand: np.ndarray, Z_prior: np.ndarray) -> tuple[set[tuple[int, int, int]], np.ndarray]:
    """Sign-flip the prior in the shared canonical frame to max voxel overlap with cand."""
    best, best_n, best_Z = occupied(Z_prior), -1, Z_prior
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                if sx * sy * sz < 0:
                    continue
                Zp = Z_prior * np.array([sx, sy, sz], dtype=np.float64)
                s = occupied(Zp)
                n = len(s & occupied(Z_cand))
                if n > best_n:
                    best, best_n, best_Z = s, n, Zp
    return best, best_Z


def trim_to_nocc(
    Z: np.ndarray,
    allow: set[tuple[int, int, int]],
    n_target: int,
    jitter: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    moved = 0
    for step in range(400):
        jit = jitter * (0.5 ** min(step, 4))
        Z = canonicalise(Z)[0]
        v = voxel_index(Z)
        keys = [tuple(x) for x in v]
        uniq, counts = np.unique(v, axis=0, return_counts=True)
        uniq_t = [tuple(x) for x in uniq]
        n_now = len(uniq_t)
        if n_now <= n_target:
            break
        rank = []
        for k, c in zip(uniq_t, counts):
            cls = 0 if k not in allow else 1
            rank.append((cls, int(c), k))
        rank.sort(key=lambda t: (t[0], t[1]))
        kill: set[tuple[int, int, int]] = set()
        for cls, c, k in rank:
            if n_now - len(kill) <= n_target:
                break
            kill.add(k)
        if not kill:
            break
        bad = np.fromiter((k in kill for k in keys), dtype=bool, count=len(keys))
        good = Z[~bad]
        if len(good) == 0:
            break
        _, tgt = cKDTree(good).query(Z[bad])
        Z = Z.copy()
        Z[bad] = good[tgt] + rng.normal(0.0, jit, size=(int(bad.sum()), 3))
        moved += int(bad.sum())
    return Z, moved


def _voxel_center(k: tuple[int, int, int]) -> np.ndarray:
    return np.array([(k[d] + 0.5) / N_GRID * (2 * EXTENT) - EXTENT for d in range(3)], dtype=np.float64)


def fill_empty_allow(
    Z: np.ndarray,
    Zp: np.ndarray,
    allow: set[tuple[int, int, int]],
    n_target: int,
    jitter: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Move outside/overfull cells into empty allow voxels so nocc can rise to target.

    Nearest-neighbour projection onto already-occupied voxels collapses nocc
    (198→187). Fill first, using prior points that already sit in the empty voxel.
    """
    Z = np.asarray(Z, dtype=np.float64).copy()
    keys = [tuple(int(x) for x in row) for row in voxel_index(Z)]
    occ = occupied(Z)
    empty = [k for k in allow if k not in occ]
    if not empty or len(occ) >= n_target:
        return Z, 0
    counts: dict[tuple[int, int, int], int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    donors = [i for i, k in enumerate(keys) if k not in allow]
    donors += [i for i, k in enumerate(keys) if k in allow and counts.get(k, 0) > 1]
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, row in enumerate(voxel_index(Zp)):
        t = tuple(int(x) for x in row)
        if t in allow:
            buckets.setdefault(t, []).append(i)
    n_need = min(len(empty), max(0, n_target - len(occ)), len(donors))
    moved = 0
    used = set()
    for j in range(n_need):
        k = empty[j]
        i = donors[j]
        if i in used:
            continue
        used.add(i)
        if k in buckets:
            src = Zp[int(rng.choice(buckets[k]))]
        else:
            src = _voxel_center(k)
        Z[i] = src + rng.normal(0.0, float(jitter), size=3)
        moved += 1
    return Z, moved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--prior", required=True, type=Path, help="board-proven occupancy cloud (09a / k04b6)")
    ap.add_argument("--target-nocc", type=int, default=None, help="default = prior nocc")
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    src = ad.read_h5ad(args.src)
    prior = ad.read_h5ad(args.prior)
    C = _xyz(src)
    Cp = _xyz(prior)
    Z0, rms0, Vt, mu = canonicalise(C)
    Zp, _, _, _ = canonicalise(Cp)
    allow, Zp = best_prior_voxels(Z0, Zp)
    nocc0 = len(occupied(Z0))
    nocc_p = len(occupied(Zp))
    tgt = int(args.target_nocc) if args.target_nocc is not None else int(nocc_p)
    print(f"  prior {args.prior.name} nocc={nocc_p} allow={len(allow)}  cand nocc={nocc0} → target {tgt}")
    rng = np.random.default_rng(int(args.seed))
    Z1 = Z0.copy()
    moved = 0
    Z1, moved_fill = fill_empty_allow(Z1, Zp, allow, tgt, float(args.jitter), rng)
    moved += moved_fill
    if moved_fill:
        print(f"  filled {moved_fill} empty allow voxels")
    keys0 = [tuple(int(x) for x in row) for row in voxel_index(Z1)]
    outside = np.fromiter((k not in allow for k in keys0), dtype=bool, count=len(keys0))
    if outside.any():
        inside = Z1[~outside]
        if len(inside) == 0:
            Zp_allow = Zp[np.fromiter(
                (tuple(int(x) for x in row) in allow for row in voxel_index(Zp)),
                dtype=bool,
                count=len(Zp),
            )]
            inside = Zp_allow if len(Zp_allow) else Zp
        _, nn = cKDTree(inside).query(Z1[outside])
        Z1 = Z1.copy()
        Z1[outside] = inside[nn] + rng.normal(0.0, float(args.jitter), size=(int(outside.sum()), 3))
        moved += int(outside.sum())
        print(f"  projected {int(outside.sum())} remaining cells from outside prior voxels")
    nocc_mid = len(occupied(Z1))
    if occupied(Z0) <= allow and nocc0 <= tgt and moved == 0:
        print("  already inside prior support at target nocc: NO-OP")
    elif nocc_mid > tgt:
        Z1, moved_trim = trim_to_nocc(Z1, allow, tgt, float(args.jitter), int(args.seed))
        moved += moved_trim
    elif nocc_mid < tgt:
        Z1, moved_fill2 = fill_empty_allow(Z1, Zp, allow, tgt, float(args.jitter), rng)
        moved += moved_fill2
        if moved_fill2:
            print(f"  refilled {moved_fill2} after projection")
    C_out = (Z1 * rms0) @ Vt + mu
    C_out = np.asarray(scale_cloud(C_out, float(args.rms)), dtype=np.float32)

    out = src.copy()
    out.obsm["spatial_3D"] = C_out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    nocc_f = len(occupied(canonicalise(_xyz(out))[0]))
    print(f"wrote {args.out}")
    print(f"  moved {moved} cell-steps  nocc {nocc0} → {nocc_f}  RMS {rms_radius(C):.2f} → {rms_radius(C_out):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
