#!/usr/bin/env python
"""Point-cloud density / neighbourhood probes behind doc/t2_p2_plan_2026-09-09.md.

Two modes:

  --files A.h5ad B.h5ad ...   official `median_nn_distance` (NN1 on a 4000 subsample,
                              n-independent), median NN15 over ALL cells (what
                              `neighborhood_mmd` actually sees), RMS and the
                              dimensionless density profile NN1*4000^(1/3)/RMS.
  --truth heart/E8.75         NFS-vs-n sensitivity: subsample the truth cloud and score
                              it against itself with the official `neighborhood_mmd`,
                              once from below (pred n < truth n) and once from above.

Read-only; writes nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.neighbors import NearestNeighbors  # noqa: E402

from common.core_metrics import mmd_unbiased  # noqa: E402  (official estimator)
from common.shape_metrics import median_nn_distance  # noqa: E402  (official NN stat)
from t2.io import load_adata, spatial_xyz  # noqa: E402
from t2.paths import load_config, resolve  # noqa: E402


def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def knn_pseudobulk(x, coords, k: int = 15):
    """Verbatim copy of T2.metrics._knn_neighborhood_pb (keeps self, rank-based)."""
    _, idx = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords)
    return to_dense(x)[idx].mean(1)


def neighborhood_mmd(pred_x, pred_c, true_x, true_c, k: int = 15, n: int = 2000):
    """Verbatim copy of T2.metrics.neighborhood_mmd."""
    return mmd_unbiased(knn_pseudobulk(pred_x, pred_c, k), knn_pseudobulk(true_x, true_c, k), n=n)


def nn15(coords) -> float:
    coords = np.asarray(coords, float)
    k = min(16, len(coords))
    return float(np.median(NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords)[0][:, -1]))


def cloud(path_or_ref: str, cfg: dict):
    if "/" in path_or_ref and not Path(path_or_ref).exists():
        setting, key = path_or_ref.split("/")
        adata = load_adata(resolve(cfg, setting, key))
    else:
        adata = load_adata(Path(path_or_ref))
    return to_dense(adata.X), np.asarray(spatial_xyz(adata), dtype=np.float64)


def report(tag: str, coords) -> None:
    coords = np.asarray(coords, float)
    rms = float(np.sqrt((coords ** 2).sum(1).mean()))
    nn1 = median_nn_distance(coords)
    print(f"{tag[:52]:<52}{len(coords):>7}{rms:>8.1f}{nn1:>10.2f}{nn15(coords):>10.2f}"
          f"{nn1 * 4000 ** (1 / 3) / rms:>9.3f}")


def sensitivity(cfg: dict, ref: str, sizes=(1000, 2000, 3000, 5000, 8000, 12000, 17616, 25179)) -> None:
    x, coords = cloud(ref, cfg)
    rng = np.random.default_rng(0)
    print(f"\n# truth = {ref} (n={len(coords)}): pred is a subsample -> UNDERSHOOTING n")
    print(f"{'pred_n':>8}{'NFS_raw':>10}")
    for n in sizes:
        if n >= len(coords):
            continue
        idx = rng.choice(len(coords), size=n, replace=False)
        print(f"{n:>8}{neighborhood_mmd(x[idx], coords[idx], x, coords):>10.5f}")
    print(f"{len(coords):>8}{neighborhood_mmd(x, coords, x, coords):>10.5f}")

    print(f"\n# truth = subsample of {ref}, pred = fuller cloud -> OVERSHOOTING n")
    print(f"{'truth_n':>8}{'pred_n':>8}{'ratio':>7}{'NFS_raw':>10}")
    for truth_n in sizes:
        if truth_n >= len(coords):
            continue
        tidx = rng.choice(len(coords), size=truth_n, replace=False)
        for pred_n in sizes:
            if pred_n < truth_n or pred_n > len(coords):
                continue
            pidx = rng.choice(len(coords), size=pred_n, replace=False)
            score = neighborhood_mmd(x[pidx], coords[pidx], x[tidx], coords[tidx])
            print(f"{truth_n:>8}{pred_n:>8}{pred_n / truth_n:>7.2f}{score:>10.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="*", default=[], help="prediction .h5ad files and/or setting/stage refs")
    parser.add_argument("--truth", default=None, help="run the NFS-vs-n sensitivity on this cloud, e.g. heart/E8.75")
    args = parser.parse_args()
    if not args.files and not args.truth:
        parser.error("need --files and/or --truth")

    cfg = load_config()
    if args.files:
        print(f"{'cloud':<52}{'n':>7}{'RMS':>8}{'NN1@4000':>10}{'NN15@all':>10}{'dens_inv':>9}")
        for item in args.files:
            _, coords = cloud(item, cfg)
            report(item, coords)
    if args.truth:
        sensitivity(cfg, args.truth)


if __name__ == "__main__":
    main()
